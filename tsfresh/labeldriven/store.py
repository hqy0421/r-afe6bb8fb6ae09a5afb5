# -*- coding: utf-8 -*-
"""On-disk store: raw sequences, reviews, generations and full provenance.

Layout (one store directory)::

    store/
      registry.pkl                    # LabelRegistry (lineages + reviews)
      sequences/
        <lineage_id>/rev_<n>.parquet # raw sequence per full revision
        <lineage_id>/current.json     # pointer to the current revision
      generations/
        index.json                    # ordered generation ids + current pointer
        <generation_id>/
          generation.pkl              # Generation (model, source, manifest, ...)
          features.parquet            # extracted current feature matrix
          probabilities.csv           # ID-aligned class-1 probabilities
          checksums.json              # integrity digests of every artifact

All writes are staged in a temporary directory and renamed into place only
after every artifact and checksum has been written successfully. A failure
mid-save therefore leaves the previously saved generation fully intact and
the store refuses to publish results that would mix new labels with an old
model.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .exceptions import StoreError
from .generation import Generation
from .models import SequenceState
from .registry import LabelRegistry, canonical_hash

_CHECKSUM_FILE = "checksums.json"
_GENERATION_PKL = "generation.pkl"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LabelDrivenStore:
    """File-backed facade combining the registry, raw sequences and models."""

    def __init__(self, root: os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sequences").mkdir(exist_ok=True)
        (self.root / "generations").mkdir(exist_ok=True)
        self._registry: Optional[LabelRegistry] = None
        # lineage_id -> {revision(int) -> DataFrame}
        self._sequences: Dict[str, Dict[int, pd.DataFrame]] = {}

    # ------------------------------------------------------------------ #
    # registry / sequence access
    # ------------------------------------------------------------------ #
    @property
    def registry(self) -> LabelRegistry:
        if self._registry is None:
            raise StoreError("Store is not open; call open() or create().")
        return self._registry

    def create(self) -> None:
        registry_path = self.root / "registry.pkl"
        if registry_path.exists():
            raise StoreError(f"Store already exists at {self.root}.")
        self._registry = LabelRegistry()
        self._persist_registry_atomic()

    def open(self) -> None:
        """Load the registry and verify every stored sequence revision."""
        registry_path = self.root / "registry.pkl"
        if not registry_path.exists():
            raise StoreError(f"No store found at {self.root}; call create().")
        with open(registry_path, "rb") as handle:
            registry = pickle.load(handle)
        if not isinstance(registry, LabelRegistry):
            raise StoreError("registry.pkl does not contain a LabelRegistry.")
        # Counters are not otherwise reconstructed from pickled state.
        registry.reindex_counters()
        registry.assert_internally_consistent()

        sequences: Dict[str, Dict[int, pd.DataFrame]] = {}
        for lineage in registry.all_lineages():
            per_revision: Dict[int, pd.DataFrame] = {}
            lineage_dir = self.root / "sequences" / lineage.lineage_id
            for revision in lineage.revisions:
                if revision.content_hash.startswith("DELETED:"):
                    continue
                path = lineage_dir / f"rev_{revision.revision}.parquet"
                if not path.exists():
                    raise StoreError(
                        f"Missing raw sequence artifact for "
                        f"{lineage.lineage_id} revision {revision.revision}."
                    )
                frame = pd.read_parquet(path)
                if canonical_hash(frame) != revision.content_hash:
                    raise StoreError(
                        f"Checksum mismatch for stored sequence "
                        f"{lineage.lineage_id} rev {revision.revision}."
                    )
                per_revision[revision.revision] = frame
            sequences[lineage.lineage_id] = per_revision
        self._registry = registry
        self._sequences = sequences

    def current_sequences(self) -> Dict[str, pd.DataFrame]:
        """Return the current raw sequence of every active lineage."""
        result = {}
        for lineage in self.registry.active_lineages():
            current = lineage.current_revision
            result[lineage.lineage_id] = self._sequences[lineage.lineage_id][
                current.revision
            ]
        return result

    def resolve_lineage_id(self, business_id: str) -> str:
        """Convenience delegation to the registry's business-ID lookup."""
        lineage_id = self.registry.resolve_lineage_id(business_id)
        if lineage_id is None:
            raise StoreError(
                f"No active lineage for business ID {business_id!r}."
            )
        return lineage_id

    def get_lineage(self, lineage_id: str):
        return self.registry.get_lineage(lineage_id)

    # ------------------------------------------------------------------ #
    # mutation helpers (registry + raw payload saved together, atomically)
    # ------------------------------------------------------------------ #
    def register_sample(
        self, business_id: str, sequence: pd.DataFrame, created_at: str
    ):
        with self._rollback_on_save_failure():
            lineage = self.registry.register_sample(
                business_id, sequence, created_at
            )
            self._write_sequence_revision(lineage.lineage_id, 1, sequence)
        return lineage

    def append_revision(
        self, business_id: str, sequence: pd.DataFrame, created_at: str
    ):
        with self._rollback_on_save_failure():
            revision = self.registry.append_revision(
                business_id, sequence, created_at
            )
            lineage_id = self.registry.resolve_lineage_id(business_id)
            self._write_sequence_revision(
                lineage_id, revision.revision, sequence
            )
        return revision

    def delete_sample(self, business_id: str, deleted_at: str) -> None:
        with self._rollback_on_save_failure():
            self.registry.delete_sample(business_id, deleted_at)

    # review pass-throughs, each persisted atomically
    def submit_review(self, business_id, label, created_at, note=None):
        with self._rollback_on_save_failure():
            record = self.registry.submit_review(
                business_id, label, created_at, note
            )
        return record

    def accept_review(self, lineage_id, review_version, decided_at):
        with self._rollback_on_save_failure():
            record = self.registry.accept_review(
                lineage_id, review_version, decided_at
            )
        return record

    def withdraw_review(self, lineage_id, review_version, decided_at):
        with self._rollback_on_save_failure():
            record = self.registry.withdraw_review(
                lineage_id, review_version, decided_at
            )
        return record

    # ------------------------------------------------------------------ #
    # generation persistence
    # ------------------------------------------------------------------ #
    def save_generation(self, generation: Generation) -> None:
        """Persist a full generation atomically; old generation survives failure."""
        gen_root = self.root / "generations"
        target = gen_root / generation.generation_id
        # Provenance is verified before the duplicate-name check so an
        # attempted "new labels + old model" re-publication is always reported
        # as the integrity violation it is, not a name collision.
        self._verify_generation_against_registry(generation)
        if target.exists():
            raise StoreError(
                f"Generation {generation.generation_id} already exists; "
                "generations are immutable."
            )
        staging = Path(
            tempfile.mkdtemp(prefix=".staging-", dir=str(gen_root))
        )
        try:
            generation_path = staging / _GENERATION_PKL
            with open(generation_path, "wb") as handle:
                pickle.dump(generation, handle)

            features_path = staging / "features.parquet"
            generation.features.to_parquet(features_path)

            proba_path = staging / "probabilities.csv"
            generation.probabilities.to_frame().to_csv(proba_path)

            checksums = {
                _GENERATION_PKL: _sha256_file(generation_path),
                "features.parquet": _sha256_file(features_path),
                "probabilities.csv": _sha256_file(proba_path),
            }
            with open(staging / _CHECKSUM_FILE, "w", encoding="utf-8") as handle:
                json.dump(checksums, handle, indent=2, sort_keys=True)

            # Verify the pickled bundle re-loads before it is promoted.
            with open(generation_path, "rb") as handle:
                reloaded = pickle.load(handle)
            if not isinstance(reloaded, Generation):
                raise StoreError("Reload check failed: wrong object type.")

            os.replace(staging, target)
            self._promote_generation_index(generation.generation_id)
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            if isinstance(exc, StoreError):
                raise
            raise StoreError(
                f"Saving generation {generation.generation_id} failed; the "
                f"previous generation remains in service: {exc}"
            ) from exc

    def load_generation(
        self, generation_id: Optional[str] = None, verify: bool = True
    ) -> Generation:
        """Re-open a saved generation, verifying checksums and provenance."""
        if generation_id is None:
            generation_id = self.current_generation_id()
            if generation_id is None:
                raise StoreError("No generation has been published yet.")
        gen_dir = self.root / "generations" / generation_id
        if not gen_dir.exists():
            raise StoreError(f"Generation {generation_id} not found.")
        checksum_path = gen_dir / _CHECKSUM_FILE
        if not checksum_path.exists():
            raise StoreError(
                f"Generation {generation_id} is missing its checksum manifest."
            )
        with open(checksum_path, encoding="utf-8") as handle:
            checksums = json.load(handle)
        for name, expected in checksums.items():
            actual = _sha256_file(gen_dir / name)
            if actual != expected:
                raise StoreError(
                    f"Integrity check failed for {name} of generation "
                    f"{generation_id}."
                )
        with open(gen_dir / _GENERATION_PKL, "rb") as handle:
            generation = pickle.load(handle)
        if verify:
            self._verify_generation_against_registry(generation)
        return generation

    def list_generations(self) -> List[str]:
        index = self._read_index()
        return list(index.get("generations", []))

    def current_generation_id(self) -> Optional[str]:
        return self._read_index().get("current")

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _verify_generation_against_registry(self, generation: Generation) -> None:
        """Refuse to publish/load a generation inconsistent with current state.

        A generation is only valid if every training lineage's bound revision
        and content hash still matches the registry, and its training labels
        are still the currently accepted labels. This blocks mixing new labels
        with an old model.
        """
        registry = self._registry
        labels = (
            registry.current_training_labels() if registry is not None else {}
        )
        for lineage_id, label in generation.source.labels.items():
            try:
                lineage = registry.get_lineage(lineage_id)
            except Exception as exc:
                raise StoreError(
                    f"Generation references unknown lineage {lineage_id}."
                ) from exc
            if lineage.state is SequenceState.DELETED:
                raise StoreError(
                    f"Generation trained on since-deleted lineage {lineage_id}."
                )
            current = lineage.current_revision
            bound = generation.sequence_manifest[lineage_id]
            if current.revision != bound["revision"]:
                raise StoreError(
                    f"Lineage {lineage_id} moved to revision "
                    f"{current.revision}; generation is bound to "
                    f"{bound['revision']}. Rebuild before publishing."
                )
            if current.content_hash != bound["content_hash"]:
                raise StoreError(
                    f"Content drift on lineage {lineage_id}; the stored raw "
                    "sequence no longer matches the generation's provenance."
                )
            if labels.get(lineage_id) != label:
                raise StoreError(
                    f"Current accepted label for {lineage_id} "
                    f"({labels.get(lineage_id)}) differs from the generation's "
                    f"training label ({label})."
                )

    def _promote_generation_index(self, generation_id: str) -> None:
        index_path = self.root / "generations" / "index.json"
        index = self._read_index()
        generations = index.get("generations", [])
        generations.append(generation_id)
        payload = {"generations": generations, "current": generation_id}
        staging = index_path.with_suffix(".json.tmp")
        with open(staging, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(staging, index_path)

    def _read_index(self) -> dict:
        index_path = self.root / "generations" / "index.json"
        if not index_path.exists():
            return {"generations": [], "current": None}
        with open(index_path, encoding="utf-8") as handle:
            return json.load(handle)

    def _write_sequence_revision(
        self, lineage_id: str, revision: int, sequence: pd.DataFrame
    ) -> None:
        lineage_dir = self.root / "sequences" / lineage_id
        lineage_dir.mkdir(parents=True, exist_ok=True)
        target = lineage_dir / f"rev_{revision}.parquet"
        staging = lineage_dir / f".rev_{revision}.tmp"
        sequence.to_parquet(staging)
        stored = pd.read_parquet(staging)
        if canonical_hash(stored) != canonical_hash(sequence):
            raise StoreError("Parquet round-trip changed sequence content.")
        os.replace(staging, target)
        pointer = {"current_revision": revision}
        with open(lineage_dir / "current.json", "w", encoding="utf-8") as handle:
            json.dump(pointer, handle)
        self._sequences.setdefault(lineage_id, {})[revision] = sequence.copy()

    def _persist_registry_atomic(self) -> None:
        target = self.root / "registry.pkl"
        staging = target.with_suffix(".pkl.tmp")
        with open(staging, "wb") as handle:
            pickle.dump(self._registry, handle)
        os.replace(staging, target)

    @contextlib.contextmanager
    def _rollback_on_save_failure(self):
        """Snapshot state *before* mutation; restore it if persistence fails.

        A deep copy is essential because registry mutation happens in place;
        a pickle taken after the mutation would already contain the change and
        could not roll anything back. The in-memory sequence cache is copied
        for the same reason.
        """
        registry_snapshot = copy.deepcopy(self._registry)
        sequences_snapshot = copy.deepcopy(self._sequences)
        try:
            yield
            self._persist_registry_atomic()
        except Exception as exc:
            self._registry = registry_snapshot
            self._sequences = sequences_snapshot
            if isinstance(exc, StoreError):
                raise
            raise StoreError(
                f"Atomic save failed; in-memory state was rolled back: {exc}"
            ) from exc


def build_and_publish(
    store: LabelDrivenStore,
    generation_id: str,
    created_at: str,
    extraction_spec,
    classifier=None,
) -> Generation:
    """End-to-end: build from current state and publish only on full success.

    On any conflict, extraction, fitting or save failure the previous
    generation remains published; nothing partial is exposed.
    """
    from .generation import build_generation

    registry_snapshot = store.registry.snapshot()
    sequences = store.current_sequences()
    # Build errors (conflicts, unusable training population, extraction or
    # fitting failures) propagate unchanged; save_generation itself guarantees
    # the previous generation remains published on persistence failures.
    generation = build_generation(
        generation_id=generation_id,
        created_at=created_at,
        registry=registry_snapshot,
        sequences=sequences,
        extraction_spec=extraction_spec,
        classifier=classifier,
    )
    store.save_generation(generation)
    return generation
