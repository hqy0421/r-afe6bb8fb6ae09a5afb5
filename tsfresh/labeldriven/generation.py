# -*- coding: utf-8 -*-
"""Building one label-driven classifier generation.

A generation is an immutable, fully provenanced bundle:

* the raw sequence revisions it was built from (lineage -> current revision
  and content hash), i.e. the **training source**;
* the exact set of currently accepted 0/1 labels that formed the training
  population — pending, withdrawn, expired and missing labels are excluded
  and never silently treated as negatives;
* the extracted **features** and the tsfresh extraction specification;
* the freshly fitted **model** (imputation + native selection + classifier);
* class-1 **probabilities** aligned by lineage ID for every current sample.

Build failure at any stage raises :class:`GenerationBuildError` and leaves
the previously published generation untouched, so results can never mix new
labels with an old model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from tsfresh.feature_extraction import extract_features

from .exceptions import GenerationBuildError, NoTrainingPopulationError
from .models import LabelState, SequenceState
from .pipeline import (
    ExtractionSpec,
    GenerationModel,
    build_sklearn_pipeline,
)
from .registry import LabelRegistry, canonical_hash


@dataclass
class TrainingSource:
    """Provenance of the population one generation learned from."""

    # lineage_id -> (business_id, revision number, content hash)
    revisions: Dict[str, Tuple[str, int, str]]
    # lineage_id -> 0/1, only currently valid accepted labels
    labels: Dict[str, int]
    # Why every non-training active lineage was excluded, for audit.
    excluded: Dict[str, str]

    def as_frame(self) -> pd.DataFrame:
        rows = []
        for lineage_id, (business_id, revision, digest) in self.revisions.items():
            rows.append(
                {
                    "lineage_id": lineage_id,
                    "business_id": business_id,
                    "revision": revision,
                    "content_hash": digest,
                    "label": self.labels.get(lineage_id),
                    "in_training": lineage_id in self.labels,
                    "exclusion_reason": self.excluded.get(lineage_id, ""),
                }
            )
        return pd.DataFrame(rows).set_index("lineage_id")


@dataclass
class Generation:
    """One immutable generation, persisted and re-openable as a unit."""

    generation_id: str
    created_at: str
    source: TrainingSource
    model: GenerationModel
    probabilities: pd.Series
    # lineage_id -> {"revision": int, "content_hash": str, "business_id": str}
    sequence_manifest: Dict[str, dict]
    features: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def probability_for(self, lineage_id: str) -> float:
        return float(self.probabilities.loc[lineage_id])


def resolve_training_source(registry: LabelRegistry) -> TrainingSource:
    """Derive the effective training population from current lifecycle state.

    Rules enforced here:

    * deleted lineages are excluded entirely;
    * only accepted reviews bound to the lineage's *current* revision yield a
      label;
    * pending, withdrawn and expired reviews exclude the lineage with an
      explicit reason rather than counting as class 0;
    * at least one class-0 and one class-1 sample must be present.
    """
    revisions: Dict[str, Tuple[str, int, str]] = {}
    labels: Dict[str, int] = {}
    excluded: Dict[str, str] = {}

    for lineage in registry.all_lineages():
        if lineage.state is SequenceState.DELETED:
            excluded[lineage.lineage_id] = "lineage_deleted"
            continue
        current = lineage.current_revision
        if current is None:
            excluded[lineage.lineage_id] = "no_revision"
            continue
        revisions[lineage.lineage_id] = (
            lineage.business_id,
            current.revision,
            current.content_hash,
        )
        label = lineage.current_accepted_label()
        if label is not None:
            labels[lineage.lineage_id] = label
            continue
        # Distinguish the non-label states for the audit trail.
        states = {r.state for r in lineage.reviews}
        if LabelState.PENDING in states:
            reason = "review_pending"
        elif LabelState.WITHDRAWN in states and LabelState.ACCEPTED not in states:
            reason = "review_withdrawn"
        elif LabelState.EXPIRED in states:
            reason = "review_expired_or_stale_revision"
        else:
            reason = "no_review"
        excluded[lineage.lineage_id] = reason

    if not labels:
        raise NoTrainingPopulationError(
            "No currently accepted 0/1 labels exist; cannot train."
        )
    present_classes = set(labels.values())
    if present_classes != {0, 1}:
        raise NoTrainingPopulationError(
            "Training needs at least one accepted class-0 and one accepted "
            f"class-1 sample; present classes: {sorted(present_classes)}."
        )
    return TrainingSource(revisions=revisions, labels=labels, excluded=excluded)


def _assemble_raw_long(
    sequences: Dict[str, pd.DataFrame],
    lineage_ids,
    spec: ExtractionSpec,
) -> pd.DataFrame:
    """Concatenate current raw sequences, tagging rows with lineage IDs."""
    frames = []
    for lineage_id in lineage_ids:
        # Tag every row with the internal lineage ID regardless of whether the
        # raw payload already carried an (business-level) id column.
        frame = sequences[lineage_id].copy()
        frame[spec.column_id] = lineage_id
        frames.append(frame)
    if not frames:
        raise GenerationBuildError("No current sequences available.")
    return pd.concat(frames, ignore_index=True)


def build_generation(
    generation_id: str,
    created_at: str,
    registry: LabelRegistry,
    sequences: Dict[str, pd.DataFrame],
    extraction_spec: ExtractionSpec,
    classifier=None,
) -> Generation:
    """Build a complete generation or raise without side effects.

    ``sequences`` maps lineage ID to the *current* raw sequence DataFrame.
    The function never mutates the registry or sequences; on failure the
    caller's previously published generation stays in service.
    """
    registry.assert_internally_consistent()
    source = resolve_training_source(registry)

    # Guard against raw-sequence / registry drift before any fitting starts.
    for lineage_id, (_, revision, digest) in source.revisions.items():
        if lineage_id not in sequences:
            raise GenerationBuildError(
                f"Missing current raw sequence for lineage {lineage_id}."
            )
        actual_hash = canonical_hash(sequences[lineage_id])
        if actual_hash != digest:
            raise GenerationBuildError(
                f"Raw sequence for lineage {lineage_id} does not match the "
                f"registered revision {revision} (hash mismatch)."
            )

    training_ids = list(source.labels.keys())
    all_ids = list(source.revisions.keys())

    # --- native extraction on the actual current sequences ---------------
    raw_long = _assemble_raw_long(sequences, all_ids, extraction_spec)
    all_features = extract_features(
        raw_long,
        default_fc_parameters=extraction_spec.default_fc_parameters,
        column_id=extraction_spec.column_id,
        column_sort=extraction_spec.column_sort,
        column_kind=extraction_spec.column_kind,
        column_value=extraction_spec.column_value,
        n_jobs=extraction_spec.n_jobs,
        disable_progressbar=extraction_spec.disable_progressbar,
        impute_function=None,
    )
    if not isinstance(all_features, pd.DataFrame) or all_features.empty:
        raise GenerationBuildError(
            "Native tsfresh feature extraction produced no features."
        )
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.reindex(index=all_ids)
    feature_columns = list(all_features.columns)

    train_frame = all_features.loc[training_ids]
    y = pd.Series(
        [source.labels[lineage_id] for lineage_id in training_ids],
        index=training_ids,
        name="label",
    )

    # --- fit imputation + native selection + classifier on train only ----
    pipeline = build_sklearn_pipeline(classifier=classifier)
    pipeline.fit(train_frame, y)

    selector = pipeline.named_steps["select"]
    fallback_used = bool(getattr(selector, "fallback_used", False))
    selected = selector.relevant_features or feature_columns

    label_counts = {
        0: sum(1 for value in y if value == 0),
        1: sum(1 for value in y if value == 1),
    }
    model = GenerationModel(
        generation_id=generation_id,
        pipeline=pipeline,
        extraction_spec=extraction_spec,
        feature_columns=feature_columns,
        training_lineage_ids=training_ids,
        label_counts=label_counts,
        all_lineage_ids=all_ids,
        selector_fallback_used=fallback_used,
        metadata={
            "selected_feature_count": len(selected),
            "extracted_feature_count": len(feature_columns),
        },
    )

    # --- score every current sample through the same fitted pipeline -----
    # Reuse the matrix already extracted above; the fitted pipeline performs
    # alignment, imputation and selection, so probabilities cover every ID.
    probabilities = model.score_features(all_features)
    if list(probabilities.index) != all_ids:
        raise GenerationBuildError(
            "Probability output is not aligned to all current lineage IDs."
        )

    sequence_manifest = {
        lineage_id: {
            "business_id": business_id,
            "revision": revision,
            "content_hash": digest,
        }
        for lineage_id, (business_id, revision, digest) in source.revisions.items()
    }
    return Generation(
        generation_id=generation_id,
        created_at=created_at,
        source=source,
        model=model,
        probabilities=probabilities,
        sequence_manifest=sequence_manifest,
        features=all_features,
        metadata={
            "training_population_size": len(training_ids),
            "current_population_size": len(all_ids),
            "label_counts": label_counts,
        },
    )
