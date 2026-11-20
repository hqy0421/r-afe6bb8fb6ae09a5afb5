# -*- coding: utf-8 -*-
"""Persistence tests: atomic saves, integrity checks and old-generation safety."""
import json

import pandas as pd
import pytest

from tsfresh.labeldriven import (
    LabelDrivenStore,
    LabelState,
    StoreError,
    build_and_publish,
)


def _seed_store(store, make_sequence):
    """Register 4 samples (2 pos / 2 neg), all accepted."""
    for business_id, positive in [
        ("P1", True), ("P2", True), ("N1", False), ("N2", False)
    ]:
        store.register_sample(
            business_id, make_sequence(business_id, positive=positive), "t0"
        )
        record = store.submit_review(business_id, int(positive), "t1")
        store.accept_review(record.lineage_id, record.review_version, "t2")


def test_create_open_roundtrip(tmp_path, make_sequence, minimal_spec):
    root = tmp_path / "store"
    store = LabelDrivenStore(root)
    store.create()
    _seed_store(store, make_sequence)
    generation = build_and_publish(store, "gen-1", "t3", minimal_spec)

    reopened = LabelDrivenStore(root)
    reopened.open()
    assert reopened.current_generation_id() == "gen-1"
    loaded = reopened.load_generation()
    assert loaded.generation_id == "gen-1"
    pd.testing.assert_series_equal(
        loaded.probabilities, generation.probabilities
    )
    pd.testing.assert_frame_equal(loaded.features, generation.features)
    assert reopened.list_generations() == ["gen-1"]


def test_raw_sequences_persisted_per_revision(tmp_path, make_sequence):
    store = LabelDrivenStore(tmp_path / "s")
    store.create()
    seq1 = make_sequence("A", positive=True)
    store.register_sample("A", seq1, "t0")
    seq2 = make_sequence("A", positive=False)
    store.append_revision("A", seq2, "t1")

    reopened = LabelDrivenStore(tmp_path / "s")
    reopened.open()
    lineage_id = reopened.resolve_lineage_id("A")
    stored = reopened._sequences[lineage_id]
    pd.testing.assert_frame_equal(stored[1], seq1)
    pd.testing.assert_frame_equal(stored[2], seq2)
    assert reopened.current_sequences()[lineage_id].equals(seq2)


def test_tampered_generation_fails_integrity_check(
    tmp_path, make_sequence, minimal_spec
):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed_store(store, make_sequence)
    build_and_publish(store, "gen-1", "t3", minimal_spec)

    gen_dir = root / "generations" / "gen-1"
    proba = gen_dir / "probabilities.csv"
    proba.write_text(proba.read_text() + "tampered-line\n")

    reopened = LabelDrivenStore(root)
    reopened.open()
    with pytest.raises(StoreError, match="Integrity check failed"):
        reopened.load_generation()


def test_tampered_sequence_fails_reopen(tmp_path, make_sequence):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    store.register_sample("A", make_sequence("A"), "t0")

    target = next((root / "sequences").rglob("rev_1.parquet"))
    frame = pd.read_parquet(target)
    frame.to_parquet(target, index=False)
    # Parquet schema/index round trip can be stable; force content drift instead.
    frame = frame.assign(value=frame.value + 1.0)
    frame.to_parquet(target, index=False)

    reopened = LabelDrivenStore(root)
    with pytest.raises(StoreError, match="Checksum mismatch"):
        reopened.open()


def test_save_failure_keeps_previous_generation(
    tmp_path, make_sequence, minimal_spec, monkeypatch
):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed_store(store, make_sequence)
    build_and_publish(store, "gen-1", "t3", minimal_spec)

    # Force the second generation's save to fail during feature serialization.
    original_to_parquet = pd.DataFrame.to_parquet

    def failing_to_parquet(self, path, *args, **kwargs):
        if str(path).endswith("features.parquet"):
            raise OSError("simulated disk failure")
        return original_to_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_to_parquet)
    with pytest.raises(StoreError, match="previous generation remains"):
        build_and_publish(store, "gen-2", "t4", minimal_spec)
    monkeypatch.undo()

    assert store.current_generation_id() == "gen-1"
    assert store.list_generations() == ["gen-1"]
    # No staging directory is left behind.
    assert not list((root / "generations").glob(".staging-*"))
    # The old generation still loads cleanly.
    assert store.load_generation("gen-1").generation_id == "gen-1"


def test_stale_generation_rejected_against_new_labels(
    tmp_path, make_sequence, minimal_spec
):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed_store(store, make_sequence)
    build_and_publish(store, "gen-1", "t3", minimal_spec)

    # New revision + new accepted label: the old generation's provenance is stale.
    store.append_revision(
        "P1", make_sequence("P1", positive=False), "t4"
    )
    record = store.submit_review("P1", 0, "t5")
    store.accept_review(record.lineage_id, record.review_version, "t6")

    with pytest.raises(StoreError, match="moved to revision"):
        store.load_generation("gen-1")

    # Building a fresh generation restores a publishable state.
    build_and_publish(store, "gen-2", "t7", minimal_spec)
    assert store.load_generation("gen-2").generation_id == "gen-2"


def test_generation_trained_on_deleted_lineage_is_rejected(
    tmp_path, make_sequence, minimal_spec
):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed_store(store, make_sequence)
    build_and_publish(store, "gen-1", "t3", minimal_spec)

    store.delete_sample("P1", "t4")
    with pytest.raises(StoreError, match="since-deleted lineage"):
        store.load_generation("gen-1")


def test_label_change_invalidates_old_generation(
    tmp_path, make_sequence, minimal_spec
):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed_store(store, make_sequence)
    build_and_publish(store, "gen-1", "t3", minimal_spec)

    lineage_id = store.resolve_lineage_id("P1")
    review = [
        r for r in store.registry.get_lineage(lineage_id).reviews
        if r.state is LabelState.ACCEPTED
    ][0]
    store.withdraw_review(lineage_id, review.review_version, "t4")
    record = store.submit_review("P1", 0, "t5")
    store.accept_review(record.lineage_id, record.review_version, "t6")

    with pytest.raises(StoreError, match="training label"):
        store.load_generation("gen-1")


def test_registry_save_failure_rolls_back_memory(
    tmp_path, make_sequence, monkeypatch
):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    store.register_sample("A", make_sequence("A"), "t0")

    def fail_dump(obj, handle):
        raise OSError("simulated pickle failure")

    monkeypatch.setattr(
        "tsfresh.labeldriven.store.pickle.dump", fail_dump
    )
    with pytest.raises(StoreError):
        store.submit_review("A", 1, "t1")
    monkeypatch.undo()

    # In-memory registry was rolled back to its persisted state.
    lineage_id = store.resolve_lineage_id("A")
    assert store.registry.get_lineage(lineage_id).reviews == []
    # And a fresh open sees the same persisted state.
    reopened = LabelDrivenStore(root)
    reopened.open()
    assert reopened.registry.get_lineage(lineage_id).reviews == []


def test_generation_ids_are_immutable(tmp_path, make_sequence, minimal_spec):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed_store(store, make_sequence)
    first = build_and_publish(store, "gen-1", "t3", minimal_spec)
    with pytest.raises(StoreError, match="immutable"):
        store.save_generation(first)


def test_checksums_manifest_complete(tmp_path, make_sequence, minimal_spec):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed_store(store, make_sequence)
    build_and_publish(store, "gen-1", "t3", minimal_spec)
    with open(
        root / "generations" / "gen-1" / "checksums.json", encoding="utf-8"
    ) as handle:
        checksums = json.load(handle)
    assert set(checksums) == {
        "generation.pkl",
        "features.parquet",
        "probabilities.csv",
    }
    assert all(len(digest) == 64 for digest in checksums.values())


def test_rescore_after_reopen_matches_published_probabilities(
    tmp_path, make_sequence, minimal_spec
):
    """Raw sequences stored with business IDs must re-score identically after
    reopening (regression: id-column alignment through lineage IDs)."""
    import pandas as pd

    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed_store(store, make_sequence)
    published = build_and_publish(store, "gen-1", "t3", minimal_spec)

    reopened = LabelDrivenStore(root)
    reopened.open()
    loaded = reopened.load_generation()
    raw = pd.concat(
        [
            reopened.current_sequences()[lineage_id]
            for lineage_id in loaded.model.all_lineage_ids
        ],
        ignore_index=True,
    )
    rescored = loaded.model.predict_proba(
        raw, loaded.model.all_lineage_ids
    )
    pd.testing.assert_series_equal(
        rescored, published.probabilities, check_names=False
    )
