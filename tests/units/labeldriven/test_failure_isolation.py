# -*- coding: utf-8 -*-
"""Tests for failure isolation: old generation stays valid when builds fail."""
import pandas as pd
import pytest

from tsfresh.labeldriven import (
    GenerationBuildError,
    LabelDrivenStore,
    build_and_publish,
)


def _seed(store, make_sequence):
    for business_id, positive in [
        ("P1", True), ("P2", True), ("N1", False), ("N2", False)
    ]:
        store.register_sample(
            business_id, make_sequence(business_id, positive=positive), "t0"
        )
        record = store.submit_review(business_id, int(positive), "t1")
        store.accept_review(record.lineage_id, record.review_version, "t2")


def test_extraction_failure_keeps_old_generation(
    tmp_path, make_sequence, minimal_spec, monkeypatch
):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed(store, make_sequence)
    first = build_and_publish(store, "gen-1", "t3", minimal_spec)

    import tsfresh.labeldriven.generation as generation_module

    def broken_extract(*args, **kwargs):
        raise RuntimeError("native extraction exploded")

    monkeypatch.setattr(
        generation_module, "extract_features", broken_extract
    )
    with pytest.raises((GenerationBuildError, RuntimeError)):
        build_and_publish(store, "gen-2", "t4", minimal_spec)
    monkeypatch.undo()

    assert store.current_generation_id() == "gen-1"
    still = store.load_generation("gen-1")
    assert still.generation_id == first.generation_id
    assert not list((root / "generations").glob(".staging-*"))


def test_fit_failure_keeps_old_generation(
    tmp_path, make_sequence, minimal_spec, monkeypatch
):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed(store, make_sequence)
    build_and_publish(store, "gen-1", "t3", minimal_spec)

    class BrokenClassifier:
        def fit(self, X, y):
            raise RuntimeError("classifier fit exploded")

    with pytest.raises(RuntimeError):
        build_and_publish(
            store, "gen-2", "t4", minimal_spec, classifier=BrokenClassifier()
        )
    assert store.current_generation_id() == "gen-1"
    assert store.list_generations() == ["gen-1"]


def test_publish_never_mixes_new_labels_with_old_model(
    tmp_path, make_sequence, minimal_spec
):
    """After labels change, the old generation refuses verification and only
    a freshly built generation is promoted."""
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    _seed(store, make_sequence)
    build_and_publish(store, "gen-1", "t3", minimal_spec)

    # Simulate an attempted "new labels + old model" publication by rebuilding
    # against changed registry state while reusing the old artifact: the store
    # must reject it on verification rather than serve mixed results.
    lineage_id = store.resolve_lineage_id("P1")
    review = [
        r for r in store.get_lineage(lineage_id).reviews
        if r.state.value == "accepted"
    ][0]
    store.withdraw_review(lineage_id, review.review_version, "t4")
    record = store.submit_review("P1", 0, "t5")
    store.accept_review(record.lineage_id, record.review_version, "t6")

    old = store.load_generation("gen-1", verify=False)
    with pytest.raises(Exception, match="training label"):
        store.save_generation(old)
    # Index untouched.
    assert store.current_generation_id() == "gen-1"

    build_and_publish(store, "gen-2", "t7", minimal_spec)
    assert store.current_generation_id() == "gen-2"


def test_counters_continue_after_reopen(tmp_path, make_sequence):
    root = tmp_path / "s"
    store = LabelDrivenStore(root)
    store.create()
    store.register_sample("A", make_sequence("A"), "t0")
    old_lineage = store.resolve_lineage_id("A")
    store.delete_sample("A", "t1")

    reopened = LabelDrivenStore(root)
    reopened.open()
    lineage = reopened.register_sample("A", make_sequence("A"), "t2")
    # Must not collide with the pre-deletion lineage id.
    assert lineage.lineage_id != old_lineage
    assert int(lineage.lineage_id.split("-")[1]) > int(
        old_lineage.split("-")[1]
    )
    record = reopened.submit_review("A", 1, "t3")
    assert record.review_version == 1
