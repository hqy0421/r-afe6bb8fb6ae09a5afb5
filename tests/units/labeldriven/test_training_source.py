# -*- coding: utf-8 -*-
"""Tests for training-population resolution: only current accepted 0/1 labels."""
import pytest

from tsfresh.labeldriven import (
    LabelState,
    NoTrainingPopulationError,
    resolve_training_source,
)


def _lineage_map(registry):
    return {
        lineage.business_id: lineage.lineage_id
        for lineage in registry.all_lineages()
    }


def test_baseline_population(populated_registry):
    registry, _ = populated_registry
    source = resolve_training_source(registry)
    ids = _lineage_map(registry)
    assert set(source.labels) == {
        ids["P1"], ids["P2"], ids["P3"],
        ids["N1"], ids["N2"], ids["N3"],
    }
    assert source.labels[ids["P1"]] == 1
    assert source.labels[ids["N1"]] == 0
    assert ids["U1"] in source.excluded
    assert source.excluded[ids["U1"]] == "no_review"


def test_pending_is_excluded_not_negative(populated_registry):
    registry, _ = populated_registry
    ids = _lineage_map(registry)
    record = registry.submit_review("U1", 0, "t")
    source = resolve_training_source(registry)
    assert ids["U1"] not in source.labels
    assert source.excluded[ids["U1"]] == "review_pending"
    assert record.state is LabelState.PENDING


def test_withdrawn_is_excluded_not_negative(populated_registry):
    registry, _ = populated_registry
    ids = _lineage_map(registry)
    lineage_id = ids["N1"]
    review = [
        r for r in registry.get_lineage(lineage_id).reviews
        if r.state is LabelState.ACCEPTED
    ][0]
    registry.withdraw_review(lineage_id, review.review_version, "t")
    source = resolve_training_source(registry)
    assert lineage_id not in source.labels
    assert source.excluded[lineage_id] == "review_withdrawn"


def test_expired_after_edit_is_excluded(populated_registry, make_sequence):
    registry, sequences = populated_registry
    ids = _lineage_map(registry)
    registry.append_revision(
        "P1", make_sequence("P1", positive=False), "t-later"
    )
    source = resolve_training_source(registry)
    assert ids["P1"] not in source.labels
    assert source.excluded[ids["P1"]] == "review_expired_or_stale_revision"


def test_deleted_lineage_excluded(populated_registry):
    registry, _ = populated_registry
    ids = _lineage_map(registry)
    registry.delete_sample("N2", "t")
    source = resolve_training_source(registry)
    assert ids["N2"] not in source.revisions
    assert source.excluded[ids["N2"]] == "lineage_deleted"


def test_missing_each_class_is_an_error(populated_registry):
    registry, _ = populated_registry
    # Withdraw every negative: class 0 disappears, training must refuse.
    for business_id in ("N1", "N2", "N3"):
        lineage_id = registry.resolve_lineage_id(business_id)
        review = [
            r for r in registry.get_lineage(lineage_id).reviews
            if r.state is LabelState.ACCEPTED
        ][0]
        registry.withdraw_review(lineage_id, review.review_version, "t")
    with pytest.raises(NoTrainingPopulationError, match="class-0"):
        resolve_training_source(registry)


def test_no_labels_at_all_is_an_error(make_sequence):
    from tsfresh.labeldriven import LabelRegistry

    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A"), "t")
    with pytest.raises(NoTrainingPopulationError):
        resolve_training_source(registry)


def test_provenance_frame_content(populated_registry):
    registry, _ = populated_registry
    frame = resolve_training_source(registry).as_frame()
    assert bool(frame.loc[registry.resolve_lineage_id("U1"), "in_training"]) is False
    train_rows = frame[frame.in_training]
    assert set(train_rows.label) == {0, 1}
    assert (train_rows.revision == 1).all()
    assert train_rows.content_hash.str.len().eq(64).all()
