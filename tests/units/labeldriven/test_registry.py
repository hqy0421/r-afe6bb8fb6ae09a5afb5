# -*- coding: utf-8 -*-
"""Tests for the sample/review lifecycle and conflict rules."""
import pytest

from tsfresh.labeldriven import (
    ConflictError,
    LabelState,
    LabelRegistry,
    canonical_hash,
)


def test_revision_numbers_and_content_hash(make_sequence):
    registry = LabelRegistry()
    seq_a = make_sequence("A", positive=True)
    lineage = registry.register_sample("A", seq_a, "t0")
    assert lineage.current_revision.revision == 1
    assert lineage.current_revision.content_hash == canonical_hash(seq_a)

    seq_b = make_sequence("A", positive=False)
    revision = registry.append_revision("A", seq_b, "t1")
    assert revision.revision == 2
    assert lineage.current_revision.revision == 2
    assert canonical_hash(seq_b) != canonical_hash(seq_a)


def test_duplicate_revision_rejected(make_sequence):
    registry = LabelRegistry()
    seq = make_sequence("A")
    registry.register_sample("A", seq, "t0")
    with pytest.raises(ConflictError):
        registry.upsert_sequence("A", seq.copy(), "t1")


def test_review_bound_to_revision_expires_on_edit(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A", positive=True), "t0")
    record = registry.submit_review("A", 1, "t1")
    registry.accept_review(record.lineage_id, record.review_version, "t2")
    lineage_id = record.lineage_id
    assert registry.get_lineage(lineage_id).current_accepted_label() == 1

    registry.append_revision("A", make_sequence("A", positive=False), "t3")
    lineage = registry.get_lineage(lineage_id)
    # The human label must not follow the edit.
    assert lineage.current_accepted_label() is None
    assert lineage.reviews[0].state is LabelState.EXPIRED
    assert lineage.reviews[0].previous_state is LabelState.ACCEPTED


def test_pending_review_also_expires_on_edit(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A"), "t0")
    record = registry.submit_review("A", 1, "t1")
    registry.append_revision("A", make_sequence("A", positive=True), "t2")
    lineage = registry.get_lineage(record.lineage_id)
    assert lineage.reviews[0].state is LabelState.EXPIRED
    assert lineage.current_accepted_label() is None


def test_rereview_required_after_edit_and_rebinds_to_new_revision(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A", positive=True), "t0")
    first = registry.submit_review("A", 1, "t1")
    registry.accept_review(first.lineage_id, first.review_version, "t2")
    registry.append_revision("A", make_sequence("A", positive=False), "t3")

    second = registry.submit_review("A", 0, "t4")
    assert second.revision == 2
    registry.accept_review(second.lineage_id, second.review_version, "t5")
    lineage = registry.get_lineage(first.lineage_id)
    assert lineage.current_accepted_label() == 0
    assert registry.current_training_labels() == {first.lineage_id: 0}


def test_cannot_accept_review_on_stale_revision(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A"), "t0")
    record = registry.submit_review("A", 1, "t1")
    registry.append_revision("A", make_sequence("A", positive=True), "t2")
    with pytest.raises(ConflictError):
        registry.accept_review(record.lineage_id, record.review_version, "t3")


def test_withdrawn_label_never_a_training_label(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A"), "t0")
    record = registry.submit_review("A", 0, "t1")
    registry.accept_review(record.lineage_id, record.review_version, "t2")
    registry.withdraw_review(record.lineage_id, record.review_version, "t3")
    assert registry.current_training_labels() == {}
    lineage = registry.get_lineage(record.lineage_id)
    assert lineage.current_accepted_label() is None


def test_illegal_transitions_rejected(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A"), "t0")
    record = registry.submit_review("A", 1, "t1")
    registry.withdraw_review(record.lineage_id, record.review_version, "t2")
    # Terminal states cannot be re-accepted.
    with pytest.raises(ConflictError):
        registry.accept_review(record.lineage_id, record.review_version, "t3")
    with pytest.raises(ConflictError):
        registry.withdraw_review(record.lineage_id, record.review_version, "t4")


def test_only_one_live_review_per_lineage(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A"), "t0")
    registry.submit_review("A", 1, "t1")
    with pytest.raises(ConflictError):
        registry.submit_review("A", 0, "t2")


def test_non_binary_labels_rejected(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A"), "t0")
    with pytest.raises(ConflictError):
        registry.submit_review("A", 2, "t1")


def test_delete_and_id_reuse_requires_new_review(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A", positive=True), "t0")
    record = registry.submit_review("A", 1, "t1")
    registry.accept_review(record.lineage_id, record.review_version, "t2")
    old_lineage_id = registry.resolve_lineage_id("A")

    registry.delete_sample("A", "t3")
    assert registry.resolve_lineage_id("A") is None
    # Old lineage stays in the log but yields no label.
    assert registry.get_lineage(old_lineage_id).current_accepted_label() is None

    registry.register_sample("A", make_sequence("A", positive=False), "t4")
    new_lineage_id = registry.resolve_lineage_id("A")
    assert new_lineage_id != old_lineage_id
    fresh = registry.get_lineage(new_lineage_id)
    assert fresh.current_revision.revision == 1
    assert fresh.current_accepted_label() is None
    # The old review cannot bind to the new lineage.
    assert registry.current_training_labels() == {}


def test_review_on_deleted_id_rejected(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A"), "t0")
    registry.delete_sample("A", "t1")
    with pytest.raises(ConflictError):
        registry.submit_review("A", 1, "t2")


def test_register_existing_active_id_rejected(make_sequence):
    registry = LabelRegistry()
    registry.register_sample("A", make_sequence("A"), "t0")
    with pytest.raises(ConflictError):
        registry.register_sample("A", make_sequence("A"), "t1")


def test_training_labels_only_contains_current_accepted(populated_registry):
    registry, _ = populated_registry
    labels = registry.current_training_labels()
    assert set(labels.values()) == {0, 1}
    # Unlabeled U1 is absent, never a zero.
    u1_lineage = registry.resolve_lineage_id("U1")
    assert u1_lineage not in labels
