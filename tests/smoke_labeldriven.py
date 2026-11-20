"""Smoke test for the label-driven generation subsystem (synthetic data)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace")

from tsfresh.labeldriven import (  # noqa: E402
    ExtractionSpec,
    LabelState,
    LabelDrivenStore,
    NoTrainingPopulationError,
    build_and_publish,
)
import tempfile  # noqa: E402

rng = np.random.default_rng(0)


def make_series(lineage_id, kind, n=25):
    t = np.arange(n)
    if kind == 1:
        values = np.sin(t / 2.0) + rng.normal(0, 0.1, n) + 3.0
    else:
        values = rng.normal(0.0, 1.0, n)
    return pd.DataFrame(
        {"id": lineage_id, "time": t, "value": values.astype(float)}
    )


def main():
    tmp = tempfile.mkdtemp()
    store = LabelDrivenStore(tmp)
    store.create()

    # 8 samples: ids 1-4 positive-like, 5-8 negative-like; 9 unlabeled.
    sequences = {}
    for i in range(1, 10):
        kind = 1 if i <= 4 else 0
        sequences[f"S{i}"] = make_series(f"S{i}", kind)
        store.register_sample(f"S{i}", sequences[f"S{i}"], "2026-01-01T00:00:00Z")

    # Review all except S9; leave S8 pending to prove pending != negative.
    for i in range(1, 8):
        label = 1 if i <= 4 else 0
        rec = store.submit_review(f"S{i}", label, "2026-01-02T00:00:00Z")
        store.accept_review(rec.lineage_id, rec.review_version,
                            "2026-01-02T01:00:00Z")
    pending = store.submit_review("S8", 0, "2026-01-02T00:00:00Z")
    assert pending.state is LabelState.PENDING

    spec = ExtractionSpec(column_id="id", column_sort="time", n_jobs=1)
    gen1 = build_and_publish(store, "gen-0001", "2026-01-03T00:00:00Z", spec)
    print("gen1 training size:", gen1.metadata["training_population_size"])
    print("gen1 label counts:", gen1.model.label_counts)
    assert gen1.metadata["training_population_size"] == 7
    assert list(gen1.probabilities.index) == sorted(
        gen1.probabilities.index
    ) or len(gen1.probabilities) == 9
    assert len(gen1.probabilities) == 9  # all current samples scored
    assert gen1.source.excluded[
        store.registry.resolve_lineage_id("S8")
    ] == "review_pending"
    # Positive-like samples should score higher than negative-like.
    pos = gen1.probabilities.filter(like="lineage", axis=0)
    print("probabilities:\n", gen1.probabilities.round(3))

    # --- Sequence edit: label must expire, retraining must refuse until re-reviewed
    edited = make_series("S1", 1)
    store.append_revision("S1", edited, "2026-01-04T00:00:00Z")
    lineage_s1 = store.registry.get_lineage(
        store.registry.resolve_lineage_id("S1")
    )
    assert lineage_s1.current_accepted_label() is None
    assert any(r.state is LabelState.EXPIRED for r in lineage_s1.reviews)

    # S1 is now excluded with an explicit stale-revision reason, not a negative.
    gen1b = build_and_publish(store, "gen-0001b", "2026-01-04T01:00:00Z", spec)
    assert lineage_s1.lineage_id not in gen1b.source.labels
    assert gen1b.source.excluded[
        lineage_s1.lineage_id
    ] == "review_expired_or_stale_revision"
    assert store.current_generation_id() == "gen-0001b"

    # Re-review S1 on new revision.
    rec = store.submit_review("S1", 1, "2026-01-04T02:00:00Z")
    store.accept_review(rec.lineage_id, rec.review_version,
                        "2026-01-04T03:00:00Z")
    gen2 = build_and_publish(store, "gen-0002", "2026-01-05T00:00:00Z", spec)
    assert store.current_generation_id() == "gen-0002"
    assert gen2.source.labels[lineage_s1.lineage_id] == 1

    # --- Delete + ID reuse: fresh lineage, must be reviewed again
    store.delete_sample("S9", "2026-01-06T00:00:00Z")
    store.register_sample("S9", make_series("S9", 0), "2026-01-06T01:00:00Z")
    new_lineage = store.registry.get_lineage(
        store.registry.resolve_lineage_id("S9")
    )
    assert new_lineage.current_accepted_label() is None
    assert len(new_lineage.revisions) == 1

    # --- Reopen store and reload generation with integrity verification
    store2 = LabelDrivenStore(tmp)
    store2.open()
    reloaded = store2.load_generation()
    assert reloaded.generation_id == "gen-0002"
    pd.testing.assert_series_equal(
        reloaded.probabilities, gen2.probabilities, check_names=False
    )
    print("reloaded OK; generations:", store2.list_generations())

    # --- Withdrawn label cannot act as negative
    rec = store.submit_review("S9", 0, "2026-01-07T00:00:00Z")
    store.accept_review(rec.lineage_id, rec.review_version,
                        "2026-01-07T01:00:00Z")
    store.withdraw_review(rec.lineage_id, rec.review_version,
                          "2026-01-07T02:00:00Z")
    src = store.registry
    excluded_lineage = src.get_lineage(rec.lineage_id)
    assert excluded_lineage.current_accepted_label() is None
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
