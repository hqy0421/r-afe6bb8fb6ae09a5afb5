# -*- coding: utf-8 -*-
"""Shared fixtures for label-driven generation tests."""
import numpy as np
import pandas as pd
import pytest

from tsfresh.feature_extraction.settings import MinimalFCParameters

from tsfresh.labeldriven import ExtractionSpec, LabelRegistry


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def make_sequence(rng):
    def _make(sample_id, positive=False, n=20, drift=0.0):
        t = np.arange(n)
        if positive:
            values = 5.0 + np.sin(t / 2.0) + rng.normal(0, 0.1, n) + drift
        else:
            values = rng.normal(0.0, 1.0, n) + drift
        return pd.DataFrame(
            {"id": sample_id, "time": t, "value": values.astype(float)}
        )

    return _make


@pytest.fixture
def minimal_spec():
    # MinimalFCParameters keeps the full pipeline native but fast in tests.
    return ExtractionSpec(
        column_id="id",
        column_sort="time",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=1,
    )


@pytest.fixture
def populated_registry(make_sequence):
    """Registry with 6 samples: P1-P3 positives, N1-N3 negatives, U1 unlabeled."""
    registry = LabelRegistry()
    plan = {
        "P1": 1, "P2": 1, "P3": 1,
        "N1": 0, "N2": 0, "N3": 0,
        "U1": None,
    }
    sequences = {}
    for business_id, label in plan.items():
        seq = make_sequence(business_id, positive=(label == 1))
        registry.register_sample(business_id, seq, "2026-01-01T00:00:00Z")
        sequences[business_id] = seq
        if label is not None:
            record = registry.submit_review(
                business_id, label, "2026-01-02T00:00:00Z"
            )
            registry.accept_review(
                record.lineage_id, record.review_version,
                "2026-01-02T01:00:00Z",
            )
    return registry, sequences
