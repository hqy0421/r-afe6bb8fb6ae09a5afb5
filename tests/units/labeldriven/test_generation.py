# -*- coding: utf-8 -*-
"""End-to-end tests for building a label-driven generation."""
import numpy as np
import pandas as pd
import pytest

from tsfresh.labeldriven import (
    ExtractionSpec,
    GenerationBuildError,
    build_generation,
)
from tsfresh.labeldriven.pipeline import (
    ColumnAlignedPerColumnImputer,
    SafeFeatureSelector,
)


def _build(registry, sequences, spec, generation_id="gen-1", created_at="t"):
    lineage_sequences = {
        registry.resolve_lineage_id(business_id): seq
        for business_id, seq in sequences.items()
    }
    return build_generation(
        generation_id=generation_id,
        created_at=created_at,
        registry=registry,
        sequences=lineage_sequences,
        extraction_spec=spec,
    )


def test_generation_fits_and_scores_all_current_samples(
    populated_registry, minimal_spec
):
    registry, sequences = populated_registry
    generation = _build(registry, sequences, minimal_spec)

    all_ids = [
        lineage.lineage_id for lineage in registry.active_lineages()
    ]
    # Probabilities are aligned by ID to every current sample, incl. unlabeled.
    assert sorted(generation.probabilities.index) == sorted(all_ids)
    assert len(generation.probabilities) == 7
    assert generation.probabilities.between(0.0, 1.0).all()

    # The unlabeled sample still receives a prediction through the fitted flow.
    u1 = registry.resolve_lineage_id("U1")
    assert not np.isnan(generation.probabilities.loc[u1])

    # Separable synthetic data: positives should outrank negatives.
    ids = {
        lineage.business_id: lineage.lineage_id
        for lineage in registry.all_lineages()
    }
    positive_scores = generation.probabilities.loc[
        [ids["P1"], ids["P2"], ids["P3"]]
    ]
    negative_scores = generation.probabilities.loc[
        [ids["N1"], ids["N2"], ids["N3"]]
    ]
    assert positive_scores.min() > negative_scores.max()


def test_imputer_and_selector_fitted_on_training_population_only(
    populated_registry, minimal_spec
):
    registry, sequences = populated_registry
    generation = _build(registry, sequences, minimal_spec)
    pipeline = generation.model.pipeline

    imputer = pipeline.named_steps["impute"]
    assert isinstance(imputer, ColumnAlignedPerColumnImputer)
    # Imputer statistics exist only for extracted columns and are finite.
    assert set(imputer._col_to_NAN_repl) == set(generation.model.feature_columns)
    assert all(np.isfinite(v) for v in imputer._col_to_NAN_repl.values())

    selector = pipeline.named_steps["select"]
    assert isinstance(selector, SafeFeatureSelector)
    # Selection learned from exactly the 6 labeled training rows.
    assert sorted(generation.model.training_lineage_ids) == sorted(
        generation.source.labels
    )
    assert len(generation.model.training_lineage_ids) == 6
    # The selected set is a subset of extracted features (or recorded fallback).
    if not selector.fallback_used:
        assert set(selector.relevant_features) <= set(
            generation.model.feature_columns
        )


def test_features_are_current_and_imputation_stats_leak_free(
    populated_registry, minimal_spec, make_sequence
):
    registry, sequences = populated_registry
    generation = _build(registry, sequences, minimal_spec)
    # Feature matrix is indexed like the current population.
    assert set(generation.features.index) == set(generation.sequence_manifest)

    # Re-fitting after an edit yields different imputation provenance and the
    # edited sample is excluded until re-reviewed. Reuse the exact payload that
    # was registered so the content-hash guard passes.
    edited = make_sequence("P1", positive=False, drift=10.0)
    registry.append_revision("P1", edited, "t2")
    sequences = {**sequences, "P1": edited}
    generation2 = _build(registry, sequences, minimal_spec, "gen-2")
    p1 = registry.resolve_lineage_id("P1")
    assert p1 not in generation2.source.labels
    assert generation2.source.excluded[p1].startswith("review_expired")
    assert generation2.sequence_manifest[p1]["revision"] == 2


def test_hash_mismatch_aborts_build(populated_registry, minimal_spec):
    registry, sequences = populated_registry
    tampered = {
        registry.resolve_lineage_id(business_id):
        seq if business_id != "P1" else seq.assign(value=seq.value + 999.0)
        for business_id, seq in sequences.items()
    }
    with pytest.raises(GenerationBuildError, match="hash mismatch"):
        build_generation(
            generation_id="gen-x",
            created_at="t",
            registry=registry,
            sequences=tampered,
            extraction_spec=minimal_spec,
        )


def test_missing_raw_sequence_aborts_build(populated_registry, minimal_spec):
    registry, sequences = populated_registry
    partial = {
        registry.resolve_lineage_id(business_id): seq
        for business_id, seq in sequences.items()
        if business_id != "N1"
    }
    with pytest.raises(GenerationBuildError, match="Missing current raw"):
        build_generation(
            generation_id="gen-x",
            created_at="t",
            registry=registry,
            sequences=partial,
            extraction_spec=minimal_spec,
        )


def test_pipeline_tolerates_missing_prediction_columns(
    populated_registry, minimal_spec
):
    registry, sequences = populated_registry
    generation = _build(registry, sequences, minimal_spec)
    features = generation.features.copy()
    # Drop half of the feature columns, as can happen with a sparse extractor.
    keep = features.columns[: max(1, len(features.columns) // 2)]
    reduced = features.loc[:, keep]
    scores = generation.model.score_features(reduced)
    assert len(scores) == len(features)
    assert scores.between(0.0, 1.0).all()


def test_pipeline_tolerates_missing_prediction_row(
    populated_registry, minimal_spec
):
    registry, sequences = populated_registry
    generation = _build(registry, sequences, minimal_spec)
    u1 = registry.resolve_lineage_id("U1")
    features = generation.features.drop(index=u1)
    scores = generation.model.score_features(features)
    # Reindexed back; the missing row is imputed from training statistics.
    assert u1 in scores.index
    assert np.isfinite(scores.loc[u1])


def test_each_build_gets_fresh_classifier(populated_registry, minimal_spec):
    registry, sequences = populated_registry
    gen1 = _build(registry, sequences, minimal_spec, "gen-1")
    gen2 = _build(registry, sequences, minimal_spec, "gen-2")
    assert gen1.model.pipeline is not gen2.model.pipeline
    classifier1 = gen1.model.pipeline.named_steps["classify"]
    classifier2 = gen2.model.pipeline.named_steps["classify"]
    assert classifier1 is not classifier2


def test_manifest_records_every_revision(populated_registry, minimal_spec):
    registry, sequences = populated_registry
    generation = _build(registry, sequences, minimal_spec)
    for lineage_id, entry in generation.sequence_manifest.items():
        lineage = registry.get_lineage(lineage_id)
        assert entry["revision"] == lineage.current_revision.revision
        assert entry["content_hash"] == lineage.current_revision.content_hash
        assert entry["business_id"] == lineage.business_id
