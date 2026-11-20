# -*- coding: utf-8 -*-
"""Fitted classification pipeline for one label-driven generation.

The pipeline composes three fitted stages plus tsfresh feature extraction:

1. **Feature extraction** is performed on the *current* raw sequences via the
   native :func:`tsfresh.feature_extraction.extraction.extract_features`
   entry point. The extracted feature matrix is re-indexed and
   column-aligned, so samples whose sequence produces no row or missing
   columns still receive a probability.
2. **Imputation** uses tsfresh's native
   :class:`~tsfresh.transformers.per_column_imputer.PerColumnImputer`, fitted
   on the effective training population only.
3. **Native feature selection** uses tsfresh's
   :class:`~tsfresh.transformers.feature_selector.FeatureSelector` (Benjamini
   Hochberg significance control), likewise fitted on the training
   population only.
4. A fresh binary classifier is fit on the selected training features.

Nothing in the imputer, selector or classifier is reused from an older
generation: a build always fits a brand new pipeline against the current
labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tsfresh.feature_extraction import extract_features
from tsfresh.transformers.feature_selector import FeatureSelector
from tsfresh.transformers.per_column_imputer import PerColumnImputer

from .exceptions import GenerationBuildError


@dataclass
class ExtractionSpec:
    """How raw sequences are laid out and which tsfresh features to compute.

    Mirrors the column arguments of
    :func:`~tsfresh.feature_extraction.extraction.extract_features`.
    """

    column_id: str
    column_sort: Optional[str] = None
    column_kind: Optional[str] = None
    column_value: Optional[str] = None
    # None means ComprehensiveFCParameters (the tsfresh default).
    default_fc_parameters: object = None
    n_jobs: int = 1
    disable_progressbar: bool = True


class ColumnAlignedPerColumnImputer(PerColumnImputer):
    """Imputer that tolerates columns unseen at prediction time.

    Extra columns are dropped; columns known from training but absent at
    prediction time are re-created and filled entirely from the fitted
    replacement dictionaries (all-NaN), so every current sample can be scored
    with the exact imputation statistics learned on the training population.
    """

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        known = list(self._col_to_NAN_repl.keys())
        aligned = X.reindex(columns=known)
        return super().transform(aligned)


class SafeFeatureSelector(FeatureSelector):
    """tsfresh selector that also handles an empty relevance result.

    If no native feature survives Benjamini Hochberg at the configured FDR
    level, the pipeline falls back to *all* imputed feature columns rather
    than crashing; the manifest records that the fallback was used, so no
    silent rule change occurs.
    """

    def fit(self, X, y):
        self.fallback_used = False
        return super().fit(X, y)

    def transform(self, X):
        if not self.relevant_features:
            self.fallback_used = True
            return X.copy()
        # Relevant columns are a subset of the fitted frame; guard against
        # prediction frames missing a column.
        return X.copy().reindex(columns=self.relevant_features, fill_value=0.0)


def build_sklearn_pipeline(classifier=None) -> Pipeline:
    """Assemble the fitted stages (imputation -> selection -> classifier)."""
    if classifier is None:
        classifier = LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=42
        )
    return Pipeline(
        steps=[
            ("impute", ColumnAlignedPerColumnImputer()),
            ("select", SafeFeatureSelector(n_jobs=1)),
            ("classify", classifier),
        ]
    )


@dataclass
class GenerationModel:
    """Everything fitted for a generation and required to re-score samples."""

    generation_id: str
    pipeline: Pipeline
    extraction_spec: ExtractionSpec
    feature_columns: List[str]
    training_lineage_ids: List[str]
    label_counts: dict
    # Every lineage ID that was current at build time, so a prediction frame
    # missing a row can be re-expanded and imputed rather than silently
    # dropping that sample from the output.
    all_lineage_ids: List[str] = field(default_factory=list)
    selector_fallback_used: bool = False
    metadata: dict = field(default_factory=dict)

    def _extract(self, raw_long: pd.DataFrame) -> pd.DataFrame:
        """Run native tsfresh extraction and align to training feature names."""
        frame = extract_features(
            raw_long,
            default_fc_parameters=self.extraction_spec.default_fc_parameters,
            column_id=self.extraction_spec.column_id,
            column_sort=self.extraction_spec.column_sort,
            column_kind=self.extraction_spec.column_kind,
            column_value=self.extraction_spec.column_value,
            n_jobs=self.extraction_spec.n_jobs,
            disable_progressbar=self.extraction_spec.disable_progressbar,
            impute_function=None,
        )
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise GenerationBuildError(
                "Native tsfresh extraction returned no features."
            )
        # Keep extraction raw; the fitted pipeline imputes. Replace infinities
        # with NaN so the imputer owns every non-finite decision.
        frame = frame.reindex(columns=self.feature_columns)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        return frame

    def score_features(self, features: pd.DataFrame) -> pd.Series:
        """Score an already-extracted, column-aligned feature matrix.

        Rows are re-indexed to the full set of lineage IDs that were current
        when the generation was built; a sample absent from the extraction
        output becomes an all-NaN row and is filled by the imputer from
        training statistics, so no current sample can disappear from the
        result.
        """
        aligned = features.reindex(
            index=self.all_lineage_ids, columns=self.feature_columns
        )
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        probabilities = self.pipeline.predict_proba(aligned)
        classes = list(self.pipeline.named_steps["classify"].classes_)
        if 1 not in classes:
            raise GenerationBuildError(
                "Fitted classifier does not expose class 1 probabilities."
            )
        positive_index = classes.index(1)
        return pd.Series(
            probabilities[:, positive_index],
            index=list(self.all_lineage_ids),
            name="p_class_1",
        )

    def predict_proba(
        self, raw_long: pd.DataFrame, all_lineage_ids: List[str]
    ) -> pd.Series:
        """Extract from raw sequences and return class-1 probabilities.

        Convenience wrapper used when a previously saved generation is
        re-opened and re-scored against a freshly extracted feature matrix.
        The raw payload's id column may carry business-level IDs, so rows are
        re-tagged with the requested lineage IDs in order before extraction.
        During a build, :meth:`score_features` is used on the already
        extracted matrix to avoid computing features twice.
        """
        ordered = []
        raw_groups = self._group_raw_by_id(raw_long)
        if len(raw_groups) != len(all_lineage_ids):
            raise GenerationBuildError(
                f"Expected {len(all_lineage_ids)} raw sequences, got "
                f"{len(raw_groups)}."
            )
        for lineage_id, group in zip(all_lineage_ids, raw_groups):
            group = group.copy()
            group[self.extraction_spec.column_id] = lineage_id
            ordered.append(group)
        tagged = pd.concat(ordered, ignore_index=True)
        features = self._extract(tagged)
        features = features.reindex(index=list(all_lineage_ids))
        return self.score_features(features)

    def _group_raw_by_id(self, raw_long: pd.DataFrame):
        """Return raw sequence groups in the order of their first occurrence."""
        id_column = self.extraction_spec.column_id
        if id_column not in raw_long.columns:
            # Single un-tagged frame: treat the whole input as one sequence.
            return [raw_long]
        ordered_ids = list(dict.fromkeys(raw_long[id_column].tolist()))
        return [
            raw_long[raw_long[id_column] == sample_id] for sample_id in ordered_ids
        ]
