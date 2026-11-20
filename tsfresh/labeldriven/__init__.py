# -*- coding: utf-8 -*-
"""Label-driven classification model generations on top of tsfresh.

This package implements a strict lifecycle for human-reviewed binary labels
and the classifier generations learned from them.

Core guarantees
---------------
* Human-reviewed labels are bound to an exact full-sequence revision plus a
  review version. Editing a sequence creates a new revision and expires the
  old review; the label never silently follows the edit.
* Only currently *accepted* 0/1 labels enter training. Pending, withdrawn and
  expired labels — as well as unlabeled samples — never act as negatives.
* Deleting a sample retires its lineage; reusing the business ID creates a
  fresh lineage that must be reviewed from scratch.
* Imputation statistics and native tsfresh feature selection are fitted on
  the effective training population only, then a brand-new classifier is
  fitted; the same fitted pipeline scores every current sample and returns
  ID-aligned class probabilities.
* Raw sequences, reviews, training sources, features, models and
  probabilities are persisted together and can be re-opened with integrity
  verification. Any conflict or failure during extraction, fitting or saving
  keeps the previous generation in service — results never mix new labels
  with an old model.
"""

from .exceptions import (
    ConflictError,
    GenerationBuildError,
    LabelDrivenError,
    NoTrainingPopulationError,
    StoreError,
)
from .generation import (
    Generation,
    TrainingSource,
    build_generation,
    resolve_training_source,
)
from .models import (
    LabelState,
    Lineage,
    ReviewRecord,
    Revision,
    SequenceState,
)
from .pipeline import ExtractionSpec, GenerationModel
from .registry import LabelRegistry, canonical_hash
from .store import LabelDrivenStore, build_and_publish

__all__ = [
    "ConflictError",
    "ExtractionSpec",
    "Generation",
    "GenerationBuildError",
    "GenerationModel",
    "LabelDrivenError",
    "LabelDrivenStore",
    "LabelRegistry",
    "LabelState",
    "Lineage",
    "NoTrainingPopulationError",
    "ReviewRecord",
    "Revision",
    "SequenceState",
    "StoreError",
    "TrainingSource",
    "build_and_publish",
    "build_generation",
    "canonical_hash",
    "resolve_training_source",
]
