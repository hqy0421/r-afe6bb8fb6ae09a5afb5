# -*- coding: utf-8 -*-
"""Exceptions for the label-driven classification generation subsystem.

The subsystem never silently degrades. Every integrity, lifecycle or build
problem is raised as a dedicated exception so callers can distinguish a
hard failure (which must keep the previous generation in service) from an
ordinary rejected state transition.
"""


class LabelDrivenError(Exception):
    """Base class for all errors raised by the ``labeldriven`` package."""


class ConflictError(LabelDrivenError):
    """Raised when an operation would violate the label/sample lifecycle.

    Conflicts include applying a human label to a sequence revision it was
    not reviewed against, attaching labels to deleted lineages, submitting a
    second active decision for the same lineage, or illegal state
    transitions. Such operations are rejected outright, never coerced.
    """


class GenerationBuildError(LabelDrivenError):
    """Raised when a new classifier generation cannot be built.

    The previous generation (if any) remains fully in service; a failed
    build never publishes a mixture of new labels and an old model.
    """


class StoreError(LabelDrivenError):
    """Raised on persistence problems or failed integrity verification."""


class NoTrainingPopulationError(GenerationBuildError):
    """Raised when the effective training population is unusable.

    Training needs at least one accepted ``0`` and one accepted ``1`` label;
    pending, withdrawn, expired or missing labels never count as negatives.
    """
