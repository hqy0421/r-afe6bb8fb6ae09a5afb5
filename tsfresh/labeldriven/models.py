# -*- coding: utf-8 -*-
"""Enumerations and immutable value objects for the label-driven subsystem.

Lifecycle summary
------------------
A *lineage* is the durable identity behind a business sample ID. A sequence
that is deleted and later recreated under the same business ID gets a fresh
lineage, so reviews bound to the old lineage can never leak onto the
re-created sample.

A *revision* is one immutable version of a lineage's full sequence. Every
human decision is bound to the exact lineage + sequence revision and carries
its own review version, making the provenance of every label explicit.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class LabelState(str, enum.Enum):
    """Lifecycle of a human review decision.

    Only ``ACCEPTED`` decisions participate in training. ``PENDING`` is an
    undecided request, ``WITHDRAWN`` retracts an earlier decision and
    ``EXPIRED`` marks a decision whose bound sequence revision is no longer
    current. None of the non-accepted states are ever treated as a negative
    class.
    """

    PENDING = "pending"
    ACCEPTED = "accepted"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"

    @property
    def is_training_label(self) -> bool:
        return self is LabelState.ACCEPTED


# Transitions that may be requested on a single review record. Anything not
# listed here is a ConflictError (reviews are otherwise immutable).
_ALLOWED_TRANSITIONS = {
    LabelState.PENDING: {LabelState.ACCEPTED, LabelState.WITHDRAWN},
    LabelState.ACCEPTED: {
        LabelState.WITHDRAWN,
        LabelState.EXPIRED,
    },
    LabelState.WITHDRAWN: set(),
    LabelState.EXPIRED: set(),
}


class SequenceState(str, enum.Enum):
    """Whether a lineage currently holds a live sequence."""

    ACTIVE = "active"
    DELETED = "deleted"


@dataclass(frozen=True)
class Revision:
    """One immutable, content-addressed sequence revision.

    :param revision: monotonically increasing revision number per lineage,
        starting at 1.
    :param content_hash: SHA-256 hex digest of the canonicalized raw
        sequence payload, proving which bytes the revision contains.
    :param created_at: ISO-8601 UTC timestamp supplied by the caller/calendar
        (the subsystem itself never relies on wall-clock randomness).
    """

    revision: int
    content_hash: str
    created_at: str


@dataclass(frozen=True)
class ReviewRecord:
    """A human review decision bound to an exact sequence revision.

    The triple (lineage_id, revision, review_version) is unique. A review is
    valid only while ``revision`` equals the lineage's current revision and
    the lineage is active; otherwise it is stale and must not drive training.
    """

    lineage_id: str
    revision: int
    review_version: int
    label: int  # 0 or 1, only meaningful once ACCEPTED
    state: LabelState
    created_at: str
    note: Optional[str] = None
    # Bookkeeping kept inside the value object so a persisted record is fully
    # self-describing.
    previous_state: Optional[LabelState] = None

    def __post_init__(self):
        if self.label not in (0, 1):
            raise ValueError("Human labels must be binary 0/1.")
        if not isinstance(self.state, LabelState):
            object.__setattr__(self, "state", LabelState(self.state))


@dataclass
class Lineage:
    """A sample identity with its full revision history and review log.

    ``business_id`` is what callers know the sample as; ``lineage_id`` is the
    internal identity that changes when the ID is reused after deletion.
    """

    lineage_id: str
    business_id: str
    state: SequenceState = SequenceState.ACTIVE
    revisions: list = field(default_factory=list)  # list[Revision]
    reviews: list = field(default_factory=list)  # list[ReviewRecord]

    @property
    def current_revision(self) -> Optional[Revision]:
        return self.revisions[-1] if self.revisions else None

    def review_for_version(self, review_version: int) -> Optional[ReviewRecord]:
        for record in self.reviews:
            if record.review_version == review_version:
                return record
        return None

    def current_accepted_label(self) -> Optional[int]:
        """Return the accepted 0/1 label bound to the *current* revision.

        Returns ``None`` when there is no usable label: no accepted review,
        the accepted review is bound to an older revision, or the lineage was
        deleted. Pending/withdrawn/expired records never produce a label.
        """
        if self.state is not SequenceState.ACTIVE or not self.revisions:
            return None
        current_number = self.revisions[-1].revision
        candidates = [
            r
            for r in self.reviews
            if r.state is LabelState.ACCEPTED and r.revision == current_number
        ]
        if not candidates:
            return None
        # At most one active accepted review per lineage is allowed by the
        # registry; defensively take the highest review version.
        return max(candidates, key=lambda r: r.review_version).label


def can_transition(current: LabelState, target: LabelState) -> bool:
    return target in _ALLOWED_TRANSITIONS[current]
