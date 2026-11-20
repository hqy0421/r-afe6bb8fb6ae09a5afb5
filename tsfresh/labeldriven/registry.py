# -*- coding: utf-8 -*-
"""In-memory registry enforcing the label/sample lifecycle.

The registry is the single authority deciding *which labels are currently
valid*. It deliberately contains no modelling code. All mutation rules that
prevent human-reviewed labels from silently following a later sequence edit
live here:

* samples are maintained by full sequence revision;
* a review is bound to one exact sequence revision and carries its own
  review version;
* only an accepted review bound to the lineage's current revision yields a
  training label;
* pending / withdrawn / expired reviews never count as negatives;
* deleting a sample retires its lineage; reusing the same business ID
  afterwards creates a brand new lineage that must be reviewed again.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Dict, List, Optional

import pandas as pd

from .exceptions import ConflictError
from .models import (
    LabelState,
    Lineage,
    ReviewRecord,
    Revision,
    SequenceState,
    can_transition,
)


def canonical_hash(sequence: pd.DataFrame) -> str:
    """Stable SHA-256 over the raw sequence payload.

    Rows are sorted on every column after resetting the index so equivalent
    content hashes identically regardless of input row order or index.
    """
    if not isinstance(sequence, pd.DataFrame):
        raise TypeError("sequence must be a pandas.DataFrame")
    frame = sequence.reset_index(drop=True)
    parts = [
        ",".join(str(c) for c in frame.columns),
        frame.sort_values(by=list(frame.columns), kind="mergesort")
        .to_csv(index=False),
    ]
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class LabelRegistry:
    """Holds every lineage, its sequence revisions and its review records."""

    def __init__(self):
        # lineage_id -> Lineage
        self._lineages: Dict[str, Lineage] = {}
        # business_id -> lineage_id (only active lineages are mapped)
        self._business_index: Dict[str, str] = {}
        self._lineage_counter = 0
        self._review_counter = 0

    def reindex_counters(self) -> None:
        """Rebuild monotonic counters from stored records.

        Required after deserialization so newly minted lineage/review ids
        never collide with ids already present in a re-opened registry.
        """
        self._lineage_counter = 0
        for lineage_id in self._lineages:
            prefix = "lineage-"
            if lineage_id.startswith(prefix):
                try:
                    self._lineage_counter = max(
                        self._lineage_counter, int(lineage_id[len(prefix):])
                    )
                except ValueError:
                    continue
        self._review_counter = max(
            (
                review.review_version
                for lineage in self._lineages.values()
                for review in lineage.reviews
            ),
            default=0,
        )
        self._rebuild_business_index()

    def _rebuild_business_index(self) -> None:
        self._business_index = {
            lineage.business_id: lineage.lineage_id
            for lineage in self._lineages.values()
            if lineage.state is SequenceState.ACTIVE
        }

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def get_lineage(self, lineage_id: str) -> Lineage:
        try:
            return self._lineages[lineage_id]
        except KeyError as exc:
            raise ConflictError(f"Unknown lineage_id {lineage_id!r}.") from exc

    def resolve_lineage_id(self, business_id: str) -> Optional[str]:
        """Return the *active* lineage id for a business ID, if any."""
        return self._business_index.get(business_id)

    def all_lineages(self) -> List[Lineage]:
        return list(self._lineages.values())

    def active_lineages(self) -> List[Lineage]:
        return [
            lineage
            for lineage in self._lineages.values()
            if lineage.state is SequenceState.ACTIVE
        ]

    def current_training_labels(self) -> Dict[str, int]:
        """Map ``lineage_id -> 0/1`` for currently valid labels only.

        This is the sole label source exposed to training. Lineages without a
        valid accepted label are absent (not zero), so they can never be
        mistaken for negative examples.
        """
        labels = {}
        for lineage in self._lineages.values():
            label = lineage.current_accepted_label()
            if label is not None:
                labels[lineage.lineage_id] = label
        return labels

    # ------------------------------------------------------------------ #
    # sequence lifecycle
    # ------------------------------------------------------------------ #
    def register_sample(
        self, business_id: str, sequence: pd.DataFrame, created_at: str
    ) -> Lineage:
        """Create the first lineage/revision for a new business ID."""
        if business_id in self._business_index:
            raise ConflictError(
                f"Business ID {business_id!r} already has an active lineage; "
                "use append_revision to replace the sequence."
            )
        self._lineage_counter += 1
        lineage_id = f"lineage-{self._lineage_counter:06d}"
        lineage = Lineage(lineage_id=lineage_id, business_id=business_id)
        lineage.revisions.append(
            Revision(
                revision=1,
                content_hash=canonical_hash(sequence),
                created_at=created_at,
            )
        )
        self._lineages[lineage_id] = lineage
        self._business_index[business_id] = lineage_id
        return lineage

    def append_revision(
        self,
        business_id: str,
        sequence: pd.DataFrame,
        created_at: str,
    ) -> Revision:
        """Store a new full-sequence revision and expire stale reviews.

        Human-reviewed labels cannot follow an edit: every accepted review
        bound to the previous current revision is moved to ``EXPIRED`` and
        must be re-reviewed against the new revision. Pending requests are
        expired as well because they refer to superseded content.
        """
        lineage_id = self._business_index.get(business_id)
        if lineage_id is None:
            raise ConflictError(
                f"Business ID {business_id!r} has no active lineage; "
                "register it (or re-register after deletion) first."
            )
        lineage = self._lineages[lineage_id]
        new_number = lineage.revisions[-1].revision + 1
        new_revision = Revision(
            revision=new_number,
            content_hash=canonical_hash(sequence),
            created_at=created_at,
        )
        lineage.revisions.append(new_revision)
        for review in lineage.reviews:
            if review.revision == new_number - 1 and review.state in (
                LabelState.ACCEPTED,
                LabelState.PENDING,
            ):
                lineage.reviews[lineage.reviews.index(review)] = ReviewRecord(
                    lineage_id=review.lineage_id,
                    revision=review.revision,
                    review_version=review.review_version,
                    label=review.label,
                    state=LabelState.EXPIRED,
                    created_at=review.created_at,
                    note=review.note,
                    previous_state=review.state,
                )
        return new_revision

    def upsert_sequence(
        self,
        business_id: str,
        sequence: pd.DataFrame,
        created_at: str,
    ) -> Lineage:
        """Convenience: register a new sample or append a revision.

        No-op duplicate content still appends a revision only when callers
        ask explicitly; this method rejects identical consecutive content to
        keep revision numbers meaningful.
        """
        lineage_id = self._business_index.get(business_id)
        if lineage_id is None:
            return self.register_sample(business_id, sequence, created_at)
        lineage = self._lineages[lineage_id]
        digest = canonical_hash(sequence)
        if digest == lineage.current_revision.content_hash:
            raise ConflictError(
                f"Sequence for {business_id!r} is identical to its current "
                "revision; no new revision created."
            )
        self.append_revision(business_id, sequence, created_at)
        return lineage

    def delete_sample(self, business_id: str, deleted_at: str) -> None:
        """Retire a lineage. Its reviews stay in the log but become unusable.

        The business ID is freed. Re-registering it later mints a new
        lineage, so old reviews can never bind to the re-created sample.
        """
        lineage_id = self._business_index.pop(business_id, None)
        if lineage_id is None:
            raise ConflictError(
                f"Business ID {business_id!r} has no active lineage to delete."
            )
        lineage = self._lineages[lineage_id]
        lineage.state = SequenceState.DELETED
        # Mark every still-live review expired as part of retirement.
        for index, review in enumerate(lineage.reviews):
            if review.state in (LabelState.ACCEPTED, LabelState.PENDING):
                lineage.reviews[index] = ReviewRecord(
                    lineage_id=review.lineage_id,
                    revision=review.revision,
                    review_version=review.review_version,
                    label=review.label,
                    state=LabelState.EXPIRED,
                    created_at=review.created_at,
                    note=review.note,
                    previous_state=review.state,
                )
        lineage.revisions.append(
            Revision(
                revision=lineage.revisions[-1].revision + 1,
                content_hash="DELETED:" + lineage.current_revision.content_hash,
                created_at=deleted_at,
            )
        )

    # ------------------------------------------------------------------ #
    # review lifecycle
    # ------------------------------------------------------------------ #
    def submit_review(
        self,
        business_id: str,
        label: int,
        created_at: str,
        note: Optional[str] = None,
    ) -> ReviewRecord:
        """Create a PENDING review bound to the current sequence revision.

        Only one live (pending/accepted) review per lineage may exist, so a
        human cannot accidentally hold two contradictory active decisions.
        """
        lineage_id = self._business_index.get(business_id)
        if lineage_id is None:
            raise ConflictError(
                f"Cannot review {business_id!r}: no active lineage (deleted "
                "IDs must be re-registered and reviewed anew)."
            )
        lineage = self._lineages[lineage_id]
        if label not in (0, 1):
            raise ConflictError("Human labels must be binary 0/1.")
        live = [
            r
            for r in lineage.reviews
            if r.state in (LabelState.PENDING, LabelState.ACCEPTED)
        ]
        if live:
            raise ConflictError(
                f"Lineage {lineage_id} already has a live review; withdraw or "
                "let it expire before submitting another."
            )
        self._review_counter += 1
        record = ReviewRecord(
            lineage_id=lineage_id,
            revision=lineage.current_revision.revision,
            review_version=self._review_counter,
            label=label,
            state=LabelState.PENDING,
            created_at=created_at,
            note=note,
        )
        lineage.reviews.append(record)
        return record

    def decide_review(
        self,
        lineage_id: str,
        review_version: int,
        target: LabelState,
        decided_at: str,
    ) -> ReviewRecord:
        """Transition one review (pending -> accepted/withdrawn, etc.)."""
        lineage = self.get_lineage(lineage_id)
        record = lineage.review_for_version(review_version)
        if record is None:
            raise ConflictError(
                f"Review version {review_version} unknown for lineage "
                f"{lineage_id}."
            )
        if target is LabelState.ACCEPTED:
            # Re-acceptance must still refer to the current revision.
            if record.revision != lineage.current_revision.revision:
                raise ConflictError(
                    "Cannot accept a review bound to a non-current revision; "
                    "submit a new review for the current sequence."
                )
            if lineage.state is not SequenceState.ACTIVE:
                raise ConflictError(
                    "Lineage is deleted; its reviews cannot be accepted."
                )
        if not can_transition(record.state, target):
            raise ConflictError(
                f"Illegal review transition {record.state.value} -> "
                f"{target.value} for review {review_version}."
            )
        index = lineage.reviews.index(record)
        updated = ReviewRecord(
            lineage_id=record.lineage_id,
            revision=record.revision,
            review_version=record.review_version,
            label=record.label,
            state=target,
            created_at=decided_at,
            note=record.note,
            previous_state=record.state,
        )
        lineage.reviews[index] = updated
        return updated

    def accept_review(
        self, lineage_id: str, review_version: int, decided_at: str
    ) -> ReviewRecord:
        return self.decide_review(
            lineage_id, review_version, LabelState.ACCEPTED, decided_at
        )

    def withdraw_review(
        self, lineage_id: str, review_version: int, decided_at: str
    ) -> ReviewRecord:
        return self.decide_review(
            lineage_id, review_version, LabelState.WITHDRAWN, decided_at
        )

    # ------------------------------------------------------------------ #
    # snapshot helpers (used by persistence and conflict checks)
    # ------------------------------------------------------------------ #
    def snapshot(self) -> "LabelRegistry":
        """Deep copy, e.g. for building a generation without later mutation."""
        return copy.deepcopy(self)

    def assert_internally_consistent(self) -> None:
        """Defense-in-depth invariants check before a build or a save."""
        active_business = {}
        for lineage in self._lineages.values():
            if lineage.state is SequenceState.ACTIVE:
                if lineage.business_id in active_business:
                    raise ConflictError(
                        "Two active lineages share business ID "
                        f"{lineage.business_id!r}."
                    )
                active_business[lineage.business_id] = lineage.lineage_id
            revisions = [r.revision for r in lineage.revisions]
            if revisions != list(range(1, len(revisions) + 1)):
                raise ConflictError(
                    f"Lineage {lineage.lineage_id} has non-contiguous "
                    "revision numbers."
                )
            live = [
                r
                for r in lineage.reviews
                if r.state in (LabelState.PENDING, LabelState.ACCEPTED)
            ]
            if len(live) > 1:
                raise ConflictError(
                    f"Lineage {lineage.lineage_id} has {len(live)} live "
                    "reviews; at most one is allowed."
                )
            for review in live:
                if review.revision != lineage.current_revision.revision:
                    raise ConflictError(
                        f"Live review {review.review_version} on lineage "
                        f"{lineage.lineage_id} is bound to a stale revision."
                    )
        if set(active_business.values()) != set(self._business_index.values()):
            raise ConflictError("Business-ID index is out of sync.")
