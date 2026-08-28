"""Deterministic aggregation of cumulative provider snapshots.

Provider captures are cumulative: a request normally repeats the messages from
earlier turns in ``history`` and adds a new response.  This module removes only
snapshots for which that relationship is *proved*.  In particular, proximity
in time, similar text, and a shared session are deliberately not evidence of a
prefix relationship.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256

from .canonical import compaction_signature
from .models import Snapshot
from .streaming import message_prefix_token

GroupKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class PrefixLink:
    """Evidence that one input snapshot is continued by another.

    The indexes refer to the sequence passed to :func:`aggregate_snapshots`.
    ``extending_index`` is a deterministic, shortest known witness.  It can
    itself be an intermediate snapshot.
    """

    covered_index: int
    extending_index: int
    matched_messages: int
    group: GroupKey


@dataclass(frozen=True, slots=True)
class AggregationResult(Sequence[Snapshot]):
    """Maximal leaves and the evidence used to suppress intermediates.

    The result behaves as a read-only sequence of leaves for convenient use in
    a pipeline, while the explicit fields retain lineage for auditing.
    """

    leaves: tuple[Snapshot, ...]
    leaf_indices: tuple[int, ...]
    intermediate: tuple[Snapshot, ...]
    intermediate_indices: tuple[int, ...]
    prefix_links: tuple[PrefixLink, ...]

    def __iter__(self) -> Iterator[Snapshot]:
        return iter(self.leaves)

    def __len__(self) -> int:
        return len(self.leaves)

    def __getitem__(self, index: int | slice) -> Snapshot | tuple[Snapshot, ...]:
        return self.leaves[index]


def _advance_digest(state: object, token: bytes) -> object:
    # ``hashlib`` hash objects intentionally have no useful public protocol
    # type before Python 3.12.  Keeping this tiny helper local avoids a runtime
    # dependency merely for typing.
    next_state = state.copy()  # type: ignore[attr-defined]
    next_state.update(len(token).to_bytes(8, "big"))  # type: ignore[attr-defined]
    next_state.update(token)  # type: ignore[attr-defined]
    return next_state


def _sequence_digest(tokens: Sequence[bytes]) -> bytes:
    state = sha256()
    for token in tokens:
        state = _advance_digest(state, token)  # type: ignore[assignment]
    return state.digest()


def _group_key(snapshot: Snapshot) -> GroupKey:
    return (
        snapshot.session_id,
        snapshot.thread_id,
    )


def _stable_snapshot_key(
    snapshot: Snapshot, transcript_digest: bytes
) -> tuple[str, ...]:
    """Order outputs independently of discovery/worker completion order."""

    return (
        snapshot.session_id,
        snapshot.thread_id,
        snapshot.captured_at,
        snapshot.turn_id,
        snapshot.request_id,
        snapshot.source_path,
        snapshot.source_sha256,
        transcript_digest.hex(),
    )


def aggregate_snapshots(snapshots: Sequence[Snapshot]) -> AggregationResult:
    """Keep maximal exact-prefix leaves from cumulative snapshots.

    ``A`` is suppressed only when all of the following are true:

    * ``A`` and ``B`` have the same ``(session, thread)``;
    * their complete opaque-compaction records are identical, including
      provider origin and item index;
    * ``A.history + A.response`` equals the beginning of ``B.history``;
    * ``B`` has a strictly longer complete transcript.

    Hashes are lookup accelerators, never the final equality check.  Each
    message participates in a constant number of hash operations and each
    suppressed snapshot is visited once after it is matched, avoiding the
    quadratic all-pairs comparison used by naive implementations.
    """

    items = tuple(snapshots)
    if not items:
        return AggregationResult((), (), (), (), ())

    history_tokens: list[tuple[bytes, ...]] = []
    full_tokens: list[tuple[bytes, ...]] = []
    full_digests: list[bytes] = []
    for snapshot in items:
        history = tuple(message_prefix_token(message) for message in snapshot.history)
        response = tuple(message_prefix_token(message) for message in snapshot.response)
        full = history + response
        history_tokens.append(history)
        full_tokens.append(full)
        full_digests.append(_sequence_digest(full))

    # Full transcripts are indexed once. The session/thread group is part of
    # the key, making cross-thread matching impossible by construction.
    compaction_signatures = [
        compaction_signature(snapshot.compaction_items) for snapshot in items
    ]
    index: dict[tuple[GroupKey, bytes, int, bytes], list[int]] = {}
    for input_index, snapshot in enumerate(items):
        key = (
            _group_key(snapshot),
            compaction_signatures[input_index],
            len(full_tokens[input_index]),
            full_digests[input_index],
        )
        index.setdefault(key, []).append(input_index)

    stable_keys = [
        _stable_snapshot_key(snapshot, full_digests[i])
        for i, snapshot in enumerate(items)
    ]
    # Shorter extenders are examined first.  Consequently, the first witness
    # saved for a covered snapshot is its shortest deterministic continuation.
    extenders = sorted(
        range(len(items)),
        key=lambda i: (len(full_tokens[i]), stable_keys[i]),
    )

    covered_by: dict[int, int] = {}
    for extending_index in extenders:
        snapshot = items[extending_index]
        history = history_tokens[extending_index]
        complete_length = len(full_tokens[extending_index])
        if complete_length == 0:
            continue

        state = sha256()
        # The empty transcript is a valid strict prefix as well.  It is rare
        # in accepted data, but treating it consistently avoids a special-case
        # leaf if upstream sends an empty first response.
        empty_key = (
            _group_key(snapshot),
            compaction_signatures[extending_index],
            0,
            state.digest(),
        )
        for covered_index in index.get(empty_key, ()):
            if covered_index != extending_index and covered_index not in covered_by:
                covered_by[covered_index] = extending_index

        for prefix_length, token in enumerate(history, start=1):
            state = _advance_digest(state, token)  # type: ignore[assignment]
            # A same-length transcript is not an earlier cumulative snapshot.
            if prefix_length >= complete_length:
                continue
            lookup_key = (
                _group_key(snapshot),
                compaction_signatures[extending_index],
                prefix_length,
                state.digest(),
            )
            for covered_index in index.get(lookup_key, ()):
                if covered_index == extending_index or covered_index in covered_by:
                    continue
                # Defend against a digest collision and document that exact
                # structural message equality, rather than hash equality, is
                # the authority.
                if full_tokens[covered_index] != history[:prefix_length]:
                    continue
                covered_by[covered_index] = extending_index

    intermediate_indices = tuple(sorted(covered_by, key=lambda i: stable_keys[i]))
    leaf_indices = tuple(
        sorted(
            (i for i in range(len(items)) if i not in covered_by),
            key=lambda i: stable_keys[i],
        )
    )
    links = tuple(
        PrefixLink(
            covered_index=i,
            extending_index=covered_by[i],
            matched_messages=len(full_tokens[i]),
            group=_group_key(items[i]),
        )
        for i in intermediate_indices
    )
    return AggregationResult(
        leaves=tuple(items[i] for i in leaf_indices),
        leaf_indices=leaf_indices,
        intermediate=tuple(items[i] for i in intermediate_indices),
        intermediate_indices=intermediate_indices,
        prefix_links=links,
    )


def maximal_prefix_leaves(snapshots: Sequence[Snapshot]) -> tuple[Snapshot, ...]:
    """Convenience wrapper returning only the maximal leaves."""

    return aggregate_snapshots(snapshots).leaves


__all__ = [
    "AggregationResult",
    "GroupKey",
    "PrefixLink",
    "aggregate_snapshots",
    "maximal_prefix_leaves",
]
