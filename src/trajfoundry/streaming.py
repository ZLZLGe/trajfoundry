"""Memory-bounded prefix aggregation for cumulative capture histories."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from .models import Message, Snapshot


def _token(message: Message) -> bytes:
    return json.dumps(
        message.model_dump(mode="json", exclude_none=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _compatibility(snapshot: Snapshot) -> tuple[str, ...]:
    return (
        snapshot.source_partition,
        snapshot.session_id,
        snapshot.thread_id,
        snapshot.provider,
        snapshot.model,
        snapshot.harness,
        snapshot.instructions,
    )


def _stable_key(snapshot: Snapshot) -> tuple[str, ...]:
    return (
        snapshot.captured_at,
        snapshot.turn_id,
        snapshot.request_id,
        snapshot.source_path,
        snapshot.source_sha256,
    )


@dataclass(slots=True)
class _TrieNode:
    # A digest is only a lookup accelerator. Each bucket retains the exact
    # canonical token so a synthetic or accidental collision cannot merge data.
    children: dict[bytes, list[tuple[bytes, int]]] = field(default_factory=dict)
    ending_paths: set[str] = field(default_factory=set)


class _TranscriptTrie:
    def __init__(self) -> None:
        self.nodes = [_TrieNode()]

    def advance(self, node_index: int, token: bytes) -> int:
        digest = sha256(token).digest()
        bucket = self.nodes[node_index].children.setdefault(digest, [])
        for existing, child_index in bucket:
            if existing == token:
                return child_index
        child_index = len(self.nodes)
        self.nodes.append(_TrieNode())
        bucket.append((token, child_index))
        return child_index


@dataclass(frozen=True, slots=True)
class StreamingAggregationResult:
    leaves: tuple[Snapshot, ...]
    contributor_paths: dict[str, tuple[str, ...]]
    intermediate_paths: tuple[str, ...]
    spawn_evidence: tuple[Snapshot, ...]


def _has_spawn_response(snapshot: Snapshot) -> bool:
    return any(
        call.function.name in {"spawn_agent", "Agent"}
        for message in snapshot.response
        for call in (message.tool_calls or [])
    )


def _has_linkage(snapshot: Snapshot) -> bool:
    return bool(
        snapshot.subagent_marker
        or snapshot.parent_thread_id
        or snapshot.parent_turn_id
        or snapshot.forked_from_thread_id
    )


def _has_agent_messages(snapshot: Snapshot) -> bool:
    return bool(snapshot.agent_messages)


def _spawn_only_messages(messages: Iterable[Message]) -> list[Message]:
    source = list(messages)
    spawn_ids = {
        call.id
        for message in source
        if message.role == "assistant"
        for call in (message.tool_calls or ())
        if call.function.name in {"spawn_agent", "Agent"}
    }
    result: list[Message] = []
    for message in source:
        if message.role == "tool" and message.tool_call_id in spawn_ids:
            result.append(message.model_copy(deep=True))
            continue
        if message.role != "assistant":
            continue
        calls = [
            call
            for call in (message.tool_calls or ())
            if call.function.name in {"spawn_agent", "Agent"}
        ]
        if not calls:
            continue
        result.append(
            message.model_copy(
                update={
                    "content": "",
                    "reasoning_content": "",
                    "reasoning_details": None,
                    "reasoning": None,
                    "tool_calls": calls,
                },
                deep=True,
            )
        )
    return result


def _routing_agent_messages(snapshot: Snapshot) -> list[Any]:
    return [
        record.model_copy(
            update={
                "item": {
                    key: record.item[key]
                    for key in ("type", "id", "author", "recipient")
                    if key in record.item
                }
            },
            deep=True,
        )
        for record in snapshot.agent_messages
    ]


def _evidence_copy(snapshot: Snapshot) -> Snapshot:
    return snapshot.model_copy(
        update={
            "history": _spawn_only_messages(snapshot.history),
            "response": _spawn_only_messages(snapshot.response),
            "tools": [],
            "server_tool_calls": [],
            "agent_messages": _routing_agent_messages(snapshot),
            "model": "",
            "harness": "",
            "instructions": "",
            "termination": "",
            "issues": [],
        },
        deep=False,
    )


def _strict_history_prefix(prefix: list[Message], candidate: Snapshot) -> bool:
    complete_length = len(candidate.history) + len(candidate.response)
    return (
        len(prefix) < complete_length
        and len(prefix) <= len(candidate.history)
        and candidate.history[: len(prefix)] == prefix
    )


def streaming_prefix_leaves(
    snapshots: Iterable[Snapshot],
) -> StreamingAggregationResult:
    """Return maximal exact-prefix leaves without retaining repeated histories.

    A trie interns each distinct canonical message once per transcript prefix.
    Active full snapshots are retained because they are the only possible
    leaves. Late, out-of-order shorter captures are checked exactly against
    those active leaves. Thus a long cumulative chain uses memory proportional
    to its unique transcript plus its branch frontier, rather than the sum of
    every repeated request history.
    """

    tries: dict[tuple[str, ...], _TranscriptTrie] = {}
    active: dict[tuple[tuple[str, ...], int], list[Snapshot]] = {}
    contributors: dict[str, set[str]] = {}
    all_paths: list[str] = []
    evidence: list[Snapshot] = []

    for snapshot in snapshots:
        all_paths.append(snapshot.source_path)
        if (
            _has_spawn_response(snapshot)
            or _has_linkage(snapshot)
            or _has_agent_messages(snapshot)
        ):
            evidence.append(_evidence_copy(snapshot))

        compatibility = _compatibility(snapshot)
        trie = tries.setdefault(compatibility, _TranscriptTrie())
        full_messages = [*snapshot.history, *snapshot.response]
        inherited: set[str] = set()

        node_index = 0
        if full_messages:
            inherited.update(trie.nodes[0].ending_paths)
            removed = active.pop((compatibility, 0), ())
            for ancestor in removed:
                contributors.pop(ancestor.source_path, None)
        for depth, message in enumerate(snapshot.history, start=1):
            node_index = trie.advance(node_index, _token(message))
            if depth < len(full_messages):
                inherited.update(trie.nodes[node_index].ending_paths)
                removed = active.pop((compatibility, node_index), ())
                for ancestor in removed:
                    contributors.pop(ancestor.source_path, None)

        for message in snapshot.response:
            node_index = trie.advance(node_index, _token(message))
        full_node = trie.nodes[node_index]
        inherited.add(snapshot.source_path)
        full_node.ending_paths.add(snapshot.source_path)
        active_key = (compatibility, node_index)

        descendants = [
            candidate
            for key, candidates in active.items()
            if key[0] == compatibility
            for candidate in candidates
            if _strict_history_prefix(full_messages, candidate)
        ]
        if descendants:
            for candidate in descendants:
                contributors[candidate.source_path].update(inherited)
            continue

        active.setdefault(active_key, []).append(snapshot)
        contributors[snapshot.source_path] = inherited

    leaves = [snapshot for candidates in active.values() for snapshot in candidates]
    ordered = sorted(leaves, key=_stable_key)
    leaf_paths = {snapshot.source_path for snapshot in ordered}
    return StreamingAggregationResult(
        leaves=tuple(ordered),
        contributor_paths={
            snapshot.source_path: tuple(sorted(contributors[snapshot.source_path]))
            for snapshot in ordered
        },
        intermediate_paths=tuple(
            sorted(path for path in all_paths if path not in leaf_paths)
        ),
        spawn_evidence=tuple(sorted(evidence, key=_stable_key)),
    )


__all__ = ["StreamingAggregationResult", "streaming_prefix_leaves"]
