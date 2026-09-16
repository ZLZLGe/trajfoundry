"""Memory-bounded prefix aggregation for cumulative capture histories."""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from .canonical import compaction_signature
from .models import Message, Snapshot
from .tool_names import is_spawn_tool_name


def message_prefix_token(message: Message) -> bytes:
    """Return the stable conversational fields used for prefix matching.

    Reasoning payloads are intentionally excluded.  Providers can replay the
    same assistant turn with a reduced reasoning envelope (for example without
    response-only metadata), while the model-visible conversation and tool
    call remain identical.
    """

    if message.role == "assistant":
        value: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
            "tool_calls": [
                call.model_dump(mode="json", exclude_none=False)
                for call in message.tool_calls or ()
            ]
            if message.tool_calls is not None
            else None,
        }
    elif message.role == "tool":
        value = {
            "role": message.role,
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content,
        }
    else:
        value = {"role": message.role, "content": message.content}
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _compatibility(snapshot: Snapshot) -> tuple[Hashable, bytes]:
    """Return the scope in which a capture may be a cumulative prefix.

    A real session is deliberately paired with its thread.  Captures whose
    session is absent are allowed to merge by user, but never across users;
    using a distinct scope tag also prevents a user id from colliding with a
    session id. Public ``no_*`` sentinels are projected only after aggregation;
    internally, only the empty string means missing.
    """

    if snapshot.session_id:
        scope: Hashable = (
            "session",
            snapshot.session_id,
            snapshot.thread_id,
        )
    else:
        scope = ("user", snapshot.user_id)
    return scope, compaction_signature(snapshot.compaction_items)


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
    # SHA-256 selects a tiny collision bucket; the canonical token comparison
    # in ``advance`` remains authoritative.
    children: dict[bytes, list[tuple[bytes, _TrieNode]]] = field(default_factory=dict)
    ending_paths: set[str] = field(default_factory=set)
    # Only presence is queried. A refcount avoids one set entry per active leaf
    # at every history prefix while still allowing covered leaves to be removed.
    extender_count: int = 0
    endpoint_leaf_ids: set[int] = field(default_factory=set)


class _TranscriptTrie:
    def __init__(self) -> None:
        self.root = _TrieNode()

    @staticmethod
    def advance(node: _TrieNode, token: bytes) -> _TrieNode:
        bucket = node.children.setdefault(sha256(token).digest(), [])
        for existing, child in bucket:
            if existing == token:
                return child
        child = _TrieNode()
        bucket.append((token, child))
        return child


@dataclass(slots=True)
class _IndexedLeaf:
    snapshot: Snapshot
    prefix_nodes: tuple[_TrieNode, ...]
    endpoint_node: _TrieNode


@dataclass(frozen=True, slots=True)
class StreamingAggregationResult:
    leaves: tuple[Snapshot, ...]
    contributor_paths: dict[str, tuple[str, ...]]
    intermediate_paths: tuple[str, ...]
    spawn_evidence: tuple[Snapshot, ...]


def _has_spawn_response(snapshot: Snapshot) -> bool:
    return any(
        is_spawn_tool_name(call.function.name)
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
    spawn_names: dict[str, set[str]] = {}
    for name, call_id in (
        (call.function.name, call.id)
        for message in source
        if message.role == "assistant"
        for call in (message.tool_calls or ())
        if is_spawn_tool_name(call.function.name)
    ):
        spawn_names.setdefault(call_id, set()).add(name)
    result: list[Message] = []
    for message in source:
        if message.role == "tool" and message.name in spawn_names.get(
            message.tool_call_id or "", set()
        ):
            result.append(message.model_copy(deep=True))
            continue
        if message.role != "assistant":
            continue
        calls = [
            call
            for call in (message.tool_calls or ())
            if is_spawn_tool_name(call.function.name)
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
            "compaction_items": [],
            "model": "",
            "harness": "",
            "instructions": "",
            "termination": "",
            "issues": [],
            "multimodal_file_mapping": [],
        },
        deep=False,
    )


def streaming_prefix_leaves(
    snapshots: Iterable[Snapshot],
    *,
    scope_fn: Callable[[Snapshot], Hashable] | None = None,
) -> StreamingAggregationResult:
    """Return maximal exact-prefix leaves without scanning the active frontier.

    The trie is an online equivalent of processing captures from longest to
    shortest. Each node indexes active leaves that end there and active leaves
    whose history extends through it, so no active-frontier scan is required.
    Exact canonical tokens remain on trie edges; SHA-256 is only a child lookup
    accelerator. ``scope_fn`` can override the default session/user scope;
    compaction signatures are still part of compatibility.
    """

    all_paths: list[str] = []
    evidence: list[Snapshot] = []
    tries: dict[tuple[Hashable, bytes], _TranscriptTrie] = {}
    # Only active leaves remain here. Prefix postings store integer ids, so a
    # covered leaf's complete Snapshot and contributor set can be released
    # immediately instead of leaving one list slot per processed capture.
    leaves: dict[int, _IndexedLeaf] = {}
    next_leaf_id = 0

    for snapshot in snapshots:
        all_paths.append(snapshot.source_path)
        if (
            _has_spawn_response(snapshot)
            or _has_linkage(snapshot)
            or _has_agent_messages(snapshot)
        ):
            evidence.append(_evidence_copy(snapshot))

        if scope_fn is None:
            compatibility = _compatibility(snapshot)
        else:
            compatibility = (
                scope_fn(snapshot),
                compaction_signature(snapshot.compaction_items),
            )
        history_length = len(snapshot.history)
        full_tokens = tuple(
            message_prefix_token(message)
            for message in (*snapshot.history, *snapshot.response)
        )
        complete_length = len(full_tokens)

        trie = tries.setdefault(compatibility, _TranscriptTrie())
        path_nodes = [trie.root]
        for token in full_tokens:
            path_nodes.append(trie.advance(path_nodes[-1], token))
        endpoint_node = path_nodes[-1]

        # These postings are exact because reaching the same trie node proves
        # equality of every canonical token along the path. Equal terminal
        # transcripts remain separate: only strict history extenders are here.
        if endpoint_node.extender_count:
            endpoint_node.ending_paths.add(snapshot.source_path)
            continue

        # Only strict history-prefix nodes can cover an earlier leaf. For an
        # empty terminal capture this tuple is empty; for any non-empty capture
        # it includes the root and stops before the complete transcript.
        prefix_count = min(history_length, complete_length - 1) + 1
        prefix_nodes = tuple(path_nodes[:prefix_count]) if complete_length else ()
        covered_leaf_ids = {
            leaf_id for node in prefix_nodes for leaf_id in node.endpoint_leaf_ids
        }

        # Remove covered leaves from the index before inserting the new leaf.
        # Endpoint paths stay on trie nodes because later divergent branches
        # must inherit them, but the heavy Snapshot is released immediately.
        for leaf_id in covered_leaf_ids:
            candidate = leaves.pop(leaf_id, None)
            if candidate is None:
                continue
            for node in candidate.prefix_nodes:
                node.extender_count -= 1
            candidate.endpoint_node.endpoint_leaf_ids.discard(leaf_id)
        if covered_leaf_ids:
            del candidate

        leaf_id = next_leaf_id
        next_leaf_id += 1
        leaves[leaf_id] = _IndexedLeaf(
            snapshot=snapshot,
            prefix_nodes=prefix_nodes,
            endpoint_node=endpoint_node,
        )
        for node in prefix_nodes:
            node.extender_count += 1
        endpoint_node.endpoint_leaf_ids.add(leaf_id)
        # Each input path is stored at exactly one endpoint. This is the
        # irreducible lineage payload required by the return contract; unlike
        # the old active-leaf representation it does not retain the Snapshot.
        endpoint_node.ending_paths.add(snapshot.source_path)

    active_leaves = sorted(
        leaves.items(), key=lambda item: _stable_key(item[1].snapshot)
    )
    ordered = [leaf.snapshot for _, leaf in active_leaves]
    leaf_paths = {snapshot.source_path for snapshot in ordered}
    return StreamingAggregationResult(
        leaves=tuple(ordered),
        contributor_paths={
            leaf.snapshot.source_path: tuple(
                sorted(
                    {
                        leaf.snapshot.source_path,
                        *(
                            path
                            for node in leaf.prefix_nodes
                            for path in node.ending_paths
                        ),
                    }
                )
            )
            for _, leaf in active_leaves
        },
        intermediate_paths=tuple(
            sorted(path for path in all_paths if path not in leaf_paths)
        ),
        spawn_evidence=tuple(sorted(evidence, key=_stable_key)),
    )


__all__ = [
    "StreamingAggregationResult",
    "message_prefix_token",
    "streaming_prefix_leaves",
]
