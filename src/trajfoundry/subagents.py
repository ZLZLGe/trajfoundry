"""Evidence-based mounting of sub-agent trajectory leaves.

The important property of this module is what it refuses to do: it never uses
timestamps, nearby files, or free-form message text to infer parentage.  A
mount requires structured thread/turn metadata and a concrete ``spawn_agent``
tool-call id.  All unresolved cases remain explicit orphans with stable
diagnostics.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .audit_codes import MountDiagnosticCode
from .models import (
    AgentMessageRecord,
    AuditIssue,
    AuditTag,
    Message,
    NormalizationAudit,
    Snapshot,
    ToolCall,
    TrajectoryNode,
)
from .quality import enrich_trajectory


@dataclass(frozen=True, slots=True)
class MountDiagnostic:
    """A stable, machine-readable reason why mounting was not complete."""

    code: MountDiagnosticCode
    detail: str
    leaf_index: int | None = None
    related_leaf_indices: tuple[int, ...] = ()
    parent_thread_id: str = ""
    parent_turn_id: str = ""
    spawn_call_id: str = ""


@dataclass(frozen=True, slots=True)
class SpawnEvidence:
    """A turn-bound ``spawn_agent`` call found in a provider response."""

    source_partition: str
    session_id: str
    parent_thread_id: str
    parent_turn_id: str
    spawn_call_id: str
    task_name: str
    agent_name: str
    agent_name_conflicted: bool
    arguments_json: str
    source_path: str

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.source_partition,
            self.session_id,
            self.parent_thread_id,
            self.parent_turn_id,
            self.spawn_call_id,
        )


@dataclass(frozen=True, slots=True)
class MountEdge:
    """One unambiguous, acyclic mount in canonical leaf-index space."""

    parent_index: int
    child_index: int
    spawn_call_id: str
    parent_turn_id: str
    task_name: str = ""
    agent_name: str = ""
    relay_id: str = ""


@dataclass(frozen=True, slots=True)
class SubagentMountPlan:
    """A deterministic mount plan over canonically ordered leaves."""

    leaves: tuple[Snapshot, ...]
    edges: tuple[MountEdge, ...]
    main_root_indices: tuple[int, ...]
    orphan_indices: tuple[int, ...]
    incomplete_parent_indices: tuple[int, ...]
    spawn_call_counts: tuple[int, ...]
    spawn_evidence: tuple[SpawnEvidence, ...]
    diagnostics: tuple[MountDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class SnapshotTrajectory:
    """Associate linkage metadata with the materialized trajectory node."""

    snapshot: Snapshot
    trajectory: TrajectoryNode


@dataclass(frozen=True, slots=True)
class MountResult:
    roots: tuple[TrajectoryNode, ...]
    root_snapshots: tuple[Snapshot, ...]
    orphans: tuple[TrajectoryNode, ...]
    orphan_snapshots: tuple[Snapshot, ...]
    plan: SubagentMountPlan


@dataclass(frozen=True, slots=True)
class _RoutingMarker:
    raw: str
    spawn_call_id: str = ""
    task_name: str = ""
    parent_thread_id: str = ""
    parent_turn_id: str = ""
    forked_from_thread_id: str = ""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _leaf_key(snapshot: Snapshot) -> tuple[str, ...]:
    transcript = [
        message.model_dump(mode="json", exclude_none=False)
        for message in (*snapshot.history, *snapshot.response)
    ]
    return (
        snapshot.source_partition,
        snapshot.session_id,
        snapshot.thread_id,
        snapshot.captured_at,
        snapshot.turn_id,
        snapshot.request_id,
        snapshot.source_path,
        snapshot.source_sha256,
        _canonical_json(transcript),
    )


def _as_object(arguments: Any) -> tuple[dict[str, Any] | None, bool]:
    if isinstance(arguments, dict):
        return arguments, False
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, True
        if isinstance(parsed, dict):
            return parsed, False
    return None, True


def _walk_dicts(value: Any) -> list[Mapping[str, Any]]:
    """Return nested metadata objects without interpreting free-form text."""

    if not isinstance(value, Mapping):
        return []
    result: list[Mapping[str, Any]] = [value]
    # Known wrappers used by Codex turn metadata.  Restricting traversal to
    # dictionaries (rather than searching arbitrary strings) keeps this
    # structured evidence.
    for key in ("sub_agent", "subagent", "routing", "thread_source"):
        child = value.get(key)
        if isinstance(child, Mapping):
            result.extend(_walk_dicts(child))
    return result


def _first_string(objects: Sequence[Mapping[str, Any]], *keys: str) -> str:
    for obj in objects:
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _parse_routing_marker(raw: str) -> _RoutingMarker:
    marker = raw.strip()
    if not marker:
        return _RoutingMarker(raw=raw)

    parsed: Any = None
    if marker.startswith("{"):
        try:
            parsed = json.loads(marker)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None

    objects = _walk_dicts(parsed)
    if objects:
        return _RoutingMarker(
            raw=raw,
            spawn_call_id=_first_string(
                objects, "spawn_call_id", "call_id", "tool_call_id"
            ),
            task_name=_first_string(objects, "task_name"),
            parent_thread_id=_first_string(objects, "parent_thread_id"),
            parent_turn_id=_first_string(objects, "parent_turn_id"),
            forked_from_thread_id=_first_string(
                objects, "forked_from_thread_id", "forked_from"
            ),
        )

    # Key/value markers are also structured and occur in a few harnesses.
    fields: dict[str, str] = {}
    for part in marker.replace(";", ",").split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    if fields:
        return _RoutingMarker(
            raw=raw,
            spawn_call_id=fields.get("spawn_call_id", fields.get("call_id", "")),
            task_name=fields.get("task_name", ""),
            parent_thread_id=fields.get("parent_thread_id", ""),
            parent_turn_id=fields.get("parent_turn_id", ""),
            forked_from_thread_id=fields.get(
                "forked_from_thread_id", fields.get("forked_from", "")
            ),
        )

    # An opaque marker still proves that this is a sub-agent.  It is used for
    # routing only when it exactly equals a call id or task_name.
    return _RoutingMarker(raw=raw)


def _spawn_calls(messages: Sequence[Message]) -> list[ToolCall]:
    result: list[ToolCall] = []
    for message in messages:
        if message.role != "assistant":
            continue
        for call in message.tool_calls or ():
            if call.function.name in {"spawn_agent", "Agent"}:
                result.append(call)
    return result


def _spawn_agent_result_names(
    snapshots: Sequence[Snapshot],
) -> dict[tuple[str, str, str, str], set[str]]:
    """Collect canonical agent names from ordered ``spawn_agent`` results.

    The tool result is accepted only when its call is present earlier in the
    same normalized transcript and its content is a JSON object containing a
    non-empty string ``task_name``.  No free-form result text is interpreted.
    """

    names: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for snapshot in snapshots:
        seen_spawn_ids: set[str] = set()
        for message in (*snapshot.history, *snapshot.response):
            if message.role == "assistant":
                seen_spawn_ids.update(
                    call.id
                    for call in (message.tool_calls or ())
                    if call.function.name == "spawn_agent" and call.id
                )
                continue
            if (
                message.role != "tool"
                or message.name != "spawn_agent"
                or not message.tool_call_id
                or message.tool_call_id not in seen_spawn_ids
            ):
                continue
            try:
                value = json.loads(message.content)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(value, Mapping):
                continue
            task_name = value.get("task_name")
            if not isinstance(task_name, str) or not task_name:
                continue
            names[
                (
                    snapshot.source_partition,
                    snapshot.session_id,
                    snapshot.thread_id,
                    message.tool_call_id,
                )
            ].add(task_name)
    return names


def _agent_field(record: Any, field: str) -> str:
    value = record.item.get(field)
    return value if isinstance(value, str) and value else ""


def _agent_message_signature(record: Any) -> str:
    # Routing conflicts concern only provider-owned routing fields.  Content
    # (including encrypted blocks) is deliberately opaque to this stage.
    return _canonical_json(
        {
            field: record.item.get(field)
            for field in ("type", "id", "author", "recipient")
        }
    )


def _agent_message_index(
    canonical_leaves: Sequence[Snapshot],
    evidence_source: Sequence[Snapshot],
) -> tuple[
    dict[tuple[str, str, str], tuple[Any, ...]],
    set[tuple[str, str, str, str]],
    list[MountDiagnostic],
]:
    """Deduplicate cumulative replays and reject ambiguous message ids."""

    variants: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(dict)
    duplicate_ids: set[tuple[str, str, str, str]] = set()
    for snapshot in sorted(evidence_source, key=_leaf_key):
        thread_key = (
            snapshot.source_partition,
            snapshot.session_id,
            snapshot.thread_id,
        )
        capture_counts: dict[str, int] = defaultdict(int)
        for record in snapshot.agent_messages:
            message_id = _agent_field(record, "id")
            if not message_id:
                continue
            key = (*thread_key, message_id)
            capture_counts[message_id] += 1
            variants[key].setdefault(_agent_message_signature(record), record)
        duplicate_ids.update(
            (*thread_key, message_id)
            for message_id, count in capture_counts.items()
            if count > 1
        )

    conflict_ids = {key for key, values in variants.items() if len(values) > 1}
    blocked_ids = duplicate_ids | conflict_ids
    records_by_thread: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for key, values in variants.items():
        if key in blocked_ids:
            continue
        records_by_thread[key[:3]].append(next(iter(values.values())))
    for records in records_by_thread.values():
        records.sort(key=lambda record: (record.origin, record.item_index))

    diagnostics: list[MountDiagnostic] = []
    for key in sorted(duplicate_ids):
        partition, session_id, thread_id, message_id = key
        targets = [
            index
            for index, leaf in enumerate(canonical_leaves)
            if (leaf.source_partition, leaf.session_id, leaf.thread_id)
            == (partition, session_id, thread_id)
        ]
        diagnostics.append(
            MountDiagnostic(
                code="duplicate_agent_message_id",
                detail=f"agent_message id {message_id!r} occurs more than once",
                leaf_index=targets[0] if len(targets) == 1 else None,
                related_leaf_indices=tuple(targets) if len(targets) > 1 else (),
                parent_thread_id=thread_id,
            )
        )
    for key in sorted(conflict_ids):
        partition, session_id, thread_id, message_id = key
        targets = [
            index
            for index, leaf in enumerate(canonical_leaves)
            if (leaf.source_partition, leaf.session_id, leaf.thread_id)
            == (partition, session_id, thread_id)
        ]
        diagnostics.append(
            MountDiagnostic(
                code="conflicting_agent_message_evidence",
                detail=f"agent_message id {message_id!r} has conflicting raw items",
                leaf_index=targets[0] if len(targets) == 1 else None,
                related_leaf_indices=tuple(targets) if len(targets) > 1 else (),
                parent_thread_id=thread_id,
            )
        )
    return (
        {key: tuple(value) for key, value in records_by_thread.items()},
        blocked_ids,
        diagnostics,
    )


def _contains_spawn(snapshot: Snapshot, evidence: SpawnEvidence) -> bool:
    for call in _spawn_calls((*snapshot.history, *snapshot.response)):
        if call.id != evidence.spawn_call_id:
            continue
        arguments, malformed = _as_object(call.function.arguments)
        if malformed or arguments is None:
            # The id is already globally scoped by parent thread and turn.  A
            # malformed repeated copy cannot improve or contradict the event.
            return True
        if _canonical_json(arguments) == evidence.arguments_json:
            return True
    return False


def _is_subagent(snapshot: Snapshot) -> bool:
    return bool(
        snapshot.subagent_marker
        or snapshot.parent_thread_id
        or snapshot.parent_turn_id
        or snapshot.forked_from_thread_id
    )


def _diagnostic_sort_key(diagnostic: MountDiagnostic) -> tuple[Any, ...]:
    return (
        diagnostic.code,
        -1 if diagnostic.leaf_index is None else diagnostic.leaf_index,
        diagnostic.related_leaf_indices,
        diagnostic.parent_thread_id,
        diagnostic.parent_turn_id,
        diagnostic.spawn_call_id,
        diagnostic.detail,
    )


def _extract_spawn_evidence(
    snapshots: Sequence[Snapshot],
) -> tuple[tuple[SpawnEvidence, ...], list[MountDiagnostic]]:
    diagnostics: list[MountDiagnostic] = []
    by_key: dict[tuple[str, str, str, str, str], SpawnEvidence] = {}
    conflicts: set[tuple[str, str, str, str, str]] = set()
    result_names = _spawn_agent_result_names(snapshots)
    reported_result_conflicts: set[tuple[str, str, str, str]] = set()

    for snapshot in sorted(snapshots, key=_leaf_key):
        calls = _spawn_calls(snapshot.response)
        if calls and not snapshot.turn_id:
            diagnostics.append(
                MountDiagnostic(
                    code="spawn_missing_turn_id",
                    detail=f"spawn response in {snapshot.source_path!r} has no turn_id",
                    parent_thread_id=snapshot.thread_id,
                )
            )
            continue
        for call in calls:
            if not call.id:
                diagnostics.append(
                    MountDiagnostic(
                        code="spawn_missing_call_id",
                        detail=f"spawn response in {snapshot.source_path!r} has no call id",
                        parent_thread_id=snapshot.thread_id,
                        parent_turn_id=snapshot.turn_id,
                    )
                )
                continue
            arguments, malformed = _as_object(call.function.arguments)
            if malformed or arguments is None:
                diagnostics.append(
                    MountDiagnostic(
                        code="malformed_spawn_arguments",
                        detail=f"spawn call {call.id!r} arguments are not a JSON object",
                        parent_thread_id=snapshot.thread_id,
                        parent_turn_id=snapshot.turn_id,
                        spawn_call_id=call.id,
                    )
                )
                arguments = {}
            task_value = arguments.get("task_name")
            task_name = task_value if isinstance(task_value, str) else ""
            result_key = (
                snapshot.source_partition,
                snapshot.session_id,
                snapshot.thread_id,
                call.id,
            )
            canonical_names = result_names.get(result_key, set())
            if len(canonical_names) > 1 and result_key not in reported_result_conflicts:
                reported_result_conflicts.add(result_key)
                diagnostics.append(
                    MountDiagnostic(
                        code="conflicting_spawn_agent_name",
                        detail=(
                            "spawn_agent result replays contain conflicting "
                            "canonical task_name values"
                        ),
                        parent_thread_id=snapshot.thread_id,
                        parent_turn_id=snapshot.turn_id,
                        spawn_call_id=call.id,
                    )
                )
            agent_name = (
                next(iter(canonical_names), "") if len(canonical_names) == 1 else ""
            )
            event = SpawnEvidence(
                source_partition=snapshot.source_partition,
                session_id=snapshot.session_id,
                parent_thread_id=snapshot.thread_id,
                parent_turn_id=snapshot.turn_id,
                spawn_call_id=call.id,
                task_name=task_name,
                agent_name=agent_name,
                agent_name_conflicted=len(canonical_names) > 1,
                arguments_json=_canonical_json(arguments),
                source_path=snapshot.source_path,
            )
            existing = by_key.get(event.key)
            if existing is None:
                by_key[event.key] = event
            elif (
                existing.arguments_json != event.arguments_json
                or existing.task_name != event.task_name
            ):
                conflicts.add(event.key)

    for key in sorted(conflicts):
        event = by_key.pop(key)
        diagnostics.append(
            MountDiagnostic(
                code="conflicting_spawn_evidence",
                detail="the same turn/call id has conflicting structured arguments",
                parent_thread_id=event.parent_thread_id,
                parent_turn_id=event.parent_turn_id,
                spawn_call_id=event.spawn_call_id,
            )
        )
    events = tuple(
        sorted(
            by_key.values(),
            key=lambda event: (
                event.source_partition,
                event.session_id,
                event.parent_thread_id,
                event.parent_turn_id,
                event.spawn_call_id,
                event.task_name,
                event.agent_name,
                event.source_path,
            ),
        )
    )
    return events, diagnostics


def _validate_child(
    snapshot: Snapshot, marker: _RoutingMarker, leaf_index: int
) -> list[MountDiagnostic]:
    diagnostics: list[MountDiagnostic] = []

    def add(code: MountDiagnosticCode, detail: str) -> None:
        diagnostics.append(
            MountDiagnostic(
                code=code,
                detail=detail,
                leaf_index=leaf_index,
                parent_thread_id=snapshot.parent_thread_id,
                parent_turn_id=snapshot.parent_turn_id,
                spawn_call_id=marker.spawn_call_id,
            )
        )

    if not snapshot.subagent_marker:
        add(
            "missing_subagent_marker",
            "sub-agent linkage fields exist but marker is empty",
        )
    if not snapshot.parent_thread_id:
        add("missing_parent_thread_id", "sub-agent has no parent_thread_id")
    if not snapshot.parent_turn_id:
        add("missing_parent_turn_id", "sub-agent has no parent_turn_id")
    marker_checks = (
        ("parent_thread_id", marker.parent_thread_id, snapshot.parent_thread_id),
        ("parent_turn_id", marker.parent_turn_id, snapshot.parent_turn_id),
        (
            "forked_from_thread_id",
            marker.forked_from_thread_id,
            snapshot.forked_from_thread_id,
        ),
    )
    for name, marker_value, snapshot_value in marker_checks:
        if marker_value and marker_value != snapshot_value:
            add(
                "conflicting_marker_metadata",
                f"marker {name} conflicts with the normalized linkage field",
            )
    return diagnostics


def _parent_agent_name(agent_name: str) -> str:
    parent, separator, _ = agent_name.rpartition("/")
    return parent if separator and parent else ""


def _child_agent_recipient(
    records: Sequence[AgentMessageRecord],
) -> tuple[str, bool]:
    """Return one trustworthy direct-child recipient and whether it conflicts.

    Only inbound history items can identify the child created by a parent turn.
    Replayed copies with the same identity collapse naturally.  Provider records
    with duplicate/conflicting ids are not safe routing evidence, while multiple
    distinct valid direct-child recipients are contradictory evidence.
    """

    recipients = {
        recipient
        for record in records
        if record.origin == "history"
        and record.item.get("type") == "agent_message"
        and (author := _agent_field(record, "author"))
        and (recipient := _agent_field(record, "recipient"))
        and _parent_agent_name(recipient) == author
    }
    if len(recipients) > 1:
        return "", True
    return next(iter(recipients), ""), False


def _filter_spawns_by_recipient(
    candidates: Sequence[SpawnEvidence], recipient: str
) -> list[SpawnEvidence]:
    """Use canonical agent identity first, then a strict task-name fallback."""

    canonical = [
        event
        for event in candidates
        if event.agent_name and event.agent_name == recipient
    ]
    if canonical:
        return canonical

    task_basename = recipient.rsplit("/", 1)[-1]
    return [
        event
        for event in candidates
        if not event.agent_name
        and not event.agent_name_conflicted
        and event.task_name
        and event.task_name == task_basename
    ]


def _attach_relay_ids(
    edges: Sequence[MountEdge],
    leaves: Sequence[Snapshot],
    agent_records_by_thread: Mapping[
        tuple[str, str, str], Sequence[AgentMessageRecord]
    ],
) -> tuple[list[MountEdge], list[MountDiagnostic]]:
    """Attach a relay only after its unique spawn call/result pair completes."""

    diagnostics: list[MountDiagnostic] = []
    by_parent: dict[int, list[MountEdge]] = defaultdict(list)
    for edge in edges:
        by_parent[edge.parent_index].append(edge)

    updated: list[MountEdge] = []
    for parent_index, parent_edges in sorted(by_parent.items()):
        parent = leaves[parent_index]
        parent_thread_key = (
            parent.source_partition,
            parent.session_id,
            parent.thread_id,
        )
        valid_records = list(agent_records_by_thread.get(parent_thread_key, ()))
        relay_eligible_edges: set[tuple[int, str]] = set()
        for edge in parent_edges:
            if not edge.agent_name:
                continue
            author_matches = [
                record
                for record in valid_records
                if _agent_field(record, "author") == edge.agent_name
            ]
            if not author_matches:
                # Relay metadata is optional.  Without a parent-side candidate
                # there is nothing to validate against the child identity.
                continue
            child = leaves[edge.child_index]
            child_thread_key = (
                child.source_partition,
                child.session_id,
                child.thread_id,
            )
            child_agent_name, conflicting_child_agent_name = _child_agent_recipient(
                agent_records_by_thread.get(child_thread_key, ())
            )
            if conflicting_child_agent_name:
                diagnostics.append(
                    MountDiagnostic(
                        code="conflicting_child_agent_name",
                        detail=(
                            "sub-agent thread has multiple structured recipient "
                            "identities; relay is not attached"
                        ),
                        leaf_index=parent_index,
                        parent_thread_id=parent.thread_id,
                        parent_turn_id=edge.parent_turn_id,
                        spawn_call_id=edge.spawn_call_id,
                    )
                )
                continue
            if not child_agent_name:
                diagnostics.append(
                    MountDiagnostic(
                        code="missing_child_agent_name",
                        detail=(
                            "spawn_agent has a canonical result name but the child "
                            "thread has no unique agent_message recipient; relay is "
                            "not attached"
                        ),
                        leaf_index=parent_index,
                        parent_thread_id=parent.thread_id,
                        parent_turn_id=edge.parent_turn_id,
                        spawn_call_id=edge.spawn_call_id,
                    )
                )
                continue
            if child_agent_name != edge.agent_name:
                diagnostics.append(
                    MountDiagnostic(
                        code="conflicting_child_agent_name",
                        detail=(
                            "child agent_message recipient does not match the "
                            "canonical spawn_agent result name; relay is not attached"
                        ),
                        leaf_index=parent_index,
                        parent_thread_id=parent.thread_id,
                        parent_turn_id=edge.parent_turn_id,
                        spawn_call_id=edge.spawn_call_id,
                    )
                )
                continue
            relay_eligible_edges.add((edge.child_index, edge.spawn_call_id))

        known_agents = {edge.agent_name for edge in parent_edges if edge.agent_name}
        known_spawns = {edge.spawn_call_id for edge in parent_edges}
        expected_parents = {
            value
            for value in (_parent_agent_name(name) for name in known_agents)
            if value
        }

        # An incoming child-shaped message after one of this parent's direct
        # spawn call/result pairs, but from no known spawned agent, is
        # structured bogus evidence.  It is never assigned by proximity.
        for record in valid_records:
            author = _agent_field(record, "author")
            recipient = _agent_field(record, "recipient")
            if (
                known_agents
                and author not in known_agents
                and recipient in expected_parents
                and author.startswith(f"{recipient}/")
                and known_spawns.intersection(record.preceding_completed_spawn_call_ids)
            ):
                diagnostics.append(
                    MountDiagnostic(
                        code="unmatched_agent_message",
                        detail=(
                            f"agent_message id {_agent_field(record, 'id')!r} "
                            "comes from no canonical direct child"
                        ),
                        leaf_index=parent_index,
                        parent_thread_id=parent.thread_id,
                    )
                )

        for edge in sorted(
            parent_edges, key=lambda value: (value.spawn_call_id, value.child_index)
        ):
            if (edge.child_index, edge.spawn_call_id) not in relay_eligible_edges:
                updated.append(edge)
                continue
            expected_recipient = _parent_agent_name(edge.agent_name)
            author_matches = [
                record
                for record in valid_records
                if _agent_field(record, "author") == edge.agent_name
            ]
            recipient_matches = [
                record
                for record in author_matches
                if not expected_recipient
                or _agent_field(record, "recipient") == expected_recipient
            ]
            if author_matches and not recipient_matches:
                diagnostics.append(
                    MountDiagnostic(
                        code="agent_relay_recipient_mismatch",
                        detail=(
                            "agent_message author matches the child but recipient "
                            "does not match the canonical parent agent"
                        ),
                        leaf_index=parent_index,
                        parent_thread_id=parent.thread_id,
                        parent_turn_id=edge.parent_turn_id,
                        spawn_call_id=edge.spawn_call_id,
                    )
                )
                updated.append(edge)
                continue
            ordered = [
                record
                for record in recipient_matches
                if edge.spawn_call_id in record.preceding_completed_spawn_call_ids
            ]
            if recipient_matches and not ordered:
                diagnostics.append(
                    MountDiagnostic(
                        code="agent_relay_before_spawn",
                        detail=(
                            "matching agent_message has no structured evidence "
                            "that the spawn call/result pair completed before it"
                        ),
                        leaf_index=parent_index,
                        parent_thread_id=parent.thread_id,
                        parent_turn_id=edge.parent_turn_id,
                        spawn_call_id=edge.spawn_call_id,
                    )
                )
                updated.append(edge)
                continue
            if len(ordered) > 1:
                diagnostics.append(
                    MountDiagnostic(
                        code="ambiguous_agent_relay",
                        detail=(
                            "multiple ordered agent_message records match the "
                            "same spawned child"
                        ),
                        leaf_index=parent_index,
                        parent_thread_id=parent.thread_id,
                        parent_turn_id=edge.parent_turn_id,
                        spawn_call_id=edge.spawn_call_id,
                    )
                )
                updated.append(edge)
                continue
            if not ordered:
                updated.append(edge)
                continue
            relay_id = _agent_field(ordered[0], "id")
            if relay_id == edge.spawn_call_id:
                diagnostics.append(
                    MountDiagnostic(
                        code="relay_id_conflict",
                        detail="agent_message relay id equals its spawn call id",
                        leaf_index=parent_index,
                        parent_thread_id=parent.thread_id,
                        parent_turn_id=edge.parent_turn_id,
                        spawn_call_id=edge.spawn_call_id,
                    )
                )
                updated.append(edge)
                continue
            updated.append(replace(edge, relay_id=relay_id))

    relay_groups: dict[tuple[int, str], list[MountEdge]] = defaultdict(list)
    for edge in updated:
        if edge.relay_id:
            relay_groups[(edge.parent_index, edge.relay_id)].append(edge)
    conflicts = {key: values for key, values in relay_groups.items() if len(values) > 1}
    if conflicts:
        conflict_edges = {
            (edge.parent_index, edge.child_index, edge.spawn_call_id)
            for values in conflicts.values()
            for edge in values
        }
        for (parent_index, relay_id), values in sorted(conflicts.items()):
            diagnostics.append(
                MountDiagnostic(
                    code="relay_id_conflict",
                    detail=(
                        f"agent_message relay id {relay_id!r} matches multiple "
                        "spawned children"
                    ),
                    leaf_index=parent_index,
                    related_leaf_indices=tuple(
                        sorted(edge.child_index for edge in values)
                    ),
                    parent_thread_id=leaves[parent_index].thread_id,
                )
            )
        updated = [
            replace(edge, relay_id="")
            if (edge.parent_index, edge.child_index, edge.spawn_call_id)
            in conflict_edges
            else edge
            for edge in updated
        ]
    return updated, diagnostics


def _cycle_nodes(edges: Sequence[MountEdge]) -> set[int]:
    """Return exactly the nodes in directed cycles (not their descendants)."""

    child_to_parent = {edge.child_index: edge.parent_index for edge in edges}
    done: set[int] = set()
    cycles: set[int] = set()
    for start in sorted(child_to_parent):
        if start in done:
            continue
        path: list[int] = []
        position: dict[int, int] = {}
        node = start
        while node in child_to_parent and node not in done:
            if node in position:
                cycles.update(path[position[node] :])
                break
            position[node] = len(path)
            path.append(node)
            node = child_to_parent[node]
        done.update(path)
    return cycles


def plan_subagent_mounts(
    leaves: Sequence[Snapshot],
    *,
    all_snapshots: Sequence[Snapshot] | None = None,
) -> SubagentMountPlan:
    """Build an auditable mount plan without mutating trajectory nodes.

    ``leaves`` should be the maximal snapshots returned by the aggregation
    stage.  Pass the pre-aggregation collection as ``all_snapshots`` so spawn
    calls can remain bound to their original ``turn_id`` even when that turn's
    snapshot was removed as an intermediate.
    """

    canonical_leaves = tuple(sorted(leaves, key=_leaf_key))
    evidence_source = (
        tuple(all_snapshots) if all_snapshots is not None else canonical_leaves
    )
    events, diagnostics = _extract_spawn_evidence(evidence_source)
    agent_records_by_thread, _blocked_agent_ids, agent_diagnostics = (
        _agent_message_index(canonical_leaves, evidence_source)
    )
    diagnostics.extend(agent_diagnostics)

    thread_leaves: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, snapshot in enumerate(canonical_leaves):
        thread_leaves[
            (snapshot.source_partition, snapshot.session_id, snapshot.thread_id)
        ].append(index)

    # Bind every spawn event to exactly one maximal parent leaf before looking
    # at children.  This also exposes a parent branch ambiguity directly.
    parent_for_event: dict[tuple[str, str, str, str, str], int] = {}
    parent_linkage_incomplete: set[int] = set()
    spawn_call_counts = [0] * len(canonical_leaves)
    for event in events:
        candidates = [
            index
            for index in thread_leaves.get(
                (
                    event.source_partition,
                    event.session_id,
                    event.parent_thread_id,
                ),
                (),
            )
            if _contains_spawn(canonical_leaves[index], event)
        ]
        if not candidates:
            diagnostics.append(
                MountDiagnostic(
                    code="missing_parent_leaf",
                    detail="no maximal parent leaf contains the indexed spawn call",
                    parent_thread_id=event.parent_thread_id,
                    parent_turn_id=event.parent_turn_id,
                    spawn_call_id=event.spawn_call_id,
                )
            )
        elif len(candidates) > 1:
            parent_linkage_incomplete.update(candidates)
            for candidate in candidates:
                spawn_call_counts[candidate] += 1
            diagnostics.append(
                MountDiagnostic(
                    code="ambiguous_parent_leaf",
                    detail="multiple maximal parent branches contain the same spawn call",
                    related_leaf_indices=tuple(candidates),
                    parent_thread_id=event.parent_thread_id,
                    parent_turn_id=event.parent_turn_id,
                    spawn_call_id=event.spawn_call_id,
                )
            )
        else:
            parent_for_event[event.key] = candidates[0]
            spawn_call_counts[candidates[0]] += 1

    events_by_turn: dict[tuple[str, str, str, str], list[SpawnEvidence]] = defaultdict(
        list
    )
    for event in events:
        events_by_turn[
            (
                event.source_partition,
                event.session_id,
                event.parent_thread_id,
                event.parent_turn_id,
            )
        ].append(event)

    subagent_indices = {
        index
        for index, snapshot in enumerate(canonical_leaves)
        if _is_subagent(snapshot)
    }
    invalid_children: set[int] = set()

    # Linkage metadata is captured on every request in a child thread.  A
    # changing parent, parent turn, fork source, or marker is contradictory
    # evidence even if prefix aggregation happened to retain only one leaf.
    linkage_evidence: dict[tuple[str, str, str], list[Snapshot]] = defaultdict(list)
    for snapshot in evidence_source:
        if _is_subagent(snapshot):
            linkage_evidence[
                (snapshot.source_partition, snapshot.session_id, snapshot.thread_id)
            ].append(snapshot)
    for child_index in sorted(subagent_indices):
        child = canonical_leaves[child_index]
        thread_key = (child.source_partition, child.session_id, child.thread_id)
        records = linkage_evidence.get(thread_key, [child])
        checks = (
            ("parent_thread_id", {record.parent_thread_id for record in records}),
            ("parent_turn_id", {record.parent_turn_id for record in records}),
            (
                "forked_from_thread_id",
                {record.forked_from_thread_id for record in records},
            ),
            ("subagent_marker", {record.subagent_marker for record in records}),
        )
        for field_name, values in checks:
            if len(values) <= 1:
                continue
            invalid_children.add(child_index)
            code: MountDiagnosticCode = (
                "conflicting_marker_metadata"
                if field_name == "subagent_marker"
                else "conflicting_parent_thread"
            )
            diagnostics.append(
                MountDiagnostic(
                    code=code,
                    detail=(
                        f"sub-agent thread has inconsistent {field_name} values "
                        "across captured turns"
                    ),
                    leaf_index=child_index,
                    parent_thread_id=child.parent_thread_id,
                    parent_turn_id=child.parent_turn_id,
                )
            )

    # More than one maximal leaf for one child thread means a genuine branch
    # (or incompatible configurations).  Metadata at thread granularity cannot
    # tell which leaf is the spawned execution, so none is guessed.
    for thread_key, indices in sorted(thread_leaves.items()):
        child_indices = [index for index in indices if index in subagent_indices]
        if len(child_indices) > 1:
            invalid_children.update(child_indices)
            for index in child_indices:
                diagnostics.append(
                    MountDiagnostic(
                        code="duplicate_child_leaves",
                        detail="multiple maximal leaves share this sub-agent thread identity",
                        leaf_index=index,
                        related_leaf_indices=tuple(child_indices),
                        parent_thread_id=canonical_leaves[index].parent_thread_id,
                        parent_turn_id=canonical_leaves[index].parent_turn_id,
                    )
                )

    proposed: list[MountEdge] = []
    for child_index in sorted(subagent_indices):
        child = canonical_leaves[child_index]
        marker = _parse_routing_marker(child.subagent_marker)
        child_diagnostics = _validate_child(child, marker, child_index)
        diagnostics.extend(child_diagnostics)
        if child_diagnostics or child_index in invalid_children:
            invalid_children.add(child_index)
            continue

        turn_key = (
            child.source_partition,
            child.session_id,
            child.parent_thread_id,
            child.parent_turn_id,
        )
        candidates = list(events_by_turn.get(turn_key, ()))
        if not candidates:
            diagnostics.append(
                MountDiagnostic(
                    code="missing_spawn_call",
                    detail="parent thread/turn has no indexed spawn_agent call",
                    leaf_index=child_index,
                    parent_thread_id=child.parent_thread_id,
                    parent_turn_id=child.parent_turn_id,
                )
            )
            invalid_children.add(child_index)
            continue

        # Explicit structured fields are constraints, never hints.  Opaque
        # markers are considered only as exact call-id/task-name matches.
        if marker.spawn_call_id:
            candidates = [
                event
                for event in candidates
                if event.spawn_call_id == marker.spawn_call_id
            ]
        if marker.task_name:
            candidates = [
                event for event in candidates if event.task_name == marker.task_name
            ]
        opaque = marker.raw.strip()
        if not marker.spawn_call_id and not marker.task_name and len(candidates) > 1:
            exact = [
                event
                for event in candidates
                if opaque in (event.spawn_call_id, event.task_name)
            ]
            if exact:
                candidates = exact

        if not candidates:
            diagnostics.append(
                MountDiagnostic(
                    code="spawn_routing_mismatch",
                    detail="structured child routing matches no spawn call on the parent turn",
                    leaf_index=child_index,
                    parent_thread_id=child.parent_thread_id,
                    parent_turn_id=child.parent_turn_id,
                    spawn_call_id=marker.spawn_call_id,
                )
            )
            invalid_children.add(child_index)
            continue

        child_thread_key = (
            child.source_partition,
            child.session_id,
            child.thread_id,
        )
        recipient, conflicting_recipient = _child_agent_recipient(
            agent_records_by_thread.get(child_thread_key, ())
        )
        if conflicting_recipient:
            diagnostics.append(
                MountDiagnostic(
                    code="conflicting_child_agent_name",
                    detail=(
                        "sub-agent thread has multiple structured direct-child "
                        "recipient identities"
                    ),
                    leaf_index=child_index,
                    parent_thread_id=child.parent_thread_id,
                    parent_turn_id=child.parent_turn_id,
                    spawn_call_id=marker.spawn_call_id,
                )
            )
            invalid_children.add(child_index)
            continue
        # Conflicting canonical-name replays make recipient validation
        # unavailable, but they do not overturn an otherwise unique primary
        # spawn edge.  The existing conflict diagnostic still quarantines it.
        recipient_can_constrain = bool(recipient) and not (
            len(candidates) == 1 and candidates[0].agent_name_conflicted
        )
        if recipient_can_constrain:
            recipient_matches = _filter_spawns_by_recipient(candidates, recipient)
            if not recipient_matches:
                diagnostics.append(
                    MountDiagnostic(
                        code="spawn_routing_mismatch",
                        detail=(
                            "structured child recipient conflicts with the spawn "
                            "routing selected on the parent turn"
                        ),
                        leaf_index=child_index,
                        parent_thread_id=child.parent_thread_id,
                        parent_turn_id=child.parent_turn_id,
                        spawn_call_id=marker.spawn_call_id,
                    )
                )
                invalid_children.add(child_index)
                continue
            candidates = recipient_matches

        if len(candidates) > 1:
            diagnostics.append(
                MountDiagnostic(
                    code="ambiguous_spawn_call",
                    detail="parent turn contains multiple spawn calls and routing is not unique",
                    leaf_index=child_index,
                    parent_thread_id=child.parent_thread_id,
                    parent_turn_id=child.parent_turn_id,
                )
            )
            invalid_children.add(child_index)
            continue

        event = candidates[0]
        parent_index = parent_for_event.get(event.key)
        if parent_index is None:
            invalid_children.add(child_index)
            # The event-specific missing/ambiguous parent diagnostic was
            # already emitted above.
            continue
        proposed.append(
            MountEdge(
                parent_index=parent_index,
                child_index=child_index,
                spawn_call_id=event.spawn_call_id,
                parent_turn_id=event.parent_turn_id,
                task_name=event.task_name,
                agent_name=event.agent_name,
            )
        )

    # A spawn call can own one child trajectory.  Multiple child leaves with
    # different thread ids are still ambiguous and are never selected by time.
    by_spawn: dict[tuple[int, str], list[MountEdge]] = defaultdict(list)
    for edge in proposed:
        by_spawn[(edge.parent_index, edge.spawn_call_id)].append(edge)
    edges: list[MountEdge] = []
    for (parent_index, call_id), grouped in sorted(by_spawn.items()):
        if len(grouped) == 1:
            edges.append(grouped[0])
            continue
        related = tuple(sorted(edge.child_index for edge in grouped))
        invalid_children.update(related)
        for edge in grouped:
            diagnostics.append(
                MountDiagnostic(
                    code="multiple_children_for_spawn",
                    detail="one spawn call is claimed by multiple child threads",
                    leaf_index=edge.child_index,
                    related_leaf_indices=related,
                    parent_thread_id=canonical_leaves[parent_index].thread_id,
                    parent_turn_id=edge.parent_turn_id,
                    spawn_call_id=call_id,
                )
            )

    cycle_members = _cycle_nodes(edges)
    if cycle_members:
        for index in sorted(cycle_members):
            diagnostics.append(
                MountDiagnostic(
                    code="mount_cycle",
                    detail="sub-agent parentage forms a directed cycle",
                    leaf_index=index,
                    related_leaf_indices=tuple(sorted(cycle_members)),
                    parent_thread_id=canonical_leaves[index].parent_thread_id,
                    parent_turn_id=canonical_leaves[index].parent_turn_id,
                )
            )
        invalid_children.update(cycle_members)
        edges = [
            edge
            for edge in edges
            if edge.child_index not in cycle_members
            and edge.parent_index not in cycle_members
        ]

    edges, relay_diagnostics = _attach_relay_ids(
        edges,
        canonical_leaves,
        agent_records_by_thread,
    )
    diagnostics.extend(relay_diagnostics)

    main_roots = tuple(
        index for index in range(len(canonical_leaves)) if index not in subagent_indices
    )
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        children_by_parent[edge.parent_index].append(edge.child_index)
    reachable = set(main_roots)
    queue = deque(main_roots)
    while queue:
        parent = queue.popleft()
        for child_index in children_by_parent.get(parent, ()):
            if child_index not in reachable:
                reachable.add(child_index)
                queue.append(child_index)

    unreachable = subagent_indices - reachable
    for index in sorted(unreachable - invalid_children):
        diagnostics.append(
            MountDiagnostic(
                code="unreachable_subagent",
                detail="valid local linkage does not lead to a main-agent root",
                leaf_index=index,
                parent_thread_id=canonical_leaves[index].parent_thread_id,
                parent_turn_id=canonical_leaves[index].parent_turn_id,
            )
        )
    # Do not silently preserve edges into an orphan component as if mounted.
    edges = [
        edge
        for edge in edges
        if edge.parent_index in reachable and edge.child_index in reachable
    ]

    mounted_event_keys = {
        (
            canonical_leaves[edge.parent_index].source_partition,
            canonical_leaves[edge.parent_index].session_id,
            canonical_leaves[edge.parent_index].thread_id,
            edge.parent_turn_id,
            edge.spawn_call_id,
        )
        for edge in edges
    }
    incomplete_parents: set[int] = set(parent_linkage_incomplete)
    for event in events:
        parent_index = parent_for_event.get(event.key)
        if parent_index is None or event.key in mounted_event_keys:
            continue
        incomplete_parents.add(parent_index)
        diagnostics.append(
            MountDiagnostic(
                code="unmounted_spawn_call",
                detail="indexed spawn call has no uniquely mounted child trajectory",
                leaf_index=parent_index,
                parent_thread_id=event.parent_thread_id,
                parent_turn_id=event.parent_turn_id,
                spawn_call_id=event.spawn_call_id,
            )
        )

    edges_tuple = tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.parent_index,
                edge.spawn_call_id,
                edge.child_index,
            ),
        )
    )
    return SubagentMountPlan(
        leaves=canonical_leaves,
        edges=edges_tuple,
        main_root_indices=main_roots,
        orphan_indices=tuple(sorted(unreachable)),
        incomplete_parent_indices=tuple(sorted(incomplete_parents)),
        spawn_call_counts=tuple(spawn_call_counts),
        spawn_evidence=events,
        diagnostics=tuple(sorted(diagnostics, key=_diagnostic_sort_key)),
    )


def _snapshot_identity(snapshot: Snapshot) -> tuple[str, ...]:
    # source sha/path identify a capture in normal operation; the remaining
    # fields make test fixtures and imported manifests safe as well.
    return (
        snapshot.source_partition,
        snapshot.session_id,
        snapshot.thread_id,
        snapshot.turn_id,
        snapshot.request_id,
        snapshot.source_path,
        snapshot.source_sha256,
    )


def mount_subagents(
    candidates: Sequence[SnapshotTrajectory],
    *,
    all_snapshots: Sequence[Snapshot] | None = None,
) -> MountResult:
    """Materialize a mount plan into deep-copied ``TrajectoryNode`` trees."""

    candidate_by_identity: dict[tuple[str, ...], SnapshotTrajectory] = {}
    for candidate in candidates:
        identity = _snapshot_identity(candidate.snapshot)
        if identity in candidate_by_identity:
            raise ValueError(f"duplicate SnapshotTrajectory identity: {identity!r}")
        candidate_by_identity[identity] = candidate

    plan = plan_subagent_mounts(
        [candidate.snapshot for candidate in candidates],
        all_snapshots=all_snapshots,
    )
    nodes: dict[int, TrajectoryNode] = {}
    for index, snapshot in enumerate(plan.leaves):
        candidate = candidate_by_identity.get(_snapshot_identity(snapshot))
        if candidate is None:
            raise ValueError(
                f"missing trajectory node for leaf {snapshot.source_path!r}"
            )
        materialized = candidate.trajectory.model_copy(deep=True)
        materialized.agent_messages = [
            AgentMessageRecord(
                origin=record.origin,
                item_index=record.item_index,
                item=record.item,
            )
            for record in snapshot.agent_messages
        ]
        nodes[index] = materialized

    edges_by_parent: dict[int, list[MountEdge]] = defaultdict(list)
    for edge in plan.edges:
        edges_by_parent[edge.parent_index].append(edge)
    incomplete = set(plan.incomplete_parent_indices)
    materialization_diagnostics: list[MountDiagnostic] = []
    for index, materialized in nodes.items():
        preexisting_call_ids = set(materialized.sub_agent_trajectory or {}) | set(
            materialized.sub_agent_relay_mounts or {}
        )
        for call_id in sorted(preexisting_call_ids):
            incomplete.add(index)
            materialization_diagnostics.append(
                MountDiagnostic(
                    code="preexisting_mount_conflict",
                    detail=(
                        "input trajectory was already mounted; the unverified "
                        "entry is ignored and rebuilt only from snapshot evidence"
                    ),
                    leaf_index=index,
                    parent_thread_id=plan.leaves[index].thread_id,
                    spawn_call_id=call_id,
                )
            )
    if materialization_diagnostics:
        plan = replace(
            plan,
            incomplete_parent_indices=tuple(sorted(incomplete)),
            diagnostics=tuple(
                sorted(
                    (*plan.diagnostics, *materialization_diagnostics),
                    key=_diagnostic_sort_key,
                )
            ),
        )

    def attach_mount_diagnostics(node: TrajectoryNode, index: int) -> TrajectoryNode:
        relevant = [
            diagnostic
            for diagnostic in plan.diagnostics
            if (
                diagnostic.leaf_index == index
                or index in diagnostic.related_leaf_indices
                or (
                    diagnostic.leaf_index is None
                    and not diagnostic.related_leaf_indices
                    and diagnostic.parent_thread_id == plan.leaves[index].thread_id
                    and (
                        not diagnostic.parent_turn_id
                        or diagnostic.parent_turn_id == plan.leaves[index].turn_id
                        or any(
                            call.id == diagnostic.spawn_call_id
                            for message in (
                                *plan.leaves[index].history,
                                *plan.leaves[index].response,
                            )
                            for call in (message.tool_calls or ())
                        )
                    )
                )
            )
        ]
        if not relevant:
            return node
        mount_issues = [
            AuditIssue(
                code=diagnostic.code,
                stage="subagents",
                detail=diagnostic.detail,
            )
            for diagnostic in relevant
        ]
        previous = node.normalization_audit
        issues = [*(previous.issues if previous else []), *mount_issues]
        return node.model_copy(
            update={
                "normalization_audit": NormalizationAudit(
                    tag=AuditTag.QUARANTINED,
                    reason_codes=sorted(
                        {
                            issue.code
                            for issue in issues
                            if issue.severity.value == "error"
                        }
                    ),
                    issues=issues,
                )
            }
        )

    def build(index: int) -> TrajectoryNode:
        node = nodes[index].model_copy(deep=True)
        child_edges = edges_by_parent.get(index, ())
        mounted: dict[str, TrajectoryNode] = {}
        relay_mounts: dict[str, str] = {}
        for edge in child_edges:
            mounted[edge.spawn_call_id] = build(edge.child_index)
            if edge.relay_id:
                relay_mounts[edge.spawn_call_id] = edge.relay_id

        # Inputs to this stage are flat nodes.  Any pre-existing mounts were
        # diagnosed above and are intentionally not trusted or propagated.
        node.sub_agent_trajectory = mounted or None
        node.sub_agent_relay_mounts = relay_mounts or None
        return attach_mount_diagnostics(node, index)

    roots = tuple(
        enrich_trajectory(build(index), top_level=True, is_subagent=False)
        for index in plan.main_root_indices
    )
    orphan_nodes: list[TrajectoryNode] = []
    for index in plan.orphan_indices:
        node = nodes[index].model_copy(deep=True)
        node.sub_agent_trajectory = None
        node.sub_agent_relay_mounts = None
        node = attach_mount_diagnostics(node, index)
        orphan_nodes.append(enrich_trajectory(node, top_level=True, is_subagent=True))

    return MountResult(
        roots=roots,
        root_snapshots=tuple(plan.leaves[i] for i in plan.main_root_indices),
        orphans=tuple(orphan_nodes),
        orphan_snapshots=tuple(plan.leaves[i] for i in plan.orphan_indices),
        plan=plan,
    )


__all__ = [
    "MountDiagnostic",
    "MountEdge",
    "MountResult",
    "SnapshotTrajectory",
    "SpawnEvidence",
    "SubagentMountPlan",
    "mount_subagents",
    "plan_subagent_mounts",
]
