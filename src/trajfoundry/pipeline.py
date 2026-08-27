"""End-to-end, resumable trajectory normalization pipeline."""

from __future__ import annotations

import fcntl
import hashlib
import json
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .canonical import canonical_json, semantic_payload, trajectory_id
from .export import SCHEMA_VERSION, OutputSet
from .io import decode_capture, discover_captures, read_capture_bytes, source_partition
from .models import (
    AgentMessageRecord,
    AuditIssue,
    AuditTag,
    Message,
    Metadata,
    NormalizationAudit,
    QuarantineRecord,
    ServerToolCall,
    Severity,
    Snapshot,
    ToolDefinition,
    TrajectoryNode,
)
from .providers.anthropic import parse_anthropic_capture
from .providers.responses import parse_responses_capture
from .quality import enrich_trajectory
from .state import StateStore
from .streaming import streaming_prefix_leaves
from .subagents import SubagentMountPlan, plan_subagent_mounts

NORMALIZER_REVISION = "2026-08-27.6"
DEFAULT_INPUT = Path("/data/回流轨迹/data_feedback_des")
DEFAULT_OUTPUT = Path("/data/trajfoundry")

# These defects are fully representable in the canonical trajectory.  They
# remain error-level audit evidence (and therefore quarantine the trajectory),
# but must not demote the whole capture to a metadata-only record.
_MATERIALIZABLE_CAPTURE_ISSUES = {
    "ambiguous_server_tool_result",
    "duplicate_server_tool_call_id",
    "duplicate_server_tool_result",
    "duplicate_tool_call_id",
    "duplicate_tool_result",
    "invalid_server_tool_call",
    "invalid_server_tool_result",
    "invalid_agent_message_author",
    "invalid_agent_message_id",
    "invalid_agent_message_recipient",
    "invalid_tool_call",
    "invalid_tool_result",
    "missing_server_tool_id",
    "missing_tool_arguments",
    "missing_tool_call_id",
    "missing_tool_name",
    "missing_tool_result_id",
    "missing_tool_result_name",
    "orphan_server_tool_result",
    "orphan_tool_result",
    "tool_result_type_mismatch",
}


class UnsupportedCaptureError(ValueError):
    """Raised for JSON files outside the two supported provider endpoints."""


@contextmanager
def _exclusive_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"lock path must not be a symlink: {path}")
    handle: BinaryIO = path.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another normalization run holds {path}") from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def _normalization_locks(config: PipelineConfig) -> Iterable[None]:
    paths = {
        config.output_root / ".normalize.lock",
        config.resolved_state_path.with_suffix(".lock"),
    }
    with ExitStack() as stack:
        for path in sorted(paths, key=str):
            stack.enter_context(_exclusive_lock(path))
        yield


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    input_root: Path = DEFAULT_INPUT
    output_root: Path = DEFAULT_OUTPUT
    state_path: Path | None = None
    resume: bool = False
    max_shard_bytes: int = 512 * 1024 * 1024

    @property
    def resolved_state_path(self) -> Path:
        return self.state_path or self.output_root / ".state" / "trajfoundry.sqlite"


@dataclass(slots=True)
class PipelineStats:
    discovered: int = 0
    reused: int = 0
    parsed: int = 0
    parse_failures: int = 0
    eligible_snapshots: int = 0
    prefix_intermediates: int = 0
    leaf_snapshots: int = 0
    stored_trajectories: int = 0
    sessions: int = 0


@dataclass(slots=True)
class _FlatLeaf:
    """Small session-level descriptor for one distinct flat trajectory.

    Full snapshots and materialized trajectories are deliberately not retained
    here.  They are reloaded one root at a time after the sub-agent graph has
    been planned, which bounds peak memory by one thread plus one output tree.
    """

    routing_snapshot: Snapshot
    contributor_paths: set[str]
    issues: dict[bytes, AuditIssue]


def config_hash(config: PipelineConfig) -> str:
    """Hash normalization semantics, excluding machine-specific paths."""

    value = {
        "normalizer_revision": NORMALIZER_REVISION,
        "schema_version": SCHEMA_VERSION,
        "aggregation_scope": ["source_partition", "session_id", "thread_id"],
        "developer_messages": "preserve",
        "instructions": "trajectory_field",
        "strict_wire_completion": True,
        "token_estimator": "unicode-word-v1",
        "max_shard_bytes": config.max_shard_bytes,
    }
    return hashlib.sha256(canonical_json(value)).hexdigest()


def parse_capture(
    capture: dict[str, Any],
    *,
    source_path: str,
    source_sha256: str,
    partition: str,
) -> Snapshot:
    endpoint_value = capture.get("path")
    endpoint = (
        endpoint_value.split("?", 1)[0].rstrip("/")
        if isinstance(endpoint_value, str)
        else ""
    )
    arguments = {
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_partition": partition,
    }
    if endpoint in {"/v1/responses", "/responses"}:
        return parse_responses_capture(capture, **arguments)
    if endpoint in {
        "/v1/messages",
        "/messages",
        "/v1/messages/count_tokens",
        "/messages/count_tokens",
    }:
        return parse_anthropic_capture(capture, **arguments)
    raise UnsupportedCaptureError(f"unsupported capture endpoint: {endpoint!r}")


def _definition_score(definition: ToolDefinition) -> tuple[int, int, int, bytes]:
    encoded = canonical_json(definition.parameters)
    return (
        int(bool(definition.parameters)),
        len(encoded),
        len(definition.description),
        canonical_json(definition.model_dump(mode="json")),
    )


def _is_structural_subset(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return all(
            key in right and _is_structural_subset(value, right[key])
            for key, value in left.items()
        )
    if isinstance(left, list) and isinstance(right, list):
        return all(item in right for item in left)
    return left == right


def _finish_tools(
    candidates: dict[str, list[ToolDefinition]],
) -> tuple[list[ToolDefinition], list[AuditIssue]]:
    merged: list[ToolDefinition] = []
    issues: list[AuditIssue] = []
    for name in sorted(candidates):
        variants = candidates[name]
        parameter_variants = [definition.parameters for definition in variants]
        incompatible = any(
            not _is_structural_subset(left, right)
            and not _is_structural_subset(right, left)
            for index, left in enumerate(parameter_variants)
            for right in parameter_variants[index + 1 :]
        )
        if incompatible:
            issues.append(
                AuditIssue(
                    code="conflicting_tool_definition",
                    stage="aggregation",
                    path=f"/tools/{name}",
                    detail="same-named tool has incompatible parameter schemas",
                )
            )
        parameter_source = max(variants, key=_definition_score)
        description = max(
            (definition.description for definition in variants),
            key=lambda value: (len(value), value),
        )
        merged.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters=parameter_source.parameters,
            )
        )
    return merged, issues


def _merge_contributor_metadata(
    snapshots: Iterable[Snapshot],
) -> tuple[list[ToolDefinition], list[ServerToolCall], list[AuditIssue]]:
    tool_candidates: dict[str, list[ToolDefinition]] = defaultdict(list)
    server_calls: list[ServerToolCall] = []
    server_slots: dict[tuple[str, int], int] = {}
    server_variants: set[tuple[str, int, bytes]] = set()
    server_issues: list[AuditIssue] = []
    for snapshot in snapshots:
        for definition in snapshot.tools:
            if definition not in tool_candidates[definition.name]:
                tool_candidates[definition.name].append(definition)
        occurrences: dict[str, int] = defaultdict(int)
        for call in snapshot.server_tool_calls:
            ordinal = occurrences[call.id]
            occurrences[call.id] += 1
            slot = (call.id, ordinal)
            signature = canonical_json(call.model_dump(mode="json", exclude_none=False))
            previous_index = server_slots.get(slot)
            if previous_index is None:
                server_slots[slot] = len(server_calls)
                server_variants.add((call.id, ordinal, signature))
                server_calls.append(call.model_copy(deep=True))
                continue
            previous = server_calls[previous_index]
            names_compatible = (
                previous.name == call.name or not previous.name or not call.name
            )
            if previous.arguments != call.arguments:
                server_issues.append(
                    AuditIssue(
                        code="conflicting_server_tool_call",
                        stage="aggregation",
                        path=f"/server_tool_calls/{call.id}",
                        detail="replayed server tool call has conflicting arguments",
                    )
                )
            elif not names_compatible:
                server_issues.append(
                    AuditIssue(
                        code="conflicting_server_tool_call",
                        stage="aggregation",
                        path=f"/server_tool_calls/{call.id}",
                        detail="replayed server tool call has conflicting names",
                    )
                )
            elif (
                previous.result is not None
                and call.result is not None
                and previous.result != call.result
            ):
                server_issues.append(
                    AuditIssue(
                        code="duplicate_server_tool_result",
                        stage="aggregation",
                        path=f"/server_tool_calls/{call.id}",
                        detail="same server tool call has conflicting result blocks",
                    )
                )
            else:
                update: dict[str, Any] = {"origin": call.origin}
                if not previous.name and call.name:
                    update["name"] = call.name
                if previous.result is None and call.result is not None:
                    update["result"] = call.result
                server_calls[previous_index] = previous.model_copy(
                    update=update, deep=True
                )
                continue

            variant = (call.id, ordinal, signature)
            if variant not in server_variants:
                server_variants.add(variant)
                server_calls.append(call.model_copy(deep=True))
    tools, tool_issues = _finish_tools(tool_candidates)
    return tools, server_calls, [*tool_issues, *server_issues]


def _initial_audit(issues: Sequence[AuditIssue]) -> NormalizationAudit:
    reasons = sorted(
        {issue.code for issue in issues if issue.severity == Severity.ERROR}
    )
    return NormalizationAudit(
        tag=AuditTag.QUARANTINED if reasons else AuditTag.PASS,
        reason_codes=reasons,
        issues=list(issues),
    )


def _trajectory_from_leaf(
    leaf: Snapshot, contributors: Iterable[Snapshot]
) -> TrajectoryNode:
    tools, server_calls, contributor_issues = _merge_contributor_metadata(contributors)
    issues = [*leaf.issues, *contributor_issues]
    basename = Path(leaf.source_path).name
    return TrajectoryNode(
        messages=[*leaf.history, *leaf.response],
        tools=tools,
        agent_messages=[
            AgentMessageRecord(
                origin=record.origin,
                item_index=record.item_index,
                item=record.item,
            )
            for record in leaf.agent_messages
        ],
        instructions=leaf.instructions,
        termination=leaf.termination,
        harness=leaf.harness,
        model=leaf.model,
        source=basename,
        server_tool_calls=server_calls,
        metadata=Metadata(
            source_file=basename,
            line_no=0,
            created_at=leaf.captured_at,
        ),
        normalization_audit=_initial_audit(issues),
    )


def _snapshot_identity(snapshot: Snapshot) -> tuple[str, ...]:
    return (
        snapshot.source_partition,
        snapshot.session_id,
        snapshot.thread_id,
        snapshot.turn_id,
        snapshot.request_id,
        snapshot.source_path,
        snapshot.source_sha256,
    )


def _mount_issues_by_index(
    plan: SubagentMountPlan,
) -> dict[int, list[AuditIssue]]:
    """Map graph diagnostics back to every affected flat leaf."""

    issues_by_index: dict[int, list[AuditIssue]] = defaultdict(list)
    for diagnostic in plan.diagnostics:
        targets: set[int] = set(diagnostic.related_leaf_indices)
        if diagnostic.leaf_index is not None:
            targets.add(diagnostic.leaf_index)
        if not targets and diagnostic.parent_thread_id:
            targets.update(
                index
                for index, leaf in enumerate(plan.leaves)
                if leaf.thread_id == diagnostic.parent_thread_id
                and (
                    not diagnostic.parent_turn_id
                    or leaf.turn_id == diagnostic.parent_turn_id
                    or any(
                        call.id == diagnostic.spawn_call_id
                        for message in (*leaf.history, *leaf.response)
                        for call in (message.tool_calls or [])
                    )
                )
            )
        for index in targets:
            issues_by_index[index].append(
                AuditIssue(
                    code=diagnostic.code,
                    stage="subagents",
                    path="",
                    detail=diagnostic.detail,
                )
            )
    return issues_by_index


def _eligible(snapshot: Snapshot) -> bool:
    if snapshot.operation == "count_tokens" or not snapshot.wire_complete:
        return False
    if snapshot.outcome == "success":
        return True
    error_codes = {
        issue.code for issue in snapshot.issues if issue.severity == Severity.ERROR
    }
    return bool(error_codes) and error_codes <= _MATERIALIZABLE_CAPTURE_ISSUES


def _spawn_only_messages(messages: Iterable[Message]) -> list[Message]:
    """Retain structured spawn calls without retaining arbitrary content."""

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
    """Strip opaque bodies from session-wide mount-planning copies."""

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


def _minimal_evidence(snapshot: Snapshot) -> Snapshot:
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


def _routing_snapshot(snapshot: Snapshot) -> Snapshot:
    """Project a full leaf to the fields needed by the mount planner.

    Spawn calls are placed in ``history`` deliberately: the planner may use
    them to prove that a terminal leaf contains an earlier spawn event, but it
    must not infer that they originated in this leaf's own ``turn_id``.
    """

    return snapshot.model_copy(
        update={
            "history": _spawn_only_messages((*snapshot.history, *snapshot.response)),
            "response": [],
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


def _snapshot_order_key(snapshot: Snapshot) -> tuple[str, ...]:
    return (
        snapshot.captured_at,
        snapshot.turn_id,
        snapshot.request_id,
        snapshot.source_path,
        snapshot.source_sha256,
    )


def _flat_semantic_key(snapshot: Snapshot, node: TrajectoryNode) -> str:
    """Identify retransmitted leaves without collapsing real mount variants."""

    response_spawns = [
        {
            "turn_id": snapshot.turn_id,
            "call": call.model_dump(mode="json", exclude_none=False),
        }
        for message in snapshot.response
        for call in (message.tool_calls or ())
        if call.function.name in {"spawn_agent", "Agent"}
    ]
    payload = {
        "source_partition": snapshot.source_partition,
        "session_id": snapshot.session_id,
        "thread_id": snapshot.thread_id,
        "parent_thread_id": snapshot.parent_thread_id,
        "parent_turn_id": snapshot.parent_turn_id,
        "forked_from_thread_id": snapshot.forked_from_thread_id,
        "subagent_marker": snapshot.subagent_marker,
        "provider": snapshot.provider,
        "operation": snapshot.operation,
        "response_spawn_bindings": response_spawns,
        "trajectory": semantic_payload(node),
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _issue_key(issue: AuditIssue) -> bytes:
    return canonical_json(issue.model_dump(mode="json", exclude_none=False))


def _merged_issues(*groups: Iterable[AuditIssue]) -> list[AuditIssue]:
    unique: dict[bytes, AuditIssue] = {}
    for group in groups:
        for issue in group:
            unique[_issue_key(issue)] = issue
    return [unique[key] for key in sorted(unique)]


def _origin_rows(state: StateStore, paths: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(set(paths)):
        metadata = state.source_metadata(path)
        if metadata is None:
            continue
        sha256, captured_at = metadata
        rows.append(
            {
                "source_ref": path,
                "sha256": sha256,
                "captured_at": captured_at,
            }
        )
    return rows


def _snapshots_for_paths(state: StateStore, paths: Iterable[str]) -> Iterable[Snapshot]:
    for path in sorted(set(paths)):
        snapshot = state.get_snapshot(path)
        if snapshot is not None:
            yield snapshot


def _descendants(plan: SubagentMountPlan, root_index: int) -> set[int]:
    children: dict[int, list[int]] = defaultdict(list)
    for edge in plan.edges:
        children[edge.parent_index].append(edge.child_index)
    found: set[int] = set()
    pending = [root_index]
    while pending:
        index = pending.pop()
        if index in found:
            continue
        found.add(index)
        pending.extend(children.get(index, ()))
    return found


def _store_trajectory(
    state: StateStore,
    node: TrajectoryNode,
    representative: Snapshot,
    origins: Sequence[dict[str, str]],
) -> None:
    audit = node.normalization_audit or _initial_audit([])
    disposition = audit.tag.value
    identifier = trajectory_id(node)
    representative_key = (
        f"{representative.captured_at}\x1f{representative.source_path}"
        f"\x1f{representative.source_sha256}"
    )
    source_rows = list(origins) or [
        {
            "source_ref": representative.source_path,
            "sha256": representative.source_sha256,
            "captured_at": representative.captured_at,
        }
    ]
    state.put_trajectory_with_origins(
        identifier,
        node,
        representative_key=representative_key,
        origins=source_rows,
        disposition=disposition,
        reason_codes=audit.reason_codes,
    )


def _load_flat_node(
    state: StateStore,
    leaf: _FlatLeaf,
    graph_issues: Iterable[AuditIssue],
) -> tuple[TrajectoryNode, Snapshot]:
    """Rebuild one flat node only when its output component is materialized."""

    representative = state.get_snapshot(leaf.routing_snapshot.source_path)
    if representative is None:
        raise RuntimeError(
            f"missing representative snapshot: {leaf.routing_snapshot.source_path}"
        )
    node = _trajectory_from_leaf(
        representative,
        _snapshots_for_paths(state, leaf.contributor_paths),
    )
    existing = node.normalization_audit
    issues = _merged_issues(
        existing.issues if existing else (),
        leaf.issues.values(),
        graph_issues,
    )
    node.normalization_audit = _initial_audit(issues)
    return node, representative


def _materialize_tree(
    state: StateStore,
    leaves_by_index: dict[int, _FlatLeaf],
    graph_issues: dict[int, list[AuditIssue]],
    edges_by_parent: dict[int, list[tuple[str, int, str]]],
    index: int,
) -> tuple[TrajectoryNode, Snapshot]:
    """Materialize one connected root so unrelated roots stay off heap."""

    node, representative = _load_flat_node(
        state,
        leaves_by_index[index],
        graph_issues.get(index, ()),
    )
    children: dict[str, TrajectoryNode] = {}
    relay_mounts: dict[str, str] = {}
    for call_id, child_index, relay_id in edges_by_parent.get(index, ()):
        child, _ = _materialize_tree(
            state,
            leaves_by_index,
            graph_issues,
            edges_by_parent,
            child_index,
        )
        children[call_id] = child
        if relay_id:
            relay_mounts[call_id] = relay_id
    node.sub_agent_trajectory = children or None
    node.sub_agent_relay_mounts = relay_mounts or None
    return node, representative


def _build_trajectories(state: StateStore, stats: PipelineStats) -> None:
    state.clear_trajectories()
    for partition, session_id in state.sessions():
        stats.sessions += 1
        flat_leaves: dict[str, _FlatLeaf] = {}
        evidence_by_path: dict[str, Snapshot] = {}
        for thread_id in state.threads_for_session(partition, session_id):
            eligible = (
                snapshot
                for snapshot in state.snapshots_for_thread(
                    partition, session_id, thread_id
                )
                if _eligible(snapshot)
            )
            result = streaming_prefix_leaves(eligible)
            stats.prefix_intermediates += len(result.intermediate_paths)
            stats.leaf_snapshots += len(result.leaves)
            stats.eligible_snapshots += len(result.intermediate_paths) + len(
                result.leaves
            )
            for evidence in result.spawn_evidence:
                evidence_by_path[evidence.source_path] = evidence
            for leaf in result.leaves:
                evidence_by_path.setdefault(leaf.source_path, _minimal_evidence(leaf))
                contributor_paths = result.contributor_paths.get(
                    leaf.source_path, (leaf.source_path,)
                )
                candidate = _trajectory_from_leaf(
                    leaf, _snapshots_for_paths(state, contributor_paths)
                )
                semantic_key = _flat_semantic_key(leaf, candidate)
                candidate_issues = (
                    candidate.normalization_audit.issues
                    if candidate.normalization_audit
                    else ()
                )
                existing = flat_leaves.get(semantic_key)
                if existing is None:
                    flat_leaves[semantic_key] = _FlatLeaf(
                        routing_snapshot=_routing_snapshot(leaf),
                        contributor_paths=set(contributor_paths),
                        issues={_issue_key(issue): issue for issue in candidate_issues},
                    )
                    continue
                existing.contributor_paths.update(contributor_paths)
                for issue in candidate_issues:
                    existing.issues[_issue_key(issue)] = issue
                if _snapshot_order_key(leaf) < _snapshot_order_key(
                    existing.routing_snapshot
                ):
                    existing.routing_snapshot = _routing_snapshot(leaf)

        if not flat_leaves:
            continue
        routing_leaves = [
            flat_leaves[key].routing_snapshot for key in sorted(flat_leaves)
        ]
        evidence = tuple(evidence_by_path.values())
        plan = plan_subagent_mounts(routing_leaves, all_snapshots=evidence)
        leaves_by_identity = {
            _snapshot_identity(leaf.routing_snapshot): leaf
            for leaf in flat_leaves.values()
        }
        leaves_by_index = {
            index: leaves_by_identity[_snapshot_identity(snapshot)]
            for index, snapshot in enumerate(plan.leaves)
        }
        graph_issues = _mount_issues_by_index(plan)
        edges_by_parent: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
        for edge in plan.edges:
            edges_by_parent[edge.parent_index].append(
                (edge.spawn_call_id, edge.child_index, edge.relay_id)
            )

        for root_index in plan.main_root_indices:
            node, snapshot = _materialize_tree(
                state,
                leaves_by_index,
                graph_issues,
                edges_by_parent,
                root_index,
            )
            origin_paths = {
                path
                for index in _descendants(plan, root_index)
                for path in leaves_by_index[index].contributor_paths
            }
            enriched = enrich_trajectory(node, top_level=True, is_subagent=False)
            _store_trajectory(
                state,
                enriched,
                snapshot,
                _origin_rows(state, origin_paths),
            )
            del node, snapshot, enriched, origin_paths

        for orphan_index in plan.orphan_indices:
            node, snapshot = _load_flat_node(
                state,
                leaves_by_index[orphan_index],
                graph_issues.get(orphan_index, ()),
            )
            node.sub_agent_trajectory = None
            node.sub_agent_relay_mounts = None
            enriched = enrich_trajectory(node, top_level=True, is_subagent=True)
            _store_trajectory(
                state,
                enriched,
                snapshot,
                _origin_rows(state, leaves_by_index[orphan_index].contributor_paths),
            )
            del node, snapshot, enriched
    stats.stored_trajectories = state.trajectory_count()


def _snapshot_record(snapshot: Snapshot) -> QuarantineRecord | None:
    if _eligible(snapshot):
        return None
    if snapshot.operation == "count_tokens":
        tag = AuditTag.EXCLUDED
        synthetic = AuditIssue(
            code="non_trajectory_operation",
            stage="pipeline",
            detail="token counting calls are not trajectory turns",
        )
    else:
        tag = AuditTag.QUARANTINED
        synthetic = AuditIssue(
            code=f"snapshot_{snapshot.outcome}",
            stage="pipeline",
            detail="capture is not a complete successful model response",
        )
    issues = [*snapshot.issues, synthetic]
    reasons = sorted(
        {issue.code for issue in issues if issue.severity == Severity.ERROR}
    )
    return QuarantineRecord(
        source_ref=snapshot.source_path,
        sha256=snapshot.source_sha256,
        endpoint=snapshot.operation,
        captured_at=snapshot.captured_at,
        normalization_audit=NormalizationAudit(
            tag=tag,
            reason_codes=reasons,
            issues=issues,
        ),
    )


def _export(state: StateStore, config: PipelineConfig, configuration_hash: str) -> None:
    output = OutputSet(config.output_root, max_shard_bytes=config.max_shard_bytes)
    try:
        output.stats.input_files = state.capture_count()
        for (
            source_ref,
            sha256,
            status,
            reason,
            endpoint,
            captured_at,
        ) in state.capture_records():
            if status == "failed":
                output.write_record(
                    QuarantineRecord(
                        source_ref=source_ref,
                        sha256=sha256,
                        endpoint=endpoint,
                        captured_at=captured_at,
                        normalization_audit=NormalizationAudit(
                            tag=AuditTag.QUARANTINED,
                            reason_codes=["capture_parse_failed"],
                            issues=[
                                AuditIssue(
                                    code="capture_parse_failed",
                                    stage="ingest",
                                    detail=reason,
                                )
                            ],
                        ),
                    )
                )
                continue
            snapshot = state.get_snapshot(source_ref)
            if snapshot is not None and (record := _snapshot_record(snapshot)):
                output.write_record(record)

        for _, node, origins in state.iter_trajectories():
            output.write_trajectory(node, origins)
        output.close(input_root=str(config.input_root), config_hash=configuration_hash)
    except BaseException:
        output.abort()
        raise


def normalize(config: PipelineConfig) -> PipelineStats:
    """Normalize a capture tree and atomically publish manifest-backed shards."""

    input_root = config.input_root.resolve()
    output_root = config.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_root}")
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise ValueError("output directory must not be inside the input tree")
    if config.max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    if output_root.exists() and any(output_root.iterdir()) and not config.resume:
        raise FileExistsError(
            f"output directory is not empty; pass --resume: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    for reserved in (".state", "accepted", "quarantine", "generations"):
        if (output_root / reserved).is_symlink():
            raise ValueError(f"reserved output path must not be a symlink: {reserved}")

    normalized_config = PipelineConfig(
        input_root=input_root,
        output_root=output_root,
        state_path=(
            config.state_path.resolve() if config.state_path is not None else None
        ),
        resume=config.resume,
        max_shard_bytes=config.max_shard_bytes,
    )
    configuration_hash = config_hash(normalized_config)
    stats = PipelineStats()
    if normalized_config.resolved_state_path.is_symlink():
        raise ValueError("state database path must not be a symlink")
    with (
        _normalization_locks(normalized_config),
        StateStore(normalized_config.resolved_state_path) as state,
    ):
        previous_hash = state.get_meta("config_hash")
        if not normalized_config.resume or previous_hash != configuration_hash:
            state.reset_ingest()
        state.set_meta("config_hash", configuration_hash)
        state.begin_scan(uuid.uuid4().hex)
        for path in discover_captures(input_root):
            stats.discovered += 1
            source_ref = path.relative_to(input_root).as_posix()
            digest = ""
            endpoint = ""
            captured_at = ""
            try:
                payload, digest = read_capture_bytes(path)
                if state.capture_matches(source_ref, digest):
                    stats.reused += 1
                    continue
                capture = decode_capture(payload)
                endpoint_value = capture.get("path")
                endpoint = (
                    endpoint_value.split("?", 1)[0].rstrip("/")
                    if isinstance(endpoint_value, str)
                    else ""
                )
                captured_value = capture.get("captured_at")
                captured_at = captured_value if isinstance(captured_value, str) else ""
                snapshot = parse_capture(
                    capture,
                    source_path=source_ref,
                    source_sha256=digest,
                    partition=source_partition(input_root, path),
                )
                state.put_snapshot(snapshot, endpoint=endpoint)
                stats.parsed += 1
            except Exception as error:  # noqa: BLE001 - quarantine per-file defects
                state.put_failure(
                    source_ref,
                    digest,
                    f"{type(error).__name__}: capture could not be normalized",
                    endpoint=endpoint,
                    captured_at=captured_at,
                )
                stats.parse_failures += 1
        state.finish_scan()
        _build_trajectories(state, stats)
        _export(state, normalized_config, configuration_hash)
    return stats


def stats_json(stats: PipelineStats) -> str:
    return json.dumps(asdict(stats), ensure_ascii=False, sort_keys=True)


__all__ = [
    "DEFAULT_INPUT",
    "DEFAULT_OUTPUT",
    "NORMALIZER_REVISION",
    "PipelineConfig",
    "PipelineStats",
    "UnsupportedCaptureError",
    "config_hash",
    "normalize",
    "parse_capture",
    "stats_json",
]
