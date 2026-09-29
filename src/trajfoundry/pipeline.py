"""End-to-end, resumable trajectory normalization pipeline."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import multiprocessing
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

from .canonical import (
    canonical_compaction_records,
    canonical_json,
    semantic_payload,
    trajectory_id,
)
from .export import SCHEMA_VERSION, OutputSet
from .io import CaptureSource, LocalCaptureSource, decode_capture
from .models import (
    AgentMessageRecord,
    AuditIssue,
    AuditTag,
    CompactionRecord,
    MediaMapping,
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
from .providers.chat import parse_chat_capture
from .providers.responses import parse_responses_capture
from .quality import enrich_trajectory
from .sources.deepinfra import DeepInfraError, adapt_deepinfra_envelope
from .sources.sxf import SXFError, adapt_sxf_envelope
from .sources.tokenplan import TokenPlanError, adapt_tokenplan_envelope
from .state import StateStore, _decompress_payload
from .streaming import streaming_prefix_leaves
from .subagents import (
    SubagentMountPlan,
    _accept_local_relay_mount,
    diagnostic_applies_to_snapshot,
    plan_subagent_mounts,
)
from .tool_names import is_spawn_tool_name

LOGGER = logging.getLogger(__name__)

NORMALIZER_REVISION = "2026-09-28.1"
DEFAULT_INPUT = Path("/data/回流轨迹/data_feedback_des")
DEFAULT_OUTPUT = Path("/data/trajfoundry")
_INGEST_BATCH_ITEMS = 512
_INGEST_BATCH_BYTES = 64 * 1024 * 1024
_TRAJECTORY_BUILD_SKIP_REASON = "trajectory_materialization_failed"
_BUILD_WORKERS_ENV = "TRAJFOUNDRY_BUILD_WORKERS"
_PROGRESS_INTERVAL_SECONDS = 60.0
_PROGRESS_ITEM_INTERVAL = 10_000

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
    input_format: Literal["freerouter", "tokenplan", "sxf", "deepinfra"] = "freerouter"
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
    skipped_inputs: int = 0
    eligible_snapshots: int = 0
    prefix_intermediates: int = 0
    leaf_snapshots: int = 0
    stored_trajectories: int = 0
    sessions: int = 0
    input_bytes: int = 0
    ingest_seconds: float = 0.0
    build_seconds: float = 0.0
    export_seconds: float = 0.0
    validation_seconds: float = 0.0


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


@dataclass(frozen=True, slots=True)
class _MaterializationLeaf:
    """Small, picklable descriptor passed to a build worker."""

    source_path: str
    contributor_paths: tuple[str, ...]
    issues: tuple[AuditIssue, ...]


@dataclass(frozen=True, slots=True)
class _MaterializationJob:
    """One independent root materialization unit.

    Prefix aggregation and sub-agent planning happen in the coordinator.  A
    worker only receives the resulting leaf graph and reads immutable snapshot
    payloads from SQLite.  This preserves the exact prefix semantics while
    parallelizing the expensive trajectory construction step.
    """

    state_path: str
    immutable_state: bool
    root_index: int
    is_subagent: bool
    leaves: dict[int, _MaterializationLeaf]
    graph_issues: dict[int, tuple[AuditIssue, ...]]
    edges_by_parent: dict[int, tuple[tuple[str, int, str], ...]]
    origin_paths: tuple[str, ...]


def config_hash(config: PipelineConfig) -> str:
    """Hash normalization semantics, excluding machine-specific paths."""

    value = {
        "normalizer_revision": NORMALIZER_REVISION,
        "schema_version": SCHEMA_VERSION,
        "input_format": config.input_format,
        "media_policy": (
            "capture_used_object_name_mapping"
            if config.input_format == "tokenplan"
            else "not_applicable"
        ),
        "aggregation_scope": {
            "with_session": ["session_id", "thread_id"],
            "without_session": ["user_id_or_no_user_id"],
        },
        "prefix_message_fields": {
            "system_developer_user": ["role", "content"],
            "assistant": ["role", "content", "tool_calls"],
            "tool": ["role", "tool_call_id", "name", "content"],
        },
        "responses_compaction": "lossless_opaque_sidecar_exact_prefix_quarantine",
        "subagent_identity_fields": [
            "subagent_marker",
            "x-openai-subagent",
            "subagent_kind",
            "parent_thread_id",
            "parent_turn_id",
            "forked_from_thread_id",
        ],
        "leaf_messages": "request_history_plus_response_without_contributor_backfill",
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
    source_name: Literal["freerouter", "tokenplan", "sxf", "deepinfra"] = "freerouter",
    response_is_normalized_final: bool = False,
    multimodal_file_mapping: Sequence[MediaMapping] | None = None,
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
    }
    if endpoint in {"/v1/responses", "/responses"}:
        snapshot = parse_responses_capture(capture, **arguments)
    elif endpoint in {"/v1/chat/completions", "/chat/completions"}:
        snapshot = parse_chat_capture(
            capture,
            response_is_normalized_final=response_is_normalized_final,
            **arguments,
        )
    elif endpoint in {
        "/v1/messages",
        "/messages",
        "/v1/messages/count_tokens",
        "/messages/count_tokens",
    }:
        snapshot = parse_anthropic_capture(
            capture,
            response_is_normalized_final=response_is_normalized_final,
            **arguments,
        )
    else:
        raise UnsupportedCaptureError(f"unsupported capture endpoint: {endpoint!r}")
    updates: dict[str, Any] = {}
    masked_fields = capture.get("_masked_identity_fields")
    if isinstance(masked_fields, (list, tuple, set, frozenset)):
        issues = list(snapshot.issues)
        for field in sorted(
            item for item in masked_fields if isinstance(item, str) and item
        ):
            issues.append(
                AuditIssue(
                    code=f"metadata_{field}_masked",
                    stage="parsing",
                    severity=Severity.WARNING,
                    path=f"/metadata/{field}",
                    detail="redacted identity value was ignored",
                )
            )
        if issues != snapshot.issues:
            updates["issues"] = issues
    if snapshot.source_name != source_name:
        updates["source_name"] = source_name
    if multimodal_file_mapping is not None:
        updates["multimodal_file_mapping"] = [
            item.model_copy(deep=True) for item in multimodal_file_mapping
        ]
    return snapshot.model_copy(update=updates) if updates else snapshot


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
    *,
    server_origin_source_path: str | None = None,
) -> tuple[
    list[ToolDefinition],
    list[ServerToolCall],
    list[AgentMessageRecord],
    list[CompactionRecord],
    list[AuditIssue],
]:
    tool_candidates: dict[str, list[ToolDefinition]] = defaultdict(list)
    server_calls: list[ServerToolCall] = []
    server_slots: dict[tuple[str, int], int] = {}
    authoritative_origin_slots: set[tuple[str, int]] = set()
    server_variants: set[tuple[str, int, bytes]] = set()
    server_issues: list[AuditIssue] = []
    agent_messages: dict[bytes, AgentMessageRecord] = {}
    compaction_items: list[CompactionRecord] = []
    contributor_issues: dict[bytes, AuditIssue] = {}
    for snapshot in snapshots:
        for definition in snapshot.tools:
            if definition not in tool_candidates[definition.name]:
                tool_candidates[definition.name].append(definition)
        for record in snapshot.agent_messages:
            projected = AgentMessageRecord(
                origin=record.origin,
                item_index=record.item_index,
                item=record.item,
            )
            key = canonical_json(projected.model_dump(mode="json", exclude_none=False))
            agent_messages.setdefault(key, projected)
        compaction_items.extend(snapshot.compaction_items)
        for issue in snapshot.issues:
            contributor_issues.setdefault(_issue_key(issue), issue)
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
                if snapshot.source_path == server_origin_source_path:
                    authoritative_origin_slots.add(slot)
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
                update: dict[str, Any] = {}
                if (
                    snapshot.source_path == server_origin_source_path
                    or slot not in authoritative_origin_slots
                ):
                    update["origin"] = call.origin
                if not previous.name and call.name:
                    update["name"] = call.name
                if previous.result is None and call.result is not None:
                    update["result"] = call.result
                server_calls[previous_index] = previous.model_copy(
                    update=update, deep=True
                )
                if snapshot.source_path == server_origin_source_path:
                    authoritative_origin_slots.add(slot)
                continue

            variant = (call.id, ordinal, signature)
            if variant not in server_variants:
                server_variants.add(variant)
                server_calls.append(call.model_copy(deep=True))
    tools, tool_issues = _finish_tools(tool_candidates)
    return (
        tools,
        server_calls,
        [agent_messages[key] for key in sorted(agent_messages)],
        [
            record.model_copy(deep=True)
            for record in canonical_compaction_records(compaction_items)
        ],
        [
            *[contributor_issues[key] for key in sorted(contributor_issues)],
            *tool_issues,
            *server_issues,
        ],
    )


def _contributor_leaf_warnings(
    leaf: Snapshot, contributors: Iterable[Snapshot]
) -> list[AuditIssue]:
    warnings: list[AuditIssue] = []
    for field in ("instructions", "model", "harness", "termination"):
        leaf_value = getattr(leaf, field)
        variants = {getattr(snapshot, field) for snapshot in contributors}
        if variants <= {leaf_value}:
            continue
        warnings.append(
            AuditIssue(
                code=f"contributor_{field}_changed",
                stage="aggregation",
                severity=Severity.WARNING,
                path=f"/{field}",
                detail=(
                    f"prefix contributors contain {field} values different "
                    "from the leaf; the leaf value was retained"
                ),
            )
        )
    return warnings


def _merged_multimodal_file_mapping(
    leaf: Snapshot, contributors: Iterable[Snapshot]
) -> list[MediaMapping]:
    """Union media mappings from a leaf and its cumulative prefix captures.

    Contributors are ordered by the same stable capture key used elsewhere in
    the pipeline.  A part is emitted once, while the selected leaf remains
    authoritative if a replay supplied a different filename for that part.
    """

    snapshots = list(contributors)
    snapshots.sort(key=_snapshot_order_key)

    by_part: dict[str, MediaMapping] = {}
    part_order: list[str] = []

    # The leaf contains the complete request history in normal cumulative
    # captures, so its order is the most faithful representation of first use
    # in the published transcript.  It is also authoritative when a replay
    # reused a part id with a different storage name.
    for entry in leaf.multimodal_file_mapping:
        if entry.part_id not in by_part:
            part_order.append(entry.part_id)
        by_part[entry.part_id] = entry.model_copy(deep=True)

    # Prefix captures can contain a media reference omitted by a later replay
    # (for example after provider-side compaction).  Retain those mappings, but
    # append them deterministically after the leaf's first-use order.
    for snapshot in snapshots:
        for entry in snapshot.multimodal_file_mapping:
            if entry.part_id not in by_part:
                part_order.append(entry.part_id)
                by_part[entry.part_id] = entry.model_copy(deep=True)
    return [by_part[part_id] for part_id in part_order]


def _initial_audit(issues: Sequence[AuditIssue]) -> NormalizationAudit:
    reasons = sorted(
        {issue.code for issue in issues if issue.severity == Severity.ERROR}
    )
    return NormalizationAudit(
        tag=AuditTag.QUARANTINED if reasons else AuditTag.PASS,
        reason_codes=reasons,
        issues=list(issues),
    )


def _source_file_and_line(source_path: str) -> tuple[str, int]:
    """Extract SXF's line marker while preserving legacy file sources."""

    base, marker, line_text = source_path.rpartition("#L")
    if base and marker and line_text.isdigit():
        return Path(base).name, int(line_text)
    return Path(source_path).name, 0


def _trajectory_from_leaf(
    leaf: Snapshot, contributors: Iterable[Snapshot]
) -> TrajectoryNode:
    contributor_list = list(contributors)
    if not any(item.source_path == leaf.source_path for item in contributor_list):
        contributor_list.append(leaf)
    tools, server_calls, agent_messages, compaction_items, contributor_issues = (
        _merge_contributor_metadata(
            contributor_list,
            server_origin_source_path=leaf.source_path,
        )
    )
    issues = _merged_issues(
        leaf.issues,
        contributor_issues,
        _contributor_leaf_warnings(leaf, contributor_list),
    )
    synthesized_identity_issues: list[AuditIssue] = []
    if not leaf.user_id:
        synthesized_identity_issues.append(
            AuditIssue(
                code="metadata_user_id_synthesized",
                stage="aggregation",
                severity=Severity.WARNING,
                path="/metadata/user_id",
                detail="missing user id was published as no_user_id",
            )
        )
    if not leaf.session_id:
        synthesized_identity_issues.append(
            AuditIssue(
                code="metadata_session_id_synthesized",
                stage="aggregation",
                severity=Severity.WARNING,
                path="/metadata/session_id",
                detail="missing session id was published as no_session_id",
            )
        )
    issues = _merged_issues(issues, synthesized_identity_issues)
    multimodal_file_mapping = _merged_multimodal_file_mapping(leaf, contributor_list)
    basename, line_no = _source_file_and_line(leaf.source_path)
    source_type, specific_source = {
        "freerouter": ("api-router", "free-router"),
        "tokenplan": ("api-router", "token-plan"),
        "sxf": ("traj-cooperate", "SXF"),
        "deepinfra": ("api-router", "deep-infra"),
    }[leaf.source_name]
    contributor_user_ids = {
        snapshot.user_id for snapshot in contributor_list if snapshot.user_id
    }
    if contributor_user_ids and any(
        snapshot.user_id != leaf.user_id
        for snapshot in contributor_list
        if snapshot.user_id
    ):
        issues = _merged_issues(
            issues,
            [
                AuditIssue(
                    code="contributor_user_id_changed",
                    stage="aggregation",
                    severity=Severity.WARNING,
                    path="/user_id",
                    detail=(
                        "prefix contributors contain user_id values different "
                        "from the leaf; the leaf value was retained"
                    ),
                )
            ],
        )
    return TrajectoryNode(
        messages=[*leaf.history, *leaf.response],
        multimodal_file_mapping=multimodal_file_mapping,
        tools=tools,
        agent_messages=agent_messages,
        compaction_items=compaction_items,
        instructions=leaf.instructions,
        termination=leaf.termination,
        harness=leaf.harness,
        model=leaf.model,
        source=basename,
        server_tool_calls=server_calls,
        metadata=Metadata(
            source_file=basename,
            source_name=leaf.source_name,
            line_no=line_no,
            created_at=leaf.captured_at,
            model=leaf.model,
            user_id=leaf.user_id or "no_user_id",
            session_id=leaf.session_id or "no_session_id",
            source_type=source_type,
            specific_source=specific_source,
        ),
        normalization_audit=_initial_audit(issues),
    )


def _snapshot_identity(snapshot: Snapshot) -> tuple[str, ...]:
    return (
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
                if diagnostic_applies_to_snapshot(diagnostic, leaf)
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
    spawn_names: dict[str, set[str]] = defaultdict(set)
    for name, call_id in (
        (call.function.name, call.id)
        for message in source
        if message.role == "assistant"
        for call in (message.tool_calls or ())
        if is_spawn_tool_name(call.function.name)
    ):
        spawn_names[call_id].add(name)
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
        if is_spawn_tool_name(call.function.name)
    ]
    payload = {
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
    if any(
        issue.code == "metadata_session_id_masked" for issue in snapshot.issues
    ):
        payload["masked_identity_source"] = snapshot.source_path
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
    return [
        {
            "source_ref": path,
            "sha256": sha256,
            "captured_at": captured_at,
        }
        for path, (sha256, captured_at) in state.source_metadata_for_paths(
            paths
        ).items()
    ]


def _snapshots_for_paths(state: StateStore, paths: Iterable[str]) -> Iterable[Snapshot]:
    yield from state.iter_snapshots_for_paths(paths)


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
        if relay_id and _accept_local_relay_mount(
            node,
            call_id=call_id,
            relay_id=relay_id,
        ):
            relay_mounts[call_id] = relay_id
    node.sub_agent_trajectory = children or None
    node.sub_agent_relay_mounts = relay_mounts or None
    return node, representative


def _read_snapshot_for_worker(
    connection: sqlite3.Connection,
    cache: dict[str, Snapshot],
    source_path: str,
) -> Snapshot:
    cached = cache.get(source_path)
    if cached is not None:
        return cached
    row = connection.execute(
        "SELECT payload FROM snapshots WHERE source_path=?",
        (source_path,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"missing representative snapshot: {source_path}")
    snapshot = Snapshot.model_validate_json(_decompress_payload(row[0]))
    cache[source_path] = snapshot
    return snapshot


def _materialize_root_worker(
    job: _MaterializationJob,
) -> tuple[int, TrajectoryNode, Snapshot, tuple[str, ...]]:
    """Materialize and enrich one root in an isolated process.

    The SQLite state is immutable during this phase.  Each worker opens its own
    read-only connection; no SQLite connection or Pydantic object is shared
    between processes.
    """

    # Ingest has completed before jobs are submitted.  Normal state databases
    # use WAL, so regular read-only mode is required to see committed WAL
    # pages.  Ephemeral scheduler state uses an in-memory journal and an
    # exclusive lock; immutable mode bypasses that retained lock safely after
    # ingest has finished.
    state_uri = Path(job.state_path).resolve().as_uri()
    connection = sqlite3.connect(
        state_uri + ("?mode=ro&immutable=1" if job.immutable_state else "?mode=ro"),
        uri=True,
    )
    cache: dict[str, Snapshot] = {}

    def load(source_path: str) -> Snapshot:
        return _read_snapshot_for_worker(connection, cache, source_path)

    leaves = job.leaves

    def load_flat(index: int) -> tuple[TrajectoryNode, Snapshot]:
        descriptor = leaves[index]
        representative = load(descriptor.source_path)
        node = _trajectory_from_leaf(
            representative,
            (load(path) for path in descriptor.contributor_paths),
        )
        existing = node.normalization_audit
        issues = _merged_issues(
            existing.issues if existing else (),
            descriptor.issues,
            job.graph_issues.get(index, ()),
        )
        node.normalization_audit = _initial_audit(issues)
        return node, representative

    def materialize(index: int) -> tuple[TrajectoryNode, Snapshot]:
        node, representative = load_flat(index)
        children: dict[str, TrajectoryNode] = {}
        relay_mounts: dict[str, str] = {}
        for call_id, child_index, relay_id in job.edges_by_parent.get(index, ()):
            child, _ = materialize(child_index)
            children[call_id] = child
            if relay_id and _accept_local_relay_mount(
                node,
                call_id=call_id,
                relay_id=relay_id,
            ):
                relay_mounts[call_id] = relay_id
        node.sub_agent_trajectory = children or None
        node.sub_agent_relay_mounts = relay_mounts or None
        return node, representative

    try:
        node, representative = materialize(job.root_index)
        enriched = enrich_trajectory(
            node,
            top_level=True,
            is_subagent=job.is_subagent,
        )
        return job.root_index, enriched, representative, job.origin_paths
    finally:
        connection.close()


def _build_worker_count(configured: int | None = None) -> int:
    """Return the configured trajectory materialization process count."""

    raw = (
        str(configured)
        if configured is not None
        else os.environ.get(_BUILD_WORKERS_ENV, "1").strip()
    )
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError(f"{_BUILD_WORKERS_ENV} must be a positive integer") from error
    if workers <= 0:
        raise ValueError(f"{_BUILD_WORKERS_ENV} must be a positive integer")
    return workers


def _state_database_path(state: StateStore) -> Path:
    """Resolve the on-disk SQLite path used by materialization workers."""

    row = state.connection.execute("PRAGMA database_list").fetchone()
    if not row or not row[2]:
        raise RuntimeError("parallel trajectory materialization requires file-backed state")
    return Path(str(row[2])).resolve()


def _state_database_is_immutable(state: StateStore) -> bool:
    """Whether workers must bypass the coordinator's ephemeral DB lock."""

    row = state.connection.execute("PRAGMA journal_mode").fetchone()
    return bool(row and str(row[0]).lower() == "memory")


def _log_trajectory_build_failure(
    error: Exception,
    source_paths: Iterable[str],
) -> None:
    paths = sorted(set(source_paths))
    LOGGER.warning(
        "skipping trajectory after %s: source_count=%d source_refs=%s",
        type(error).__name__,
        len(paths),
        paths[:3],
    )


def _skip_failed_trajectory_sources(
    state: StateStore,
    failed_paths: Iterable[str],
    represented_paths: Iterable[str],
) -> int:
    """Mark failed leaves skipped after all successful origins are known."""

    paths = sorted(set(failed_paths) - set(represented_paths))
    skipped = 0
    with state.write_batch():
        for source_path in paths:
            metadata = state.source_metadata(source_path)
            if metadata is None:
                continue
            sha256, captured_at = metadata
            state.put_skipped(
                source_path,
                sha256,
                _TRAJECTORY_BUILD_SKIP_REASON,
                captured_at=captured_at,
            )
            skipped += 1
    return skipped


def _build_trajectories(
    state: StateStore,
    stats: PipelineStats,
    *,
    build_workers: int | None = None,
) -> None:
    started_at = time.monotonic()
    last_progress_at = started_at
    completed_scopes = 0
    completed_roots = 0
    failed_roots = 0
    total_scopes = sum(1 for _ in state.aggregation_scopes())

    def log_progress(*, force: bool = False) -> None:
        nonlocal last_progress_at
        now = time.monotonic()
        finished_roots = completed_roots + failed_roots
        if not force and (
            finished_roots % _PROGRESS_ITEM_INTERVAL != 0
            and completed_scopes % _PROGRESS_ITEM_INTERVAL != 0
            and now - last_progress_at < _PROGRESS_INTERVAL_SECONDS
        ):
            return
        elapsed = max(now - started_at, 0.001)
        LOGGER.info(
            "【构建阶段】已完成范围=%d/%d，已处理根轨迹=%d，已生成轨迹=%d，"
            "失败跳过=%d，耗时=%.1f秒，范围速度=%.2f个/秒",
            completed_scopes,
            total_scopes,
            completed_roots,
            state.trajectory_count(),
            failed_roots,
            elapsed,
            completed_scopes / elapsed,
        )
        last_progress_at = now

    LOGGER.info(
        "【构建阶段】开始构建轨迹，聚合范围总数=%d，并发进程数=%s",
        total_scopes,
        build_workers if build_workers is not None else os.environ.get(_BUILD_WORKERS_ENV, "1"),
    )
    state.clear_trajectories()
    failed_paths: set[str] = set()
    represented_paths: set[str] = set()
    # A real session is one mount-planning universe; the prefix index itself
    # keeps its threads separate.  Captures without a session are partitioned
    # by user (or the explicit no-user bucket) so they can merge across noisy
    # thread/request labels without ever crossing user boundaries.
    for scope in state.aggregation_scopes():
        stats.sessions += 1
        flat_leaves: dict[str, _FlatLeaf] = {}
        evidence_by_path: dict[str, Snapshot] = {}
        eligible = (
            snapshot
            for snapshot in state.snapshots_for_aggregation_scope(scope)
            if _eligible(snapshot)
        )
        result = streaming_prefix_leaves(eligible)
        stats.prefix_intermediates += len(result.intermediate_paths)
        stats.leaf_snapshots += len(result.leaves)
        stats.eligible_snapshots += len(result.intermediate_paths) + len(result.leaves)
        for evidence in result.spawn_evidence:
            evidence_by_path[evidence.source_path] = evidence
        for leaf in result.leaves:
            evidence_by_path.setdefault(leaf.source_path, _minimal_evidence(leaf))
            contributor_paths = result.contributor_paths.get(
                leaf.source_path, (leaf.source_path,)
            )
            try:
                candidate = _trajectory_from_leaf(
                    leaf, _snapshots_for_paths(state, contributor_paths)
                )
                semantic_key = _flat_semantic_key(leaf, candidate)
            except Exception as error:  # noqa: BLE001 - skip one bad trajectory
                failed_paths.update(contributor_paths)
                _log_trajectory_build_failure(error, contributor_paths)
                continue
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
            completed_scopes += 1
            log_progress()
            continue
        routing_leaves = [
            flat_leaves[key].routing_snapshot for key in sorted(flat_leaves)
        ]
        evidence = tuple(evidence_by_path.values())
        all_flat_paths = {
            path
            for flat_leaf in flat_leaves.values()
            for path in flat_leaf.contributor_paths
        }
        try:
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
        except Exception as error:  # noqa: BLE001 - skip one broken scope
            failed_paths.update(all_flat_paths)
            _log_trajectory_build_failure(error, all_flat_paths)
            completed_scopes += 1
            log_progress()
            continue

        # Build one independent job per root.  The coordinator owns prefix
        # aggregation and mount planning; workers only read the immutable
        # SQLite snapshot and return enriched trees.  Jobs are assembled with
        # root-local graph slices so large scopes are not pickled repeatedly.
        worker_count = _build_worker_count(build_workers)
        root_specs: list[tuple[int, tuple[str, ...], bool]] = []
        for root_index, is_subagent in [
            *((index, False) for index in plan.main_root_indices),
            *((index, True) for index in plan.orphan_indices),
        ]:
            descendants = _descendants(plan, root_index)
            origin_paths = tuple(
                sorted(
                    {
                        path
                        for index in descendants
                        for path in leaves_by_index[index].contributor_paths
                    }
                )
            )
            root_specs.append((root_index, origin_paths, is_subagent))

        # Keep a single root in-process even when a larger worker count is
        # configured; process startup and SQLite handoff would otherwise cost
        # more than the materialization itself.
        use_pool = worker_count > 1 and len(root_specs) > 1
        root_jobs: list[tuple[_MaterializationJob, tuple[str, ...], bool]] = []
        if use_pool:
            state_path = _state_database_path(state)
            immutable_state = _state_database_is_immutable(state)
            for root_index, origin_paths, is_subagent in root_specs:
                descendants = _descendants(plan, root_index)
                root_jobs.append(
                    (
                        _MaterializationJob(
                            state_path=str(state_path),
                            immutable_state=immutable_state,
                            root_index=root_index,
                            is_subagent=is_subagent,
                            leaves={
                                index: _MaterializationLeaf(
                                    source_path=leaves_by_index[index]
                                    .routing_snapshot.source_path,
                                    contributor_paths=tuple(
                                        sorted(leaves_by_index[index].contributor_paths)
                                    ),
                                    issues=tuple(leaves_by_index[index].issues.values()),
                                )
                                for index in descendants
                            },
                            graph_issues={
                                index: tuple(graph_issues.get(index, ()))
                                for index in descendants
                            },
                            edges_by_parent={
                                index: tuple(edges_by_parent.get(index, ()))
                                for index in descendants
                            },
                            origin_paths=origin_paths,
                        ),
                        origin_paths,
                        is_subagent,
                    )
                )

        with state.write_batch():
            if not use_pool:
                for root_index, origin_paths, is_subagent in root_specs:
                    try:
                        node, snapshot = (
                            _load_flat_node(
                                state,
                                leaves_by_index[root_index],
                                graph_issues.get(root_index, ()),
                            )
                            if is_subagent
                            else _materialize_tree(
                                state,
                                leaves_by_index,
                                graph_issues,
                                edges_by_parent,
                                root_index,
                            )
                        )
                        if is_subagent:
                            node.sub_agent_trajectory = None
                            node.sub_agent_relay_mounts = None
                        enriched = enrich_trajectory(
                            node,
                            top_level=True,
                            is_subagent=is_subagent,
                        )
                        try:
                            _store_trajectory(
                                state,
                                enriched,
                                snapshot,
                                _origin_rows(state, origin_paths),
                            )
                        except Exception as error:  # noqa: BLE001
                            failed_roots += 1
                            failed_paths.update(origin_paths)
                            _log_trajectory_build_failure(error, origin_paths)
                        else:
                            completed_roots += 1
                            represented_paths.update(origin_paths)
                    except Exception as error:  # noqa: BLE001 - skip one bad trajectory
                        failed_roots += 1
                        failed_paths.update(origin_paths)
                        _log_trajectory_build_failure(error, origin_paths)
                    log_progress()
            elif root_jobs:
                # Submit in deterministic root order, but consume and persist in
                # that same order so trajectory IDs and representative selection
                # remain independent of process completion timing.
                # The scheduler executes generated Python nodes at module
                # scope, so ``spawn`` would recursively re-enter the node
                # instead of starting a worker.  Fork is safe here because
                # workers only use their own read-only SQLite connection and
                # the coordinator no longer holds an exclusive SQLite lock.
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("fork"),
                ) as pool:
                    iterator = iter(root_jobs)
                    futures: list[tuple[tuple[_MaterializationJob, tuple[str, ...], bool], Any]] = []
                    for _ in range(min(worker_count * 2, len(root_jobs))):
                        item = next(iterator, None)
                        if item is None:
                            break
                        futures.append((item, pool.submit(_materialize_root_worker, item[0])))
                    while futures:
                        (_, origin_paths, _), future = futures.pop(0)
                        paths = origin_paths
                        try:
                            _, enriched, snapshot, returned_paths = future.result()
                            paths = tuple(returned_paths) or origin_paths
                            _store_trajectory(
                                state,
                                enriched,
                                snapshot,
                                _origin_rows(state, paths),
                            )
                            completed_roots += 1
                            represented_paths.update(paths)
                        except Exception as error:  # noqa: BLE001 - skip one bad trajectory
                            failed_roots += 1
                            failed_paths.update(paths)
                            _log_trajectory_build_failure(error, paths)
                        log_progress()
                        item = next(iterator, None)
                        if item is not None:
                            futures.append((item, pool.submit(_materialize_root_worker, item[0])))
        completed_scopes += 1
        log_progress()
    _skip_failed_trajectory_sources(state, failed_paths, represented_paths)
    state.assign_sub_session_ids()
    # Sub-session assignment validates the materialized rows a second time.
    # A malformed cached row can therefore be skipped after the initial
    # materialization pass; count those inputs before exporting the manifest.
    stats.skipped_inputs = sum(
        status == "skipped" for _, _, status, _, _, _ in state.capture_records()
    )
    stats.stored_trajectories = state.trajectory_count()
    log_progress(force=True)
    LOGGER.info(
        "【构建阶段】完成：范围=%d，生成轨迹=%d，失败跳过=%d，耗时=%.1f秒",
        stats.sessions,
        stats.stored_trajectories,
        failed_roots,
        time.monotonic() - started_at,
    )


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


def _export(
    state: StateStore,
    config: PipelineConfig,
    configuration_hash: str,
    *,
    output: Any | None = None,
    input_root_label: str | None = None,
) -> Any:
    output = output or OutputSet(
        config.output_root,
        max_shard_bytes=config.max_shard_bytes,
    )
    started_at = time.monotonic()
    last_progress_at = started_at
    processed_records = 0
    processed_trajectories = 0
    total_records = state.capture_count()
    total_trajectories = state.trajectory_count()

    def log_progress(*, force: bool = False) -> None:
        nonlocal last_progress_at
        now = time.monotonic()
        completed = processed_records + processed_trajectories
        if not force and (
            completed % _PROGRESS_ITEM_INTERVAL != 0
            and now - last_progress_at < _PROGRESS_INTERVAL_SECONDS
        ):
            return
        LOGGER.info(
            "【输出阶段】已处理输入记录=%d/%d，已处理轨迹=%d/%d，耗时=%.1f秒",
            processed_records,
            total_records,
            processed_trajectories,
            total_trajectories,
            now - started_at,
        )
        last_progress_at = now

    LOGGER.info(
        "【输出阶段】开始写入结果：输入记录总数=%d，轨迹总数=%d",
        total_records,
        total_trajectories,
    )
    try:
        output.stats.input_files = total_records
        for (
            source_ref,
            sha256,
            status,
            reason,
            endpoint,
            captured_at,
        ) in state.capture_records():
            if status == "skipped":
                output.write_skipped(reason)
                processed_records += 1
                log_progress()
                continue
            if status == "failed":
                reason_code = reason.partition(":")[0]
                if reason_code not in {
                    "tokenplan_empty_envelope",
                    "invalid_tokenplan_envelope",
                    "sxf_error",
                    "invalid_deepinfra_envelope",
                    "deepinfra_incomplete_envelope",
                }:
                    reason_code = "capture_parse_failed"
                output.write_record(
                    QuarantineRecord(
                        source_ref=source_ref,
                        sha256=sha256,
                        endpoint=endpoint,
                        captured_at=captured_at,
                        normalization_audit=NormalizationAudit(
                            tag=AuditTag.QUARANTINED,
                            reason_codes=[reason_code],
                            issues=[
                                AuditIssue(
                                    code=reason_code,
                                    stage="ingest",
                                    detail=reason,
                                )
                            ],
                        ),
                    )
                )
                processed_records += 1
                log_progress()
                continue
            snapshot = state.get_snapshot(source_ref)
            if snapshot is not None and (record := _snapshot_record(snapshot)):
                output.write_record(record)
            processed_records += 1
            log_progress()

        for _, node, origins in state.iter_trajectories():
            output.write_trajectory(node, origins)
            processed_trajectories += 1
            log_progress()
        manifest = output.close(
            input_root=input_root_label or str(config.input_root),
            input_format=config.input_format,
            config_hash=configuration_hash,
        )
        log_progress(force=True)
        LOGGER.info(
            "【输出阶段】完成：已处理输入记录=%d/%d，已处理轨迹=%d/%d，耗时=%.1f秒",
            processed_records,
            total_records,
            processed_trajectories,
            total_trajectories,
            time.monotonic() - started_at,
        )
        return manifest
    except BaseException:
        output.abort()
        raise


def _ingest_source(
    source: CaptureSource,
    state: StateStore,
    config: PipelineConfig,
    stats: PipelineStats,
    *,
    source_errors_fatal: bool = False,
) -> None:
    started_at = time.monotonic()
    last_progress_at = started_at
    LOGGER.info("【读取阶段】开始读取输入数据")

    def log_progress(*, force: bool = False) -> None:
        nonlocal last_progress_at
        now = time.monotonic()
        if not force and (
            stats.discovered % _PROGRESS_ITEM_INTERVAL != 0
            and now - last_progress_at < _PROGRESS_INTERVAL_SECONDS
        ):
            return
        elapsed = max(now - started_at, 0.001)
        rate = stats.discovered / elapsed
        LOGGER.info(
            "【读取阶段】已发现=%d，已解析=%d，已复用=%d，解析失败=%d，"
            "耗时=%.1f秒，速度=%.1f对象/秒",
            stats.discovered,
            stats.parsed,
            stats.reused,
            stats.parse_failures,
            elapsed,
            rate,
        )
        last_progress_at = now

    state.begin_scan(uuid.uuid4().hex)
    batch_items = 0
    batch_bytes = 0
    batch: Any | None = None

    def open_batch() -> None:
        nonlocal batch
        context = state.write_batch()
        context.__enter__()
        batch = context

    def close_batch(
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: Any = None,
    ) -> None:
        nonlocal batch
        context, batch = batch, None
        if context is not None:
            context.__exit__(exc_type, exc_value, traceback)

    open_batch()

    def rotate_batch(payload_bytes: int) -> None:
        nonlocal batch_items, batch_bytes
        batch_items += 1
        batch_bytes += payload_bytes
        if batch_items < _INGEST_BATCH_ITEMS and batch_bytes < _INGEST_BATCH_BYTES:
            return
        close_batch()
        open_batch()
        batch_items = 0
        batch_bytes = 0

    def process_payload(source_ref: str, payload: bytes, digest: str) -> None:
        stats.discovered += 1
        stats.input_bytes += len(payload)
        endpoint = ""
        captured_at = ""
        if state.capture_matches(source_ref, digest):
            stats.reused += 1
            log_progress()
            return

        try:
            capture = decode_capture(payload, input_format=config.input_format)
            endpoint_value = capture.get("path")
            if config.input_format == "deepinfra":
                outer_request = capture.get("request")
                if isinstance(outer_request, Mapping):
                    endpoint_value = outer_request.get("path")
                request_time = capture.get("request_time")
                if isinstance(request_time, str):
                    captured_at = request_time
            endpoint = (
                endpoint_value.split("?", 1)[0].rstrip("/")
                if isinstance(endpoint_value, str)
                else ""
            )
            source_name: Literal["freerouter", "tokenplan", "sxf", "deepinfra"] = (
                "freerouter"
            )
            response_is_normalized_final = False
            multimodal_file_mapping: list[MediaMapping] = []
            if config.input_format == "tokenplan":
                adapted = adapt_tokenplan_envelope(capture)
                capture = adapted.capture
                source_name = adapted.source_name
                endpoint = adapted.endpoint
                captured_at = adapted.captured_at
                response_is_normalized_final = adapted.response_is_normalized_final
                multimodal_file_mapping = adapted.multimodal_file_mapping
            elif config.input_format == "sxf":
                capture = adapt_sxf_envelope(capture)
                source_name = "sxf"
            elif config.input_format == "deepinfra":
                capture = adapt_deepinfra_envelope(capture)
                source_name = "deepinfra"
            captured_value = capture.get("captured_at")
            captured_at = captured_value if isinstance(captured_value, str) else ""
            snapshot = parse_capture(
                capture,
                source_path=source_ref,
                source_sha256=digest,
                source_name=source_name,
                response_is_normalized_final=response_is_normalized_final,
                multimodal_file_mapping=multimodal_file_mapping,
            )
            state.put_snapshot(snapshot, endpoint=endpoint)
            stats.parsed += 1
        except TokenPlanError as error:
            state.put_failure(
                source_ref,
                digest,
                f"{error.code}: {error.detail}",
                endpoint=endpoint,
                captured_at=captured_at,
            )
            stats.parse_failures += 1
        except SXFError as error:
            state.put_failure(
                source_ref,
                digest,
                f"sxf_error: {error}",
                endpoint=endpoint,
                captured_at=captured_at,
            )
            stats.parse_failures += 1
        except DeepInfraError as error:
            state.put_failure(
                source_ref,
                digest,
                f"{error.code}: {error.detail}",
                endpoint=endpoint,
                captured_at=captured_at,
            )
            stats.parse_failures += 1
        except Exception as error:  # noqa: BLE001 - quarantine per-file defects
            state.put_failure(
                source_ref,
                digest,
                f"{type(error).__name__}: capture could not be normalized",
                endpoint=endpoint,
                captured_at=captured_at,
            )
            stats.parse_failures += 1
        log_progress()

    try:
        streaming = getattr(source, "iter_capture_payloads", None)
        payload_formats = getattr(source, "payload_formats", frozenset({"sxf"}))
        if config.input_format in payload_formats and callable(streaming):
            for source_ref, payload, digest in streaming(config.input_format):
                process_payload(source_ref, payload, digest)
                rotate_batch(len(payload))
        else:
            for capture_ref in source.iter_captures(config.input_format):
                source_ref = capture_ref.source_ref
                try:
                    payload, digest = source.read_capture_bytes(capture_ref)
                except Exception as error:
                    if source_errors_fatal:
                        raise
                    stats.discovered += 1
                    state.put_failure(
                        source_ref,
                        "",
                        f"{type(error).__name__}: capture could not be normalized",
                        endpoint="",
                        captured_at="",
                    )
                    stats.parse_failures += 1
                    log_progress()
                    rotate_batch(0)
                    continue
                process_payload(source_ref, payload, digest)
                rotate_batch(len(payload))
    except BaseException as error:
        close_batch(type(error), error, error.__traceback__)
        raise
    else:
        close_batch()
    state.finish_scan()
    stats.skipped_inputs = sum(
        status == "skipped" for _, _, status, _, _, _ in state.capture_records()
    )
    log_progress(force=True)
    LOGGER.info(
        "【读取阶段】完成：已发现=%d，已解析=%d，解析失败=%d，耗时=%.1f秒",
        stats.discovered,
        stats.parsed,
        stats.parse_failures,
        time.monotonic() - started_at,
    )


def normalize_source(
    source: CaptureSource,
    *,
    input_format: Literal["freerouter", "tokenplan", "sxf", "deepinfra"],
    state_path: Path,
    output_factory: Callable[[], Any],
    max_shard_bytes: int = 512 * 1024 * 1024,
    build_workers: int | None = None,
) -> tuple[PipelineStats, Any]:
    """Normalize a non-filesystem source with ephemeral local state.

    The source and output implementation own remote I/O.  SQLite remains
    local because aggregation requires ordered, random access to snapshots.
    This boundary deliberately has no resume mode: scheduler retries rebuild
    the ephemeral state from the immutable input inventory.
    """

    if input_format not in {"freerouter", "tokenplan", "sxf", "deepinfra"}:
        raise ValueError(f"unsupported input format: {input_format!r}")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    resolved_state = state_path.resolve()
    if resolved_state.is_symlink():
        raise ValueError("state database path must not be a symlink")
    runtime_root = resolved_state.parent
    runtime_config = PipelineConfig(
        input_root=Path("."),
        input_format=input_format,
        output_root=runtime_root,
        state_path=resolved_state,
        resume=False,
        max_shard_bytes=max_shard_bytes,
    )
    configuration_hash = config_hash(runtime_config)
    stats = PipelineStats()
    with (
        _exclusive_lock(resolved_state.with_suffix(".lock")),
        StateStore(resolved_state, ephemeral=True) as state,
    ):
        state.reset_ingest()
        state.set_meta("config_hash", configuration_hash)
        # Remote read errors are infrastructure failures, rather than content
        # defects, and must fail the scheduler task without publication.
        stage_started_at = time.monotonic()
        _ingest_source(
            source,
            state,
            runtime_config,
            stats,
            source_errors_fatal=True,
        )
        stats.ingest_seconds = time.monotonic() - stage_started_at
        if stats.discovered == 0:
            raise ValueError(f"capture source is empty: {source.label}")
        stage_started_at = time.monotonic()
        _build_trajectories(state, stats, build_workers=build_workers)
        stats.build_seconds = time.monotonic() - stage_started_at
        stage_started_at = time.monotonic()
        output_result = _export(
            state,
            runtime_config,
            configuration_hash,
            output=output_factory(),
            input_root_label=source.label,
        )
        stats.export_seconds = time.monotonic() - stage_started_at
        LOGGER.info(
            "【阶段汇总】读取耗时=%.1f秒，构建耗时=%.1f秒，输出耗时=%.1f秒",
            stats.ingest_seconds,
            stats.build_seconds,
            stats.export_seconds,
        )
    return stats, output_result


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
        input_format=config.input_format,
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
        stage_started_at = time.monotonic()
        _ingest_source(
            LocalCaptureSource(input_root),
            state,
            normalized_config,
            stats,
        )
        stats.ingest_seconds = time.monotonic() - stage_started_at
        stage_started_at = time.monotonic()
        _build_trajectories(state, stats)
        stats.build_seconds = time.monotonic() - stage_started_at
        stage_started_at = time.monotonic()
        _export(state, normalized_config, configuration_hash)
        stats.export_seconds = time.monotonic() - stage_started_at
        LOGGER.info(
            "【阶段汇总】读取耗时=%.1f秒，构建耗时=%.1f秒，输出耗时=%.1f秒",
            stats.ingest_seconds,
            stats.build_seconds,
            stats.export_seconds,
        )
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
    "normalize_source",
    "parse_capture",
    "stats_json",
]
