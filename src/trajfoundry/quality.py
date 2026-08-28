"""Deterministic trajectory statistics, tool checks, and strict gating."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

from .audit_codes import (
    DERIVED_SUBAGENT_AUDIT_CODES,
    MOUNT_ONLY_AUDIT_CODES,
    OPAQUE_COMPACTION_CONTEXT,
    PRIMARY_MOUNT_DIAGNOSTIC_CODES,
    RESPONSES_UNSUPPORTED_CALL_EVIDENCE,
)
from .models import (
    AuditIssue,
    AuditTag,
    Completeness,
    Message,
    NormalizationAudit,
    Severity,
    ToolCallCheck,
    ToolCallMismatch,
    TrajectoryNode,
    TypeMismatch,
)
from .tool_names import is_spawn_tool_name

_TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic approximation; never use for billing."""

    return len(_TOKEN_PATTERN.findall(text))


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _declared_types(schema: Any) -> list[str]:
    if isinstance(schema, dict):
        value = schema.get("type")
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
    return []


def _matches_type(value: Any, declared: list[str]) -> bool:
    if not declared:
        return True
    actual = _json_type(value)
    if actual == "integer" and "number" in declared:
        return True
    return actual in declared


def check_tool_calls(
    messages: list[Message],
    tools: list[Any],
) -> tuple[ToolCallCheck, list[str]]:
    definitions = {tool.name: tool for tool in tools}
    check = ToolCallCheck()
    missing: set[str] = set()
    details: list[ToolCallMismatch] = []

    for message_index, message in enumerate(messages):
        for call in message.tool_calls or []:
            check.total_calls += 1
            definition = definitions.get(call.function.name)
            if definition is None:
                check.undefined_tool_calls += 1
                check.mismatch_calls += 1
                missing.add(call.function.name)
                details.append(
                    ToolCallMismatch(
                        tool=call.function.name,
                        tool_call_id=call.id,
                        message_index=message_index,
                        reasons=["undefined_tool"],
                    )
                )
                continue

            arguments = call.function.arguments
            parameters = definition.parameters
            properties = (
                parameters.get("properties") if isinstance(parameters, dict) else None
            )
            if not isinstance(arguments, dict) or not isinstance(properties, dict):
                check.unverifiable_calls += 1
                continue

            check.checked_calls += 1
            reasons: list[str] = []
            extra = sorted(set(arguments) - set(properties))
            required = parameters.get("required", [])
            required = required if isinstance(required, list) else []
            missing_required: list[str] = []
            defaulted_missing = False
            for name in required:
                if name in arguments:
                    continue
                schema = properties.get(name, {})
                declared = _declared_types(schema)
                if isinstance(schema, dict) and (
                    "default" in schema or "null" in declared
                ):
                    defaulted_missing = True
                else:
                    missing_required.append(name)

            mismatched: list[TypeMismatch] = []
            for name, value in arguments.items():
                if name not in properties:
                    continue
                declared = _declared_types(properties[name])
                if not _matches_type(value, declared):
                    mismatched.append(
                        TypeMismatch(
                            arg=name, declared=declared, actual=_json_type(value)
                        )
                    )

            if extra:
                reasons.append("extra_args")
                check.extra_arg_calls += 1
            if missing_required:
                reasons.append("missing_required")
                check.missing_required_calls += 1
            if mismatched:
                reasons.append("type_mismatch")
                check.type_mismatch_calls += 1
            if defaulted_missing:
                check.defaulted_missing_calls += 1
            if reasons:
                check.mismatch_calls += 1
                details.append(
                    ToolCallMismatch(
                        tool=call.function.name,
                        tool_call_id=call.id,
                        message_index=message_index,
                        reasons=reasons,
                        extra_args=extra or None,
                        missing_required=missing_required or None,
                        type_mismatch=mismatched or None,
                    )
                )

    check.mismatches = details[:50]
    check.mismatches_truncated = max(0, len(details) - 50)
    return check, sorted(missing)


def _provider_hallucination_count(audit: NormalizationAudit | None) -> int:
    if audit is None:
        return 0
    evidence = {
        (issue.path, issue.detail)
        for issue in audit.issues
        if issue.code == RESPONSES_UNSUPPORTED_CALL_EVIDENCE
        and issue.stage == "responses"
        and issue.severity == Severity.WARNING
    }
    return len(evidence)


def _pairing_issues(messages: list[Message]) -> list[AuditIssue]:
    calls: dict[str, tuple[str, int]] = {}
    results: dict[str, list[tuple[int, str]]] = defaultdict(list)
    issues: list[AuditIssue] = []
    for index, message in enumerate(messages):
        for call in message.tool_calls or []:
            if call.id in calls:
                issues.append(
                    AuditIssue(
                        code="duplicate_tool_call_id",
                        stage="quality",
                        path=f"/messages/{index}/tool_calls",
                        detail=f"duplicate call id {call.id}",
                    )
                )
            else:
                calls[call.id] = (call.function.name, index)
        if message.role == "tool" and message.tool_call_id:
            results[message.tool_call_id].append((index, message.name or ""))

    for call_id, (name, index) in calls.items():
        matched = results.get(call_id, [])
        if not matched:
            issues.append(
                AuditIssue(
                    code="missing_tool_result",
                    stage="quality",
                    path=f"/messages/{index}",
                    detail=f"{name} call {call_id} has no result",
                )
            )
        elif len(matched) > 1:
            issues.append(
                AuditIssue(
                    code="duplicate_tool_result",
                    stage="quality",
                    path=f"/messages/{matched[1][0]}",
                    detail=f"call {call_id} has {len(matched)} results",
                )
            )
        for result_index, result_name in matched:
            if result_index < index:
                issues.append(
                    AuditIssue(
                        code="tool_result_before_call",
                        stage="quality",
                        path=f"/messages/{result_index}",
                        detail=f"result {call_id} appears before its call",
                    )
                )
            if result_name != name:
                issues.append(
                    AuditIssue(
                        code="tool_result_name_mismatch",
                        stage="quality",
                        path=f"/messages/{result_index}/name",
                        detail=(
                            f"result name {result_name!r} does not match "
                            f"call name {name!r}"
                        ),
                    )
                )
    for call_id, indexed_results in results.items():
        if call_id not in calls:
            issues.append(
                AuditIssue(
                    code="orphan_tool_result",
                    stage="quality",
                    path=f"/messages/{indexed_results[0][0]}",
                    detail=f"result {call_id} has no call",
                )
            )
    return issues


def _local_complete(node: TrajectoryNode, pairing: list[AuditIssue]) -> bool:
    if pairing:
        return False
    if not node.messages or node.messages[-1].role != "assistant":
        return False
    last = node.messages[-1]
    return not bool(last.tool_calls)


def _descendants(children: Iterable[TrajectoryNode]) -> Iterator[TrajectoryNode]:
    for child in children:
        yield child
        yield from _descendants((child.sub_agent_trajectory or {}).values())


def _spawn_call_ids(messages: list[Message]) -> list[str]:
    return [
        call.id
        for message in messages
        for call in (message.tool_calls or [])
        if is_spawn_tool_name(call.function.name)
    ]


def _has_mount_failure(node: TrajectoryNode) -> bool:
    return bool(
        node.normalization_audit
        and any(
            issue.severity == Severity.ERROR
            and issue.code in PRIMARY_MOUNT_DIAGNOSTIC_CODES
            for issue in node.normalization_audit.issues
        )
    )


def _mount_state(
    node: TrajectoryNode,
    children: dict[str, TrajectoryNode] | None = None,
) -> tuple[bool, int]:
    """Return recursive spawn-mount completeness and unkeyed mount count."""

    mounted = children if children is not None else (node.sub_agent_trajectory or {})
    spawn_calls = _spawn_call_ids(node.messages)
    unique_spawn_ids = set(spawn_calls)
    mounted_ids = set(mounted)
    local_complete = (
        len(spawn_calls) == len(unique_spawn_ids)
        and mounted_ids == unique_spawn_ids
        and not _has_mount_failure(node)
    )
    unkeyed_mounts = len(mounted_ids - unique_spawn_ids)
    for child in mounted.values():
        child_complete, child_unkeyed = _mount_state(child)
        local_complete = local_complete and child_complete
        unkeyed_mounts += child_unkeyed
    return local_complete, unkeyed_mounts


def _audit_has_non_mount_failure(audit: NormalizationAudit | None) -> bool:
    if audit is None or audit.tag == AuditTag.PASS:
        return False
    error_codes = {
        issue.code for issue in audit.issues if issue.severity == Severity.ERROR
    } | set(audit.reason_codes)
    if not error_codes:
        return True
    return bool(error_codes - MOUNT_ONLY_AUDIT_CODES)


def _is_derived_audit_issue(issue: AuditIssue) -> bool:
    """Return whether quality enrichment must replace this issue on rerun."""

    return issue.stage == "quality" or issue.code in DERIVED_SUBAGENT_AUDIT_CODES


def _trailing_unanswered_call(
    messages: list[Message], pairing: list[AuditIssue]
) -> bool:
    missing_indices: list[int] = []
    for issue in pairing:
        if issue.code != "missing_tool_result":
            continue
        match = re.fullmatch(r"/messages/(\d+)", issue.path)
        if match:
            missing_indices.append(int(match.group(1)))
    return any(
        not any(message.role == "assistant" for message in messages[index + 1 :])
        for index in missing_indices
    )


def enrich_trajectory(
    node: TrajectoryNode,
    *,
    top_level: bool = True,
    is_subagent: bool = False,
) -> TrajectoryNode:
    """Recompute all derived fields and return a validated copy."""

    children: dict[str, TrajectoryNode] = {}
    for call_id, child in (node.sub_agent_trajectory or {}).items():
        children[call_id] = enrich_trajectory(child, top_level=False, is_subagent=True)

    check, missing_defs = check_tool_calls(node.messages, node.tools)
    check.hallucinated_calls = _provider_hallucination_count(node.normalization_audit)
    pairing = _pairing_issues(node.messages)
    counts = Counter(
        call.function.name
        for message in node.messages
        for call in (message.tool_calls or [])
    )
    total_rounds = sum(message.role == "assistant" for message in node.messages)
    reasoning_tokens = sum(
        estimate_tokens(message.reasoning_content or "")
        for message in node.messages
        if message.role == "assistant"
    )

    local_tool_defs = "incomplete" if missing_defs else "complete"
    descendants = tuple(_descendants(children.values()))
    child_defs_bad = any(child.tool_defs_tag == "incomplete" for child in descendants)
    if local_tool_defs == "incomplete":
        tool_defs_tag = "incomplete"
    elif top_level and child_defs_bad:
        tool_defs_tag = "complete_with_incomplete_sub"
    else:
        tool_defs_tag = "complete"

    local_call_tag = "inconsistent" if check.mismatch_calls else "consistent"
    child_calls_bad = any(
        child.tool_call_tag == "inconsistent" for child in descendants
    )
    child_quality_bad = any(
        child.tool_defs_tag == "incomplete"
        or child.tool_call_tag == "inconsistent"
        or _audit_has_non_mount_failure(child.normalization_audit)
        for child in descendants
    )
    if local_call_tag == "inconsistent":
        tool_call_tag = "inconsistent"
    elif top_level and child_calls_bad:
        tool_call_tag = "consistent_with_inconsistent_sub"
    else:
        tool_call_tag = "consistent"

    spawn_calls = _spawn_call_ids(node.messages)
    subtree_complete, unkeyed_mounts = _mount_state(node, children)
    local_complete = _local_complete(node, pairing)
    has_sub = bool(spawn_calls or children or _has_mount_failure(node))
    if is_subagent and top_level:
        completeness_tag = "orphan_sub"
    elif local_complete and not has_sub:
        completeness_tag = "complete_main_no_sub"
    elif not local_complete and not has_sub:
        completeness_tag = "incomplete_main_no_sub"
    elif local_complete and subtree_complete:
        completeness_tag = "complete_main_with_complete_sub"
    elif local_complete:
        completeness_tag = "complete_main_with_incomplete_sub"
    elif subtree_complete:
        completeness_tag = "incomplete_main_with_complete_sub"
    else:
        completeness_tag = "incomplete_main_with_incomplete_sub"

    audit_issues = [
        *(
            issue
            for issue in (
                node.normalization_audit.issues if node.normalization_audit else []
            )
            if not _is_derived_audit_issue(issue)
        ),
        *pairing,
    ]
    if node.compaction_items:
        audit_issues.append(
            AuditIssue(
                code=OPAQUE_COMPACTION_CONTEXT,
                stage="quality",
                path="/compaction_items",
                detail=(
                    f"trajectory contains {len(node.compaction_items)} opaque "
                    "Responses compaction item(s) that cannot be interpreted"
                ),
            )
        )
    if check.hallucinated_calls:
        audit_issues.append(
            AuditIssue(
                code="hallucinated_tool_call",
                stage="quality",
                detail=(
                    f"{check.hallucinated_calls} provider-confirmed tool call(s) "
                    "were rejected as unsupported"
                ),
            )
        )
    if has_sub and not subtree_complete:
        audit_issues.append(
            AuditIssue(
                code="incomplete_subagent_mount",
                stage="subagents",
                detail="one or more spawn calls could not be mounted uniquely",
            )
        )
    if child_quality_bad:
        audit_issues.append(
            AuditIssue(
                code="subagent_quality_failure",
                stage="quality",
                detail=(
                    "one or more sub-agent trajectories failed audit or "
                    "tool-quality checks"
                ),
            )
        )
    if is_subagent and top_level:
        audit_issues.append(
            AuditIssue(
                code="orphan_subagent",
                stage="subagents",
                detail="sub-agent trajectory has no verified parent mount",
            )
        )
    deduplicated_issues: list[AuditIssue] = []
    seen_issues: set[tuple[str, str, str, str, str]] = set()
    for issue in audit_issues:
        key = (issue.code, issue.stage, issue.severity.value, issue.path, issue.detail)
        if key not in seen_issues:
            seen_issues.add(key)
            deduplicated_issues.append(issue)
    audit_issues = deduplicated_issues
    reason_codes = sorted(
        {issue.code for issue in audit_issues if issue.severity == Severity.ERROR}
    )
    acceptable = (
        not reason_codes
        and tool_defs_tag == "complete"
        and tool_call_tag == "consistent"
        and check.hallucinated_calls == 0
        and completeness_tag
        in {"complete_main_no_sub", "complete_main_with_complete_sub"}
    )
    audit = NormalizationAudit(
        tag=AuditTag.PASS if acceptable else AuditTag.QUARANTINED,
        reason_codes=reason_codes,
        issues=audit_issues,
    )

    update: dict[str, Any] = {
        "sub_agent_trajectory": children or None,
        "total_rounds": total_rounds,
        "total_tool_calls": sum(counts.values()),
        "tool_counts": dict(sorted(counts.items())),
        "reasoning_total_tokens": reasoning_tokens,
        "tool_defs_tag": tool_defs_tag,
        "missing_tool_defs": missing_defs,
        "tool_call_tag": tool_call_tag,
        "tool_call_check": check,
    }
    if top_level:
        update["completeness_tag"] = completeness_tag
        update["completeness"] = Completeness(
            is_subagent=is_subagent,
            spawn_calls=len(spawn_calls),
            mounted_subs=len(children),
            subtree_complete=False if is_subagent else subtree_complete,
            unkeyed_mounts=unkeyed_mounts,
            relay_mounts=len(node.sub_agent_relay_mounts or {}),
            trailing_unanswered_call=_trailing_unanswered_call(node.messages, pairing),
            no_final_assistant_turn=not node.messages
            or node.messages[-1].role != "assistant"
            or bool(node.messages[-1].tool_calls),
        )
        update["normalization_audit"] = audit
    else:
        update["completeness_tag"] = None
        update["completeness"] = None
        update["normalization_audit"] = audit
    return node.model_copy(update=update)


def is_strict_sample(node: TrajectoryNode) -> bool:
    return bool(
        node.normalization_audit
        and node.normalization_audit.tag == AuditTag.PASS
        and node.tool_defs_tag == "complete"
        and node.tool_call_tag == "consistent"
        and node.tool_call_check.hallucinated_calls == 0
        and node.completeness_tag
        in {"complete_main_no_sub", "complete_main_with_complete_sub"}
    )


class StaleDerivedFieldsError(ValueError):
    """Raised when a trajectory's stored quality fields need recomputation."""


def validate_derived_fields(node: TrajectoryNode) -> None:
    """Reject a top-level trajectory whose derived fields are not canonical.

    This check intentionally lives above the generic output projector. Internal
    aggregation hashes also project flat, not-yet-enriched nodes and must remain
    usable while a trajectory is still being assembled.
    """

    if node.completeness is None or node.completeness_tag is None:
        raise StaleDerivedFieldsError(
            "trajectory is missing top-level derived completeness fields"
        )
    recomputed = enrich_trajectory(
        node,
        top_level=True,
        is_subagent=node.completeness.is_subagent,
    )
    if recomputed != node:
        raise StaleDerivedFieldsError(
            "trajectory has stale or inconsistent derived quality fields"
        )
