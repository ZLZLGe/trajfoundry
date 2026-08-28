"""Canonical serialization and provenance-independent hashes."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import orjson

from .audit_codes import RESPONSES_UNSUPPORTED_CALL_EVIDENCE
from .models import CompactionRecord, Message, Severity, TrajectoryNode
from .output_contract import project_message, project_trajectory


def canonical_json(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def message_fingerprint(message: Message) -> str:
    payload = project_message(message)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def compaction_record_key(record: CompactionRecord) -> tuple[int, int, bytes]:
    """Return the stable provider-position order for opaque compaction items."""

    return (
        0 if record.origin == "history" else 1,
        record.item_index,
        canonical_json(record.item),
    )


def canonical_compaction_records(
    records: Iterable[CompactionRecord],
) -> list[CompactionRecord]:
    """Deduplicate exact replay records without interpreting their contents."""

    unique: dict[bytes, CompactionRecord] = {}
    for record in records:
        encoded = canonical_json(record.model_dump(mode="json", exclude_none=False))
        unique.setdefault(encoded, record)
    return sorted(unique.values(), key=compaction_record_key)


def compaction_signature(records: Iterable[CompactionRecord]) -> bytes:
    """Return an exact, position-sensitive opaque-context signature."""

    ordered = canonical_compaction_records(records)
    return canonical_json(
        [record.model_dump(mode="json", exclude_none=False) for record in ordered]
    )


def _semantic_provider_evidence(
    node: TrajectoryNode,
) -> dict[str, list[dict[str, str]]]:
    audit = node.normalization_audit
    if audit is None:
        return {}
    responses_rejections = {
        (issue.path, issue.detail)
        for issue in audit.issues
        if issue.code == RESPONSES_UNSUPPORTED_CALL_EVIDENCE
        and issue.stage == "responses"
        and issue.severity == Severity.WARNING
    }
    if not responses_rejections:
        return {}
    return {
        RESPONSES_UNSUPPORTED_CALL_EVIDENCE: [
            {"path": path, "detail": detail}
            for path, detail in sorted(responses_rejections)
        ]
    }


def semantic_payload(node: TrajectoryNode) -> dict[str, Any]:
    # Use the same field projection as persisted and published trajectories.
    # Completeness is deliberately omitted because it is derived below the
    # semantic identity boundary.
    has_completeness = (
        node.completeness_tag is not None or node.completeness is not None
    )
    payload = project_trajectory(node, top_level=has_completeness)
    provider_evidence = _semantic_provider_evidence(node)
    for key in (
        "source",
        "metadata",
        "normalization_audit",
        "completeness",
        "completeness_tag",
        "total_rounds",
        "total_tool_calls",
        "tool_counts",
        "reasoning_total_tokens",
        "tool_defs_tag",
        "missing_tool_defs",
        "tool_call_tag",
        "tool_call_check",
    ):
        payload.pop(key, None)
    if provider_evidence:
        payload["provider_evidence"] = provider_evidence
    children = payload.get("sub_agent_trajectory")
    if isinstance(children, dict):
        payload["sub_agent_trajectory"] = {
            call_id: semantic_payload((node.sub_agent_trajectory or {})[call_id])
            for call_id in sorted(children)
        }
    return payload


def trajectory_id(node: TrajectoryNode) -> str:
    return hashlib.sha256(canonical_json(semantic_payload(node))).hexdigest()
