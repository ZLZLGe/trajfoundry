"""Shared audit-code classifications used across normalization stages."""

from __future__ import annotations

from typing import Literal, TypeAlias, get_args

RESPONSES_UNSUPPORTED_CALL_EVIDENCE = "responses_unsupported_tool_call"

MountDiagnosticCode: TypeAlias = Literal[
    "missing_subagent_marker",
    "missing_parent_thread_id",
    "missing_parent_turn_id",
    "conflicting_parent_thread",
    "conflicting_marker_metadata",
    "duplicate_child_leaves",
    "missing_spawn_call",
    "spawn_routing_mismatch",
    "ambiguous_spawn_call",
    "spawn_missing_call_id",
    "spawn_missing_turn_id",
    "malformed_spawn_arguments",
    "missing_spawn_agent_name",
    "conflicting_spawn_agent_name",
    "conflicting_spawn_evidence",
    "missing_parent_leaf",
    "ambiguous_parent_leaf",
    "multiple_children_for_spawn",
    "missing_child_agent_name",
    "conflicting_child_agent_name",
    "duplicate_agent_message_id",
    "conflicting_agent_message_evidence",
    "unmatched_agent_message",
    "agent_relay_recipient_mismatch",
    "agent_relay_before_spawn",
    "ambiguous_agent_relay",
    "relay_id_conflict",
    "mount_cycle",
    "unreachable_subagent",
    "unmounted_spawn_call",
    "preexisting_mount_conflict",
]

# All graph and relay diagnostics emitted by sub-agent mount planning.
MOUNT_DIAGNOSTIC_CODES = frozenset(get_args(MountDiagnosticCode))

# Relay metadata is optional evidence layered on top of an already-proven
# parent/child mount.  Failures in this set may quarantine the record, but
# cannot make the primary spawn graph incomplete.
_RELAY_MOUNT_DIAGNOSTIC_CODES = frozenset(
    {
        "missing_spawn_agent_name",
        "conflicting_spawn_agent_name",
        "missing_child_agent_name",
        "conflicting_child_agent_name",
        "duplicate_agent_message_id",
        "conflicting_agent_message_evidence",
        "unmatched_agent_message",
        "agent_relay_recipient_mismatch",
        "agent_relay_before_spawn",
        "ambiguous_agent_relay",
        "relay_id_conflict",
    }
)
PRIMARY_MOUNT_DIAGNOSTIC_CODES = MOUNT_DIAGNOSTIC_CODES - _RELAY_MOUNT_DIAGNOSTIC_CODES

# These aggregate/quality codes also describe mounting only.  They must not be
# promoted to ``subagent_quality_failure`` in ancestors.
MOUNT_ONLY_AUDIT_CODES = MOUNT_DIAGNOSTIC_CODES | {
    "incomplete_subagent_mount",
    "orphan_subagent",
}

# Quality enrichment owns every issue emitted at the quality stage, plus these
# two aggregate sub-agent issues.  They are discarded before every enrichment
# pass and deterministically recomputed from messages and primary evidence.
DERIVED_SUBAGENT_AUDIT_CODES = {
    "incomplete_subagent_mount",
    "orphan_subagent",
}


__all__ = [
    "DERIVED_SUBAGENT_AUDIT_CODES",
    "MOUNT_DIAGNOSTIC_CODES",
    "MOUNT_ONLY_AUDIT_CODES",
    "PRIMARY_MOUNT_DIAGNOSTIC_CODES",
    "RESPONSES_UNSUPPORTED_CALL_EVIDENCE",
    "MountDiagnosticCode",
]
