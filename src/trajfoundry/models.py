"""Canonical and intermediate data contracts for TrajFoundry."""

from __future__ import annotations

import ipaddress
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

_REDACTED = "[REDACTED]"
_CREDENTIAL_KEY = r"""
    (?:
        authorization|proxy[-_ ]?authorization|cookie|set[-_ ]?cookie|
        api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|client[-_ ]?secret|
        proxy[-_ ]?(?:credential|password|secret|token)|password|passwd
    )
"""
_QUOTED_CREDENTIAL = re.compile(
    rf"(?ix)(?P<prefix>[\"']?{_CREDENTIAL_KEY}[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*')"
)
_UNQUOTED_CREDENTIAL = re.compile(
    rf"(?ix)(?P<prefix>[\"']?{_CREDENTIAL_KEY}[\"']?\s*[:=]\s*)"
    r"[^,;\r\n}\]]+"
)
_AUTH_SCHEME = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=:-]+")
_CREDENTIAL_URL = re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@")
_OPENAI_STYLE_KEY = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk|rk)-(?:proj-)?[A-Za-z0-9_-]{8,}"
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
_MAX_DIAGNOSTIC_DETAIL = 1024


def _redact_ip(match: re.Match[str]) -> str:
    candidate = match.group(0)
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return candidate
    return _REDACTED


def sanitize_diagnostic_detail(detail: str) -> str:
    """Remove credentials and network identifiers from non-trajectory text.

    This function is intentionally scoped to audit diagnostics. Message bodies,
    reasoning, tool arguments, and tool results never pass through it.
    """

    value = _AUTH_SCHEME.sub(lambda match: f"{match.group(1)} {_REDACTED}", detail)
    value = _CREDENTIAL_URL.sub(rf"\1{_REDACTED}@", value)
    value = _QUOTED_CREDENTIAL.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED}", value
    )
    value = _UNQUOTED_CREDENTIAL.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED}", value
    )
    value = _OPENAI_STYLE_KEY.sub(_REDACTED, value)
    value = _JWT.sub(_REDACTED, value)
    value = _IPV4.sub(_redact_ip, value)
    value = _IPV6.sub(_redact_ip, value)
    if len(value) > _MAX_DIAGNOSTIC_DETAIL:
        value = f"{value[:_MAX_DIAGNOSTIC_DETAIL]}...[TRUNCATED]"
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AuditTag(StrEnum):
    PASS = "pass"
    QUARANTINED = "quarantined"
    EXCLUDED = "excluded"


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class AuditIssue(StrictModel):
    code: str
    stage: str
    severity: Severity = Severity.ERROR
    path: str = ""
    detail: str = ""

    @field_validator("detail")
    @classmethod
    def sanitize_detail(cls, value: str) -> str:
        return sanitize_diagnostic_detail(value)


class NormalizationAudit(StrictModel):
    tag: AuditTag
    reason_codes: list[str] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)


class FunctionCall(StrictModel):
    name: str
    arguments: Any


class ToolCall(StrictModel):
    type: Literal["function"] = "function"
    id: str
    function: FunctionCall


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    reasoning: JsonValue | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def validate_role_fields(self) -> Message:
        if self.role == "assistant":
            if self.reasoning_content is None:
                raise ValueError("assistant messages require reasoning_content")
            if self.tool_calls is not None and not self.tool_calls:
                raise ValueError("assistant tool_calls must contain at least one call")
            if self.tool_call_id is not None or self.name is not None:
                raise ValueError("assistant messages cannot be tool results")
        elif self.role == "tool":
            if not self.tool_call_id or self.name is None:
                raise ValueError("tool messages require tool_call_id and name")
            if any(
                value is not None
                for value in (
                    self.reasoning_content,
                    self.reasoning_details,
                    self.reasoning,
                    self.tool_calls,
                )
            ):
                raise ValueError("tool messages cannot contain assistant fields")
        elif any(
            value is not None
            for value in (
                self.reasoning_content,
                self.reasoning_details,
                self.reasoning,
                self.tool_calls,
                self.tool_call_id,
                self.name,
            )
        ):
            raise ValueError(f"{self.role} messages may only contain role and content")
        return self


class ToolDefinition(StrictModel):
    type: Literal["function"] = "function"
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ServerToolCall(StrictModel):
    name: str
    id: str
    arguments: Any
    origin: Literal["history", "response"]
    result: dict[str, Any] | None = None


class AgentMessageRecord(StrictModel):
    """Lossless provider record for a Responses ``agent_message`` item."""

    origin: Literal["history", "response"]
    item_index: int = Field(ge=0)
    item: dict[str, JsonValue]


class AgentMessageEvidence(AgentMessageRecord):
    """Raw agent message plus structured provider-order routing evidence."""

    preceding_completed_spawn_call_ids: list[str] = Field(default_factory=list)


class TypeMismatch(StrictModel):
    arg: str
    declared: list[str]
    actual: str


class ToolCallMismatch(StrictModel):
    tool: str
    tool_call_id: str
    message_index: int
    reasons: list[
        Literal["undefined_tool", "extra_args", "missing_required", "type_mismatch"]
    ]
    extra_args: list[str] | None = None
    missing_required: list[str] | None = None
    type_mismatch: list[TypeMismatch] | None = None


class ToolCallCheck(StrictModel):
    total_calls: int = 0
    hallucinated_calls: int = 0
    undefined_tool_calls: int = 0
    unverifiable_calls: int = 0
    checked_calls: int = 0
    extra_arg_calls: int = 0
    missing_required_calls: int = 0
    type_mismatch_calls: int = 0
    defaulted_missing_calls: int = 0
    mismatch_calls: int = 0
    mismatches: list[ToolCallMismatch] = Field(default_factory=list)
    mismatches_truncated: int = 0


class Metadata(StrictModel):
    source_file: str
    source_name: Literal["freerouter"] = "freerouter"
    line_no: int = 0
    created_at: str = ""


class Completeness(StrictModel):
    is_subagent: bool
    spawn_calls: int = 0
    mounted_subs: int = 0
    subtree_complete: bool = True
    unkeyed_mounts: int = 0
    relay_mounts: int = 0
    trailing_unanswered_call: bool = False
    no_final_assistant_turn: bool = False


class TrajectoryNode(StrictModel):
    messages: list[Message]
    tools: list[ToolDefinition]
    agent_messages: list[AgentMessageRecord] = Field(default_factory=list)
    instructions: str = ""
    termination: str = ""
    harness: str = "unknown"
    model: str = ""
    source: str
    total_rounds: int = 0
    total_tool_calls: int = 0
    tool_counts: dict[str, int] = Field(default_factory=dict)
    reasoning_total_tokens: int = 0
    tool_defs_tag: Literal["complete", "incomplete", "complete_with_incomplete_sub"] = (
        "complete"
    )
    missing_tool_defs: list[str] = Field(default_factory=list)
    tool_call_tag: Literal[
        "consistent", "inconsistent", "consistent_with_inconsistent_sub"
    ] = "consistent"
    tool_call_check: ToolCallCheck = Field(default_factory=ToolCallCheck)
    server_tool_calls: list[ServerToolCall] = Field(default_factory=list)
    metadata: Metadata
    sub_agent_trajectory: dict[str, TrajectoryNode] | None = None
    sub_agent_relay_mounts: dict[str, str] | None = None
    completeness_tag: (
        Literal[
            "orphan_sub",
            "complete_main_no_sub",
            "incomplete_main_no_sub",
            "complete_main_with_complete_sub",
            "complete_main_with_incomplete_sub",
            "incomplete_main_with_complete_sub",
            "incomplete_main_with_incomplete_sub",
        ]
        | None
    ) = None
    completeness: Completeness | None = None
    normalization_audit: NormalizationAudit | None = None


class Snapshot(StrictModel):
    source_path: str
    source_sha256: str
    source_partition: str
    session_id: str
    thread_id: str
    turn_id: str = ""
    parent_thread_id: str = ""
    parent_turn_id: str = ""
    forked_from_thread_id: str = ""
    subagent_marker: str = ""
    provider: Literal["openai", "anthropic"]
    operation: Literal["responses", "messages", "count_tokens"]
    outcome: Literal[
        "success", "api_error", "transport_error", "truncated", "capture_invalid"
    ]
    captured_at: str = ""
    request_id: str = ""
    model: str = ""
    harness: str = "unknown"
    instructions: str = ""
    history: list[Message] = Field(default_factory=list)
    response: list[Message] = Field(default_factory=list)
    tools: list[ToolDefinition] = Field(default_factory=list)
    server_tool_calls: list[ServerToolCall] = Field(default_factory=list)
    agent_messages: list[AgentMessageEvidence] = Field(default_factory=list)
    termination: str = ""
    wire_complete: bool = False
    issues: list[AuditIssue] = Field(default_factory=list)


class QuarantineRecord(StrictModel):
    source_ref: str
    sha256: str
    endpoint: str = ""
    captured_at: str = ""
    normalization_audit: NormalizationAudit


TrajectoryNode.model_rebuild()
