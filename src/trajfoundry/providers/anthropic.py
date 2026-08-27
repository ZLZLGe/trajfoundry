"""Normalize Anthropic Messages API captures into provider-neutral snapshots.

The freerouter recorder stores streaming responses as decoded SSE envelopes of
the form ``{"seq": 1, "event": "message_start", "data": {...}}``.  The
adapter also accepts raw SSE text, which makes the boundary useful outside the
recorder and keeps all wire-completeness checks in one place.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from trajfoundry.models import (
    AuditIssue,
    FunctionCall,
    Message,
    ServerToolCall,
    Severity,
    Snapshot,
    ToolCall,
    ToolDefinition,
)

_STAGE = "provider.anthropic"
_KNOWN_NON_TEXT_BLOCKS = {"document", "image", "search_result"}
_SERVER_RESULT_TYPES = {
    "bash_code_execution_tool_result",
    "code_execution_tool_result",
    "mcp_tool_result",
    "server_tool_result",
    "text_editor_code_execution_tool_result",
    "web_fetch_tool_result",
    "web_search_tool_result",
}


class AnthropicCaptureError(ValueError):
    """Raised when the supplied object is not an Anthropic capture."""


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _issue(
    code: str,
    detail: str,
    *,
    path: str = "",
    severity: Severity = Severity.ERROR,
) -> AuditIssue:
    return AuditIssue(
        code=code,
        stage=_STAGE,
        severity=severity,
        path=path,
        detail=detail,
    )


def _stable_api_error_detail(error: Any, *, status: int | None = None) -> str:
    """Summarize an API error without retaining its free-form message."""

    parts: list[str] = []
    if status is not None:
        parts.append(f"HTTP {status}")
    if isinstance(error, dict):
        for key in ("type", "code"):
            value = error.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                atom = str(value)
                if 0 < len(atom) <= 80 and all(
                    character.isalnum() or character in "._-:" for character in atom
                ):
                    parts.append(f"{key}={atom}")
    suffix = f" ({'; '.join(parts)})" if parts else ""
    return f"Anthropic API error{suffix}"


def _json_string(value: Any) -> str:
    """Project a tool result to a string without reparsing string payloads."""

    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
            else:
                parts.append(_compact_json(block))
        return "".join(parts)
    return _compact_json(value)


def _message_content(
    content: Any,
    *,
    path: str,
    issues: list[AuditIssue],
) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        issues.append(
            _issue(
                "invalid_message_content",
                "message content must be a string or an array",
                path=path,
            )
        )
        return _json_string(content)

    parts: list[str] = []
    for index, block in enumerate(content):
        block_path = f"{path}[{index}]"
        if not isinstance(block, dict):
            issues.append(
                _issue(
                    "invalid_content_block",
                    "content block must be an object",
                    path=block_path,
                )
            )
            parts.append(_compact_json(block))
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block_type in _KNOWN_NON_TEXT_BLOCKS:
            semantic = {
                key: value for key, value in block.items() if key != "cache_control"
            }
            parts.append(_compact_json(semantic))
        else:
            issues.append(
                _issue(
                    "unsupported_content_block",
                    f"unsupported content block type: {block_type!r}",
                    path=block_path,
                )
            )
            parts.append(_compact_json(block))
    return "".join(parts)


def _system_content(
    value: Any,
    *,
    path: str,
    issues: list[AuditIssue],
) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        issues.append(
            _issue(
                "invalid_system_prompt",
                "system must be a string or an array of text blocks",
                path=path,
            )
        )
        return _json_string(value)

    parts: list[str] = []
    for index, block in enumerate(value):
        block_path = f"{path}[{index}]"
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            # cache_control is intentionally transport metadata, not content.
            parts.append(block["text"])
        else:
            issues.append(
                _issue(
                    "unsupported_system_block",
                    "system arrays may only contain text blocks",
                    path=block_path,
                )
            )
            parts.append(_compact_json(block))
    return "".join(parts)


def _normalize_tool_input(value: Any, *, present: bool) -> Any:
    """Normalize Anthropic tool input without losing malformed wire values."""

    if not present:
        return {}
    if not isinstance(value, str):
        return value

    def reject_nonstandard_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant: {constant}")

    try:
        return json.loads(value, parse_constant=reject_nonstandard_constant)
    except (json.JSONDecodeError, ValueError):
        return {"raw": value}


@dataclass
class _PendingClientResult:
    message: Message
    identifier: str
    path: str


@dataclass
class _PendingServerResult:
    identifier: str
    payload: dict[str, Any]
    origin: str
    path: str


@dataclass
class _ParseContext:
    issues: list[AuditIssue] = field(default_factory=list)
    tool_names: dict[str, str] = field(default_factory=dict)
    server_calls: list[ServerToolCall] = field(default_factory=list)
    server_by_id: dict[str, ServerToolCall] = field(default_factory=dict)
    pending_client_results: list[_PendingClientResult] = field(default_factory=list)
    pending_server_results: list[_PendingServerResult] = field(default_factory=list)
    missing_id_counts: dict[str, int] = field(default_factory=dict)

    def missing_id(self, kind: str) -> str:
        index = self.missing_id_counts.get(kind, 0) + 1
        self.missing_id_counts[kind] = index
        return f"missing-{kind}:{index}"

    def add_server_call(
        self,
        block: dict[str, Any],
        *,
        origin: str,
        path: str,
    ) -> None:
        identifier = block.get("id")
        name = block.get("name")
        if not isinstance(identifier, str) or not identifier:
            self.issues.append(
                _issue(
                    "invalid_server_tool_call",
                    "server tool call requires a non-empty id",
                    path=path,
                )
            )
            identifier = self.missing_id("server-tool-call")
        if not isinstance(name, str) or not name:
            self.issues.append(
                _issue(
                    "invalid_server_tool_call",
                    "server tool call requires a non-empty name",
                    path=path,
                )
            )
            name = _text(name)
        call = ServerToolCall(
            name=name,
            id=identifier,
            arguments=_normalize_tool_input(
                block.get("input"), present="input" in block
            ),
            origin=origin,  # type: ignore[arg-type]
            result={
                key: block[key]
                for key in ("result", "results", "output")
                if key in block
            }
            or None,
        )
        self.server_calls.append(call)
        if identifier in self.server_by_id:
            self.issues.append(
                _issue(
                    "duplicate_server_tool_call_id",
                    f"duplicate server tool call id: {identifier}",
                    path=path,
                )
            )
        else:
            self.server_by_id[identifier] = call

    def add_server_result(
        self,
        block: dict[str, Any],
        *,
        origin: str,
        path: str,
    ) -> None:
        tool_use_id = block.get("tool_use_id")
        fallback_id = block.get("id")
        identifier = (
            tool_use_id if isinstance(tool_use_id, str) and tool_use_id else fallback_id
        )
        if not isinstance(identifier, str) or not identifier:
            self.issues.append(
                _issue(
                    "invalid_server_tool_result",
                    "server tool result requires a non-empty tool_use_id or id",
                    path=path,
                )
            )
            identifier = f"missing:server-result:{origin}:{path}"
        self.pending_server_results.append(
            _PendingServerResult(
                identifier=identifier,
                payload=dict(block),
                origin=origin,
                path=path,
            )
        )

    def add_client_result(
        self,
        message: Message,
        *,
        identifier: str,
        path: str,
    ) -> None:
        self.pending_client_results.append(
            _PendingClientResult(
                message=message,
                identifier=identifier,
                path=path,
            )
        )

    def resolve_pending_results(self) -> None:
        seen_client_results: set[str] = set()
        for pending in self.pending_client_results:
            if pending.identifier in seen_client_results:
                self.issues.append(
                    _issue(
                        "duplicate_tool_result",
                        f"multiple client tool results use id: {pending.identifier}",
                        path=pending.path,
                    )
                )
            seen_client_results.add(pending.identifier)

            matched = pending.identifier in self.tool_names
            if not pending.message.name and matched:
                pending.message.name = self.tool_names[pending.identifier]
            if not matched:
                self.issues.append(
                    _issue(
                        "orphan_tool_result",
                        f"no client tool call found for result id: {pending.identifier!r}",
                        path=pending.path,
                    )
                )
            if not pending.message.name:
                self.issues.append(
                    _issue(
                        "missing_tool_result_name",
                        "tool message has no resolvable name",
                        path=pending.path,
                    )
                )

        seen_server_results: set[str] = {
            call.id for call in self.server_calls if call.result is not None
        }
        for pending in self.pending_server_results:
            call = self.server_by_id.get(pending.identifier)
            if pending.identifier in seen_server_results:
                self.issues.append(
                    _issue(
                        "duplicate_server_tool_result",
                        f"multiple server tool results use id: {pending.identifier}",
                        path=pending.path,
                    )
                )
            seen_server_results.add(pending.identifier)

            if call is None:
                self.issues.append(
                    _issue(
                        "orphan_server_tool_result",
                        "no server tool call found for result id: "
                        f"{pending.identifier!r}",
                        path=pending.path,
                    )
                )
                self.server_calls.append(
                    ServerToolCall(
                        name="",
                        id=pending.identifier,
                        arguments={},
                        origin=pending.origin,  # type: ignore[arg-type]
                        result=pending.payload,
                    )
                )
            elif call.result is None:
                call.result = pending.payload
            else:
                # The schema has one result per record.  A repeated result is
                # represented by a second record so no provider evidence is lost.
                self.server_calls.append(
                    call.model_copy(update={"result": pending.payload}, deep=True)
                )


def _assistant_from_blocks(
    blocks: Any,
    *,
    context: _ParseContext,
    origin: str,
    path: str,
) -> Message:
    if isinstance(blocks, str):
        return Message(role="assistant", content=blocks, reasoning_content="")
    if not isinstance(blocks, list):
        context.issues.append(
            _issue(
                "invalid_assistant_content",
                "assistant content must be a string or an array",
                path=path,
            )
        )
        return Message(
            role="assistant",
            content=_json_string(blocks),
            reasoning_content="",
        )

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    reasoning_details: list[Any] = []
    tool_calls: list[ToolCall] = []
    for index, block in enumerate(blocks):
        block_path = f"{path}[{index}]"
        if not isinstance(block, dict):
            context.issues.append(
                _issue(
                    "invalid_content_block",
                    "content block must be an object",
                    path=block_path,
                )
            )
            text_parts.append(_compact_json(block))
            continue

        block_type = block.get("type")
        if block_type == "text":
            value = block.get("text")
            if isinstance(value, str):
                text_parts.append(value)
            else:
                context.issues.append(
                    _issue(
                        "invalid_text_block",
                        "text block requires string text",
                        path=block_path,
                    )
                )
                text_parts.append(_json_string(value))
        elif block_type == "thinking":
            reasoning_details.append(dict(block))
            value = block.get("thinking")
            if isinstance(value, str):
                thinking_parts.append(value)
            else:
                context.issues.append(
                    _issue(
                        "invalid_thinking_block",
                        "thinking block requires string thinking",
                        path=block_path,
                    )
                )
                thinking_parts.append(_json_string(value))
        elif block_type == "redacted_thinking":
            reasoning_details.append(dict(block))
            context.issues.append(
                _issue(
                    "redacted_thinking_preserved",
                    "redacted thinking is preserved only in reasoning_details",
                    path=block_path,
                    severity=Severity.WARNING,
                )
            )
        elif block_type == "tool_use":
            identifier = block.get("id")
            name = block.get("name")
            if not isinstance(identifier, str) or not identifier:
                context.issues.append(
                    _issue(
                        "invalid_tool_call",
                        "tool_use requires a non-empty id",
                        path=block_path,
                    )
                )
                identifier = context.missing_id("client-tool-call")
            if not isinstance(name, str) or not name:
                context.issues.append(
                    _issue(
                        "invalid_tool_call",
                        "tool_use requires a non-empty name",
                        path=block_path,
                    )
                )
                name = _text(name)
            tool_calls.append(
                ToolCall(
                    id=identifier,
                    function=FunctionCall(
                        name=name,
                        arguments=_normalize_tool_input(
                            block.get("input"), present="input" in block
                        ),
                    ),
                )
            )
            if identifier in context.tool_names:
                context.issues.append(
                    _issue(
                        "duplicate_tool_call_id",
                        f"duplicate client tool call id: {identifier}",
                        path=block_path,
                    )
                )
            else:
                context.tool_names[identifier] = name
        elif block_type in {"server_tool_use", "mcp_tool_use"}:
            context.add_server_call(
                block,
                origin=origin,
                path=block_path,
            )
        elif block_type in _SERVER_RESULT_TYPES or (
            isinstance(block_type, str) and block_type.endswith("_tool_result")
        ):
            context.add_server_result(block, origin=origin, path=block_path)
        elif block_type in _KNOWN_NON_TEXT_BLOCKS:
            semantic = {
                key: value for key, value in block.items() if key != "cache_control"
            }
            text_parts.append(_compact_json(semantic))
        else:
            context.issues.append(
                _issue(
                    "unsupported_content_block",
                    f"unsupported assistant content block type: {block_type!r}",
                    path=block_path,
                )
            )
            text_parts.append(_compact_json(block))

    return Message(
        role="assistant",
        content="".join(text_parts),
        reasoning_content="".join(thinking_parts),
        reasoning_details=reasoning_details or None,
        tool_calls=tool_calls or None,
    )


def _user_messages(
    content: Any,
    *,
    context: _ParseContext,
    path: str,
) -> list[Message]:
    if isinstance(content, str):
        return [Message(role="user", content=content)]
    if not isinstance(content, list):
        context.issues.append(
            _issue(
                "invalid_user_content",
                "user content must be a string or an array",
                path=path,
            )
        )
        return [Message(role="user", content=_json_string(content))]

    result: list[Message] = []
    plain_parts: list[str] = []

    def flush_plain() -> None:
        if plain_parts:
            result.append(Message(role="user", content="".join(plain_parts)))
            plain_parts.clear()

    for index, block in enumerate(content):
        block_path = f"{path}[{index}]"
        if not isinstance(block, dict):
            context.issues.append(
                _issue(
                    "invalid_content_block",
                    "content block must be an object",
                    path=block_path,
                )
            )
            plain_parts.append(_compact_json(block))
            continue
        block_type = block.get("type")
        if block_type == "tool_result":
            flush_plain()
            identifier = block.get("tool_use_id")
            if not isinstance(identifier, str) or not identifier:
                context.issues.append(
                    _issue(
                        "invalid_tool_result",
                        "tool_result requires a non-empty tool_use_id",
                        path=block_path,
                    )
                )
                identifier = context.missing_id("client-tool-result")
            message = Message(
                role="tool",
                content=_json_string(block.get("content", "")),
                tool_call_id=identifier,
                name="",
            )
            result.append(message)
            context.add_client_result(
                message,
                identifier=identifier,
                path=block_path,
            )
        elif block_type == "text" and isinstance(block.get("text"), str):
            plain_parts.append(block["text"])
        elif block_type in _KNOWN_NON_TEXT_BLOCKS:
            semantic = {
                key: value for key, value in block.items() if key != "cache_control"
            }
            plain_parts.append(_compact_json(semantic))
        else:
            context.issues.append(
                _issue(
                    "unsupported_content_block",
                    f"unsupported user content block type: {block_type!r}",
                    path=block_path,
                )
            )
            plain_parts.append(_compact_json(block))
    flush_plain()
    if not result and not content:
        result.append(Message(role="user", content=""))
    return result


def _parse_history(request: dict[str, Any], context: _ParseContext) -> list[Message]:
    history: list[Message] = []
    if "system" in request and request["system"] is not None:
        history.append(
            Message(
                role="system",
                content=_system_content(
                    request["system"],
                    path="request_body.system",
                    issues=context.issues,
                ),
            )
        )

    raw_messages = request.get("messages", [])
    if not isinstance(raw_messages, list):
        context.issues.append(
            _issue(
                "invalid_messages",
                "request messages must be an array",
                path="request_body.messages",
            )
        )
        return history

    for index, raw_message in enumerate(raw_messages):
        path = f"request_body.messages[{index}]"
        if not isinstance(raw_message, dict):
            context.issues.append(
                _issue("invalid_message", "message must be an object", path=path)
            )
            continue
        role = raw_message.get("role")
        content = raw_message.get("content", "")
        if role == "assistant":
            history.append(
                _assistant_from_blocks(
                    content,
                    context=context,
                    origin="history",
                    path=f"{path}.content",
                )
            )
        elif role == "user":
            history.extend(
                _user_messages(content, context=context, path=f"{path}.content")
            )
        elif role in {"system", "developer"}:
            history.append(
                Message(
                    role=role,
                    content=_message_content(
                        content,
                        path=f"{path}.content",
                        issues=context.issues,
                    ),
                )
            )
        elif role == "tool":
            identifier = raw_message.get("tool_call_id")
            if not isinstance(identifier, str) or not identifier:
                context.issues.append(
                    _issue(
                        "invalid_tool_result",
                        "tool message requires a non-empty tool_call_id",
                        path=path,
                    )
                )
                identifier = context.missing_id("client-tool-result")
            message = Message(
                role="tool",
                content=_json_string(content),
                tool_call_id=identifier,
                name="",
            )
            history.append(message)
            context.add_client_result(
                message,
                identifier=identifier,
                path=path,
            )
        else:
            context.issues.append(
                _issue(
                    "unsupported_message_role",
                    f"unsupported Anthropic message role: {role!r}",
                    path=path,
                )
            )
    return history


def _parse_tools(
    request: dict[str, Any], issues: list[AuditIssue]
) -> list[ToolDefinition]:
    raw_tools = request.get("tools", [])
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        issues.append(
            _issue(
                "invalid_tools",
                "request tools must be an array",
                path="request_body.tools",
            )
        )
        return []

    tools: list[ToolDefinition] = []
    for index, raw_tool in enumerate(raw_tools):
        path = f"request_body.tools[{index}]"
        if not isinstance(raw_tool, dict):
            issues.append(
                _issue("invalid_tool_definition", "tool must be an object", path=path)
            )
            continue

        tool_type = raw_tool.get("type")
        is_client_tool = tool_type in {None, "custom", "function"} and (
            "input_schema" in raw_tool or tool_type in {None, "custom"}
        )
        if not is_client_tool:
            # Anthropic's provider-executed tools use versioned type names and
            # intentionally do not enter the client tool-definition set.
            continue

        name = raw_tool.get("name")
        if not isinstance(name, str) or not name:
            issues.append(
                _issue(
                    "invalid_tool_definition",
                    "client tool definition requires a non-empty name",
                    path=path,
                )
            )
            continue
        description = raw_tool.get("description", "")
        if not isinstance(description, str):
            issues.append(
                _issue(
                    "invalid_tool_definition",
                    "client tool description must be a string",
                    path=f"{path}.description",
                )
            )
            description = _json_string(description)
        schema = raw_tool.get("input_schema", {})
        if not isinstance(schema, dict):
            issues.append(
                _issue(
                    "invalid_tool_schema",
                    "client tool input_schema must be an object",
                    path=f"{path}.input_schema",
                )
            )
            schema = {}
        tools.append(
            ToolDefinition(name=name, description=description, parameters=schema)
        )
    return tools


def _decode_sse_text(body: str, issues: list[AuditIssue]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name = ""
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if not event_name and not data_lines:
            return
        raw_data = "\n".join(data_lines)
        if raw_data == "[DONE]":
            data: Any = {"type": "message_stop"}
            name = event_name or "message_stop"
        else:
            try:
                data = json.loads(raw_data)
            except (TypeError, json.JSONDecodeError):
                issues.append(
                    _issue(
                        "invalid_sse_data",
                        "SSE data is not valid JSON",
                        path=f"response_body[{len(events)}]",
                    )
                )
                data = {}
            name = event_name or (data.get("type") if isinstance(data, dict) else "")
        events.append({"event": name, "data": data})
        event_name = ""
        data_lines = []

    for raw_line in body.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            flush()
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_name = line[6:].lstrip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line.startswith(("id:", "retry:")):
            continue
        else:
            issues.append(
                _issue(
                    "invalid_sse_line",
                    "unrecognized SSE line",
                    path="response_body",
                )
            )
    flush()
    return events


def _event_list(body: Any, issues: list[AuditIssue]) -> list[Any]:
    if isinstance(body, list):
        return body
    if isinstance(body, str):
        return _decode_sse_text(body, issues)
    if isinstance(body, dict) and (
        body.get("type") == "error" or body.get("error") is not None
    ):
        return [{"event": "error", "data": body}]
    if isinstance(body, dict) and isinstance(body.get("events"), list):
        return body["events"]
    issues.append(
        _issue(
            "invalid_stream_body",
            "stream response body must be an event array or SSE text",
            path="response_body",
        )
    )
    return []


def _normalized_events(
    body: Any, issues: list[AuditIssue]
) -> tuple[list[dict[str, Any]], bool]:
    issue_count = len(issues)
    raw_events = _event_list(body, issues)
    events: list[dict[str, Any]] = []
    sequence_valid = not any(
        issue.severity == Severity.ERROR for issue in issues[issue_count:]
    )
    previous_seq: int | None = None
    previous_wire: str | None = None
    saw_sequence = False
    saw_unsequenced = False

    for raw_index, envelope in enumerate(raw_events):
        path = f"response_body[{raw_index}]"
        if not isinstance(envelope, dict):
            issues.append(
                _issue("invalid_sse_event", "SSE event must be an object", path=path)
            )
            sequence_valid = False
            continue
        data = envelope.get("data", envelope)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                issues.append(
                    _issue("invalid_sse_data", "SSE data is not valid JSON", path=path)
                )
                sequence_valid = False
                continue
        if not isinstance(data, dict):
            issues.append(
                _issue("invalid_sse_data", "SSE data must be an object", path=path)
            )
            sequence_valid = False
            continue
        event_name = envelope.get("event") or data.get("type")
        if not isinstance(event_name, str) or not event_name:
            issues.append(
                _issue("invalid_sse_event", "SSE event has no type", path=path)
            )
            sequence_valid = False
            continue
        if isinstance(data.get("type"), str) and data["type"] != event_name:
            issues.append(
                _issue(
                    "sse_event_type_mismatch",
                    f"event field {event_name!r} differs from data type {data['type']!r}",
                    path=path,
                )
            )
            sequence_valid = False

        seq = envelope.get("seq")
        wire = _compact_json({"event": event_name, "data": data})
        if seq is None:
            saw_unsequenced = True
        elif isinstance(seq, int) and not isinstance(seq, bool):
            saw_sequence = True
            if previous_seq is not None:
                if seq == previous_seq:
                    if wire == previous_wire:
                        continue
                    issues.append(
                        _issue(
                            "conflicting_sse_sequence",
                            f"sequence {seq} is reused with different content",
                            path=path,
                        )
                    )
                    sequence_valid = False
                elif seq != previous_seq + 1:
                    issues.append(
                        _issue(
                            "sse_sequence_gap",
                            f"expected sequence {previous_seq + 1}, got {seq}",
                            path=path,
                        )
                    )
                    sequence_valid = False
            previous_seq = seq
            previous_wire = wire
        else:
            issues.append(
                _issue("invalid_sse_sequence", "SSE seq must be an integer", path=path)
            )
            sequence_valid = False
        events.append({"event": event_name, "data": data, "path": path})

    if saw_sequence and saw_unsequenced:
        issues.append(
            _issue(
                "mixed_sse_sequence",
                "only some SSE events carry sequence numbers",
                path="response_body",
            )
        )
        sequence_valid = False
    return events, sequence_valid


@dataclass
class _StreamBlock:
    value: dict[str, Any]
    path: str
    json_parts: list[str] = field(default_factory=list)
    closed: bool = False


def _parse_stream_response(
    body: Any,
    context: _ParseContext,
) -> tuple[list[Message], str, bool]:
    events, sequence_valid = _normalized_events(body, context.issues)
    blocks: dict[int, _StreamBlock] = {}
    block_order: list[int] = []
    initial_blocks: list[dict[str, Any]] = []
    saw_start = False
    saw_stop = False
    saw_error = False
    state_valid = sequence_valid
    model = ""

    for envelope in events:
        event = envelope["event"]
        data = envelope["data"]
        path = envelope["path"]
        if saw_stop:
            context.issues.append(
                _issue(
                    "event_after_message_stop",
                    f"event {event!r} appears after message_stop",
                    path=path,
                )
            )
            state_valid = False
            continue

        if event == "message_start":
            if saw_start:
                context.issues.append(
                    _issue(
                        "duplicate_message_start", "duplicate message_start", path=path
                    )
                )
                state_valid = False
                continue
            saw_start = True
            message = data.get("message", {})
            if isinstance(message, dict):
                if message.get("role") != "assistant":
                    context.issues.append(
                        _issue(
                            "invalid_message_start",
                            "message_start role must be assistant",
                            path=f"{path}.message.role",
                        )
                    )
                    state_valid = False
                model = _text(message.get("model"))
                content = message.get("content", [])
                if isinstance(content, list):
                    for content_index, block in enumerate(content):
                        if isinstance(block, dict):
                            initial_blocks.append(block)
                        else:
                            context.issues.append(
                                _issue(
                                    "invalid_message_start",
                                    "message_start content blocks must be objects",
                                    path=f"{path}.message.content[{content_index}]",
                                )
                            )
                            state_valid = False
                elif content is not None and content != "":
                    context.issues.append(
                        _issue(
                            "invalid_message_start",
                            "message_start content must be an array",
                            path=path,
                        )
                    )
                    state_valid = False
            else:
                context.issues.append(
                    _issue(
                        "invalid_message_start", "message must be an object", path=path
                    )
                )
                state_valid = False
        elif event == "content_block_start":
            if not saw_start:
                context.issues.append(
                    _issue(
                        "content_before_message_start",
                        "content block starts before message_start",
                        path=path,
                    )
                )
                state_valid = False
            index = data.get("index")
            block = data.get("content_block")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not isinstance(block, dict)
            ):
                context.issues.append(
                    _issue(
                        "invalid_content_block_start",
                        "content_block_start requires integer index and object block",
                        path=path,
                    )
                )
                state_valid = False
                continue
            if index in blocks:
                context.issues.append(
                    _issue(
                        "duplicate_content_block_start",
                        f"content block {index} started more than once",
                        path=path,
                    )
                )
                state_valid = False
                continue
            blocks[index] = _StreamBlock(value=dict(block), path=path)
            block_order.append(index)
        elif event == "content_block_delta":
            if not saw_start:
                context.issues.append(
                    _issue(
                        "content_before_message_start",
                        "content block delta appears before message_start",
                        path=path,
                    )
                )
                state_valid = False
            index = data.get("index")
            delta = data.get("delta")
            current = blocks.get(index) if isinstance(index, int) else None
            if current is None or current.closed or not isinstance(delta, dict):
                context.issues.append(
                    _issue(
                        "orphan_content_block_delta",
                        f"delta has no open content block at index {index!r}",
                        path=path,
                    )
                )
                state_valid = False
                continue
            delta_type = delta.get("type")
            if delta_type == "text_delta" and isinstance(delta.get("text"), str):
                current.value["text"] = _text(current.value.get("text")) + delta["text"]
            elif delta_type == "thinking_delta" and isinstance(
                delta.get("thinking"), str
            ):
                current.value["thinking"] = (
                    _text(current.value.get("thinking")) + delta["thinking"]
                )
            elif delta_type == "input_json_delta" and isinstance(
                delta.get("partial_json"), str
            ):
                current.json_parts.append(delta["partial_json"])
            elif delta_type == "signature_delta":
                signature = delta.get("signature")
                if isinstance(signature, str):
                    current.value["signature"] = (
                        _text(current.value.get("signature")) + signature
                    )
                else:
                    context.issues.append(
                        _issue(
                            "invalid_signature_delta",
                            "signature delta requires string signature",
                            path=path,
                        )
                    )
                    state_valid = False
            elif delta_type == "citations_delta":
                context.issues.append(
                    _issue(
                        "citation_metadata_omitted",
                        "citation metadata is not part of the V1 message schema",
                        path=path,
                        severity=Severity.WARNING,
                    )
                )
            else:
                context.issues.append(
                    _issue(
                        "unsupported_content_block_delta",
                        f"unsupported content delta type: {delta_type!r}",
                        path=path,
                    )
                )
                state_valid = False
        elif event == "content_block_stop":
            if not saw_start:
                context.issues.append(
                    _issue(
                        "content_before_message_start",
                        "content block stops before message_start",
                        path=path,
                    )
                )
                state_valid = False
            index = data.get("index")
            current = blocks.get(index) if isinstance(index, int) else None
            if current is None or current.closed:
                context.issues.append(
                    _issue(
                        "orphan_content_block_stop",
                        f"stop has no open content block at index {index!r}",
                        path=path,
                    )
                )
                state_valid = False
                continue
            current.closed = True
        elif event == "message_delta":
            if not saw_start:
                context.issues.append(
                    _issue(
                        "message_delta_before_start",
                        "message_delta appears before message_start",
                        path=path,
                    )
                )
                state_valid = False
            continue
        elif event == "message_stop":
            if not saw_start:
                context.issues.append(
                    _issue(
                        "message_stop_before_start",
                        "message_stop appears before message_start",
                        path=path,
                    )
                )
                state_valid = False
            saw_stop = True
        elif event == "ping":
            continue
        elif event == "error":
            error = data.get("error", {})
            context.issues.append(
                _issue(
                    "anthropic_api_error",
                    _stable_api_error_detail(error),
                    path=path,
                )
            )
            saw_error = True
        else:
            context.issues.append(
                _issue(
                    "unknown_sse_event",
                    f"unsupported SSE event: {event!r}",
                    path=path,
                )
            )
            state_valid = False

    if not saw_start and not saw_error:
        context.issues.append(
            _issue(
                "missing_message_start",
                "stream has no message_start",
                path="response_body",
            )
        )
        state_valid = False
    if not saw_stop and not saw_error:
        context.issues.append(
            _issue(
                "missing_message_stop",
                "stream ended before message_stop",
                path="response_body",
            )
        )
        state_valid = False
    for index in block_order:
        current = blocks[index]
        if not current.closed:
            context.issues.append(
                _issue(
                    "unclosed_content_block",
                    f"content block {index} was not closed",
                    path=current.path,
                )
            )
            state_valid = False
        if current.json_parts:
            raw_json = "".join(current.json_parts)
            if raw_json:
                try:
                    current.value["input"] = json.loads(raw_json)
                except json.JSONDecodeError:
                    current.value["input"] = raw_json
                    context.issues.append(
                        _issue(
                            "invalid_partial_json",
                            f"content block {index} contains incomplete tool JSON",
                            path=current.path,
                        )
                    )
                    state_valid = False

    decoded_blocks = initial_blocks + [blocks[index].value for index in block_order]
    if saw_error and not decoded_blocks:
        return [], model, False
    response = [
        _assistant_from_blocks(
            decoded_blocks,
            context=context,
            origin="response",
            path="response_body.content",
        )
    ]
    wire_complete = saw_start and saw_stop and state_valid and not saw_error
    if saw_error:
        outcome = "api_error"
    elif not saw_stop or not state_valid:
        outcome = "truncated"
    else:
        outcome = "success"
    return response, model, wire_complete if outcome == "success" else False


def _parse_nonstream_response(
    body: Any,
    context: _ParseContext,
) -> tuple[list[Message], str, bool, str]:
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            context.issues.append(
                _issue(
                    "invalid_response_json",
                    "non-stream response is not valid JSON",
                    path="response_body",
                )
            )
            return [], "", False, "capture_invalid"
    if not isinstance(body, dict):
        context.issues.append(
            _issue(
                "invalid_response_body",
                "non-stream response must be an object",
                path="response_body",
            )
        )
        return [], "", False, "capture_invalid"
    if body.get("type") == "error" or "error" in body:
        error = body.get("error", {})
        context.issues.append(
            _issue(
                "anthropic_api_error",
                _stable_api_error_detail(error),
                path="response_body",
            )
        )
        return [], "", False, "api_error"
    if body.get("type") != "message" or body.get("role") != "assistant":
        context.issues.append(
            _issue(
                "invalid_message_response",
                "response is not an Anthropic assistant message",
                path="response_body",
            )
        )
        return [], _text(body.get("model")), False, "capture_invalid"
    response = [
        _assistant_from_blocks(
            body.get("content", []),
            context=context,
            origin="response",
            path="response_body.content",
        )
    ]
    return response, _text(body.get("model")), True, "success"


def _request_metadata(request: dict[str, Any]) -> dict[str, Any]:
    raw = request.get("metadata")
    metadata = dict(raw) if isinstance(raw, dict) else {}
    encoded = metadata.get("user_id")
    if isinstance(encoded, str):
        try:
            decoded = json.loads(encoded)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            # Explicit metadata fields take precedence over the user_id envelope.
            metadata = {**decoded, **metadata}
    return metadata


def _snapshot_base(
    capture: dict[str, Any],
    *,
    source_path: str,
    source_sha256: str,
    source_partition: str,
    operation: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    headers = capture.get("request_headers")
    if not isinstance(headers, dict):
        headers = {}
    metadata = _request_metadata(request)
    session_id = (
        _text(metadata.get("session_id"))
        or _text(capture.get("session_id"))
        or _text(headers.get("x-claude-code-session-id"))
    )
    thread_id = (
        _text(metadata.get("thread_id"))
        or _text(capture.get("thread_id"))
        or _text(headers.get("x-claude-code-session-id"))
        or session_id
    )
    marker = (
        _text(metadata.get("subagent_marker"))
        or _text(metadata.get("thread_source"))
        or _text(capture.get("subagent_marker"))
    )
    if marker in {"main", "user"}:
        marker = ""
    explicit_harness = _text(capture.get("harness")) or _text(metadata.get("harness"))
    harness = explicit_harness or (
        "claude-code" if _text(headers.get("x-claude-code-session-id")) else "unknown"
    )
    return {
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_partition": source_partition,
        "session_id": session_id,
        "thread_id": thread_id,
        "turn_id": _text(metadata.get("turn_id")) or _text(capture.get("turn_id")),
        "parent_thread_id": _text(metadata.get("parent_thread_id"))
        or _text(capture.get("parent_thread_id")),
        "parent_turn_id": _text(metadata.get("parent_turn_id"))
        or _text(capture.get("parent_turn_id")),
        "forked_from_thread_id": _text(metadata.get("forked_from_thread_id"))
        or _text(capture.get("forked_from_thread_id")),
        "subagent_marker": marker,
        "provider": "anthropic",
        "operation": operation,
        "captured_at": _text(capture.get("captured_at")),
        "request_id": _text(capture.get("request_id")),
        "model": _text(request.get("model")),
        "harness": harness,
        "instructions": "",
        "termination": "",
    }


def _has_error(issues: Iterable[AuditIssue]) -> bool:
    return any(issue.severity == Severity.ERROR for issue in issues)


def parse_anthropic_capture(
    capture: dict[str, Any],
    *,
    source_path: str,
    source_sha256: str,
    source_partition: str,
) -> Snapshot:
    """Parse one freerouter Anthropic capture.

    ``/v1/messages/count_tokens`` is returned as a successful
    ``operation='count_tokens'`` snapshot when its response is valid.  The
    pipeline can therefore exclude it without manufacturing a trajectory.
    """

    if not isinstance(capture, dict):
        raise AnthropicCaptureError("capture must be an object")
    path = _text(capture.get("path"))
    endpoint = path.split("?", 1)[0].rstrip("/")
    if endpoint in {"/v1/messages/count_tokens", "/messages/count_tokens"}:
        operation = "count_tokens"
    elif endpoint in {"/v1/messages", "/messages"}:
        operation = "messages"
    else:
        raise AnthropicCaptureError(f"unsupported Anthropic endpoint: {path!r}")

    request = capture.get("request_body")
    request_issue: AuditIssue | None = None
    if not isinstance(request, dict):
        request_issue = _issue(
            "invalid_request_body",
            "request_body must be an object",
            path="request_body",
        )
        request = {}
    base = _snapshot_base(
        capture,
        source_path=source_path,
        source_sha256=source_sha256,
        source_partition=source_partition,
        operation=operation,
        request=request,
    )
    status = capture.get("status_code")

    if operation == "count_tokens":
        issues = [request_issue] if request_issue else []
        body = capture.get("response_body")
        if not isinstance(status, int) or isinstance(status, bool) or status <= 0:
            outcome = "transport_error"
            issues.append(
                _issue("missing_http_status", "capture has no valid HTTP status code")
            )
            wire_complete = False
        elif not 200 <= status < 300:
            outcome = "api_error"
            issues.append(
                _issue("anthropic_api_error", f"Anthropic returned HTTP {status}")
            )
            wire_complete = False
        elif (
            isinstance(body, dict)
            and isinstance(body.get("input_tokens"), int)
            and not isinstance(body.get("input_tokens"), bool)
        ):
            outcome = "capture_invalid" if _has_error(issues) else "success"
            wire_complete = not _has_error(issues)
        else:
            outcome = "capture_invalid"
            wire_complete = False
            issues.append(
                _issue(
                    "invalid_count_tokens_response",
                    "count_tokens response requires integer input_tokens",
                    path="response_body",
                )
            )
        return Snapshot(
            **base,
            outcome=outcome,
            history=[],
            response=[],
            tools=[],
            server_tool_calls=[],
            wire_complete=wire_complete,
            issues=issues,
        )

    context = _ParseContext()
    if request_issue:
        context.issues.append(request_issue)
    if not base["session_id"]:
        context.issues.append(
            _issue("missing_session_id", "capture has no session id", path="session_id")
        )
    if not base["thread_id"]:
        context.issues.append(
            _issue("missing_thread_id", "capture has no thread id", path="request_body")
        )
    history = _parse_history(request, context)
    tools = _parse_tools(request, context.issues)

    if not isinstance(status, int) or isinstance(status, bool) or status <= 0:
        context.issues.append(
            _issue("missing_http_status", "capture has no valid HTTP status code")
        )
        response: list[Message] = []
        response_model = ""
        outcome = "transport_error"
        wire_complete = False
    elif not 200 <= status < 300:
        body = capture.get("response_body")
        error: Any = None
        if isinstance(body, dict):
            error = body.get("error")
        context.issues.append(
            _issue(
                "anthropic_api_error",
                _stable_api_error_detail(error, status=status),
                path="response_body",
            )
        )
        response = []
        response_model = ""
        outcome = "api_error"
        wire_complete = False
    elif (
        capture.get("is_stream") is True
        or request.get("stream") is True
        or isinstance(capture.get("response_body"), list)
        or (
            isinstance(capture.get("response_body"), str)
            and capture["response_body"].lstrip().startswith(("event:", "data:"))
        )
    ):
        before = len(context.issues)
        response, response_model, wire_complete = _parse_stream_response(
            capture.get("response_body"), context
        )
        new_codes = {issue.code for issue in context.issues[before:]}
        if "anthropic_api_error" in new_codes:
            outcome = "api_error"
        elif not wire_complete:
            outcome = "truncated"
        elif _has_error(context.issues):
            outcome = "capture_invalid"
        else:
            outcome = "success"
    else:
        response, response_model, wire_complete, outcome = _parse_nonstream_response(
            capture.get("response_body"), context
        )

    context.resolve_pending_results()
    if outcome == "success" and _has_error(context.issues):
        outcome = "capture_invalid"

    if response_model and not base["model"]:
        base["model"] = response_model
    return Snapshot(
        **base,
        outcome=outcome,
        history=history,
        response=response,
        tools=tools,
        server_tool_calls=context.server_calls,
        wire_complete=wire_complete,
        issues=context.issues,
    )


normalize_anthropic_capture = parse_anthropic_capture


__all__ = [
    "AnthropicCaptureError",
    "normalize_anthropic_capture",
    "parse_anthropic_capture",
]
