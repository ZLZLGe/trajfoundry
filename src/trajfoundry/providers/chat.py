"""Normalize OpenAI Chat Completions captures into provider-neutral snapshots.

TokenPlan stores streamed Chat Completions as a normalized final
``chat.completion`` object.  Consequently this adapter intentionally parses a
single final object for both streaming and non-streaming requests; raw SSE or
chunk arrays are structural errors rather than alternate transcript sources.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from trajfoundry.models import (
    AuditIssue,
    FunctionCall,
    Message,
    Severity,
    Snapshot,
    ToolCall,
    ToolDefinition,
)

_STAGE = "provider.chat"
_ENDPOINTS = {"/v1/chat/completions", "/chat/completions"}
_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
_KNOWN_NON_TEXT_PART_TYPES = {
    "audio",
    "file",
    "image",
    "image_url",
    "input_audio",
    "input_file",
    "input_image",
    "refusal",
}


class ChatCompletionsAdapterError(ValueError):
    """Raised when a value is not a Chat Completions capture at all."""


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _strict_json_loads(value: str) -> Any:
    def reject_nonstandard_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant: {constant}")

    return json.loads(value, parse_constant=reject_nonstandard_constant)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _issue(
    issues: list[AuditIssue],
    code: str,
    path: str,
    detail: str,
    *,
    severity: Severity = Severity.ERROR,
) -> None:
    issues.append(
        AuditIssue(
            code=code,
            stage=_STAGE,
            severity=severity,
            path=path,
            detail=detail,
        )
    )


def _render_content(content: Any, *, path: str, issues: list[AuditIssue]) -> str:
    """Render Chat content without dropping ordered multimodal evidence."""

    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        _issue(
            issues,
            "invalid_message_content",
            path,
            f"expected string, array, or null; got {type(content).__name__}",
        )
        return _compact_json(content)

    rendered: list[str] = []
    for index, part in enumerate(content):
        part_path = f"{path}[{index}]"
        if not isinstance(part, Mapping):
            _issue(
                issues,
                "invalid_content_part",
                part_path,
                f"expected object; got {type(part).__name__}",
            )
            rendered.append(_compact_json(part))
            continue

        part_type = part.get("type")
        if part_type in _TEXT_PART_TYPES:
            value = part.get("text")
            if not isinstance(value, str):
                _issue(
                    issues,
                    "invalid_text_part",
                    part_path,
                    f"text is {type(value).__name__}, not string",
                )
                rendered.append(_compact_json(value))
            else:
                rendered.append(value)
        elif part_type in _KNOWN_NON_TEXT_PART_TYPES:
            rendered.append(_compact_json(dict(part)))
        else:
            _issue(
                issues,
                "unknown_content_part",
                part_path,
                f"unsupported content part type {part_type!r}; retained as JSON",
            )
            rendered.append(_compact_json(dict(part)))
    return "".join(rendered)


def _parse_arguments(
    value: Any, *, present: bool, path: str, issues: list[AuditIssue]
) -> Any:
    if not present:
        _issue(issues, "missing_tool_arguments", path, "tool arguments are absent")
        return {"raw": ""}
    if not isinstance(value, str):
        return value
    try:
        return _strict_json_loads(value)
    except (TypeError, ValueError):
        return {"raw": value}


def _reasoning_content(
    message: Mapping[str, Any], *, path: str, issues: list[AuditIssue]
) -> str:
    """Prefer ``reasoning_content`` and fall back to legacy ``reasoning``."""

    if message.get("reasoning_content") is not None:
        value = message["reasoning_content"]
        if isinstance(value, str):
            return value
        _issue(
            issues,
            "invalid_reasoning_content",
            f"{path}.reasoning_content",
            f"expected string or null; got {type(value).__name__}",
        )
        return _compact_json(value)

    value = message.get("reasoning")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    _issue(
        issues,
        "invalid_reasoning",
        f"{path}.reasoning",
        f"expected string or null; got {type(value).__name__}",
    )
    return _compact_json(value)


def _decoded_turn_metadata(
    value: Any, *, path: str, issues: list[AuditIssue]
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        _issue(
            issues,
            "invalid_turn_metadata",
            path,
            f"expected JSON string or object; got {type(value).__name__}",
        )
        return None
    try:
        decoded = _strict_json_loads(value)
    except (TypeError, ValueError):
        _issue(issues, "invalid_turn_metadata", path, "value is not valid JSON")
        return None
    if not isinstance(decoded, Mapping):
        _issue(issues, "invalid_turn_metadata", path, "JSON value is not an object")
        return None
    return dict(decoded)


def _identity_metadata(
    capture: Mapping[str, Any],
    request: Mapping[str, Any],
    issues: list[AuditIssue],
) -> tuple[dict[str, str], bool]:
    """Select request identity before transport-level fallbacks."""

    direct_sources: list[tuple[str, Mapping[str, Any]]] = []
    encoded_sources: list[tuple[str, Mapping[str, Any]]] = []
    saw_codex_metadata = False
    for metadata_name in ("client_metadata", "metadata"):
        value = request.get(metadata_name)
        if not isinstance(value, Mapping):
            continue
        direct_sources.append((f"request_body.{metadata_name}", value))
        encoded = _decoded_turn_metadata(
            value.get("x-codex-turn-metadata"),
            path=f"request_body.{metadata_name}.x-codex-turn-metadata",
            issues=issues,
        )
        if encoded is not None:
            encoded_sources.append(
                (f"request_body.{metadata_name}.x-codex-turn-metadata", encoded)
            )
            saw_codex_metadata = True

    metadata = request.get("metadata")
    if isinstance(metadata, Mapping):
        encoded_user = metadata.get("user_id")
        if isinstance(encoded_user, str):
            try:
                user_identity = _strict_json_loads(encoded_user)
            except (TypeError, ValueError):
                user_identity = None
            if isinstance(user_identity, Mapping):
                encoded_sources.append(("request_body.metadata.user_id", user_identity))

    raw_headers = capture.get("request_headers")
    headers: dict[str, Any] = {}
    if isinstance(raw_headers, Mapping):
        headers = {
            str(key).lower(): value
            for key, value in raw_headers.items()
            if str(key).lower()
            in {
                "x-codex-turn-metadata",
                "x-codex-parent-thread-id",
                "x-openai-subagent",
            }
        }
    header_metadata = _decoded_turn_metadata(
        headers.get("x-codex-turn-metadata"),
        path="request_headers.x-codex-turn-metadata",
        issues=issues,
    )
    if header_metadata is not None:
        encoded_sources.append(
            ("request_headers.x-codex-turn-metadata", header_metadata)
        )
        saw_codex_metadata = True
    header_sources = [("request_headers", headers)] if headers else []
    if headers:
        saw_codex_metadata = True

    sources = [*direct_sources, *encoded_sources, *header_sources]
    aliases: dict[str, tuple[str, ...]] = {
        "session_id": ("session_id",),
        "thread_id": ("thread_id",),
        "turn_id": ("turn_id",),
        "parent_thread_id": (
            "parent_thread_id",
            "parentThreadId",
            "x-codex-parent-thread-id",
        ),
        "parent_turn_id": ("parent_turn_id", "parentTurnId"),
        "forked_from_thread_id": (
            "forked_from_thread_id",
            "forkedFromThreadId",
        ),
    }

    def values_for(keys: tuple[str, ...]) -> list[tuple[str, str]]:
        values: list[tuple[str, str]] = []
        for source_name, source in sources:
            for key in keys:
                value = source.get(key)
                if isinstance(value, str) and value:
                    values.append((f"{source_name}.{key}", value))
                    break
        return values

    identity: dict[str, str] = {}
    for identity_field, field_aliases in aliases.items():
        candidates = values_for(field_aliases)
        if candidates:
            identity[identity_field] = candidates[0][1]
        if len({value for _, value in candidates}) > 1:
            detail = "; ".join(f"{source}={value!r}" for source, value in candidates)
            _issue(
                issues,
                "metadata_conflict",
                identity_field,
                f"conflicting {identity_field}: {detail}",
            )

    explicit_markers = values_for(("subagent_marker",))
    transport_markers = values_for(("x-openai-subagent",))
    subagent_kinds = values_for(("subagent_kind",))
    selected_markers = explicit_markers or transport_markers or subagent_kinds
    if selected_markers:
        identity["subagent_marker"] = selected_markers[0][1]
    harness_candidates = values_for(("harness",))
    if harness_candidates:
        identity["harness"] = harness_candidates[0][1]
    return identity, saw_codex_metadata


@dataclass
class _PendingToolResult:
    message: Message
    identifier: str
    path: str


@dataclass
class _ParseContext:
    issues: list[AuditIssue] = field(default_factory=list)
    tool_names: dict[str, str] = field(default_factory=dict)
    pending_results: list[_PendingToolResult] = field(default_factory=list)
    missing_id_counts: dict[str, int] = field(default_factory=dict)

    def missing_id(self, kind: str) -> str:
        count = self.missing_id_counts.get(kind, 0) + 1
        self.missing_id_counts[kind] = count
        return f"missing-{kind}:{count}"

    def register_call(self, identifier: str, name: str, *, path: str) -> None:
        if identifier in self.tool_names:
            _issue(
                self.issues,
                "duplicate_tool_call_id",
                path,
                f"duplicate client tool call id: {identifier}",
            )
            return
        self.tool_names[identifier] = name

    def add_result(self, message: Message, identifier: str, *, path: str) -> None:
        self.pending_results.append(
            _PendingToolResult(message=message, identifier=identifier, path=path)
        )

    def resolve_results(self) -> None:
        seen: set[str] = set()
        for pending in self.pending_results:
            if pending.identifier in seen:
                _issue(
                    self.issues,
                    "duplicate_tool_result",
                    pending.path,
                    f"multiple client tool results use id: {pending.identifier}",
                )
            seen.add(pending.identifier)

            name = self.tool_names.get(pending.identifier)
            if name is None:
                _issue(
                    self.issues,
                    "orphan_tool_result",
                    pending.path,
                    f"no client tool call found for result id: {pending.identifier!r}",
                )
                name = ""
            pending.message.name = name


def _parse_tool_calls(
    value: Any,
    *,
    path: str,
    context: _ParseContext,
) -> list[ToolCall] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        _issue(
            context.issues,
            "invalid_tool_calls",
            path,
            f"assistant tool_calls must be an array; got {type(value).__name__}",
        )
        return None

    calls: list[ToolCall] = []
    for index, raw_call in enumerate(value):
        call_path = f"{path}[{index}]"
        if not isinstance(raw_call, Mapping):
            _issue(
                context.issues,
                "invalid_tool_call",
                call_path,
                f"tool call must be an object; got {type(raw_call).__name__}",
            )
            continue
        if raw_call.get("type") not in {None, "function"}:
            _issue(
                context.issues,
                "unsupported_tool_call",
                call_path,
                f"unsupported Chat tool call type: {raw_call.get('type')!r}",
            )
            continue
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            _issue(
                context.issues,
                "invalid_tool_call",
                f"{call_path}.function",
                "tool call function must be an object",
            )
            function = {}

        identifier = raw_call.get("id")
        if not isinstance(identifier, str) or not identifier:
            _issue(
                context.issues,
                "invalid_tool_call",
                f"{call_path}.id",
                "tool call requires a non-empty id",
            )
            identifier = context.missing_id("client-tool-call")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            _issue(
                context.issues,
                "invalid_tool_call",
                f"{call_path}.function.name",
                "tool call function requires a non-empty name",
            )
            name = _text(name)
        calls.append(
            ToolCall(
                id=identifier,
                function=FunctionCall(
                    name=name,
                    arguments=_parse_arguments(
                        function.get("arguments"),
                        present="arguments" in function,
                        path=f"{call_path}.function.arguments",
                        issues=context.issues,
                    ),
                ),
            )
        )
        context.register_call(identifier, name, path=call_path)
    return calls or None


def _assistant_message(
    raw_message: Mapping[str, Any],
    *,
    path: str,
    context: _ParseContext,
) -> Message:
    return Message(
        role="assistant",
        content=_render_content(
            raw_message.get("content"),
            path=f"{path}.content",
            issues=context.issues,
        ),
        reasoning_content=_reasoning_content(
            raw_message,
            path=path,
            issues=context.issues,
        ),
        tool_calls=_parse_tool_calls(
            raw_message.get("tool_calls"),
            path=f"{path}.tool_calls",
            context=context,
        ),
    )


def _parse_history(request: Mapping[str, Any], context: _ParseContext) -> list[Message]:
    raw_messages = request.get("messages")
    if not isinstance(raw_messages, list):
        _issue(
            context.issues,
            "invalid_messages",
            "request_body.messages",
            "request messages must be an array",
        )
        return []

    history: list[Message] = []
    for index, raw_message in enumerate(raw_messages):
        path = f"request_body.messages[{index}]"
        if not isinstance(raw_message, Mapping):
            _issue(
                context.issues,
                "invalid_message",
                path,
                f"message must be an object; got {type(raw_message).__name__}",
            )
            continue

        role = raw_message.get("role")
        if role == "assistant":
            history.append(_assistant_message(raw_message, path=path, context=context))
        elif role in {"system", "developer", "user"}:
            history.append(
                Message(
                    role=role,
                    content=_render_content(
                        raw_message.get("content"),
                        path=f"{path}.content",
                        issues=context.issues,
                    ),
                )
            )
        elif role == "tool":
            identifier = raw_message.get("tool_call_id")
            if not isinstance(identifier, str) or not identifier:
                _issue(
                    context.issues,
                    "invalid_tool_result",
                    f"{path}.tool_call_id",
                    "tool message requires a non-empty tool_call_id",
                )
                identifier = context.missing_id("client-tool-result")
            message = Message(
                role="tool",
                content=_render_content(
                    raw_message.get("content"),
                    path=f"{path}.content",
                    issues=context.issues,
                ),
                tool_call_id=identifier,
                name="",
            )
            history.append(message)
            context.add_result(message, identifier, path=path)
        else:
            _issue(
                context.issues,
                "unsupported_message_role",
                path,
                f"unsupported Chat message role: {role!r}",
            )
    return history


def _parse_tools(
    request: Mapping[str, Any], issues: list[AuditIssue]
) -> list[ToolDefinition]:
    raw_tools = request.get("tools")
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        _issue(
            issues,
            "invalid_tools",
            "request_body.tools",
            "request tools must be an array or null",
        )
        return []

    tools: list[ToolDefinition] = []
    for index, raw_tool in enumerate(raw_tools):
        path = f"request_body.tools[{index}]"
        if not isinstance(raw_tool, Mapping):
            _issue(
                issues,
                "invalid_tool_definition",
                path,
                f"tool must be an object; got {type(raw_tool).__name__}",
            )
            continue
        if raw_tool.get("type") != "function":
            _issue(
                issues,
                "unsupported_tool_definition",
                path,
                f"unsupported Chat tool type: {raw_tool.get('type')!r}",
            )
            continue
        function = raw_tool.get("function")
        if not isinstance(function, Mapping):
            _issue(
                issues,
                "invalid_tool_definition",
                f"{path}.function",
                "tool function must be an object",
            )
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            _issue(
                issues,
                "invalid_tool_definition",
                f"{path}.function.name",
                "tool function requires a non-empty name",
            )
            continue
        description = function.get("description")
        if description is None:
            description = ""
        elif not isinstance(description, str):
            _issue(
                issues,
                "invalid_tool_definition",
                f"{path}.function.description",
                "tool function description must be a string or null",
            )
            description = _compact_json(description)
        parameters = function.get("parameters", {})
        if not isinstance(parameters, Mapping):
            _issue(
                issues,
                "invalid_tool_schema",
                f"{path}.function.parameters",
                "tool function parameters must be an object",
            )
            parameters = {}
        tools.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters=dict(parameters),
            )
        )
    return tools


def _stable_api_error_detail(value: Any, *, status: int | None = None) -> str:
    parts: list[str] = []
    if status is not None:
        parts.append(f"HTTP {status}")
    error = value.get("error") if isinstance(value, Mapping) else None
    if isinstance(error, Mapping):
        for key in ("type", "code"):
            atom = error.get(key)
            if isinstance(atom, (str, int)) and not isinstance(atom, bool):
                rendered = str(atom)
                if 0 < len(rendered) <= 80 and all(
                    character.isalnum() or character in "._-:" for character in rendered
                ):
                    parts.append(f"{key}={rendered}")
    suffix = f" ({'; '.join(parts)})" if parts else ""
    return f"OpenAI-compatible API error{suffix}"


def _decode_response_body(
    body: Any, issues: list[AuditIssue]
) -> Mapping[str, Any] | None:
    if isinstance(body, str):
        try:
            body = _strict_json_loads(body)
        except (TypeError, ValueError):
            _issue(
                issues,
                "invalid_response_json",
                "response_body",
                "Chat Completions response is not valid JSON",
            )
            return None
    if not isinstance(body, Mapping):
        _issue(
            issues,
            "invalid_response_body",
            "response_body",
            "Chat Completions response must be a final object",
        )
        return None
    return body


def _parse_response(
    body: Any,
    *,
    status_code: int,
    context: _ParseContext,
) -> tuple[list[Message], str, str, bool]:
    issue_start = len(context.issues)
    decoded = _decode_response_body(body, context.issues)
    if decoded is None:
        return [], "", "capture_invalid", False

    response_model = _text(decoded.get("model"))
    if decoded.get("error") is not None or decoded.get("object") == "error":
        _issue(
            context.issues,
            "openai_api_error",
            "response_body",
            _stable_api_error_detail(decoded, status=status_code),
        )
        return [], response_model, "api_error", False

    object_type = decoded.get("object")
    if object_type is not None and object_type != "chat.completion":
        _issue(
            context.issues,
            "invalid_chat_completion_object",
            "response_body.object",
            f"expected 'chat.completion'; got {object_type!r}",
        )

    choices = decoded.get("choices")
    if not isinstance(choices, list):
        _issue(
            context.issues,
            "invalid_chat_choices",
            "response_body.choices",
            "Chat Completions choices must be an array",
        )
        return [], response_model, "capture_invalid", False
    if len(choices) != 1:
        _issue(
            context.issues,
            "invalid_chat_choice_count",
            "response_body.choices",
            f"expected exactly one Chat Completions choice; got {len(choices)}",
        )
    if not choices:
        return [], response_model, "capture_invalid", False

    choice = choices[0]
    if not isinstance(choice, Mapping):
        _issue(
            context.issues,
            "invalid_chat_choice",
            "response_body.choices[0]",
            "Chat Completions choice must be an object",
        )
        return [], response_model, "capture_invalid", False
    finish_reason = choice.get("finish_reason")
    truncated = finish_reason in {"length", "content_filter"}
    if truncated:
        _issue(
            context.issues,
            "chat_completion_truncated",
            "response_body.choices[0].finish_reason",
            f"Chat Completions stopped with finish_reason={finish_reason!r}",
        )
    elif finish_reason not in {"stop", "tool_calls"}:
        _issue(
            context.issues,
            "invalid_chat_finish_reason",
            "response_body.choices[0].finish_reason",
            f"unsupported final Chat Completions finish_reason: {finish_reason!r}",
        )
    raw_message = choice.get("message")
    if not isinstance(raw_message, Mapping):
        _issue(
            context.issues,
            "invalid_chat_response_message",
            "response_body.choices[0].message",
            "Chat Completions choice requires a message object",
        )
        return [], response_model, "capture_invalid", False
    if raw_message.get("role") != "assistant":
        _issue(
            context.issues,
            "invalid_chat_response_role",
            "response_body.choices[0].message.role",
            "Chat Completions response message must have role 'assistant'",
        )

    response = [
        _assistant_message(
            raw_message,
            path="response_body.choices[0].message",
            context=context,
        )
    ]
    if finish_reason == "tool_calls" and not response[0].tool_calls:
        _issue(
            context.issues,
            "invalid_chat_finish_reason",
            "response_body.choices[0].finish_reason",
            "finish_reason='tool_calls' requires assistant tool_calls",
        )
    if truncated:
        return response, response_model, "truncated", False
    has_new_errors = any(
        issue.severity == Severity.ERROR for issue in context.issues[issue_start:]
    )
    outcome = "capture_invalid" if has_new_errors else "success"
    return response, response_model, outcome, not has_new_errors


def parse_chat_capture(
    capture: dict[str, Any],
    *,
    source_path: str,
    source_sha256: str,
    response_is_normalized_final: bool = False,
) -> Snapshot:
    """Parse one OpenAI-compatible ``/v1/chat/completions`` capture.

    ``response_is_normalized_final`` is accepted for source-adapter symmetry.
    Chat captures always require a final ChatCompletion object, including when
    the original client request asked for streaming.
    """

    if not isinstance(capture, dict):
        raise ChatCompletionsAdapterError("capture must be an object")
    endpoint = capture.get("path")
    if (
        not isinstance(endpoint, str)
        or endpoint.split("?", 1)[0].rstrip("/") not in _ENDPOINTS
    ):
        raise ChatCompletionsAdapterError(
            f"not a Chat Completions capture: {endpoint!r}"
        )

    # The marker is intentionally not used as an SSE switch.  It documents
    # that source adapters may explicitly identify a normalized final object.
    _ = response_is_normalized_final
    context = _ParseContext()
    raw_request = capture.get("request_body")
    if isinstance(raw_request, Mapping):
        request: Mapping[str, Any] = raw_request
    else:
        _issue(
            context.issues,
            "invalid_request_body",
            "request_body",
            "Chat Completions request_body must be an object",
        )
        request = {}

    history = _parse_history(request, context)
    tools = _parse_tools(request, context.issues)

    identity, saw_codex_metadata = _identity_metadata(capture, request, context.issues)

    def identity_value(field: str) -> str:
        selected = identity.get(field, "")
        fallback = _text(capture.get(field))
        if selected and fallback and selected != fallback:
            _issue(
                context.issues,
                "metadata_conflict",
                field,
                f"capture {field} conflicts with request metadata {field}",
            )
        return selected or fallback

    session_id = identity_value("session_id")
    turn_id = identity_value("turn_id")
    parent_thread_id = identity_value("parent_thread_id")
    parent_turn_id = identity_value("parent_turn_id")
    forked_from_thread_id = identity_value("forked_from_thread_id")
    subagent_marker = identity.get("subagent_marker", "") or _text(
        capture.get("subagent_marker")
    )
    if subagent_marker in {"main", "user"}:
        subagent_marker = ""
    explicit_thread_id = identity_value("thread_id")
    has_subagent_linkage = bool(
        subagent_marker or parent_thread_id or parent_turn_id or forked_from_thread_id
    )
    if explicit_thread_id:
        thread_id = explicit_thread_id
    elif has_subagent_linkage:
        thread_id = f"__isolated_subagent__:{source_path}"
        _issue(
            context.issues,
            "isolated_subagent_missing_thread_id",
            "request_body",
            "sub-agent linkage has no explicit thread id; capture was isolated",
            severity=Severity.WARNING,
        )
    else:
        thread_id = session_id
    if not session_id:
        _issue(
            context.issues,
            "missing_session_id",
            "session_id",
            "capture has no session id",
        )
    if not thread_id:
        _issue(
            context.issues,
            "missing_thread_id",
            "thread_id",
            "capture has no thread id",
        )

    status = capture.get("status_code")
    response: list[Message] = []
    response_model = ""
    if not isinstance(status, int) or isinstance(status, bool) or status <= 0:
        _issue(
            context.issues,
            "missing_http_status",
            "status_code",
            "capture has no valid HTTP status code",
        )
        outcome = "transport_error"
        wire_complete = False
    elif not 200 <= status < 300:
        _issue(
            context.issues,
            "openai_api_error",
            "response_body",
            _stable_api_error_detail(capture.get("response_body"), status=status),
        )
        outcome = "api_error"
        wire_complete = False
    else:
        response, response_model, outcome, wire_complete = _parse_response(
            capture.get("response_body"),
            status_code=status,
            context=context,
        )

    context.resolve_results()
    if outcome == "success" and any(
        issue.severity == Severity.ERROR for issue in context.issues
    ):
        outcome = "capture_invalid"
        wire_complete = False

    request_model = request.get("model")
    model = request_model if isinstance(request_model, str) else response_model
    explicit_harness = identity.get("harness", "") or _text(capture.get("harness"))
    harness = explicit_harness or (
        "codex"
        if saw_codex_metadata or isinstance(request.get("client_metadata"), Mapping)
        else "unknown"
    )
    return Snapshot(
        source_path=source_path,
        source_sha256=source_sha256,
        session_id=session_id,
        thread_id=thread_id,
        turn_id=turn_id,
        parent_thread_id=parent_thread_id,
        parent_turn_id=parent_turn_id,
        forked_from_thread_id=forked_from_thread_id,
        subagent_marker=subagent_marker,
        provider="openai",
        operation="chat_completions",
        outcome=outcome,
        captured_at=_text(capture.get("captured_at")),
        request_id=_text(capture.get("request_id")),
        model=model,
        harness=harness,
        instructions="",
        history=history,
        response=response,
        tools=tools,
        server_tool_calls=[],
        agent_messages=[],
        compaction_items=[],
        termination="",
        wire_complete=wire_complete,
        issues=context.issues,
    )


parse_chat_completions_capture = parse_chat_capture
normalize_chat_capture = parse_chat_capture
normalize_chat_completions_capture = parse_chat_capture


__all__ = [
    "ChatCompletionsAdapterError",
    "normalize_chat_capture",
    "normalize_chat_completions_capture",
    "parse_chat_capture",
    "parse_chat_completions_capture",
]
