"""Normalize OpenAI Responses captures into :class:`~trajfoundry.models.Snapshot`.

Freerouter stores streaming responses as an array of parsed SSE frames.  This
module deliberately consumes that representation instead of treating the last
frame as authoritative: some captured terminal payloads have already projected
custom tool calls into incomplete function calls, while ``output_item.done``
still contains the lossless item.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from typing import Any

from trajfoundry.audit_codes import RESPONSES_UNSUPPORTED_CALL_EVIDENCE
from trajfoundry.models import (
    AgentMessageEvidence,
    AuditIssue,
    CompactionRecord,
    FunctionCall,
    Message,
    ServerToolCall,
    Severity,
    Snapshot,
    ToolCall,
    ToolDefinition,
)
from trajfoundry.tool_names import is_spawn_tool_name, qualify_tool_name


class ResponsesAdapterError(ValueError):
    """Raised when a value is not a Responses capture at all."""


_CLIENT_CALL_TYPES = {
    "apply_patch_call",
    "computer_call",
    "custom_tool_call",
    "function_call",
    "local_shell_call",
}
_CLIENT_RESULT_TYPES = {
    "apply_patch_call_output",
    "computer_call_output",
    "custom_tool_call_output",
    "function_call_output",
    "local_shell_call_output",
}
_CLIENT_RESULT_TO_CALL = {
    "apply_patch_call_output": "apply_patch_call",
    "computer_call_output": "computer_call",
    "custom_tool_call_output": "custom_tool_call",
    "function_call_output": "function_call",
    "local_shell_call_output": "local_shell_call",
    "shell_call_output": "shell_call",
    "tool_search_output": "tool_search_call",
}
_CLIENT_CALL_TO_RESULT = {
    call_type: result_type for result_type, call_type in _CLIENT_RESULT_TO_CALL.items()
}
_CLIENT_BUILTIN_NAMES = {
    "apply_patch_call": "apply_patch",
    "computer_call": "computer",
    "local_shell_call": "local_shell",
    "shell_call": "shell",
    "tool_search_call": "tool_search",
}
_CONDITIONAL_CALL_TYPES = {"shell_call", "tool_search_call"}
_CONDITIONAL_RESULT_TYPES = {"shell_call_output", "tool_search_output"}
_TERMINAL_EVENTS = {
    "response.completed",
    "response.incomplete",
    "response.failed",
    "error",
}
_KNOWN_EVENT_PREFIXES = (
    "response.output_item.",
    "response.content_part.",
    "response.output_text.",
    "response.refusal.",
    "response.reasoning_summary_part.",
    "response.reasoning_summary_text.",
    "response.reasoning_text.",
    "response.function_call_arguments.",
    "response.custom_tool_call_input.",
    "response.web_search_call.",
    "response.file_search_call.",
    "response.code_interpreter_call.",
    "response.computer_call.",
    "response.apply_patch_call.",
    "response.image_generation_call.",
    "response.local_shell_call.",
    "response.mcp_call.",
    "response.shell_call.",
    "response.tool_search_call.",
    "response.audio.",
)
_LIFECYCLE_EVENTS = {
    "response.created",
    "response.queued",
    "response.in_progress",
    *_TERMINAL_EVENTS,
}
_KNOWN_NON_TEXT_CONTENT = {
    "input_audio",
    "input_file",
    "input_image",
    "computer_screenshot",
    "output_audio",
    "output_file",
    "output_image",
}
_SERVER_CALL_TYPES = {
    "code_interpreter_call",
    "file_search_call",
    "image_generation_call",
    "mcp_call",
    "web_search_call",
}
_SERVER_RESULT_TYPES = {
    "code_interpreter_call_output",
    "file_search_call_output",
    "image_generation_call_output",
    "mcp_call_output",
    "web_search_call_output",
}
_HOSTED_TOOL_TYPES = {
    "code_interpreter",
    "file_search",
    "image_generation",
    "mcp",
    "web_search",
}
_EXECUTION_CLIENT = "client"
_EXECUTION_SERVER = "server"


def _json(value: Any) -> str:
    """Return a stable, compact JSON representation."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _strict_json_loads(value: str) -> Any:
    """Parse an RFC-compatible JSON value without Python's NaN extensions."""

    def reject_nonstandard_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant: {constant}")

    return json.loads(value, parse_constant=reject_nonstandard_constant)


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
            stage="responses",
            severity=severity,
            path=path,
            detail=detail,
        )
    )


def _error_payload(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        error = value.get("error")
        if isinstance(error, Mapping):
            return error
        response = value.get("response")
        if isinstance(response, Mapping):
            nested = _error_payload(response)
            if nested is not None:
                return nested
        data = value.get("data")
        if isinstance(data, Mapping):
            nested = _error_payload(data)
            if nested is not None:
                return nested
        if value.get("type") == "error":
            return value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in reversed(value):
            payload = _error_payload(item)
            if payload is not None:
                return payload
    return None


def _stable_api_error_detail(value: Any, *, status: int | None = None) -> str:
    """Summarize a provider error without copying its free-form body."""

    parts: list[str] = []
    if status is not None:
        parts.append(f"HTTP {status}")
    payload = _error_payload(value)
    if payload is not None:
        for key in ("type", "code"):
            field = payload.get(key)
            if isinstance(field, (str, int)) and not isinstance(field, bool):
                atom = str(field)
                if 0 < len(atom) <= 80 and all(
                    character.isalnum() or character in "._-:" for character in atom
                ):
                    parts.append(f"{key}={atom}")
    suffix = f" ({'; '.join(parts)})" if parts else ""
    return f"OpenAI-compatible API error{suffix}"


def _as_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _json(value)


def _render_content(content: Any, *, path: str, issues: list[AuditIssue]) -> str:
    """Convert a Responses message content value without silently dropping parts."""

    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, Mapping):
        parts: Sequence[Any] = [content]
    elif isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        parts = content
    else:
        _issue(
            issues,
            "invalid_message_content",
            path,
            f"expected string, object, or array; got {type(content).__name__}",
        )
        return _as_string(content)

    rendered: list[str] = []
    for index, part in enumerate(parts):
        part_path = f"{path}[{index}]"
        if not isinstance(part, Mapping):
            _issue(
                issues,
                "invalid_content_part",
                part_path,
                f"expected object; got {type(part).__name__}",
            )
            rendered.append(_as_string(part))
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text", "summary_text"}:
            text = part.get("text", "")
            if not isinstance(text, str):
                _issue(
                    issues,
                    "invalid_text_part",
                    part_path,
                    f"text is {type(text).__name__}, not string",
                )
                text = _as_string(text)
            rendered.append(text)
        elif part_type == "refusal":
            refusal = part.get("refusal", "")
            if not isinstance(refusal, str):
                _issue(
                    issues,
                    "invalid_refusal_part",
                    part_path,
                    f"refusal is {type(refusal).__name__}, not string",
                )
                refusal = _as_string(refusal)
            rendered.append(refusal)
        elif part_type in _KNOWN_NON_TEXT_CONTENT:
            rendered.append(_json(dict(part)))
        else:
            _issue(
                issues,
                "unknown_content_part",
                part_path,
                f"unsupported content part type {part_type!r}; retained as JSON",
            )
            rendered.append(_json(dict(part)))
    return "\n".join(rendered)


def _parse_arguments(value: Any, *, path: str, issues: list[AuditIssue]) -> Any:
    """Apply the trajectory contract's JSON-string argument compatibility rule."""

    if not isinstance(value, str):
        if value is None:
            _issue(issues, "missing_tool_arguments", path, "tool arguments are absent")
            return {"raw": ""}
        return value
    try:
        return _strict_json_loads(value)
    except (TypeError, ValueError):
        return {"raw": value}


def _reasoning_text(item: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    for field in ("summary", "content"):
        value = item.get(field)
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for part in value:
                if isinstance(part, Mapping):
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
    text = item.get("text")
    if isinstance(text, str):
        chunks.append(text)
    return "\n".join(chunks)


class _AssistantAccumulator:
    def __init__(self) -> None:
        self.contents: list[str] = []
        self.reasoning_content: list[str] = []
        self.reasoning: Any | None = None
        self.tool_calls: list[ToolCall] = []
        self.has_message_item = False

    @property
    def populated(self) -> bool:
        return bool(
            self.contents
            or self.reasoning_content
            or self.reasoning is not None
            or self.tool_calls
            or self.has_message_item
        )

    def message(self) -> Message:
        return Message(
            role="assistant",
            content="\n".join(self.contents),
            reasoning_content="\n".join(
                chunk for chunk in self.reasoning_content if chunk
            ),
            reasoning=deepcopy(self.reasoning),
            tool_calls=self.tool_calls or None,
        )


def _server_call_name(item_type: str, item: Mapping[str, Any]) -> str:
    """Return a name for a real server call, deriving it from its call type."""

    explicit = item.get("name")
    if isinstance(explicit, str):
        return explicit
    for suffix in ("_call_output", "_call", "_output"):
        if item_type.endswith(suffix):
            return item_type.removesuffix(suffix)
    return ""


def _server_result_name(item: Mapping[str, Any]) -> str:
    """Return only a name explicitly present on an unpaired result block."""

    explicit = item.get("name")
    return explicit if isinstance(explicit, str) else ""


def _server_arguments(item: Mapping[str, Any]) -> Any:
    if "action" in item:
        return item["action"]
    if "arguments" in item:
        arguments = item["arguments"]
        if isinstance(arguments, str):
            try:
                return _strict_json_loads(arguments)
            except (TypeError, ValueError):
                return {"raw": arguments}
        return arguments
    if "input" in item:
        return item["input"]
    omitted = {
        "type",
        "id",
        "call_id",
        "status",
        "name",
        "namespace",
        "execution",
        "environment",
        "result",
        "results",
        "output",
    }
    return {key: value for key, value in item.items() if key not in omitted}


def _server_inline_result(item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Retain inline result fields without discarding their provider keys."""

    result = {
        field: deepcopy(item[field])
        for field in ("result", "results", "output")
        if field in item
    }
    return result or None


def _is_server_call(item_type: Any) -> bool:
    return isinstance(item_type, str) and item_type in _SERVER_CALL_TYPES


def _is_server_result(item_type: Any) -> bool:
    return isinstance(item_type, str) and item_type in _SERVER_RESULT_TYPES


def _qualified_item_name(item: Mapping[str, Any]) -> str:
    name = item.get("name")
    if not isinstance(name, str):
        return ""
    namespace = item.get("namespace")
    if namespace is not None and not isinstance(namespace, str):
        namespace = None
    return qualify_tool_name(namespace, name)


def _client_call_name(item_type: str, item: Mapping[str, Any]) -> str:
    builtin = _CLIENT_BUILTIN_NAMES.get(item_type)
    return builtin if builtin is not None else _qualified_item_name(item)


def _client_arguments(
    item_type: str,
    item: Mapping[str, Any],
    *,
    path: str,
    issues: list[AuditIssue],
) -> Any:
    if item_type == "custom_tool_call":
        raw_input = item.get("input", "")
        if not isinstance(raw_input, str):
            _issue(
                issues,
                "invalid_custom_tool_input",
                f"{path}.input",
                f"expected string; got {type(raw_input).__name__}",
            )
            raw_input = _as_string(raw_input)
        return {"input": raw_input}
    if item_type == "function_call":
        return _parse_arguments(
            item.get("arguments"),
            path=f"{path}.arguments",
            issues=issues,
        )

    preferred_fields = {
        "apply_patch_call": ("operation",),
        "computer_call": ("actions", "action"),
        "local_shell_call": ("action",),
        "shell_call": ("action",),
        "tool_search_call": ("arguments",),
    }
    for field in preferred_fields.get(item_type, ()):
        if field not in item:
            continue
        value = item[field]
        if field == "arguments" and isinstance(value, str):
            try:
                return _strict_json_loads(value)
            except (TypeError, ValueError):
                return {"raw": value}
        return deepcopy(value)
    return deepcopy(_server_arguments(item))


def _client_result_content(item_type: str, item: Mapping[str, Any]) -> str:
    if item_type in {"function_call_output", "custom_tool_call_output"}:
        output = item.get("output", "")
        return output if isinstance(output, str) else _json(output)

    # ``status`` is part of the client result itself for built-ins such as
    # apply_patch (``completed`` versus ``failed``), rather than disposable
    # provider-envelope metadata.  Preserve it with the result payload.
    envelope = {"type", "id", "call_id", "name", "namespace", "execution"}
    payload = {
        key: deepcopy(value) for key, value in item.items() if key not in envelope
    }
    return _json(payload)


def _raw_call_id(item: Mapping[str, Any]) -> str | None:
    """Return only a real Responses ``call_id`` suitable for linkage."""

    value = item.get("call_id")
    return value if isinstance(value, str) and value else None


def _stable_missing_id(origin: str, index: int) -> str:
    return f"missing:{origin}:{index}"


def _display_item_id(item: Mapping[str, Any], *, origin: str, index: int) -> str:
    """Choose a display ID without allowing ``item.id`` to imply linkage."""

    call_id = _raw_call_id(item)
    if call_id is not None:
        return call_id
    item_id = item.get("id")
    if isinstance(item_id, str) and item_id:
        return item_id
    return _stable_missing_id(origin, index)


def _conditional_family(item_type: Any) -> str | None:
    if item_type in {"tool_search_call", "tool_search_output"}:
        return "tool_search"
    if item_type in {"shell_call", "shell_call_output"}:
        return "shell"
    return None


def _explicit_item_execution(
    family: str,
    item: Mapping[str, Any],
    *,
    path: str,
    issues: list[AuditIssue],
) -> str | None:
    if family == "tool_search":
        if "execution" not in item or item.get("execution") is None:
            return None
        execution = item.get("execution")
        if execution in {_EXECUTION_CLIENT, _EXECUTION_SERVER}:
            return str(execution)
        _issue(
            issues,
            "ambiguous_tool_execution",
            f"{path}.execution",
            f"unsupported tool_search execution {execution!r}",
        )
        return None

    if "environment" not in item or item.get("environment") is None:
        return None
    environment = item.get("environment")
    if not isinstance(environment, Mapping):
        _issue(
            issues,
            "ambiguous_tool_execution",
            f"{path}.environment",
            "shell environment is not an object",
        )
        return None
    environment_type = environment.get("type")
    if environment_type == "local":
        return _EXECUTION_CLIENT
    if environment_type in {"container_auto", "container_reference"}:
        return _EXECUTION_SERVER
    _issue(
        issues,
        "ambiguous_tool_execution",
        f"{path}.environment.type",
        f"unsupported shell environment type {environment_type!r}",
    )
    return None


def _classify_items(
    raw_items: Sequence[Any],
    *,
    origin: str,
    issues: list[AuditIssue],
    path: str,
    execution_modes: Mapping[str, frozenset[str]],
) -> dict[int, str]:
    """Classify provider items by the component that actually executes them."""

    classifications: dict[int, str] = {}
    conditional_indexes: list[int] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            continue
        item_type = raw_item.get("type")
        if item_type in _CLIENT_CALL_TYPES:
            classifications[index] = "client_call"
        elif item_type in _CLIENT_RESULT_TYPES:
            classifications[index] = "client_result"
        elif item_type in _SERVER_CALL_TYPES:
            classifications[index] = "server_call"
        elif item_type in _SERVER_RESULT_TYPES:
            classifications[index] = "server_result"
        elif _conditional_family(item_type) is not None:
            conditional_indexes.append(index)

    def group_key(index: int) -> tuple[str, str, Any]:
        item = raw_items[index]
        assert isinstance(item, Mapping)
        item_type = item.get("type")
        family = _conditional_family(item_type)
        assert family is not None
        call_id = _raw_call_id(item)
        if call_id is not None:
            return family, "call_id", call_id
        if family == "tool_search":
            if item_type == "tool_search_call" and index + 1 < len(raw_items):
                neighbor = raw_items[index + 1]
                if (
                    isinstance(neighbor, Mapping)
                    and neighbor.get("type") == "tool_search_output"
                    and _raw_call_id(neighbor) is None
                ):
                    return family, "adjacent", index
            if item_type == "tool_search_output" and index > 0:
                neighbor = raw_items[index - 1]
                if (
                    isinstance(neighbor, Mapping)
                    and neighbor.get("type") == "tool_search_call"
                    and _raw_call_id(neighbor) is None
                ):
                    return family, "adjacent", index - 1
        return family, "index", index

    groups: dict[tuple[str, str, Any], list[int]] = {}
    for index in conditional_indexes:
        groups.setdefault(group_key(index), []).append(index)

    for (family, _, _), indexes in groups.items():
        explicit_modes: set[str] = set()
        for index in indexes:
            item = raw_items[index]
            assert isinstance(item, Mapping)
            explicit = _explicit_item_execution(
                family,
                item,
                path=f"{path}[{index}]",
                issues=issues,
            )
            if explicit is not None:
                explicit_modes.add(explicit)

        if len(explicit_modes) > 1:
            _issue(
                issues,
                "tool_execution_mismatch",
                f"{path}[{indexes[0]}]",
                f"paired {family} items disagree on execution",
            )
            continue

        definition_modes = execution_modes.get(family, frozenset())
        if explicit_modes:
            mode = next(iter(explicit_modes))
            if (
                origin == "response"
                and len(definition_modes) == 1
                and mode not in definition_modes
            ):
                _issue(
                    issues,
                    "tool_execution_mismatch",
                    f"{path}[{indexes[0]}]",
                    (
                        f"{family} item execution {mode!r} conflicts with "
                        f"the active definition"
                    ),
                )
        elif len(definition_modes) == 1:
            mode = next(iter(definition_modes))
        else:
            detail = (
                f"multiple active {family} definitions disagree on execution"
                if definition_modes
                else f"{family} item has no execution evidence"
            )
            _issue(
                issues,
                "ambiguous_tool_execution",
                f"{path}[{indexes[0]}]",
                detail,
            )
            continue

        for index in indexes:
            item = raw_items[index]
            assert isinstance(item, Mapping)
            item_type = item.get("type")
            is_result = item_type in _CONDITIONAL_RESULT_TYPES
            classifications[index] = f"{mode}_{'result' if is_result else 'call'}"

    return classifications


def _normalize_server_items(
    raw_items: Sequence[tuple[int, Mapping[str, Any], str]],
    *,
    origin: str,
    issues: list[AuditIssue],
    path: str,
) -> list[ServerToolCall]:
    """Pair server items by ``call_id`` plus the hosted tool-search exception."""

    def is_result(item_type: str) -> bool:
        return item_type in _SERVER_RESULT_TYPES | _CONDITIONAL_RESULT_TYPES

    call_indexes: dict[str, list[int]] = {}
    result_indexes: dict[str, list[int]] = {}
    for event_index, (_, item, item_type) in enumerate(raw_items):
        call_id = _raw_call_id(item)
        if call_id is None:
            continue
        target = result_indexes if is_result(item_type) else call_indexes
        target.setdefault(call_id, []).append(event_index)

    for call_id, indexes in call_indexes.items():
        if len(indexes) > 1:
            source_index = raw_items[indexes[1]][0]
            _issue(
                issues,
                "duplicate_server_tool_call_id",
                f"{path}[{source_index}]",
                f"multiple server tool calls use call_id {call_id!r}",
            )
    for call_id, indexes in result_indexes.items():
        if len(indexes) > 1:
            source_index = raw_items[indexes[1]][0]
            _issue(
                issues,
                "duplicate_server_tool_result",
                f"{path}[{source_index}]",
                f"multiple server tool results use call_id {call_id!r}",
            )
        calls = call_indexes.get(call_id, [])
        if not calls:
            source_index = raw_items[indexes[0]][0]
            _issue(
                issues,
                "orphan_server_tool_result",
                f"{path}[{source_index}]",
                f"no server tool call found for call_id {call_id!r}",
            )
        elif len(calls) > 1:
            source_index = raw_items[indexes[0]][0]
            _issue(
                issues,
                "ambiguous_server_tool_result",
                f"{path}[{source_index}]",
                f"server result call_id {call_id!r} matches multiple calls",
            )

    paired_results: dict[int, int] = {}
    claimed_results: set[int] = set()
    for call_id, calls in call_indexes.items():
        results = result_indexes.get(call_id, [])
        if len(calls) != 1 or not results:
            continue
        call_event_index = calls[0]
        _, call_item, _ = raw_items[call_event_index]
        if _server_inline_result(call_item) is not None:
            continue
        result_event_index = results[0]
        paired_results[call_event_index] = result_event_index
        claimed_results.add(result_event_index)

    # Hosted tool search currently emits ``call_id: null``.  Its immediately
    # adjacent call/output pair is still one provider-side operation; item IDs
    # remain display identifiers and are never generalized into linkage keys.
    null_tool_search_pairs: dict[int, int] = {}
    for event_index, (source_index, item, item_type) in enumerate(raw_items[:-1]):
        if item_type != "tool_search_call" or _raw_call_id(item) is not None:
            continue
        next_source_index, next_item, next_type = raw_items[event_index + 1]
        if (
            next_source_index == source_index + 1
            and next_type == "tool_search_output"
            and _raw_call_id(next_item) is None
        ):
            null_tool_search_pairs[event_index] = event_index + 1
            claimed_results.add(event_index + 1)

    pending: list[tuple[int, int, dict[str, Any]]] = []
    for event_index, (source_index, item, item_type) in enumerate(raw_items):
        raw_call_id = _raw_call_id(item)
        item_id = item.get("id")
        is_null_tool_search_pair = (
            event_index in null_tool_search_pairs or event_index in claimed_results
        ) and item_type in {"tool_search_call", "tool_search_output"}
        if (
            raw_call_id is None
            and not is_null_tool_search_pair
            and not (isinstance(item_id, str) and item_id)
        ):
            _issue(
                issues,
                "missing_server_tool_id",
                f"{path}[{source_index}]",
                f"{item_type} has no id",
            )

        if (
            is_result(item_type)
            and raw_call_id is None
            and not is_null_tool_search_pair
        ):
            display_id = (
                item_id
                if isinstance(item_id, str) and item_id
                else f"missing:server-result:{origin}:{path}[{source_index}]"
            )
            _issue(
                issues,
                "orphan_server_tool_result",
                f"{path}[{source_index}]",
                "server tool result has no linkable call_id",
            )
        else:
            display_id = _display_item_id(item, origin=origin, index=source_index)

        if is_result(item_type):
            if event_index in claimed_results:
                continue
            pending.append(
                (
                    source_index,
                    event_index,
                    {
                        "name": _server_result_name(item),
                        "id": display_id,
                        "arguments": {},
                        "origin": origin,
                        "result": deepcopy(dict(item)),
                    },
                )
            )
            continue

        result = _server_inline_result(item)
        order_index = source_index
        paired_index = paired_results.get(
            event_index, null_tool_search_pairs.get(event_index)
        )
        if paired_index is not None:
            result_source_index, result_item, _ = raw_items[paired_index]
            order_index = min(order_index, result_source_index)
            if raw_call_id is None and not (isinstance(item_id, str) and item_id):
                result_item_id = result_item.get("id")
                if isinstance(result_item_id, str) and result_item_id:
                    display_id = result_item_id
            # A standalone server result is itself the provider's result block.
            result = deepcopy(dict(result_item))
        pending.append(
            (
                order_index,
                event_index,
                {
                    "name": _server_call_name(item_type, item),
                    "id": display_id,
                    "arguments": deepcopy(_server_arguments(item)),
                    "origin": origin,
                    "result": result,
                },
            )
        )

    # Keep provider order, including results that appeared before their call.
    pending.sort(key=lambda entry: (entry[0], entry[1]))
    return [ServerToolCall(**record) for _, _, record in pending]


def _normalize_items(
    raw_items: Any,
    *,
    origin: str,
    issues: list[AuditIssue],
    path: str,
    execution_modes: Mapping[str, frozenset[str]],
    prior_completed_spawn_call_ids: Sequence[str] = (),
) -> tuple[
    list[Message],
    list[ServerToolCall],
    list[AgentMessageEvidence],
    list[CompactionRecord],
    list[str],
]:
    if raw_items is None:
        return [], [], [], [], list(dict.fromkeys(prior_completed_spawn_call_ids))
    if isinstance(raw_items, str):
        raw_items = [{"type": "message", "role": "user", "content": raw_items}]
    if not isinstance(raw_items, Sequence) or isinstance(
        raw_items, (str, bytes, bytearray)
    ):
        _issue(
            issues,
            "invalid_item_list",
            path,
            f"expected array; got {type(raw_items).__name__}",
        )
        return [], [], [], [], list(dict.fromkeys(prior_completed_spawn_call_ids))

    classifications = _classify_items(
        raw_items,
        origin=origin,
        issues=issues,
        path=path,
        execution_modes=execution_modes,
    )

    call_names: dict[tuple[str, str], list[str]] = {}
    call_indexes: dict[str, list[int]] = {}
    result_indexes: dict[str, list[int]] = {}
    call_types_by_id: dict[str, set[str]] = {}
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        call_id = _raw_call_id(item)
        if call_id is None:
            continue
        classification = classifications.get(index)
        if classification == "client_result":
            result_indexes.setdefault(call_id, []).append(index)
            continue
        if classification != "client_call" or not isinstance(item_type, str):
            continue
        call_type = item_type
        name = _client_call_name(call_type, item)
        result_type = _CLIENT_CALL_TO_RESULT.get(call_type)
        if result_type is not None:
            call_names.setdefault((result_type, call_id), []).append(name)
        call_indexes.setdefault(call_id, []).append(index)
        call_types_by_id.setdefault(call_id, set()).add(call_type)

    messages: list[Message] = []
    current = _AssistantAccumulator()
    server_items: list[tuple[int, Mapping[str, Any], str]] = []
    agent_messages: list[AgentMessageEvidence] = []
    compaction_items: list[CompactionRecord] = []
    completed_spawn_call_ids = list(dict.fromkeys(prior_completed_spawn_call_ids))
    completed_spawn_call_id_set = set(completed_spawn_call_ids)

    def flush_assistant() -> None:
        nonlocal current
        if current.populated:
            messages.append(current.message())
            current = _AssistantAccumulator()

    for index, raw_item in enumerate(raw_items):
        item_path = f"{path}[{index}]"
        if not isinstance(raw_item, Mapping):
            flush_assistant()
            _issue(
                issues,
                "invalid_response_item",
                item_path,
                f"expected object; got {type(raw_item).__name__}",
            )
            continue
        item = dict(raw_item)
        item_type = item.get("type")
        classification = classifications.get(index)

        if item_type == "compaction":
            flush_assistant()
            compaction_items.append(
                CompactionRecord(
                    origin=origin,
                    item_index=index,
                    item=deepcopy(item),
                )
            )
            continue

        if item_type == "agent_message":
            flush_assistant()
            fields: dict[str, str] = {}
            for field in ("id", "author", "recipient"):
                value = item.get(field)
                if isinstance(value, str) and value:
                    fields[field] = value
                    continue
                fields[field] = ""
                _issue(
                    issues,
                    f"invalid_agent_message_{field}",
                    f"{item_path}.{field}",
                    (
                        f"expected non-empty string; got {type(value).__name__}"
                        if value is not None
                        else "field is absent"
                    ),
                )
            agent_messages.append(
                AgentMessageEvidence(
                    origin=origin,
                    item_index=index,
                    item=deepcopy(item),
                    preceding_completed_spawn_call_ids=list(completed_spawn_call_ids),
                )
            )
            continue

        if item_type == "message" or (item_type is None and "role" in item):
            role = item.get("role")
            if role not in {"system", "developer", "user", "assistant"}:
                flush_assistant()
                _issue(
                    issues,
                    "invalid_message_role",
                    f"{item_path}.role",
                    f"unsupported role {role!r}",
                )
                continue
            content = _render_content(
                item.get("content", ""),
                path=f"{item_path}.content",
                issues=issues,
            )
            if role == "assistant":
                if current.has_message_item:
                    flush_assistant()
                current.contents.append(content)
                current.has_message_item = True
            else:
                flush_assistant()
                messages.append(Message(role=role, content=content))
            continue

        if item_type == "reasoning":
            # The target contract has one raw OpenAI reasoning value per
            # assistant message. A new item therefore starts a new canonical
            # assistant segment instead of inventing an array wrapper.
            if current.populated:
                flush_assistant()
            text = _reasoning_text(item)
            if text:
                current.reasoning_content.append(text)
            current.reasoning = deepcopy(item)
            continue

        if classification == "client_call" and isinstance(item_type, str):
            raw_call_id = _raw_call_id(item)
            call_id = raw_call_id or _stable_missing_id(origin, index)
            name = _client_call_name(item_type, item)
            if raw_call_id is None:
                _issue(
                    issues,
                    "missing_tool_call_id",
                    item_path,
                    f"{item_type} has no call_id",
                )
            if not name:
                _issue(
                    issues,
                    "missing_tool_name",
                    item_path,
                    f"{item_type} has no name",
                )
            arguments = _client_arguments(
                item_type,
                item,
                path=item_path,
                issues=issues,
            )
            current.tool_calls.append(
                ToolCall(
                    id=call_id,
                    function=FunctionCall(name=name, arguments=arguments),
                )
            )
            continue

        if classification == "client_result" and isinstance(item_type, str):
            flush_assistant()
            raw_call_id = _raw_call_id(item)
            call_id = raw_call_id or _stable_missing_id(origin, index)
            if raw_call_id is None:
                _issue(
                    issues,
                    "missing_tool_result_id",
                    item_path,
                    f"{item_type} has no call_id",
                )
            linked_names = (
                call_names.get((str(item_type), raw_call_id), []) if raw_call_id else []
            )
            linked = len(linked_names) == 1
            raw_name = item.get("name")
            if linked:
                name = linked_names[0]
            elif raw_call_id and raw_call_id in call_types_by_id:
                name = ""
                actual_types = ", ".join(sorted(call_types_by_id[raw_call_id]))
                _issue(
                    issues,
                    "tool_result_type_mismatch",
                    item_path,
                    f"{item_type} does not match client call type(s) {actual_types}",
                )
            else:
                result_call_type = _CLIENT_RESULT_TO_CALL.get(item_type)
                name = _CLIENT_BUILTIN_NAMES.get(result_call_type or "", "")
                if not name and isinstance(raw_name, str):
                    name = _qualified_item_name(item)
            if not linked and not (raw_call_id and raw_call_id in call_types_by_id):
                _issue(
                    issues,
                    "orphan_tool_result",
                    item_path,
                    f"no unique client tool call found for {call_id!r}",
                )
            output = item.get("output", "")
            if (
                raw_call_id is not None
                and linked
                and len(call_indexes.get(raw_call_id, ())) == 1
                and len(result_indexes.get(raw_call_id, ())) == 1
                and call_indexes[raw_call_id][0] < index
                and isinstance(output, str)
                and output
                in {
                    f"unsupported call: {name}",
                    f"unsupported custom tool call: {name}",
                }
            ):
                _issue(
                    issues,
                    RESPONSES_UNSUPPORTED_CALL_EVIDENCE,
                    item_path,
                    (
                        f"provider rejected client tool call id={call_id!r} "
                        f"name={name!r} as unsupported"
                    ),
                    severity=Severity.WARNING,
                )
            messages.append(
                Message(
                    role="tool",
                    content=_client_result_content(item_type, item),
                    tool_call_id=call_id,
                    name=name,
                )
            )
            if (
                raw_call_id is not None
                and linked
                and is_spawn_tool_name(name)
                and len(call_indexes.get(raw_call_id, ())) == 1
                and len(result_indexes.get(raw_call_id, ())) == 1
                and call_indexes[raw_call_id][0] < index
                and raw_call_id not in completed_spawn_call_id_set
            ):
                completed_spawn_call_id_set.add(raw_call_id)
                completed_spawn_call_ids.append(raw_call_id)
            continue

        if classification in {"server_call", "server_result"}:
            server_items.append((index, item, str(item_type)))
            continue

        if _conditional_family(item_type) is not None:
            # Classification already emitted the precise ambiguity/conflict.
            # Preserve message boundaries around an unprojectable tool item.
            flush_assistant()
            continue

        if item_type == "additional_tools":
            # Definitions are collected separately by _normalize_tools.
            continue

        flush_assistant()
        _issue(
            issues,
            "unknown_response_item",
            item_path,
            f"unsupported item type {item_type!r}",
        )

    flush_assistant()
    return (
        messages,
        _normalize_server_items(
            server_items,
            origin=origin,
            issues=issues,
            path=path,
        ),
        agent_messages,
        compaction_items,
        completed_spawn_call_ids,
    )


def _custom_parameters(tool: Mapping[str, Any]) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
        "additionalProperties": False,
    }
    if "format" in tool:
        parameters["x-openai-custom-tool-format"] = tool["format"]
    return parameters


def _normalize_tools(
    request: Mapping[str, Any],
    response_items: Sequence[Mapping[str, Any]],
    issues: list[AuditIssue],
) -> tuple[list[ToolDefinition], dict[str, frozenset[str]]]:
    candidates: list[tuple[Mapping[str, Any], str | None, str, str | None]] = []
    execution_modes: dict[str, set[str]] = {
        "shell": set(),
        "tool_search": set(),
    }
    request_tools = request.get("tools", [])
    if request_tools is None:
        request_tools = []
    if not isinstance(request_tools, Sequence) or isinstance(
        request_tools, (str, bytes, bytearray)
    ):
        _issue(
            issues,
            "invalid_tool_definitions",
            "request_body.tools",
            f"expected array; got {type(request_tools).__name__}",
        )
        request_tools = []

    def add_builtin_candidate(
        tool: Mapping[str, Any], path: str, canonical_name: str
    ) -> None:
        candidates.append((tool, None, path, canonical_name))

    def add_candidates(tools: Iterable[Any], base_path: str) -> None:
        for index, raw_tool in enumerate(tools):
            if not isinstance(raw_tool, Mapping):
                _issue(
                    issues,
                    "invalid_tool_definition",
                    f"{base_path}[{index}]",
                    f"expected object; got {type(raw_tool).__name__}",
                )
                continue
            tool_type = raw_tool.get("type")
            if tool_type in {"function", "custom"}:
                namespace = raw_tool.get("namespace")
                candidates.append(
                    (
                        raw_tool,
                        namespace if isinstance(namespace, str) else None,
                        f"{base_path}[{index}]",
                        None,
                    )
                )
            elif tool_type == "namespace":
                namespace = raw_tool.get("name")
                nested = raw_tool.get("tools", [])
                if not isinstance(namespace, str) or not namespace:
                    _issue(
                        issues,
                        "invalid_namespace_definition",
                        f"{base_path}[{index}]",
                        "namespace has no name",
                    )
                    continue
                if not isinstance(nested, Sequence) or isinstance(
                    nested, (str, bytes, bytearray)
                ):
                    _issue(
                        issues,
                        "invalid_namespace_definition",
                        f"{base_path}[{index}].tools",
                        "namespace tools are not an array",
                    )
                    continue
                for nested_index, nested_tool in enumerate(nested):
                    if isinstance(nested_tool, Mapping) and nested_tool.get("type") in {
                        "function",
                        "custom",
                    }:
                        candidates.append(
                            (
                                nested_tool,
                                namespace,
                                f"{base_path}[{index}].tools[{nested_index}]",
                                None,
                            )
                        )
                    else:
                        _issue(
                            issues,
                            "unsupported_namespace_tool",
                            f"{base_path}[{index}].tools[{nested_index}]",
                            "namespace member is not a function/custom tool",
                        )
            elif tool_type == "tool_search":
                if "execution" not in raw_tool:
                    execution = _EXECUTION_SERVER
                else:
                    raw_execution = raw_tool.get("execution")
                    execution = (
                        str(raw_execution)
                        if raw_execution in {_EXECUTION_CLIENT, _EXECUTION_SERVER}
                        else None
                    )
                    if execution is None:
                        _issue(
                            issues,
                            "ambiguous_tool_execution",
                            f"{base_path}[{index}].execution",
                            f"unsupported tool_search execution {raw_execution!r}",
                        )
                if execution is not None:
                    execution_modes["tool_search"].add(execution)
                    if execution == _EXECUTION_CLIENT:
                        add_builtin_candidate(
                            raw_tool, f"{base_path}[{index}]", "tool_search"
                        )
            elif tool_type == "shell":
                environment = raw_tool.get("environment")
                environment_type = (
                    environment.get("type")
                    if isinstance(environment, Mapping)
                    else None
                )
                if environment_type == "local":
                    execution_modes["shell"].add(_EXECUTION_CLIENT)
                    add_builtin_candidate(raw_tool, f"{base_path}[{index}]", "shell")
                elif environment_type in {"container_auto", "container_reference"}:
                    execution_modes["shell"].add(_EXECUTION_SERVER)
                else:
                    _issue(
                        issues,
                        "ambiguous_tool_execution",
                        f"{base_path}[{index}].environment",
                        f"unsupported shell environment type {environment_type!r}",
                    )
            elif tool_type in {"computer", "computer_use_preview"}:
                add_builtin_candidate(raw_tool, f"{base_path}[{index}]", "computer")
            elif tool_type == "apply_patch":
                add_builtin_candidate(raw_tool, f"{base_path}[{index}]", "apply_patch")
            elif tool_type == "local_shell":
                add_builtin_candidate(raw_tool, f"{base_path}[{index}]", "local_shell")
            elif tool_type in _HOSTED_TOOL_TYPES:
                # Provider-hosted definitions do not enter the client tool set.
                continue

    add_candidates(request_tools, "request_body.tools")

    def add_item_tools(items: Any, base_path: str) -> None:
        if not isinstance(items, Sequence) or isinstance(
            items, (str, bytes, bytearray)
        ):
            return
        for index, item in enumerate(items):
            if not isinstance(item, Mapping) or item.get("type") not in {
                "additional_tools",
                "tool_search_output",
            }:
                continue
            additional = item.get("tools", [])
            if isinstance(additional, Sequence) and not isinstance(
                additional, (str, bytes, bytearray)
            ):
                add_candidates(additional, f"{base_path}[{index}].tools")
            else:
                issue_code = (
                    "invalid_additional_tools"
                    if item.get("type") == "additional_tools"
                    else "invalid_tool_search_output"
                )
                _issue(
                    issues,
                    issue_code,
                    f"{base_path}[{index}].tools",
                    f"{item.get('type')}.tools is not an array",
                )

    add_item_tools(request.get("input"), "request_body.input")
    add_item_tools(response_items, "response.output")

    definitions: list[ToolDefinition] = []
    seen: set[str] = set()
    for tool, namespace, tool_path, canonical_name in candidates:
        name = canonical_name if canonical_name is not None else tool.get("name")
        if not isinstance(name, str) or not name:
            _issue(
                issues, "missing_tool_definition_name", tool_path, "tool has no name"
            )
            continue
        name = qualify_tool_name(namespace, name)
        description = tool.get("description", "")
        if not isinstance(description, str):
            _issue(
                issues,
                "invalid_tool_description",
                f"{tool_path}.description",
                f"expected string; got {type(description).__name__}",
            )
            description = _as_string(description)
        if tool.get("type") == "custom":
            parameters = _custom_parameters(tool)
        else:
            parameters = tool.get("parameters", {})
            if not isinstance(parameters, Mapping):
                _issue(
                    issues,
                    "invalid_tool_parameters",
                    f"{tool_path}.parameters",
                    f"expected object; got {type(parameters).__name__}",
                )
                parameters = {}
            else:
                parameters = dict(parameters)
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        )
        signature = _json(definition.model_dump(mode="json"))
        if signature not in seen:
            seen.add(signature)
            definitions.append(definition)
    return definitions, {
        family: frozenset(modes) for family, modes in execution_modes.items()
    }


def _decode_event_data(
    data: Any, *, frame_path: str, issues: list[AuditIssue]
) -> list[dict[str, Any]]:
    if isinstance(data, Mapping):
        return [dict(data)]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        events: list[dict[str, Any]] = []
        for index, value in enumerate(data):
            events.extend(
                _decode_event_data(
                    value,
                    frame_path=f"{frame_path}[{index}]",
                    issues=issues,
                )
            )
        return events
    if isinstance(data, str):
        if data.strip() == "[DONE]":
            return []
        try:
            decoded = json.loads(data)
        except ValueError:
            _issue(
                issues,
                "invalid_sse_data",
                frame_path,
                "SSE data is not JSON",
            )
            return []
        return _decode_event_data(decoded, frame_path=frame_path, issues=issues)
    _issue(
        issues,
        "invalid_sse_data",
        frame_path,
        f"expected object or JSON string; got {type(data).__name__}",
    )
    return []


def _extract_events(
    body: Sequence[Any], issues: list[AuditIssue]
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    capture_sequences: dict[int, str] = {}
    previous_capture_seq: int | None = None

    for index, raw_frame in enumerate(body):
        frame_path = f"response_body[{index}]"
        if not isinstance(raw_frame, Mapping):
            _issue(
                issues,
                "invalid_sse_frame",
                frame_path,
                f"expected object; got {type(raw_frame).__name__}",
            )
            continue
        frame = dict(raw_frame)
        capture_seq = frame.get("seq")
        if isinstance(capture_seq, int):
            encoded = _json(frame)
            previous = capture_sequences.get(capture_seq)
            if previous is not None:
                if previous != encoded:
                    _issue(
                        issues,
                        "conflicting_sse_sequence",
                        f"{frame_path}.seq",
                        f"sequence {capture_seq} has conflicting frames",
                    )
                # Identical retransmissions have no semantic effect.
                continue
            if (
                previous_capture_seq is not None
                and capture_seq != previous_capture_seq + 1
            ):
                _issue(
                    issues,
                    "sse_sequence_gap",
                    f"{frame_path}.seq",
                    f"expected {previous_capture_seq + 1}, got {capture_seq}",
                )
            capture_sequences[capture_seq] = encoded
            previous_capture_seq = capture_seq

        if "data" in frame:
            decoded = _decode_event_data(
                frame["data"], frame_path=f"{frame_path}.data", issues=issues
            )
            envelope_event = frame.get("event")
            for event in decoded:
                event_type = event.get("type")
                if (
                    isinstance(envelope_event, str)
                    and isinstance(event_type, str)
                    and envelope_event != event_type
                ):
                    _issue(
                        issues,
                        "sse_event_type_mismatch",
                        frame_path,
                        f"event={envelope_event!r}, data.type={event_type!r}",
                    )
                events.append(event)
        elif isinstance(frame.get("type"), str):
            events.append(frame)
        else:
            _issue(
                issues,
                "invalid_sse_frame",
                frame_path,
                "frame has neither data nor type",
            )

    previous_event_seq: int | None = None
    event_sequences: dict[int, str] = {}
    deduplicated: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        event_path = f"response_body.events[{index}]"
        sequence = event.get("sequence_number")
        if isinstance(sequence, int):
            encoded = _json(event)
            previous = event_sequences.get(sequence)
            if previous is not None:
                if previous != encoded:
                    _issue(
                        issues,
                        "conflicting_event_sequence",
                        f"{event_path}.sequence_number",
                        f"sequence {sequence} has conflicting events",
                    )
                # Do not apply a retransmitted sequence twice.  A conflicting
                # duplicate remains quarantinable through the issue above.
                continue
            if previous_event_seq is not None and sequence != previous_event_seq + 1:
                _issue(
                    issues,
                    "event_sequence_gap",
                    f"{event_path}.sequence_number",
                    f"expected {previous_event_seq + 1}, got {sequence}",
                )
            event_sequences[sequence] = encoded
            previous_event_seq = sequence

        event_type = event.get("type")
        known = event_type in _LIFECYCLE_EVENTS or (
            isinstance(event_type, str) and event_type.startswith(_KNOWN_EVENT_PREFIXES)
        )
        if not known:
            _issue(
                issues,
                "unknown_sse_event",
                f"{event_path}.type",
                f"unsupported SSE event {event_type!r}",
            )
        deduplicated.append(event)
    return deduplicated


def _ensure_output_item(
    items: MutableMapping[int, dict[str, Any]],
    event: Mapping[str, Any],
    item_type: str,
) -> dict[str, Any]:
    index = event.get("output_index")
    if not isinstance(index, int):
        index = max(items, default=-1) + 1
    if index not in items:
        item: dict[str, Any] = {
            "type": item_type,
            "id": event.get("item_id", ""),
        }
        if item_type == "message":
            item.update(role="assistant", content=[])
        elif item_type == "reasoning":
            item.update(summary=[])
        items[index] = item
    return items[index]


def _ensure_part(parts: list[Any], index: int, part_type: str) -> dict[str, Any]:
    while len(parts) <= index:
        parts.append({"type": part_type, "text": ""})
    part = parts[index]
    if not isinstance(part, dict):
        part = {"type": part_type, "text": ""}
        parts[index] = part
    part.setdefault("type", part_type)
    return part


def _items_from_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items: dict[int, dict[str, Any]] = {}
    completed_indexes: set[int] = set()

    for event in events:
        event_type = event.get("type")
        output_index = event.get("output_index")
        if event_type == "response.output_item.added" and isinstance(output_index, int):
            item = event.get("item")
            if isinstance(item, Mapping):
                items[output_index] = deepcopy(dict(item))
            continue
        if event_type == "response.output_item.done" and isinstance(output_index, int):
            item = event.get("item")
            if isinstance(item, Mapping):
                items[output_index] = deepcopy(dict(item))
                completed_indexes.add(output_index)
            continue

        if not isinstance(event_type, str):
            continue
        is_done = event_type.endswith(".done")
        value = event.get("text") if is_done else event.get("delta")
        if event_type.startswith(("response.output_text.", "response.refusal.")):
            if not isinstance(value, str) or output_index in completed_indexes:
                continue
            item = _ensure_output_item(items, event, "message")
            content = item.setdefault("content", [])
            if not isinstance(content, list):
                content = []
                item["content"] = content
            content_index = event.get("content_index", 0)
            if not isinstance(content_index, int):
                content_index = 0
            part_type = (
                "refusal"
                if event_type.startswith("response.refusal.")
                else "output_text"
            )
            part = _ensure_part(content, content_index, part_type)
            field = "refusal" if part_type == "refusal" else "text"
            part[field] = value if is_done else f"{part.get(field, '')}{value}"
        elif event_type.startswith(
            ("response.reasoning_summary_text.", "response.reasoning_text.")
        ):
            if not isinstance(value, str) or output_index in completed_indexes:
                continue
            item = _ensure_output_item(items, event, "reasoning")
            field_name = (
                "summary"
                if event_type.startswith("response.reasoning_summary_text.")
                else "content"
            )
            parts = item.setdefault(field_name, [])
            if not isinstance(parts, list):
                parts = []
                item[field_name] = parts
            part_index = event.get("summary_index", event.get("content_index", 0))
            if not isinstance(part_index, int):
                part_index = 0
            part_type = "summary_text" if field_name == "summary" else "reasoning_text"
            part = _ensure_part(parts, part_index, part_type)
            part["text"] = value if is_done else f"{part.get('text', '')}{value}"
        elif event_type.startswith("response.function_call_arguments."):
            if output_index in completed_indexes:
                continue
            value = event.get("arguments") if is_done else event.get("delta")
            if not isinstance(value, str):
                continue
            item = _ensure_output_item(items, event, "function_call")
            item["arguments"] = (
                value if is_done else f"{item.get('arguments', '')}{value}"
            )
        elif event_type.startswith("response.custom_tool_call_input."):
            if output_index in completed_indexes:
                continue
            value = event.get("input") if is_done else event.get("delta")
            if not isinstance(value, str):
                continue
            item = _ensure_output_item(items, event, "custom_tool_call")
            item["input"] = value if is_done else f"{item.get('input', '')}{value}"

    terminal_output: Sequence[Any] = []
    for event in reversed(events):
        response = event.get("response")
        if isinstance(response, Mapping) and isinstance(
            response.get("output"), Sequence
        ):
            terminal_output = response["output"]
            break
    for index, item in enumerate(terminal_output):
        if index not in items and isinstance(item, Mapping):
            items[index] = deepcopy(dict(item))
    return [items[index] for index in sorted(items)]


def _validate_stream_structure(
    events: Sequence[Mapping[str, Any]], issues: list[AuditIssue]
) -> None:
    terminal_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("type") in _TERMINAL_EVENTS
    ]
    if terminal_indexes and terminal_indexes[-1] != len(events) - 1:
        _issue(
            issues,
            "events_after_terminal",
            "response_body",
            "SSE events occur after the terminal event",
        )

    families = (
        ("response.output_item.added", "response.output_item.done", "output_index"),
        ("response.content_part.added", "response.content_part.done", "content_index"),
        (
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "summary_index",
        ),
    )
    for added_type, done_type, sub_index_field in families:
        added: set[tuple[Any, Any]] = set()
        done: set[tuple[Any, Any]] = set()
        for event in events:
            event_type = event.get("type")
            if event_type not in {added_type, done_type}:
                continue
            key = (event.get("output_index"), event.get(sub_index_field))
            (added if event_type == added_type else done).add(key)
        for key in sorted(added - done, key=repr):
            _issue(
                issues,
                "unclosed_sse_item",
                "response_body",
                f"{added_type} for {key!r} has no matching {done_type}",
            )
        for key in sorted(done - added, key=repr):
            _issue(
                issues,
                "missing_sse_item_start",
                "response_body",
                f"{done_type} for {key!r} has no matching {added_type}",
            )


def _parse_response_body(
    body: Any,
    *,
    status_code: int | None,
    issues: list[AuditIssue],
) -> tuple[list[dict[str, Any]], str, bool, str]:
    """Return output items, outcome, wire completeness, and response model."""

    response_issue_start = len(issues)

    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            _issue(
                issues,
                "invalid_response_body",
                "response_body",
                "response body is not JSON",
            )
            return [], "capture_invalid", False, ""

    if isinstance(body, Mapping):
        response = dict(body)
        response_model = response.get("model")
        model = response_model if isinstance(response_model, str) else ""
        response_status = response.get("status")
        if (
            status_code is not None
            and not 200 <= status_code < 300
            or response.get("error") is not None
            or response_status == "failed"
        ):
            outcome = "api_error"
        elif response_status in {"incomplete", "cancelled"}:
            outcome = "truncated"
        else:
            outcome = "success"
        if outcome == "api_error" and (status_code is None or 200 <= status_code < 300):
            _issue(
                issues,
                "openai_api_error",
                "response_body",
                _stable_api_error_detail(response, status=status_code),
            )
        output = response.get("output", [])
        if not isinstance(output, Sequence) or isinstance(
            output, (str, bytes, bytearray)
        ):
            _issue(
                issues,
                "invalid_response_output",
                "response_body.output",
                "response output is not an array",
            )
            output = []
        new_errors = any(
            issue.severity == Severity.ERROR for issue in issues[response_issue_start:]
        )
        if outcome == "success" and new_errors:
            outcome = "capture_invalid"
        wire_complete = outcome == "success" and not new_errors
        return (
            [dict(item) for item in output if isinstance(item, Mapping)],
            outcome,
            wire_complete,
            model,
        )

    if not isinstance(body, Sequence) or isinstance(body, (str, bytes, bytearray)):
        _issue(
            issues,
            "invalid_response_body",
            "response_body",
            f"expected response object or SSE array; got {type(body).__name__}",
        )
        return [], "capture_invalid", False, ""

    events = _extract_events(body, issues)
    _validate_stream_structure(events, issues)
    terminals = [event for event in events if event.get("type") in _TERMINAL_EVENTS]
    if len(terminals) > 1:
        _issue(
            issues,
            "multiple_terminal_events",
            "response_body",
            f"found {len(terminals)} terminal SSE events",
        )
    terminal = terminals[-1] if terminals else None

    if terminal is None:
        outcome = (
            "api_error"
            if status_code is not None and not 200 <= status_code < 300
            else "truncated"
        )
        _issue(
            issues,
            "missing_terminal_event",
            "response_body",
            "SSE stream ended without completed, incomplete, failed, or error",
        )
    else:
        terminal_type = terminal.get("type")
        response = terminal.get("response")
        response_status = (
            response.get("status") if isinstance(response, Mapping) else None
        )
        if terminal_type in {"response.failed", "error"} or (
            status_code is not None and not 200 <= status_code < 300
        ):
            outcome = "api_error"
        elif terminal_type == "response.incomplete" or response_status in {
            "incomplete",
            "cancelled",
        }:
            outcome = "truncated"
        elif response_status == "failed":
            outcome = "api_error"
        else:
            outcome = "success"
        if outcome == "api_error" and (status_code is None or 200 <= status_code < 300):
            _issue(
                issues,
                "openai_api_error",
                "response_body",
                _stable_api_error_detail(terminal, status=status_code),
            )

    corrupt_codes = {
        "conflicting_sse_sequence",
        "sse_sequence_gap",
        "invalid_sse_frame",
        "invalid_sse_data",
        "conflicting_event_sequence",
        "event_sequence_gap",
        "sse_event_type_mismatch",
        "unknown_sse_event",
        "multiple_terminal_events",
        "events_after_terminal",
    }
    response_issues = issues[response_issue_start:]
    corrupt = any(issue.code in corrupt_codes for issue in response_issues)
    if terminal is not None and terminal.get("type") == "response.completed":
        corrupt = corrupt or any(
            issue.code in {"unclosed_sse_item", "missing_sse_item_start"}
            for issue in response_issues
        )
    if corrupt and outcome not in {"api_error"}:
        outcome = "capture_invalid"
    response_status = None
    if terminal is not None and isinstance(terminal.get("response"), Mapping):
        response_status = terminal["response"].get("status")
    wire_complete = (
        terminal is not None
        and terminal.get("type") == "response.completed"
        and response_status not in {"failed", "incomplete", "cancelled"}
        and outcome == "success"
        and not corrupt
    )

    model = ""
    for event in reversed(events):
        response = event.get("response")
        if isinstance(response, Mapping) and isinstance(response.get("model"), str):
            model = response["model"]
            break
    return _items_from_events(events), outcome, wire_complete, model


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
        decoded = json.loads(value)
    except ValueError:
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
    """Merge identity fields without carrying raw capture headers downstream."""

    direct_body_sources: list[tuple[str, Mapping[str, Any]]] = []
    encoded_sources: list[tuple[str, Mapping[str, Any]]] = []
    saw_codex_metadata = False

    # Body metadata is already semantic request data.  Keep explicit values at
    # higher priority than the JSON-encoded compatibility copy.
    for field in ("client_metadata", "metadata"):
        value = request.get(field)
        if not isinstance(value, Mapping):
            continue
        direct_body_sources.append((f"request_body.{field}", value))
        encoded = _decoded_turn_metadata(
            value.get("x-codex-turn-metadata"),
            path=f"request_body.{field}.x-codex-turn-metadata",
            issues=issues,
        )
        if encoded is not None:
            encoded_sources.append(
                (f"request_body.{field}.x-codex-turn-metadata", encoded)
            )
            saw_codex_metadata = True

    # Headers are untrusted transport metadata.  Read only the three explicitly
    # allowed keys and retain only their decoded identity values.
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
    header_turn_metadata = _decoded_turn_metadata(
        headers.get("x-codex-turn-metadata"),
        path="request_headers.x-codex-turn-metadata",
        issues=issues,
    )
    if header_turn_metadata is not None:
        encoded_sources.append(
            ("request_headers.x-codex-turn-metadata", header_turn_metadata)
        )
        saw_codex_metadata = True
    header_sources: list[tuple[str, Mapping[str, Any]]] = []
    if headers:
        header_sources.append(("request_headers", headers))
        saw_codex_metadata = True

    # The encoded copies rank below explicit body values but above header-only
    # aliases.  Conflicts in trajectory identity fields are still surfaced
    # regardless of priority.
    all_sources = [*direct_body_sources, *encoded_sources, *header_sources]

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
        for source_name, source in all_sources:
            for key in keys:
                value = source.get(key)
                if isinstance(value, str) and value:
                    values.append((f"{source_name}.{key}", value))
                    break
        return values

    identity: dict[str, str] = {}
    for field, field_aliases in aliases.items():
        candidates = values_for(field_aliases)
        if candidates:
            identity[field] = candidates[0][1]
        distinct = {value for _, value in candidates}
        if len(distinct) > 1:
            detail = "; ".join(f"{source}={value!r}" for source, value in candidates)
            _issue(
                issues,
                "metadata_conflict",
                field,
                f"conflicting {field}: {detail}",
            )

    # These labels describe different aspects of sub-agent execution and may
    # legitimately use different vocabularies.  Their transport copies are
    # advisory, so select deterministically without comparing them or turning
    # disagreement into a capture-level integrity failure.
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


def parse_responses_capture(
    capture: dict[str, Any],
    *,
    source_path: str,
    source_sha256: str,
) -> Snapshot:
    """Parse one freerouter ``/v1/responses`` capture.

    The adapter returns a snapshot even for API failures and truncated streams;
    structural/semantic defects are recorded in ``Snapshot.issues`` so the
    admission stage can quarantine them with their lineage intact.
    """

    if not isinstance(capture, dict):
        raise ResponsesAdapterError("capture must be an object")
    endpoint = capture.get("path")
    if not isinstance(endpoint, str) or endpoint.split("?", 1)[0].rstrip("/") not in {
        "/v1/responses",
        "/responses",
    }:
        raise ResponsesAdapterError(f"not a Responses capture: {endpoint!r}")
    request = capture.get("request_body")
    if not isinstance(request, Mapping):
        raise ResponsesAdapterError("request_body must be an object")

    issues: list[AuditIssue] = []
    instructions_value = request.get("instructions")
    if instructions_value is None:
        instructions = ""
    elif isinstance(instructions_value, str):
        instructions = instructions_value
    else:
        instructions = ""
        _issue(
            issues,
            "invalid_instructions",
            "request_body.instructions",
            f"expected string or null; got {type(instructions_value).__name__}",
        )

    request_input = request.get("input", [])
    status_code_raw = capture.get("status_code")
    status_code = (
        status_code_raw
        if isinstance(status_code_raw, int)
        and not isinstance(status_code_raw, bool)
        and status_code_raw > 0
        else None
    )
    transport_error = status_code is None
    if transport_error:
        _issue(
            issues,
            "missing_http_status",
            "status_code",
            "capture has no valid HTTP status code",
        )
    elif not 200 <= status_code < 300:
        _issue(
            issues,
            "openai_api_error",
            "response_body",
            _stable_api_error_detail(capture.get("response_body"), status=status_code),
        )
    response_items, outcome, wire_complete, response_model = _parse_response_body(
        capture.get("response_body"),
        status_code=status_code,
        issues=issues,
    )
    tools, execution_modes = _normalize_tools(request, response_items, issues)
    (
        history,
        history_server_tools,
        history_agent_messages,
        history_compaction_items,
        history_completed_spawn_ids,
    ) = _normalize_items(
        request_input,
        origin="history",
        issues=issues,
        path="request_body.input",
        execution_modes=execution_modes,
    )
    (
        response,
        response_server_tools,
        response_agent_messages,
        response_compaction_items,
        _,
    ) = _normalize_items(
        response_items,
        origin="response",
        issues=issues,
        path="response.output",
        execution_modes=execution_modes,
        prior_completed_spawn_call_ids=history_completed_spawn_ids,
    )

    identity, saw_codex_metadata = _identity_metadata(capture, request, issues)
    capture_session = capture.get("session_id")
    session_id = identity.get("session_id", "") or (
        capture_session if isinstance(capture_session, str) else ""
    )
    if (
        isinstance(capture_session, str)
        and identity.get("session_id")
        and capture_session != identity["session_id"]
    ):
        _issue(
            issues,
            "metadata_conflict",
            "session_id",
            "capture session_id conflicts with request metadata session_id",
        )
    turn_id = identity.get("turn_id", "")
    parent_thread_id = identity.get("parent_thread_id", "")
    parent_turn_id = identity.get("parent_turn_id", "")
    forked_from_thread_id = identity.get("forked_from_thread_id", "")
    subagent_marker = identity.get("subagent_marker", "")
    explicit_thread_id = identity.get("thread_id", "")
    has_subagent_linkage = bool(
        subagent_marker or parent_thread_id or parent_turn_id or forked_from_thread_id
    )
    if explicit_thread_id:
        thread_id = explicit_thread_id
    elif has_subagent_linkage:
        # A session id identifies the whole run, not a particular child.  A
        # per-capture namespace is intentionally conservative: without a real
        # child thread id we may retain multiple orphans, but can never merge a
        # child into the main transcript or into an unrelated child.
        thread_id = f"__isolated_subagent__:{source_path}"
        _issue(
            issues,
            "isolated_subagent_missing_thread_id",
            "request_body",
            "sub-agent linkage has no explicit thread id; capture was isolated",
            severity=Severity.WARNING,
        )
    else:
        thread_id = session_id

    request_model = request.get("model")
    model = request_model if isinstance(request_model, str) else response_model
    request_id = capture.get("request_id")
    captured_at = capture.get("captured_at")
    harness_value = identity.get("harness")
    if harness_value:
        harness = harness_value
    elif saw_codex_metadata or isinstance(request.get("client_metadata"), Mapping):
        harness = "codex"
    else:
        harness = "unknown"

    if not session_id:
        _issue(issues, "missing_session_id", "session_id", "capture has no session id")
    if not thread_id:
        _issue(issues, "missing_thread_id", "request_body", "capture has no thread id")

    if transport_error:
        outcome = "transport_error"
        wire_complete = False
    elif outcome == "success" and any(
        issue.severity == Severity.ERROR for issue in issues
    ):
        outcome = "capture_invalid"

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
        operation="responses",
        outcome=outcome,
        captured_at=captured_at if isinstance(captured_at, str) else "",
        request_id=request_id if isinstance(request_id, str) else "",
        model=model,
        harness=harness,
        instructions=instructions,
        history=history,
        response=response,
        tools=tools,
        server_tool_calls=[*history_server_tools, *response_server_tools],
        agent_messages=[*history_agent_messages, *response_agent_messages],
        compaction_items=[*history_compaction_items, *response_compaction_items],
        termination="",
        wire_complete=wire_complete,
        issues=issues,
    )


# A readable alias for callers that dispatch by provider and operation.
normalize_responses_capture = parse_responses_capture


__all__ = [
    "ResponsesAdapterError",
    "normalize_responses_capture",
    "parse_responses_capture",
]
