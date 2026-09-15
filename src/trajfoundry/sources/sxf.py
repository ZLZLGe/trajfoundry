"""Adapters for SXF JSONL capture envelopes.

SXF stores one raw API capture per JSONL row.  Responses and Chat Completions
responses are commonly encoded as Server-Sent Events (SSE) text, while the
normal provider adapters consume decoded event objects or a final completion
object.  This module deliberately keeps the transport adaptation separate from
the provider semantics so the existing prefix aggregation and quality gates
remain the single source of truth.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class SXFError(ValueError):
    """Raised when an SXF envelope or transport body cannot be adapted."""


_RAW_SCHEMA = "tap_raw_capture_record.v1"
_IDENTITY_HEADER_ALIASES = {
    "session-id": "session_id",
    "session_id": "session_id",
    "thread-id": "thread_id",
    "thread_id": "thread_id",
    "turn-id": "turn_id",
    "turn_id": "turn_id",
    "parent-thread-id": "parent_thread_id",
    "parent_thread_id": "parent_thread_id",
    "parent-turn-id": "parent_turn_id",
    "parent_turn_id": "parent_turn_id",
    "forked-from-thread-id": "forked_from_thread_id",
    "forked_from_thread_id": "forked_from_thread_id",
}
_PROMOTED_IDENTITY_FIELDS = (
    "session_id",
    "thread_id",
    "turn_id",
    "parent_thread_id",
    "parent_turn_id",
    "forked_from_thread_id",
)


def _header_scalar(value: Any) -> Any:
    """Unwrap the singleton-list representation used by SXF headers."""

    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        return value
    return value


def sanitize_identity_headers(value: Any) -> dict[str, Any]:
    """Keep only identity-related request headers and flatten their values.

    Header values in the raw SXF envelope are usually arrays.  Provider
    identity extraction expects semantic scalar values, and no other headers
    should be carried into the normalization pipeline.
    """

    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip().lower()
        if key == "x-codex-turn-metadata":
            result[key] = _header_scalar(raw_value)
            continue
        if key in {
            "x-codex-parent-thread-id",
            "x-openai-subagent",
            "x-claude-code-session-id",
        }:
            result[key] = _header_scalar(raw_value)
            continue
        canonical = _IDENTITY_HEADER_ALIASES.get(key)
        if canonical is not None:
            result[canonical] = _header_scalar(raw_value)
    return result


def _json_loads(value: str, *, path: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError) as error:
        raise SXFError(f"{path} is not valid JSON") from error


def _sse_field(line: str) -> tuple[str, str] | None:
    if line.startswith(":"):
        return None
    if ":" not in line:
        return (line, "")
    name, value = line.split(":", 1)
    value = value.removeprefix(" ")
    return name, value


def parse_sse_events(text: str, *, allow_done: bool = False) -> list[dict[str, Any]]:
    """Decode an SSE body into JSON event objects.

    SSE data may span multiple ``data:`` lines.  Comment-only keepalives and
    transport fields such as ``id``/``retry`` are ignored.  ``[DONE]`` is a
    Chat Completions marker and is not returned as an event.
    """

    if not isinstance(text, str):
        raise SXFError("SSE body must be text")

    events: list[dict[str, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            data_lines = []
            return
        payload = "\n".join(data_lines)
        data_lines = []
        current_event = event_name
        event_name = None
        if payload.strip() == "[DONE]":
            if not allow_done:
                raise SXFError("unexpected [DONE] marker in Responses SSE")
            return
        decoded = _json_loads(payload, path="response_body SSE data")
        values: list[Any]
        if isinstance(decoded, list):
            values = decoded
        else:
            values = [decoded]
        for value in values:
            if not isinstance(value, Mapping):
                raise SXFError("SSE data must decode to an object")
            event = dict(value)
            if current_event is not None:
                event_type = event.get("type")
                if event_type is not None and event_type != current_event:
                    raise SXFError(
                        "SSE event name does not match the event object's type"
                    )
                if event_type is None:
                    event["type"] = current_event
            events.append(event)

    # splitlines() handles CRLF, LF, and a final unterminated line without
    # materializing any additional copy of the response body.
    for line in text.splitlines():
        if line == "":
            flush()
            continue
        field = _sse_field(line)
        if field is None:
            continue
        name, value = field
        if name == "event":
            event_name = value
        elif name == "data":
            data_lines.append(value)
        # id/retry and unknown extension fields are transport metadata.
    flush()
    return events


def _looks_like_sse(text: str) -> bool:
    return any(line.startswith(("event:", "data:", ":")) for line in text.splitlines())


def _aggregate_chat_chunks(text: str) -> dict[str, Any]:
    events = parse_sse_events(text, allow_done=True)
    if not text.rstrip().endswith("[DONE]"):
        raise SXFError("Chat Completions SSE stream is missing [DONE]")
    chunks = [
        event for event in events if event.get("object") == "chat.completion.chunk"
    ]
    if not chunks:
        raise SXFError("Chat Completions SSE stream contains no completion chunks")

    first = chunks[0]
    choices: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        raw_choices = chunk.get("choices", [])
        if not isinstance(raw_choices, list):
            raise SXFError("Chat Completions chunk choices must be an array")
        for fallback_index, raw_choice in enumerate(raw_choices):
            if not isinstance(raw_choice, Mapping):
                raise SXFError("Chat Completions chunk choice must be an object")
            index = raw_choice.get("index", fallback_index)
            if not isinstance(index, int) or index < 0:
                raise SXFError(
                    "Chat Completions chunk choice index must be non-negative"
                )
            state = choices.setdefault(
                index,
                {
                    "role": "assistant",
                    "content": [],
                    "reasoning_content": [],
                    "reasoning": [],
                    "tool_calls": {},
                    "finish_reason": None,
                },
            )
            delta = raw_choice.get("delta", {})
            if not isinstance(delta, Mapping):
                raise SXFError("Chat Completions chunk delta must be an object")
            role = delta.get("role")
            if isinstance(role, str) and role:
                state["role"] = role
            for field in ("content", "reasoning_content", "reasoning"):
                value = delta.get(field)
                if isinstance(value, str):
                    state[field].append(value)
            raw_tool_calls = delta.get("tool_calls")
            if raw_tool_calls is not None:
                if not isinstance(raw_tool_calls, list):
                    raise SXFError("Chat Completions delta tool_calls must be an array")
                for call_fallback, raw_call in enumerate(raw_tool_calls):
                    if not isinstance(raw_call, Mapping):
                        raise SXFError(
                            "Chat Completions tool call delta must be an object"
                        )
                    call_index = raw_call.get("index", call_fallback)
                    if not isinstance(call_index, int) or call_index < 0:
                        raise SXFError(
                            "Chat Completions tool call index must be non-negative"
                        )
                    call = state["tool_calls"].setdefault(
                        call_index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    for key in ("id", "type"):
                        value = raw_call.get(key)
                        if isinstance(value, str) and value:
                            call[key] = value
                    function = raw_call.get("function")
                    if isinstance(function, Mapping):
                        name = function.get("name")
                        if isinstance(name, str):
                            call["function"]["name"] += name
                        arguments = function.get("arguments")
                        if isinstance(arguments, str):
                            call["function"]["arguments"] += arguments
            finish_reason = raw_choice.get("finish_reason")
            if finish_reason is not None:
                state["finish_reason"] = finish_reason

    final_choices: list[dict[str, Any]] = []
    for index in sorted(choices):
        state = choices[index]
        message: dict[str, Any] = {
            "role": state["role"],
            "content": "".join(state["content"]),
        }
        reasoning = "".join(state["reasoning_content"])
        if reasoning:
            message["reasoning_content"] = reasoning
        legacy_reasoning = "".join(state["reasoning"])
        if legacy_reasoning:
            message["reasoning"] = legacy_reasoning
        tool_calls = [
            call
            for _, call in sorted(state["tool_calls"].items())
            if call.get("id") or call.get("function", {}).get("name")
        ]
        if tool_calls:
            message["tool_calls"] = tool_calls
        final_choices.append(
            {
                "index": index,
                "message": message,
                "finish_reason": state["finish_reason"],
            }
        )

    result: dict[str, Any] = {
        "id": first.get("id", ""),
        "object": "chat.completion",
        "created": first.get("created", 0),
        "model": first.get("model", ""),
        "choices": final_choices,
    }
    for chunk in reversed(chunks):
        usage = chunk.get("usage")
        if isinstance(usage, Mapping):
            result["usage"] = dict(usage)
            break
    return result


def normalize_response_body(path: str, body: Any) -> Any:
    """Convert SXF raw response text to provider-parser input."""

    if not isinstance(body, str):
        return body
    if not _looks_like_sse(body):
        try:
            return _json_loads(body, path="response_body")
        except SXFError:
            # Let the SSE parser produce a more useful error for malformed
            # event framing; both outcomes are quarantined by the pipeline.
            return body
    endpoint = path.split("?", 1)[0].rstrip("/")
    if endpoint in {"/v1/chat/completions", "/chat/completions"}:
        return _aggregate_chat_chunks(body)
    return parse_sse_events(body)


def adapt_sxf_envelope(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Map either observed SXF envelope variant to a common capture shape."""

    if not isinstance(capture, Mapping):
        raise SXFError("SXF capture root must be an object")

    if capture.get("schema") == _RAW_SCHEMA:
        result = dict(capture)
        started = capture.get("started_at")
        completed = capture.get("completed_at")
        result["captured_at"] = (
            started
            if isinstance(started, str)
            else completed
            if isinstance(completed, str)
            else ""
        )
        capture_id = capture.get("capture_id")
        if isinstance(capture_id, str) and capture_id:
            result["request_id"] = capture_id
        identity_headers = sanitize_identity_headers(capture.get("request_headers"))
        result["request_headers"] = identity_headers
        # The provider adapters intentionally read only a small, explicit
        # request-header allowlist.  SXF's Session-Id/Thread-Id headers are
        # semantic identity, not credentials, so expose scalar values at the
        # capture boundary where every provider can apply the normal fallback
        # and conflict checks.  Multi-valued headers remain unpromoted rather
        # than guessing which identity is authoritative.
        for field in _PROMOTED_IDENTITY_FIELDS:
            value = identity_headers.get(field)
            if isinstance(value, str) and value:
                result.setdefault(field, value)
    elif isinstance(capture.get("capture_meta"), Mapping):
        meta = dict(capture["capture_meta"])
        result = {
            "path": capture.get("path", ""),
            "method": capture.get("method", "POST"),
            "model": capture.get("model", ""),
            "request_body": capture.get("request_body"),
            "response_body": capture.get("response_body"),
            "status_code": meta.get("status_code"),
            "captured_at": capture.get("created_at", ""),
            "session_id": capture.get("conversation_session_id", ""),
            "user_id": capture.get("user_id", ""),
            "request_headers": {},
        }
        agent_kind = meta.get("agent_kind")
        if isinstance(agent_kind, str) and agent_kind:
            request = result.get("request_body")
            if isinstance(request, Mapping):
                request_copy = dict(request)
                metadata = request_copy.get("metadata")
                metadata_copy = dict(metadata) if isinstance(metadata, Mapping) else {}
                metadata_copy.setdefault("harness", agent_kind)
                request_copy["metadata"] = metadata_copy
                result["request_body"] = request_copy
    else:
        raise SXFError("unrecognized SXF envelope")

    endpoint = result.get("path")
    if not isinstance(endpoint, str) or not endpoint:
        raise SXFError("SXF envelope has no API path")
    result["response_body"] = normalize_response_body(
        endpoint, result.get("response_body")
    )
    return result


__all__ = [
    "SXFError",
    "adapt_sxf_envelope",
    "normalize_response_body",
    "parse_sse_events",
    "sanitize_identity_headers",
]
