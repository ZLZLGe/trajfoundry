"""Adapt Deep Infra request archives to the provider capture contract."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .sxf import SXFError, normalize_response_body, sanitize_identity_headers


class DeepInfraError(ValueError):
    """A stable, source-specific Deep Infra envelope failure."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


_INVALID = "invalid_deepinfra_envelope"
_INCOMPLETE = "deepinfra_incomplete_envelope"
_SUPPORTED_ENDPOINTS = {
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/responses",
    "/responses",
    "/v1/messages",
    "/messages",
    "/v1/messages/count_tokens",
    "/messages/count_tokens",
}


def _error(code: str, detail: str) -> DeepInfraError:
    return DeepInfraError(code, detail)


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(_INVALID, f"{path} must be an object")
    return value


def _required_mapping(
    value: Mapping[str, Any], key: str, *, path: str
) -> Mapping[str, Any]:
    if key not in value:
        raise _error(_INVALID, f"{path}.{key} is required")
    return _mapping(value[key], path=f"{path}.{key}")


def _required_text(value: Mapping[str, Any], key: str, *, path: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise _error(_INVALID, f"{path}.{key} must be a non-empty string")
    return item


def _body_text(
    value: Mapping[str, Any], *, path: str, allow_transport_missing: bool = False
) -> str:
    truncated = value.get("body_truncated", False)
    if not isinstance(truncated, bool):
        raise _error(_INVALID, f"{path}.body_truncated must be a boolean")
    if truncated:
        if allow_transport_missing:
            body = value.get("body")
            if isinstance(body, str):
                return body
            return ""
        raise _error(_INCOMPLETE, f"{path}.body is truncated")

    if "body" not in value or value.get("body") is None:
        if allow_transport_missing:
            return ""
        raise _error(_INCOMPLETE, f"{path}.body is missing")
    body = value["body"]
    if not isinstance(body, str):
        raise _error(_INVALID, f"{path}.body must be a string")
    if not body.strip():
        if allow_transport_missing:
            return ""
        raise _error(_INCOMPLETE, f"{path}.body is missing")
    return body


def _decoded_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


_IDENTITY_ALIASES = {
    "session_id": ("session_id", "sessionId"),
    "thread_id": ("thread_id", "threadId"),
}

# The archive redaction layer replaces sensitive identity values with this
# marker.  It is not a usable conversation identity: treating the marker as a
# real session would merge unrelated captures and it cannot be used in the
# flat output filename.  Keep the value out of the projected capture instead.
_REDACTED_IDENTITY_VALUES = frozenset(
    {"<SENSITIVE>", "<REDACTED>", "[REDACTED]", "***"}
)
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def _identity_text(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    value = value.strip()
    if not value or value.upper() in _REDACTED_IDENTITY_VALUES:
        return ""
    return value


def _session_identity_text(value: object) -> str:
    candidate = _identity_text(value)
    return candidate if candidate and _SAFE_SESSION_ID.fullmatch(candidate) else ""


def _identity_value(source: Mapping[str, Any], field: str) -> str:
    for alias in _IDENTITY_ALIASES[field]:
        value = (
            _session_identity_text(source.get(alias))
            if field == "session_id"
            else _identity_text(source.get(alias))
        )
        if value:
            return value
    return ""


def _clean_identity_container(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove redacted identity values from one provider metadata object."""

    result = dict(value)
    for field in (
        "session_id",
        "sessionId",
        "thread_id",
        "threadId",
        "turn_id",
        "turnId",
        "parent_thread_id",
        "parentThreadId",
        "parent_turn_id",
        "parentTurnId",
        "forked_from_thread_id",
        "forkedFromThreadId",
        "user_id",
        "userId",
    ):
        raw = result.get(field)
        identity_text = (
            _session_identity_text(raw)
            if field in {"session_id", "sessionId"}
            else _identity_text(raw)
        )
        if isinstance(raw, str) and raw and not identity_text:
            result.pop(field, None)

    for field in ("user_id", "x-codex-turn-metadata"):
        raw = result.get(field)
        decoded = _decoded_mapping(raw)
        if decoded is None:
            continue
        cleaned = _clean_identity_container(decoded)
        if cleaned != dict(decoded):
            result[field] = json.dumps(
                cleaned, ensure_ascii=False, separators=(",", ":")
            )
    return result


def _clean_request_identity(request_body: Mapping[str, Any]) -> dict[str, Any]:
    """Project request metadata without archive redaction placeholders.

    Provider adapters inspect the request body again after this source adapter
    returns. Cleaning only the projected capture identity would let a nested
    ``metadata.user_id`` value reintroduce the placeholder later.
    """

    result = dict(request_body)
    for field in ("client_metadata", "metadata"):
        container = result.get(field)
        if isinstance(container, Mapping):
            result[field] = _clean_identity_container(container)
    return result


def _redacted_identity_fields(value: object) -> set[str]:
    """Return semantic identity fields replaced by the archive redactor."""

    if not isinstance(value, Mapping):
        return set()
    fields: set[str] = set()
    aliases = {
        "session_id": "session_id",
        "sessionid": "session_id",
        "session-id": "session_id",
        "x-session-id": "session_id",
        "thread_id": "thread_id",
        "threadid": "thread_id",
        "thread-id": "thread_id",
        "x-thread-id": "thread_id",
        "turn_id": "turn_id",
        "turnid": "turn_id",
        "turn-id": "turn_id",
        "user_id": "user_id",
        "userid": "user_id",
        "user-id": "user_id",
        "x-user-id": "user_id",
        "parent_thread_id": "parent_thread_id",
        "parentthreadid": "parent_thread_id",
        "parent-thread-id": "parent_thread_id",
        "x-parent-session-id": "parent_thread_id",
        "parent_turn_id": "parent_turn_id",
        "parentturnid": "parent_turn_id",
        "parent-turn-id": "parent_turn_id",
    }
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip().lower()
        field = aliases.get(key)
        if field and isinstance(raw_value, str) and raw_value:
            identity_text = (
                _session_identity_text(raw_value)
                if field == "session_id"
                else _identity_text(raw_value)
            )
            if not identity_text:
                fields.add(field)
        if key in {
            "client_metadata",
            "metadata",
            "user_id",
            "x-codex-turn-metadata",
        }:
            decoded = _decoded_mapping(raw_value)
            if decoded is not None:
                fields.update(_redacted_identity_fields(decoded))
    return fields


def _body_identity(request_body: Mapping[str, Any], field: str) -> str:
    """Read only identity containers understood by provider adapters."""

    direct: list[Mapping[str, Any]] = []
    encoded: list[Mapping[str, Any]] = []
    for key in ("client_metadata", "metadata"):
        value = request_body.get(key)
        if not isinstance(value, Mapping):
            continue
        direct.append(value)
        turn_metadata = _decoded_mapping(value.get("x-codex-turn-metadata"))
        if turn_metadata is not None:
            encoded.append(turn_metadata)

    # Anthropic clients may serialize their semantic metadata inside
    # metadata.user_id.  The provider parser gives direct fields precedence.
    metadata = request_body.get("metadata")
    if isinstance(metadata, Mapping):
        user_metadata = _decoded_mapping(metadata.get("user_id"))
        if user_metadata is not None:
            encoded.append(user_metadata)

    for source in (*direct, *encoded):
        if value := _identity_value(source, field):
            return value
    return ""


def _header_scalar(value: object) -> object:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _identity_headers(value: object) -> tuple[dict[str, Any], str]:
    """Sanitize headers and project Deep Infra's additional identity aliases."""

    result = sanitize_identity_headers(value)
    for key in list(result):
        if key == "x-codex-turn-metadata":
            continue
        raw = result[key]
        identity_text = (
            _session_identity_text(raw)
            if key in {"session_id", "x-claude-code-session-id"}
            else _identity_text(raw)
        )
        if isinstance(raw, str) and raw and not identity_text:
            result.pop(key, None)
    encoded_headers = _decoded_mapping(result.get("x-codex-turn-metadata"))
    if encoded_headers is not None:
        cleaned_headers = _clean_identity_container(encoded_headers)
        if cleaned_headers != dict(encoded_headers):
            result["x-codex-turn-metadata"] = json.dumps(
                cleaned_headers, ensure_ascii=False, separators=(",", ":")
            )
    user_id = ""
    if not isinstance(value, Mapping):
        return result, user_id

    for raw_key, raw_value in value.items():
        key = str(raw_key).strip().lower()
        scalar = _header_scalar(raw_value)
        if key == "x-session-id":
            if identity := _session_identity_text(scalar):
                result.setdefault("session_id", identity)
        elif key == "x-thread-id":
            if identity := _identity_text(scalar):
                result.setdefault("thread_id", identity)
        elif key in {
            "x-claude-code-agent-id",
            "x-claude-code-session-id",
        }:
            # These are semantic routing identities, not credentials.  Keep
            # them only after the generic identity allowlist has removed all
            # unrelated transport headers.
            identity = (
                _session_identity_text(scalar)
                if key == "x-claude-code-session-id"
                else _identity_text(scalar)
            )
            if not identity:
                continue
            result[key] = identity
            if key == "x-claude-code-session-id":
                result.setdefault("session_id", identity)
            else:
                result.setdefault("thread_id", identity)
        elif key == "x-parent-session-id":
            if identity := _identity_text(scalar):
                result.setdefault("parent_thread_id", identity)
        elif key == "x-deepseek-harness-session-id":
            if identity := _session_identity_text(scalar):
                result.setdefault("session_id", identity)
        elif key == "x-deepseek-harness-user-id":
            user_id = _identity_text(scalar)
        elif key == "x-user-id":
            user_id = user_id or _identity_text(scalar)
    return result, user_id


def _header_identity(headers: Mapping[str, Any], field: str) -> str:
    value = (
        _session_identity_text(headers.get(field))
        if field == "session_id"
        else _identity_text(headers.get(field))
    )
    if value:
        return value
    if field == "session_id":
        for key in (
            "x-claude-code-session-id",
            "x-deepseek-harness-session-id",
        ):
            if semantic_session := _session_identity_text(headers.get(key)):
                return semantic_session
    if field == "thread_id" and (
        claude_agent := _identity_text(headers.get("x-claude-code-agent-id"))
    ):
        return claude_agent
    turn_metadata = _decoded_mapping(headers.get("x-codex-turn-metadata"))
    return _identity_value(turn_metadata, field) if turn_metadata is not None else ""


def adapt_deepinfra_envelope(value: object) -> dict[str, Any]:
    """Validate and project one Deep Infra archive record.

    Request metadata remains inside the provider body and is authoritative.
    Sanitized transport identity is used only as a fallback.  A request id is
    retained only as request identity; it must never stand in for a missing
    semantic session.
    """

    root = _mapping(value, path="$")
    request = _required_mapping(root, "request", path="$")
    response = _required_mapping(root, "response", path="$")
    _required_mapping(root, "access_log", path="$")

    request_id = _required_text(root, "request_id", path="$")
    captured_at = _required_text(root, "request_time", path="$")
    path = _required_text(request, "path", path="$.request")
    endpoint = path.split("?", 1)[0].rstrip("/") or "/"
    if endpoint not in _SUPPORTED_ENDPOINTS:
        raise _error(
            _INVALID,
            "$.request.path is not a supported provider endpoint",
        )

    request_text = _body_text(request, path="$.request")
    try:
        request_body = json.loads(request_text)
    except (TypeError, ValueError) as error:
        raise _error(_INVALID, "$.request.body is not valid JSON") from error
    if not isinstance(request_body, Mapping):
        raise _error(_INVALID, "$.request.body must decode to an object")
    masked_identity_fields = _redacted_identity_fields(request_body)
    request_body = _clean_request_identity(request_body)

    status_code = response.get("status_code")
    if (
        not isinstance(status_code, int)
        or isinstance(status_code, bool)
        or status_code < 0
    ):
        raise _error(_INVALID, "$.response.status_code must be a non-negative integer")

    transport_or_api_error = status_code == 0 or not 200 <= status_code < 300
    response_text = _body_text(
        response,
        path="$.response",
        allow_transport_missing=transport_or_api_error,
    )

    raw_headers = request.get("headers", {})
    if raw_headers is None:
        raw_headers = {}
    if not isinstance(raw_headers, Mapping):
        raise _error(_INVALID, "$.request.headers must be an object")
    masked_identity_fields.update(_redacted_identity_fields(raw_headers))
    request_headers, header_user_id = _identity_headers(raw_headers)

    try:
        response_body = (
            {}
            if not response_text.strip() and transport_or_api_error
            else normalize_response_body(path, response_text)
        )
    except SXFError as error:
        if not transport_or_api_error:
            raise _error(
                _INVALID, "$.response.body is not valid provider SSE"
            ) from error
        # Provider adapters retain the transport/API disposition even when an
        # upstream gateway returned a non-JSON error body.
        response_body = response_text
    if not isinstance(response_body, (Mapping, list)) and not transport_or_api_error:
        raise _error(_INVALID, "$.response.body is not valid JSON or SSE")

    capture: dict[str, Any] = {
        "path": path,
        "request_body": request_body,
        "response_body": response_body,
        "status_code": status_code,
        "captured_at": captured_at,
        "request_id": request_id,
        "request_headers": request_headers,
        "is_stream": request_body.get("stream") is True,
    }
    if masked_identity_fields:
        capture["_masked_identity_fields"] = sorted(masked_identity_fields)

    body_session = _body_identity(request_body, "session_id")
    header_session = _header_identity(request_headers, "session_id")
    if not body_session and header_session:
        capture["session_id"] = header_session

    body_thread = _body_identity(request_body, "thread_id")
    header_thread = _header_identity(request_headers, "thread_id")
    if not body_thread and header_thread:
        capture["thread_id"] = header_thread
    parent_thread = request_headers.get("parent_thread_id")
    if isinstance(parent_thread, str) and parent_thread:
        capture["parent_thread_id"] = parent_thread
    if header_user_id:
        capture["user_id"] = header_user_id
    return capture


__all__ = ["DeepInfraError", "adapt_deepinfra_envelope"]
