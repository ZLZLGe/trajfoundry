"""Adapt TokenPlan feedback envelopes to the existing provider capture shape.

This module deliberately does not parse provider payloads or media.  It only
validates the TokenPlan envelope and projects its transport metadata into the
small capture contract consumed by provider adapters.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any, Literal


class TokenPlanError(ValueError):
    """A stable, source-specific TokenPlan envelope failure."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class AdaptedCapture:
    """A validated TokenPlan record projected into a provider capture.

    ``capture`` is suitable for the existing provider adapters.  The other
    attributes are source facts for the pipeline to use without re-opening the
    envelope.  ``has_media`` is intentionally only a routing signal: media is
    neither read nor copied here.
    """

    capture: dict[str, Any]
    source_name: Literal["tokenplan"]
    endpoint: str
    captured_at: str
    response_is_normalized_final: bool
    has_media: bool


_NORMALIZED_FINAL_FORMATS = {
    "/v1/chat/completions": "OPENAI_CHAT_COMPLETIONS_NORMALIZED",
    "/chat/completions": "OPENAI_CHAT_COMPLETIONS_NORMALIZED",
    "/v1/messages": "ANTHROPIC_MESSAGES_NORMALIZED",
    "/messages": "ANTHROPIC_MESSAGES_NORMALIZED",
    "/v1/responses": "OPENAI_RESPONSES_NORMALIZED",
    "/responses": "OPENAI_RESPONSES_NORMALIZED",
}

_PROTOCOL_ENDPOINTS = {
    "openai_chat": {"/v1/chat/completions", "/chat/completions"},
    "openai_responses": {"/v1/responses", "/responses"},
    "anthropic_messages": {"/v1/messages", "/messages"},
}


def _error(code: str, detail: str) -> TokenPlanError:
    return TokenPlanError(code, detail)


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("invalid_tokenplan_envelope", f"{path} must be an object")
    return value


def _required(mapping: Mapping[str, Any], key: str, *, path: str) -> Any:
    if key not in mapping:
        raise _error("invalid_tokenplan_envelope", f"{path}.{key} is required")
    return mapping[key]


def _optional_string(value: object, *, path: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _error("invalid_tokenplan_envelope", f"{path} must be a string")
    return value


def _timestamp(value: object, *, path: str) -> str:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or int(value) != value
    ):
        raise _error(
            "invalid_tokenplan_envelope", f"{path} must be an integer epoch-ms"
        )
    try:
        instant = datetime.fromtimestamp(int(value) / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise _error("invalid_tokenplan_envelope", f"{path} is out of range") from error
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _decode_identity(value: object) -> Mapping[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _first_identity(request_body: Mapping[str, Any], field: str) -> str:
    """Read provider request identity without changing the request body.

    This mirrors the semantic metadata containers used by the two existing
    adapters.  It is deliberately only used to decide whether TokenPlan's
    envelope identity is a fallback; the provider adapter remains the source
    of truth for identity extraction and conflict diagnostics.
    """

    direct: list[Mapping[str, Any]] = []
    encoded: list[Mapping[str, Any]] = []
    for key in ("client_metadata", "metadata"):
        value = request_body.get(key)
        if not isinstance(value, Mapping):
            continue
        direct.append(value)
        decoded = _decode_identity(value.get("x-codex-turn-metadata"))
        if decoded is not None:
            encoded.append(decoded)

    # Anthropic stores compatibility identity in metadata.user_id.  Its parser
    # gives explicit metadata fields precedence over this encoded copy.
    user_id = request_body.get("metadata")
    user_identity = (
        _decode_identity(user_id.get("user_id"))
        if isinstance(user_id, Mapping)
        else None
    )
    sources = [*direct, *encoded]
    if user_identity is not None:
        sources.append(user_identity)
    for source in sources:
        value = source.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _has_media(value: object) -> bool:
    if isinstance(value, Mapping):
        if isinstance(value.get("$media_ref"), str) and value["$media_ref"]:
            return True
        return any(_has_media(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_media(child) for child in value)
    if isinstance(value, str):
        return value.startswith("$media_ref:")
    return False


def _validate_capture_complete(capture: Mapping[str, Any], *, path: str) -> None:
    if capture.get("status") != "COMPLETE":
        raise _error(
            "invalid_tokenplan_envelope", f"{path}.status must equal 'COMPLETE'"
        )


def _validate_data_quality(metadata: Mapping[str, Any]) -> None:
    quality = _mapping(metadata.get("data_quality"), path="metadata.data_quality")
    expected = {
        "client_request_complete": True,
        "client_response_complete": True,
        "all_attachments_available": True,
        "truncated": False,
    }
    for key, required in expected.items():
        if quality.get(key) is not required:
            raise _error(
                "invalid_tokenplan_envelope",
                f"metadata.data_quality.{key} must be {required!r}",
            )


def adapt_tokenplan_envelope(value: object) -> AdaptedCapture:
    """Validate and adapt one TokenPlan ``data_feedback_des.v1`` record.

    Empty objects get a dedicated error code so callers can quarantine the
    known two-byte placeholder files distinctly from malformed envelopes.
    """

    root = _mapping(value, path="$")
    if not root:
        raise _error("tokenplan_empty_envelope", "envelope is an empty object")

    request = _mapping(
        _required(root, "client_request", path="$"), path="client_request"
    )
    response = _mapping(
        _required(root, "client_response", path="$"), path="client_response"
    )
    metadata = _mapping(_required(root, "metadata", path="$"), path="metadata")
    request_capture = _mapping(
        _required(request, "capture", path="client_request"),
        path="client_request.capture",
    )
    response_capture = _mapping(
        _required(response, "capture", path="client_response"),
        path="client_response.capture",
    )

    raw_endpoint = _required(request, "path", path="client_request")
    if not isinstance(raw_endpoint, str) or not raw_endpoint:
        raise _error(
            "invalid_tokenplan_envelope", "client_request.path must be a string"
        )
    endpoint = raw_endpoint.split("?", 1)[0].rstrip("/") or "/"
    protocol = _required(request, "protocol", path="client_request")
    allowed_endpoints = (
        _PROTOCOL_ENDPOINTS.get(protocol) if isinstance(protocol, str) else None
    )
    if allowed_endpoints is None or endpoint not in allowed_endpoints:
        raise _error(
            "invalid_tokenplan_envelope",
            "client_request.protocol/path is not a supported TokenPlan combination",
        )
    request_body = _required(request_capture, "body", path="client_request.capture")
    response_body = _required(response_capture, "body", path="client_response.capture")
    _validate_capture_complete(request_capture, path="client_request.capture")
    _validate_capture_complete(response_capture, path="client_response.capture")
    _validate_data_quality(metadata)
    http_status = _required(metadata, "http_status_code", path="metadata")
    if (
        not isinstance(http_status, int)
        or isinstance(http_status, bool)
        or http_status <= 0
    ):
        raise _error(
            "invalid_tokenplan_envelope",
            "metadata.http_status_code must be a positive integer",
        )

    completed_at = metadata.get("completed_at_ms")
    if completed_at is None:
        captured_at = _timestamp(
            _required(metadata, "received_at_ms", path="metadata"),
            path="metadata.received_at_ms",
        )
    else:
        captured_at = _timestamp(completed_at, path="metadata.completed_at_ms")

    request_id = _optional_string(
        metadata.get("request_id"), path="metadata.request_id"
    )
    envelope_session = _optional_string(
        metadata.get("session_id"), path="metadata.session_id"
    )
    envelope_task = _optional_string(metadata.get("task_id"), path="metadata.task_id")
    request_mapping = _mapping(request_body, path="client_request.capture.body")
    has_body_session = bool(_first_identity(request_mapping, "session_id"))
    has_body_turn = bool(_first_identity(request_mapping, "turn_id"))

    # Existing providers already consume these top-level fallbacks where
    # appropriate.  Do not inject or overwrite request metadata: it is actual
    # trajectory content, while TokenPlan identity is transport bookkeeping.
    capture: dict[str, Any] = {
        "path": raw_endpoint,
        "request_body": request_body,
        "response_body": response_body,
        "status_code": http_status,
        "request_id": request_id,
        "captured_at": captured_at,
        "is_stream": request.get("stream") is True,
        "response_is_normalized_final": response_capture.get("format")
        == _NORMALIZED_FINAL_FORMATS.get(endpoint),
    }
    if envelope_session and not has_body_session:
        capture["session_id"] = envelope_session
    if envelope_task and not has_body_turn:
        capture["turn_id"] = envelope_task

    return AdaptedCapture(
        capture=capture,
        source_name="tokenplan",
        endpoint=endpoint,
        captured_at=captured_at,
        response_is_normalized_final=bool(capture["response_is_normalized_final"]),
        has_media=bool(request.get("media"))
        or bool(response.get("media"))
        or _has_media(request_body)
        or _has_media(response_body),
    )


__all__ = ["AdaptedCapture", "TokenPlanError", "adapt_tokenplan_envelope"]
