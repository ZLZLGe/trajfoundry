"""Adapt TokenPlan feedback envelopes to the existing provider capture shape.

This module deliberately does not parse provider payload semantics or media
bytes.  It validates the TokenPlan envelope, extracts explicit media sentinels,
and projects transport metadata into the small capture contract consumed by
provider adapters.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from pathlib import PurePath
from typing import Any, Literal

from ..models import MediaMapping


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
    envelope.  Media bytes are never read or copied; only the stable object
    names referenced by the request/response bodies are retained.
    """

    capture: dict[str, Any]
    source_name: Literal["tokenplan"]
    endpoint: str
    captured_at: str
    response_is_normalized_final: bool
    has_media: bool
    multimodal_file_mapping: list[MediaMapping] = field(default_factory=list)


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


_MEDIA_REF = re.compile(r"\$media_ref:([^\s\"'<>},\]\)`]+)")
_STRUCTURED_MEDIA_REF = re.compile(
    r"(?P<key>[\"']?\$media_ref[\"']?)\s*:\s*"
    r"(?P<quote>[\"'])(?P<part_id>[^\"'\r\n]*)(?P=quote)"
)
_MEDIA_SENTINEL = "$media_ref:"


@dataclass(frozen=True)
class _MediaReference:
    """One explicit media sentinel found in a captured body."""

    part_id: str
    path: str


def _normalise_part_id(value: object, *, path: str) -> str:
    """Validate a structured media reference and return its part id."""

    if not isinstance(value, str):
        raise _error("invalid_tokenplan_envelope", f"{path} must be a string")
    part_id = value.removeprefix(_MEDIA_SENTINEL)
    if not part_id or any(character.isspace() for character in part_id):
        raise _error(
            "invalid_tokenplan_envelope", f"{path} must contain a non-empty part id"
        )
    return part_id


def _string_media_references(value: str, *, path: str) -> list[_MediaReference]:
    """Extract sentinels from a string, including sentinels in embedded JSON.

    The request/response bodies frequently contain JSON serialized inside a
    normal text field.  Scanning the string itself therefore preserves the
    provider payload while still finding the explicit TokenPlan sentinel.
    ``$media_ref:{part_id}`` is a documented template that appears in captured
    prompt text; it is not a concrete reference and is intentionally ignored.
    """

    # Most message strings are ordinary prose.  Avoid a JSON parse attempt when
    # there is no control marker at all; this matters for large code blocks.
    if "$media_ref" not in value:
        return []

    # A body can carry a JSON object as a text value.  Parsing only strings
    # that are themselves complete JSON objects/arrays lets structured refs
    # retain their true object order without changing the captured body.
    stripped = value.strip()
    if stripped[:1] in {"{", "["}:
        try:
            decoded = json.loads(stripped)
        except (TypeError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, (Mapping, list)):
            return _body_media_references(decoded, path=path)

    references: list[_MediaReference] = []
    structured_matches = list(_STRUCTURED_MEDIA_REF.finditer(value))
    events: list[tuple[int, int, str]] = []
    for match in structured_matches:
        raw_part_id = match.group("part_id")
        # Captured implementation docs occasionally show a quoted template
        # (`$media_ref:"+partID`).  It is prose, not a concrete part id.
        if not raw_part_id or raw_part_id.startswith("{") or raw_part_id == "+partID":
            if not raw_part_id:
                raise _error(
                    "invalid_tokenplan_envelope",
                    f"{path} contains an empty structured $media_ref sentinel",
                )
            continue
        part_id = _normalise_part_id(raw_part_id, path=path)
        events.append((match.start(), 0, part_id))

    matches = list(_MEDIA_REF.finditer(value))
    structured_spans = [match.span() for match in structured_matches]
    for match in matches:
        # A quoted structured value may itself contain the normal sentinel
        # spelling; let the structured event represent it once.
        if any(
            start <= match.start() and match.end() <= end
            for start, end in structured_spans
        ):
            continue
        raw_part_id = match.group(1).rstrip(
            ".,;:!?)]}`\uff0c\u3002\uff1b\uff1a\uff01\uff1f\uff09\u3011\u300b"
        )
        # Some captured system prompts describe the sentinel using a template.
        # Treat that prose as text, rather than turning every ordinary record
        # containing documentation into a malformed media capture.
        if raw_part_id.startswith("{"):
            continue
        if not raw_part_id:
            raise _error(
                "invalid_tokenplan_envelope",
                f"{path} contains an empty $media_ref sentinel",
            )
        events.append((match.start(), 1, raw_part_id))

    for _, _, part_id in sorted(events):
        references.append(_MediaReference(part_id=part_id, path=path))

    # An exact empty sentinel is malformed.  A sentinel-looking fragment inside
    # prose (for example a code sample containing ``$media_ref:"+partID``) is
    # left as text; it is not a concrete media reference.
    if value.strip() == _MEDIA_SENTINEL and not matches:
        raise _error(
            "invalid_tokenplan_envelope",
            f"{path} contains a malformed $media_ref sentinel",
        )
    return references


def _body_media_references(
    value: object, *, path: str = "body"
) -> list[_MediaReference]:
    """Walk a body in provider order and collect only explicit media refs.

    In addition to the normal string form (``$media_ref:media_0``), TokenPlan
    producers may emit a structured ``{"$media_ref": "media_0"}`` value.  The
    key itself is control syntax; all other strings are scanned literally.
    """

    references: list[_MediaReference] = []
    if isinstance(value, str):
        return _string_media_references(value, path=path)
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if key == "$media_ref":
                if isinstance(child, list):
                    for index, item in enumerate(child):
                        part_id = _normalise_part_id(item, path=f"{child_path}/{index}")
                        references.append(
                            _MediaReference(part_id=part_id, path=child_path)
                        )
                else:
                    part_id = _normalise_part_id(child, path=child_path)
                    references.append(_MediaReference(part_id=part_id, path=child_path))
                continue
            references.extend(_body_media_references(child, path=child_path))
        return references
    if isinstance(value, list):
        for index, child in enumerate(value):
            references.extend(_body_media_references(child, path=f"{path}/{index}"))
    return references


def _valid_object_name(value: object) -> str | None:
    """Return a safe storage basename, or ``None`` for malformed metadata."""

    if not isinstance(value, str) or not value or not value.strip():
        return None
    if (
        PurePath(value).name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or value in {".", ".."}
    ):
        return None
    return value


@dataclass
class _MediaMetadata:
    """Candidate metadata grouped by TokenPlan part id."""

    names: set[str]
    malformed: bool = False


def _media_metadata(
    request: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, _MediaMetadata]:
    """Collect usable and malformed media records without rejecting unused ones."""

    metadata: dict[str, _MediaMetadata] = {}
    for container in (request, response):
        records = container.get("media")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                # There is no part id to match, so this malformed unused
                # attachment can be ignored.  A referenced id will fail the
                # lookup below with a stable envelope error.
                continue
            part_id = record.get("part_id")
            if not isinstance(part_id, str) or not part_id:
                continue
            entry = metadata.setdefault(part_id, _MediaMetadata(names=set()))
            object_name = _valid_object_name(record.get("object_name"))
            if object_name is None:
                entry.malformed = True
                continue
            entry.names.add(object_name)
    return metadata


def _multimodal_file_mapping(
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    request_body: object,
    response_body: object,
) -> list[MediaMapping]:
    """Map body-used media parts to their stored TokenPlan object names.

    Only concrete sentinels in request/response bodies are emitted.  Available
    attachments that never occur in a body are deliberately omitted.  The
    first occurrence determines output order; repeated references are deduped.
    """

    references = [
        *_body_media_references(request_body, path="client_request.capture.body"),
        *_body_media_references(response_body, path="client_response.capture.body"),
    ]
    if not references:
        return []

    metadata = _media_metadata(request, response)
    mappings: list[MediaMapping] = []
    seen: set[str] = set()
    for reference in references:
        if reference.part_id in seen:
            continue
        seen.add(reference.part_id)
        entry = metadata.get(reference.part_id)
        if entry is None or entry.malformed or len(entry.names) != 1:
            if entry is None:
                detail = "has no media metadata"
            elif entry.malformed:
                detail = "has missing or invalid object_name metadata"
            else:
                detail = "has conflicting object_name metadata"
            raise _error(
                "invalid_tokenplan_envelope",
                f"{reference.path} references media part_id "
                f"{reference.part_id!r} which {detail}",
            )
        mappings.append(
            MediaMapping(part_id=reference.part_id, object_name=next(iter(entry.names)))
        )
    return mappings


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

    multimodal_file_mapping = _multimodal_file_mapping(
        request,
        response,
        request_body,
        response_body,
    )
    return AdaptedCapture(
        capture=capture,
        source_name="tokenplan",
        endpoint=endpoint,
        captured_at=captured_at,
        response_is_normalized_final=bool(capture["response_is_normalized_final"]),
        has_media=bool(request.get("media"))
        or bool(response.get("media"))
        or bool(multimodal_file_mapping),
        multimodal_file_mapping=multimodal_file_mapping,
    )


__all__ = ["AdaptedCapture", "TokenPlanError", "adapt_tokenplan_envelope"]
