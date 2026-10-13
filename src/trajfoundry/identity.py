"""Shared rules for selecting real session identities from capture metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def normalize_session_id(value: object) -> str:
    """Treat the public missing-session sentinel as missing at input boundaries."""

    return value if isinstance(value, str) and value != "no_session_id" else ""


def request_session_id(request: Mapping[str, Any]) -> str:
    """Select a real request session before any transport-level fallback.

    Explicit metadata takes precedence over encoded compatibility copies, but
    an empty value or ``no_session_id`` cannot hide a real session in either.
    """

    direct: list[Mapping[str, Any]] = []
    encoded: list[Mapping[str, Any]] = []

    def decoded(value: object) -> Mapping[str, Any] | None:
        if isinstance(value, Mapping):
            return value
        if not isinstance(value, str):
            return None
        try:
            result = json.loads(value)
        except (TypeError, ValueError):
            return None
        return result if isinstance(result, Mapping) else None

    for key in ("client_metadata", "metadata"):
        container = request.get(key)
        if not isinstance(container, Mapping):
            continue
        direct.append(container)
        identity = decoded(container.get("x-codex-turn-metadata"))
        if identity is not None:
            encoded.append(identity)
    metadata = request.get("metadata")
    if isinstance(metadata, Mapping):
        # Unlike turn metadata, this compatibility field is a JSON string,
        # not an object. Match provider parsing so malformed values cannot
        # suppress a valid transport fallback.
        encoded_user = metadata.get("user_id")
        identity = decoded(encoded_user) if isinstance(encoded_user, str) else None
        if identity is not None:
            encoded.append(identity)
    for container in (*direct, *encoded):
        value = normalize_session_id(container.get("session_id"))
        if value:
            return value
    return ""
