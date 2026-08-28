"""Shared helpers for preserving and interpreting provider tool names."""

from __future__ import annotations


def qualify_tool_name(namespace: str | None, name: str) -> str:
    """Return a tool's full name without duplicating an existing namespace."""

    if not namespace or not name:
        return name
    prefix = f"{namespace}."
    return name if name.startswith(prefix) else f"{prefix}{name}"


def is_spawn_tool_name(name: str) -> bool:
    """Return whether a full tool name has spawn-agent semantics."""

    if name in {"Agent", "spawn_agent"}:
        return True
    namespace, separator, member = name.rpartition(".")
    return bool(namespace and separator and member == "spawn_agent")
