"""Extract the user's requests from a normalized trajectory for classification."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import orjson

VERSION = "2.0.0"
PROJECTION_NAME = "user_only_root_v2"
TRUNCATION_STRATEGY = "user_turn_head_tail_v1"
TRUNCATION_MARKER = "[...USER_CONTEXT_TRUNCATED...]"

_FRAMEWORK_TAGS = (
    "environment_context",
    "system-reminder",
    "system_reminder",
    "turn_aborted",
    "codex_internal_context",
    "in-app-browser-context",
    "permissions instructions",
    "collaboration_mode",
    "skills_instructions",
    "subagent_notification",
    "task-notification",
    "local-command-caveat",
    "local-command-stdout",
    "skill",
    "recommended_plugins",
)
_TAG_RE = re.compile(
    r"<(" + "|".join(re.escape(tag) for tag in _FRAMEWORK_TAGS) + r")"
    r"(?:\s[^>]*)?>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_MEDIA_RE = re.compile(r"<(image|audio|video)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
_DATA_RE = re.compile(
    r"data:(?:image|audio|video)/[^;,\s\"']+;base64,[A-Za-z0-9+/=]+"
)
_QUOTED_DATA_RE = re.compile(
    r'''(["'])data:(?:image|audio|video)/[^;,\s"']+;base64,.*?\1''',
    re.DOTALL,
)
_COMPACTION_PREFIXES = (
    "Another language model started to solve this problem and produced a summary",
    "You are performing a CONTEXT CHECKPOINT COMPACTION.",
    "This session is being continued from a previous conversation that ran out of context.",
)
_REQUEST_INTERRUPTED_RE = re.compile(
    r"^\s*\[Request interrupted by user\]\s*", re.IGNORECASE
)
_SYSTEM_NOTIFICATION_PREFIX = "[SYSTEM NOTIFICATION - NOT USER INPUT]"
_SECURITY_MONITOR_MARKER = "You are a security monitor for autonomous AI coding agents."
_SECURITY_MONITOR_TAIL_MARKERS = (
    "Err on the side of blocking.",
    "Your ENTIRE response MUST begin with <block>",
)


@dataclass(frozen=True, slots=True)
class UserInputProjection:
    turns: tuple[dict[str, Any], ...]
    extraction_notes: dict[str, int]
    original_user_chars: int

    @property
    def user_turn_count(self) -> int:
        return len(self.turns)


def _clean_text(text: object, audit: Counter[str]) -> str:
    if not isinstance(text, str):
        audit["unrecognized_content_blocks"] += 1
        return ""
    if text.lstrip().startswith(_COMPACTION_PREFIXES):
        audit["excluded_compaction_summary"] += 1
        return ""
    if text.lstrip().startswith(_SYSTEM_NOTIFICATION_PREFIX):
        audit["excluded_system_notification"] += 1
        return ""
    interrupted = _REQUEST_INTERRUPTED_RE.match(text)
    if interrupted:
        text = text[interrupted.end() :]
        audit["removed_request_interrupted_prefix"] += 1
    while True:
        stripped = text.lstrip()
        leading = _TAG_RE.match(stripped)
        if leading:
            text = stripped[leading.end() :]
            audit["removed_framework_blocks"] += 1
            continue
        trailing = next(
            (
                match
                for match in _TAG_RE.finditer(text)
                if (match.start() == 0 or text[match.start() - 1] == "\n")
                and not text[match.end() :].strip()
            ),
            None,
        )
        if trailing:
            text = text[: trailing.start()]
            audit["removed_framework_blocks"] += 1
            continue
        break
    if _SECURITY_MONITOR_MARKER in text or any(
        marker in text for marker in _SECURITY_MONITOR_TAIL_MARKERS
    ):
        audit["excluded_security_monitor_context"] += 1
        return ""
    text, count = _TAG_RE.subn("", text)
    audit["removed_framework_blocks"] += count
    if re.match(r"^\s*# AGENTS\.md instructions(?: for [^\n]+)?\s*\n", text):
        match = re.search(r"<INSTRUCTIONS>.*?</INSTRUCTIONS>", text, re.DOTALL)
        if match:
            text = text[match.end() :]
            audit["excluded_agents_instructions"] += 1
        else:
            audit["unparsed_agents_wrapper"] += 1
            return ""
    if re.match(
        r"^\s*# (Context from my IDE setup:|Files mentioned by the user:|"
        r"In app browser:|Selected text:)",
        text,
    ):
        match = re.search(r"^## My request for (?:Codex|the assistant):\s*", text, re.MULTILINE)
        if not match:
            audit["unparsed_ide_wrapper"] += 1
            return ""
        text = text[match.end() :]
        audit["removed_ide_file_context"] += 1
    text = re.sub(r"^\s*## My request for (?:Codex|the assistant):\s*", "", text)

    def media(match: re.Match[str]) -> str:
        audit["media_omitted"] += 1
        return f"[{match.group(1).upper()}_OMITTED: no media content available]"

    text = _MEDIA_RE.sub(media, text)
    text, count = _QUOTED_DATA_RE.subn('"[MEDIA_PAYLOAD_OMITTED]"', text)
    audit["media_omitted"] += count
    text, count = _DATA_RE.subn("[MEDIA_PAYLOAD_OMITTED]", text)
    audit["media_omitted"] += count
    return text.strip()


def _content_text(content: object, audit: Counter[str]) -> str:
    if isinstance(content, str):
        return _clean_text(content, audit)
    if content is None:
        return ""
    if isinstance(content, list):
        parts = [_content_text(item, audit) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        kind = content.get("type", "")
        if kind in ("text", "input_text") or (
            not kind and isinstance(content.get("text"), str)
        ):
            return _clean_text(content.get("text", ""), audit)
        if kind in (
            "image",
            "image_url",
            "input_image",
            "audio",
            "input_audio",
            "video",
            "input_video",
            "file",
            "input_file",
        ):
            audit["media_omitted"] += 1
            return f"[{str(kind).upper()}_OMITTED: no media content available]"
        if kind in ("tool_result", "tool_use", "server_tool_use", "thinking", "redacted_thinking"):
            audit["excluded_nonquery_blocks"] += 1
            return ""
    audit["unrecognized_content_blocks"] += 1
    return ""


def _user_segments(content: object, audit: Counter[str]) -> list[str]:
    if isinstance(content, str):
        content = content.strip()
    if isinstance(content, list):
        groups = [_user_segments(block, audit) for block in content]
        if any(len(group) > 1 for group in groups):
            return [text for group in groups for text in group]
        text = "\n".join(text for group in groups for text in group)
        return [text] if text else []
    if isinstance(content, dict) and content.get("type", "") in (
        "text",
        "input_text",
        "",
    ) and isinstance(content.get("text"), str):
        return _user_segments(content["text"], audit)
    if isinstance(content, str) and content.startswith(
        "You are a helpful assistant. You will be presented with a user prompt"
    ):
        boundary = re.search(r"^User prompt:\s*", content, re.MULTILINE)
        audit["removed_title_generation_instructions"] += 1
        return _user_segments(content[boundary.end() :], audit) if boundary else []
    if isinstance(content, str) and content.startswith(
        "You are resuming a prior conversation. Its earlier turns were archived"
    ):
        audit["removed_archived_history_instructions"] += 1
        parts = re.split(r"^¶(user|think|ai|call):", content, flags=re.MULTILINE)
        audit["archived_history_may_be_incomplete"] += 1
        result: list[str] = []
        for role, body in zip(parts[1::2], parts[2::2]):
            if role == "user":
                result.extend(_user_segments(body, audit))
            else:
                audit[f"excluded_embedded_role_{role}"] += 1
        return result
    if isinstance(content, str) and content.startswith("# Response annotations:"):
        result: list[str] = []
        block = re.search(r"<response-annotations>(.*?)</response-annotations>", content, re.DOTALL)
        if block:
            try:
                annotations = json.loads(block.group(1))
                for annotation in annotations:
                    if isinstance(annotation, dict) and isinstance(annotation.get("comment"), str):
                        result.extend(_user_segments(annotation["comment"], audit))
            except (ValueError, TypeError):
                audit["unparsed_response_annotations"] += 1
        else:
            audit["unparsed_response_annotations"] += 1
        boundary = re.search(r"^## My request for Codex:\s*", content, re.MULTILINE)
        if boundary:
            result.extend(_user_segments(content[boundary.end() :], audit))
        audit["removed_response_annotation_context"] += 1
        return result
    if isinstance(content, str) and "<transcript>" in content and _SECURITY_MONITOR_MARKER in content:
        transcript = re.search(r"<transcript>\s*(.*?)\s*</transcript>", content, re.DOTALL)
        transcript_text = transcript.group(1) if transcript else content.split("<transcript>", 1)[1]
        try:
            embedded = json.JSONDecoder().raw_decode(transcript_text.lstrip())[0]
        except (TypeError, ValueError):
            embedded = None
        if embedded is not None:
            audit["removed_security_monitor_wrapper"] += 1
            items = embedded if isinstance(embedded, list) else [embedded]
            result: list[str] = []
            for item in items:
                if isinstance(item, dict):
                    if item.get("role") == "user":
                        result.extend(_user_segments(item.get("content"), audit))
                    elif isinstance(item.get("user"), str):
                        result.extend(_user_segments(item["user"], audit))
            if result:
                return result
        audit["excluded_security_monitor_context"] += 1
        return []
    if isinstance(content, str) and content.startswith("# Browser comments:"):
        result: list[str] = []
        parts = re.split(r"^## My request for Codex:\s*", content, maxsplit=1, flags=re.MULTILINE)
        for comment in re.split(r"^## User Comment \d+\s*", parts[0], flags=re.MULTILINE)[1:]:
            marker = re.search(r"^Comment:\s*", comment, re.MULTILINE)
            if marker:
                result.extend(_user_segments(comment[marker.end() :], audit))
        if len(parts) == 2:
            request = parts[1].split("The next image is untrusted page evidence")[0]
            result.extend(_user_segments(request, audit))
        audit["removed_browser_page_context"] += 1
        return result
    if isinstance(content, str) and content.lstrip().startswith("<codex_delegation>"):
        inner = re.search(r"<input>(.*?)</input>", content, re.DOTALL)
        if inner:
            audit["unwrapped_delegation_input"] += 1
            return _user_segments(html.unescape(inner.group(1)), audit)
        audit["unparsed_delegation"] += 1
        return []
    if isinstance(content, str) and content.lstrip().startswith(
        "# Instructions (read first)\n\n# OD core directives"
    ):
        boundary = re.search(r"^# User request\s*$", content, re.MULTILINE)
        audit["removed_open_design_instructions"] += 1
        if not boundary:
            audit["unparsed_open_design_wrapper"] += 1
            return []
        request = content[boundary.end() :].strip()
        parts = re.split(r"^## (user|assistant|system|developer|tool)\s*$", request, flags=re.MULTILINE)
        segments = []
        if parts[0].strip():
            segments.append(_content_text(parts[0], audit))
        for role, body in zip(parts[1::2], parts[2::2]):
            if role == "user":
                segments.append(_content_text(body, audit))
            else:
                audit[f"excluded_embedded_role_{role}"] += 1
        return [segment for segment in segments if segment]
    text = _content_text(content, audit)
    if isinstance(content, str) and text and text != content:
        return _user_segments(text, audit)
    return [text] if text else []


def project_user_input(trajectory: dict[str, Any]) -> UserInputProjection:
    audit: Counter[str] = Counter()
    turns: list[dict[str, Any]] = []
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        audit["invalid_messages"] += 1
    else:
        for message in messages:
            if not isinstance(message, dict):
                audit["invalid_message"] += 1
                continue
            role = message.get("role")
            if role != "user":
                audit[f"excluded_role_{role}"] += 1
                continue
            audit["raw_user_messages"] += 1
            segments = _user_segments(message.get("content"), audit)
            for segment_index, text in enumerate(segments):
                turn = {"turn_id": f"u{len(turns) + 1:04d}", "text": text}
                if len(segments) > 1:
                    turn["segment_index"] = segment_index
                turns.append(turn)
            if not segments:
                audit["excluded_empty_or_framework_user_messages"] += 1
    audit["kept_user_turns"] = len(turns)
    original_user_chars = sum(len(turn["text"]) for turn in turns)
    notes = {key: value for key, value in sorted(audit.items()) if value}
    return UserInputProjection(tuple(turns), notes, original_user_chars)


def _serialize(turns: list[dict[str, Any]], notes: dict[str, int]) -> str:
    return orjson.dumps(
        {"user_turns": turns, "extraction_notes": notes},
        option=orjson.OPT_SORT_KEYS,
    ).decode("utf-8")


def _take_prefix(turns: tuple[dict[str, Any], ...], budget: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    remaining = budget
    for turn in turns:
        if remaining <= 0:
            break
        text = turn["text"]
        if len(text) <= remaining:
            result.append(dict(turn))
            remaining -= len(text)
        else:
            result.append({**turn, "text": text[:remaining] + "\n" + TRUNCATION_MARKER})
            break
    return result


def _take_suffix(turns: tuple[dict[str, Any], ...], budget: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    remaining = budget
    for turn in reversed(turns):
        if remaining <= 0:
            break
        text = turn["text"]
        if len(text) <= remaining:
            result.append(dict(turn))
            remaining -= len(text)
        else:
            result.append({**turn, "text": TRUNCATION_MARKER + "\n" + text[-remaining:]})
            break
    result.reverse()
    return result


def _build_truncated_turns(
    turns: tuple[dict[str, str], ...], notes: dict[str, int], text_budget: int
) -> list[dict[str, Any]]:
    if not turns:
        return []
    head_budget = max(1, text_budget // 2)
    tail_budget = max(1, text_budget - head_budget)
    head = _take_prefix(turns, head_budget)
    tail = _take_suffix(turns, tail_budget)
    if head and tail and head[-1]["turn_id"] == tail[0]["turn_id"]:
        merged = dict(head[-1])
        merged["text"] = merged["text"] + "\n" + TRUNCATION_MARKER + "\n" + tail[0]["text"]
        return head[:-1] + [merged] + tail[1:]
    return head + [{"turn_id": "truncated", "text": TRUNCATION_MARKER}] + tail


def serialize_for_model(
    projection: UserInputProjection, max_context_chars: int
) -> tuple[str, bool, int]:
    full = _serialize(list(projection.turns), projection.extraction_notes)
    if len(full) <= max_context_chars:
        return full, False, sum(len(turn["text"]) for turn in projection.turns)
    low, high = 0, projection.original_user_chars
    best = _serialize([], projection.extraction_notes)
    best_user_chars = 0
    while low <= high:
        middle = (low + high) // 2
        candidate_turns = _build_truncated_turns(
            projection.turns, projection.extraction_notes, middle
        )
        candidate = _serialize(candidate_turns, projection.extraction_notes)
        if len(candidate) <= max_context_chars:
            best = candidate
            best_user_chars = sum(len(turn["text"]) for turn in candidate_turns)
            low = middle + 1
        else:
            high = middle - 1
    return best, True, best_user_chars


__all__ = [
    "PROJECTION_NAME",
    "TRUNCATION_MARKER",
    "TRUNCATION_STRATEGY",
    "VERSION",
    "UserInputProjection",
    "project_user_input",
    "serialize_for_model",
]
