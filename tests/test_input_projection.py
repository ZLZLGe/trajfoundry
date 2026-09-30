from __future__ import annotations

import orjson

from trajfoundry.classification.input_projection import (
    PROJECTION_NAME,
    TRUNCATION_MARKER,
    project_user_input,
    serialize_for_model,
)


def test_projection_keeps_clean_user_requests_and_excludes_other_roles() -> None:
    projection = project_user_input(
        {
            "messages": [
                {"role": "system", "content": "hidden system"},
                {"role": "assistant", "content": "hidden answer"},
                {"role": "tool", "content": "hidden tool result"},
                {
                    "role": "user",
                    "content": (
                        "<environment_context>internal</environment_context>\n"
                        "Please inspect the project."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first request"},
                        {"type": "image", "url": "discard-me"},
                        {"type": "text", "text": "second request"},
                    ],
                },
            ]
        }
    )

    assert PROJECTION_NAME == "user_only_root_v2"
    assert projection.user_turn_count == 2
    assert [turn["text"] for turn in projection.turns] == [
        "Please inspect the project.",
        "first request\n[IMAGE_OMITTED: no media content available]\nsecond request",
    ]
    assert projection.extraction_notes["excluded_role_assistant"] == 1
    assert projection.extraction_notes["excluded_role_tool"] == 1


def test_projection_serialization_is_bounded_and_marks_truncation() -> None:
    projection = project_user_input(
        {"messages": [{"role": "user", "content": "x" * 20_000}]}
    )

    payload, truncated, sent_chars = serialize_for_model(projection, 1_024)

    assert truncated is True
    assert sent_chars > 0
    assert len(payload) <= 1_024
    assert TRUNCATION_MARKER in payload
    assert orjson.loads(payload)["user_turns"]


def test_projection_drops_harness_notifications() -> None:
    projection = project_user_input(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "[SYSTEM NOTIFICATION - NOT USER INPUT]\nbackground event",
                },
                {
                    "role": "user",
                    "content": "[Request interrupted by user]\nContinue with the task",
                },
                {
                    "role": "user",
                    "content": "You are a security monitor for autonomous AI coding agents.",
                },
            ]
        }
    )

    assert [turn["text"] for turn in projection.turns] == [
        "Continue with the task"
    ]


def test_projection_ignores_nested_subagent_user_requests() -> None:
    projection = project_user_input(
        {
            "messages": [{"role": "user", "content": "root request"}],
            "sub_agent_trajectory": {
                "b": {
                    "messages": [{"role": "user", "content": "child B"}],
                },
                "a": {
                    "messages": [{"role": "user", "content": "child A"}],
                },
            },
        }
    )

    assert [turn["text"] for turn in projection.turns] == ["root request"]
    assert "nested_trajectory_nodes" not in projection.extraction_notes
