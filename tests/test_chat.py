from __future__ import annotations

import pytest

from trajfoundry.providers.chat import (
    ChatCompletionsAdapterError,
    parse_chat_capture,
)


def _capture(
    *,
    request_body: object,
    response_body: object,
    status_code: object = 200,
    stream: bool = False,
) -> dict:
    return {
        "path": "/v1/chat/completions",
        "request_body": request_body,
        "response_body": response_body,
        "status_code": status_code,
        "is_stream": stream,
        "session_id": "session-1",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "parent_thread_id": "parent-thread",
        "parent_turn_id": "parent-turn",
        "forked_from_thread_id": "fork-thread",
        "subagent_marker": "child",
        "request_id": "request-1",
        "captured_at": "2026-08-30T00:00:00Z",
        "harness": "qwen-code",
    }


def _completion(message: object, *, model: str = "chat-model") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": message,
                "provider_trace": {"ignored": True},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def _parse(capture: dict, **kwargs):
    return parse_chat_capture(
        capture,
        source_path="tokenplan/one.json",
        source_sha256="a" * 64,
        **kwargs,
    )


def test_parses_multiturn_nested_tools_and_resolves_result_names() -> None:
    request = {
        "model": "request-model",
        "messages": [
            {"role": "system", "content": "be exact"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect "},
                    {
                        "type": "image_url",
                        "image_url": {"url": "$media_ref:media_0"},
                    },
                    {"type": "text", "text": " now"},
                ],
                "agent": "ignored-upstream-label",
                "traceId": "ignored-trace",
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "preferred thought",
                "reasoning": "ignored fallback",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"weather"}',
                        },
                    },
                    {
                        "type": "function",
                        "id": "call-2",
                        "function": {"name": "raw_tool", "arguments": "not-json"},
                    },
                ],
                "usage": {"ignored": True},
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "wrong-upstream-name",
                "content": [
                    {"type": "text", "text": "sunny"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "$media_ref:result"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "continue",
                "reasoning_content": None,
                "reasoning": "fallback thought",
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "raw result"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look something up",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "raw_tool",
                    "parameters": {"type": "object"},
                },
            },
        ],
        "agent": {"ignored": True},
    }
    response = _completion(
        {
            "role": "assistant",
            "content": "done",
            "reasoning_content": "final thought",
            "reasoning": "ignored final fallback",
            "agent": "ignored",
        },
        model="response-model",
    )

    snapshot = _parse(_capture(request_body=request, response_body=response))

    assert snapshot.provider == "openai"
    assert snapshot.operation == "chat_completions"
    assert snapshot.model == "request-model"
    assert snapshot.harness == "qwen-code"
    assert snapshot.session_id == "session-1"
    assert snapshot.thread_id == "thread-1"
    assert snapshot.turn_id == "turn-1"
    assert snapshot.parent_thread_id == "parent-thread"
    assert snapshot.parent_turn_id == "parent-turn"
    assert snapshot.forked_from_thread_id == "fork-thread"
    assert snapshot.subagent_marker == "child"
    assert snapshot.request_id == "request-1"
    assert snapshot.captured_at == "2026-08-30T00:00:00Z"
    assert snapshot.instructions == ""
    assert snapshot.termination == ""
    assert [message.role for message in snapshot.history] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert snapshot.history[1].content == (
        'inspect {"image_url":{"url":"$media_ref:media_0"},"type":"image_url"} now'
    )
    first_assistant = snapshot.history[2]
    assert first_assistant.content == ""
    assert first_assistant.reasoning_content == "preferred thought"
    assert [call.function.name for call in first_assistant.tool_calls or []] == [
        "lookup",
        "raw_tool",
    ]
    assert first_assistant.tool_calls is not None
    assert first_assistant.tool_calls[0].function.arguments == {"query": "weather"}
    assert first_assistant.tool_calls[1].function.arguments == {"raw": "not-json"}
    assert snapshot.history[3].name == "lookup"
    assert snapshot.history[3].content == (
        'sunny{"image_url":{"url":"$media_ref:result"},"type":"image_url"}'
    )
    assert snapshot.history[4].reasoning_content == "fallback thought"
    assert snapshot.history[5].name == "raw_tool"
    assert [tool.name for tool in snapshot.tools] == ["lookup", "raw_tool"]
    assert snapshot.tools[0].description == "Look something up"
    assert snapshot.tools[1].description == ""
    assert snapshot.response[0].content == "done"
    assert snapshot.response[0].reasoning_content == "final thought"
    assert snapshot.server_tool_calls == []
    assert snapshot.agent_messages == []
    assert snapshot.compaction_items == []
    assert snapshot.outcome == "success"
    assert snapshot.wire_complete is True
    assert snapshot.issues == []


def test_normalized_stream_is_parsed_as_one_final_object_with_array_content() -> None:
    request = {
        "model": "stream-model",
        "stream": True,
        "messages": [{"role": "user", "content": "go"}],
    }
    response = _completion(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first"},
                {
                    "type": "input_audio",
                    "input_audio": {"format": "wav", "data": "$media_ref:audio"},
                },
                {"type": "output_text", "text": "last"},
            ],
            "reasoning_content": None,
            "reasoning": "stream reasoning",
        }
    )

    snapshot = _parse(
        _capture(
            request_body=request,
            response_body=response,
            stream=True,
        ),
        response_is_normalized_final=True,
    )

    assert snapshot.response[0].content == (
        'first{"input_audio":{"data":"$media_ref:audio","format":"wav"},'
        '"type":"input_audio"}last'
    )
    assert snapshot.response[0].reasoning_content == "stream reasoning"
    assert snapshot.outcome == "success"
    assert snapshot.wire_complete is True


def test_response_tool_call_accepts_null_content() -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "chat-model",
                "messages": [{"role": "user", "content": "lookup"}],
            },
            response_body=_completion(
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "response-call",
                            "function": {
                                "name": "search",
                                "arguments": '{"q":"x"}',
                            },
                        }
                    ],
                }
            ),
        )
    )

    assert snapshot.response[0].content == ""
    assert snapshot.response[0].tool_calls is not None
    assert snapshot.response[0].tool_calls[0].id == "response-call"
    assert snapshot.response[0].tool_calls[0].function.arguments == {"q": "x"}
    assert snapshot.outcome == "success"


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_truncated_finish_reason_is_not_accepted(finish_reason: str) -> None:
    response = _completion({"role": "assistant", "content": "partial"})
    response["choices"][0]["finish_reason"] = finish_reason

    snapshot = _parse(
        _capture(
            request_body={
                "model": "chat-model",
                "messages": [{"role": "user", "content": "continue"}],
            },
            response_body=response,
        )
    )

    assert snapshot.response[0].content == "partial"
    assert snapshot.outcome == "truncated"
    assert snapshot.wire_complete is False
    assert "chat_completion_truncated" in {issue.code for issue in snapshot.issues}


@pytest.mark.parametrize(
    ("status_code", "response_body", "outcome", "code"),
    [
        (None, None, "transport_error", "missing_http_status"),
        (
            429,
            {"error": {"type": "rate_limit_error", "message": "secret body"}},
            "api_error",
            "openai_api_error",
        ),
        (200, [], "capture_invalid", "invalid_response_body"),
        (200, {"model": "m"}, "capture_invalid", "invalid_chat_choices"),
        (
            200,
            {"choices": [{}], "model": "m"},
            "capture_invalid",
            "invalid_chat_response_message",
        ),
    ],
)
def test_http_and_response_structure_failures_are_audited(
    status_code: object,
    response_body: object,
    outcome: str,
    code: str,
) -> None:
    snapshot = _parse(
        _capture(
            request_body={"model": "m", "messages": []},
            response_body=response_body,
            status_code=status_code,
        )
    )

    assert snapshot.outcome == outcome
    assert snapshot.wire_complete is False
    assert code in {issue.code for issue in snapshot.issues}
    assert all("secret body" not in issue.detail for issue in snapshot.issues)


def test_multiple_choices_preserves_only_first_and_is_invalid() -> None:
    response = _completion(
        {"role": "assistant", "content": "first", "reasoning": "one"}
    )
    response["choices"].append(
        {
            "index": 1,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "must not be concatenated",
                "reasoning": "two",
            },
        }
    )

    snapshot = _parse(
        _capture(
            request_body={"model": "m", "messages": []},
            response_body=response,
        )
    )

    assert [message.content for message in snapshot.response] == ["first"]
    assert snapshot.outcome == "capture_invalid"
    assert snapshot.wire_complete is False
    assert {issue.code for issue in snapshot.issues} == {"invalid_chat_choice_count"}


def test_wrong_response_object_is_a_structural_error() -> None:
    response = _completion(
        {"role": "assistant", "content": "not really chat", "reasoning": ""}
    )
    response["object"] = "response"

    snapshot = _parse(
        _capture(
            request_body={"model": "m", "messages": []},
            response_body=response,
        )
    )

    assert snapshot.response[0].content == "not really chat"
    assert snapshot.outcome == "capture_invalid"
    assert snapshot.wire_complete is False
    assert {issue.code for issue in snapshot.issues} == {
        "invalid_chat_completion_object"
    }


def test_orphan_tool_result_and_invalid_request_shape_are_audited() -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "m",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "unknown-call",
                        "name": "must-be-ignored",
                        "content": "result",
                    }
                ],
                "tools": "not-an-array",
            },
            response_body=_completion(
                {"role": "assistant", "content": "done", "reasoning": None}
            ),
        )
    )

    assert snapshot.history[0].name == ""
    assert snapshot.outcome == "capture_invalid"
    assert snapshot.wire_complete is False
    assert {issue.code for issue in snapshot.issues} == {
        "invalid_tools",
        "orphan_tool_result",
    }


@pytest.mark.parametrize(
    "capture",
    [None, {}, {"path": "/v1/responses"}],
)
def test_rejects_values_that_are_not_chat_captures(capture: object) -> None:
    with pytest.raises(ChatCompletionsAdapterError):
        parse_chat_capture(  # type: ignore[arg-type]
            capture,
            source_path="one.json",
            source_sha256="a" * 64,
        )
