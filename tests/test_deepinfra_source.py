from __future__ import annotations

import json

import pytest

from trajfoundry.sources.deepinfra import DeepInfraError, adapt_deepinfra_envelope


def _envelope(
    *,
    path: str = "/v1/chat/completions",
    request_body: object | str | None = None,
    response_body: object | str | None = None,
    headers: object | None = None,
) -> dict:
    if request_body is None:
        request_body = {"model": "test-model", "messages": [], "stream": False}
    if response_body is None:
        response_body = {
            "id": "chat-test",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [],
        }
    return {
        "request_id": "request-1",
        "request_time": "2026-09-14T12:00:00.000Z",
        "request": {
            "method": "POST",
            "path": path,
            "headers": headers if headers is not None else {},
            "body": request_body
            if isinstance(request_body, str)
            else json.dumps(request_body),
            "body_truncated": False,
        },
        "response": {
            "status_code": 200,
            "headers": {},
            "body": response_body
            if isinstance(response_body, str)
            else json.dumps(response_body),
            "body_truncated": False,
        },
        "access_log": {
            "path": path,
            "request_id": "request-1",
            "response_code": 200,
        },
    }


def _chat_sse() -> str:
    chunks = [
        {
            "id": "chat-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hel"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chat-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    )


def test_adapt_deepinfra_json_projects_only_provider_capture_fields() -> None:
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    response_body = {
        "id": "chat-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [],
    }

    capture = adapt_deepinfra_envelope(
        _envelope(request_body=request_body, response_body=response_body)
    )

    assert capture == {
        "path": "/v1/chat/completions",
        "request_body": request_body,
        "response_body": response_body,
        "status_code": 200,
        "captured_at": "2026-09-14T12:00:00.000Z",
        "request_id": "request-1",
        "request_headers": {},
        "is_stream": False,
        "session_id": "request-1",
    }


def test_adapt_deepinfra_aggregates_chat_sse_to_final_response() -> None:
    capture = adapt_deepinfra_envelope(
        _envelope(
            request_body={"model": "test-model", "messages": [], "stream": True},
            response_body=_chat_sse(),
        )
    )

    assert capture["is_stream"] is True
    assert capture["response_body"]["object"] == "chat.completion"
    assert capture["response_body"]["choices"][0]["message"]["content"] == "hello"


@pytest.mark.parametrize(
    "path", ["/v1/messages/count_tokens", "/messages/count_tokens"]
)
def test_adapt_deepinfra_accepts_anthropic_count_tokens(path: str) -> None:
    capture = adapt_deepinfra_envelope(
        _envelope(
            path=path,
            request_body={"model": "test-model", "messages": []},
            response_body={"input_tokens": 12},
        )
    )

    assert capture["path"] == path
    assert capture["response_body"] == {"input_tokens": 12}


@pytest.mark.parametrize(
    ("path", "response_body", "expected_types"),
    [
        (
            "/v1/responses",
            (
                "event: response.created\n"
                'data: {"type":"response.created","sequence_number":0}\n\n'
                "event: response.completed\n"
                'data: {"type":"response.completed","sequence_number":1,'
                '"response":{"status":"completed","output":[]}}\n\n'
            ),
            ["response.created", "response.completed"],
        ),
        (
            "/v1/messages",
            (
                "event: message_start\n"
                'data: {"type":"message_start","message":{"role":"assistant",'
                '"content":[],"model":"test-model"}}\n\n'
                "event: message_stop\n"
                'data: {"type":"message_stop"}\n\n'
            ),
            ["message_start", "message_stop"],
        ),
    ],
    ids=["responses", "anthropic"],
)
def test_adapt_deepinfra_preserves_event_streams_for_event_providers(
    path: str, response_body: str, expected_types: list[str]
) -> None:
    request = (
        {"model": "test-model", "input": [], "stream": True}
        if path.endswith("responses")
        else {"model": "test-model", "messages": [], "stream": True}
    )

    capture = adapt_deepinfra_envelope(
        _envelope(path=path, request_body=request, response_body=response_body)
    )

    assert [event["type"] for event in capture["response_body"]] == expected_types


def test_identity_headers_are_sanitized_promoted_and_body_metadata_stays_primary() -> (
    None
):
    capture = adapt_deepinfra_envelope(
        _envelope(
            request_body={
                "model": "test-model",
                "messages": [],
                "metadata": {
                    "session_id": "body-session",
                    "thread_id": "body-thread",
                    "user_id": "body-user",
                },
            },
            headers={
                "X-Session-Id": "header-session",
                "X-Thread-Id": "header-thread",
                "X-User-Id": "header-user",
                "Authorization": "secret-token",
                "X-Forwarded-For": "192.0.2.1",
                "Username": "private-name",
            },
        )
    )

    assert capture["request_headers"] == {
        "session_id": "header-session",
        "thread_id": "header-thread",
    }
    assert "session_id" not in capture
    assert "thread_id" not in capture
    assert capture["user_id"] == "header-user"
    serialized = repr(capture)
    assert "secret-token" not in serialized
    assert "192.0.2.1" not in serialized
    assert "private-name" not in serialized


def test_identity_headers_are_capture_fallbacks() -> None:
    capture = adapt_deepinfra_envelope(
        _envelope(
            headers={
                "Session-Id": "session-alias",
                "X-Session-Id": "session-alias",
                "X-Thread-Id": "thread-alias",
                "X-User-Id": "user-alias",
            }
        )
    )

    assert capture["session_id"] == "session-alias"
    assert capture["thread_id"] == "thread-alias"
    assert capture["user_id"] == "user-alias"
    assert capture["request_headers"] == {
        "session_id": "session-alias",
        "thread_id": "thread-alias",
    }


def test_claude_agent_and_parent_headers_define_thread_boundaries() -> None:
    capture = adapt_deepinfra_envelope(
        _envelope(
            headers={
                "X-Claude-Code-Session-Id": "claude-session",
                "X-Claude-Code-Agent-Id": "claude-agent",
                "X-Parent-Session-Id": "claude-parent-thread",
                "X-User-Id": "user-alias",
                "Authorization": "secret-token",
            }
        )
    )

    assert capture["session_id"] == "claude-session"
    assert capture["thread_id"] == "claude-agent"
    assert capture["request_headers"] == {
        "session_id": "claude-session",
        "thread_id": "claude-agent",
        "x-claude-code-session-id": "claude-session",
        "x-claude-code-agent-id": "claude-agent",
        "parent_thread_id": "claude-parent-thread",
    }
    assert capture["user_id"] == "user-alias"
    assert "secret-token" not in repr(capture)


def test_deepseek_identity_headers_are_promoted_without_transport_headers() -> None:
    capture = adapt_deepinfra_envelope(
        _envelope(
            headers={
                "X-DeepSeek-Harness-Session-Id": "deepseek-session",
                "X-DeepSeek-Harness-User-Id": "deepseek-user",
                "X-Forwarded-For": "192.0.2.1",
            }
        )
    )

    assert capture["session_id"] == "deepseek-session"
    assert capture["user_id"] == "deepseek-user"
    assert "x-forwarded-for" not in capture["request_headers"]


def test_transport_failure_with_missing_response_body_reaches_provider_parser() -> None:
    value = _envelope()
    value["response"] = {"status_code": 0, "headers": {}}

    capture = adapt_deepinfra_envelope(value)

    assert capture["status_code"] == 0
    assert capture["response_body"] == {}


@pytest.mark.parametrize("status_code", [400, 404, 503])
def test_api_failure_with_plaintext_or_missing_response_body_is_preserved(
    status_code: int,
) -> None:
    value = _envelope()
    value["response"] = {
        "status_code": status_code,
        "headers": {},
        "body": "upstream failure",
    }

    capture = adapt_deepinfra_envelope(value)

    assert capture["status_code"] == status_code
    assert capture["response_body"] == "upstream failure"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["request"].pop("body"),
        lambda value: value["response"].update(body=""),
        lambda value: value["request"].update(body_truncated=True),
        lambda value: value["response"].update(body_truncated=True),
    ],
    ids=[
        "request-body-missing",
        "response-body-empty",
        "request-truncated",
        "response-truncated",
    ],
)
def test_incomplete_deepinfra_capture_has_stable_error(mutate) -> None:
    value = _envelope()
    mutate(value)

    with pytest.raises(DeepInfraError) as error:
        adapt_deepinfra_envelope(value)

    assert error.value.code == "deepinfra_incomplete_envelope"
    assert error.value.detail.startswith(("$.request.body", "$.response.body"))


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"request": {}, "response": {}, "access_log": {}},
        _envelope(path="/v1/unknown"),
        _envelope(request_body="not-json"),
        _envelope(request_body="[]"),
    ],
    ids=["root", "unknown-shape", "unknown-endpoint", "request-json", "request-shape"],
)
def test_invalid_deepinfra_envelope_has_stable_error(value: object) -> None:
    with pytest.raises(DeepInfraError) as error:
        adapt_deepinfra_envelope(value)

    assert error.value.code == "invalid_deepinfra_envelope"
    assert error.value.detail


@pytest.mark.parametrize(
    "body",
    [
        "not-json-or-sse",
        "null",
        "event: response.created\ndata: not-json\n\n",
    ],
    ids=["text", "json-scalar", "malformed-sse"],
)
def test_unknown_response_body_shape_is_rejected(body: str) -> None:
    with pytest.raises(DeepInfraError) as error:
        adapt_deepinfra_envelope(_envelope(response_body=body))

    assert error.value.code == "invalid_deepinfra_envelope"
    assert error.value.detail.startswith("$.response.body")
