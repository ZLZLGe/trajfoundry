from __future__ import annotations

import pytest

from trajfoundry.providers.anthropic import parse_anthropic_capture
from trajfoundry.providers.chat import parse_chat_capture
from trajfoundry.sources.tokenplan import TokenPlanError, adapt_tokenplan_envelope


def _envelope(
    *,
    request_body: dict | None = None,
    response_body: object | None = None,
    request_path: str = "/v1/messages",
    response_format: str = "ANTHROPIC_MESSAGES_NORMALIZED",
    stream: bool = True,
    metadata: dict | None = None,
    media: list[dict] | None = None,
) -> dict:
    return {
        "client_request": {
            "path": request_path,
            "protocol": {
                "/v1/messages": "anthropic_messages",
                "/v1/responses": "openai_responses",
                "/v1/chat/completions": "openai_chat",
            }[request_path],
            "stream": stream,
            "capture": {
                "body": request_body
                if request_body is not None
                else {"model": "claude-test", "messages": [], "stream": stream},
                "status": "COMPLETE",
            },
            "media": media or [],
        },
        "client_response": {
            "capture": {
                "body": response_body
                if response_body is not None
                else {
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": "done"}],
                },
                "format": response_format,
                "status": "COMPLETE",
            },
            "media": [],
        },
        "metadata": metadata
        or {
            "request_id": "request-1",
            "session_id": "envelope-session",
            "task_id": "envelope-task",
            "received_at_ms": 0,
            "completed_at_ms": 1234,
            "http_status_code": 200,
            "data_quality": {
                "client_request_complete": True,
                "client_response_complete": True,
                "all_attachments_available": True,
                "truncated": False,
            },
        },
    }


def test_adapt_tokenplan_envelope_projects_transport_fields_without_overwriting_identity() -> (
    None
):
    request_body = {
        "model": "claude-test",
        "stream": True,
        "metadata": {"session_id": "body-session", "turn_id": "body-turn"},
        "messages": [{"role": "user", "content": "go"}],
    }

    adapted = adapt_tokenplan_envelope(
        _envelope(
            request_body=request_body,
            media=[{"part_id": "media_0"}],
        )
    )

    assert adapted.source_name == "tokenplan"
    assert adapted.endpoint == "/v1/messages"
    assert adapted.captured_at == "1970-01-01T00:00:01.234Z"
    assert adapted.response_is_normalized_final is True
    assert adapted.has_media is True
    assert adapted.capture["request_body"] is request_body
    assert adapted.capture["response_body"]["type"] == "message"
    assert adapted.capture["status_code"] == 200
    assert "session_id" not in adapted.capture
    assert "turn_id" not in adapted.capture


def test_adapt_tokenplan_envelope_uses_outer_identity_and_received_timestamp_as_fallback() -> (
    None
):
    adapted = adapt_tokenplan_envelope(
        _envelope(
            request_body={"model": "claude-test", "messages": []},
            metadata={
                "request_id": "request-1",
                "session_id": "envelope-session",
                "task_id": "envelope-task",
                "received_at_ms": 1000,
                "completed_at_ms": None,
                "http_status_code": 200,
                "data_quality": {
                    "client_request_complete": True,
                    "client_response_complete": True,
                    "all_attachments_available": True,
                    "truncated": False,
                },
            },
        )
    )

    assert adapted.capture["session_id"] == "envelope-session"
    assert adapted.capture["turn_id"] == "envelope-task"
    assert adapted.captured_at == "1970-01-01T00:00:01.000Z"


def test_empty_tokenplan_envelope_has_a_stable_error_code() -> None:
    with pytest.raises(TokenPlanError) as error:
        adapt_tokenplan_envelope({})

    assert error.value.code == "tokenplan_empty_envelope"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record["client_request"].update(protocol="openai_responses"),
        lambda record: record["client_response"]["capture"].update(status="PARTIAL"),
        lambda record: record["metadata"]["data_quality"].update(truncated=True),
    ],
    ids=["protocol-path", "capture-status", "data-quality"],
)
def test_invalid_tokenplan_envelope_integrity_is_rejected(mutate) -> None:
    record = _envelope()
    mutate(record)

    with pytest.raises(TokenPlanError) as error:
        adapt_tokenplan_envelope(record)

    assert error.value.code == "invalid_tokenplan_envelope"


def test_tokenplan_anthropic_normalized_final_uses_final_object_not_sse() -> None:
    adapted = adapt_tokenplan_envelope(
        _envelope(
            request_body={
                "model": "claude-test",
                "stream": True,
                "messages": [{"role": "user", "content": "go"}],
            }
        )
    )

    unchanged_sse_behavior = parse_anthropic_capture(
        adapted.capture,
        source_path="tokenplan.json",
        source_sha256="a" * 64,
    )
    normalized_final = parse_anthropic_capture(
        adapted.capture,
        source_path="tokenplan.json",
        source_sha256="a" * 64,
        response_is_normalized_final=adapted.response_is_normalized_final,
    )

    assert unchanged_sse_behavior.outcome == "truncated"
    assert normalized_final.outcome == "success"
    assert normalized_final.wire_complete is True
    assert normalized_final.response[0].content == "done"


def test_non_normalized_response_does_not_receive_final_object_marker() -> None:
    adapted = adapt_tokenplan_envelope(
        _envelope(
            request_path="/v1/responses",
            request_body={"model": "gpt-test", "input": []},
            response_body={"object": "response", "status": "completed", "output": []},
            response_format="ORIGINAL",
            stream=False,
        )
    )

    assert adapted.response_is_normalized_final is False


def test_tokenplan_chat_marker_and_request_identity_are_preserved() -> None:
    adapted = adapt_tokenplan_envelope(
        _envelope(
            request_path="/v1/chat/completions",
            request_body={
                "model": "chat-test",
                "messages": [{"role": "user", "content": "go"}],
                "client_metadata": {
                    "session_id": "body-session",
                    "thread_id": "body-thread",
                    "turn_id": "body-turn",
                },
            },
            response_body={
                "object": "chat.completion",
                "model": "chat-test",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"},
                    }
                ],
            },
            response_format="OPENAI_CHAT_COMPLETIONS_NORMALIZED",
            stream=True,
        )
    )

    assert adapted.response_is_normalized_final is True
    assert "session_id" not in adapted.capture
    assert "turn_id" not in adapted.capture

    snapshot = parse_chat_capture(
        adapted.capture,
        source_path="tokenplan.json",
        source_sha256="a" * 64,
        response_is_normalized_final=adapted.response_is_normalized_final,
    )

    assert snapshot.session_id == "body-session"
    assert snapshot.thread_id == "body-thread"
    assert snapshot.turn_id == "body-turn"
    assert snapshot.outcome == "success"
    assert snapshot.issues == []
