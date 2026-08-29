from __future__ import annotations

import json

import pytest

from trajfoundry.audit_codes import RESPONSES_UNSUPPORTED_CALL_EVIDENCE
from trajfoundry.models import (
    AuditTag,
    Metadata,
    NormalizationAudit,
    Severity,
    TrajectoryNode,
)
from trajfoundry.providers.anthropic import (
    AnthropicCaptureError,
    parse_anthropic_capture,
)
from trajfoundry.quality import enrich_trajectory


def _capture(
    *,
    request_body: dict | None = None,
    response_body: object = None,
    status_code: int = 200,
    is_stream: bool = False,
    path: str = "/v1/messages",
) -> dict:
    return {
        "request_id": "req-1",
        "session_id": "session-1",
        "captured_at": "2026-08-16T01:02:03+00:00",
        "path": path,
        "request_headers": {"x-claude-code-session-id": "thread-1"},
        "request_body": request_body
        if request_body is not None
        else {"model": "claude-test", "messages": []},
        "response_body": response_body,
        "status_code": status_code,
        "is_stream": is_stream,
    }


def _parse(capture: dict):
    return parse_anthropic_capture(
        capture,
        source_path="/captures/example.json",
        source_sha256="a" * 64,
    )


def _event(seq: int, event: str, **data: object) -> dict:
    return {"seq": seq, "event": event, "data": {"type": event, **data}}


def test_anthropic_unsupported_text_is_not_responses_provider_evidence() -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "claude-test",
                "messages": [
                    {"role": "user", "content": "go"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call-ghost",
                                "name": "ghost",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-ghost",
                                "content": "unsupported call: ghost",
                            }
                        ],
                    },
                ],
                "tools": [{"name": "ghost", "input_schema": {}}],
            },
            response_body={
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            },
        )
    )

    assert RESPONSES_UNSUPPORTED_CALL_EVIDENCE not in {
        issue.code for issue in snapshot.issues
    }
    node = TrajectoryNode(
        messages=[*snapshot.history, *snapshot.response],
        tools=snapshot.tools,
        harness=snapshot.harness,
        source="example.json",
        metadata=Metadata(source_file="example.json"),
        normalization_audit=NormalizationAudit(
            tag=AuditTag.PASS,
            issues=snapshot.issues,
        ),
    )
    enriched = enrich_trajectory(node)
    assert enriched.tool_call_check.hallucinated_calls == 0
    assert enriched.normalization_audit is not None
    assert enriched.normalization_audit.tag == AuditTag.PASS


def test_stream_reconstructs_messages_reasoning_and_tool_calls() -> None:
    request = {
        "model": "claude-test",
        "stream": True,
        "system": [
            {"type": "text", "text": "first\n", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "second"},
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "old thought",
                        "signature": "drop-me",
                    },
                    {"type": "text", "text": "working"},
                    {
                        "type": "tool_use",
                        "id": "old-call",
                        "name": "lookup",
                        "input": {"q": 1},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "old-call",
                        "content": [{"type": "text", "text": "result"}],
                        "is_error": False,
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "lookup",
                "description": "Look something up",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "integer"}},
                },
            },
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
        ],
    }
    response = [
        _event(
            1,
            "message_start",
            message={"role": "assistant", "model": "claude-test", "content": []},
        ),
        _event(
            2,
            "content_block_start",
            index=0,
            content_block={"type": "thinking", "thinking": "", "signature": ""},
        ),
        _event(
            3,
            "content_block_delta",
            index=0,
            delta={"type": "thinking_delta", "thinking": "new thought"},
        ),
        _event(
            4,
            "content_block_delta",
            index=0,
            delta={"type": "signature_delta", "signature": "drop-me-too"},
        ),
        _event(5, "content_block_stop", index=0),
        _event(
            6,
            "content_block_start",
            index=1,
            content_block={"type": "text", "text": ""},
        ),
        _event(
            7,
            "content_block_delta",
            index=1,
            delta={"type": "text_delta", "text": "answer"},
        ),
        _event(8, "content_block_stop", index=1),
        _event(
            9,
            "content_block_start",
            index=2,
            content_block={
                "type": "tool_use",
                "id": "new-call",
                "name": "lookup",
                "input": {},
            },
        ),
        _event(
            10,
            "content_block_delta",
            index=2,
            delta={"type": "input_json_delta", "partial_json": '{"q"'},
        ),
        _event(
            11,
            "content_block_delta",
            index=2,
            delta={"type": "input_json_delta", "partial_json": ":2}"},
        ),
        _event(12, "content_block_stop", index=2),
        _event(13, "message_delta", delta={"stop_reason": "tool_use"}),
        _event(14, "message_stop"),
    ]

    snapshot = _parse(
        _capture(request_body=request, response_body=response, is_stream=True)
    )

    assert snapshot.outcome == "success"
    assert snapshot.wire_complete is True
    assert snapshot.instructions == ""
    assert snapshot.thread_id == "thread-1"
    assert [message.role for message in snapshot.history] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert snapshot.history[0].content == "first\nsecond"
    assert snapshot.history[2].reasoning_content == "old thought"
    assert snapshot.history[3].content == "result"
    assert snapshot.history[3].name == "lookup"
    assert [tool.name for tool in snapshot.tools] == ["lookup"]
    assert snapshot.response[0].content == "answer"
    assert snapshot.response[0].reasoning_content == "new thought"
    assert snapshot.response[0].tool_calls is not None
    assert snapshot.response[0].tool_calls[0].function.arguments == {"q": 2}
    assert snapshot.history[2].reasoning_details is not None
    assert snapshot.history[2].reasoning_details[0]["signature"] == "drop-me"
    assert snapshot.response[0].reasoning_details is not None
    assert snapshot.response[0].reasoning_details[0]["signature"] == "drop-me-too"
    assert snapshot.issues == []


def test_nonstream_response_and_mid_conversation_system_message() -> None:
    capture = _capture(
        request_body={
            "model": "claude-request",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "date changed"},
            ],
        },
        response_body={
            "type": "message",
            "role": "assistant",
            "model": "claude-response",
            "content": [
                {"type": "thinking", "thinking": "think", "signature": "ignored"},
                {"type": "text", "text": "done"},
            ],
            "stop_reason": "end_turn",
        },
    )

    snapshot = _parse(capture)

    assert snapshot.outcome == "success"
    assert snapshot.wire_complete is True
    assert [message.role for message in snapshot.history] == ["user", "system"]
    assert snapshot.response[0].content == "done"
    assert snapshot.response[0].reasoning_content == "think"
    # Request model is authoritative when present.
    assert snapshot.model == "claude-request"
    assert snapshot.termination == ""


@pytest.mark.parametrize(
    ("is_stream", "response_is_normalized_final"),
    [(False, False), (True, True)],
    ids=["nonstream", "normalized-final"],
)
def test_final_response_max_tokens_is_truncated_and_preserves_content(
    is_stream: bool,
    response_is_normalized_final: bool,
) -> None:
    snapshot = parse_anthropic_capture(
        _capture(
            request_body={
                "model": "claude-test",
                "stream": is_stream,
                "messages": [{"role": "user", "content": "continue"}],
            },
            response_body={
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": "partial answer"}],
            },
            is_stream=is_stream,
        ),
        source_path="/captures/example.json",
        source_sha256="a" * 64,
        response_is_normalized_final=response_is_normalized_final,
    )

    assert snapshot.outcome == "truncated"
    assert snapshot.wire_complete is False
    assert snapshot.response[0].content == "partial answer"
    assert snapshot.termination == ""
    assert "anthropic_max_tokens_truncated" in {issue.code for issue in snapshot.issues}


@pytest.mark.parametrize("stop_location", ["message_start", "message_delta"])
def test_stream_max_tokens_is_truncated_and_preserves_content(
    stop_location: str,
) -> None:
    start_message: dict[str, object] = {
        "role": "assistant",
        "model": "claude-test",
        "content": [],
    }
    if stop_location == "message_start":
        start_message["stop_reason"] = "max_tokens"
    response = [
        _event(1, "message_start", message=start_message),
        _event(
            2,
            "content_block_start",
            index=0,
            content_block={"type": "text", "text": ""},
        ),
        _event(
            3,
            "content_block_delta",
            index=0,
            delta={"type": "text_delta", "text": "partial answer"},
        ),
        _event(4, "content_block_stop", index=0),
    ]
    if stop_location == "message_delta":
        response.append(_event(5, "message_delta", delta={"stop_reason": "max_tokens"}))
    response.append(
        _event(6 if stop_location == "message_delta" else 5, "message_stop")
    )

    snapshot = _parse(_capture(response_body=response, is_stream=True))

    assert snapshot.outcome == "truncated"
    assert snapshot.wire_complete is False
    assert snapshot.response[0].content == "partial answer"
    assert snapshot.termination == ""
    assert "anthropic_max_tokens_truncated" in {issue.code for issue in snapshot.issues}


def test_server_tools_are_separate_from_client_tools_and_results_attach() -> None:
    capture = _capture(
        request_body={
            "model": "claude-test",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "srv-history",
                            "name": "web_search",
                            "input": {"query": "past"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srv-history",
                            "content": [{"type": "web_search_result", "title": "Past"}],
                        },
                    ],
                }
            ],
            "tools": [
                {
                    "name": "local",
                    "description": "",
                    "input_schema": {"type": "object"},
                },
                {"type": "web_search_20250305", "name": "web_search"},
            ],
        },
        response_body={
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srv-response",
                    "name": "web_search",
                    "input": {"query": "now"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srv-response",
                    "content": [{"type": "web_search_result", "title": "Now"}],
                },
                {"type": "text", "text": "summary"},
            ],
        },
    )

    snapshot = _parse(capture)

    assert [tool.name for tool in snapshot.tools] == ["local"]
    assert [(call.id, call.origin) for call in snapshot.server_tool_calls] == [
        ("srv-history", "history"),
        ("srv-response", "response"),
    ]
    assert snapshot.server_tool_calls[0].result == {
        "type": "web_search_tool_result",
        "tool_use_id": "srv-history",
        "content": [{"type": "web_search_result", "title": "Past"}],
    }
    assert snapshot.server_tool_calls[1].result == {
        "type": "web_search_tool_result",
        "tool_use_id": "srv-response",
        "content": [{"type": "web_search_result", "title": "Now"}],
    }
    assert snapshot.response[0].content == "summary"


@pytest.mark.parametrize(
    ("input_fields", "expected"),
    [
        ({}, {}),
        ({"input": '{"query":"weather"}'}, {"query": "weather"}),
        ({"input": "not-json"}, {"raw": "not-json"}),
        ({"input": ["already", "decoded"]}, ["already", "decoded"]),
    ],
)
def test_tool_use_input_is_normalized_without_losing_invalid_strings(
    input_fields: dict, expected: object
) -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "claude-test",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call-1",
                                "name": "lookup",
                                **input_fields,
                            }
                        ],
                    }
                ],
            },
            response_body={
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        )
    )

    calls = snapshot.history[0].tool_calls
    assert calls is not None
    assert calls[0].function.arguments == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("result", {"status": "ok"}),
        ("results", [{"status": "ok"}]),
        ("output", "complete"),
    ],
)
def test_server_tool_embedded_result_preserves_its_field_name(
    field: str, value: object
) -> None:
    call = {
        "type": "server_tool_use",
        "id": "srv-1",
        "name": "web_search",
        "input": {"query": "now"},
        field: value,
    }
    snapshot = _parse(
        _capture(
            response_body={
                "type": "message",
                "role": "assistant",
                "content": [call],
            }
        )
    )

    assert snapshot.server_tool_calls[0].result == {field: value}


def test_results_before_calls_are_resolved_after_the_whole_capture() -> None:
    server_result = {
        "type": "web_search_tool_result",
        "tool_use_id": "srv-late",
        "content": [{"type": "web_search_result", "title": "Found"}],
    }
    snapshot = _parse(
        _capture(
            request_body={
                "model": "claude-test",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-late",
                                "content": "client result",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call-late",
                                "name": "lookup",
                                "input": {},
                            }
                        ],
                    },
                ],
            },
            response_body={
                "type": "message",
                "role": "assistant",
                "content": [
                    server_result,
                    {
                        "type": "server_tool_use",
                        "id": "srv-late",
                        "name": "web_search",
                        "input": {"query": "late"},
                    },
                ],
            },
        )
    )

    assert snapshot.history[0].role == "tool"
    assert snapshot.history[0].name == "lookup"
    assert snapshot.server_tool_calls[0].name == "web_search"
    assert snapshot.server_tool_calls[0].result == server_result
    assert snapshot.issues == []


def test_orphan_results_survive_without_fabricated_tool_names() -> None:
    server_result = {
        "type": "web_search_tool_result",
        "tool_use_id": "orphan-server",
        "content": [{"type": "web_search_result", "title": "Found"}],
    }
    snapshot = _parse(
        _capture(
            request_body={
                "model": "claude-test",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "orphan-client",
                                "content": "result",
                            }
                        ],
                    }
                ],
            },
            response_body={
                "type": "message",
                "role": "assistant",
                "content": [server_result],
            },
        )
    )

    assert snapshot.history[0].role == "tool"
    assert snapshot.history[0].name == ""
    assert snapshot.server_tool_calls[0].name == ""
    assert snapshot.server_tool_calls[0].result == server_result
    assert {issue.code for issue in snapshot.issues} >= {
        "orphan_tool_result",
        "orphan_server_tool_result",
    }
    assert "__unknown_tool__" not in snapshot.model_dump_json()


def test_duplicate_results_are_preserved_and_diagnosed() -> None:
    client_results = [
        {"type": "tool_result", "tool_use_id": "call-1", "content": "first"},
        {"type": "tool_result", "tool_use_id": "call-1", "content": "second"},
    ]
    server_results = [
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srv-1",
            "content": [{"title": value}],
        }
        for value in ("first", "second")
    ]
    snapshot = _parse(
        _capture(
            request_body={
                "model": "claude-test",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call-1",
                                "name": "lookup",
                                "input": {},
                            },
                            {
                                "type": "server_tool_use",
                                "id": "srv-1",
                                "name": "web_search",
                                "input": {},
                            },
                        ],
                    },
                    {"role": "user", "content": client_results},
                ],
            },
            response_body={
                "type": "message",
                "role": "assistant",
                "content": server_results,
            },
        )
    )

    assert [
        message.content for message in snapshot.history if message.role == "tool"
    ] == [
        "first",
        "second",
    ]
    assert [call.result for call in snapshot.server_tool_calls] == server_results
    assert {issue.code for issue in snapshot.issues} >= {
        "duplicate_tool_result",
        "duplicate_server_tool_result",
    }


def test_generated_missing_ids_are_stable_across_repeated_parses() -> None:
    capture = _capture(
        request_body={
            "model": "claude-test",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "lookup", "input": {}},
                        {
                            "type": "server_tool_use",
                            "name": "web_search",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "orphan"}],
                },
            ],
        },
        response_body={
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "web_search_tool_result", "content": []},
                {"type": "text", "text": "done"},
            ],
        },
    )

    first = _parse(capture)
    second = _parse(capture)
    first_calls = first.history[0].tool_calls
    second_calls = second.history[0].tool_calls
    assert first_calls is not None
    assert second_calls is not None
    assert first_calls[0].id == second_calls[0].id == "missing-client-tool-call:1"
    assert (
        first.server_tool_calls[0].id
        == second.server_tool_calls[0].id
        == "missing-server-tool-call:1"
    )
    assert (
        first.history[1].tool_call_id
        == second.history[1].tool_call_id
        == "missing-client-tool-result:1"
    )
    assert (
        first.server_tool_calls[1].id
        == second.server_tool_calls[1].id
        == "missing:server-result:response:response_body.content[0]"
    )
    assert "__unknown_tool__" not in first.model_dump_json()


def test_raw_sse_text_is_supported() -> None:
    payloads = [
        (
            "message_start",
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    ]
    raw_sse = "".join(
        f"event: {event}\ndata: {json.dumps(data)}\n\n" for event, data in payloads
    )

    snapshot = _parse(_capture(response_body=raw_sse, is_stream=True))

    assert snapshot.outcome == "success"
    assert snapshot.response[0].content == "hello"


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (
            [
                _event(
                    1, "message_start", message={"role": "assistant", "content": []}
                ),
                _event(
                    2,
                    "content_block_start",
                    index=0,
                    content_block={"type": "text", "text": "partial"},
                ),
            ],
            "missing_message_stop",
        ),
        (
            [
                _event(
                    1, "message_start", message={"role": "assistant", "content": []}
                ),
                _event(3, "message_stop"),
            ],
            "sse_sequence_gap",
        ),
    ],
)
def test_incomplete_stream_is_truncated(
    response: list[dict], expected_code: str
) -> None:
    snapshot = _parse(_capture(response_body=response, is_stream=True))

    assert snapshot.outcome == "truncated"
    assert snapshot.wire_complete is False
    assert expected_code in {issue.code for issue in snapshot.issues}


def test_stream_error_and_http_error_are_api_errors() -> None:
    stream_error = _parse(
        _capture(
            response_body=[
                _event(
                    1, "error", error={"type": "overloaded_error", "message": "busy"}
                )
            ],
            is_stream=True,
        )
    )
    http_error = _parse(
        _capture(
            status_code=429,
            response_body={"type": "error", "error": {"message": "rate limited"}},
        )
    )

    assert stream_error.outcome == "api_error"
    assert stream_error.wire_complete is False
    assert http_error.outcome == "api_error"
    assert http_error.issues[0].detail == "Anthropic API error (HTTP 429)"


def test_http_error_detail_never_retains_sensitive_provider_body() -> None:
    secret_values = (
        "TOP-SECRET",
        "proxy-password",
        "203.0.113.42",
    )
    snapshot = _parse(
        _capture(
            status_code=401,
            response_body={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                    "message": (
                        "Authorization: Bearer TOP-SECRET; "
                        "api_key=TOP-SECRET; proxy credential=proxy-password; "
                        "x-forwarded-for=203.0.113.42"
                    ),
                },
            },
        )
    )

    serialized = snapshot.model_dump_json()
    assert snapshot.issues[0].detail == (
        "Anthropic API error "
        "(HTTP 401; type=authentication_error; code=invalid_api_key)"
    )
    assert all(secret not in serialized for secret in secret_values)


def test_count_tokens_is_marked_as_non_trajectory_operation() -> None:
    snapshot = _parse(
        _capture(
            path="/v1/messages/count_tokens?beta=true",
            request_body={
                "model": "claude-test",
                "messages": [{"role": "user", "content": "x"}],
            },
            response_body={"input_tokens": 17},
        )
    )

    assert snapshot.operation == "count_tokens"
    assert snapshot.outcome == "success"
    assert snapshot.wire_complete is True
    assert snapshot.history == []
    assert snapshot.response == []


def test_unknown_endpoint_is_rejected() -> None:
    with pytest.raises(AnthropicCaptureError):
        _parse(_capture(path="/v1/complete", response_body={}))


@pytest.mark.parametrize("envelope", ["metadata", "user_id"])
@pytest.mark.parametrize("field", ["thread_source", "threadSource"])
@pytest.mark.parametrize("value", ["system", "automation", "subagent", "custom"])
def test_thread_source_is_ignored_for_subagent_identity(
    envelope: str,
    field: str,
    value: str,
) -> None:
    metadata: dict[str, object]
    if envelope == "metadata":
        metadata = {field: value}
    else:
        metadata = {"user_id": json.dumps({field: value})}

    snapshot = _parse(
        _capture(
            request_body={
                "model": "claude-test",
                "metadata": metadata,
                "messages": [],
            },
            response_body={
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [],
            },
        )
    )

    assert snapshot.subagent_marker == ""
    assert snapshot.thread_id == "thread-1"
    assert "isolated_subagent_missing_thread_id" not in {
        issue.code for issue in snapshot.issues
    }
    assert snapshot.outcome == "success"


def test_subagent_without_explicit_thread_id_ignores_session_header_fallback() -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "claude-test",
                "metadata": {
                    "parent_thread_id": "main-thread",
                    "parent_turn_id": "turn-1",
                    "subagent_marker": "collab_spawn",
                },
                "messages": [],
            },
            response_body={
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [],
            },
        )
    )

    assert snapshot.thread_id == "__isolated_subagent__:/captures/example.json"
    warning = next(
        issue
        for issue in snapshot.issues
        if issue.code == "isolated_subagent_missing_thread_id"
    )
    assert warning.severity == Severity.WARNING
