from __future__ import annotations

import json

import pytest

from trajfoundry.audit_codes import RESPONSES_UNSUPPORTED_CALL_EVIDENCE
from trajfoundry.models import (
    AuditIssue,
    AuditTag,
    Metadata,
    NormalizationAudit,
    Severity,
    TrajectoryNode,
)
from trajfoundry.providers.responses import (
    ResponsesAdapterError,
    parse_responses_capture,
)
from trajfoundry.quality import enrich_trajectory
from trajfoundry.streaming import streaming_prefix_leaves


def _capture(
    *, request_body: dict, response_body: object, status_code: int = 200
) -> dict:
    client_metadata = request_body.get("client_metadata")
    session_id = (
        client_metadata.get("session_id", "capture-session")
        if isinstance(client_metadata, dict)
        else "capture-session"
    )
    return {
        "path": "/v1/responses",
        "captured_at": "2026-08-27T01:02:03+00:00",
        "session_id": session_id,
        "request_id": "request-1",
        "status_code": status_code,
        "is_stream": isinstance(response_body, list),
        "request_body": request_body,
        "response_body": response_body,
    }


def _parse(capture: dict):
    return parse_responses_capture(
        capture,
        source_path="/captures/one.json",
        source_sha256="abc123",
    )


def _frame(seq: int, event: dict) -> dict:
    event = {**event, "sequence_number": seq - 1}
    return {"seq": seq, "event": event["type"], "data": event}


def test_preserves_developer_and_keeps_instructions_out_of_messages() -> None:
    instructions = "  exact instructions\nwith final newline\n"
    request = {
        "model": "gpt-test",
        "instructions": instructions,
        "client_metadata": {
            "session_id": "session-1",
            "thread_id": "thread-1",
            "turn_id": "turn-1",
        },
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "developer text"}],
            },
            {"type": "message", "role": "user", "content": "question"},
            {
                "type": "reasoning",
                "id": "rs-1",
                "summary": [{"type": "summary_text", "text": "inspect"}],
                "encrypted_content": "opaque",
            },
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "read",
                "arguments": '{"path":"/tmp/a"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": '{"ok":true}',
            },
            {
                "type": "custom_tool_call",
                "call_id": "call-2",
                "name": "patch",
                "input": "*** Begin Patch\n",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call-2",
                "output": "Done!",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "type": "custom",
                "name": "patch",
                "description": "Apply a patch",
                "format": {"type": "grammar", "syntax": "lark"},
            },
            {"type": "web_search"},
        ],
    }
    response = {
        "id": "resp-1",
        "status": "completed",
        "model": "gpt-test",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "answer now"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            },
        ],
    }

    snapshot = _parse(_capture(request_body=request, response_body=response))

    assert snapshot.instructions == instructions
    assert snapshot.session_id == "session-1"
    assert snapshot.thread_id == "thread-1"
    assert snapshot.turn_id == "turn-1"
    assert [message.role for message in snapshot.history] == [
        "developer",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert snapshot.history[0].content == "developer text"
    assert all(message.content != instructions for message in snapshot.history)
    first_call = snapshot.history[2]
    assert first_call.reasoning_content == "inspect"
    assert first_call.reasoning == {
        "type": "reasoning",
        "id": "rs-1",
        "summary": [{"type": "summary_text", "text": "inspect"}],
        "encrypted_content": "opaque",
    }
    assert first_call.reasoning_details is None
    assert first_call.tool_calls[0].function.arguments == {"path": "/tmp/a"}
    assert snapshot.history[3].content == '{"ok":true}'
    custom_call = snapshot.history[4].tool_calls[0]
    assert custom_call.function.arguments == {"input": "*** Begin Patch\n"}
    assert snapshot.history[5].name == "patch"
    assert snapshot.response[0].content == "answer"
    assert snapshot.response[0].reasoning_content == "answer now"
    assert snapshot.response[0].reasoning == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "answer now"}],
    }
    assert [tool.name for tool in snapshot.tools] == ["read", "patch"]
    assert snapshot.tools[1].parameters["properties"]["input"] == {"type": "string"}
    assert snapshot.tools[1].parameters["x-openai-custom-tool-format"] == {
        "type": "grammar",
        "syntax": "lark",
    }
    assert snapshot.outcome == "success"
    assert snapshot.wire_complete is True
    assert snapshot.termination == ""
    assert snapshot.issues == []


def test_agent_message_is_preserved_raw_without_fabricating_a_role() -> None:
    raw_agent_message = {
        "type": "agent_message",
        "id": "amsg-1",
        "author": "/root/child",
        "recipient": "/root",
        "content": [
            {"type": "input_text", "text": "routing envelope"},
            {"type": "encrypted_content", "encrypted_content": "opaque"},
        ],
        "future_provider_field": {"keep": [1, None, True]},
    }
    request = {
        "model": "gpt-test",
        "input": [
            {"type": "message", "role": "user", "content": "go"},
            {
                "type": "function_call",
                "call_id": "spawn-1",
                "name": "spawn_agent",
                "arguments": '{"task_name":"child","message":"work"}',
            },
            {
                "type": "function_call_output",
                "call_id": "spawn-1",
                "output": '{"task_name":"/root/child"}',
            },
            raw_agent_message,
        ],
    }
    snapshot = _parse(
        _capture(
            request_body=request,
            response_body={
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": "done"}],
            },
        )
    )

    assert [message.role for message in snapshot.history] == [
        "user",
        "assistant",
        "tool",
    ]
    assert len(snapshot.agent_messages) == 1
    record = snapshot.agent_messages[0]
    assert record.origin == "history"
    assert record.item_index == 3
    assert record.item == raw_agent_message
    assert record.preceding_completed_spawn_call_ids == ["spawn-1"]
    assert "unknown_response_item" not in {issue.code for issue in snapshot.issues}


def test_agent_message_between_spawn_call_and_result_has_no_completed_pair() -> None:
    raw_agent_message = {
        "type": "agent_message",
        "id": "amsg-early",
        "author": "/root/child",
        "recipient": "/root",
        "content": [],
    }
    snapshot = _parse(
        _capture(
            request_body={
                "model": "gpt-test",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "spawn-1",
                        "name": "spawn_agent",
                        "arguments": '{"task_name":"child","message":"work"}',
                    },
                    raw_agent_message,
                    {
                        "type": "function_call_output",
                        "call_id": "spawn-1",
                        "output": '{"task_name":"/root/child"}',
                    },
                ],
            },
            response_body={"status": "completed", "output": []},
        )
    )

    assert snapshot.agent_messages[0].item == raw_agent_message
    assert snapshot.agent_messages[0].preceding_completed_spawn_call_ids == []


@pytest.mark.parametrize("field", ["id", "author", "recipient"])
def test_malformed_agent_message_routing_field_is_diagnosed(field: str) -> None:
    item = {
        "type": "agent_message",
        "id": "amsg-1",
        "author": "/root/child",
        "recipient": "/root",
        "content": [],
    }
    item[field] = None
    snapshot = _parse(
        _capture(
            request_body={"model": "gpt-test", "input": [item]},
            response_body={"status": "completed", "output": []},
        )
    )

    assert snapshot.agent_messages[0].item == item
    assert f"invalid_agent_message_{field}" in {issue.code for issue in snapshot.issues}


def test_exact_unique_unsupported_result_emits_provider_evidence() -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "gpt-test",
                "input": [
                    {"type": "message", "role": "user", "content": "go"},
                    {
                        "type": "function_call",
                        "call_id": "call-ghost",
                        "name": "ghost",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call-ghost",
                        "output": "unsupported call: ghost",
                    },
                ],
                "tools": [{"type": "function", "name": "ghost"}],
            },
            response_body={
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": "done"}],
            },
        )
    )

    evidence = [
        issue
        for issue in snapshot.issues
        if issue.code == RESPONSES_UNSUPPORTED_CALL_EVIDENCE
    ]
    assert len(evidence) == 1
    assert evidence[0].stage == "responses"
    assert evidence[0].severity == Severity.WARNING

    node = TrajectoryNode(
        messages=[*snapshot.history, *snapshot.response],
        tools=snapshot.tools,
        harness=snapshot.harness,
        source="one.json",
        metadata=Metadata(source_file="one.json"),
        normalization_audit=NormalizationAudit(
            tag=AuditTag.PASS,
            issues=snapshot.issues,
        ),
    )
    enriched = enrich_trajectory(node)
    assert enriched.tool_call_check.hallucinated_calls == 1
    assert enriched.normalization_audit is not None
    assert enriched.normalization_audit.tag == AuditTag.QUARANTINED


@pytest.mark.parametrize(
    "output",
    [
        " unsupported call: ghost",
        "unsupported call: ghost ",
        "\nunsupported call: ghost",
        "unsupported call: ghost\n",
        "Unsupported call: ghost",
    ],
)
def test_unsupported_result_text_must_match_exactly(output: str) -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "gpt-test",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call-ghost",
                        "name": "ghost",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call-ghost",
                        "output": output,
                    },
                ],
                "tools": [{"type": "function", "name": "ghost"}],
            },
            response_body={
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": "done"}],
            },
        )
    )

    assert RESPONSES_UNSUPPORTED_CALL_EVIDENCE not in {
        issue.code for issue in snapshot.issues
    }


def test_exact_custom_tool_unsupported_result_emits_provider_evidence() -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "gpt-test",
                "input": [
                    {
                        "type": "custom_tool_call",
                        "call_id": "call-patch",
                        "name": "patch",
                        "input": "*** Begin Patch\n",
                    },
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "call-patch",
                        "output": "unsupported custom tool call: patch",
                    },
                ],
                "tools": [{"type": "custom", "name": "patch"}],
            },
            response_body={
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": "done"}],
            },
        )
    )

    evidence = [
        issue
        for issue in snapshot.issues
        if issue.code == RESPONSES_UNSUPPORTED_CALL_EVIDENCE
    ]
    assert len(evidence) == 1
    assert evidence[0].severity == Severity.WARNING


@pytest.mark.parametrize("defect", ["duplicate-call", "duplicate-result", "reversed"])
def test_unsupported_result_requires_unique_forward_pair(defect: str) -> None:
    call = {
        "type": "function_call",
        "call_id": "call-ghost",
        "name": "ghost",
        "arguments": "{}",
    }
    result = {
        "type": "function_call_output",
        "call_id": "call-ghost",
        "output": "unsupported call: ghost",
    }
    if defect == "duplicate-call":
        items = [call, {**call}, result]
    elif defect == "duplicate-result":
        items = [call, result, {**result}]
    else:
        items = [result, call]

    snapshot = _parse(
        _capture(
            request_body={
                "model": "gpt-test",
                "input": [
                    {"type": "message", "role": "user", "content": "go"},
                    *items,
                ],
                "tools": [{"type": "function", "name": "ghost"}],
            },
            response_body={
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": "done"}],
            },
        )
    )

    assert RESPONSES_UNSUPPORTED_CALL_EVIDENCE not in {
        issue.code for issue in snapshot.issues
    }


def test_reconstructs_lossless_custom_call_from_parsed_sse_frames() -> None:
    frames = [
        _frame(
            1,
            {
                "type": "response.created",
                "response": {"id": "resp-1", "status": "in_progress", "output": []},
            },
        ),
        _frame(
            2,
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "rs-1",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "incomplete-ciphertext",
                },
            },
        ),
        _frame(
            3,
            {
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "item_id": "rs-1",
                "summary_index": 0,
                "delta": "think",
            },
        ),
        _frame(
            4,
            {
                "type": "response.reasoning_summary_text.done",
                "output_index": 0,
                "item_id": "rs-1",
                "summary_index": 0,
                "text": "thinking",
            },
        ),
        _frame(
            5,
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "rs-1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thinking"}],
                    "encrypted_content": "complete-ciphertext",
                    "status": "completed",
                },
            },
        ),
        _frame(
            6,
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "ctc-1",
                    "type": "custom_tool_call",
                    "call_id": "call-1",
                    "name": "apply_patch",
                    "input": "",
                },
            },
        ),
        _frame(
            7,
            {
                "type": "response.custom_tool_call_input.delta",
                "output_index": 1,
                "item_id": "ctc-1",
                "delta": "*** Begin ",
            },
        ),
        _frame(
            8,
            {
                "type": "response.custom_tool_call_input.delta",
                "output_index": 1,
                "item_id": "ctc-1",
                "delta": "Patch",
            },
        ),
        _frame(
            9,
            {
                "type": "response.custom_tool_call_input.done",
                "output_index": 1,
                "item_id": "ctc-1",
                "input": "*** Begin Patch",
            },
        ),
        _frame(
            10,
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "id": "ctc-1",
                    "type": "custom_tool_call",
                    "call_id": "call-1",
                    "name": "apply_patch",
                    "input": "*** Begin Patch",
                },
            },
        ),
        _frame(
            11,
            {
                "type": "response.completed",
                "response": {
                    "id": "resp-1",
                    "status": "completed",
                    "model": "gpt-test",
                    # Freerouter terminal projection seen in production: the
                    # custom call is mislabeled and its input is absent.
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "thinking"}],
                            "encrypted_content": "terminal-projection",
                        },
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "apply_patch",
                        },
                    ],
                },
            },
        ),
    ]
    request = {
        "model": "gpt-test",
        "instructions": "",
        "input": [{"type": "message", "role": "user", "content": "go"}],
        "tools": [{"type": "custom", "name": "apply_patch"}],
    }

    snapshot = _parse(_capture(request_body=request, response_body=frames))

    assert snapshot.outcome == "success"
    assert snapshot.wire_complete is True
    assert len(snapshot.response) == 1
    assistant = snapshot.response[0]
    assert assistant.reasoning_content == "thinking"
    assert assistant.reasoning == {
        "id": "rs-1",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "thinking"}],
        "encrypted_content": "complete-ciphertext",
        "status": "completed",
    }
    assert assistant.reasoning_details is None
    assert assistant.tool_calls[0].function.name == "apply_patch"
    assert assistant.tool_calls[0].function.arguments == {"input": "*** Begin Patch"}
    assert snapshot.issues == []


def test_multiple_reasoning_items_start_separate_assistant_segments() -> None:
    first_reasoning = {
        "type": "reasoning",
        "id": "rs-1",
        "summary": [{"type": "summary_text", "text": "first summary"}],
        "encrypted_content": "cipher-one",
        "status": "completed",
    }
    second_reasoning = {
        "type": "reasoning",
        "id": "rs-2",
        "summary": [{"type": "summary_text", "text": "second summary"}],
        "content": [{"type": "reasoning_text", "text": "visible detail"}],
        "text": "legacy text",
        "encrypted_content": "cipher-two",
    }
    response = {
        "status": "completed",
        "output": [
            first_reasoning,
            second_reasoning,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "working"}],
            },
            {
                "type": "custom_tool_call",
                "call_id": "call-1",
                "name": "exec",
                "input": "run",
            },
        ],
    }
    request = {
        "model": "gpt-test",
        "input": [{"type": "message", "role": "user", "content": "go"}],
        "tools": [{"type": "custom", "name": "exec"}],
    }

    snapshot = _parse(_capture(request_body=request, response_body=response))

    assert len(snapshot.response) == 2
    first, second = snapshot.response
    assert first.content == ""
    assert first.reasoning_content == "first summary"
    assert first.reasoning == first_reasoning
    assert first.tool_calls is None
    assert second.content == "working"
    assert second.reasoning_content == "second summary\nvisible detail\nlegacy text"
    assert second.reasoning == second_reasoning
    assert second.tool_calls[0].function.name == "exec"
    assert all(message.reasoning_details is None for message in snapshot.response)
    assert snapshot.issues == []


def test_adjacent_assistant_message_items_preserve_their_boundaries() -> None:
    request = {
        "model": "gpt-test",
        "input": [
            {"type": "message", "role": "user", "content": "go"},
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [
                    {"type": "output_text", "text": "working"},
                    {"type": "output_text", "text": "still working"},
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "history final"}],
            },
        ],
        "tools": [],
    }
    response = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "response update"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "response final"}],
            },
        ],
    }

    snapshot = _parse(_capture(request_body=request, response_body=response))

    assert [message.content for message in snapshot.history] == [
        "go",
        "working\nstill working",
        "history final",
    ]
    assert [message.content for message in snapshot.response] == [
        "response update",
        "response final",
    ]


def test_empty_assistant_message_item_still_preserves_a_boundary() -> None:
    snapshot = _parse(
        _capture(
            request_body={"model": "gpt-test", "input": [], "tools": []},
            response_body={
                "status": "completed",
                "output": [
                    {"type": "message", "role": "assistant", "content": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "done",
                    },
                ],
            },
        )
    )

    assert [message.content for message in snapshot.response] == ["", "done"]


def test_reasoning_and_tool_calls_stay_with_their_assistant_message_item() -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "gpt-test",
                "input": [],
                "tools": [{"type": "function", "name": "read"}],
            },
            response_body={
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "inspect"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "working",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "read",
                        "arguments": "{}",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "done",
                    },
                ],
            },
        )
    )

    assert len(snapshot.response) == 2
    first, second = snapshot.response
    assert first.content == "working"
    assert first.reasoning_content == "inspect"
    assert first.tool_calls is not None
    assert first.tool_calls[0].function.name == "read"
    assert second.content == "done"
    assert second.reasoning_content == ""
    assert second.tool_calls is None


def test_preserved_assistant_boundaries_enable_prefix_matching() -> None:
    first_capture = _capture(
        request_body={
            "model": "gpt-test",
            "input": [{"type": "message", "role": "user", "content": "go"}],
            "tools": [],
        },
        response_body={
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": "working",
                }
            ],
        },
    )
    second_capture = _capture(
        request_body={
            "model": "gpt-test",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": "working",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": "finished",
                },
                {"type": "message", "role": "user", "content": "next"},
            ],
            "tools": [],
        },
        response_body={
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": "next done"}
            ],
        },
    )
    first = parse_responses_capture(
        first_capture,
        source_path="first.json",
        source_sha256="a" * 64,
    )
    second = parse_responses_capture(
        second_capture,
        source_path="second.json",
        source_sha256="b" * 64,
    )

    result = streaming_prefix_leaves([first, second])

    assert [snapshot.source_path for snapshot in result.leaves] == ["second.json"]
    assert result.contributor_paths == {"second.json": ("first.json", "second.json")}


def test_opaque_reasoning_item_is_not_dropped_when_visible_text_is_empty() -> None:
    reasoning = {
        "type": "reasoning",
        "id": "rs-opaque",
        "summary": [],
        "content": [],
        "encrypted_content": "opaque",
    }

    snapshot = _parse(
        _capture(
            request_body={"model": "gpt-test", "input": [], "tools": []},
            response_body={"status": "completed", "output": [reasoning]},
        )
    )

    assert len(snapshot.response) == 1
    assert snapshot.response[0].content == ""
    assert snapshot.response[0].reasoning_content == ""
    assert snapshot.response[0].reasoning == reasoning


def test_server_tools_are_audited_but_not_projected_as_client_messages() -> None:
    request = {
        "model": "gpt-test",
        "input": [
            {"type": "message", "role": "user", "content": "search"},
            {
                "type": "web_search_call",
                "id": "ws-history",
                "status": "completed",
                "action": {"type": "search", "query": "history query"},
            },
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "collaboration",
                        "tools": [
                            {
                                "type": "function",
                                "name": "spawn_agent",
                                "description": "spawn",
                                "parameters": {"type": "object"},
                            }
                        ],
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "local",
                "parameters": {"type": "object"},
            },
            {"type": "web_search"},
        ],
    }
    response = {
        "status": "completed",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws-response",
                "status": "completed",
                "action": {"type": "search", "query": "current query"},
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "found"}],
            },
        ],
    }

    snapshot = _parse(_capture(request_body=request, response_body=response))

    assert [message.role for message in snapshot.history] == ["user"]
    assert [message.role for message in snapshot.response] == ["assistant"]
    assert [tool.name for tool in snapshot.tools] == [
        "local",
        "collaboration.spawn_agent",
    ]
    assert [
        (call.name, call.id, call.origin) for call in snapshot.server_tool_calls
    ] == [
        ("web_search", "ws-history", "history"),
        ("web_search", "ws-response", "response"),
    ]
    assert snapshot.server_tool_calls[0].arguments["query"] == "history query"


def test_server_results_are_lossless_and_link_only_by_call_id() -> None:
    leading_result = {
        "type": "shell_call_output",
        "call_id": "srv-1",
        "status": "completed",
        "output": {"stdout": "first", "exit_code": 0},
        "provider_extra": {"kept": True},
    }
    duplicate_result = {
        "type": "shell_call_output",
        "call_id": "srv-1",
        "id": "result-duplicate",
        "output": "second",
    }
    orphan_result = {
        "type": "shell_call_output",
        "call_id": "orphan",
        "output": ["orphan", {"kept": True}],
    }
    id_only_result = {
        "type": "code_interpreter_call_output",
        "id": "same-item-id",
        "output": "must not link by item.id",
    }
    request = {
        "model": "gpt-test",
        "input": [
            leading_result,
            {
                "type": "shell_call",
                "call_id": "srv-1",
                "id": "shell-item",
                "name": "",
                "arguments": '{"cmd":"pwd"}',
            },
            duplicate_result,
            orphan_result,
            {
                "type": "code_interpreter_call",
                "id": "same-item-id",
                "arguments": {"code": "print(1)"},
            },
            id_only_result,
            {
                "type": "file_search_call",
                "id": "file-item",
                "name": "",
                "queries": ["needle"],
                "results": [{"filename": "a.txt", "score": 0.9}],
            },
            {
                "type": "web_search_call",
                "action": {"type": "search", "query": "missing id"},
            },
        ],
        "tools": [],
    }

    snapshot = _parse(
        _capture(
            request_body=request,
            response_body={"status": "completed", "output": []},
        )
    )

    calls = snapshot.server_tool_calls
    assert len(calls) == 7
    assert calls[0].name == ""
    assert calls[0].id == "srv-1"
    assert calls[0].arguments == {"cmd": "pwd"}
    assert calls[0].result == leading_result
    assert calls[1].id == "srv-1"
    assert calls[1].result == duplicate_result
    assert calls[2].id == "orphan"
    assert calls[2].result == orphan_result

    # Equal item.id values are display identifiers only and never establish
    # call/result linkage without a non-empty call_id on both items.
    assert calls[3].id == "same-item-id"
    assert calls[3].arguments == {"code": "print(1)"}
    assert calls[3].result is None
    assert calls[4].id == "same-item-id"
    assert calls[4].arguments == {}
    assert calls[4].result == id_only_result

    assert calls[5].name == ""
    assert calls[5].arguments == {"queries": ["needle"]}
    assert calls[5].result == {"results": [{"filename": "a.txt", "score": 0.9}]}
    assert calls[6].id == "missing:history:7"
    assert calls[6].arguments == {"type": "search", "query": "missing id"}
    assert {
        "duplicate_server_tool_result",
        "missing_server_tool_id",
        "orphan_server_tool_result",
    }.issubset({issue.code for issue in snapshot.issues})


def test_orphan_server_result_does_not_infer_name_from_result_type() -> None:
    unnamed = {
        "type": "shell_call_output",
        "call_id": "orphan-unnamed",
        "output": "lost",
    }
    explicitly_named = {
        "type": "shell_call_output",
        "call_id": "orphan-named",
        "name": "provider-explicit-name",
        "output": "also lost",
    }
    snapshot = _parse(
        _capture(
            request_body={
                "model": "gpt-test",
                "input": [unnamed, explicitly_named],
                "tools": [],
            },
            response_body={"status": "completed", "output": []},
        )
    )

    assert [call.name for call in snapshot.server_tool_calls] == [
        "",
        "provider-explicit-name",
    ]
    assert [call.result for call in snapshot.server_tool_calls] == [
        unnamed,
        explicitly_named,
    ]
    assert (
        sum(issue.code == "orphan_server_tool_result" for issue in snapshot.issues) == 2
    )


def test_ambiguous_server_result_does_not_infer_name_from_result_type() -> None:
    result = {
        "type": "shell_call_output",
        "call_id": "duplicate-call-id",
        "output": "ambiguous",
    }
    snapshot = _parse(
        _capture(
            request_body={
                "model": "gpt-test",
                "input": [
                    {
                        "type": "shell_call",
                        "call_id": "duplicate-call-id",
                        "arguments": '{"cmd":"first"}',
                    },
                    {
                        "type": "shell_call",
                        "call_id": "duplicate-call-id",
                        "arguments": '{"cmd":"second"}',
                    },
                    result,
                ],
                "tools": [],
            },
            response_body={"status": "completed", "output": []},
        )
    )

    calls = snapshot.server_tool_calls
    assert [call.name for call in calls] == ["shell", "shell", ""]
    assert calls[2].arguments == {}
    assert calls[2].result == result
    assert {
        "duplicate_server_tool_call_id",
        "ambiguous_server_tool_result",
    }.issubset({issue.code for issue in snapshot.issues})


def test_missing_server_result_id_contains_origin_and_json_path() -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "gpt-test",
                "input": [{"type": "shell_call_output", "output": "lost"}],
                "tools": [],
            },
            response_body={"status": "completed", "output": []},
        )
    )

    assert snapshot.server_tool_calls[0].id == (
        "missing:server-result:history:request_body.input[0]"
    )
    assert snapshot.server_tool_calls[0].name == ""
    assert snapshot.server_tool_calls[0].result == {
        "type": "shell_call_output",
        "output": "lost",
    }
    assert "orphan_server_tool_result" in {issue.code for issue in snapshot.issues}


def test_client_linkage_never_falls_back_to_item_id_and_names_may_be_empty() -> None:
    unsupported = "unsupported call: read"
    request = {
        "model": "gpt-test",
        "input": [
            {
                "type": "function_call",
                "id": "shared-item-id",
                "name": "read",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "id": "shared-item-id",
                "output": unsupported,
            },
            {
                "type": "function_call",
                "call_id": "call-empty-name",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call-empty-name",
                "output": "unsupported call: ",
            },
            {
                "type": "custom_tool_call",
                "call_id": "call-wrong-result-type",
                "name": "exec",
                "input": "command",
            },
            {
                "type": "function_call_output",
                "call_id": "call-wrong-result-type",
                "name": "must-not-mask-type-mismatch",
                "output": "unsupported call: exec",
            },
        ],
        "tools": [],
    }

    snapshot = _parse(
        _capture(
            request_body=request,
            response_body={"status": "completed", "output": []},
        )
    )

    assert snapshot.history[0].tool_calls[0].id == "missing:history:0"
    assert snapshot.history[1].tool_call_id == "missing:history:1"
    assert snapshot.history[1].name == ""
    assert snapshot.history[1].content == unsupported
    assert snapshot.history[2].tool_calls[0].function.name == ""
    assert snapshot.history[3].tool_call_id == "call-empty-name"
    assert snapshot.history[3].name == ""
    assert snapshot.history[4].tool_calls[0].function.name == "exec"
    assert snapshot.history[5].tool_call_id == "call-wrong-result-type"
    assert snapshot.history[5].name == ""
    assert "__unknown_tool__" not in snapshot.model_dump_json()
    assert {
        "missing_tool_call_id",
        "missing_tool_result_id",
        "missing_tool_name",
        "orphan_tool_result",
        "tool_result_type_mismatch",
    }.issubset({issue.code for issue in snapshot.issues})


@pytest.mark.parametrize(
    ("events", "expected_outcome", "expected_wire_complete", "issue_code"),
    [
        (
            [
                _frame(
                    1,
                    {
                        "type": "response.failed",
                        "response": {
                            "status": "failed",
                            "error": {"code": "upstream_error"},
                            "output": [],
                        },
                    },
                )
            ],
            "api_error",
            False,
            None,
        ),
        (
            [
                _frame(
                    1,
                    {
                        "type": "response.incomplete",
                        "response": {
                            "status": "incomplete",
                            "incomplete_details": {"reason": "max_output_tokens"},
                            "output": [],
                        },
                    },
                )
            ],
            "truncated",
            False,
            None,
        ),
        (
            [
                _frame(
                    1,
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                ),
                _frame(
                    2,
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "partial",
                    },
                ),
            ],
            "truncated",
            False,
            "missing_terminal_event",
        ),
    ],
)
def test_sse_terminal_outcomes(
    events: list[dict],
    expected_outcome: str,
    expected_wire_complete: bool,
    issue_code: str | None,
) -> None:
    request = {"model": "gpt-test", "input": [], "tools": []}

    snapshot = _parse(_capture(request_body=request, response_body=events))

    assert snapshot.outcome == expected_outcome
    assert snapshot.wire_complete is expected_wire_complete
    if issue_code:
        assert issue_code in {issue.code for issue in snapshot.issues}
        assert snapshot.response[0].content == "partial"


def test_sequence_gap_makes_completed_stream_capture_invalid() -> None:
    events = [
        _frame(
            1,
            {
                "type": "response.created",
                "response": {"status": "in_progress", "output": []},
            },
        ),
        # Both freerouter's frame sequence and OpenAI's sequence_number skip one.
        {
            "seq": 3,
            "event": "response.completed",
            "data": {
                "type": "response.completed",
                "sequence_number": 2,
                "response": {"status": "completed", "output": []},
            },
        },
    ]

    snapshot = _parse(
        _capture(
            request_body={"model": "gpt-test", "input": [], "tools": []},
            response_body=events,
        )
    )

    assert snapshot.outcome == "capture_invalid"
    assert snapshot.wire_complete is False
    assert {"sse_sequence_gap", "event_sequence_gap"}.issubset(
        {issue.code for issue in snapshot.issues}
    )


def test_non_string_instructions_are_quarantinable_without_string_coercion() -> None:
    capture = _capture(
        request_body={"model": "gpt-test", "instructions": ["not", "valid"]},
        response_body={"status": "completed", "output": []},
    )

    snapshot = _parse(capture)

    assert snapshot.instructions == ""
    assert "invalid_instructions" in {issue.code for issue in snapshot.issues}
    assert snapshot.outcome == "capture_invalid"


def test_whitelisted_headers_fill_identity_and_conflicts_are_reported() -> None:
    request = {
        "model": "gpt-test",
        "client_metadata": {
            "session_id": "session-1",
            "thread_id": "body-thread",
            "turn_id": "turn-1",
        },
        "input": [],
        "tools": [],
    }
    capture = _capture(
        request_body=request,
        response_body={"status": "completed", "output": []},
    )
    capture["session_id"] = "session-1"
    capture["request_headers"] = {
        "Authorization": "Bearer must-not-survive",
        "X-Codex-Turn-Metadata": json.dumps(
            {
                "session_id": "session-1",
                "thread_id": "header-thread",
                "turn_id": "turn-1",
                "forked_from_thread_id": "fork-1",
            }
        ),
        "x-codex-parent-thread-id": "parent-1",
        "x-openai-subagent": "guardian",
    }

    snapshot = _parse(capture)

    assert snapshot.thread_id == "body-thread"
    assert snapshot.parent_thread_id == "parent-1"
    assert snapshot.forked_from_thread_id == "fork-1"
    assert snapshot.subagent_marker == "guardian"
    assert "metadata_conflict" in {issue.code for issue in snapshot.issues}
    assert snapshot.outcome == "capture_invalid"
    assert "must-not-survive" not in snapshot.model_dump_json()


def test_missing_http_status_is_transport_error_even_with_a_body() -> None:
    capture = _capture(
        request_body={"model": "gpt-test", "input": [], "tools": []},
        response_body={"status": "completed", "output": []},
    )
    capture["status_code"] = None

    snapshot = _parse(capture)

    assert snapshot.outcome == "transport_error"
    assert snapshot.wire_complete is False
    assert "missing_http_status" in {issue.code for issue in snapshot.issues}


def test_http_error_body_is_summarized_without_rewriting_trajectory_content() -> None:
    user_content = "Keep this user payload verbatim: Bearer USER-PAYLOAD"
    request = {
        "model": "gpt-test",
        "input": [{"type": "message", "role": "user", "content": user_content}],
        "tools": [],
    }
    snapshot = _parse(
        _capture(
            request_body=request,
            status_code=401,
            response_body={
                "error": {
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                    "message": (
                        "Authorization: Bearer TOP-SECRET, api_key=TOP-SECRET, "
                        "proxy credential=proxy-password, forwarded IP=198.51.100.8"
                    ),
                }
            },
        )
    )

    assert snapshot.history[0].content == user_content
    assert snapshot.issues[0].detail == (
        "OpenAI-compatible API error "
        "(HTTP 401; type=authentication_error; code=invalid_api_key)"
    )
    serialized = snapshot.model_dump_json()
    for secret in ("TOP-SECRET", "proxy-password", "198.51.100.8"):
        assert secret not in serialized


def test_audit_issue_detail_has_defense_in_depth_redaction() -> None:
    issue = AuditIssue(
        code="unsafe_exception",
        stage="test",
        detail=(
            'authorization="Bearer TOP-SECRET"; api_key=TOP-SECRET; '
            "proxy-authorization: Basic cHJveHk6cGFzcw==; "
            "forwarded IP 192.0.2.25"
        ),
    )

    serialized = issue.model_dump_json()
    assert "TOP-SECRET" not in serialized
    assert "cHJveHk6cGFzcw==" not in serialized
    assert "192.0.2.25" not in serialized
    assert "[REDACTED]" in issue.detail


def test_rejects_non_responses_endpoint() -> None:
    capture = _capture(request_body={}, response_body={})
    capture["path"] = "/v1/messages"

    with pytest.raises(ResponsesAdapterError):
        _parse(capture)


def test_json_sse_data_and_unparseable_arguments_are_preserved() -> None:
    event = {
        "type": "response.completed",
        "sequence_number": 0,
        "response": {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-raw",
                    "name": "raw_tool",
                    "arguments": "not-json",
                }
            ],
        },
    }
    capture = _capture(
        request_body={"model": "gpt-test", "input": [], "tools": []},
        response_body=[
            {"seq": 1, "event": "response.completed", "data": json.dumps(event)}
        ],
    )

    snapshot = _parse(capture)

    assert snapshot.response[0].tool_calls[0].function.arguments == {"raw": "not-json"}


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_constants_in_client_and_server_arguments_are_raw(
    constant: str,
) -> None:
    snapshot = _parse(
        _capture(
            request_body={"model": "gpt-test", "input": [], "tools": []},
            response_body={
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "client-raw",
                        "name": "local",
                        "arguments": constant,
                    },
                    {
                        "type": "shell_call",
                        "call_id": "server-raw",
                        "arguments": constant,
                    },
                ],
            },
        )
    )

    assert snapshot.response[0].tool_calls
    assert snapshot.response[0].tool_calls[0].function.arguments == {"raw": constant}
    assert snapshot.server_tool_calls[0].arguments == {"raw": constant}


def test_subagent_without_explicit_thread_id_is_isolated() -> None:
    snapshot = _parse(
        _capture(
            request_body={
                "model": "gpt-test",
                "client_metadata": {
                    "session_id": "session-1",
                    "parent_thread_id": "main-thread",
                    "parent_turn_id": "turn-1",
                    "subagent_marker": "collab_spawn",
                },
                "input": [],
                "tools": [],
            },
            response_body={"status": "completed", "output": []},
        )
    )

    assert snapshot.thread_id == "__isolated_subagent__:/captures/one.json"
    issue = next(
        issue
        for issue in snapshot.issues
        if issue.code == "isolated_subagent_missing_thread_id"
    )
    assert issue.severity == Severity.WARNING


def test_main_thread_without_explicit_thread_id_uses_session_id() -> None:
    snapshot = _parse(
        _capture(
            request_body={"model": "gpt-test", "input": [], "tools": []},
            response_body={"status": "completed", "output": []},
        )
    )

    assert snapshot.thread_id == "capture-session"
