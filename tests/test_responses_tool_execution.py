from __future__ import annotations

import json
from typing import Any

import pytest

from trajfoundry.models import Message, Severity, Snapshot, ToolCall
from trajfoundry.providers.responses import parse_responses_capture


def _capture(
    *,
    tools: list[dict[str, Any]] | None = None,
    input_items: list[dict[str, Any]] | None = None,
    output_items: list[dict[str, Any]] | None = None,
    response_body: object | None = None,
) -> dict[str, Any]:
    if response_body is None:
        response_body = {
            "id": "resp-1",
            "model": "gpt-test",
            "status": "completed",
            "output": output_items or [],
        }
    return {
        "path": "/v1/responses",
        "captured_at": "2026-08-28T01:02:03+00:00",
        "session_id": "session-1",
        "request_id": "request-1",
        "status_code": 200,
        "is_stream": isinstance(response_body, list),
        "request_body": {
            "model": "gpt-test",
            "client_metadata": {
                "session_id": "session-1",
                "thread_id": "thread-1",
                "turn_id": "turn-1",
            },
            "input": input_items or [],
            "tools": tools or [],
        },
        "response_body": response_body,
    }


def _parse(capture: dict[str, Any]) -> Snapshot:
    return parse_responses_capture(
        capture,
        source_path="/captures/tool-execution.json",
        source_sha256="a" * 64,
    )


def _messages(snapshot: Snapshot) -> list[Message]:
    return [*snapshot.history, *snapshot.response]


def _client_calls(snapshot: Snapshot) -> list[ToolCall]:
    return [
        call for message in _messages(snapshot) for call in (message.tool_calls or [])
    ]


def _tool_results(snapshot: Snapshot) -> list[Message]:
    return [message for message in _messages(snapshot) if message.role == "tool"]


def _error_codes(snapshot: Snapshot) -> set[str]:
    return {issue.code for issue in snapshot.issues if issue.severity == Severity.ERROR}


def _tool_search_definition(
    execution: str | None,
) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "type": "tool_search",
        "description": "Find project-specific tools.",
        "parameters": {
            "type": "object",
            "properties": {"goal": {"type": "string"}},
            "required": ["goal"],
            "additionalProperties": False,
        },
    }
    if execution is not None:
        definition["execution"] = execution
    return definition


def _loaded_function(name: str = "lookup_eta") -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": "Look up an ETA.",
        "defer_loading": True,
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    }


def test_client_tool_search_is_a_client_call_and_preserves_loaded_tools() -> None:
    loaded = _loaded_function()
    output = {
        "type": "tool_search_output",
        "execution": "client",
        "call_id": "search-1",
        "status": "completed",
        "tools": [loaded],
    }
    snapshot = _parse(
        _capture(
            tools=[_tool_search_definition("client")],
            input_items=[
                {
                    "type": "tool_search_call",
                    "execution": "client",
                    "call_id": "search-1",
                    "status": "completed",
                    "arguments": {"goal": "find an ETA tool"},
                },
                output,
            ],
        )
    )

    calls = _client_calls(snapshot)
    results = _tool_results(snapshot)
    assert snapshot.server_tool_calls == []
    assert [
        (call.id, call.function.name, call.function.arguments) for call in calls
    ] == [("search-1", "tool_search", {"goal": "find an ETA tool"})]
    assert [(result.tool_call_id, result.name) for result in results] == [
        ("search-1", "tool_search")
    ]
    assert json.loads(results[0].content) == {
        "status": "completed",
        "tools": [loaded],
    }
    assert [tool.name for tool in snapshot.tools] == ["tool_search", "lookup_eta"]
    assert (
        snapshot.tools[0].parameters == _tool_search_definition("client")["parameters"]
    )


def test_hosted_tool_search_is_only_a_server_call() -> None:
    output = {
        "type": "tool_search_output",
        "execution": "server",
        "call_id": "search-2",
        "status": "completed",
        "tools": [],
    }
    snapshot = _parse(
        _capture(
            tools=[_tool_search_definition("server")],
            output_items=[
                {
                    "type": "tool_search_call",
                    "execution": "server",
                    "call_id": "search-2",
                    "status": "completed",
                    "arguments": {"paths": ["crm"]},
                },
                output,
            ],
        )
    )

    assert _client_calls(snapshot) == []
    assert _tool_results(snapshot) == []
    assert snapshot.tools == []
    assert len(snapshot.server_tool_calls) == 1
    call = snapshot.server_tool_calls[0]
    assert (call.id, call.name, call.arguments, call.origin) == (
        "search-2",
        "tool_search",
        {"paths": ["crm"]},
        "response",
    )
    assert call.result == output


def test_tool_search_definition_without_execution_defaults_to_hosted() -> None:
    snapshot = _parse(
        _capture(
            tools=[_tool_search_definition(None)],
            output_items=[
                {
                    "type": "tool_search_call",
                    "call_id": "search-default",
                    "arguments": {"paths": ["crm"]},
                }
            ],
        )
    )

    assert _client_calls(snapshot) == []
    assert snapshot.tools == []
    assert [call.name for call in snapshot.server_tool_calls] == ["tool_search"]
    assert "ambiguous_tool_execution" not in _error_codes(snapshot)


def test_missing_tool_search_execution_falls_back_to_paired_item() -> None:
    output = {
        "type": "tool_search_output",
        "execution": "client",
        "call_id": "search-paired",
        "status": "completed",
        "tools": [],
    }
    snapshot = _parse(
        _capture(
            input_items=[
                {
                    "type": "tool_search_call",
                    "call_id": "search-paired",
                    "arguments": {"goal": "find a tool"},
                },
                output,
            ]
        )
    )

    assert snapshot.server_tool_calls == []
    assert [call.function.name for call in _client_calls(snapshot)] == ["tool_search"]
    assert [result.name for result in _tool_results(snapshot)] == ["tool_search"]
    assert "ambiguous_tool_execution" not in _error_codes(snapshot)


def test_tool_search_without_execution_evidence_is_ambiguous() -> None:
    snapshot = _parse(
        _capture(
            output_items=[
                {
                    "type": "tool_search_call",
                    "call_id": "search-ambiguous",
                    "arguments": {"goal": "find a tool"},
                }
            ]
        )
    )

    assert _client_calls(snapshot) == []
    assert snapshot.server_tool_calls == []
    assert snapshot.outcome == "capture_invalid"
    assert "ambiguous_tool_execution" in _error_codes(snapshot)


def test_conflicting_tool_search_execution_is_quarantinable() -> None:
    snapshot = _parse(
        _capture(
            tools=[_tool_search_definition("client")],
            output_items=[
                {
                    "type": "tool_search_call",
                    "execution": "client",
                    "call_id": "search-conflict",
                    "arguments": {"goal": "find a tool"},
                },
                {
                    "type": "tool_search_output",
                    "execution": "server",
                    "call_id": "search-conflict",
                    "status": "completed",
                    "tools": [],
                },
            ],
        )
    )

    assert snapshot.outcome == "capture_invalid"
    assert "tool_execution_mismatch" in _error_codes(snapshot)


def test_tool_search_item_conflicting_with_current_definition_is_reported() -> None:
    snapshot = _parse(
        _capture(
            tools=[_tool_search_definition("client")],
            output_items=[
                {
                    "type": "tool_search_call",
                    "execution": "server",
                    "call_id": "search-definition-conflict",
                    "arguments": {"paths": ["crm"]},
                }
            ],
        )
    )

    assert snapshot.outcome == "capture_invalid"
    assert [call.name for call in snapshot.server_tool_calls] == ["tool_search"]
    assert _client_calls(snapshot) == []
    assert "tool_execution_mismatch" in _error_codes(snapshot)


@pytest.mark.parametrize(
    ("definition_type", "item_type", "expected_name"),
    [
        ("computer", "computer_call", "computer"),
        ("apply_patch", "apply_patch_call", "apply_patch"),
        ("local_shell", "local_shell_call", "local_shell"),
    ],
)
def test_client_builtins_are_projected_as_client_tool_messages(
    definition_type: str,
    item_type: str,
    expected_name: str,
) -> None:
    call_id = f"{expected_name}-1"
    snapshot = _parse(
        _capture(
            tools=[{"type": definition_type}],
            output_items=[
                {
                    "type": item_type,
                    "call_id": call_id,
                    "arguments": {"marker": expected_name},
                },
                {
                    "type": f"{item_type}_output",
                    "call_id": call_id,
                    "output": {"marker": expected_name, "ok": True},
                },
            ],
        )
    )

    calls = _client_calls(snapshot)
    results = _tool_results(snapshot)
    assert snapshot.server_tool_calls == []
    assert [(call.id, call.function.name) for call in calls] == [
        (call_id, expected_name)
    ]
    assert calls[0].function.arguments == {"marker": expected_name}
    assert [(result.tool_call_id, result.name) for result in results] == [
        (call_id, expected_name)
    ]
    assert expected_name in results[0].content
    assert [(tool.name, tool.parameters) for tool in snapshot.tools] == [
        (expected_name, {})
    ]


def test_client_builtin_without_definition_does_not_invent_one() -> None:
    snapshot = _parse(
        _capture(
            output_items=[
                {
                    "type": "apply_patch_call",
                    "call_id": "patch-undefined",
                    "operation": {
                        "type": "delete_file",
                        "path": "obsolete.txt",
                    },
                },
                {
                    "type": "apply_patch_call_output",
                    "call_id": "patch-undefined",
                    "status": "failed",
                    "output": "file not found",
                },
            ]
        )
    )

    assert [call.function.name for call in _client_calls(snapshot)] == ["apply_patch"]
    assert snapshot.tools == []
    assert json.loads(_tool_results(snapshot)[0].content) == {
        "output": "file not found",
        "status": "failed",
    }


def test_local_shell_environment_is_a_client_tool() -> None:
    snapshot = _parse(
        _capture(
            tools=[{"type": "shell", "environment": {"type": "local"}}],
            output_items=[
                {
                    "type": "shell_call",
                    "call_id": "shell-local",
                    "arguments": {"command": ["pwd"]},
                },
                {
                    "type": "shell_call_output",
                    "call_id": "shell-local",
                    "output": {"stdout": "/workspace\n", "exit_code": 0},
                },
            ],
        )
    )

    assert snapshot.server_tool_calls == []
    assert [call.function.name for call in _client_calls(snapshot)] == ["shell"]
    assert [result.name for result in _tool_results(snapshot)] == ["shell"]
    assert [(tool.name, tool.parameters) for tool in snapshot.tools] == [("shell", {})]


def test_shell_without_environment_evidence_is_ambiguous() -> None:
    snapshot = _parse(
        _capture(
            output_items=[
                {
                    "type": "shell_call",
                    "call_id": "shell-ambiguous",
                    "arguments": {"command": ["pwd"]},
                },
                {
                    "type": "shell_call_output",
                    "call_id": "shell-ambiguous",
                    "output": {"stdout": "/workspace\n", "exit_code": 0},
                },
            ]
        )
    )

    assert _client_calls(snapshot) == []
    assert _tool_results(snapshot) == []
    assert snapshot.server_tool_calls == []
    assert snapshot.outcome == "capture_invalid"
    assert "ambiguous_tool_execution" in _error_codes(snapshot)


def test_shell_item_environment_takes_precedence_and_reports_definition_conflict() -> (
    None
):
    snapshot = _parse(
        _capture(
            tools=[{"type": "shell", "environment": {"type": "container_auto"}}],
            output_items=[
                {
                    "type": "shell_call",
                    "call_id": "shell-conflict",
                    "environment": {"type": "local"},
                    "arguments": {"command": ["pwd"]},
                }
            ],
        )
    )

    assert [call.function.name for call in _client_calls(snapshot)] == ["shell"]
    assert snapshot.server_tool_calls == []
    assert snapshot.outcome == "capture_invalid"
    assert "tool_execution_mismatch" in _error_codes(snapshot)


@pytest.mark.parametrize("environment_type", ["container_auto", "container_reference"])
def test_hosted_shell_environment_is_a_server_tool(environment_type: str) -> None:
    result = {
        "type": "shell_call_output",
        "call_id": "shell-hosted",
        "output": {"stdout": "/mnt/data\n", "exit_code": 0},
    }
    snapshot = _parse(
        _capture(
            tools=[
                {
                    "type": "shell",
                    "environment": {
                        "type": environment_type,
                        "container_id": "container-1",
                    },
                }
            ],
            output_items=[
                {
                    "type": "shell_call",
                    "call_id": "shell-hosted",
                    "arguments": {"command": ["pwd"]},
                },
                result,
            ],
        )
    )

    assert _client_calls(snapshot) == []
    assert _tool_results(snapshot) == []
    assert snapshot.tools == []
    assert len(snapshot.server_tool_calls) == 1
    call = snapshot.server_tool_calls[0]
    assert (call.name, call.id, call.arguments) == (
        "shell",
        "shell-hosted",
        {"command": ["pwd"]},
    )
    assert call.result == result


def test_web_search_remains_a_server_tool() -> None:
    snapshot = _parse(
        _capture(
            tools=[{"type": "web_search"}],
            output_items=[
                {
                    "type": "web_search_call",
                    "call_id": "web-1",
                    "status": "completed",
                    "action": {"type": "search", "query": "OpenAI"},
                }
            ],
        )
    )

    assert _client_calls(snapshot) == []
    assert snapshot.tools == []
    assert [
        (call.name, call.id, call.arguments) for call in snapshot.server_tool_calls
    ] == [("web_search", "web-1", {"type": "search", "query": "OpenAI"})]


def test_unknown_call_suffixes_are_not_silently_classified_as_server_tools() -> None:
    snapshot = _parse(
        _capture(
            output_items=[
                {
                    "type": "future_widget_call",
                    "call_id": "future-1",
                    "arguments": {"value": 1},
                },
                {
                    "type": "future_widget_call_output",
                    "call_id": "future-1",
                    "output": {"ok": True},
                },
            ]
        )
    )

    assert _client_calls(snapshot) == []
    assert _tool_results(snapshot) == []
    assert snapshot.server_tool_calls == []
    assert snapshot.outcome == "capture_invalid"
    assert {
        issue.path for issue in snapshot.issues if issue.severity == Severity.ERROR
    } >= {"response.output[0]", "response.output[1]"}


@pytest.mark.parametrize(
    ("execution", "origin"),
    [("client", "history"), ("server", "response")],
)
def test_tool_search_output_adds_dynamic_client_definitions(
    execution: str,
    origin: str,
) -> None:
    loaded_tools = [
        _loaded_function("lookup_eta"),
        {
            "type": "custom",
            "name": "edit_file",
            "description": "Edit a file.",
            "format": {"type": "text"},
        },
        {
            "type": "namespace",
            "name": "ops",
            "description": "Operational tools.",
            "tools": [
                {
                    "type": "function",
                    "name": "restart",
                    "description": "Restart a service.",
                    "parameters": {
                        "type": "object",
                        "properties": {"service": {"type": "string"}},
                    },
                }
            ],
        },
    ]
    call = {
        "type": "tool_search_call",
        "execution": execution,
        "call_id": "dynamic-search",
        "arguments": {"goal": "load operational tools"},
    }
    output = {
        "type": "tool_search_output",
        "execution": execution,
        "call_id": "dynamic-search",
        "status": "completed",
        "tools": loaded_tools,
    }
    items = [call, output]
    snapshot = _parse(
        _capture(
            tools=[_tool_search_definition(execution)],
            input_items=items if origin == "history" else [],
            output_items=items if origin == "response" else [],
        )
    )

    definitions = {tool.name: tool for tool in snapshot.tools}
    assert {"lookup_eta", "edit_file", "ops.restart"} <= definitions.keys()
    assert definitions["lookup_eta"].parameters == loaded_tools[0]["parameters"]
    assert definitions["edit_file"].parameters["properties"] == {
        "input": {"type": "string"}
    }
    assert (
        definitions["ops.restart"].parameters
        == loaded_tools[2]["tools"][0]["parameters"]
    )


def test_namespace_calls_and_results_use_the_full_name() -> None:
    namespace = lambda name: {
        "type": "namespace",
        "name": name,
        "description": f"{name} tools",
        "tools": [
            {
                "type": "function",
                "name": "run",
                "description": f"Run {name}.",
                "parameters": {"type": "object"},
            }
        ],
    }
    snapshot = _parse(
        _capture(
            tools=[namespace("alpha"), namespace("beta")],
            input_items=[
                {
                    "type": "function_call",
                    "namespace": "alpha",
                    "name": "run",
                    "call_id": "alpha-run",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "alpha-run",
                    "output": "alpha done",
                },
                {
                    "type": "function_call",
                    "namespace": "beta",
                    "name": "run",
                    "call_id": "beta-run",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "beta-run",
                    "output": "beta done",
                },
            ],
        )
    )

    assert [tool.name for tool in snapshot.tools] == ["alpha.run", "beta.run"]
    assert [call.function.name for call in _client_calls(snapshot)] == [
        "alpha.run",
        "beta.run",
    ]
    assert [result.name for result in _tool_results(snapshot)] == [
        "alpha.run",
        "beta.run",
    ]


@pytest.mark.parametrize("provider_id", [None, "tool-search-item-1"])
def test_hosted_tool_search_with_null_call_id_pairs_adjacent_items(
    provider_id: str | None,
) -> None:
    call: dict[str, Any] = {
        "type": "tool_search_call",
        "execution": "server",
        "call_id": None,
        "status": "completed",
        "arguments": {"paths": ["crm"]},
    }
    if provider_id is not None:
        call["id"] = provider_id
    output = {
        "type": "tool_search_output",
        "execution": "server",
        "call_id": None,
        "status": "completed",
        "tools": [],
    }
    capture = _capture(
        tools=[_tool_search_definition(None)],
        output_items=[call, output],
    )

    first = _parse(capture)
    second = _parse(capture)

    assert len(first.server_tool_calls) == 1
    normalized = first.server_tool_calls[0]
    assert normalized.name == "tool_search"
    assert normalized.arguments == {"paths": ["crm"]}
    assert normalized.result == output
    assert normalized.id == second.server_tool_calls[0].id
    if provider_id is not None:
        assert normalized.id == provider_id
    assert {
        "missing_server_tool_id",
        "orphan_server_tool_result",
    }.isdisjoint(_error_codes(first))


def test_streamed_tool_search_uses_the_same_execution_classification() -> None:
    terminal = {
        "type": "response.completed",
        "sequence_number": 0,
        "response": {
            "id": "resp-stream",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "tool_search_call",
                    "execution": "client",
                    "call_id": "search-stream",
                    "arguments": {"goal": "find a stream tool"},
                }
            ],
        },
    }
    snapshot = _parse(
        _capture(
            tools=[_tool_search_definition("client")],
            response_body=[
                {
                    "seq": 1,
                    "event": "response.completed",
                    "data": terminal,
                }
            ],
        )
    )

    assert snapshot.server_tool_calls == []
    assert [(call.id, call.function.name) for call in _client_calls(snapshot)] == [
        ("search-stream", "tool_search")
    ]
