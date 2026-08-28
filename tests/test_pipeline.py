from __future__ import annotations

import sqlite3
from pathlib import Path

import orjson
import pytest

from trajfoundry.models import (
    AgentMessageEvidence,
    AuditIssue,
    AuditTag,
    FunctionCall,
    Message,
    ServerToolCall,
    Severity,
    Snapshot,
    ToolCall,
    ToolDefinition,
)
from trajfoundry.pipeline import (
    PipelineConfig,
    PipelineStats,
    _build_trajectories,
    _exclusive_lock,
    _flat_semantic_key,
    _merge_contributor_metadata,
    _trajectory_from_leaf,
    normalize,
)
from trajfoundry.state import StateStore
from trajfoundry.validation import validate_output


def _message(role: str, text: str) -> dict[str, object]:
    content_type = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": content_type, "text": text}],
    }


def _capture(
    *,
    captured_at: str,
    turn_id: str,
    request_input: list[dict[str, object]],
    response_output: list[dict[str, object]],
    tools: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "path": "/v1/responses",
        "session_id": "session",
        "request_id": turn_id,
        "captured_at": captured_at,
        "status_code": 200,
        "is_stream": False,
        "request_headers": {"authorization": "must-not-leak"},
        "response_headers": {"set-cookie": "must-not-leak"},
        "request_body": {
            "model": "gpt-test",
            "instructions": "stay exact",
            "client_metadata": {
                "session_id": "session",
                "thread_id": "thread",
                "turn_id": turn_id,
            },
            "input": request_input,
            "tools": tools,
        },
        "response_body": {
            "id": f"response-{turn_id}",
            "status": "completed",
            "model": "gpt-test",
            "output": response_output,
        },
    }


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(value))


def _rows(path: Path) -> list[dict[str, object]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines()]


def _output_file(root: Path, suffix: str) -> Path:
    manifest = orjson.loads((root / "manifest.json").read_bytes())
    relative = next(
        entry["path"] for entry in manifest["files"] if entry["path"].endswith(suffix)
    )
    return root / relative


def test_end_to_end_prefix_tool_union_resume_and_deleted_input(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    session = input_root / "v1" / "dt=2026-08-27" / "partition" / "session"
    developer = _message("developer", "keep developer")
    user = _message("user", "question")
    call = {
        "type": "function_call",
        "call_id": "call-1",
        "name": "lookup",
        "arguments": '{"query":"x"}',
    }
    result = {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "result",
    }
    raw_agent_message = {
        "type": "agent_message",
        "id": "amsg-1",
        "author": "/root/child",
        "recipient": "/root",
        "content": [
            {"type": "input_text", "text": "envelope"},
            {"type": "encrypted_content", "encrypted_content": "opaque"},
        ],
        "unknown": {"preserve": True},
    }
    tool = {
        "type": "function",
        "name": "lookup",
        "description": "Look up a value",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    first_path = session / "a.json"
    # The same logical session can be physically split below different storage
    # parents.  Input location is provenance, not an aggregation boundary.
    second_path = (
        input_root / "v1" / "dt=2026-08-27" / "other-partition" / "session" / "b.json"
    )
    _write(
        first_path,
        _capture(
            captured_at="2026-08-27T00:00:00Z",
            turn_id="turn-1",
            request_input=[developer, user],
            response_output=[call],
            tools=[tool],
        ),
    )
    _write(
        second_path,
        _capture(
            captured_at="2026-08-27T00:01:00Z",
            turn_id="turn-2",
            request_input=[developer, user, call, result, raw_agent_message],
            response_output=[_message("assistant", "done")],
            tools=[],
        ),
    )

    first = normalize(PipelineConfig(input_root=input_root, output_root=output_root))
    assert first.discovered == 2
    assert first.prefix_intermediates == 1
    accepted_path = _output_file(output_root, "/accepted/trajectories-00000.jsonl")
    row = _rows(accepted_path)[0]
    assert row["instructions"] == "stay exact"
    assert [message["role"] for message in row["messages"][:2]] == [
        "developer",
        "user",
    ]
    assert [definition["name"] for definition in row["tools"]] == ["lookup"]
    assert row["agent_messages"] == [
        {
            "origin": "history",
            "item_index": 4,
            "item": raw_agent_message,
        }
    ]
    serialized = orjson.dumps(row)
    assert b"must-not-leak" not in serialized
    lineage_path = _output_file(output_root, "/lineage.jsonl")
    lineage = _rows(lineage_path)
    assert lineage[0]["origin_count"] == 2
    assert validate_output(output_root).valid
    accepted_bytes = accepted_path.read_bytes()
    lineage_bytes = lineage_path.read_bytes()

    resumed = normalize(
        PipelineConfig(input_root=input_root, output_root=output_root, resume=True)
    )
    assert resumed.reused == 2
    assert (
        _output_file(output_root, "/accepted/trajectories-00000.jsonl").read_bytes()
        == accepted_bytes
    )
    assert _output_file(output_root, "/lineage.jsonl").read_bytes() == lineage_bytes

    first_path.unlink()
    normalize(
        PipelineConfig(input_root=input_root, output_root=output_root, resume=True)
    )
    manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    assert not any("/accepted/" in entry["path"] for entry in manifest["files"])
    quarantined_path = _output_file(output_root, "/quarantine/trajectories-00000.jsonl")
    quarantined = _rows(quarantined_path)[0]
    assert quarantined["tool_defs_tag"] == "incomplete"
    assert validate_output(output_root).valid


def test_provider_rejection_prevents_cross_provider_trajectory_dedup(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    responses_path = input_root / "partition" / "responses" / "one.json"
    anthropic_path = input_root / "partition" / "anthropic" / "two.json"
    tool_name = "ghost"
    call_id = "call-ghost"
    rejection = "unsupported call: ghost"

    _write(
        responses_path,
        {
            "path": "/v1/responses",
            "session_id": "shared-session",
            "request_id": "responses-turn",
            "captured_at": "2026-08-27T00:00:00Z",
            "status_code": 200,
            "is_stream": False,
            "request_body": {
                "model": "same-model",
                "client_metadata": {
                    "session_id": "shared-session",
                    "thread_id": "responses-thread",
                    "turn_id": "responses-turn",
                    "harness": "shared-harness",
                },
                "input": [
                    {"type": "message", "role": "user", "content": "go"},
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": rejection,
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": tool_name,
                        "description": "",
                        "parameters": {},
                    }
                ],
            },
            "response_body": {
                "status": "completed",
                "model": "same-model",
                "output": [{"type": "message", "role": "assistant", "content": "done"}],
            },
        },
    )
    _write(
        anthropic_path,
        {
            "path": "/v1/messages",
            "session_id": "shared-session",
            "request_id": "anthropic-turn",
            "captured_at": "2026-08-27T00:01:00Z",
            "status_code": 200,
            "is_stream": False,
            "harness": "shared-harness",
            "request_body": {
                "model": "same-model",
                "metadata": {
                    "session_id": "shared-session",
                    "thread_id": "anthropic-thread",
                    "turn_id": "anthropic-turn",
                },
                "messages": [
                    {"role": "user", "content": "go"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": call_id,
                                "name": tool_name,
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": rejection,
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "name": tool_name,
                        "description": "",
                        "input_schema": {},
                    }
                ],
            },
            "response_body": {
                "type": "message",
                "role": "assistant",
                "model": "same-model",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            },
        },
    )

    stats = normalize(PipelineConfig(input_root=input_root, output_root=output_root))

    accepted = _rows(_output_file(output_root, "/accepted/trajectories-00000.jsonl"))
    quarantined = _rows(
        _output_file(output_root, "/quarantine/trajectories-00000.jsonl")
    )
    lineage = _rows(_output_file(output_root, "/lineage.jsonl"))
    assert stats.stored_trajectories == 2
    assert len(accepted) == len(quarantined) == 1
    assert accepted[0]["messages"] == quarantined[0]["messages"]
    assert accepted[0]["tools"] == quarantined[0]["tools"]
    assert accepted[0]["normalization_audit"]["tag"] == "pass"
    assert quarantined[0]["normalization_audit"]["tag"] == "quarantined"
    assert quarantined[0]["tool_call_check"]["hallucinated_calls"] == 1
    assert len(lineage) == 2
    assert len({row["trajectory_id"] for row in lineage}) == 2
    assert validate_output(output_root).valid


def test_nonempty_output_requires_resume(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "owned-by-user.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        normalize(PipelineConfig(input_root=input_root, output_root=output_root))


@pytest.mark.parametrize("stored_hash", [None, "stale-configuration"])
def test_resume_rebuilds_when_configuration_hash_is_missing_or_changed(
    tmp_path: Path, stored_hash: str | None
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write(
        input_root / "partition" / "session" / "one.json",
        _capture(
            captured_at="2026-08-27T00:00:00Z",
            turn_id="turn-1",
            request_input=[_message("user", "hello")],
            response_output=[_message("assistant", "done")],
            tools=[],
        ),
    )
    normalize(PipelineConfig(input_root=input_root, output_root=output_root))
    state_path = output_root / ".state" / "trajfoundry.sqlite"
    with sqlite3.connect(state_path) as connection:
        connection.execute("DELETE FROM meta WHERE key='config_hash'")
        if stored_hash is not None:
            connection.execute(
                "INSERT INTO meta(key,value) VALUES('config_hash',?)", (stored_hash,)
            )

    resumed = normalize(
        PipelineConfig(input_root=input_root, output_root=output_root, resume=True)
    )

    assert resumed.reused == 0
    assert resumed.parsed == 1
    assert validate_output(output_root).valid


def test_server_tool_duplicate_results_survive_contributor_merge() -> None:
    snapshot = Snapshot(
        source_path="one.json",
        source_sha256="a" * 64,
        session_id="session",
        thread_id="thread",
        provider="anthropic",
        operation="messages",
        outcome="success",
        server_tool_calls=[
            ServerToolCall(
                name="web_search",
                id="server-1",
                arguments={"query": "x"},
                origin="response",
                result={"type": "web_search_tool_result", "content": ["first"]},
            ),
            ServerToolCall(
                name="web_search",
                id="server-1",
                arguments={"query": "x"},
                origin="response",
                result={"type": "web_search_tool_result", "content": ["second"]},
            ),
        ],
    )

    _, server_calls, _, _ = _merge_contributor_metadata([snapshot])

    assert [call.result for call in server_calls] == [
        {"type": "web_search_tool_result", "content": ["first"]},
        {"type": "web_search_tool_result", "content": ["second"]},
    ]


def test_conflicting_replayed_server_results_are_preserved_and_quarantined() -> None:
    base = Snapshot(
        source_path="first.json",
        source_sha256="a" * 64,
        session_id="session",
        thread_id="thread",
        provider="anthropic",
        operation="messages",
        outcome="success",
        server_tool_calls=[
            ServerToolCall(
                name="web_search",
                id="server-1",
                arguments={"query": "x"},
                origin="response",
                result={"type": "web_search_tool_result", "content": ["first"]},
            )
        ],
    )
    replay = base.model_copy(
        update={
            "source_path": "second.json",
            "server_tool_calls": [
                base.server_tool_calls[0].model_copy(
                    update={
                        "origin": "history",
                        "result": {
                            "type": "web_search_tool_result",
                            "content": ["second"],
                        },
                    }
                )
            ],
        }
    )

    _, server_calls, _, issues = _merge_contributor_metadata([base, replay])

    assert len(server_calls) == 2
    assert {issue.code for issue in issues} == {"duplicate_server_tool_result"}


def test_leaf_messages_stay_exact_while_contributor_evidence_is_unioned() -> None:
    question = Message(role="user", content="q")
    earlier_assistant = Message(
        role="assistant",
        content="",
        reasoning_content="summary",
        reasoning={
            "type": "reasoning",
            "id": "rs-1",
            "encrypted_content": "opaque",
            "metadata": {"response_only": True},
        },
    )
    replayed_assistant = earlier_assistant.model_copy(
        update={
            "reasoning_content": "",
            "reasoning": {
                "type": "reasoning",
                "id": "rs-1",
                "encrypted_content": "opaque",
            },
        }
    )
    contributor = Snapshot(
        source_path="old/short.json",
        source_sha256="a" * 64,
        session_id="session",
        thread_id="thread",
        provider="openai",
        operation="responses",
        outcome="success",
        model="old-model",
        harness="old-harness",
        instructions="old instructions",
        termination="old termination",
        history=[question],
        response=[earlier_assistant],
        tools=[ToolDefinition(name="old-tool")],
        agent_messages=[
            AgentMessageEvidence(
                origin="response",
                item_index=1,
                item={"type": "agent_message", "id": "old-message"},
            )
        ],
        issues=[
            AuditIssue(
                code="contributor_evidence",
                stage="test",
                severity=Severity.WARNING,
            )
        ],
    )
    leaf = contributor.model_copy(
        update={
            "source_path": "new/leaf.json",
            "source_sha256": "b" * 64,
            "model": "leaf-model",
            "harness": "leaf-harness",
            "instructions": "leaf instructions",
            "termination": "leaf termination",
            "history": [question, replayed_assistant],
            "response": [
                Message(role="assistant", content="done", reasoning_content="")
            ],
            "tools": [ToolDefinition(name="leaf-tool")],
            "agent_messages": [
                AgentMessageEvidence(
                    origin="history",
                    item_index=2,
                    item={"type": "agent_message", "id": "leaf-message"},
                )
            ],
            "issues": [],
        }
    )

    node = _trajectory_from_leaf(leaf, [contributor, leaf])

    assert node.messages == [*leaf.history, *leaf.response]
    assert node.messages[1].reasoning == replayed_assistant.reasoning
    assert {tool.name for tool in node.tools} == {"old-tool", "leaf-tool"}
    assert {record.item["id"] for record in node.agent_messages} == {
        "old-message",
        "leaf-message",
    }
    assert node.instructions == "leaf instructions"
    assert node.model == "leaf-model"
    assert node.harness == "leaf-harness"
    assert node.termination == "leaf termination"
    assert node.normalization_audit is not None
    assert {issue.code for issue in node.normalization_audit.issues} >= {
        "contributor_evidence",
        "contributor_instructions_changed",
        "contributor_model_changed",
        "contributor_harness_changed",
        "contributor_termination_changed",
    }
    assert node.normalization_audit.tag == AuditTag.PASS


def test_output_may_not_be_nested_under_input(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    with pytest.raises(ValueError):
        normalize(
            PipelineConfig(
                input_root=input_root,
                output_root=input_root / "normalized",
            )
        )


def test_failed_endpoint_never_persists_query_secrets(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write(
        input_root / "partition" / "session" / "bad.json",
        {
            "path": "/unsupported?proxy_token=TOP-SECRET",
            "captured_at": "2026-08-27T00:00:00Z",
            "request_headers": {"authorization": "TOP-SECRET"},
        },
    )

    normalize(PipelineConfig(input_root=input_root, output_root=output_root))

    record_path = _output_file(output_root, "/quarantine/records-00000.jsonl")
    serialized = record_path.read_bytes()
    assert b"TOP-SECRET" not in serialized
    assert b'"endpoint":"/unsupported"' in serialized
    assert validate_output(output_root).valid


def test_expressible_orphan_server_result_is_a_quarantined_trajectory(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    orphan = {
        "type": "shell_call_output",
        "call_id": "server-orphan",
        "output": {"stdout": "kept", "exit_code": 0},
    }
    _write(
        input_root / "partition" / "session" / "one.json",
        _capture(
            captured_at="2026-08-27T00:00:00Z",
            turn_id="turn-1",
            request_input=[_message("user", "run"), orphan],
            response_output=[_message("assistant", "done")],
            tools=[{"type": "shell", "environment": {"type": "container_auto"}}],
        ),
    )

    normalize(PipelineConfig(input_root=input_root, output_root=output_root))

    trajectory = _rows(
        _output_file(output_root, "/quarantine/trajectories-00000.jsonl")
    )[0]
    assert trajectory["server_tool_calls"][0]["result"] == orphan
    assert (
        "orphan_server_tool_result" in trajectory["normalization_audit"]["reason_codes"]
    )
    manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    assert manifest["counts"]["quarantined_trajectories"] == 1
    assert manifest["counts"]["quarantined_records"] == 0
    assert validate_output(output_root).valid


@pytest.mark.parametrize("field", ["id", "author", "recipient"])
def test_invalid_agent_message_routing_field_preserves_raw_trajectory(
    tmp_path: Path,
    field: str,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    raw_agent_message = {
        "type": "agent_message",
        "id": "amsg-1",
        "author": "/root/child",
        "recipient": "/root",
        "content": [{"type": "encrypted_content", "encrypted_content": "opaque"}],
        "unknown": {"preserve": True},
    }
    raw_agent_message[field] = None
    _write(
        input_root / "partition" / "session" / "one.json",
        _capture(
            captured_at="2026-08-27T00:00:00Z",
            turn_id="turn-1",
            request_input=[_message("user", "run"), raw_agent_message],
            response_output=[_message("assistant", "done")],
            tools=[],
        ),
    )

    normalize(PipelineConfig(input_root=input_root, output_root=output_root))

    trajectory = _rows(
        _output_file(output_root, "/quarantine/trajectories-00000.jsonl")
    )[0]
    assert trajectory["agent_messages"] == [
        {
            "origin": "history",
            "item_index": 1,
            "item": raw_agent_message,
        }
    ]
    assert (
        f"invalid_agent_message_{field}"
        in trajectory["normalization_audit"]["reason_codes"]
    )
    manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    assert manifest["counts"]["quarantined_trajectories"] == 1
    assert manifest["counts"]["quarantined_records"] == 0
    assert validate_output(output_root).valid


def test_anthropic_orphan_server_result_is_a_quarantined_trajectory(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    orphan = {
        "type": "web_search_tool_result",
        "tool_use_id": "server-orphan",
        "content": [{"type": "web_search_result", "url": "https://example.com"}],
    }
    _write(
        input_root / "partition" / "session" / "one.json",
        {
            "path": "/v1/messages",
            "session_id": "session",
            "status_code": 200,
            "request_body": {
                "model": "claude-test",
                "messages": [{"role": "user", "content": "search"}],
            },
            "response_body": {
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [orphan, {"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
            },
        },
    )

    normalize(PipelineConfig(input_root=input_root, output_root=output_root))

    trajectory = _rows(
        _output_file(output_root, "/quarantine/trajectories-00000.jsonl")
    )[0]
    assert trajectory["server_tool_calls"][0]["result"] == orphan
    assert (
        "orphan_server_tool_result" in trajectory["normalization_audit"]["reason_codes"]
    )
    assert validate_output(output_root).valid


def test_normalization_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    lock = tmp_path / "normalize.lock"
    with _exclusive_lock(lock), pytest.raises(RuntimeError), _exclusive_lock(lock):
        pass


def test_duplicate_child_leaf_is_merged_before_mounting(tmp_path: Path) -> None:
    spawn = ToolCall(
        id="spawn-1",
        function=FunctionCall(
            name="spawn_agent",
            arguments={"task_name": "child", "message": "do work"},
        ),
    )
    user = Message(role="user", content="start")
    spawn_message = Message(
        role="assistant",
        content="",
        reasoning_content="",
        tool_calls=[spawn],
    )
    spawn_result = Message(
        role="tool",
        content='{"task_name":"/root/child"}',
        tool_call_id="spawn-1",
        name="spawn_agent",
    )
    final = Message(role="assistant", content="done", reasoning_content="")
    spawn_definition = ToolDefinition(
        name="spawn_agent",
        parameters={
            "type": "object",
            "properties": {
                "task_name": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["task_name", "message"],
        },
    )

    def snapshot(
        name: str,
        *,
        thread_id: str,
        turn_id: str,
        history: list[Message],
        response: list[Message],
        captured_at: str,
        tools: list[ToolDefinition] | None = None,
        parent_thread_id: str = "",
        parent_turn_id: str = "",
        marker: str = "",
        agent_messages: list[AgentMessageEvidence] | None = None,
    ) -> Snapshot:
        return Snapshot(
            source_path=f"/input/{name}.json",
            source_sha256=name,
            session_id="session",
            thread_id=thread_id,
            turn_id=turn_id,
            parent_thread_id=parent_thread_id,
            parent_turn_id=parent_turn_id,
            forked_from_thread_id=parent_thread_id,
            subagent_marker=marker,
            provider="openai",
            operation="responses",
            outcome="success",
            captured_at=captured_at,
            request_id=name,
            model="gpt-test",
            harness="codex",
            history=history,
            response=response,
            tools=tools or [],
            agent_messages=agent_messages or [],
            termination="completed",
            wire_complete=True,
        )

    parent_event = snapshot(
        "parent-event",
        thread_id="main",
        turn_id="parent-turn",
        history=[user],
        response=[spawn_message],
        captured_at="2026-08-27T00:00:00Z",
        tools=[spawn_definition],
    )
    parent_leaf = snapshot(
        "parent-leaf",
        thread_id="main",
        turn_id="parent-final",
        history=[user, spawn_message, spawn_result],
        response=[final],
        captured_at="2026-08-27T00:01:00Z",
        agent_messages=[
            AgentMessageEvidence(
                origin="history",
                item_index=3,
                item={
                    "type": "agent_message",
                    "id": "relay-1",
                    "author": "/root/child",
                    "recipient": "/root",
                    "content": [
                        {
                            "type": "encrypted_content",
                            "encrypted_content": "opaque-relay",
                        }
                    ],
                    "unknown": "preserved",
                },
                preceding_completed_spawn_call_ids=["spawn-1"],
            )
        ],
    )
    marker = '{"spawn_call_id":"spawn-1","task_name":"child"}'
    duplicate_children = [
        snapshot(
            f"child-{suffix}",
            thread_id="child",
            turn_id=f"child-turn-{suffix}",
            history=[],
            response=[
                Message(role="assistant", content="result", reasoning_content="")
            ],
            captured_at=f"2026-08-27T00:02:0{index}Z",
            parent_thread_id="main",
            parent_turn_id="parent-turn",
            marker=marker,
            agent_messages=[
                AgentMessageEvidence(
                    origin="history",
                    item_index=0,
                    item={
                        "type": "agent_message",
                        "id": "child-input",
                        "author": "/root",
                        "recipient": "/root/child",
                        "content": [],
                    },
                    preceding_completed_spawn_call_ids=[],
                )
            ],
        )
        for index, suffix in enumerate(("a", "b"))
    ]

    with StateStore(tmp_path / "state.sqlite") as state:
        for item in (parent_event, parent_leaf, *duplicate_children):
            state.put_snapshot(item, endpoint="/v1/responses")
        stats = PipelineStats()
        _build_trajectories(state, stats)
        stored = list(state.iter_trajectories())

    assert stats.prefix_intermediates == 1
    assert stats.leaf_snapshots == 3
    assert stats.stored_trajectories == 1
    assert len(stored) == 1
    _, root, origins = stored[0]
    assert root.normalization_audit is not None
    assert root.normalization_audit.tag == AuditTag.PASS
    assert root.sub_agent_trajectory is not None
    assert set(root.sub_agent_trajectory) == {"spawn-1"}
    assert root.sub_agent_relay_mounts == {"spawn-1": "relay-1"}
    assert root.agent_messages[0].item["unknown"] == "preserved"
    assert {origin["source_ref"] for origin in origins} == {
        "/input/parent-event.json",
        "/input/parent-leaf.json",
        "/input/child-a.json",
        "/input/child-b.json",
    }

    child = duplicate_children[0]
    variant = child.model_copy(update={"termination": "length"})
    assert _flat_semantic_key(
        child, _trajectory_from_leaf(child, [child])
    ) != _flat_semantic_key(variant, _trajectory_from_leaf(variant, [variant]))
