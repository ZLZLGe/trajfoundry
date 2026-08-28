from __future__ import annotations

import pytest

from trajfoundry.models import (
    FunctionCall,
    Message,
    Metadata,
    Snapshot,
    ToolCall,
    ToolDefinition,
    TrajectoryNode,
)
from trajfoundry.output_contract import project_trajectory
from trajfoundry.pipeline import _spawn_only_messages as pipeline_spawn_only_messages
from trajfoundry.quality import enrich_trajectory
from trajfoundry.streaming import _spawn_only_messages as streaming_spawn_only_messages
from trajfoundry.subagents import plan_subagent_mounts
from trajfoundry.tool_names import is_spawn_tool_name, qualify_tool_name


@pytest.mark.parametrize(
    ("namespace", "name", "expected"),
    [
        (None, "spawn_agent", "spawn_agent"),
        ("", "spawn_agent", "spawn_agent"),
        ("collaboration", "spawn_agent", "collaboration.spawn_agent"),
        (
            "collaboration",
            "collaboration.spawn_agent",
            "collaboration.spawn_agent",
        ),
    ],
)
def test_qualify_tool_name_preserves_full_names(
    namespace: str | None, name: str, expected: str
) -> None:
    assert qualify_tool_name(namespace, name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Agent", True),
        ("spawn_agent", True),
        ("collaboration.spawn_agent", True),
        ("multi_agent_v1.spawn_agent", True),
        ("collaboration.Agent", False),
        ("not_spawn_agent", False),
        (".spawn_agent", False),
    ],
)
def test_is_spawn_tool_name_uses_the_full_name(name: str, expected: bool) -> None:
    assert is_spawn_tool_name(name) is expected


def _spawn_call(
    call_id: str = "spawn-1", name: str = "collaboration.spawn_agent"
) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=FunctionCall(name=name, arguments={"task_name": "child"}),
    )


@pytest.mark.parametrize(
    "filter_messages",
    [pipeline_spawn_only_messages, streaming_spawn_only_messages],
)
def test_spawn_evidence_requires_exact_namespaced_result_name(filter_messages) -> None:
    call = _spawn_call()
    messages = [
        Message(role="assistant", content="", reasoning_content="", tool_calls=[call]),
        Message(
            role="tool",
            content='{"task_name":"wrong"}',
            tool_call_id=call.id,
            name="multi_agent_v1.spawn_agent",
        ),
        Message(
            role="tool",
            content='{"task_name":"right"}',
            tool_call_id=call.id,
            name=call.function.name,
        ),
    ]

    filtered = filter_messages(messages)

    assert [message.name for message in filtered if message.role == "tool"] == [
        "collaboration.spawn_agent"
    ]
    assert filtered[0].tool_calls
    assert filtered[0].tool_calls[0].function.name == "collaboration.spawn_agent"


def _snapshot(
    source: str,
    *,
    thread: str,
    turn: str,
    history: list[Message] | None = None,
    response: list[Message] | None = None,
    parent_thread: str = "",
    parent_turn: str = "",
    marker: str = "",
) -> Snapshot:
    return Snapshot(
        source_path=f"/{source}.json",
        source_sha256=source,
        session_id="session",
        thread_id=thread,
        turn_id=turn,
        parent_thread_id=parent_thread,
        parent_turn_id=parent_turn,
        subagent_marker=marker,
        provider="openai",
        operation="responses",
        outcome="success",
        request_id=source,
        history=history or [],
        response=response or [],
        wire_complete=True,
    )


def test_namespaced_spawn_result_is_used_only_by_its_full_call_name() -> None:
    call = _spawn_call()
    event = _snapshot(
        "event",
        thread="parent",
        turn="spawn-turn",
        response=[
            Message(
                role="assistant", content="", reasoning_content="", tool_calls=[call]
            )
        ],
    )
    mismatched_parent = _snapshot(
        "parent-wrong",
        thread="parent",
        turn="final-turn",
        history=[
            Message(
                role="assistant", content="", reasoning_content="", tool_calls=[call]
            ),
            Message(
                role="tool",
                content='{"task_name":"/root/wrong"}',
                tool_call_id=call.id,
                name="multi_agent_v1.spawn_agent",
            ),
        ],
        response=[Message(role="assistant", content="done", reasoning_content="")],
    )
    child = _snapshot(
        "child",
        thread="child",
        turn="child-turn",
        parent_thread="parent",
        parent_turn="spawn-turn",
        marker=call.id,
        response=[Message(role="assistant", content="done", reasoning_content="")],
    )

    mismatched = plan_subagent_mounts(
        [mismatched_parent, child],
        all_snapshots=[event, mismatched_parent, child],
    )
    assert len(mismatched.edges) == 1
    assert mismatched.edges[0].agent_name == ""

    matching_parent = mismatched_parent.model_copy(
        update={
            "source_path": "/parent-right.json",
            "source_sha256": "parent-right",
            "history": [
                mismatched_parent.history[0],
                mismatched_parent.history[1].model_copy(
                    update={
                        "content": '{"task_name":"/root/right"}',
                        "name": call.function.name,
                    }
                ),
            ],
        }
    )
    matching = plan_subagent_mounts(
        [matching_parent, child], all_snapshots=[event, matching_parent, child]
    )
    assert len(matching.edges) == 1
    assert matching.edges[0].agent_name == "/root/right"


def test_namespaced_spawn_counts_as_a_mounted_spawn_in_output_contract() -> None:
    call = _spawn_call()
    child = TrajectoryNode(
        messages=[Message(role="assistant", content="done", reasoning_content="")],
        tools=[],
        source="child.json",
        metadata=Metadata(source_file="child.json"),
    )
    parent = TrajectoryNode(
        messages=[
            Message(
                role="assistant", content="", reasoning_content="", tool_calls=[call]
            ),
            Message(
                role="tool",
                content="spawned",
                tool_call_id=call.id,
                name=call.function.name,
            ),
            Message(role="assistant", content="done", reasoning_content=""),
        ],
        tools=[ToolDefinition(name=call.function.name)],
        source="parent.json",
        metadata=Metadata(source_file="parent.json"),
        sub_agent_trajectory={call.id: child},
    )

    enriched = enrich_trajectory(parent)
    projected = project_trajectory(enriched)

    assert enriched.completeness is not None
    assert enriched.completeness.spawn_calls == 1
    assert enriched.completeness.subtree_complete
    assert projected["completeness"]["spawn_calls"] == 1
