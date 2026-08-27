from __future__ import annotations

import json

from trajfoundry.models import (
    AgentMessageEvidence,
    FunctionCall,
    Message,
    Metadata,
    Snapshot,
    ToolCall,
    TrajectoryNode,
)
from trajfoundry.output_contract import project_trajectory
from trajfoundry.subagents import (
    SnapshotTrajectory,
    mount_subagents,
    plan_subagent_mounts,
)


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str = "", *, calls: list[ToolCall] | None = None) -> Message:
    return Message(
        role="assistant",
        content=content,
        reasoning_content="",
        tool_calls=calls,
    )


def spawn(call_id: str, task_name: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=FunctionCall(
            name="spawn_agent",
            arguments={"task_name": task_name, "message": f"do {task_name}"},
        ),
    )


def anthropic_spawn(call_id: str, task_name: str) -> ToolCall:
    call = spawn(call_id, task_name)
    return call.model_copy(
        update={"function": call.function.model_copy(update={"name": "Agent"})}
    )


def snapshot(
    name: str,
    *,
    thread: str,
    turn: str,
    history: list[Message] | None = None,
    response: list[Message] | None = None,
    parent_thread: str = "",
    parent_turn: str = "",
    forked_from: str = "",
    marker: str = "",
    agent_messages: list[AgentMessageEvidence] | None = None,
) -> Snapshot:
    return Snapshot(
        source_path=f"/{name}.json",
        source_sha256=name,
        source_partition="p",
        session_id="s",
        thread_id=thread,
        turn_id=turn,
        parent_thread_id=parent_thread,
        parent_turn_id=parent_turn,
        forked_from_thread_id=forked_from,
        subagent_marker=marker,
        provider="openai",
        operation="responses",
        outcome="success",
        request_id=name,
        model="gpt",
        harness="codex",
        history=history or [],
        response=response or [],
        agent_messages=agent_messages or [],
        wire_complete=True,
    )


def agent_message(
    message_id: str,
    *,
    author: str,
    recipient: str,
    preceding: list[str],
    item_index: int = 0,
) -> AgentMessageEvidence:
    return AgentMessageEvidence(
        origin="history",
        item_index=item_index,
        item={
            "type": "agent_message",
            "id": message_id,
            "author": author,
            "recipient": recipient,
            "content": [{"type": "encrypted_content", "encrypted_content": "x"}],
        },
        preceding_completed_spawn_call_ids=preceding,
    )


def node(item: Snapshot) -> TrajectoryNode:
    return TrajectoryNode(
        messages=[*item.history, *item.response],
        tools=[],
        source=item.source_path.rsplit("/", 1)[-1],
        model=item.model,
        harness=item.harness,
        instructions=item.instructions,
        metadata=Metadata(source_file=item.source_path.rsplit("/", 1)[-1]),
    )


def codes(plan: object) -> set[str]:
    return {diagnostic.code for diagnostic in plan.diagnostics}  # type: ignore[attr-defined]


def test_mounts_by_parent_turn_call_id_and_structured_task_routing() -> None:
    spawn_a, spawn_b = spawn("call-a", "research"), spawn("call-b", "tests")
    event_snapshot = snapshot(
        "parent-event",
        thread="main",
        turn="turn-1",
        history=[user("go")],
        response=[assistant(calls=[spawn_a, spawn_b])],
    )
    # The maximal parent leaf repeats the spawn turn in history.  It no longer
    # carries turn-1 as its own turn id, so all_snapshots is required to retain
    # the exact turn binding.
    parent_leaf = snapshot(
        "parent-leaf",
        thread="main",
        turn="turn-2",
        history=[user("go"), assistant(calls=[spawn_a, spawn_b])],
        response=[assistant("done")],
    )
    child_a = snapshot(
        "child-a",
        thread="child-a",
        turn="child-turn-a",
        response=[assistant("research result")],
        parent_thread="main",
        parent_turn="turn-1",
        forked_from="main",
        marker=json.dumps({"spawn_call_id": "call-a", "task_name": "research"}),
    )
    child_b = snapshot(
        "child-b",
        thread="child-b",
        turn="child-turn-b",
        response=[assistant("test result")],
        parent_thread="main",
        parent_turn="turn-1",
        forked_from="main",
        marker=json.dumps({"spawn_call_id": "call-b", "task_name": "tests"}),
    )

    plan = plan_subagent_mounts(
        [child_b, parent_leaf, child_a],
        all_snapshots=[parent_leaf, event_snapshot, child_a, child_b],
    )

    assert codes(plan) == set()
    paths = [leaf.source_path for leaf in plan.leaves]
    assert {
        (paths[edge.parent_index], paths[edge.child_index], edge.spawn_call_id)
        for edge in plan.edges
    } == {
        ("/parent-leaf.json", "/child-a.json", "call-a"),
        ("/parent-leaf.json", "/child-b.json", "call-b"),
    }
    assert tuple(paths[index] for index in plan.main_root_indices) == (
        "/parent-leaf.json",
    )
    assert plan.orphan_indices == ()


def _responses_relay_snapshots(
    *,
    parent_agent_messages: list[AgentMessageEvidence],
    child_agent_messages: list[AgentMessageEvidence] | None = None,
) -> tuple[Snapshot, Snapshot, Snapshot]:
    call = spawn("spawn-1", "child")
    event = snapshot(
        "parent-event",
        thread="main-thread",
        turn="parent-turn",
        response=[assistant(calls=[call])],
    )
    parent = snapshot(
        "parent-leaf",
        thread="main-thread",
        turn="parent-final",
        history=[
            assistant(calls=[call]),
            Message(
                role="tool",
                content='{"task_name":"/root/child"}',
                tool_call_id="spawn-1",
                name="spawn_agent",
            ),
        ],
        response=[assistant("done")],
        agent_messages=parent_agent_messages,
    )
    child = snapshot(
        "child-leaf",
        thread="child-thread",
        turn="child-turn",
        response=[assistant("child done")],
        parent_thread="main-thread",
        parent_turn="parent-turn",
        forked_from="main-thread",
        marker="collab_spawn",
        agent_messages=(
            child_agent_messages
            if child_agent_messages is not None
            else [
                agent_message(
                    "child-input",
                    author="/root",
                    recipient="/root/child",
                    preceding=[],
                )
            ]
        ),
    )
    return event, parent, child


def test_responses_mount_uses_canonical_agent_name_and_unique_ordered_relay() -> None:
    event, parent, child = _responses_relay_snapshots(
        parent_agent_messages=[
            agent_message(
                "relay-1",
                author="/root/child",
                recipient="/root",
                preceding=["spawn-1"],
            )
        ]
    )

    plan = plan_subagent_mounts([parent, child], all_snapshots=[event, parent, child])

    assert codes(plan) == set()
    assert len(plan.edges) == 1
    assert plan.edges[0].spawn_call_id == "spawn-1"
    assert plan.edges[0].agent_name == "/root/child"
    assert plan.edges[0].relay_id == "relay-1"

    result = mount_subagents(
        [
            SnapshotTrajectory(parent, node(parent)),
            SnapshotTrajectory(child, node(child)),
        ],
        all_snapshots=[event, parent, child],
    )
    assert result.roots[0].sub_agent_relay_mounts == {"spawn-1": "relay-1"}
    assert result.roots[0].agent_messages[0].item == parent.agent_messages[0].item


def test_multiple_ordered_agent_messages_do_not_guess_a_relay() -> None:
    event, parent, child = _responses_relay_snapshots(
        parent_agent_messages=[
            agent_message(
                f"relay-{index}",
                author="/root/child",
                recipient="/root",
                preceding=["spawn-1"],
                item_index=index,
            )
            for index in (1, 2)
        ]
    )

    plan = plan_subagent_mounts([parent, child], all_snapshots=[event, parent, child])

    assert len(plan.edges) == 1
    assert plan.edges[0].relay_id == ""
    assert "ambiguous_agent_relay" in codes(plan)

    result = mount_subagents(
        [
            SnapshotTrajectory(parent, node(parent)),
            SnapshotTrajectory(child, node(child)),
        ],
        all_snapshots=[event, parent, child],
    )
    root = result.roots[0]
    assert root.sub_agent_trajectory is not None
    assert set(root.sub_agent_trajectory) == {"spawn-1"}
    assert root.sub_agent_relay_mounts is None
    assert root.completeness is not None
    assert root.completeness.subtree_complete
    assert root.normalization_audit is not None
    assert "ambiguous_agent_relay" in root.normalization_audit.reason_codes
    assert "incomplete_subagent_mount" not in root.normalization_audit.reason_codes
    project_trajectory(root)


def test_missing_spawn_agent_name_does_not_block_primary_mount() -> None:
    call = spawn("spawn-1", "child")
    event = snapshot(
        "parent-event",
        thread="main-thread",
        turn="parent-turn",
        response=[assistant(calls=[call])],
    )
    parent = snapshot(
        "parent-leaf",
        thread="main-thread",
        turn="parent-final",
        history=[
            assistant(calls=[call]),
            Message(
                role="tool",
                content="{}",
                tool_call_id="spawn-1",
                name="spawn_agent",
            ),
        ],
        response=[assistant("done")],
        agent_messages=[
            agent_message(
                "relay-1",
                author="/root/child",
                recipient="/root",
                preceding=["spawn-1"],
            )
        ],
    )
    child = snapshot(
        "child-leaf",
        thread="child-thread",
        turn="child-turn",
        response=[assistant("child done")],
        parent_thread="main-thread",
        parent_turn="parent-turn",
        forked_from="main-thread",
        marker="spawn-1",
        agent_messages=[
            agent_message(
                "child-input",
                author="/root",
                recipient="/root/child",
                preceding=[],
            )
        ],
    )

    result = mount_subagents(
        [
            SnapshotTrajectory(parent, node(parent)),
            SnapshotTrajectory(child, node(child)),
        ],
        all_snapshots=[event, parent, child],
    )

    assert len(result.plan.edges) == 1
    assert result.plan.edges[0].agent_name == ""
    assert result.plan.edges[0].relay_id == ""
    assert result.plan.orphan_indices == ()
    assert "unmounted_spawn_call" not in codes(result.plan)
    root = result.roots[0]
    assert root.sub_agent_trajectory is not None
    assert set(root.sub_agent_trajectory) == {"spawn-1"}
    assert root.completeness is not None
    assert root.completeness.subtree_complete
    assert root.normalization_audit is not None
    assert "incomplete_subagent_mount" not in root.normalization_audit.reason_codes
    project_trajectory(root)


def test_absent_parent_relay_candidate_requires_no_child_identity() -> None:
    event, parent, child = _responses_relay_snapshots(
        parent_agent_messages=[],
        child_agent_messages=[],
    )

    plan = plan_subagent_mounts([parent, child], all_snapshots=[event, parent, child])

    assert len(plan.edges) == 1
    assert plan.edges[0].agent_name == "/root/child"
    assert plan.edges[0].relay_id == ""
    assert plan.orphan_indices == ()
    assert codes(plan) == set()


def test_conflicting_spawn_agent_names_do_not_remove_primary_spawn_event() -> None:
    event, parent, child = _responses_relay_snapshots(parent_agent_messages=[])
    conflicting_replay = parent.model_copy(
        update={
            "source_path": "/parent-conflicting-name.json",
            "source_sha256": "parent-conflicting-name",
            "request_id": "parent-conflicting-name",
            "history": [
                parent.history[0],
                parent.history[1].model_copy(
                    update={"content": '{"task_name":"/root/other"}'}
                ),
            ],
        },
        deep=True,
    )

    result = mount_subagents(
        [
            SnapshotTrajectory(parent, node(parent)),
            SnapshotTrajectory(child, node(child)),
        ],
        all_snapshots=[event, conflicting_replay, parent, child],
    )

    assert len(result.plan.edges) == 1
    assert result.plan.edges[0].agent_name == ""
    assert result.plan.orphan_indices == ()
    assert "conflicting_spawn_agent_name" in codes(result.plan)
    assert "conflicting_spawn_evidence" not in codes(result.plan)
    root = result.roots[0]
    assert root.completeness is not None
    assert root.completeness.subtree_complete
    assert root.normalization_audit is not None
    assert "conflicting_spawn_agent_name" in root.normalization_audit.reason_codes
    assert "incomplete_subagent_mount" not in root.normalization_audit.reason_codes
    project_trajectory(root)


def test_conflicting_child_agent_names_do_not_orphan_primary_mount() -> None:
    event, parent, child = _responses_relay_snapshots(
        parent_agent_messages=[
            agent_message(
                "relay-1",
                author="/root/child",
                recipient="/root",
                preceding=["spawn-1"],
            )
        ],
        child_agent_messages=[
            agent_message(
                "child-input-1",
                author="/root",
                recipient="/root/child",
                preceding=[],
            ),
            agent_message(
                "child-input-2",
                author="/root",
                recipient="/root/other",
                preceding=[],
                item_index=1,
            ),
        ],
    )

    plan = plan_subagent_mounts([parent, child], all_snapshots=[event, parent, child])

    assert len(plan.edges) == 1
    assert plan.edges[0].relay_id == ""
    assert plan.orphan_indices == ()
    assert "conflicting_child_agent_name" in codes(plan)
    assert "unmounted_spawn_call" not in codes(plan)


def test_agent_message_before_spawn_completion_is_not_used_as_relay() -> None:
    event, parent, child = _responses_relay_snapshots(
        parent_agent_messages=[
            agent_message(
                "relay-early",
                author="/root/child",
                recipient="/root",
                preceding=[],
            )
        ]
    )

    plan = plan_subagent_mounts([parent, child], all_snapshots=[event, parent, child])

    assert plan.edges[0].relay_id == ""
    assert "agent_relay_before_spawn" in codes(plan)


def test_bogus_and_duplicate_agent_messages_are_diagnosed_not_guessed() -> None:
    duplicate = agent_message(
        "relay-duplicate",
        author="/root/child",
        recipient="/root",
        preceding=["spawn-1"],
    )
    event, parent, child = _responses_relay_snapshots(
        parent_agent_messages=[
            duplicate,
            duplicate.model_copy(deep=True),
            agent_message(
                "relay-bogus",
                author="/root/not-the-child",
                recipient="/root",
                preceding=["spawn-1"],
                item_index=2,
            ),
        ]
    )

    plan = plan_subagent_mounts([parent, child], all_snapshots=[event, parent, child])

    assert plan.edges[0].relay_id == ""
    assert {"duplicate_agent_message_id", "unmatched_agent_message"} <= codes(plan)


def test_conflicting_agent_message_id_is_not_used_as_relay() -> None:
    first = agent_message(
        "relay-conflict",
        author="/root/child",
        recipient="/root",
        preceding=["spawn-1"],
    )
    second = agent_message(
        "relay-conflict",
        author="/root/other",
        recipient="/root",
        preceding=["spawn-1"],
        item_index=1,
    )
    event, parent, child = _responses_relay_snapshots(
        parent_agent_messages=[first, second]
    )

    plan = plan_subagent_mounts([parent, child], all_snapshots=[event, parent, child])

    assert plan.edges[0].relay_id == ""
    assert "conflicting_agent_message_evidence" in codes(plan)


def test_replayed_relay_body_difference_does_not_affect_routing() -> None:
    relay = agent_message(
        "relay-1",
        author="/root/child",
        recipient="/root",
        preceding=["spawn-1"],
    )
    replay = relay.model_copy(deep=True)
    replay.item["content"] = [
        {"type": "encrypted_content", "encrypted_content": "different-opaque"}
    ]
    event, parent, child = _responses_relay_snapshots(parent_agent_messages=[relay])
    earlier_parent = parent.model_copy(
        update={
            "source_path": "/parent-replay.json",
            "source_sha256": "parent-replay",
            "request_id": "parent-replay",
            "agent_messages": [replay],
        },
        deep=True,
    )

    plan = plan_subagent_mounts(
        [parent, child], all_snapshots=[event, earlier_parent, parent, child]
    )

    assert plan.edges[0].relay_id == "relay-1"
    assert "conflicting_agent_message_evidence" not in codes(plan)


def test_agent_alias_is_valid_spawn_evidence() -> None:
    call = anthropic_spawn("call-a", "research")
    parent = snapshot(
        "parent",
        thread="main",
        turn="turn-1",
        response=[assistant(calls=[call])],
    )
    child = snapshot(
        "child",
        thread="child",
        turn="child-turn",
        response=[assistant("done")],
        parent_thread="main",
        parent_turn="turn-1",
        forked_from="main",
        marker="call-a",
    )
    plan = plan_subagent_mounts([parent, child])
    assert len(plan.edges) == 1
    assert plan.orphan_indices == ()


def test_opaque_marker_does_not_guess_between_multiple_spawn_calls() -> None:
    calls = [spawn("call-a", "a"), spawn("call-b", "b")]
    parent = snapshot(
        "parent",
        thread="main",
        turn="turn-1",
        response=[assistant(calls=calls)],
    )
    child = snapshot(
        "child",
        thread="child",
        turn="child-turn",
        response=[assistant("result")],
        parent_thread="main",
        parent_turn="turn-1",
        forked_from="main",
        marker="subagent",
    )

    plan = plan_subagent_mounts([parent, child])

    assert plan.edges == ()
    assert "ambiguous_spawn_call" in codes(plan)
    assert "unmounted_spawn_call" in codes(plan)
    assert tuple(plan.leaves[index].source_path for index in plan.orphan_indices) == (
        "/child.json",
    )


def test_explicit_routing_mismatch_never_falls_back_to_only_spawn() -> None:
    parent = snapshot(
        "parent",
        thread="main",
        turn="turn-1",
        response=[assistant(calls=[spawn("actual", "task")])],
    )
    child = snapshot(
        "child",
        thread="child",
        turn="child-turn",
        response=[assistant("result")],
        parent_thread="main",
        parent_turn="turn-1",
        forked_from="main",
        marker=json.dumps({"spawn_call_id": "wrong"}),
    )

    plan = plan_subagent_mounts([parent, child])

    assert plan.edges == ()
    assert "spawn_routing_mismatch" in codes(plan)


def test_conflicting_structured_child_metadata_is_diagnosed_as_orphan() -> None:
    parent = snapshot(
        "parent",
        thread="main",
        turn="turn-1",
        response=[assistant(calls=[spawn("call", "task")])],
    )
    child = snapshot(
        "child",
        thread="child",
        turn="child-turn",
        response=[assistant("result")],
        parent_thread="main",
        parent_turn="turn-1",
        forked_from="fork-source",
        marker=json.dumps(
            {"spawn_call_id": "call", "parent_thread_id": "different-parent"}
        ),
    )

    plan = plan_subagent_mounts([parent, child])

    assert plan.edges == ()
    assert "conflicting_marker_metadata" in codes(plan)
    assert len(plan.orphan_indices) == 1


def test_fork_source_may_differ_from_explicit_spawn_parent() -> None:
    parent = snapshot(
        "parent",
        thread="main",
        turn="turn-1",
        response=[assistant(calls=[spawn("call", "task")])],
    )
    child = snapshot(
        "child",
        thread="child",
        turn="child-turn",
        response=[assistant("result")],
        parent_thread="main",
        parent_turn="turn-1",
        # Real nested/guardian metadata can record a fork source that differs
        # from the thread owning the spawn turn.
        forked_from="earlier-subagent-thread",
        marker="call",
    )

    plan = plan_subagent_mounts([parent, child])

    assert len(plan.edges) == 1
    assert plan.orphan_indices == ()


def test_linkage_must_be_consistent_across_preaggregation_snapshots() -> None:
    call = spawn("call", "task")
    parent = snapshot(
        "parent",
        thread="main",
        turn="turn-1",
        response=[assistant(calls=[call])],
    )
    child_leaf = snapshot(
        "child-leaf",
        thread="child",
        turn="child-turn-2",
        history=[user("work")],
        response=[assistant("result")],
        parent_thread="main",
        parent_turn="turn-1",
        marker="call",
    )
    conflicting_intermediate = snapshot(
        "child-intermediate",
        thread="child",
        turn="child-turn-1",
        response=[assistant("started")],
        parent_thread="other-main",
        parent_turn="turn-1",
        marker="call",
    )

    plan = plan_subagent_mounts(
        [parent, child_leaf],
        all_snapshots=[parent, child_leaf, conflicting_intermediate],
    )

    assert plan.edges == ()
    assert "conflicting_parent_thread" in codes(plan)
    assert len(plan.orphan_indices) == 1


def test_multiple_children_claiming_one_call_are_all_orphaned() -> None:
    parent = snapshot(
        "parent",
        thread="main",
        turn="turn-1",
        response=[assistant(calls=[spawn("call", "task")])],
    )
    children = [
        snapshot(
            f"child-{suffix}",
            thread=f"child-{suffix}",
            turn=f"child-turn-{suffix}",
            response=[assistant("result")],
            parent_thread="main",
            parent_turn="turn-1",
            forked_from="main",
            marker="call",
        )
        for suffix in ("a", "b")
    ]

    plan = plan_subagent_mounts([parent, *children])

    assert plan.edges == ()
    assert "multiple_children_for_spawn" in codes(plan)
    assert {plan.leaves[index].source_path for index in plan.orphan_indices} == {
        "/child-a.json",
        "/child-b.json",
    }


def test_cycle_is_reported_and_never_materialized() -> None:
    leaf_a = snapshot(
        "a",
        thread="a",
        turn="turn-a",
        response=[assistant(calls=[spawn("spawn-b", "b")])],
        parent_thread="b",
        parent_turn="turn-b",
        forked_from="b",
        marker="spawn-a",
    )
    leaf_b = snapshot(
        "b",
        thread="b",
        turn="turn-b",
        response=[assistant(calls=[spawn("spawn-a", "a")])],
        parent_thread="a",
        parent_turn="turn-a",
        forked_from="a",
        marker="spawn-b",
    )

    plan = plan_subagent_mounts([leaf_b, leaf_a])

    assert plan.edges == ()
    assert "mount_cycle" in codes(plan)
    assert len(plan.orphan_indices) == 2


def test_parent_branch_ambiguity_is_explicit() -> None:
    call = spawn("call", "task")
    event = snapshot(
        "event",
        thread="main",
        turn="turn-1",
        response=[assistant(calls=[call])],
    )
    branches = [
        snapshot(
            f"branch-{suffix}",
            thread="main",
            turn=f"turn-{suffix}",
            history=[assistant(calls=[call])],
            response=[assistant(suffix)],
        )
        for suffix in ("a", "b")
    ]
    child = snapshot(
        "child",
        thread="child",
        turn="child-turn",
        response=[assistant("result")],
        parent_thread="main",
        parent_turn="turn-1",
        forked_from="main",
        marker="call",
    )

    plan = plan_subagent_mounts(
        [*branches, child], all_snapshots=[event, *branches, child]
    )

    assert plan.edges == ()
    assert "ambiguous_parent_leaf" in codes(plan)
    assert len(plan.orphan_indices) == 1


def test_materialization_uses_call_id_keys_and_marks_orphans() -> None:
    call = spawn("call", "task")
    parent = snapshot(
        "parent",
        thread="main",
        turn="turn-1",
        response=[assistant(calls=[call])],
    )
    child = snapshot(
        "child",
        thread="child",
        turn="child-turn",
        response=[assistant("result")],
        parent_thread="main",
        parent_turn="turn-1",
        forked_from="main",
        marker=json.dumps({"spawn_call_id": "call", "task_name": "task"}),
    )
    orphan = snapshot(
        "orphan",
        thread="orphan",
        turn="orphan-turn",
        response=[assistant("lost")],
        parent_thread="missing",
        parent_turn="missing-turn",
        forked_from="missing",
        marker="subagent",
    )

    result = mount_subagents(
        [
            SnapshotTrajectory(orphan, node(orphan)),
            SnapshotTrajectory(child, node(child)),
            SnapshotTrajectory(parent, node(parent)),
        ]
    )

    assert len(result.roots) == 1
    root = result.roots[0]
    assert root.sub_agent_trajectory is not None
    assert set(root.sub_agent_trajectory) == {"call"}
    assert root.completeness_tag == "incomplete_main_with_complete_sub"
    assert root.completeness is not None
    assert root.completeness.spawn_calls == 1
    assert root.completeness.mounted_subs == 1
    assert root.completeness.subtree_complete
    assert root.completeness.trailing_unanswered_call
    assert root.completeness.no_final_assistant_turn
    assert len(result.orphans) == 1
    assert result.orphans[0].completeness_tag == "orphan_sub"
    assert result.orphans[0].completeness is not None
    assert not result.orphans[0].completeness.subtree_complete


def test_materialization_preserves_explicit_mount_planning_failure() -> None:
    call = spawn("call", "task")
    parent = snapshot(
        "parent",
        thread="main",
        turn="turn-1",
        response=[assistant(calls=[call])],
    )
    child = snapshot(
        "child",
        thread="child",
        turn="child-turn",
        response=[assistant("result")],
        parent_thread="main",
        parent_turn="turn-1",
        forked_from="main",
        marker="call",
    )
    parent_node = node(parent)
    parent_node.sub_agent_trajectory = {"call": node(child)}

    result = mount_subagents(
        [
            SnapshotTrajectory(parent, parent_node),
            SnapshotTrajectory(child, node(child)),
        ]
    )

    assert "preexisting_mount_conflict" in codes(result.plan)
    root = result.roots[0]
    assert root.completeness is not None
    assert not root.completeness.subtree_complete
    assert root.normalization_audit is not None
    assert "preexisting_mount_conflict" in root.normalization_audit.reason_codes
    assert "incomplete_subagent_mount" in root.normalization_audit.reason_codes
