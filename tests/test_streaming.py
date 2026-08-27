from trajfoundry.models import AgentMessageEvidence, Message, Snapshot
from trajfoundry.streaming import streaming_prefix_leaves


def snap(
    path: str,
    history: list[Message],
    response: list[Message],
    *,
    instructions: str = "",
) -> Snapshot:
    return Snapshot(
        source_path=path,
        source_sha256=path,
        source_partition="partition",
        session_id="session",
        thread_id="thread",
        provider="openai",
        operation="responses",
        outcome="success",
        instructions=instructions,
        history=history,
        response=response,
    )


def test_streaming_keeps_all_divergent_leaves() -> None:
    q1 = Message(role="user", content="q1")
    a1 = Message(role="assistant", content="a1", reasoning_content="")
    q2 = Message(role="user", content="q2")
    a2 = Message(role="assistant", content="a2", reasoning_content="")
    alternate = Message(role="assistant", content="a2-alt", reasoning_content="")
    result = streaming_prefix_leaves(
        [
            snap("a", [q1], [a1]),
            snap("b", [q1, a1, q2], [a2]),
            snap("c", [q1, a1, q2], [alternate]),
        ]
    )
    assert [leaf.source_path for leaf in result.leaves] == ["b", "c"]
    assert result.contributor_paths["b"] == ("a", "b")
    assert result.contributor_paths["c"] == ("a", "c")


def test_instructions_split_prefix_chains() -> None:
    q = Message(role="user", content="q")
    a = Message(role="assistant", content="a", reasoning_content="")
    q2 = Message(role="user", content="q2")
    b = Message(role="assistant", content="b", reasoning_content="")
    result = streaming_prefix_leaves(
        [
            snap("a", [q], [a], instructions="one"),
            snap("b", [q, a, q2], [b], instructions="two"),
        ]
    )
    assert {leaf.source_path for leaf in result.leaves} == {"a", "b"}


def test_late_shorter_snapshot_is_still_intermediate() -> None:
    q = Message(role="user", content="q")
    a = Message(role="assistant", content="a", reasoning_content="")
    q2 = Message(role="user", content="q2")
    b = Message(role="assistant", content="b", reasoning_content="")
    result = streaming_prefix_leaves(
        [snap("long", [q, a, q2], [b]), snap("short", [q], [a])]
    )
    assert [leaf.source_path for leaf in result.leaves] == ["long"]
    assert result.contributor_paths["long"] == ("long", "short")


def test_intermediate_subagent_linkage_is_retained_as_lightweight_evidence() -> None:
    first = snap(
        "child-first",
        [Message(role="user", content="q")],
        [Message(role="assistant", content="a", reasoning_content="")],
    ).model_copy(
        update={
            "parent_thread_id": "parent",
            "parent_turn_id": "turn-1",
            "subagent_marker": "call-1",
        }
    )
    leaf = snap(
        "child-leaf",
        [
            Message(role="user", content="q"),
            Message(role="assistant", content="a", reasoning_content=""),
            Message(role="user", content="next"),
        ],
        [Message(role="assistant", content="done", reasoning_content="")],
    ).model_copy(
        update={
            "parent_thread_id": "parent",
            "parent_turn_id": "turn-1",
            "subagent_marker": "call-1",
        }
    )

    result = streaming_prefix_leaves([first, leaf])

    assert [item.source_path for item in result.spawn_evidence] == [
        "child-first",
        "child-leaf",
    ]
    assert all(not item.history and not item.tools for item in result.spawn_evidence)


def test_agent_message_routing_evidence_drops_opaque_content_only_in_copy() -> None:
    record = AgentMessageEvidence(
        origin="history",
        item_index=7,
        item={
            "type": "agent_message",
            "id": "amsg-1",
            "author": "/root/child",
            "recipient": "/root",
            "content": [
                {"type": "encrypted_content", "encrypted_content": "very-large"}
            ],
            "future": "kept-in-source",
        },
        preceding_completed_spawn_call_ids=["spawn-1"],
    )
    source = snap(
        "with-agent-message",
        [Message(role="user", content="q")],
        [Message(role="assistant", content="done", reasoning_content="")],
    ).model_copy(update={"agent_messages": [record]})

    result = streaming_prefix_leaves([source])

    assert result.leaves[0].agent_messages[0].item == record.item
    compact = result.spawn_evidence[0].agent_messages[0]
    assert compact.item == {
        "type": "agent_message",
        "id": "amsg-1",
        "author": "/root/child",
        "recipient": "/root",
    }
    assert compact.preceding_completed_spawn_call_ids == ["spawn-1"]


def test_equal_terminal_transcripts_remain_separate_leaves() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")

    result = streaming_prefix_leaves(
        [snap("first", [question], [answer]), snap("second", [question], [answer])]
    )

    assert [leaf.source_path for leaf in result.leaves] == ["first", "second"]
    assert result.contributor_paths == {
        "first": ("first",),
        "second": ("second",),
    }


def test_late_empty_snapshot_is_covered_by_existing_response() -> None:
    answer = Message(role="assistant", content="a", reasoning_content="")
    result = streaming_prefix_leaves(
        [snap("long", [], [answer]), snap("empty", [], [])]
    )
    assert [leaf.source_path for leaf in result.leaves] == ["long"]
    assert result.contributor_paths["long"] == ("empty", "long")
