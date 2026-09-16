import gc
import weakref
from typing import Literal

from trajfoundry import streaming
from trajfoundry.models import (
    AgentMessageEvidence,
    CompactionRecord,
    FunctionCall,
    MediaMapping,
    Message,
    Snapshot,
    ToolCall,
)
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
        session_id="session",
        thread_id="thread",
        provider="openai",
        operation="responses",
        outcome="success",
        instructions=instructions,
        history=history,
        response=response,
    )


def compaction(
    *,
    origin: Literal["history", "response"] = "history",
    item_index: int = 1,
    encrypted_content: str = "opaque-a",
) -> CompactionRecord:
    return CompactionRecord(
        origin=origin,
        item_index=item_index,
        item={
            "type": "compaction",
            "id": "cmp-1",
            "encrypted_content": encrypted_content,
        },
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


def test_streaming_preserves_leaf_media_mapping_across_prefix_fold() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    final_question = Message(role="user", content="next")
    final = Message(role="assistant", content="done", reasoning_content="")
    short = snap("short-media", [question], [answer]).model_copy(
        update={
            "multimodal_file_mapping": [
                MediaMapping(part_id="media_0", object_name="first.png")
            ]
        }
    )
    leaf = snap(
        "leaf-media",
        [question, answer, final_question],
        [final],
    ).model_copy(
        update={
            "multimodal_file_mapping": [
                MediaMapping(part_id="media_0", object_name="first.png"),
                MediaMapping(part_id="media_1", object_name="second.jpg"),
            ]
        }
    )

    result = streaming_prefix_leaves([leaf, short])

    assert result.leaves == (leaf,)
    assert result.leaves[0].multimodal_file_mapping == [
        MediaMapping(part_id="media_0", object_name="first.png"),
        MediaMapping(part_id="media_1", object_name="second.jpg"),
    ]
    assert result.contributor_paths["leaf-media"] == ("leaf-media", "short-media")


def test_instructions_do_not_split_proven_prefix_chains() -> None:
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
    assert [leaf.source_path for leaf in result.leaves] == ["b"]
    assert result.contributor_paths["b"] == ("a", "b")


def test_reasoning_envelope_differences_do_not_split_prefix() -> None:
    question = Message(role="user", content="q")
    response_assistant = Message(
        role="assistant",
        content="",
        reasoning_content="visible summary",
        reasoning={
            "type": "reasoning",
            "id": "rs-1",
            "summary": [{"type": "summary_text", "text": "visible summary"}],
            "content": [],
            "encrypted_content": "opaque",
            "metadata": {"response_only": True},
        },
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(name="exec", arguments={"cmd": "pwd"}),
            )
        ],
    )
    replay_assistant = response_assistant.model_copy(
        update={
            "reasoning_content": "different visible replay",
            "reasoning": {
                "type": "reasoning",
                "id": "rs-1",
                "summary": [],
                "encrypted_content": "opaque",
            },
        }
    )
    tool_result = Message(role="tool", content="{}", tool_call_id="call-1", name="exec")
    leaf_answer = Message(role="assistant", content="done", reasoning_content="")

    short = snap("old/short", [question], [response_assistant])
    leaf = snap(
        "new/leaf",
        [question, replay_assistant, tool_result],
        [leaf_answer],
    )
    result = streaming_prefix_leaves([short, leaf])

    assert result.leaves == (leaf,)
    assert result.contributor_paths["new/leaf"] == ("new/leaf", "old/short")
    assert result.leaves[0].history[1] == replay_assistant


def test_identical_compaction_signature_allows_streaming_prefix_fold() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    next_question = Message(role="user", content="next")
    final = Message(role="assistant", content="done", reasoning_content="")
    record = compaction()
    short = snap("short", [question], [answer]).model_copy(
        update={"compaction_items": [record]}
    )
    long = snap("long", [question, answer, next_question], [final]).model_copy(
        update={"compaction_items": [record.model_copy(deep=True)]}
    )

    result = streaming_prefix_leaves([short, long])

    assert result.leaves == (long,)
    assert result.contributor_paths["long"] == ("long", "short")


def test_streaming_prefix_never_crosses_compaction_signature() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    next_question = Message(role="user", content="next")
    final = Message(role="assistant", content="done", reasoning_content="")
    short = snap("short", [question], [answer]).model_copy(
        update={"compaction_items": [compaction()]}
    )
    variants = (
        snap("missing", [question, answer, next_question], [final]),
        snap("origin", [question, answer, next_question], [final]).model_copy(
            update={"compaction_items": [compaction(origin="response")]}
        ),
        snap("index", [question, answer, next_question], [final]).model_copy(
            update={"compaction_items": [compaction(item_index=2)]}
        ),
        snap("content", [question, answer, next_question], [final]).model_copy(
            update={"compaction_items": [compaction(encrypted_content="opaque-b")]}
        ),
    )

    for variant in variants:
        result = streaming_prefix_leaves([short, variant])
        assert {leaf.source_path for leaf in result.leaves} == {
            "short",
            variant.source_path,
        }
        assert result.intermediate_paths == ()


def test_content_and_tool_call_differences_still_branch() -> None:
    question = Message(role="user", content="q")
    call_a = ToolCall(
        id="call-a", function=FunctionCall(name="exec", arguments={"cmd": "a"})
    )
    call_b = ToolCall(
        id="call-b", function=FunctionCall(name="exec", arguments={"cmd": "b"})
    )
    short_content = snap(
        "content-short",
        [question],
        [Message(role="assistant", content="a", reasoning_content="")],
    )
    long_content = snap(
        "content-long",
        [
            question,
            Message(role="assistant", content="b", reasoning_content="different"),
            Message(role="user", content="next"),
        ],
        [Message(role="assistant", content="done", reasoning_content="")],
    )
    short_call = snap(
        "call-short",
        [question],
        [
            Message(
                role="assistant",
                content="",
                reasoning_content="x",
                tool_calls=[call_a],
            )
        ],
    )
    long_call = snap(
        "call-long",
        [
            question,
            Message(
                role="assistant",
                content="",
                reasoning_content="y",
                tool_calls=[call_b],
            ),
            Message(role="tool", content="{}", tool_call_id="call-b", name="exec"),
        ],
        [Message(role="assistant", content="done", reasoning_content="")],
    )

    content_result = streaming_prefix_leaves([short_content, long_content])
    call_result = streaming_prefix_leaves([short_call, long_call])

    assert {leaf.source_path for leaf in content_result.leaves} == {
        "content-short",
        "content-long",
    }
    assert {leaf.source_path for leaf in call_result.leaves} == {
        "call-short",
        "call-long",
    }

    shared_call_message = Message(
        role="assistant",
        content="",
        reasoning_content="",
        tool_calls=[call_a],
    )
    completed_result = snap(
        "result-short",
        [question, shared_call_message],
        [Message(role="tool", content="one", tool_call_id="call-a", name="exec")],
    )
    different_result = snap(
        "result-long",
        [
            question,
            shared_call_message,
            Message(role="tool", content="two", tool_call_id="call-a", name="exec"),
            Message(role="user", content="next"),
        ],
        [Message(role="assistant", content="done", reasoning_content="")],
    )
    result_result = streaming_prefix_leaves([completed_result, different_result])
    assert {leaf.source_path for leaf in result_result.leaves} == {
        "result-short",
        "result-long",
    }


def test_prefix_never_crosses_session_or_thread() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    next_question = Message(role="user", content="next")
    final = Message(role="assistant", content="done", reasoning_content="")
    base = snap("base", [question], [answer])

    for different in (
        snap("session", [question, answer, next_question], [final]).model_copy(
            update={"session_id": "other-session"}
        ),
        snap("thread", [question, answer, next_question], [final]).model_copy(
            update={"thread_id": "other-thread"}
        ),
    ):
        result = streaming_prefix_leaves([base, different])
        assert {leaf.source_path for leaf in result.leaves} == {
            "base",
            different.source_path,
        }


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


def test_routing_evidence_drops_compaction_only_from_lightweight_copy() -> None:
    record = compaction(encrypted_content="must-not-enter-routing")
    source = snap(
        "child",
        [Message(role="user", content="q")],
        [Message(role="assistant", content="done", reasoning_content="")],
    ).model_copy(
        update={
            "parent_thread_id": "parent",
            "compaction_items": [record],
        }
    )

    result = streaming_prefix_leaves([source])

    assert result.leaves[0].compaction_items == [record]
    assert result.spawn_evidence[0].compaction_items == []


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


def test_missing_session_uses_user_scope_and_can_merge_across_threads() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    short = snap("short-no-session", [question], [answer]).model_copy(
        update={"session_id": "", "user_id": "user-a", "thread_id": "thread-1"}
    )
    long = snap(
        "long-no-session",
        [question, answer, Message(role="user", content="next")],
        [Message(role="assistant", content="done", reasoning_content="")],
    ).model_copy(
        update={"session_id": "", "user_id": "user-a", "thread_id": "thread-2"}
    )

    result = streaming_prefix_leaves(item for item in (short, long))

    assert [leaf.source_path for leaf in result.leaves] == ["long-no-session"]
    assert result.contributor_paths["long-no-session"] == (
        "long-no-session",
        "short-no-session",
    )


def test_missing_session_does_not_merge_different_users_or_known_sessions() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    base = snap("base", [question], [answer]).model_copy(
        update={"session_id": "", "user_id": "user-a"}
    )
    other_user = snap(
        "other-user",
        [question, answer, Message(role="user", content="next")],
        [Message(role="assistant", content="done", reasoning_content="")],
    ).model_copy(update={"session_id": "", "user_id": "user-b"})
    known_session = snap(
        "known-session",
        [question, answer, Message(role="user", content="next")],
        [Message(role="assistant", content="done", reasoning_content="")],
    ).model_copy(update={"session_id": "session-known", "user_id": "user-a"})

    result = streaming_prefix_leaves([base, other_user, known_session])

    assert {leaf.source_path for leaf in result.leaves} == {
        "base",
        "other-user",
        "known-session",
    }
    assert result.intermediate_paths == ()


def test_public_identity_sentinel_strings_remain_explicit_internal_ids() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    short = snap("short", [question], [answer]).model_copy(
        update={"session_id": "", "user_id": ""}
    )
    literal_user = snap(
        "literal-user",
        [question, answer, Message(role="user", content="next")],
        [Message(role="assistant", content="done", reasoning_content="")],
    ).model_copy(update={"session_id": "", "user_id": "no_user_id"})
    literal_session = literal_user.model_copy(
        update={
            "source_path": "literal-session",
            "session_id": "no_session_id",
            "user_id": "",
        }
    )

    result = streaming_prefix_leaves([short, literal_user, literal_session])

    assert {leaf.source_path for leaf in result.leaves} == {
        "short",
        "literal-user",
        "literal-session",
    }
    assert result.intermediate_paths == ()


def test_explicit_request_id_session_is_a_real_session_scope() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    first = snap("request-1", [question], [answer]).model_copy(
        update={"session_id": "request-1", "request_id": "request-1", "user_id": "u"}
    )
    second = snap(
        "request-2",
        [question, answer, Message(role="user", content="next")],
        [Message(role="assistant", content="done", reasoning_content="")],
    ).model_copy(
        update={"session_id": "request-2", "request_id": "request-2", "user_id": "u"}
    )

    result = streaming_prefix_leaves([first, second])

    assert {leaf.source_path for leaf in result.leaves} == {"request-1", "request-2"}


def test_custom_scope_function_overrides_default_session_scope() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    first = snap("first", [question], [answer]).model_copy(
        update={"session_id": "session-a"}
    )
    second = snap(
        "second",
        [question, answer, Message(role="user", content="next")],
        [Message(role="assistant", content="done", reasoning_content="")],
    ).model_copy(update={"session_id": "session-b"})

    result = streaming_prefix_leaves([first, second], scope_fn=lambda _: "shared")

    assert [leaf.source_path for leaf in result.leaves] == ["second"]
    assert result.contributor_paths["second"] == ("first", "second")


def test_generator_releases_covered_snapshot_before_input_is_exhausted() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    references: list[weakref.ReferenceType[Snapshot]] = []

    def source():
        short = snap("short", [question], [answer])
        references.append(weakref.ref(short))
        yield short
        del short

        long = snap(
            "long",
            [question, answer, Message(role="user", content="next")],
            [Message(role="assistant", content="done", reasoning_content="")],
        )
        yield long
        del long

        gc.collect()
        assert references[0]() is None

    result = streaming_prefix_leaves(source())

    assert [leaf.source_path for leaf in result.leaves] == ["long"]


def test_short_prefix_contributes_to_every_divergent_maximal_leaf() -> None:
    question = Message(role="user", content="q")
    answer = Message(role="assistant", content="a", reasoning_content="")
    follow_up = Message(role="user", content="next")
    short = snap("short", [question], [answer])
    middle = snap(
        "middle",
        [question, answer, follow_up],
        [Message(role="assistant", content="middle", reasoning_content="")],
    )
    branch_a = snap(
        "branch-a",
        [
            question,
            answer,
            follow_up,
            Message(role="assistant", content="middle", reasoning_content=""),
            Message(role="user", content="branch"),
        ],
        [Message(role="assistant", content="a", reasoning_content="")],
    )
    branch_b = snap(
        "branch-b",
        [question, answer, follow_up],
        [Message(role="assistant", content="other", reasoning_content="")],
    )

    result = streaming_prefix_leaves([short, middle, branch_a, branch_b])

    assert [leaf.source_path for leaf in result.leaves] == ["branch-a", "branch-b"]
    assert result.contributor_paths == {
        "branch-a": ("branch-a", "middle", "short"),
        "branch-b": ("branch-b", "short"),
    }


def test_unrelated_leaves_do_not_scan_the_active_frontier(monkeypatch) -> None:
    advance_calls = 0
    original_advance = streaming._TranscriptTrie.advance

    def counted_advance(node, token):
        nonlocal advance_calls
        advance_calls += 1
        return original_advance(node, token)

    monkeypatch.setattr(
        streaming._TranscriptTrie, "advance", staticmethod(counted_advance)
    )
    snapshots = [
        snap(
            f"leaf-{index}",
            [Message(role="user", content=f"question-{index}")],
            [
                Message(
                    role="assistant", content=f"answer-{index}", reasoning_content=""
                )
            ],
        )
        for index in range(500)
    ]

    result = streaming_prefix_leaves(snapshots)

    assert len(result.leaves) == len(snapshots)
    assert advance_calls == sum(
        len(snapshot.history) + len(snapshot.response) for snapshot in snapshots
    )
