from __future__ import annotations

from typing import Literal

from trajfoundry.aggregate import aggregate_snapshots
from trajfoundry.models import (
    CompactionRecord,
    FunctionCall,
    Message,
    Snapshot,
    ToolCall,
)


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str) -> Message:
    return Message(role="assistant", content=content, reasoning_content="")


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


def snapshot(
    name: str,
    history: list[Message],
    response: list[Message],
    *,
    session: str = "s",
    thread: str = "t",
    provider: str = "openai",
    model: str = "gpt",
    harness: str = "codex",
    instructions: str = "rules",
) -> Snapshot:
    return Snapshot(
        source_path=f"/{name}.json",
        source_sha256=name,
        session_id=session,
        thread_id=thread,
        turn_id=name,
        provider=provider,
        operation="responses" if provider == "openai" else "messages",
        outcome="success",
        request_id=name,
        model=model,
        harness=harness,
        instructions=instructions,
        history=history,
        response=response,
        wire_complete=True,
    )


def test_linear_cumulative_snapshots_keep_only_maximal_leaf() -> None:
    q1, a1, q2, a2 = user("q1"), assistant("a1"), user("q2"), assistant("a2")
    first = snapshot("first", [q1], [a1])
    second = snapshot("second", [q1, a1, q2], [a2])

    result = aggregate_snapshots([second, first])

    assert result.leaves == (second,)
    assert result.intermediate == (first,)
    assert len(result.prefix_links) == 1
    link = result.prefix_links[0]
    assert [second, first][link.covered_index] is first
    assert [second, first][link.extending_index] is second
    assert link.matched_messages == 2


def test_real_branches_are_all_preserved() -> None:
    q1, a1, q2 = user("q1"), assistant("a1"), user("q2")
    first = snapshot("first", [q1], [a1])
    branch_a = snapshot("branch-a", [q1, a1, q2], [assistant("answer-a")])
    branch_b = snapshot("branch-b", [q1, a1, q2], [assistant("answer-b")])

    result = aggregate_snapshots([branch_b, first, branch_a])

    assert result.leaves == (branch_a, branch_b)
    assert result.intermediate == (first,)


def test_identical_compaction_signature_allows_batch_prefix_fold() -> None:
    q1, a1, q2, a2 = user("q1"), assistant("a1"), user("q2"), assistant("a2")
    record = compaction()
    first = snapshot("first", [q1], [a1]).model_copy(
        update={"compaction_items": [record]}
    )
    second = snapshot("second", [q1, a1, q2], [a2]).model_copy(
        update={"compaction_items": [record.model_copy(deep=True)]}
    )

    result = aggregate_snapshots([second, first])

    assert result.leaves == (second,)
    assert result.intermediate == (first,)


def test_batch_prefix_never_crosses_compaction_signature() -> None:
    q1, a1, q2, a2 = user("q1"), assistant("a1"), user("q2"), assistant("a2")
    first = snapshot("first", [q1], [a1]).model_copy(
        update={"compaction_items": [compaction()]}
    )
    variants = (
        snapshot("missing", [q1, a1, q2], [a2]),
        snapshot("origin", [q1, a1, q2], [a2]).model_copy(
            update={"compaction_items": [compaction(origin="response")]}
        ),
        snapshot("index", [q1, a1, q2], [a2]).model_copy(
            update={"compaction_items": [compaction(item_index=2)]}
        ),
        snapshot("content", [q1, a1, q2], [a2]).model_copy(
            update={"compaction_items": [compaction(encrypted_content="opaque-b")]}
        ),
    )

    for variant in variants:
        result = aggregate_snapshots([first, variant])
        assert {leaf.source_path for leaf in result.leaves} == {
            first.source_path,
            variant.source_path,
        }
        assert result.intermediate == ()


def test_prefix_never_crosses_session_or_thread() -> None:
    q1, a1, q2 = user("q1"), assistant("a1"), user("q2")
    base = snapshot("base", [q1], [a1])
    variants = [
        snapshot("session", [q1, a1, q2], [assistant("a2")], session="s2"),
        snapshot("thread", [q1, a1, q2], [assistant("a2")], thread="t2"),
    ]

    result = aggregate_snapshots([base, *variants])

    assert len(result.leaves) == 1 + len(variants)
    assert result.intermediate == ()


def test_semantic_request_settings_do_not_block_proven_prefix() -> None:
    q1, a1, q2 = user("q1"), assistant("a1"), user("q2")
    variants = [
        snapshot(
            "provider",
            [q1, a1, q2],
            [assistant("a2")],
            provider="anthropic",
        ),
        snapshot("model", [q1, a1, q2], [assistant("a2")], model="other"),
        snapshot("harness", [q1, a1, q2], [assistant("a2")], harness="other"),
        snapshot(
            "instructions",
            [q1, a1, q2],
            [assistant("a2")],
            instructions="changed",
        ),
    ]

    for variant in variants:
        base = snapshot("base", [q1], [a1])
        result = aggregate_snapshots([base, variant])
        assert result.leaves == (variant,)
        assert result.intermediate == (base,)


def test_reasoning_differences_are_ignored_but_tool_results_are_not() -> None:
    q1 = user("q1")
    call = ToolCall(
        id="call-1",
        function=FunctionCall(name="exec", arguments={"cmd": "pwd"}),
    )
    response_form = Message(
        role="assistant",
        content="",
        reasoning_content="summary",
        reasoning={"type": "reasoning", "metadata": {"response_only": True}},
        tool_calls=[call],
    )
    replay_form = response_form.model_copy(
        update={
            "reasoning_content": "",
            "reasoning": {"type": "reasoning"},
        }
    )
    short = snapshot("short", [q1], [response_form])
    matching = snapshot(
        "matching",
        [
            q1,
            replay_form,
            Message(role="tool", content="{}", tool_call_id="call-1", name="exec"),
        ],
        [assistant("done")],
    )

    assert aggregate_snapshots([short, matching]).intermediate == (short,)

    completed = snapshot(
        "completed",
        [q1, replay_form],
        [Message(role="tool", content="one", tool_call_id="call-1", name="exec")],
    )
    different_result = snapshot(
        "different-result",
        [
            q1,
            replay_form,
            Message(role="tool", content="two", tool_call_id="call-1", name="exec"),
            user("next"),
        ],
        [assistant("done")],
    )
    result = aggregate_snapshots([completed, different_result])
    assert {item.source_path for item in result.leaves} == {
        completed.source_path,
        different_result.source_path,
    }


def test_complete_snapshot_must_prefix_later_history_not_later_response() -> None:
    q1, a1 = user("q1"), assistant("a1")
    first = snapshot("first", [q1], [a1])
    # The concatenated transcript starts like first, but a1 belongs to the
    # second response rather than its history.  This is not turn continuation
    # evidence and must not suppress first.
    malformed_continuation = snapshot("malformed-continuation", [q1], [a1, user("q2")])

    result = aggregate_snapshots([first, malformed_continuation])

    assert result.leaves == (first, malformed_continuation)


def test_equal_terminal_duplicates_are_not_treated_as_prefixes() -> None:
    q1, a1 = user("q1"), assistant("a1")
    left = snapshot("left", [q1], [a1])
    right = snapshot("right", [q1], [a1])

    result = aggregate_snapshots([right, left])

    # Exact global deduplication is a later pipeline stage with lineage.  The
    # cumulative-snapshot stage has no later-history evidence here.
    assert result.leaves == (left, right)
    assert result.intermediate == ()


def test_empty_snapshot_is_suppressed_only_by_a_strict_continuation() -> None:
    empty = snapshot("empty", [], [])
    later = snapshot("later", [], [assistant("answer")])

    result = aggregate_snapshots([later, empty])

    assert result.leaves == (later,)
    assert result.intermediate == (empty,)


def test_output_and_shortest_prefix_witness_are_input_order_independent() -> None:
    q1, a1, q2, a2, q3, a3 = (
        user("q1"),
        assistant("a1"),
        user("q2"),
        assistant("a2"),
        user("q3"),
        assistant("a3"),
    )
    first = snapshot("first", [q1], [a1])
    middle = snapshot("middle", [q1, a1, q2], [a2])
    last = snapshot("last", [q1, a1, q2, a2, q3], [a3])

    forward = aggregate_snapshots([first, middle, last])
    backward = aggregate_snapshots([last, middle, first])

    assert [item.source_path for item in forward.leaves] == ["/last.json"]
    assert [item.source_path for item in backward.leaves] == ["/last.json"]
    forward_links = {
        [first, middle, last][link.covered_index].source_path: [first, middle, last][
            link.extending_index
        ].source_path
        for link in forward.prefix_links
    }
    backward_links = {
        [last, middle, first][link.covered_index].source_path: [last, middle, first][
            link.extending_index
        ].source_path
        for link in backward.prefix_links
    }
    assert (
        forward_links
        == backward_links
        == {
            "/first.json": "/middle.json",
            "/middle.json": "/last.json",
        }
    )
