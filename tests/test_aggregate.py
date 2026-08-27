from __future__ import annotations

from trajfoundry.aggregate import aggregate_snapshots
from trajfoundry.models import Message, Snapshot


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str) -> Message:
    return Message(role="assistant", content=content, reasoning_content="")


def snapshot(
    name: str,
    history: list[Message],
    response: list[Message],
    *,
    partition: str = "p",
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
        source_partition=partition,
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


def test_prefix_never_crosses_primary_group_or_semantic_configuration() -> None:
    q1, a1, q2 = user("q1"), assistant("a1"), user("q2")
    base = snapshot("base", [q1], [a1])
    variants = [
        snapshot("partition", [q1, a1, q2], [assistant("a2")], partition="p2"),
        snapshot("session", [q1, a1, q2], [assistant("a2")], session="s2"),
        snapshot("thread", [q1, a1, q2], [assistant("a2")], thread="t2"),
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

    result = aggregate_snapshots([base, *variants])

    assert len(result.leaves) == 1 + len(variants)
    assert result.intermediate == ()


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
