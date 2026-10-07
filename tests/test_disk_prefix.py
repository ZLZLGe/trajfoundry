import gc
import random
import sqlite3
import weakref
from itertools import permutations

import pytest

from trajfoundry import disk_prefix
from trajfoundry.disk_prefix import disk_prefix_leaves
from trajfoundry.models import (
    AgentMessageEvidence,
    AuditIssue,
    CompactionRecord,
    FunctionCall,
    Message,
    Severity,
    Snapshot,
    ToolCall,
)
from trajfoundry.streaming import streaming_prefix_leaves


def snapshot(path, history, response, **metadata):
    return Snapshot(
        source_path=path,
        source_sha256=path,
        session_id="session",
        thread_id="thread",
        provider="openai",
        operation="responses",
        outcome="success",
        history=history,
        response=response,
    ).model_copy(update=metadata)


def user(text):
    return Message(role="user", content=text)


def assistant(text):
    return Message(role="assistant", content=text, reasoning_content="")


def assert_matches_reference(snapshots, directory, **kwargs):
    expected = streaming_prefix_leaves(snapshots, **kwargs)
    with disk_prefix_leaves(snapshots, directory, **kwargs) as actual:
        assert tuple(actual.leaves) == expected.leaves
        assert len(actual.leaves) == len(expected.leaves)
        assert tuple(actual.leaves) == expected.leaves  # Reiterable, not a cursor.
        assert tuple(actual.intermediate_paths) == expected.intermediate_paths
        assert len(actual.intermediate_paths) == len(expected.intermediate_paths)
        assert tuple(actual.spawn_evidence) == expected.spawn_evidence
        assert {
            path: tuple(values) for path, values in actual.contributor_paths.items()
        } == expected.contributor_paths
        assert len(actual.contributor_paths) == len(expected.contributor_paths)
        for path, values in actual.contributor_paths.items():
            assert len(values) == len(expected.contributor_paths[path])
            assert tuple(values) == expected.contributor_paths[path]
        with pytest.raises(KeyError):
            actual.contributor_paths["not-an-input-source"]


def test_prefix_chain_and_divergent_leaves_in_every_input_order(tmp_path):
    q, a, q2 = user("q"), assistant("a"), user("next")
    short = snapshot("short", [q], [a])
    left = snapshot("left", [q, a, q2], [assistant("left")])
    right = snapshot("right", [q, a, q2], [assistant("right")])
    for items in permutations([short, left, right]):
        assert_matches_reference(items, tmp_path)


def test_equal_empty_and_history_only_transcripts(tmp_path):
    q, a = user("q"), assistant("a")
    items = [
        snapshot("empty", [], []),
        snapshot("empty-equal", [], []),
        snapshot("first", [q], [a]),
        snapshot("equal", [q], [a]),
        snapshot("history-only", [q, a], []),
    ]
    assert_matches_reference(items, tmp_path)
    assert_matches_reference(items[::-1], tmp_path)
    with disk_prefix_leaves(items, tmp_path) as result:
        assert tuple(item.source_path for item in result.leaves) == (
            "equal",
            "first",
            "history-only",
        )
        assert tuple(result.contributor_paths["first"]) == (
            "empty",
            "empty-equal",
            "first",
        )


@pytest.mark.parametrize(
    "short_fields,long_fields",
    [
        ({}, {"session_id": "different"}),
        ({}, {"thread_id": "different"}),
        (
            {"session_id": "", "user_id": "u"},
            {"session_id": "", "user_id": "u", "thread_id": "other"},
        ),
        ({"session_id": "", "user_id": "u"}, {"session_id": "", "user_id": "v"}),
        (
            {"session_id": "", "user_id": ""},
            {"session_id": "", "user_id": "no_user_id"},
        ),
        (
            {"session_id": "", "user_id": "u"},
            {"session_id": "no_session_id", "user_id": "u"},
        ),
    ],
)
def test_scope_isolation_and_missing_session_rules(tmp_path, short_fields, long_fields):
    q, a = user("q"), assistant("a")
    items = [
        snapshot("short", [q], [a], **short_fields),
        snapshot("long", [q, a, user("next")], [assistant("done")], **long_fields),
    ]
    assert_matches_reference(items, tmp_path)


def test_masked_session_compaction_and_custom_scope(tmp_path):
    q, a = user("q"), assistant("a")
    masked = AuditIssue(
        code="metadata_session_id_masked", stage="metadata", severity=Severity.WARNING
    )
    compact = CompactionRecord(
        origin="history",
        item_index=1,
        item={"type": "compaction", "encrypted_content": "opaque"},
    )
    short = snapshot("short", [q], [a])
    long = snapshot("long", [q, a, user("next")], [assistant("done")])
    assert_matches_reference(
        [
            item.model_copy(update={"session_id": "", "issues": [masked]})
            for item in (short, long)
        ],
        tmp_path,
    )
    assert_matches_reference(
        [short.model_copy(update={"compaction_items": [compact]}), long], tmp_path
    )
    assert_matches_reference(
        [
            item.model_copy(update={"compaction_items": [compact]})
            for item in (short, long)
        ],
        tmp_path,
    )
    assert_matches_reference(
        [short, long.model_copy(update={"session_id": "different"})],
        tmp_path,
        scope_fn=lambda _: "shared",
    )


def test_tool_calls_reasoning_and_routing_evidence_preserved(tmp_path):
    question = user("q")
    spawn = Message(
        role="assistant",
        content="",
        reasoning_content="private reasoning",
        tool_calls=[
            ToolCall(
                id="spawn-1",
                function=FunctionCall(name="spawn_agent", arguments={"task": "test"}),
            )
        ],
    )
    response = Message(
        role="tool",
        content='{"id":"child"}',
        tool_call_id="spawn-1",
        name="spawn_agent",
    )
    short = snapshot("short", [question], [spawn], parent_thread_id="parent")
    long = snapshot(
        "long",
        [
            question,
            spawn.model_copy(update={"reasoning_content": "replayed"}),
            response,
        ],
        [assistant("done")],
        agent_messages=[
            AgentMessageEvidence(
                origin="history",
                item_index=2,
                item={
                    "type": "agent_message",
                    "id": "m",
                    "author": "a",
                    "recipient": "b",
                    "content": "opaque",
                },
            )
        ],
    )
    alternate = snapshot(
        "different-call",
        [question],
        [
            spawn.model_copy(
                update={
                    "tool_calls": [
                        ToolCall(
                            id="spawn-2",
                            function=FunctionCall(
                                name="spawn_agent", arguments={"task": "other"}
                            ),
                        )
                    ]
                }
            )
        ],
    )
    assert_matches_reference([short, long, alternate], tmp_path)


def test_hash_collisions_still_compare_exact_bytes(tmp_path, monkeypatch):
    class Collision:
        def digest(self):
            return b"collision"

    monkeypatch.setattr(disk_prefix, "sha256", lambda _: Collision())
    items = [
        snapshot("one", [user("different-one")], [assistant("answer")]),
        snapshot("two", [user("different-two")], [assistant("answer")]),
    ]
    assert_matches_reference(items, tmp_path)
    with disk_prefix_leaves(items, tmp_path, lookup_cache_bytes=0) as result:
        assert len(result.leaves) == 2
        assert [item.source_path for item in result.leaves] == ["one", "two"]


def test_external_fetch_is_lazy_and_payload_is_not_duplicated(tmp_path):
    q, a = user("q"), assistant("a")
    items = [
        snapshot("short", [q], [a]),
        snapshot("long", [q, a, user("next")], [assistant("done")]),
    ]
    indexed = {item.source_path: item for item in items}
    fetched = []

    def fetch(path):
        fetched.append(path)
        return indexed[path]

    with disk_prefix_leaves(items, tmp_path, fetch_snapshot=fetch) as result:
        assert not fetched
        assert len(result.leaves) == 1
        assert not fetched
        assert result.leaves[0] == items[1]
        assert fetched == ["long"]
        with sqlite3.connect(result.database_path) as connection:
            assert (
                connection.execute(
                    "SELECT count(*) FROM snapshots WHERE payload IS NOT NULL"
                ).fetchone()[0]
                == 0
            )
        path = result.database_path
    assert not path.exists()


def test_missing_external_snapshot_is_not_silently_dropped(tmp_path):
    item = snapshot("missing", [user("q")], [assistant("a")])
    with (
        disk_prefix_leaves([item], tmp_path, fetch_snapshot=lambda _: None) as result,
        pytest.raises(RuntimeError, match="disappeared"),
    ):
        list(result.leaves)


def test_input_snapshots_are_released_and_empty_input_is_supported(tmp_path):
    references = []

    def source():
        for index in range(3):
            item = snapshot(str(index), [user(str(index))], [assistant(str(index))])
            references.append(weakref.ref(item))
            yield item
            del item
        gc.collect()
        assert references[0]() is None

    with disk_prefix_leaves(source(), tmp_path, lookup_cache_bytes=1024) as result:
        gc.collect()
        assert all(reference() is None for reference in references)
        assert len(result.leaves) == 3
        assert result.leaves[-1].source_path == "2"
        with pytest.raises(IndexError):
            result.leaves[3]
    assert_matches_reference([], tmp_path)


def test_result_and_scratch_cleanup_when_consumer_raises(tmp_path):
    with (
        pytest.raises(RuntimeError, match="caller failed"),
        disk_prefix_leaves([], tmp_path) as result,
    ):
        path = result.database_path
        assert path.exists()
        raise RuntimeError("caller failed")
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_bounded_lookup_cache_does_not_retain_oversized_tokens():
    cache = disk_prefix._ByteCache(1024)
    cache.put("first", 1, 100)
    cache.put("second", 2, 100)
    cache.put("third", 3, 100)
    assert cache.bytes <= 1024
    assert cache.get("first") is None
    cache.put("huge", 4, 4096)
    assert cache.get("huge") is None
    assert cache.bytes <= 1024


def test_small_randomized_differential_cases(tmp_path):
    # Functional equivalence across randomized ordering/history boundaries;
    # intentionally small, not a load or memory stress test.
    randomizer = random.Random(309)
    messages = [user("a"), user("b"), assistant("a"), assistant("b")]
    for case in range(40):
        items = []
        for index in range(12):
            transcript = [
                randomizer.choice(messages) for _ in range(randomizer.randrange(7))
            ]
            split = randomizer.randrange(len(transcript) + 1)
            items.append(
                snapshot(
                    f"{case}-{index}",
                    transcript[:split],
                    transcript[split:],
                    captured_at=str(randomizer.randrange(3)),
                )
            )
        randomizer.shuffle(items)
        assert_matches_reference(items, tmp_path)
