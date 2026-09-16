import sqlite3
import zlib
from pathlib import Path

import orjson
import zstandard

from trajfoundry.canonical import trajectory_id
from trajfoundry.export import OutputSet
from trajfoundry.io import JsonlShardWriter, load_capture
from trajfoundry.models import (
    AuditTag,
    CompactionRecord,
    MediaMapping,
    Message,
    Metadata,
    NormalizationAudit,
    Snapshot,
    TrajectoryNode,
)
from trajfoundry.quality import enrich_trajectory
from trajfoundry.state import StateStore


def test_capture_loader_drops_sensitive_headers(tmp_path: Path) -> None:
    path = tmp_path / "v1" / "dt=x" / "partition" / "session" / "one.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        orjson.dumps(
            {
                "path": "/v1/responses",
                "request_headers": {
                    "authorization": "secret",
                    "x-forwarded-for": "127.0.0.1",
                    "x-openai-subagent": "collab_spawn",
                },
                "response_headers": {"set-cookie": "secret"},
            }
        )
    )
    capture = load_capture(path)
    assert capture["request_headers"] == {"x-openai-subagent": "collab_spawn"}
    assert "response_headers" not in capture


def test_state_round_trip_compresses_snapshot(tmp_path: Path) -> None:
    snapshot = Snapshot(
        source_path="v1/p/s/a.json",
        source_sha256="a" * 64,
        session_id="s",
        thread_id="t",
        provider="openai",
        operation="responses",
        outcome="success",
        history=[Message(role="user", content="hello")],
        multimodal_file_mapping=[
            MediaMapping(part_id="media_0", object_name="media_0-image.png")
        ],
        compaction_items=[
            CompactionRecord(
                origin="history",
                item_index=1,
                item={
                    "type": "compaction",
                    "id": "cmp-1",
                    "encrypted_content": "opaque",
                },
            )
        ],
    )
    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_snapshot(snapshot)
        payload = state.connection.execute(
            "SELECT payload FROM snapshots WHERE source_path=?",
            (snapshot.source_path,),
        ).fetchone()[0]
        assert payload.startswith(b"TFZ1")
        assert zstandard.ZstdDecompressor().decompress(payload[4:])
        restored = list(state.iter_snapshots())
    assert restored == [snapshot]


def test_state_reads_legacy_zlib_snapshot_payload_in_v2_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite"
    snapshot = Snapshot(
        source_path="legacy.json",
        source_sha256="a" * 64,
        session_id="s",
        thread_id="t",
        provider="openai",
        operation="responses",
        outcome="success",
    )
    legacy = zlib.compress(snapshot.model_dump_json(exclude_none=True).encode())
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE captures (
                source_path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                endpoint TEXT NOT NULL DEFAULT '',
                captured_at TEXT NOT NULL DEFAULT '',
                scan_id TEXT NOT NULL DEFAULT '',
                user_id TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE snapshots (
                source_path TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX idx_snapshots_group
            ON snapshots(session_id, thread_id, captured_at, source_path);
            PRAGMA user_version=2;
            """
        )
        connection.execute(
            "INSERT INTO snapshots("
            "source_path,session_id,thread_id,captured_at,payload"
            ") VALUES(?,?,?,?,?)",
            (
                snapshot.source_path,
                snapshot.session_id,
                snapshot.thread_id,
                snapshot.captured_at,
                legacy,
            ),
        )
        connection.execute(
            "INSERT INTO captures(source_path,sha256,status) VALUES(?,?,?)",
            (snapshot.source_path, snapshot.source_sha256, "parsed"),
        )

    with StateStore(path) as state:
        assert state.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert state.connection.execute("PRAGMA page_size").fetchone()[0] == 4096
        assert list(state.iter_snapshots()) == [snapshot]
        assert (
            state.connection.execute("SELECT payload FROM snapshots").fetchone()[0]
            == legacy
        )

        state.put_snapshot(snapshot)
        assert (
            state.connection.execute("SELECT payload FROM snapshots")
            .fetchone()[0]
            .startswith(b"TFZ1")
        )


def test_state_rebuilds_v1_cache_before_sessionless_user_aggregation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite"
    snapshots = [
        Snapshot(
            source_path=f"{user}.json",
            source_sha256=character * 64,
            session_id="",
            thread_id=f"request-{user}",
            user_id=user,
            provider="openai",
            operation="responses",
            outcome="success",
        )
        for user, character in (("alice", "a"), ("bob", "b"))
    ]
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE captures (
                source_path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                endpoint TEXT NOT NULL DEFAULT '',
                captured_at TEXT NOT NULL DEFAULT '',
                scan_id TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE snapshots (
                source_path TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            PRAGMA user_version=1;
            """
        )
        for snapshot in snapshots:
            connection.execute(
                "INSERT INTO captures(source_path,sha256,status) VALUES(?,?,?)",
                (snapshot.source_path, snapshot.source_sha256, "parsed"),
            )
            connection.execute(
                "INSERT INTO snapshots VALUES(?,?,?,?,?)",
                (
                    snapshot.source_path,
                    snapshot.session_id,
                    snapshot.thread_id,
                    snapshot.captured_at,
                    zlib.compress(snapshot.model_dump_json().encode()),
                ),
            )

    with StateStore(path) as state:
        assert state.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert state.capture_count() == 0
        assert list(state.iter_snapshots()) == []
        assert list(state.aggregation_scopes()) == []


def test_state_write_batch_commits_and_rolls_back(tmp_path: Path) -> None:
    def make_snapshot(path: str) -> Snapshot:
        return Snapshot(
            source_path=path,
            source_sha256="a" * 64,
            session_id="s",
            thread_id="t",
            provider="openai",
            operation="responses",
            outcome="success",
        )

    with StateStore(tmp_path / "state.sqlite") as state:
        with state.write_batch():
            state.put_snapshot(make_snapshot("one.json"))
            state.put_snapshot(make_snapshot("two.json"))
        assert state.capture_count() == 2

        try:
            with state.write_batch():
                state.put_snapshot(make_snapshot("three.json"))
                raise RuntimeError("abort batch")
        except RuntimeError:
            pass
        assert state.capture_count() == 2
        assert state.get_snapshot("three.json") is None

    with StateStore(tmp_path / "state.sqlite") as state:
        assert [snapshot.source_path for snapshot in state.iter_snapshots()] == [
            "one.json",
            "two.json",
        ]


def test_state_configures_page_size_and_ephemeral_pragmas(tmp_path: Path) -> None:
    durable_path = tmp_path / "durable.sqlite"
    with StateStore(durable_path) as state:
        assert state.connection.execute("PRAGMA page_size").fetchone()[0] == 32 * 1024
        assert state.connection.execute("PRAGMA cache_size").fetchone()[0] == -(
            128 * 1024
        )
        assert state.connection.execute("PRAGMA mmap_size").fetchone()[0] >= (
            256 * 1024 * 1024
        )
        assert (
            state.connection.execute("PRAGMA journal_mode").fetchone()[0].lower()
            == "wal"
        )
        assert state.connection.execute("PRAGMA synchronous").fetchone()[0] == 1

    ephemeral_path = tmp_path / "ephemeral.sqlite"
    with StateStore(ephemeral_path, ephemeral=True) as state:
        assert state.connection.execute("PRAGMA page_size").fetchone()[0] == 32 * 1024
        assert (
            state.connection.execute("PRAGMA journal_mode").fetchone()[0].lower()
            == "memory"
        )
        assert state.connection.execute("PRAGMA synchronous").fetchone()[0] == 0


def test_iter_snapshot_groups_uses_ordered_groups(tmp_path: Path) -> None:
    def make_snapshot(path: str, session: str, thread: str) -> Snapshot:
        return Snapshot(
            source_path=path,
            source_sha256="a" * 64,
            session_id=session,
            thread_id=thread,
            provider="openai",
            operation="responses",
            outcome="success",
        )

    with StateStore(tmp_path / "state.sqlite") as state:
        for snapshot in (
            make_snapshot("b.json", "s2", "t1"),
            make_snapshot("a.json", "s1", "t2"),
            make_snapshot("c.json", "s1", "t1"),
            make_snapshot("empty.json", "s3", ""),
        ):
            state.put_snapshot(snapshot)
        groups = list(state.iter_snapshot_groups())
        assert [(session, thread) for session, thread, _ in groups] == [
            ("s1", "t1"),
            ("s1", "t2"),
            ("s2", "t1"),
        ]
        assert [snapshot.source_path for snapshot in groups[0][2]] == ["c.json"]
        all_groups = list(state.iter_snapshot_groups(include_empty_threads=True))
        assert ("s3", "") in [(session, thread) for session, thread, _ in all_groups]


def test_state_streams_session_and_missing_session_user_scopes(
    tmp_path: Path,
) -> None:
    def make_snapshot(
        path: str,
        *,
        session: str,
        thread: str,
        user: str,
        captured_at: str,
    ) -> Snapshot:
        return Snapshot(
            source_path=path,
            source_sha256="a" * 64,
            session_id=session,
            thread_id=thread,
            user_id=user,
            captured_at=captured_at,
            provider="openai",
            operation="responses",
            outcome="success",
        )

    snapshots = (
        make_snapshot(
            "session-thread-2.json",
            session="session-1",
            thread="thread-2",
            user="ignored-user",
            captured_at="2026-09-16T00:00:02Z",
        ),
        make_snapshot(
            "session-thread-1.json",
            session="session-1",
            thread="thread-1",
            user="different-ignored-user",
            captured_at="2026-09-16T00:00:01Z",
        ),
        make_snapshot(
            "alice-later.json",
            session="",
            thread="ignored-2",
            user="alice",
            captured_at="2026-09-16T00:00:04Z",
        ),
        make_snapshot(
            "alice-earlier.json",
            session="",
            thread="ignored-1",
            user="alice",
            captured_at="2026-09-16T00:00:03Z",
        ),
        make_snapshot(
            "bob.json",
            session="",
            thread="",
            user="bob",
            captured_at="2026-09-16T00:00:05Z",
        ),
        make_snapshot(
            "missing-user.json",
            session="",
            thread="",
            user="",
            captured_at="2026-09-16T00:00:06Z",
        ),
        make_snapshot(
            "literal-sentinel-session.json",
            session="no_session_id",
            thread="",
            user="ignored-user",
            captured_at="2026-09-16T00:00:07Z",
        ),
        make_snapshot(
            "literal-sentinel-user.json",
            session="",
            thread="",
            user="no_user_id",
            captured_at="2026-09-16T00:00:08Z",
        ),
    )

    with StateStore(tmp_path / "state.sqlite") as state:
        with state.write_batch():
            for snapshot in snapshots:
                state.put_snapshot(snapshot)

        assert list(state.aggregation_scopes()) == [
            ("session", "no_session_id"),
            ("session", "session-1"),
            ("user", ""),
            ("user", "alice"),
            ("user", "bob"),
            ("user", "no_user_id"),
        ]
        assert [
            snapshot.source_path
            for snapshot in state.snapshots_for_aggregation_scope(
                ("session", "session-1")
            )
        ] == ["session-thread-1.json", "session-thread-2.json"]
        assert [
            snapshot.source_path
            for snapshot in state.snapshots_for_aggregation_scope(("user", "alice"))
        ] == ["alice-earlier.json", "alice-later.json"]
        assert [
            snapshot.source_path
            for snapshot in state.snapshots_for_aggregation_scope(
                ("session", "no_session_id")
            )
        ] == ["literal-sentinel-session.json"]
        assert [
            snapshot.source_path
            for snapshot in state.snapshots_for_aggregation_scope(("user", ""))
        ] == ["missing-user.json"]
        assert [
            snapshot.source_path
            for snapshot in state.snapshots_for_aggregation_scope(
                ("user", "no_user_id")
            )
        ] == ["literal-sentinel-user.json"]


def test_state_updates_capture_user_id_with_snapshot_status(tmp_path: Path) -> None:
    snapshot = Snapshot(
        source_path="capture.json",
        source_sha256="a" * 64,
        session_id="",
        thread_id="",
        user_id="user-1",
        provider="openai",
        operation="responses",
        outcome="success",
    )
    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_snapshot(snapshot)
        assert state.connection.execute(
            "SELECT user_id FROM captures WHERE source_path=?",
            (snapshot.source_path,),
        ).fetchone() == ("user-1",)

        state.put_failure(snapshot.source_path, "b" * 64, "invalid json")
        assert state.connection.execute(
            "SELECT user_id FROM captures WHERE source_path=?",
            (snapshot.source_path,),
        ).fetchone() == ("",)


def test_state_batch_path_reads_are_ordered_and_chunked(tmp_path: Path) -> None:
    snapshots = [
        Snapshot(
            source_path=f"capture-{index:04d}.json",
            source_sha256=f"{index:064x}",
            session_id="session",
            thread_id="thread",
            captured_at=f"2026-09-16T00:{index // 60:02d}:{index % 60:02d}Z",
            provider="openai",
            operation="responses",
            outcome="success",
        )
        for index in range(514)
    ]
    requested_paths = [
        snapshots[-1].source_path,
        "missing.json",
        *(snapshot.source_path for snapshot in reversed(snapshots)),
        snapshots[0].source_path,
    ]

    with StateStore(tmp_path / "state.sqlite") as state:
        with state.write_batch():
            for snapshot in snapshots:
                state.put_snapshot(snapshot)

        restored = list(state.iter_snapshots_for_paths(requested_paths))
        metadata = state.source_metadata_for_paths(requested_paths)

    expected_paths = [snapshot.source_path for snapshot in snapshots]
    assert [snapshot.source_path for snapshot in restored] == expected_paths
    assert list(metadata) == expected_paths
    assert metadata[snapshots[0].source_path] == (
        snapshots[0].source_sha256,
        snapshots[0].captured_at,
    )


def test_state_round_trip_preserves_trajectory_media_mapping(tmp_path: Path) -> None:
    node = enrich_trajectory(
        TrajectoryNode(
            messages=[
                Message(role="user", content="look"),
                Message(role="assistant", content="done", reasoning_content=""),
            ],
            tools=[],
            source="media.json",
            metadata=Metadata(source_file="media.json"),
            multimodal_file_mapping=[
                MediaMapping(part_id="media_0", object_name="stored.png")
            ],
        )
    )
    identifier = trajectory_id(node)
    assert node.normalization_audit is not None

    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_trajectory(
            identifier,
            node,
            representative_key="media.json",
            source_ref="media.json",
            sha256="a" * 64,
            captured_at="2026-09-03T00:00:00Z",
            disposition=node.normalization_audit.tag.value,
            reason_codes=node.normalization_audit.reason_codes,
        )
        restored = list(state.iter_trajectories())

    assert len(restored) == 1
    assert restored[0][0] == identifier
    assert restored[0][1].multimodal_file_mapping == [
        MediaMapping(part_id="media_0", object_name="stored.png")
    ]


def test_failed_reparse_removes_stale_snapshot(tmp_path: Path) -> None:
    snapshot = Snapshot(
        source_path="v1/p/s/a.json",
        source_sha256="a" * 64,
        session_id="s",
        thread_id="t",
        provider="openai",
        operation="responses",
        outcome="success",
    )
    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_snapshot(snapshot)
        state.put_failure(snapshot.source_path, "b" * 64, "invalid json")
        assert state.get_snapshot(snapshot.source_path) is None


def test_skipped_capture_is_resumable_and_removes_stale_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = Snapshot(
        source_path="test/sample.json",
        source_sha256="a" * 64,
        session_id="s",
        thread_id="t",
        provider="openai",
        operation="responses",
        outcome="success",
    )
    with StateStore(tmp_path / "state.sqlite") as state:
        state.begin_scan("first")
        state.put_snapshot(snapshot)
        state.put_skipped(snapshot.source_path, "b" * 64, "test_input")
        assert state.get_snapshot(snapshot.source_path) is None
        assert state.capture_matches(snapshot.source_path, "b" * 64)
        assert list(state.capture_records()) == [
            (snapshot.source_path, "b" * 64, "skipped", "test_input", "", "")
        ]


def test_completed_inventory_removes_deleted_sources(tmp_path: Path) -> None:
    snapshot = Snapshot(
        source_path="gone.json",
        source_sha256="a" * 64,
        session_id="s",
        thread_id="t",
        provider="openai",
        operation="responses",
        outcome="success",
    )
    with StateStore(tmp_path / "state.sqlite") as state:
        state.begin_scan("first")
        state.put_snapshot(snapshot)
        state.finish_scan()
        state.begin_scan("second")
        state.finish_scan()
        assert state.capture_count() == 0
        assert list(state.iter_snapshots()) == []


def test_old_partitioned_state_is_rebuilt_before_resume(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE captures (
                source_path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                endpoint TEXT NOT NULL DEFAULT '',
                captured_at TEXT NOT NULL DEFAULT '',
                scan_id TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE snapshots (
                source_path TEXT PRIMARY KEY,
                source_partition TEXT NOT NULL,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE trajectories (
                trajectory_id TEXT PRIMARY KEY,
                disposition TEXT NOT NULL,
                representative_key TEXT NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE trajectory_origins (
                trajectory_id TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                disposition TEXT NOT NULL,
                reason_codes TEXT NOT NULL,
                PRIMARY KEY(trajectory_id,source_ref,sha256)
            );
            INSERT INTO captures(source_path,sha256,status)
            VALUES('old.json','deadbeef','parsed');
            INSERT INTO snapshots(
                source_path,source_partition,session_id,thread_id,captured_at,payload
            ) VALUES('old.json','old-partition','s','t','','not-a-current-snapshot');
            """
        )

    with StateStore(path) as state:
        columns = tuple(
            row[1] for row in state.connection.execute("PRAGMA table_info(snapshots)")
        )
        version = state.connection.execute("PRAGMA user_version").fetchone()[0]
        assert columns == (
            "source_path",
            "session_id",
            "thread_id",
            "captured_at",
            "payload",
        )
        assert version == 2
        assert state.capture_count() == 0
        assert list(state.iter_snapshots()) == []


def test_jsonl_writer_shards_without_splitting_lines(tmp_path: Path) -> None:
    with JsonlShardWriter(tmp_path, "rows", max_bytes=12) as writer:
        writer.write({"a": 1})
        writer.write({"b": 2})
    assert len(list(tmp_path.glob("rows-*.jsonl"))) == 2


def test_export_writes_manifest_and_lineage(tmp_path: Path) -> None:
    node = enrich_trajectory(
        TrajectoryNode(
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi", reasoning_content=""),
            ],
            tools=[],
            source="a.json",
            metadata=Metadata(source_file="a.json"),
            normalization_audit=NormalizationAudit(tag=AuditTag.PASS),
        )
    )
    output = OutputSet(tmp_path)
    output.stats.input_files = 1
    output.write_trajectory(node, [{"source_ref": "a.json", "sha256": "a" * 64}])
    output.close(input_root="/input", config_hash="config")
    manifest = orjson.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["counts"]["accepted"] == 1
    lineage = next(
        entry["path"]
        for entry in manifest["files"]
        if entry["path"].endswith("/lineage.jsonl")
    )
    assert (tmp_path / lineage).is_file()


def test_export_counts_skipped_inputs_without_emitting_rows(tmp_path: Path) -> None:
    output = OutputSet(tmp_path)
    output.stats.input_files = 2
    output.write_skipped("test_input")
    output.write_skipped("empty_envelope")
    output.close(input_root="/input", config_hash="config", input_format="tokenplan")

    manifest = orjson.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["schema_version"] == "trajfoundry-v3"
    assert manifest["input_format"] == "tokenplan"
    assert manifest["counts"]["skipped_inputs"] == 2
    assert manifest["counts"]["skip_reason_counts"] == {
        "empty_envelope": 1,
        "test_input": 1,
    }
    assert not any("accepted/" in item["path"] for item in manifest["files"])
    assert not any("quarantine/" in item["path"] for item in manifest["files"])


def test_aborted_export_publishes_no_partial_run(tmp_path: Path) -> None:
    node = enrich_trajectory(
        TrajectoryNode(
            messages=[Message(role="assistant", content="done", reasoning_content="")],
            tools=[],
            source="a.json",
            metadata=Metadata(source_file="a.json"),
        )
    )
    output = OutputSet(tmp_path, max_shard_bytes=1)
    output.write_trajectory(node, [{"source_ref": "a.json", "sha256": "a" * 64}])
    output.abort()

    assert not (tmp_path / "manifest.json").exists()
    assert not list(tmp_path.rglob("*.jsonl"))
    assert not list(tmp_path.glob(".staging-*"))


def test_publish_keeps_previous_immutable_generation(tmp_path: Path) -> None:
    node = enrich_trajectory(
        TrajectoryNode(
            messages=[Message(role="assistant", content="done", reasoning_content="")],
            tools=[],
            source="a.json",
            metadata=Metadata(source_file="a.json"),
        )
    )
    first = OutputSet(tmp_path)
    first.stats.input_files = 1
    first.write_trajectory(node, [{"source_ref": "a.json", "sha256": "a" * 64}])
    first.close(input_root="/input", config_hash="config")
    first_manifest = orjson.loads((tmp_path / "manifest.json").read_bytes())
    first_paths = {tmp_path / entry["path"] for entry in first_manifest["files"]}

    second = OutputSet(tmp_path)
    second.stats.input_files = 1
    second.write_trajectory(node, [{"source_ref": "a.json", "sha256": "a" * 64}])
    second.close(input_root="/input", config_hash="config")
    second_manifest = orjson.loads((tmp_path / "manifest.json").read_bytes())

    assert first_manifest["files"] != second_manifest["files"]
    assert all(path.is_file() for path in first_paths)
    assert all(
        entry["path"].startswith("generations/") for entry in second_manifest["files"]
    )
