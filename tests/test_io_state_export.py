import sqlite3
from pathlib import Path

import orjson

from trajfoundry.export import OutputSet
from trajfoundry.io import JsonlShardWriter, load_capture
from trajfoundry.models import (
    AuditTag,
    CompactionRecord,
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
        restored = list(state.iter_snapshots())
    assert restored == [snapshot]


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
        assert version == 1
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
