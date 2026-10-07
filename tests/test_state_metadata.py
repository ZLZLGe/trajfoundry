from pathlib import Path

import pytest

from trajfoundry.models import (
    AuditIssue,
    MediaMapping,
    Message,
    Severity,
    Snapshot,
    ToolDefinition,
)
from trajfoundry.state import StateStore, read_snapshot_metadata


def snapshot(path: str = "one.json") -> Snapshot:
    return Snapshot(
        source_path=path,
        source_sha256="a" * 64,
        session_id="s",
        thread_id="t",
        provider="openai",
        operation="responses",
        outcome="success",
        captured_at="2026-10-07",
        instructions="Keep this instruction",
        model="model",
        harness="agent",
        termination="complete",
        history=[Message(role="user", content="long history" * 1000)],
        response=[Message(role="assistant", content="done", reasoning_content="")],
        tools=[ToolDefinition(name="tool", description="tool", parameters={})],
        multimodal_file_mapping=[MediaMapping(part_id="part", object_name="object")],
        issues=[AuditIssue(code="warning", stage="parse", severity=Severity.WARNING)],
    )


def test_metadata_projection_preserves_fields_and_drops_transcript(
    tmp_path: Path,
) -> None:
    original = snapshot()
    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_snapshot(original)
        state.put_snapshot_metadata(original)
        projected = state.get_snapshot_metadata(original.source_path)
        assert projected is not None
        assert projected.model_dump(
            exclude={"history", "response"}
        ) == original.model_dump(exclude={"history", "response"})
        assert projected.history == projected.response == []
        assert state.get_snapshot(original.source_path) == original
        assert state.snapshot_payload_bytes_for_paths([original.source_path]) == {
            original.source_path: len(
                original.model_dump_json(exclude_none=True).encode()
            )
        }
        assert state.snapshot_metadata_bytes_for_paths([original.source_path]) == {
            original.source_path: len(
                original.model_dump_json(
                    exclude={"history", "response"}, exclude_none=True
                ).encode()
            )
        }


def test_metadata_cache_is_invalidated_by_snapshot_replacement_and_removal(
    tmp_path: Path,
) -> None:
    original = snapshot()
    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_snapshot(original)
        state.put_snapshot_metadata(original)
        changed = original.model_copy(update={"instructions": "changed"})
        state.put_snapshot(changed)
        assert (
            state.connection.execute(
                "SELECT COUNT(*) FROM snapshot_metadata"
            ).fetchone()[0]
            == 0
        )
        projected = list(state.iter_snapshot_metadata_for_paths([original.source_path]))
        assert projected[0].instructions == "changed"
        assert (
            state.connection.execute(
                "SELECT COUNT(*) FROM snapshot_metadata"
            ).fetchone()[0]
            == 1
        )
        state.put_failure(original.source_path, "b" * 64, "failed")
        assert state.get_snapshot_metadata(original.source_path) is None
        assert (
            state.connection.execute(
                "SELECT COUNT(*) FROM snapshot_metadata"
            ).fetchone()[0]
            == 0
        )


def test_metadata_cached_lookup_does_not_decode_full_snapshot(tmp_path: Path) -> None:
    original = snapshot()
    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_snapshot(original)
        state.put_snapshot_metadata(original)
        # Keep the metadata row while deliberately making the full payload bad.
        state.connection.execute("DROP TRIGGER snapshot_metadata_after_update")
        state.connection.execute("UPDATE snapshots SET payload=?", (b"invalid",))
        projected = read_snapshot_metadata(state.connection, original.source_path)
        assert projected is not None and projected.instructions == original.instructions
        assert list(state.iter_snapshot_metadata_for_paths([original.source_path])) == [
            projected
        ]


def test_metadata_fallback_read_only_does_not_write(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "state.sqlite"
    original = snapshot()
    with StateStore(path) as state:
        state.put_snapshot(original)
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as reader:
            projected = read_snapshot_metadata(reader, original.source_path)
            assert projected is not None and projected.history == []
        assert (
            state.connection.execute(
                "SELECT COUNT(*) FROM snapshot_metadata"
            ).fetchone()[0]
            == 0
        )


def test_unknown_payload_size_is_explicit_not_compressed_length(tmp_path: Path) -> None:
    import zlib

    original = snapshot()
    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_snapshot(original)
        state.connection.execute(
            "UPDATE snapshots SET payload=?",
            (zlib.compress(original.model_dump_json().encode()),),
        )
        assert state.snapshot_payload_bytes_for_paths([original.source_path]) == {
            original.source_path: None
        }


def test_ephemeral_disk_journal_retains_transaction_rollback(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.sqlite", ephemeral=True) as state:
        with pytest.raises(RuntimeError), state.write_batch():
            state.put_snapshot(snapshot())
            state.put_snapshot_metadata(snapshot())
            raise RuntimeError("rollback")
        assert list(state.iter_snapshots()) == []
        assert (
            state.connection.execute(
                "SELECT COUNT(*) FROM snapshot_metadata"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("ephemeral", [False, True])
def test_prepare_concurrent_reads_allows_reader_during_parent_write(
    tmp_path: Path,
    ephemeral: bool,
) -> None:
    import sqlite3

    path = tmp_path / "state.sqlite"
    original = snapshot()
    with StateStore(path, ephemeral=ephemeral) as state:
        state.put_snapshot(original)
        state.put_snapshot_metadata(original)
        state.set_meta("visible", "committed")
        synchronous = state.connection.execute("PRAGMA synchronous").fetchone()[0]
        state.prepare_concurrent_build_reads()
        assert state.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert state.connection.execute("PRAGMA locking_mode").fetchone()[0] == "normal"
        assert (
            state.connection.execute("PRAGMA synchronous").fetchone()[0] == synchronous
        )
        assert state.connection.execute("PRAGMA temp_store").fetchone()[0] == 1
        assert state.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        # No immutable=1: a real read-only connection sees committed changes
        # and can read input while the coordinator has a live write transaction.
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.2) as reader:
            with state.write_batch():
                state.set_meta("visible", "new")
                projected = read_snapshot_metadata(reader, original.source_path)
                assert projected is not None and projected.history == []
                assert (
                    reader.execute(
                        "SELECT value FROM meta WHERE key='visible'"
                    ).fetchone()[0]
                    == "committed"
                )
                # Check OS-level lock behavior using an independent process,
                # not only SQLite's in-process connection coordination.
                import subprocess
                import sys

                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import sqlite3,sys; "
                            "db=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True,timeout=.2); "
                            "assert db.execute(\"SELECT value FROM meta WHERE key='visible'\").fetchone()[0]=='committed'; "
                            "assert db.execute('SELECT COUNT(*) FROM snapshot_metadata').fetchone()[0]==1; "
                            "db.close()"
                        ),
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                assert result.returncode == 0, result.stderr
            assert (
                reader.execute("SELECT value FROM meta WHERE key='visible'").fetchone()[
                    0
                ]
                == "new"
            )
            # A long read snapshot must not block parent commits either.
            reader.execute("BEGIN")
            reader.execute("SELECT value FROM meta WHERE key='visible'").fetchall()
            state.set_meta("visible", "last")
            assert (
                reader.execute("SELECT value FROM meta WHERE key='visible'").fetchone()[
                    0
                ]
                == "new"
            )
            reader.commit()
            assert (
                reader.execute("SELECT value FROM meta WHERE key='visible'").fetchone()[
                    0
                ]
                == "last"
            )
        state.prepare_concurrent_build_reads()  # idempotent


def test_prepare_concurrent_reads_does_not_commit_an_active_batch(
    tmp_path: Path,
) -> None:
    with StateStore(tmp_path / "state.sqlite", ephemeral=True) as state:
        with pytest.raises(RuntimeError, match="write_batch"), state.write_batch():
            state.put_snapshot(snapshot())
            state.prepare_concurrent_build_reads()
        assert state.get_snapshot("one.json") is None


def test_prepare_concurrent_reads_leaves_exclusive_wal_before_normal(
    tmp_path: Path,
) -> None:
    with StateStore(tmp_path / "state.sqlite", ephemeral=True) as state:
        state.put_snapshot(snapshot())
        state.connection.execute("PRAGMA journal_mode=WAL").fetchall()
        state.connection.execute("SELECT source_path FROM snapshots").fetchall()
        assert (
            state.connection.execute("PRAGMA locking_mode").fetchone()[0] == "exclusive"
        )
        state.prepare_concurrent_build_reads()
        assert state.connection.execute("PRAGMA locking_mode").fetchone()[0] == "normal"
        assert state.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
