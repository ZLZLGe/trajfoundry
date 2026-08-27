"""Durable, resumable normalization state."""

from __future__ import annotations

import json
import sqlite3
import zlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Self

import orjson

from .canonical import trajectory_id as compute_trajectory_id
from .models import Snapshot, TrajectoryNode
from .output_contract import parse_trajectory_record, project_trajectory
from .quality import validate_derived_fields


class StateStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS captures (
                source_path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                endpoint TEXT NOT NULL DEFAULT '',
                captured_at TEXT NOT NULL DEFAULT '',
                scan_id TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                source_path TEXT PRIMARY KEY,
                source_partition TEXT NOT NULL,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_group
            ON snapshots(source_partition, session_id, thread_id, captured_at, source_path);
            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                disposition TEXT NOT NULL,
                representative_key TEXT NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trajectory_origins (
                trajectory_id TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                disposition TEXT NOT NULL,
                reason_codes TEXT NOT NULL,
                PRIMARY KEY(trajectory_id,source_ref,sha256)
            );
            """
        )
        self._ensure_column("captures", "endpoint", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("captures", "captured_at", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("captures", "scan_id", "TEXT NOT NULL DEFAULT ''")
        self._scan_id: str | None = None

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        columns = {
            row[1] for row in self.connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            self.connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    def begin_scan(self, scan_id: str) -> None:
        if not scan_id:
            raise ValueError("scan_id must not be empty")
        self._scan_id = scan_id

    def finish_scan(self) -> None:
        """Forget sources removed since the previous completed input scan."""

        if self._scan_id is None:
            return
        with self.connection:
            self.connection.execute(
                "DELETE FROM snapshots WHERE source_path IN "
                "(SELECT source_path FROM captures WHERE scan_id != ?)",
                (self._scan_id,),
            )
            self.connection.execute(
                "DELETE FROM captures WHERE scan_id != ?", (self._scan_id,)
            )
        self._scan_id = None

    def reset_ingest(self) -> None:
        """Drop cached inputs when normalization semantics have changed."""

        with self.connection:
            self.connection.execute("DELETE FROM trajectory_origins")
            self.connection.execute("DELETE FROM trajectories")
            self.connection.execute("DELETE FROM snapshots")
            self.connection.execute("DELETE FROM captures")

    def close(self) -> None:
        self.connection.close()

    def set_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.connection.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None

    def capture_matches(self, source_path: str, sha256: str) -> bool:
        row = self.connection.execute(
            "SELECT sha256, status FROM captures WHERE source_path=?", (source_path,)
        ).fetchone()
        matched = bool(row and row[0] == sha256 and row[1] == "parsed")
        if matched and self._scan_id is not None:
            self.connection.execute(
                "UPDATE captures SET scan_id=? WHERE source_path=?",
                (self._scan_id, source_path),
            )
        return matched

    def put_snapshot(self, snapshot: Snapshot, *, endpoint: str = "") -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO snapshots(
                    source_path,source_partition,session_id,thread_id,captured_at,payload
                ) VALUES(?,?,?,?,?,?)
                ON CONFLICT(source_path) DO UPDATE SET
                    source_partition=excluded.source_partition,
                    session_id=excluded.session_id,
                    thread_id=excluded.thread_id,
                    captured_at=excluded.captured_at,
                    payload=excluded.payload
                """,
                (
                    snapshot.source_path,
                    snapshot.source_partition,
                    snapshot.session_id,
                    snapshot.thread_id,
                    snapshot.captured_at,
                    zlib.compress(
                        snapshot.model_dump_json(exclude_none=True).encode("utf-8"),
                        level=3,
                    ),
                ),
            )
            self.connection.execute(
                """
                INSERT INTO captures(
                    source_path,sha256,status,reason,endpoint,captured_at,scan_id
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(source_path) DO UPDATE SET
                    sha256=excluded.sha256,status=excluded.status,reason=excluded.reason,
                    endpoint=excluded.endpoint,captured_at=excluded.captured_at,
                    scan_id=excluded.scan_id
                """,
                (
                    snapshot.source_path,
                    snapshot.source_sha256,
                    "parsed",
                    "",
                    endpoint,
                    snapshot.captured_at,
                    self._scan_id or "",
                ),
            )

    def put_failure(
        self,
        source_path: str,
        sha256: str,
        reason: str,
        *,
        endpoint: str = "",
        captured_at: str = "",
    ) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM snapshots WHERE source_path=?", (source_path,)
            )
            self.connection.execute(
                """
                INSERT INTO captures(
                    source_path,sha256,status,reason,endpoint,captured_at,scan_id
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(source_path) DO UPDATE SET
                    sha256=excluded.sha256,status=excluded.status,reason=excluded.reason,
                    endpoint=excluded.endpoint,captured_at=excluded.captured_at,
                    scan_id=excluded.scan_id
                """,
                (
                    source_path,
                    sha256,
                    "failed",
                    reason,
                    endpoint,
                    captured_at,
                    self._scan_id or "",
                ),
            )

    def groups(self) -> Iterator[tuple[str, str, str]]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT source_partition,session_id,thread_id FROM snapshots
            ORDER BY source_partition,session_id,thread_id
            """
        )
        yield from rows

    def sessions(self) -> Iterator[tuple[str, str]]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT source_partition,session_id FROM snapshots
            ORDER BY source_partition,session_id
            """
        )
        yield from rows

    @staticmethod
    def _decode_snapshot(payload: str | bytes) -> Snapshot:
        if isinstance(payload, str):
            return Snapshot.model_validate_json(payload)
        return Snapshot.model_validate_json(zlib.decompress(payload))

    def snapshots_for_session(
        self, source_partition: str, session_id: str
    ) -> Iterator[Snapshot]:
        rows = self.connection.execute(
            """
            SELECT payload FROM snapshots
            WHERE source_partition=? AND session_id=?
            ORDER BY thread_id,captured_at,source_path
            """,
            (source_partition, session_id),
        )
        for (payload,) in rows:
            yield self._decode_snapshot(payload)

    def threads_for_session(
        self, source_partition: str, session_id: str
    ) -> Iterator[str]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT thread_id FROM snapshots
            WHERE source_partition=? AND session_id=? AND thread_id != ''
            ORDER BY thread_id
            """,
            (source_partition, session_id),
        )
        for (thread_id,) in rows:
            yield thread_id

    def snapshots_for_thread(
        self, source_partition: str, session_id: str, thread_id: str
    ) -> Iterator[Snapshot]:
        rows = self.connection.execute(
            """
            SELECT payload FROM snapshots
            WHERE source_partition=? AND session_id=? AND thread_id=?
            ORDER BY captured_at,source_path
            """,
            (source_partition, session_id, thread_id),
        )
        for (payload,) in rows:
            yield self._decode_snapshot(payload)

    def get_snapshot(self, source_path: str) -> Snapshot | None:
        row = self.connection.execute(
            "SELECT payload FROM snapshots WHERE source_path=?", (source_path,)
        ).fetchone()
        return self._decode_snapshot(row[0]) if row else None

    def source_metadata(self, source_path: str) -> tuple[str, str] | None:
        """Return provenance fields without inflating the snapshot payload."""

        row = self.connection.execute(
            """
            SELECT sha256,captured_at FROM captures
            WHERE source_path=? AND status='parsed'
            """,
            (source_path,),
        ).fetchone()
        return (str(row[0]), str(row[1])) if row else None

    def failures(self) -> Iterator[tuple[str, str, str]]:
        rows = self.connection.execute(
            "SELECT source_path,sha256,reason FROM captures WHERE status='failed' ORDER BY source_path"
        )
        yield from rows

    def capture_records(
        self,
    ) -> Iterator[tuple[str, str, str, str, str, str]]:
        rows = self.connection.execute(
            """
            SELECT source_path,sha256,status,reason,endpoint,captured_at
            FROM captures ORDER BY source_path
            """
        )
        yield from rows

    def capture_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM captures").fetchone()
        return int(row[0]) if row else 0

    def trajectory_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM trajectories").fetchone()
        return int(row[0]) if row else 0

    def clear_trajectories(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM trajectory_origins")
            self.connection.execute("DELETE FROM trajectories")

    def put_trajectory(
        self,
        trajectory_id: str,
        node: TrajectoryNode,
        *,
        representative_key: str,
        source_ref: str,
        sha256: str,
        captured_at: str,
        disposition: str,
        reason_codes: list[str],
    ) -> None:
        self.put_trajectory_with_origins(
            trajectory_id,
            node,
            representative_key=representative_key,
            origins=(
                {
                    "source_ref": source_ref,
                    "sha256": sha256,
                    "captured_at": captured_at,
                },
            ),
            disposition=disposition,
            reason_codes=reason_codes,
        )

    def put_trajectory_with_origins(
        self,
        trajectory_id: str,
        node: TrajectoryNode,
        *,
        representative_key: str,
        origins: Sequence[Mapping[str, str]],
        disposition: str,
        reason_codes: list[str],
    ) -> None:
        """Validate and persist one trajectory with all provenance rows."""

        if not origins:
            raise ValueError("trajectory must have at least one origin")
        projected = project_trajectory(node)
        validate_derived_fields(node)
        audit = node.normalization_audit
        if audit is None:  # guarded by validate_derived_fields
            raise ValueError("trajectory has no normalization audit")
        if disposition != audit.tag.value:
            raise ValueError(
                "trajectory disposition does not match normalization audit"
            )
        normalized_reasons = sorted(set(reason_codes))
        if normalized_reasons != audit.reason_codes:
            raise ValueError("trajectory reason_codes do not match normalization audit")
        expected_id = compute_trajectory_id(node)
        if trajectory_id != expected_id:
            raise ValueError("trajectory_id does not match trajectory content")
        rank = {"pass": 0, "quarantined": 1, "excluded": 2}
        node_json = orjson.dumps(projected, option=orjson.OPT_SORT_KEYS)
        existing = self.connection.execute(
            "SELECT disposition,representative_key FROM trajectories WHERE trajectory_id=?",
            (trajectory_id,),
        ).fetchone()
        replace_payload = existing is None or (
            rank.get(disposition, 9),
            representative_key,
        ) < (rank.get(existing[0], 9), existing[1])
        with self.connection:
            if existing is None:
                self.connection.execute(
                    "INSERT INTO trajectories VALUES(?,?,?,?)",
                    (
                        trajectory_id,
                        disposition,
                        representative_key,
                        zlib.compress(node_json, level=3),
                    ),
                )
            elif replace_payload:
                self.connection.execute(
                    """
                    UPDATE trajectories
                    SET disposition=?,representative_key=?,payload=?
                    WHERE trajectory_id=?
                    """,
                    (
                        disposition,
                        representative_key,
                        zlib.compress(node_json, level=3),
                        trajectory_id,
                    ),
                )
            self.connection.executemany(
                """
                INSERT OR IGNORE INTO trajectory_origins(
                    trajectory_id,source_ref,sha256,captured_at,disposition,reason_codes
                ) VALUES(?,?,?,?,?,?)
                """,
                [
                    (
                        trajectory_id,
                        origin["source_ref"],
                        origin["sha256"],
                        origin["captured_at"],
                        disposition,
                        json.dumps(normalized_reasons, separators=(",", ":")),
                    )
                    for origin in origins
                ],
            )

    def iter_trajectories(
        self,
    ) -> Iterator[tuple[str, TrajectoryNode, list[dict[str, object]]]]:
        rows = self.connection.execute(
            "SELECT trajectory_id,payload FROM trajectories ORDER BY trajectory_id"
        )
        for identifier, payload in rows:
            node = parse_trajectory_record(orjson.loads(zlib.decompress(payload)))
            validate_derived_fields(node)
            if identifier != compute_trajectory_id(node):
                raise ValueError(
                    "stored trajectory_id does not match trajectory content"
                )
            origin_rows = self.connection.execute(
                """
                SELECT source_ref,sha256,captured_at,disposition,reason_codes
                FROM trajectory_origins WHERE trajectory_id=?
                ORDER BY captured_at,source_ref,sha256
                """,
                (identifier,),
            )
            origins = [
                {
                    "source_ref": source_ref,
                    "sha256": sha256,
                    "captured_at": captured_at,
                    "disposition": disposition,
                    "reason_codes": json.loads(reason_codes),
                }
                for source_ref, sha256, captured_at, disposition, reason_codes in origin_rows
            ]
            yield identifier, node, origins

    def iter_snapshots(self) -> Iterator[Snapshot]:
        rows = self.connection.execute(
            "SELECT payload FROM snapshots ORDER BY source_partition,session_id,thread_id,captured_at,source_path"
        )
        for (payload,) in rows:
            yield self._decode_snapshot(payload)

    def commit(self) -> None:
        self.connection.commit()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
