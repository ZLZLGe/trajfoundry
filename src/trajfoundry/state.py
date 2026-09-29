"""Durable, resumable normalization state."""

from __future__ import annotations

import json
import sqlite3
import zlib
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Self, TypeAlias

import zstandard

from .canonical import trajectory_id as compute_trajectory_id
from .json_codec import dumps, loads
from .models import Snapshot, TrajectoryNode
from .output_contract import parse_trajectory_record, project_trajectory
from .quality import validate_derived_fields

STATE_SCHEMA_VERSION = 2
AggregationScope: TypeAlias = tuple[Literal["session", "user"], str]
_PAGE_SIZE = 32 * 1024
_CACHE_SIZE_KIB = 128 * 1024
_MMAP_SIZE = 256 * 1024 * 1024
_PATH_QUERY_BATCH_SIZE = 512
_TRAJECTORY_UPDATE_BATCH_SIZE = 512
_PAYLOAD_HEADER = b"TFZ1"
_ZSTD_COMPRESSOR = zstandard.ZstdCompressor(level=1)
_ZSTD_DECOMPRESSOR = zstandard.ZstdDecompressor()
_TRAJECTORY_ASSIGN_SKIP_REASON = "trajectory_materialization_failed"
_SNAPSHOT_COLUMNS = (
    "source_path",
    "session_id",
    "thread_id",
    "captured_at",
    "payload",
)


def _compress_payload(payload: bytes) -> bytes:
    return _PAYLOAD_HEADER + _ZSTD_COMPRESSOR.compress(payload)


def _decompress_payload(payload: str | bytes) -> bytes:
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if payload.startswith(_PAYLOAD_HEADER):
        return _ZSTD_DECOMPRESSOR.decompress(payload[len(_PAYLOAD_HEADER) :])
    # Accept unenveloped zlib rows defensively in an otherwise compatible v2
    # database; replacing either row rewrites it with the TFZ1 envelope.
    return zlib.decompress(payload)


def _sorted_path_batches(paths: Iterable[str]) -> Iterator[list[str]]:
    ordered_paths = sorted(set(paths))
    for offset in range(0, len(ordered_paths), _PATH_QUERY_BATCH_SIZE):
        yield ordered_paths[offset : offset + _PATH_QUERY_BATCH_SIZE]


def _has_synthesized_session(node: TrajectoryNode) -> bool:
    if not node.metadata.session_id:
        return True
    audit = node.normalization_audit
    return bool(
        audit
        and any(
            issue.code == "metadata_session_id_synthesized" for issue in audit.issues
        )
    )


def _set_sub_session_id(node: TrajectoryNode, value: int) -> None:
    """Assign one output-branch number to a root and all mounted children."""

    node.metadata.sub_session_id = value
    for child in (node.sub_agent_trajectory or {}).values():
        _set_sub_session_id(child, value)


class StateStore:
    def __init__(self, path: Path, *, ephemeral: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self._ephemeral = ephemeral
        self._write_batch_depth = 0
        if int(self.connection.execute("PRAGMA page_count").fetchone()[0]) == 0:
            self.connection.execute(f"PRAGMA page_size={_PAGE_SIZE}")
        if ephemeral:
            self.connection.execute("PRAGMA journal_mode=MEMORY")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        else:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(f"PRAGMA cache_size=-{_CACHE_SIZE_KIB}")
        self.connection.execute(f"PRAGMA mmap_size={_MMAP_SIZE}")
        stored_version = int(
            self.connection.execute("PRAGMA user_version").fetchone()[0]
        )
        if stored_version > STATE_SCHEMA_VERSION:
            self.connection.close()
            raise RuntimeError(
                "state database schema is newer than this TrajFoundry version: "
                f"{stored_version} > {STATE_SCHEMA_VERSION}"
            )
        existing_tables = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        snapshot_columns = self._table_columns("snapshots")
        had_cached_state = bool(
            existing_tables
            & {"captures", "snapshots", "trajectories", "trajectory_origins"}
        )
        # v2 adds the capture-level user id used to partition sessionless
        # aggregation. A v1 cache cannot be upgraded by merely adding the
        # column: every retained row would receive the empty default and users
        # could then be merged together. Rebuild it from source instead.
        incompatible_version = stored_version != STATE_SCHEMA_VERSION
        if had_cached_state and (
            incompatible_version or snapshot_columns != _SNAPSHOT_COLUMNS
        ):
            self._rebuild_cached_state(existing_tables)
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
                scan_id TEXT NOT NULL DEFAULT '',
                user_id TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                source_path TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_group
            ON snapshots(session_id, thread_id, captured_at, source_path);
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
        self._ensure_column("captures", "user_id", "TEXT NOT NULL DEFAULT ''")
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_captures_user "
            "ON captures(user_id,source_path)"
        )
        self.connection.execute(f"PRAGMA user_version={STATE_SCHEMA_VERSION}")
        self.connection.commit()
        self._scan_id: str | None = None
        # Rows written through put_trajectory_with_origins have already passed
        # canonical projection, derived-field validation, and trajectory-id
        # verification. Keep that fact for the current process so sub-session
        # assignment does not repeat the expensive quality pass. Rows loaded
        # from a reopened state database are intentionally not trusted.
        self._trusted_trajectory_ids: set[str] = set()

    def _table_columns(self, table: str) -> tuple[str, ...]:
        return tuple(
            str(row[1])
            for row in self.connection.execute(f"PRAGMA table_info({table})")
        )

    def _rebuild_cached_state(self, existing_tables: set[str]) -> None:
        """Discard caches whose payload/schema cannot satisfy the current model.

        ``captures`` must be cleared together with ``snapshots``.  Otherwise a
        resumed scan would trust the old ``parsed`` marker and never recreate
        the snapshot that was removed during the schema transition.
        """

        with self.connection:
            self.connection.execute("DROP INDEX IF EXISTS idx_snapshots_group")
            self.connection.execute("DROP TABLE IF EXISTS snapshots")
            for table in ("trajectory_origins", "trajectories", "captures"):
                if table in existing_tables:
                    self.connection.execute(f"DELETE FROM {table}")

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

    @contextmanager
    def write_batch(self) -> Iterator[Self]:
        """Commit a group of existing write operations as one transaction.

        Nested batches use savepoints. Existing ``put_*`` methods remain
        independently atomic when called outside this context manager.
        """

        savepoint = f"trajfoundry_batch_{self._write_batch_depth}"
        self.connection.execute(f"SAVEPOINT {savepoint}")
        self._write_batch_depth += 1
        try:
            yield self
        except BaseException:
            self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        finally:
            self._write_batch_depth -= 1

    @contextmanager
    def _write_scope(self) -> Iterator[None]:
        if self._write_batch_depth:
            yield
            return
        with self.connection:
            yield

    def finish_scan(self) -> None:
        """Forget sources removed since the previous completed input scan."""

        if self._scan_id is None:
            return
        with self._write_scope():
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

        with self._write_scope():
            self.connection.execute("DELETE FROM trajectory_origins")
            self.connection.execute("DELETE FROM trajectories")
            self.connection.execute("DELETE FROM snapshots")
            self.connection.execute("DELETE FROM captures")

    @contextmanager
    def ingest_write_mode(self) -> Iterator[Self]:
        """Suspend aggregation indexes while loading an ephemeral input.

        Remote jobs rebuild their temporary database from scratch, so the
        grouping indexes cannot serve reads during ingest.  Deferring their
        maintenance until the single writer has finished reduces SQLite work;
        durable local/resumable databases keep the historical behavior.
        """

        if not self._ephemeral:
            yield self
            return
        with self._write_scope():
            self.connection.execute("DROP INDEX IF EXISTS idx_snapshots_group")
            self.connection.execute("DROP INDEX IF EXISTS idx_captures_user")
        try:
            yield self
        finally:
            with self._write_scope():
                self.connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_snapshots_group "
                    "ON snapshots(session_id, thread_id, captured_at, source_path)"
                )
                self.connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_captures_user "
                    "ON captures(user_id, source_path)"
                )

    def close(self) -> None:
        self.connection.close()

    def set_meta(self, key: str, value: str) -> None:
        with self._write_scope():
            self.connection.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None

    def capture_matches(self, source_path: str, sha256: str) -> bool:
        row = self.connection.execute(
            "SELECT sha256, status FROM captures WHERE source_path=?", (source_path,)
        ).fetchone()
        # A skipped input is a deliberate, terminal ingest decision just like a
        # parsed snapshot.  It must participate in a resumed scan so that a
        # stable non-trajectory file is not repeatedly reclassified.
        matched = bool(row and row[0] == sha256 and row[1] in {"parsed", "skipped"})
        if matched and self._scan_id is not None:
            self.connection.execute(
                "UPDATE captures SET scan_id=? WHERE source_path=?",
                (self._scan_id, source_path),
            )
        return matched

    def put_snapshot(self, snapshot: Snapshot, *, endpoint: str = "") -> None:
        self.put_snapshot_payload(
            source_path=snapshot.source_path,
            source_sha256=snapshot.source_sha256,
            session_id=snapshot.session_id,
            thread_id=snapshot.thread_id,
            captured_at=snapshot.captured_at,
            user_id=snapshot.user_id,
            payload=_compress_payload(
                snapshot.model_dump_json(exclude_none=True).encode("utf-8")
            ),
            endpoint=endpoint,
        )

    def put_snapshot_payload(
        self,
        *,
        source_path: str,
        source_sha256: str,
        session_id: str,
        thread_id: str,
        captured_at: str,
        user_id: str,
        payload: bytes,
        endpoint: str = "",
    ) -> None:
        """Store an already serialized snapshot from a parse worker."""

        with self._write_scope():
            self.connection.execute(
                """
                INSERT INTO snapshots(
                    source_path,session_id,thread_id,captured_at,payload
                ) VALUES(?,?,?,?,?)
                ON CONFLICT(source_path) DO UPDATE SET
                    session_id=excluded.session_id,
                    thread_id=excluded.thread_id,
                    captured_at=excluded.captured_at,
                    payload=excluded.payload
                """,
                (
                    source_path,
                    session_id,
                    thread_id,
                    captured_at,
                    payload,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO captures(
                    source_path,sha256,status,reason,endpoint,captured_at,
                    scan_id,user_id
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(source_path) DO UPDATE SET
                    sha256=excluded.sha256,
                    status=excluded.status,
                    reason=excluded.reason,
                    endpoint=excluded.endpoint,captured_at=excluded.captured_at,
                    scan_id=excluded.scan_id,user_id=excluded.user_id
                """,
                (
                    source_path,
                    source_sha256,
                    "parsed",
                    "",
                    endpoint,
                    captured_at,
                    self._scan_id or "",
                    user_id,
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
        with self._write_scope():
            self.connection.execute(
                "DELETE FROM snapshots WHERE source_path=?", (source_path,)
            )
            self.connection.execute(
                """
                INSERT INTO captures(
                    source_path,sha256,status,reason,endpoint,captured_at,
                    scan_id,user_id
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(source_path) DO UPDATE SET
                    sha256=excluded.sha256,
                    status=excluded.status,
                    reason=excluded.reason,
                    endpoint=excluded.endpoint,captured_at=excluded.captured_at,
                    scan_id=excluded.scan_id,user_id=excluded.user_id
                """,
                (
                    source_path,
                    sha256,
                    "failed",
                    reason,
                    endpoint,
                    captured_at,
                    self._scan_id or "",
                    "",
                ),
            )

    def put_skipped(
        self,
        source_path: str,
        sha256: str,
        reason: str,
        *,
        endpoint: str = "",
        captured_at: str = "",
    ) -> None:
        """Persist a non-exported input and its stable skip reason."""

        if not reason:
            raise ValueError("skipped capture reason must not be empty")
        with self._write_scope():
            self.connection.execute(
                "DELETE FROM snapshots WHERE source_path=?", (source_path,)
            )
            self.connection.execute(
                """
                INSERT INTO captures(
                    source_path,sha256,status,reason,endpoint,captured_at,
                    scan_id,user_id
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(source_path) DO UPDATE SET
                    sha256=excluded.sha256,
                    status=excluded.status,
                    reason=excluded.reason,
                    endpoint=excluded.endpoint,captured_at=excluded.captured_at,
                    scan_id=excluded.scan_id,user_id=excluded.user_id
                """,
                (
                    source_path,
                    sha256,
                    "skipped",
                    reason,
                    endpoint,
                    captured_at,
                    self._scan_id or "",
                    "",
                ),
            )

    def groups(self) -> Iterator[tuple[str, str]]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT session_id,thread_id FROM snapshots
            ORDER BY session_id,thread_id
            """
        )
        yield from rows

    def aggregation_scopes(self) -> Iterator[AggregationScope]:
        """Yield each prefix-aggregation scope in deterministic order."""

        rows = self.connection.execute(
            """
            SELECT scope_kind,scope_id FROM (
                SELECT
                    'session' AS scope_kind,
                    snapshots.session_id AS scope_id
                FROM snapshots
                WHERE snapshots.session_id != ''
                GROUP BY snapshots.session_id
                UNION ALL
                SELECT
                    'user' AS scope_kind,
                    captures.user_id AS scope_id
                FROM captures
                JOIN snapshots USING(source_path)
                WHERE snapshots.session_id = ''
                GROUP BY captures.user_id
            )
            ORDER BY scope_kind,scope_id
            """
        )
        for scope_kind, scope_id in rows:
            yield (scope_kind, str(scope_id))

    def snapshots_for_aggregation_scope(
        self, scope: AggregationScope
    ) -> Iterator[Snapshot]:
        """Stream snapshots belonging to one aggregation scope."""

        scope_kind, scope_id = scope
        if scope_kind == "session":
            if not scope_id:
                raise ValueError("session aggregation scope requires a real session id")
            rows = self.connection.execute(
                """
                SELECT payload FROM snapshots
                WHERE session_id=?
                ORDER BY thread_id,captured_at,source_path
                """,
                (scope_id,),
            )
        elif scope_kind == "user":
            rows = self.connection.execute(
                """
                SELECT snapshots.payload FROM captures
                JOIN snapshots USING(source_path)
                WHERE captures.user_id=?
                AND snapshots.session_id=''
                ORDER BY snapshots.captured_at,snapshots.source_path
                """,
                (scope_id,),
            )
        else:
            raise ValueError(f"unknown aggregation scope kind: {scope_kind!r}")
        for (payload,) in rows:
            yield self._decode_snapshot(payload)

    def iter_snapshot_groups(
        self, *, include_empty_threads: bool = False
    ) -> Iterator[tuple[str, str, tuple[Snapshot, ...]]]:
        """Yield ordered snapshot groups from one database cursor."""

        where = "" if include_empty_threads else "WHERE thread_id != ''"
        rows = self.connection.execute(
            f"""
            SELECT session_id,thread_id,payload FROM snapshots
            {where}
            ORDER BY session_id,thread_id,captured_at,source_path
            """
        )
        current_key: tuple[str, str] | None = None
        current_snapshots: list[Snapshot] = []
        for session_id, thread_id, payload in rows:
            key = (str(session_id), str(thread_id))
            if current_key is not None and key != current_key:
                yield (*current_key, tuple(current_snapshots))
                current_snapshots = []
            current_key = key
            current_snapshots.append(self._decode_snapshot(payload))
        if current_key is not None:
            yield (*current_key, tuple(current_snapshots))

    def sessions(self) -> Iterator[str]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT session_id FROM snapshots
            ORDER BY session_id
            """
        )
        for (session_id,) in rows:
            yield session_id

    @staticmethod
    def _decode_snapshot(payload: str | bytes) -> Snapshot:
        return Snapshot.model_validate_json(_decompress_payload(payload))

    def snapshots_for_session(self, session_id: str) -> Iterator[Snapshot]:
        rows = self.connection.execute(
            """
            SELECT payload FROM snapshots
            WHERE session_id=?
            ORDER BY thread_id,captured_at,source_path
            """,
            (session_id,),
        )
        for (payload,) in rows:
            yield self._decode_snapshot(payload)

    def threads_for_session(self, session_id: str) -> Iterator[str]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT thread_id FROM snapshots
            WHERE session_id=? AND thread_id != ''
            ORDER BY thread_id
            """,
            (session_id,),
        )
        for (thread_id,) in rows:
            yield thread_id

    def snapshots_for_thread(
        self, session_id: str, thread_id: str
    ) -> Iterator[Snapshot]:
        rows = self.connection.execute(
            """
            SELECT payload FROM snapshots
            WHERE session_id=? AND thread_id=?
            ORDER BY captured_at,source_path
            """,
            (session_id, thread_id),
        )
        for (payload,) in rows:
            yield self._decode_snapshot(payload)

    def get_snapshot(self, source_path: str) -> Snapshot | None:
        row = self.connection.execute(
            "SELECT payload FROM snapshots WHERE source_path=?", (source_path,)
        ).fetchone()
        return self._decode_snapshot(row[0]) if row else None

    def iter_snapshots_for_paths(self, paths: Iterable[str]) -> Iterator[Snapshot]:
        """Stream existing snapshots for paths in stable source-path order."""

        for batch in _sorted_path_batches(paths):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT payload FROM snapshots "
                f"WHERE source_path IN ({placeholders}) ORDER BY source_path",
                batch,
            )
            for (payload,) in rows:
                yield self._decode_snapshot(payload)

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

    def source_metadata_for_paths(
        self, paths: Iterable[str]
    ) -> dict[str, tuple[str, str]]:
        """Return parsed provenance keyed in stable source-path order."""

        result: dict[str, tuple[str, str]] = {}
        for batch in _sorted_path_batches(paths):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT source_path,sha256,captured_at FROM captures "
                f"WHERE status='parsed' AND source_path IN ({placeholders}) "
                f"ORDER BY source_path",
                batch,
            )
            for source_path, sha256, captured_at in rows:
                result[str(source_path)] = (str(sha256), str(captured_at))
        return result

    def failures(self) -> Iterator[tuple[str, str, str]]:
        rows = self.connection.execute(
            "SELECT source_path,sha256,reason FROM captures "
            "WHERE status='failed' ORDER BY source_path"
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
        self._trusted_trajectory_ids.clear()
        with self._write_scope():
            self.connection.execute("DELETE FROM trajectory_origins")
            self.connection.execute("DELETE FROM trajectories")

    def _discard_trajectories(
        self,
        identifiers: Iterable[str],
        *,
        reason: str = _TRAJECTORY_ASSIGN_SKIP_REASON,
    ) -> int:
        """Remove malformed final rows and skip their unrepresented inputs.

        A trajectory row is normally accompanied by one or more origin rows.
        Removing only the trajectory would leave those captures marked as
        parsed without a lineage row, which makes the published manifest
        unverifiable.  Delete the bad rows first, then mark an origin skipped
        only when no surviving trajectory still represents that source.
        """

        unique_ids = sorted({str(identifier) for identifier in identifiers})
        if not unique_ids:
            return 0
        placeholders = ",".join("?" for _ in unique_ids)
        skipped_sources = 0
        with self.write_batch():
            origin_rows = list(
                self.connection.execute(
                    "SELECT source_ref FROM trajectory_origins "
                    f"WHERE trajectory_id IN ({placeholders})",
                    unique_ids,
                )
            )
            failed_sources = {str(row[0]) for row in origin_rows}
            self.connection.execute(
                "DELETE FROM trajectory_origins "
                f"WHERE trajectory_id IN ({placeholders})",
                unique_ids,
            )
            self.connection.execute(
                "DELETE FROM trajectories "
                f"WHERE trajectory_id IN ({placeholders})",
                unique_ids,
            )
            if not failed_sources:
                return skipped_sources

            source_placeholders = ",".join("?" for _ in failed_sources)
            represented_sources = {
                str(row[0])
                for row in self.connection.execute(
                    "SELECT DISTINCT source_ref FROM trajectory_origins "
                    f"WHERE source_ref IN ({source_placeholders})",
                    sorted(failed_sources),
                )
            }
            for source_path in sorted(failed_sources - represented_sources):
                metadata = self.source_metadata(source_path)
                if metadata is None:
                    continue
                sha256, captured_at = metadata
                self.put_skipped(
                    source_path,
                    sha256,
                    reason,
                    captured_at=captured_at,
                )
                skipped_sources += 1
        return skipped_sources

    @staticmethod
    def _trusted_session_metadata(value: object) -> tuple[str, bool]:
        """Read assignment metadata from an already canonical trajectory dict."""

        if not isinstance(value, Mapping):
            raise TypeError("trajectory payload is not an object")
        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("trajectory payload has no metadata object")
        session_id = metadata.get("session_id")
        if not isinstance(session_id, str):
            raise TypeError("trajectory metadata has invalid session_id")
        audit = value.get("normalization_audit")
        synthesized = False
        if isinstance(audit, Mapping):
            issues = audit.get("issues")
            if isinstance(issues, list):
                synthesized = any(
                    isinstance(issue, Mapping)
                    and issue.get("code") == "metadata_session_id_synthesized"
                    for issue in issues
                )
        return session_id, (not session_id or synthesized)

    @staticmethod
    def _set_sub_session_id_in_projection(value: object, sub_session_id: int) -> None:
        """Update canonical JSON in place without reparsing a Pydantic tree."""

        if not isinstance(value, Mapping):
            raise TypeError("trajectory payload is not an object")
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            raise TypeError("trajectory payload has no mutable metadata object")
        metadata["sub_session_id"] = sub_session_id
        children = value.get("sub_agent_trajectory")
        if isinstance(children, Mapping):
            for child in children.values():
                StateStore._set_sub_session_id_in_projection(
                    child, sub_session_id
                )

    def assign_sub_session_ids(self) -> int:
        """Number final, deduplicated branches within each real session.

        Missing provider sessions are recognizable by their normalization
        audit marker even though their published value is ``no_session_id``.
        Those user-scoped fallback groups are not real session branches and
        therefore always receive zero.  A literal provider session whose value
        is ``no_session_id`` has no marker and is numbered normally.
        """

        skipped = 0
        while True:
            assignments: dict[str, int] = {}
            real_sessions: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
            bad_ids: list[str] = []
            rows = self.connection.execute(
                "SELECT trajectory_id,payload FROM trajectories ORDER BY trajectory_id"
            )
            for identifier, payload in rows:
                identifier = str(identifier)
                try:
                    projected = loads(_decompress_payload(payload))
                    if identifier in self._trusted_trajectory_ids:
                        session_id, synthesized = self._trusted_session_metadata(projected)
                        metadata = projected["metadata"]
                        created_at = metadata.get("created_at")
                        source_file = metadata.get("source_file")
                        if not isinstance(created_at, str) or not isinstance(source_file, str):
                            raise ValueError("trajectory metadata has invalid ordering fields")
                        if synthesized:
                            assignments[identifier] = 0
                            continue
                        real_sessions[session_id].append((created_at, identifier, source_file))
                        continue

                    # Rows loaded from a previous process are not trusted and
                    # retain the original full contract and identity checks.
                    node = parse_trajectory_record(projected)
                    if identifier != compute_trajectory_id(node):
                        raise ValueError(
                            "stored trajectory_id does not match trajectory content"
                        )
                    if _has_synthesized_session(node):
                        assignments[identifier] = 0
                        continue
                    real_sessions[node.metadata.session_id].append(
                        (
                            node.metadata.created_at,
                            identifier,
                            node.metadata.source_file,
                        )
                    )
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    zlib.error,
                    zstandard.ZstdError,
                ):
                    bad_ids.append(identifier)

            if bad_ids:
                skipped += self._discard_trajectories(bad_ids)
                continue

            for session_rows in real_sessions.values():
                session_rows.sort()
                for sub_session_id, (_, identifier, _) in enumerate(session_rows):
                    assignments[identifier] = sub_session_id

            ordered_ids = sorted(assignments)
            update_bad_ids: list[str] = []
            with self._write_scope():
                for offset in range(
                    0,
                    len(ordered_ids),
                    _TRAJECTORY_UPDATE_BATCH_SIZE,
                ):
                    identifiers = ordered_ids[
                        offset : offset + _TRAJECTORY_UPDATE_BATCH_SIZE
                    ]
                    placeholders = ",".join("?" for _ in identifiers)
                    payloads = {
                        str(identifier): payload
                        for identifier, payload in self.connection.execute(
                            "SELECT trajectory_id,payload FROM trajectories "
                            f"WHERE trajectory_id IN ({placeholders})",
                            identifiers,
                        )
                    }
                    updates: list[tuple[bytes, str]] = []
                    for identifier in identifiers:
                        try:
                            projected = loads(_decompress_payload(payloads[identifier]))
                            if identifier in self._trusted_trajectory_ids:
                                # sub_session_id is excluded from semantic identity;
                                # the row was fully validated on insertion.
                                self._set_sub_session_id_in_projection(
                                    projected, assignments[identifier]
                                )
                                node_json = dumps(projected, sort_keys=True)
                            else:
                                node = parse_trajectory_record(projected)
                                _set_sub_session_id(node, assignments[identifier])
                                validate_derived_fields(node)
                                if identifier != compute_trajectory_id(node):
                                    raise ValueError(
                                        "sub_session_id changed canonical trajectory identity"
                                    )
                                node_json = dumps(project_trajectory(node), sort_keys=True)
                            updates.append(
                                (_compress_payload(node_json), identifier)
                            )
                        except (
                            ValueError,
                            TypeError,
                            KeyError,
                            zlib.error,
                            zstandard.ZstdError,
                        ):
                            update_bad_ids.append(identifier)
                    if updates:
                        self.connection.executemany(
                            "UPDATE trajectories SET payload=? WHERE trajectory_id=?",
                            updates,
                        )

            if update_bad_ids:
                skipped += self._discard_trajectories(update_bad_ids)
                continue
            return skipped

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
        node_json = dumps(projected, sort_keys=True)
        existing = self.connection.execute(
            "SELECT disposition,representative_key FROM trajectories "
            "WHERE trajectory_id=?",
            (trajectory_id,),
        ).fetchone()
        replace_payload = existing is None or (
            rank.get(disposition, 9),
            representative_key,
        ) < (rank.get(existing[0], 9), existing[1])
        with self._write_scope():
            if existing is None:
                self.connection.execute(
                    "INSERT INTO trajectories VALUES(?,?,?,?)",
                    (
                        trajectory_id,
                        disposition,
                        representative_key,
                        _compress_payload(node_json),
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
                        _compress_payload(node_json),
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
        # Only trust the payload that was actually inserted/replaced.  An
        # existing row may have come from an earlier process and must retain
        # the defensive validation path on this run.
        if existing is None or replace_payload:
            self._trusted_trajectory_ids.add(trajectory_id)

    def iter_trajectories(
        self,
    ) -> Iterator[tuple[str, TrajectoryNode, list[dict[str, object]]]]:
        rows = self.connection.execute(
            "SELECT trajectory_id,payload FROM trajectories ORDER BY trajectory_id"
        )
        for identifier, payload in rows:
            node = parse_trajectory_record(loads(_decompress_payload(payload)))
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
                for (
                    source_ref,
                    sha256,
                    captured_at,
                    disposition,
                    reason_codes,
                ) in origin_rows
            ]
            yield identifier, node, origins

    def iter_snapshots(self) -> Iterator[Snapshot]:
        rows = self.connection.execute(
            "SELECT payload FROM snapshots "
            "ORDER BY session_id,thread_id,captured_at,source_path"
        )
        for (payload,) in rows:
            yield self._decode_snapshot(payload)

    def commit(self) -> None:
        if self._write_batch_depth:
            raise RuntimeError("cannot commit inside a StateStore write_batch")
        self.connection.commit()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
