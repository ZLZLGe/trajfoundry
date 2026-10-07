"""Disk-backed, exact cumulative-history prefix aggregation.

Only a single input snapshot and byte-bounded lookup caches reside in Python.
The trie, source endpoints and routing evidence live in a disposable SQLite
database.  Contributor relationships are queried rather than expanded into a
potentially quadratic leaf/source matrix.  Result views are valid only inside
``disk_prefix_leaves``'s context manager.
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain
from pathlib import Path
from typing import Any, TypeVar

import zstandard

from .canonical import compaction_signature
from .models import Snapshot
from .streaming import (
    _compatibility,
    _evidence_copy,
    _has_agent_messages,
    _has_linkage,
    _has_spawn_response,
    message_prefix_token,
)

logger = logging.getLogger(__name__)
_STABLE_ORDER = "captured_at,turn_id,request_id,source_path,source_sha256,id"
_T = TypeVar("_T")


class _ByteCache:
    """Small LRU with conservative accounting, including per-entry overhead."""

    def __init__(self, limit: int) -> None:
        self.limit = max(0, limit)
        self.bytes = 0
        self.items: OrderedDict[Any, tuple[Any, int]] = OrderedDict()

    def get(self, key: Any) -> Any:
        item = self.items.get(key)
        if item is None:
            return None
        self.items.move_to_end(key)
        return item[0]

    def put(self, key: Any, value: Any, payload_bytes: int) -> None:
        weight = payload_bytes + 384
        previous = self.items.pop(key, None)
        if previous is not None:
            self.bytes -= previous[1]
        if weight > self.limit:
            return
        while self.items and self.bytes + weight > self.limit:
            _, (_, old_weight) = self.items.popitem(last=False)
            self.bytes -= old_weight
        self.items[key] = value, weight
        self.bytes += weight


class _QuerySequence(Sequence[_T]):
    """Reiterable SQL-backed collection; ordinary iteration never fetches all."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        sql: str,
        parameters: tuple[Any, ...],
        decode: Callable[[tuple[Any, ...]], _T],
    ) -> None:
        self.connection = connection
        self.sql = sql
        self.parameters = parameters
        self.decode = decode
        self._count: int | None = None

    def __iter__(self) -> Iterator[_T]:
        cursor = self.connection.execute(self.sql, self.parameters)
        try:
            for row in cursor:
                yield self.decode(row)
        finally:
            cursor.close()

    def __len__(self) -> int:
        if self._count is None:
            self._count = self.connection.execute(
                f"SELECT count(*) FROM ({self.sql})", self.parameters
            ).fetchone()[0]
        return self._count

    def __getitem__(self, index: int | slice) -> _T | list[_T]:
        if isinstance(index, slice):
            # Slicing explicitly requests materialization; pipeline consumers
            # use iteration instead. Keep tuple-like convenience for callers.
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        row = self.connection.execute(
            f"SELECT * FROM ({self.sql}) LIMIT 1 OFFSET ?",
            (*self.parameters, index),
        ).fetchone()
        if row is None:
            raise IndexError(index)
        return self.decode(row)


class _ContributorMapping(Mapping[str, Sequence[str]]):
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __iter__(self) -> Iterator[str]:
        cursor = self.connection.execute(
            "SELECT source_path FROM active_snapshots "
            "GROUP BY source_path ORDER BY source_path"
        )
        try:
            for (source_path,) in cursor:
                yield source_path
        finally:
            cursor.close()

    def __len__(self) -> int:
        return self.connection.execute(
            "SELECT count(DISTINCT source_path) FROM active_snapshots"
        ).fetchone()[0]

    def __getitem__(self, source_path: str) -> Sequence[str]:
        row = self.connection.execute(
            "SELECT prefix_end FROM active_snapshots WHERE source_path=? "
            "ORDER BY captured_at DESC,turn_id DESC,request_id DESC,"
            "source_sha256 DESC,id DESC LIMIT 1",
            (source_path,),
        ).fetchone()
        if row is None:
            raise KeyError(source_path)
        # A capture contributes only when its COMPLETE transcript lies on a
        # strict HISTORY prefix. Equal terminal transcripts deliberately do not
        # become each other's contributors. The capture itself is always kept.
        sql = """
            WITH RECURSIVE ancestors(id) AS (
                SELECT ? WHERE ? IS NOT NULL
                UNION ALL
                SELECT nodes.parent FROM nodes JOIN ancestors ON nodes.id=ancestors.id
                WHERE nodes.parent IS NOT NULL
            )
            SELECT source_path FROM snapshots
            JOIN ancestors ON snapshots.endpoint=ancestors.id
            UNION SELECT ?
            ORDER BY source_path
        """
        return _QuerySequence(
            self.connection, sql, (row[0], row[0], source_path), lambda item: item[0]
        )


@dataclass(frozen=True, slots=True)
class DiskAggregationResult:
    leaves: Sequence[Snapshot]
    contributor_paths: Mapping[str, Sequence[str]]
    intermediate_paths: Sequence[str]
    spawn_evidence: Sequence[Snapshot]
    database_path: Path


class _Builder:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        fetch_snapshot: Callable[[str], Snapshot | None] | None,
        lookup_cache_bytes: int,
    ) -> None:
        self.connection = connection
        self.fetch_snapshot = fetch_snapshot
        self.nodes = _ByteCache(lookup_cache_bytes)
        self.scopes = _ByteCache(min(4 * 1024**2, lookup_cache_bytes))
        self.compressor = zstandard.ZstdCompressor(level=1)
        self.decompressor = zstandard.ZstdDecompressor()
        self.node_count = 0

    def scope_root(self, compatibility: tuple[Hashable, bytes]) -> int:
        # Serialized keys are stored, never deserialized/executed. Default
        # compatibility consists solely of tuples of strings and bytes.
        key = pickle.dumps(compatibility, protocol=5)
        cached = self.scopes.get(key)
        if cached is not None:
            return cached
        row = self.connection.execute(
            "SELECT root FROM scopes WHERE scope_key=?", (key,)
        ).fetchone()
        if row is not None:
            root = row[0]
        else:
            root = self.connection.execute(
                "INSERT INTO nodes(parent,token_hash,token,covered) VALUES(NULL,NULL,NULL,0)"
            ).lastrowid
            self.connection.execute(
                "INSERT INTO scopes(scope_key,root) VALUES(?,?)", (key, root)
            )
            self.node_count += 1
        self.scopes.put(key, root, len(key))
        return root

    def advance(self, parent: int, token: bytes, *, covered: bool) -> int:
        digest = sha256(token).digest()
        key = parent, digest, token
        cached = self.nodes.get(key)
        if cached is not None:
            node, was_covered = cached
        else:
            # Hashes accelerate lookup; exact canonical bytes remain the
            # authority even under an adversarial or accidental hash collision.
            row = self.connection.execute(
                "SELECT id,covered FROM nodes WHERE parent=? AND token_hash=? AND token=?",
                (parent, digest, token),
            ).fetchone()
            if row is None:
                node = self.connection.execute(
                    "INSERT INTO nodes(parent,token_hash,token,covered) VALUES(?,?,?,?)",
                    (parent, digest, token, int(covered)),
                ).lastrowid
                was_covered = covered
                self.node_count += 1
            else:
                node, was_covered = row
        if covered and not was_covered:
            self.connection.execute("UPDATE nodes SET covered=1 WHERE id=?", (node,))
            was_covered = True
        self.nodes.put(key, (node, was_covered), len(token) + len(digest))
        return node

    def encode(self, snapshot: Snapshot) -> bytes:
        return self.compressor.compress(snapshot.model_dump_json().encode("utf-8"))

    def decode(self, row: tuple[Any, ...]) -> Snapshot:
        source_path, payload = row
        if payload is not None:
            # JSON-mode validation handles strict enum fields correctly.
            return Snapshot.model_validate_json(self.decompressor.decompress(payload))
        assert self.fetch_snapshot is not None
        snapshot = self.fetch_snapshot(source_path)
        if snapshot is None:
            raise RuntimeError(f"prefix leaf snapshot disappeared: {source_path}")
        return snapshot

    def add(
        self, snapshot: Snapshot, scope_fn: Callable[[Snapshot], Hashable] | None
    ) -> None:
        compatibility = (
            _compatibility(snapshot)
            if scope_fn is None
            else (scope_fn(snapshot), compaction_signature(snapshot.compaction_items))
        )
        node = self.scope_root(compatibility)
        complete_length = len(snapshot.history) + len(snapshot.response)
        prefix_depth = min(len(snapshot.history), complete_length - 1)
        prefix_end = None
        if complete_length:
            self.connection.execute(
                "UPDATE nodes SET covered=1 WHERE id=? AND covered=0", (node,)
            )
            if prefix_depth == 0:
                prefix_end = node
        for depth, message in enumerate(chain(snapshot.history, snapshot.response), 1):
            node = self.advance(
                node, message_prefix_token(message), covered=depth <= prefix_depth
            )
            if depth == prefix_depth:
                prefix_end = node

        # Coverage is monotone: if an extender is itself removed, its longer
        # replacement extends all the same prefix nodes. Keeping one bit per
        # trie node therefore produces the same leaves as active refcounts.
        evidence = None
        if (
            _has_spawn_response(snapshot)
            or _has_linkage(snapshot)
            or _has_agent_messages(snapshot)
        ):
            evidence = self.encode(_evidence_copy(snapshot))
        self.connection.execute(
            "INSERT INTO snapshots(source_path,source_sha256,captured_at,turn_id,"
            "request_id,endpoint,prefix_end,payload,evidence) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                snapshot.source_path,
                snapshot.source_sha256,
                snapshot.captured_at,
                snapshot.turn_id,
                snapshot.request_id,
                node,
                prefix_end,
                self.encode(snapshot) if self.fetch_snapshot is None else None,
                evidence,
            ),
        )


@contextmanager
def disk_prefix_leaves(
    snapshots: Iterable[Snapshot],
    directory: Path,
    *,
    fetch_snapshot: Callable[[str], Snapshot | None] | None = None,
    scope_fn: Callable[[Snapshot], Hashable] | None = None,
    lookup_cache_bytes: int = 128 * 1024**2,
    sqlite_cache_mib: int = 64,
) -> Iterator[DiskAggregationResult]:
    """Aggregate exact prefixes with disk-backed lazy result collections.

    ``directory`` must be Worker-local storage (normally the job's ``/tmp``
    workspace). ``fetch_snapshot`` should retrieve an unchanged input snapshot
    by source path; supplying it avoids storing a second copy of input bodies.
    The database is scratch, not a durable checkpoint, and is removed on exit.
    Memory is bounded by configured caches plus the current single snapshot.
    """

    if sqlite_cache_mib < 1 or lookup_cache_bytes < 0:
        raise ValueError(
            "disk prefix cache sizes must be nonnegative; SQLite cache >= 1 MiB"
        )
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prefix-", dir=directory) as scratch:
        database_path = Path(scratch) / "prefix.sqlite"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            # Scratch data can be rebuilt. A killed process must not publish a
            # successful job, but there is no reason to fsync every scratch batch.
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(f"PRAGMA cache_size={-sqlite_cache_mib * 1024}")
            connection.execute("PRAGMA mmap_size=0")
            connection.executescript("""
                CREATE TABLE scopes(scope_key BLOB PRIMARY KEY, root INTEGER NOT NULL);
                CREATE TABLE nodes(
                    id INTEGER PRIMARY KEY, parent INTEGER, token_hash BLOB,
                    token BLOB, covered INTEGER NOT NULL
                );
                CREATE INDEX nodes_children ON nodes(parent,token_hash);
                CREATE TABLE snapshots(
                    id INTEGER PRIMARY KEY, source_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL, captured_at TEXT NOT NULL,
                    turn_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    endpoint INTEGER NOT NULL, prefix_end INTEGER,
                    payload BLOB, evidence BLOB
                );
                CREATE VIEW active_snapshots AS
                    SELECT snapshots.* FROM snapshots JOIN nodes
                    ON snapshots.endpoint=nodes.id WHERE nodes.covered=0;
            """)
            builder = _Builder(
                connection,
                fetch_snapshot=fetch_snapshot,
                lookup_cache_bytes=lookup_cache_bytes,
            )
            started = time.monotonic()
            last_log = started
            count = 0
            for count, snapshot in enumerate(snapshots, 1):
                builder.add(snapshot, scope_fn)
                if count % 1000 == 0:
                    connection.commit()
                now = time.monotonic()
                if now - last_log >= 60:
                    logger.info(
                        "【磁盘前缀】已处理来源=%d，索引节点=%d，查找缓存=%.1fMiB，数据库=%.1fMiB，耗时=%.1f秒",
                        count,
                        builder.node_count,
                        builder.nodes.bytes / 1024**2,
                        database_path.stat().st_size / 1024**2,
                        now - started,
                    )
                    last_log = now
            if count:
                del snapshot
            connection.commit()
            connection.executescript("""
                CREATE INDEX snapshots_endpoint ON snapshots(endpoint,source_path);
                CREATE INDEX snapshots_source ON snapshots(source_path);
                CREATE INDEX snapshots_stable ON snapshots(
                    captured_at,turn_id,request_id,source_path,source_sha256,id
                );
            """)
            # No lookup needs remain; release token bodies before the caller
            # starts building trajectories or launches worker processes.
            builder.nodes.items.clear()
            builder.nodes.bytes = 0
            builder.scopes.items.clear()
            builder.scopes.bytes = 0
            result = DiskAggregationResult(
                leaves=_QuerySequence(
                    connection,
                    f"SELECT source_path,payload FROM active_snapshots ORDER BY {_STABLE_ORDER}",
                    (),
                    builder.decode,
                ),
                contributor_paths=_ContributorMapping(connection),
                intermediate_paths=_QuerySequence(
                    connection,
                    "SELECT source_path FROM snapshots s WHERE NOT EXISTS("
                    "SELECT 1 FROM active_snapshots a WHERE a.source_path=s.source_path) "
                    "ORDER BY source_path",
                    (),
                    lambda row: row[0],
                ),
                spawn_evidence=_QuerySequence(
                    connection,
                    "SELECT source_path,evidence FROM snapshots WHERE evidence IS NOT NULL "
                    f"ORDER BY {_STABLE_ORDER}",
                    (),
                    builder.decode,
                ),
                database_path=database_path,
            )
            logger.info(
                "【磁盘前缀】完成：来源=%d，叶子=%d，中间来源=%d，索引节点=%d，数据库=%.1fMiB，耗时=%.1f秒",
                count,
                len(result.leaves),
                len(result.intermediate_paths),
                builder.node_count,
                database_path.stat().st_size / 1024**2,
                time.monotonic() - started,
            )
            yield result
        finally:
            connection.close()


__all__ = ["DiskAggregationResult", "disk_prefix_leaves"]
