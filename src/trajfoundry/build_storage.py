"""Small disk-backed collections for normalization planning.

These collections hold provenance strings, never full capture objects.  They
keep shared-prefix fanout from creating a second in-memory copy of every
source relationship while preserving deterministic source ordering.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path


class BuildCollections:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=TRUNCATE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute(
            "CREATE TABLE paths (collection TEXT NOT NULL, path TEXT NOT NULL, "
            "PRIMARY KEY(collection,path)) WITHOUT ROWID"
        )

    def paths(self, key: str) -> DiskPathSet:
        return DiskPathSet(self, key)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class DiskPathSet:
    def __init__(self, store: BuildCollections, key: str) -> None:
        self.store = store
        self.key = key

    def update(self, paths: Iterable[str]) -> None:
        self.store.connection.executemany(
            "INSERT OR IGNORE INTO paths VALUES(?,?)",
            ((self.key, path) for path in paths),
        )

    def __iter__(self) -> Iterator[str]:
        for row in self.store.connection.execute(
            "SELECT path FROM paths WHERE collection=? ORDER BY path", (self.key,)
        ):
            yield str(row[0])

    def __len__(self) -> int:
        return int(
            self.store.connection.execute(
                "SELECT count(*) FROM paths WHERE collection=?", (self.key,)
            ).fetchone()[0]
        )


__all__ = ["BuildCollections", "DiskPathSet"]
