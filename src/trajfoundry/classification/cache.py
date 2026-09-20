"""Small SQLite cache for restartable trajectory classification."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Self

import orjson


class ClassificationCache:
    """Thread-safe cache that stores only validated successful classifications."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS classifications (
                cache_key TEXT PRIMARY KEY,
                result BLOB NOT NULL
            )
            """
        )
        self._connection.commit()
        self._lock = threading.Lock()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT result FROM classifications WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            value = orjson.loads(row[0])
        except orjson.JSONDecodeError:
            return None
        if type(value) is not dict or value.get("status") != "accepted":
            return None
        return value

    def put(self, cache_key: str, result: dict[str, Any]) -> None:
        if result.get("status") != "accepted":
            raise ValueError("only accepted classifications may be cached")
        payload = orjson.dumps(result, option=orjson.OPT_SORT_KEYS)
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO classifications(cache_key, result) VALUES (?, ?)",
                (cache_key, payload),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["ClassificationCache"]
