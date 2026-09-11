"""Filesystem and JSONL helpers."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import orjson

SAFE_REQUEST_HEADERS = {
    "x-claude-code-session-id",
    "x-codex-parent-thread-id",
    "x-codex-turn-metadata",
    "x-openai-subagent",
}


@dataclass(frozen=True, slots=True)
class LocalCapture:
    """One capture rooted at a local input directory."""

    source_ref: str
    path: Path


class CaptureSource(Protocol):
    """Minimal source boundary consumed by the normalization pipeline."""

    @property
    def label(self) -> str: ...

    def iter_captures(
        self,
        input_format: Literal["freerouter", "tokenplan"],
    ) -> Iterator[Any]: ...

    def read_capture_bytes(self, capture: Any) -> tuple[bytes, str]: ...


class LocalCaptureSource:
    """Expose the existing read-only filesystem behavior as a capture source."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def label(self) -> str:
        return str(self.root)

    def iter_captures(
        self,
        input_format: Literal["freerouter", "tokenplan"],
    ) -> Iterator[LocalCapture]:
        for path in discover_captures(self.root, input_format=input_format):
            yield LocalCapture(
                source_ref=path.relative_to(self.root).as_posix(),
                path=path,
            )

    def read_capture_bytes(self, capture: LocalCapture) -> tuple[bytes, str]:
        return read_capture_bytes(capture.path)


def discover_captures(
    root: Path,
    *,
    input_format: Literal["freerouter", "tokenplan"] = "freerouter",
) -> Iterator[Path]:
    """Yield only capture files for the selected source format.

    TokenPlan partitions also contain manifests and media assets.  Request
    envelopes have a stable ``req_*.json`` basename, so selecting that shape
    keeps manifests out of ingest without inspecting or mutating the source
    tree.  A caller can include a top-level ``test`` partition simply by
    choosing an input root that contains it.
    """

    if input_format == "freerouter":
        pattern = "*.json"
    elif input_format == "tokenplan":
        pattern = "req_*.json"
    else:  # pragma: no cover - PipelineConfig constrains the public value.
        raise ValueError(f"unsupported input format: {input_format!r}")
    yield from sorted(path for path in root.rglob(pattern) if path.is_file())


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_capture_bytes(path: Path) -> tuple[bytes, str]:
    """Read and hash one stable file descriptor exactly once."""

    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise OSError("capture changed while it was being read")
    return payload, hashlib.sha256(payload).hexdigest()


def decode_capture(
    payload: bytes,
    *,
    input_format: Literal["freerouter", "tokenplan"] = "freerouter",
) -> dict[str, Any]:
    """Decode one capture while discarding sensitive headers immediately.

    Memory is bounded by an individual capture, never by a complete session.
    """

    capture = orjson.loads(payload)
    if not isinstance(capture, dict):
        raise TypeError("capture root must be a JSON object")
    if input_format == "freerouter":
        raw_headers = capture.get("request_headers")
        if isinstance(raw_headers, dict):
            capture["request_headers"] = {
                str(key).lower(): value
                for key, value in raw_headers.items()
                if str(key).lower() in SAFE_REQUEST_HEADERS
            }
        else:
            capture["request_headers"] = {}
        capture.pop("response_headers", None)
        capture.pop("query_string", None)
    elif input_format != "tokenplan":
        raise ValueError(f"unsupported input format: {input_format!r}")
    return capture


def load_capture(path: Path) -> dict[str, Any]:
    payload, _ = read_capture_bytes(path)
    return decode_capture(payload)


class JsonlShardWriter:
    def __init__(
        self, directory: Path, prefix: str, max_bytes: int = 512 * 1024 * 1024
    ):
        self.directory = directory
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index = 0
        self._size = 0
        self._handle = None
        self._pending_path: Path | None = None
        self.paths: list[Path] = []

    def _open(self) -> None:
        path = self.directory / f"{self.prefix}-{self._index:05d}.jsonl"
        pending = path.with_suffix(path.suffix + ".tmp")
        self._index += 1
        self._handle = pending.open("wb")
        self._pending_path = pending
        self.paths.append(path)
        self._size = 0

    def write(self, value: Any) -> None:
        line = orjson.dumps(value, option=orjson.OPT_SORT_KEYS) + b"\n"
        if self._handle is None or (
            self._size and self._size + len(line) > self.max_bytes
        ):
            self.close_current()
            self._open()
        assert self._handle is not None
        self._handle.write(line)
        self._size += len(line)

    def close_current(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            assert self._pending_path is not None
            self._pending_path.replace(self.paths[-1])
            self._handle = None
            self._pending_path = None

    def close(self) -> None:
        self.close_current()

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._pending_path is not None:
            self._pending_path.unlink(missing_ok=True)
            self._pending_path = None
        for path in self.paths:
            path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
