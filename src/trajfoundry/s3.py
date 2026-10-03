"""Strict S3 locations and stable capture reads.

The module deliberately keeps the AWS SDK behind :func:`create_s3_client` so
importing TrajFoundry does not initialize an SDK session or require boto3 for
local-only jobs.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sqlite3
import tempfile
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from .credentials import S3Credentials

InputFormat = Literal["freerouter", "tokenplan", "sxf", "deepinfra"]
_READ_CHUNK_BYTES = 1024 * 1024
_DEFAULT_READ_WORKERS = 4
_MAX_POOL_CONNECTIONS = 8
_PROGRESS_INTERVAL_SECONDS = 60.0
_DEFAULT_READ_PREFETCH_BYTES = 512 * 1024 * 1024
_INVENTORY_INSERT_BATCH = 10_000

LOGGER = logging.getLogger(__name__)


class _CountingBody:
    """Adapt an S3 body for zstandard while counting compressed bytes."""

    def __init__(self, body: Any) -> None:
        self.body = body
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        value = self.body.read(size)
        if not isinstance(value, bytes):
            raise TypeError("S3 response body must yield bytes")
        self.bytes_read += len(value)
        return value

    def readinto(self, buffer: bytearray | memoryview) -> int:
        value = self.read(len(buffer))
        buffer[: len(value)] = value
        return len(value)

    def close(self) -> None:
        close = getattr(self.body, "close", None)
        if callable(close):
            close()


def _validate_bucket(bucket: str) -> None:
    if not isinstance(bucket, str):
        raise TypeError("S3 bucket must be a string")
    if not bucket:
        raise ValueError("S3 bucket must not be empty")
    if bucket in {".", ".."}:
        raise ValueError("S3 bucket is invalid")
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in bucket
    ):
        raise ValueError("S3 bucket must not contain whitespace or control characters")
    if any(character in bucket for character in "/\\@?#:"):
        raise ValueError("S3 bucket must not contain URI delimiters")


def _validate_path_segments(value: str, *, name: str, directory: bool) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if value.startswith("/"):
        raise ValueError(f"{name} must be relative")
    if "\\" in value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(f"{name} must not contain a backslash or control character")
    if directory:
        if not value.endswith("/"):
            raise ValueError(f"{name} must end with '/'")
        segment_value = value[:-1]
    else:
        if value.endswith("/"):
            raise ValueError(f"{name} must identify an object, not a directory")
        segment_value = value
    segments = segment_value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"{name} contains an unsafe path segment")


@dataclass(frozen=True, slots=True)
class S3Location:
    """A bucket and a canonical, non-root directory prefix."""

    bucket: str
    prefix: str

    def __post_init__(self) -> None:
        _validate_bucket(self.bucket)
        _validate_path_segments(self.prefix, name="S3 prefix", directory=True)

    @property
    def uri(self) -> str:
        """Return the location without embedding credentials or query data."""

        return f"s3://{self.bucket}/{self.prefix}"

    def key(self, relative: str) -> str:
        """Join a safe object-relative path beneath this location."""

        _validate_path_segments(relative, name="relative S3 key", directory=False)
        return f"{self.prefix}{relative}"


def parse_s3_uri(uri: str) -> S3Location:
    """Parse a strict ``s3://bucket/non-empty/prefix/`` directory URI.

    Valid key text is preserved byte-for-byte.  Ambiguous path spellings are
    rejected instead of being silently normalized to a different S3 key.
    """

    if not isinstance(uri, str):
        raise TypeError("S3 URI must be a string")
    if any(ord(character) < 32 or ord(character) == 127 for character in uri):
        raise ValueError("S3 URI must not contain control characters")
    if not uri.startswith("s3://"):
        raise ValueError("S3 URI must start with 's3://'")
    if "?" in uri or "#" in uri:
        raise ValueError("S3 URI must not contain a query or fragment")

    parsed = urlsplit(uri)
    if parsed.scheme != "s3":  # Defensive; the exact prefix check is authoritative.
        raise ValueError("S3 URI must use the s3 scheme")
    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
    ):
        raise ValueError("S3 URI must not contain credentials")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("S3 URI contains an invalid authority") from error
    if port is not None:
        raise ValueError("S3 URI must not contain a port")
    if parsed.netloc != parsed.hostname:
        raise ValueError("S3 URI authority must contain only a bucket")
    if not parsed.path.startswith("/"):
        raise ValueError("S3 URI must contain an absolute URI path")

    # Remove only the separator introduced by URI syntax.  A second leading
    # slash remains visible to validation and is rejected as an empty segment.
    prefix = parsed.path[1:]
    return S3Location(bucket=parsed.netloc, prefix=prefix)


def create_s3_client(
    endpoint_url: str,
    region_name: str,
    *,
    credentials: S3Credentials | None = None,
    max_pool_connections: int = _MAX_POOL_CONNECTIONS,
) -> Any:
    """Create the configured S3-compatible client, importing boto3 lazily."""

    import boto3
    from botocore.config import Config

    if max_pool_connections <= 0:
        raise ValueError("max pool connections must be positive")
    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        retries={"mode": "standard", "max_attempts": 10},
        max_pool_connections=max_pool_connections,
    )
    client_kwargs: dict[str, Any] = {
        "endpoint_url": endpoint_url,
        "region_name": region_name,
        "config": config,
    }
    if credentials is not None:
        client_kwargs["aws_access_key_id"] = credentials.aws_access_key_id
        client_kwargs["aws_secret_access_key"] = credentials.aws_secret_access_key
        if credentials.aws_session_token is not None:
            client_kwargs["aws_session_token"] = credentials.aws_session_token

    return boto3.client("s3", **client_kwargs)


@dataclass(frozen=True, slots=True)
class S3Capture:
    """Immutable identity metadata for one capture in an S3 input scan."""

    source_ref: str
    key: str
    size: int
    etag: str


class S3CaptureSource:
    """List and read stable capture objects under one S3 directory prefix."""

    def __init__(
        self,
        client: Any,
        location: S3Location,
        *,
        read_workers: int = _DEFAULT_READ_WORKERS,
        read_prefetch: int | None = None,
        read_prefetch_bytes: int | None = None,
        inventory_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if read_workers <= 0:
            raise ValueError("read workers must be positive")
        if read_prefetch is None:
            read_prefetch = read_workers
        if read_prefetch <= 0:
            raise ValueError("read prefetch must be positive")
        if read_prefetch_bytes is None:
            read_prefetch_bytes = _DEFAULT_READ_PREFETCH_BYTES
        if read_prefetch_bytes <= 0:
            raise ValueError("read prefetch bytes must be positive")
        self.client = client
        self.location = location
        self.read_workers = read_workers
        self.read_prefetch = read_prefetch
        self.read_prefetch_bytes = read_prefetch_bytes
        self.inventory_dir = (
            None if inventory_dir is None else os.fspath(inventory_dir)
        )

    @property
    def label(self) -> str:
        return self.location.uri

    @property
    def payload_formats(self) -> frozenset[InputFormat]:
        """Formats this source can yield directly to the ingest pipeline."""

        return frozenset({"freerouter", "tokenplan", "sxf", "deepinfra"})

    def iter_captures(self, input_format: InputFormat) -> Iterator[S3Capture]:
        """Return capture metadata in stable relative-key order.

        Listing is started before this method returns, preserving the
        historical behavior where listing/service errors are raised at call
        time.  The returned iterator still streams rows from the temporary
        inventory instead of materializing all captures in memory.
        """

        captures = self._iter_captures(input_format)
        try:
            first = next(captures)
        except StopIteration:
            return iter(())

        def prepend_first() -> Iterator[S3Capture]:
            try:
                yield first
                yield from captures
            finally:
                close = getattr(captures, "close", None)
                if callable(close):
                    close()

        return prepend_first()

    def _iter_captures(self, input_format: InputFormat) -> Iterator[S3Capture]:
        """Return capture metadata in stable relative-key order.

        The listing can contain millions of objects.  Keep the inventory in a
        temporary SQLite file rather than retaining Python objects and a
        duplicate-detection set for the entire prefix.  The database is
        deleted when this iterator is exhausted (or closed).
        """

        if input_format not in {"freerouter", "tokenplan", "sxf", "deepinfra"}:
            raise ValueError(f"unsupported input format: {input_format!r}")

        inventory_dir = self.inventory_dir
        if inventory_dir is not None:
            os.makedirs(inventory_dir, exist_ok=True)
        fd, inventory_name = tempfile.mkstemp(
            prefix="trajfoundry-inventory-",
            suffix=".sqlite",
            dir=inventory_dir,
        )
        os.close(fd)
        inventory_path = inventory_name
        connection: sqlite3.Connection | None = None
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.location.bucket,
            Prefix=self.location.prefix,
        )
        started_at = time.monotonic()
        last_progress_at = started_at
        page_count = 0
        selected_count = 0
        selected_bytes = 0
        try:
            connection = sqlite3.connect(inventory_path)
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA locking_mode=EXCLUSIVE")
            connection.execute(
                """
                CREATE TABLE inventory (
                    key TEXT PRIMARY KEY,
                    source_ref TEXT,
                    size INTEGER,
                    etag TEXT,
                    selected INTEGER NOT NULL
                ) WITHOUT ROWID
                """
            )
            connection.commit()
            pending_rows: list[tuple[str, str, int | None, str | None, int]] = []

            def flush_rows() -> None:
                if not pending_rows:
                    return
                try:
                    connection.executemany(
                        "INSERT INTO inventory(key, source_ref, size, etag, selected) "
                        "VALUES (?, ?, ?, ?, ?)",
                        pending_rows,
                    )
                except sqlite3.IntegrityError as error:
                    raise OSError("S3 listing returned a duplicate key") from error
                connection.commit()
                pending_rows.clear()

            for page in pages:
                page_count += 1
                contents = page.get("Contents", [])
                if contents is None:
                    contents = []
                if not isinstance(contents, list):
                    raise TypeError("S3 listing Contents must be a list")
                for item in contents:
                    if not isinstance(item, dict):
                        raise TypeError("S3 listing entry must be an object")
                    key = item["Key"]
                    if not isinstance(key, str) or not key.startswith(
                        self.location.prefix
                    ):
                        raise OSError(
                            "S3 listing returned a key outside the requested prefix"
                        )
                    if key.endswith("/"):
                        # Directory markers were previously checked for duplicate
                        # keys too.  Keep them in the disk-backed inventory so the
                        # duplicate check remains complete without a Python set.
                        pending_rows.append((key, "", None, None, 0))
                        if len(pending_rows) >= _INVENTORY_INSERT_BATCH:
                            flush_rows()
                        continue
                    size = item["Size"]
                    etag = item["ETag"]
                    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                        raise TypeError("S3 listing Size must be a non-negative integer")
                    if not isinstance(etag, str) or not etag:
                        raise TypeError("S3 listing ETag must be a non-empty string")

                    source_ref = key[len(self.location.prefix) :]
                    if not source_ref:
                        continue
                    if self.location.key(source_ref) != key:  # Defensive exact join check.
                        raise OSError("S3 listing returned an unsafe relative key")
                    basename = source_ref.rsplit("/", 1)[-1]
                    if input_format in {"freerouter", "deepinfra"}:
                        selected = basename.endswith(".json")
                    elif input_format == "tokenplan":
                        selected = basename.startswith("req_") and basename.endswith(
                            ".json"
                        )
                    else:
                        selected = basename.endswith(".jsonl.zst")
                    pending_rows.append((key, source_ref, size, etag, int(selected)))
                    if selected:
                        selected_count += 1
                        selected_bytes += size
                    if len(pending_rows) >= _INVENTORY_INSERT_BATCH:
                        flush_rows()

                flush_rows()
                now = time.monotonic()
                if page_count == 1 or now - last_progress_at >= _PROGRESS_INTERVAL_SECONDS:
                    LOGGER.info(
                        "【输入清单】已扫描分页=%d，已发现候选对象=%d，候选字节=%d，耗时=%.1f秒",
                        page_count,
                        selected_count,
                        selected_bytes,
                        now - started_at,
                    )
                    last_progress_at = now

            # The primary key detects duplicates only when rows are inserted.
            # Querying this index also avoids materializing all captures in RAM.
            connection.execute(
                "CREATE INDEX inventory_source_ref_idx ON inventory(source_ref) "
                "WHERE selected = 1"
            )
            connection.commit()
            LOGGER.info(
                "【输入清单】扫描完成：分页=%d，候选对象=%d，候选字节=%d，清单文件=%.1fMiB，耗时=%.1f秒",
                page_count,
                selected_count,
                selected_bytes,
                os.path.getsize(inventory_path) / 1024**2,
                time.monotonic() - started_at,
            )
            rows = connection.execute(
                "SELECT source_ref, key, size, etag FROM inventory "
                "WHERE selected = 1 ORDER BY source_ref"
            )
            for source_ref, key, size, etag in rows:
                yield S3Capture(
                    source_ref=source_ref,
                    key=key,
                    size=size,
                    etag=etag,
                )
        finally:
            if connection is not None:
                connection.close()
            try:
                os.unlink(inventory_path)
            except FileNotFoundError:
                pass

    def iter_capture_payloads(
        self, input_format: InputFormat
    ) -> Iterator[tuple[str, bytes, str]]:
        """Yield stable payloads with bounded prefetch for ordinary objects.

        Ordinary JSON captures are fetched by a bounded worker pool while
        results remain ordered by source reference.  SXF remains a sequential
        decompression stream so a compressed object is never accumulated in
        memory.  Each SXF hash covers the exact JSONL row without its line
        ending.
        """

        if input_format != "sxf":
            yield from self._iter_prefetched_capture_payloads(input_format)
            return
        try:
            import zstandard
        except ImportError as error:  # pragma: no cover - environment setup issue
            raise RuntimeError(
                "SXF input requires the 'zstandard' package in the runtime environment"
            ) from error

        for capture in self.iter_captures("sxf"):
            response = self.client.get_object(
                Bucket=self.location.bucket,
                Key=capture.key,
                IfMatch=capture.etag,
            )
            body = response.get("Body")
            if body is None or not callable(getattr(body, "read", None)):
                raise OSError("S3 SXF object response has no readable body")
            response_etag = response.get("ETag")
            if response_etag != capture.etag:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
                raise OSError("S3 SXF object ETag changed while it was being read")
            content_length = response.get("ContentLength")
            if (
                not isinstance(content_length, int)
                or isinstance(content_length, bool)
                or content_length != capture.size
            ):
                close = getattr(body, "close", None)
                if callable(close):
                    close()
                raise OSError("S3 SXF object size changed while it was being read")
            counted_body = _CountingBody(body)
            try:
                with io.BufferedReader(
                    zstandard.ZstdDecompressor().stream_reader(counted_body)
                ) as reader:
                    line_no = 0
                    while True:
                        line = reader.readline()
                        if not line:
                            break
                        payload = line.rstrip(b"\r\n")
                        source_ref = f"{capture.source_ref}#L{line_no:08d}"
                        yield source_ref, payload, hashlib.sha256(payload).hexdigest()
                        line_no += 1
                if counted_body.bytes_read != capture.size:
                    raise OSError("S3 SXF object ended before its advertised size")
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()

    def _iter_prefetched_capture_payloads(
        self, input_format: InputFormat
    ) -> Iterator[tuple[str, bytes, str]]:
        captures = iter(self.iter_captures(input_format))
        executor = ThreadPoolExecutor(
            max_workers=self.read_workers,
            thread_name_prefix="trajfoundry-s3-read",
        )
        pending: deque[tuple[S3Capture, int, Future[tuple[bytes, str]]]] = deque()
        pending_bytes = 0
        peak_pending_bytes = 0
        submitted = 0
        completed = 0
        completed_bytes = 0
        deferred_capture: S3Capture | None = None
        started_at = time.monotonic()
        last_progress_at = started_at

        def fill() -> None:
            nonlocal deferred_capture, pending_bytes, peak_pending_bytes, submitted
            while len(pending) < self.read_prefetch:
                if deferred_capture is None:
                    try:
                        deferred_capture = next(captures)
                    except StopIteration:
                        return
                capture = deferred_capture
                reservation = max(capture.size, 1)
                # Permit one oversized object so a single large capture cannot
                # deadlock the iterator, but never submit another object while
                # the byte budget is full.
                if pending and pending_bytes + reservation > self.read_prefetch_bytes:
                    return
                deferred_capture = None
                pending.append(
                    (capture, reservation, executor.submit(self.read_capture_bytes, capture))
                )
                pending_bytes += reservation
                peak_pending_bytes = max(peak_pending_bytes, pending_bytes)
                submitted += 1

        try:
            fill()
            while pending:
                capture, reservation, future = pending.popleft()
                pending_bytes -= reservation
                payload, digest = future.result()
                completed += 1
                completed_bytes += len(payload)
                fill()
                yield capture.source_ref, payload, digest
                now = time.monotonic()
                if now - last_progress_at >= _PROGRESS_INTERVAL_SECONDS:
                    LOGGER.info(
                        "【读取预取】已提交=%d，已完成=%d，等待任务=%d，等待字节=%d，峰值等待字节=%d，耗时=%.1f秒",
                        submitted,
                        completed,
                        len(pending),
                        pending_bytes,
                        peak_pending_bytes,
                        now - started_at,
                    )
                    last_progress_at = now
            LOGGER.info(
                "【读取预取】完成：对象=%d，读取字节=%d，峰值等待字节=%d，耗时=%.1f秒",
                completed,
                completed_bytes,
                peak_pending_bytes,
                time.monotonic() - started_at,
            )
        finally:
            for _, _, future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def read_capture_bytes(self, capture: S3Capture) -> tuple[bytes, str]:
        """Conditionally read, close, length-check, and hash one S3 object."""

        if self.location.key(capture.source_ref) != capture.key:
            raise ValueError("S3 capture key is outside its source location")
        if (
            not isinstance(capture.size, int)
            or isinstance(capture.size, bool)
            or capture.size < 0
        ):
            raise ValueError("S3 capture size must be a non-negative integer")
        if not isinstance(capture.etag, str) or not capture.etag:
            raise ValueError("S3 capture ETag must be a non-empty string")
        response = self.client.get_object(
            Bucket=self.location.bucket,
            Key=capture.key,
            IfMatch=capture.etag,
        )
        body = response["Body"]
        payload = bytearray()
        digest = hashlib.sha256()
        try:
            while True:
                # Keep an inaccurate/changed ContentLength from causing an
                # unbounded bytearray before the metadata check below.  Read at
                # most one byte beyond the advertised size so we can still
                # detect an oversized response.
                remaining = capture.size - len(payload)
                chunk = body.read(
                    min(_READ_CHUNK_BYTES, remaining + 1)
                    if remaining >= 0
                    else _READ_CHUNK_BYTES
                )
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise TypeError("S3 response body must yield bytes")
                if len(payload) + len(chunk) > capture.size:
                    raise OSError("S3 capture size changed while it was being read")
                payload.extend(chunk)
                digest.update(chunk)
        finally:
            body.close()

        content_length = response.get("ContentLength")
        response_etag = response.get("ETag")
        if (
            not isinstance(content_length, int)
            or isinstance(content_length, bool)
            or content_length != capture.size
            or len(payload) != capture.size
        ):
            raise OSError("S3 capture size changed while it was being read")
        if not isinstance(response_etag, str) or response_etag != capture.etag:
            raise OSError("S3 capture ETag changed while it was being read")
        return bytes(payload), digest.hexdigest()


__all__ = [
    "InputFormat",
    "S3Capture",
    "S3CaptureSource",
    "S3Location",
    "create_s3_client",
    "parse_s3_uri",
]
