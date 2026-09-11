from __future__ import annotations

from io import BytesIO
from pathlib import Path

import orjson

from trajfoundry.export import OutputSet
from trajfoundry.models import Message, Metadata, TrajectoryNode
from trajfoundry.quality import enrich_trajectory
from trajfoundry.validation import (
    ValidationBackend,
    ValidationFile,
    validate_output_backend,
)


class _TrackingStream(BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.bytes_returned = 0

    def read(self, size: int = -1) -> bytes:
        chunk = super().read(size)
        self.bytes_returned += len(chunk)
        return chunk


class _FailingStream(_TrackingStream):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self._first_read = True

    def read(self, size: int = -1) -> bytes:
        if not self._first_read:
            raise RuntimeError("TOP-SECRET backend response")
        self._first_read = False
        return super().read(max(1, len(self.getbuffer()) // 2))


class _FailingCloseStream(_TrackingStream):
    def close(self) -> None:
        super().close()
        raise RuntimeError("TOP-SECRET backend response")


class _MemoryBackend(ValidationBackend):
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.discovered = set(objects)
        self.reported_sizes: dict[str, int] = {}
        self.fail_open: set[str] = set()
        self.fail_read: set[str] = set()
        self.fail_close: set[str] = set()
        self.fail_listing = False
        self.streams: list[_TrackingStream] = []

    def open_file(self, relative_path: str) -> ValidationFile | None:
        if relative_path in self.fail_open:
            raise RuntimeError("TOP-SECRET backend response")
        payload = self.objects.get(relative_path)
        if payload is None:
            return None
        if relative_path in self.fail_read:
            stream_type = _FailingStream
        elif relative_path in self.fail_close:
            stream_type = _FailingCloseStream
        else:
            stream_type = _TrackingStream
        stream = stream_type(payload)
        self.streams.append(stream)
        return ValidationFile(
            size=self.reported_sizes.get(relative_path, len(payload)),
            stream=stream,
        )

    def iter_generated_jsonl_paths(self, generation_ids: frozenset[str]) -> set[str]:
        del generation_ids
        if self.fail_listing:
            raise RuntimeError("TOP-SECRET backend response")
        return self.discovered


def _build_output(root: Path) -> tuple[bytes, dict[str, bytes]]:
    source = "accepted.json"
    trajectory = enrich_trajectory(
        TrajectoryNode(
            messages=[
                Message(role="user", content="request"),
                Message(role="assistant", content="done", reasoning_content=""),
            ],
            tools=[],
            source=source,
            metadata=Metadata(source_file=source),
        )
    )
    output = OutputSet(root)
    output.stats.input_files = 1
    output.write_trajectory(
        trajectory,
        [
            {
                "source_ref": source,
                "sha256": "a" * 64,
                "captured_at": "2026-08-27T00:00:00Z",
                "disposition": "pass",
                "reason_codes": [],
            }
        ],
    )
    output.close(input_root="s3://bucket/input", config_hash="config")

    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = orjson.loads(manifest_bytes)
    objects = {
        entry["path"]: (root / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    return manifest_bytes, objects


def test_validate_output_backend_accepts_memory_streams_and_closes_them(
    tmp_path: Path,
) -> None:
    manifest_bytes, objects = _build_output(tmp_path)
    backend = _MemoryBackend(objects)

    report = validate_output_backend(manifest_bytes, backend)

    assert report.valid
    assert len(backend.streams) == len(objects)
    assert all(stream.closed for stream in backend.streams)
    assert sum(stream.bytes_returned for stream in backend.streams) == sum(
        map(len, objects.values())
    )


def test_validate_output_backend_checks_reported_and_streamed_size(
    tmp_path: Path,
) -> None:
    manifest_bytes, objects = _build_output(tmp_path)
    backend = _MemoryBackend(objects)
    first_path = next(iter(objects))
    backend.reported_sizes[first_path] = len(objects[first_path]) + 1

    report = validate_output_backend(manifest_bytes, backend)

    assert not report.valid
    assert any("byte count does not match" in error for error in report.errors)


def test_validate_output_backend_sanitizes_open_and_listing_errors(
    tmp_path: Path,
) -> None:
    manifest_bytes, objects = _build_output(tmp_path)
    backend = _MemoryBackend(objects)
    backend.fail_open.add(next(iter(objects)))
    backend.fail_listing = True

    report = validate_output_backend(manifest_bytes, backend)

    assert not report.valid
    assert any("could not be opened" in error for error in report.errors)
    assert any("could not be enumerated" in error for error in report.errors)
    assert "TOP-SECRET" not in "\n".join(report.errors)
    assert all(stream.closed for stream in backend.streams)


def test_validate_output_backend_sanitizes_read_errors_and_closes_stream(
    tmp_path: Path,
) -> None:
    manifest_bytes, objects = _build_output(tmp_path)
    backend = _MemoryBackend(objects)
    backend.fail_read.add(next(iter(objects)))

    report = validate_output_backend(manifest_bytes, backend)

    assert not report.valid
    assert any("could not be read completely" in error for error in report.errors)
    assert "TOP-SECRET" not in "\n".join(report.errors)
    assert all(stream.closed for stream in backend.streams)


def test_validate_output_backend_sanitizes_close_errors(tmp_path: Path) -> None:
    manifest_bytes, objects = _build_output(tmp_path)
    backend = _MemoryBackend(objects)
    backend.fail_close.add(next(iter(objects)))

    report = validate_output_backend(manifest_bytes, backend)

    assert not report.valid
    assert any("could not be read completely" in error for error in report.errors)
    assert "TOP-SECRET" not in "\n".join(report.errors)
    assert all(stream.closed for stream in backend.streams)


def test_validate_output_backend_detects_unlisted_generated_jsonl(
    tmp_path: Path,
) -> None:
    manifest_bytes, objects = _build_output(tmp_path)
    backend = _MemoryBackend(objects)
    backend.discovered.add(
        "generations/00000000000000000000000000000000/accepted/trajectories-99999.jsonl"
    )

    report = validate_output_backend(manifest_bytes, backend)

    assert not report.valid
    assert any("not listed in manifest" in error for error in report.errors)
