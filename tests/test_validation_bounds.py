import hashlib
import io
import logging

import orjson
import pytest

from trajfoundry import validation


class ChunkStream(io.BytesIO):
    def __init__(self, content, chunk_size=7, as_view=False):
        super().__init__(content)
        self.chunk_size = chunk_size
        self.as_view = as_view

    def read(self, size=-1):
        value = super().read(min(size, self.chunk_size))
        return memoryview(value) if self.as_view else value


def scan(content, *, limit=None, chunk_size=7, as_view=False):
    stream = ChunkStream(content, chunk_size, as_view)
    observed = validation._Observed()
    errors = validation._ErrorCollector()
    result = validation._validate_jsonl_stream(
        stream,
        kind=None,
        file_index=0,
        relative_path="test.jsonl",
        allow_legacy_metadata=False,
        observed=observed,
        errors=errors,
        max_jsonl_row_bytes=limit,
    )
    assert stream.closed
    assert result.complete
    assert result.bytes_read == len(content)
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    return observed, errors


@pytest.mark.parametrize("trailing_newline", [True, False])
def test_oversized_row_is_drained_without_losing_hash_or_next_rows(
    trailing_newline, monkeypatch
):
    decoded_sizes = []
    original = validation._validate_jsonl_row

    def record_size(raw_line, **kwargs):
        decoded_sizes.append(len(raw_line))
        return original(raw_line, **kwargs)

    monkeypatch.setattr(validation, "_validate_jsonl_row", record_size)
    oversized = b'{"secret":"' + b"x" * 150 + b'"}'
    content = oversized + b"\n{}\n" + oversized + (b"\n" if trailing_newline else b"")
    observed, errors = scan(content, limit=20)
    assert decoded_sizes == [3]
    assert observed.counts["jsonl_rows"] == 3
    assert errors.total == 2
    assert all("exceeds" in message for message in errors.messages)
    assert "secret" not in " ".join(errors.messages)


def test_row_limit_includes_newline_and_accepts_exact_boundary():
    observed, errors = scan(b'{"x":1}\n{}', limit=8)
    assert observed.counts["jsonl_rows"] == 2
    assert errors.total == 0
    _, errors = scan(b'{"x":1}\n', limit=7)
    assert errors.total == 1


def test_total_file_size_is_not_a_row_limit_and_memoryview_is_supported():
    observed, errors = scan(b"{}\n" * 20, limit=3, chunk_size=5, as_view=True)
    assert observed.counts["jsonl_rows"] == 20
    assert errors.total == 0


def test_legacy_default_does_not_introduce_row_rejections():
    observed, errors = scan(b'{"x":"' + b"a" * 200 + b'"}\n')
    assert observed.counts["jsonl_rows"] == 1
    assert errors.total == 0


def manifest(files):
    return orjson.dumps(
        {
            "schema_version": "trajfoundry-v4",
            "created_at": "2026-10-07",
            "input_root": "/input",
            "input_format": "tokenplan",
            "config_hash": "test",
            "token_estimator": "test",
            "counts": {
                "accepted": 0,
                "quarantined_trajectories": 0,
                "quarantined_records": 0,
                "excluded_records": 0,
                "duplicate_trajectories": 0,
                "input_files": 0,
                "reason_counts": {},
                "skipped_inputs": 0,
                "skip_reason_counts": {},
            },
            "files": [
                {"path": path, "bytes": size, "sha256": "0" * 64}
                for path, size in files
            ],
        }
    )


class SchedulingBackend:
    def __init__(self, files):
        self.files = files

    def iter_generated_jsonl_paths(self, generation_ids):
        return [path for path, _ in self.files]


def run_scheduling(
    monkeypatch, files, *, byte_limit, row_limit=None, timeout_first=False
):
    inflight = []
    windows = []
    completions = []

    def validate_file(backend, **kwargs):
        assert kwargs["max_jsonl_row_bytes"] == row_limit
        observed = validation._Observed()
        observed.counts["checked_files"] = 1
        return observed, validation._ErrorCollector()

    class FakeFuture:
        def __init__(self, function, item):
            self.function = function
            self.item = item
            self.waits = 0

        def result(self, timeout=None):
            self.waits += 1
            if timeout_first and self.waits == 1:
                raise TimeoutError()
            inflight.remove(self.item)
            completions.append(self.item[0])
            return self.function(self.item)

        def done(self):
            return False

    class FakeExecutor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 16

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, item):
            inflight.append(item)
            windows.append([(entry.path, entry.bytes) for _, entry, _ in inflight])
            return FakeFuture(function, item)

    monkeypatch.setattr(validation, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(validation, "_validate_manifest_file", validate_file)
    report = validation.validate_output_backend(
        manifest(files),
        SchedulingBackend(files),
        max_workers=16,
        max_pending_bytes=byte_limit,
        max_jsonl_row_bytes=row_limit,
    )
    assert report.valid, report.errors
    assert not inflight
    assert completions == list(range(len(files)))
    return windows


def test_large_files_use_byte_budget_and_oversized_file_runs_alone(monkeypatch):
    windows = run_scheduling(
        monkeypatch,
        [
            ("a.jsonl", 60),
            ("b.jsonl", 60),
            ("huge.jsonl", 200),
            ("c.jsonl", 40),
            ("lineage.jsonl", 5),
        ],
        byte_limit=100,
    )
    for window in windows:
        assert sum(size for _, size in window) <= 100 or len(window) == 1
        if any(path == "huge.jsonl" for path, _ in window):
            assert len(window) == 1


def test_small_files_keep_full_worker_window(monkeypatch):
    files = [(f"small-{index}.jsonl", 1) for index in range(40)] + [
        ("lineage.jsonl", 1)
    ]
    windows = run_scheduling(monkeypatch, files, byte_limit=100)
    assert max(map(len, windows)) == 32


def test_large_lineage_is_weighted_by_row_bound_not_rejected(monkeypatch):
    windows = run_scheduling(
        monkeypatch,
        [("a.jsonl", 10), ("lineage.jsonl", 1000), ("b.jsonl", 10)],
        byte_limit=100,
        row_limit=20,
    )
    assert max(map(len, windows)) == 3


def test_waiting_for_large_validation_file_emits_progress(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="trajfoundry.validation")
    run_scheduling(
        monkeypatch,
        [("a.jsonl", 60), ("lineage.jsonl", 5)],
        byte_limit=100,
        timeout_first=True,
    )
    assert any("在途原始字节=" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "kwargs",
    [{"max_pending_bytes": 0}, {"max_jsonl_row_bytes": 0}, {"max_jsonl_row_bytes": -1}],
)
def test_invalid_memory_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        validation.validate_output_backend(b"{}", SchedulingBackend([]), **kwargs)
