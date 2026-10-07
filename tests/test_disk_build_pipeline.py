"""Tiny functional integrations through the actual multiprocessing build pool."""

import hashlib
import os
import time
from pathlib import Path

import orjson
import pytest

from trajfoundry import pipeline
from trajfoundry.models import AuditIssue, Message, Severity, Snapshot
from trajfoundry.pipeline import PipelineConfig, PipelineStats
from trajfoundry.state import StateStore
from trajfoundry.validation import validate_output

_ORIGINAL_MATERIALIZE = pipeline._materialize_root_worker
_PID_LOG: Path | None = None


def _recording_materialize(job):
    assert _PID_LOG is not None
    descriptor = os.open(_PID_LOG, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
    finally:
        os.close(descriptor)
    return _ORIGINAL_MATERIALIZE(job)


def _raise_worker_failure(job):
    raise ValueError("synthetic private worker detail must not be copied into errors")


def _reordered_materialize(job):
    # A tiny synchronization delay forces completion out of submission order;
    # this is a functional scheduling test, not a throughput/load test.
    if job.root_index == 0:
        time.sleep(0.1)
    return _ORIGINAL_MATERIALIZE(job)


def _sources(source_name):
    question = Message(role="user", content="question")
    answer = Message(role="assistant", content="answer", reasoning_content="")
    next_question = Message(role="user", content="next")

    def capture(path, history, response, **updates):
        return Snapshot(
            source_path=path,
            source_sha256=hashlib.sha256(path.encode()).hexdigest(),
            source_name=source_name,
            session_id="session-1",
            thread_id="thread-1",
            user_id="user",
            provider="openai",
            operation="responses",
            outcome="success",
            captured_at="2026-10-07T00:00:00Z",
            history=history,
            response=response,
            wire_complete=True,
            termination="completed",
            issues=[
                AuditIssue(
                    code="synthetic_warning", stage="test", severity=Severity.WARNING
                )
            ],
        ).model_copy(update=updates)

    return [
        capture("short.json", [question], [answer]),
        capture(
            "left.json",
            [question, answer, next_question],
            [Message(role="assistant", content="left", reasoning_content="")],
        ),
        capture(
            "left-copy.json",
            [question, answer, next_question],
            [Message(role="assistant", content="left", reasoning_content="")],
        ),
        capture(
            "right.json",
            [question, answer, next_question],
            [Message(role="assistant", content="right", reasoning_content="")],
        ),
        capture(
            "other-session.json",
            [Message(role="user", content="other")],
            [answer],
            session_id="session-2",
        ),
    ]


@pytest.mark.parametrize("source_name", ["tokenplan", "freerouter"])
@pytest.mark.parametrize(
    "ephemeral", [False, True], ids=["wal-state", "exclusive-local-state"]
)
def test_serial_and_actual_process_pool_preserve_output_and_lineage(
    tmp_path, monkeypatch, source_name, ephemeral
):
    global _PID_LOG
    monkeypatch.setattr(pipeline, "_materialize_root_worker", _recording_materialize)
    snapshots = _sources(source_name)
    outputs = []
    for workers in (1, 2):
        root = tmp_path / f"workers-{workers}"
        root.mkdir()
        _PID_LOG = root / "worker-pids.txt"
        with StateStore(root / "state.sqlite", ephemeral=ephemeral) as state:
            with state.write_batch():
                for snapshot in reversed(snapshots):
                    state.put_snapshot(snapshot, endpoint="/v1/responses")
            stats = PipelineStats()
            pipeline._build_trajectories(state, stats, build_workers=workers)
            assert stats.sessions == 2
            assert stats.eligible_snapshots == 5
            assert stats.prefix_intermediates == 1
            assert stats.stored_trajectories == 3
            assert stats.skipped_inputs == 0
            stored = list(state.iter_trajectories())
            assert {
                origin["source_ref"] for _, _, origins in stored for origin in origins
            } == {snapshot.source_path for snapshot in snapshots}
            assert all(node.normalization_audit is not None for _, node, _ in stored)
            config = PipelineConfig(
                input_root=root / "input",
                output_root=root / "output",
                input_format=source_name,
            )
            pipeline._export(state, config, "test")
        assert validate_output(config.output_root, max_workers=2).valid
        manifest = orjson.loads((config.output_root / "manifest.json").read_bytes())
        outputs.append(
            {
                "counts": manifest["counts"],
                "files": {
                    entry["path"]: (config.output_root / entry["path"]).read_bytes()
                    for entry in manifest["files"]
                },
            }
        )
        pids = {int(value) for value in _PID_LOG.read_text().splitlines()}
        if workers == 1:
            assert pids == {os.getpid()}
        else:
            assert pids and os.getpid() not in pids
    assert outputs[0] == outputs[1]


def test_actual_worker_error_fails_closed_without_skipping_sources(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "_materialize_root_worker", _raise_worker_failure)
    with StateStore(tmp_path / "state.sqlite", ephemeral=True) as state:
        for snapshot in _sources("tokenplan"):
            state.put_snapshot(snapshot)
        with pytest.raises(
            pipeline.TrajectoryBuildError, match="worker-result"
        ) as failure:
            pipeline._build_trajectories(state, PipelineStats(), build_workers=2)
        assert "private worker detail" not in str(failure.value)
        assert not any(
            status == "skipped" for _, _, status, _, _, _ in state.capture_records()
        )
        assert state.capture_count() == 5
        assert not (tmp_path / "manifest.json").exists()


def test_parallel_duplicate_roots_preserve_shared_origin_disposition(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "_materialize_root_worker", _reordered_materialize)
    original = _sources("tokenplan")
    short = original[0].model_copy(update={"session_id": "", "issues": []})
    first = original[1].model_copy(
        update={"session_id": "", "issues": [], "thread_id": "first-thread"}
    )
    second = first.model_copy(
        update={
            "source_path": "duplicate.json",
            "source_sha256": "f" * 64,
            "thread_id": "second-thread",
            "issues": [
                AuditIssue(
                    code="synthetic_error", stage="test", severity=Severity.ERROR
                )
            ],
        }
    )
    results = []
    for workers in (1, 2):
        with StateStore(
            tmp_path / f"origins-{workers}.sqlite", ephemeral=True
        ) as state:
            for snapshot in (short, first, second):
                state.put_snapshot(snapshot)
            stats = PipelineStats()
            pipeline._build_trajectories(state, stats, build_workers=workers)
            assert stats.stored_trajectories == 1
            assert stats.skipped_inputs == 0
            results.append(
                [
                    (identifier, node.model_dump(mode="json"), origins)
                    for identifier, node, origins in state.iter_trajectories()
                ]
            )
    assert results[0] == results[1]
