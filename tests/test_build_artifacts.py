from __future__ import annotations

import multiprocessing
import weakref
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from trajfoundry.build_budget import BuildMemoryError
from trajfoundry.canonical import canonical_json
from trajfoundry.models import (
    AuditIssue,
    AuditTag,
    FunctionCall,
    MediaMapping,
    Message,
    Severity,
    Snapshot,
    ToolCall,
    ToolDefinition,
    TrajectoryNode,
)
from trajfoundry.pipeline import (
    _contributor_leaf_warnings,
    _estimate_root_working_bytes,
    _MaterializationJob,
    _MaterializationLeaf,
    _materialize_root_worker,
    _merged_multimodal_file_mapping,
    _read_materialization_artifact,
    _trajectory_from_leaf,
)
from trajfoundry.quality import check_tool_calls, estimate_tokens
from trajfoundry.state import StateStore, _compress_payload


def _snapshot(index: int = 0) -> Snapshot:
    return Snapshot(
        source_path=f"source-{index}.json",
        source_sha256=str(index) * 64,
        session_id="session",
        thread_id=f"thread-{index}",
        captured_at=f"2026-10-01T00:00:{index:02d}Z",
        provider="openai",
        operation="responses",
        outcome="success",
        user_id="user",
        history=[Message(role="user", content=f"question-{index}")],
        response=[Message(role="assistant", content="answer", reasoning_content="")],
        wire_complete=True,
        issues=[
            AuditIssue(
                code="test_warning",
                stage="test",
                severity=Severity.WARNING,
                detail="round-trip must preserve strict enums",
            )
        ],
    )


def test_worker_artifact_preserves_enriched_enums_across_processes(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.sqlite"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    jobs: list[_MaterializationJob] = []
    with StateStore(state_path) as state:
        for index in range(2):
            snapshot = _snapshot(index)
            state.put_snapshot(snapshot)
            jobs.append(
                _MaterializationJob(
                    state_path=str(state_path),
                    root_index=index,
                    is_subagent=False,
                    leaves={
                        index: _MaterializationLeaf(
                            source_path=snapshot.source_path,
                            contributor_paths=(snapshot.source_path,),
                            issues=(),
                        )
                    },
                    graph_issues={},
                    edges_by_parent={},
                    origin_paths=(snapshot.source_path,),
                    artifact_dir=str(artifact_dir),
                )
            )
    # Use real child processes: inline mocks previously missed this exact
    # enriched-audit serialization path in production.
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        outputs = list(pool.map(_materialize_root_worker, jobs))
    for job, (_, artifact, representative, origins) in zip(jobs, outputs, strict=True):
        assert isinstance(artifact, str)
        node, metadata = _read_materialization_artifact(artifact)
        assert metadata == representative
        assert origins == job.origin_paths
        assert node.normalization_audit is not None
        assert isinstance(node.normalization_audit.tag, AuditTag)
        assert all(
            isinstance(issue.severity, Severity)
            for issue in node.normalization_audit.issues
        )
        assert not Path(artifact).exists()
        _, inline, _, _ = _materialize_root_worker(replace(job, artifact_dir=None))
        assert isinstance(inline, TrajectoryNode)
        assert node.model_dump() == inline.model_dump()


def test_invalid_artifact_is_not_deleted_before_validation(tmp_path: Path) -> None:
    artifact = tmp_path / "bad.json.zst"
    node = _trajectory_from_leaf(_snapshot(), [])
    document = node.model_dump(mode="json")
    document["normalization_audit"]["tag"] = "not-a-tag"
    artifact.write_bytes(
        _compress_payload(
            canonical_json(
                {
                    "trajectory": document,
                    "representative": {
                        "source_path": "source.json",
                        "source_sha256": "hash",
                        "captured_at": "",
                    },
                }
            )
        )
    )
    with pytest.raises(ValidationError):
        _read_materialization_artifact(str(artifact))
    assert artifact.is_file()


@pytest.mark.parametrize(
    "body_sizes,metadata_sizes",
    [
        ({}, {"source-0.json": 100}),
        ({"source-0.json": 100}, {}),
        ({"source-0.json": None}, {"source-0.json": 100}),
        ({"source-0.json": 100}, {"source-0.json": None}),
    ],
    ids=["missing-body", "missing-projection", "unknown-body", "unknown-projection"],
)
def test_root_admission_does_not_treat_unknown_sizes_as_zero(
    tmp_path: Path, monkeypatch, body_sizes, metadata_sizes
) -> None:
    path = tmp_path / "state.sqlite"
    job = _MaterializationJob(
        state_path=str(path),
        root_index=0,
        is_subagent=False,
        leaves={0: _MaterializationLeaf("source-0.json", ("source-0.json",), ())},
        graph_issues={},
        edges_by_parent={},
        origin_paths=("source-0.json",),
    )
    with StateStore(path) as state:
        monkeypatch.setattr(
            state, "snapshot_payload_bytes_for_paths", lambda _: body_sizes
        )
        monkeypatch.setattr(
            state, "snapshot_metadata_bytes_for_paths", lambda _: metadata_sizes
        )
        with pytest.raises(
            BuildMemoryError, match="unknown or a source/projection is missing"
        ):
            _estimate_root_working_bytes(state, job)


def test_contributor_histories_are_released_while_iterating() -> None:
    leaf = _snapshot(40)
    message_refs: list[weakref.ReferenceType[Message]] = []

    def contributors():
        for index in range(20):
            # Allow the previous iterator item to remain in for-loop locals,
            # but not a history retained from multiple contributors ago.
            if index >= 3:
                assert message_refs[index - 3]() is None
            snapshot = _snapshot(index)
            message_refs.append(weakref.ref(snapshot.history[0]))
            yield snapshot
            del snapshot

    node = _trajectory_from_leaf(leaf, contributors())
    assert node.messages == [*leaf.history, *leaf.response]
    assert all(reference() is None for reference in message_refs)


def test_projected_contributor_metadata_preserves_complete_merge() -> None:
    leaf = _snapshot(2)
    earlier = _snapshot(1).model_copy(
        update={
            "instructions": "earlier instruction",
            "model": "earlier model",
            "harness": "earlier harness",
            "termination": "earlier termination",
            "user_id": "earlier user",
            "tools": [ToolDefinition(name="earlier-tool")],
            "multimodal_file_mapping": [
                MediaMapping(part_id="old", object_name="old.png")
            ],
        }
    )
    expected = _trajectory_from_leaf(leaf, [earlier, leaf])
    actual = _trajectory_from_leaf(
        leaf,
        (
            item.model_copy(update={"history": [], "response": []})
            for item in (earlier, leaf)
        ),
    )
    assert actual.model_dump() == expected.model_dump()
    warning_codes = {issue.code for issue in actual.normalization_audit.issues}
    for field in ("instructions", "model", "harness", "termination", "user_id"):
        assert f"contributor_{field}_changed" in warning_codes


def test_contributor_warning_helper_accepts_one_shot_iterator() -> None:
    leaf = _snapshot()
    contributor = _snapshot(1).model_copy(
        update={"model": "old", "termination": "old", "harness": "old"}
    )
    codes = {
        issue.code for issue in _contributor_leaf_warnings(leaf, iter([contributor]))
    }
    assert codes == {
        "contributor_model_changed",
        "contributor_harness_changed",
        "contributor_termination_changed",
    }


def test_media_merge_keeps_capture_order_without_buffering_snapshots() -> None:
    leaf = _snapshot(4).model_copy(
        update={
            "multimodal_file_mapping": [
                MediaMapping(part_id="leaf", object_name="authoritative.png")
            ]
        }
    )
    earliest = _snapshot(0).model_copy(
        update={
            "multimodal_file_mapping": [
                MediaMapping(part_id="shared", object_name="earliest.png"),
                MediaMapping(part_id="first", object_name="first.png"),
            ]
        }
    )
    later = _snapshot(2).model_copy(
        update={
            "multimodal_file_mapping": [
                MediaMapping(part_id="leaf", object_name="ignored.png"),
                MediaMapping(part_id="later", object_name="later.png"),
                MediaMapping(part_id="shared", object_name="ignored.png"),
            ]
        }
    )
    merged = _merged_multimodal_file_mapping(leaf, iter([later, earliest]))
    assert [(item.part_id, item.object_name) for item in merged] == [
        ("leaf", "authoritative.png"),
        ("shared", "earliest.png"),
        ("first", "first.png"),
        ("later", "later.png"),
    ]


def test_quality_mismatch_limit_preserves_full_counts() -> None:
    calls = [
        ToolCall(
            id=f"call-{index}", function=FunctionCall(name="missing", arguments={})
        )
        for index in range(75)
    ]
    calls.extend(
        ToolCall(
            id=f"schema-{index}",
            function=FunctionCall(name="known", arguments={"unexpected": True}),
        )
        for index in range(25)
    )
    check, missing = check_tool_calls(
        [Message(role="assistant", content="", reasoning_content="", tool_calls=calls)],
        [ToolDefinition(name="known", parameters={"type": "object", "properties": {}})],
    )
    assert check.total_calls == check.mismatch_calls == 100
    assert check.undefined_tool_calls == 75
    assert check.extra_arg_calls == 25
    assert check.checked_calls == 25
    assert check.mismatches_truncated == 50
    assert len(check.mismatches) == 50
    assert check.mismatches[0].tool_call_id == "call-0"
    assert check.mismatches[-1].tool_call_id == "call-49"
    assert missing == ["missing"]


@pytest.mark.parametrize("text", ["", "hello world", "中文 English_123，！", "x\n y😀"])
def test_incremental_token_count_preserves_previous_formula(text: str) -> None:
    from trajfoundry.quality import _TOKEN_PATTERN

    assert estimate_tokens(text) == len(_TOKEN_PATTERN.findall(text))
