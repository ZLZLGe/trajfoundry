from pathlib import Path

import orjson
import pytest

from trajfoundry.export import OutputSet
from trajfoundry.feedback import feedback_fields, unix_ms
from trajfoundry.models import (
    AuditTag,
    Message,
    Metadata,
    NormalizationAudit,
    QuarantineRecord,
    TrajectoryNode,
)
from trajfoundry.quality import enrich_trajectory
from trajfoundry.validation import validate_output

FEEDBACK_KEYS = {
    "status",
    "started_at_ms",
    "completed_at_ms",
    "total_bytes",
    "record_count",
    "accepted_trajectory_count",
    "quarantined_trajectory_count",
    "parse_failure_count",
}


def write_batch(root: Path) -> dict:
    output = OutputSet(root)
    output.stats.input_files = 4
    for name, complete in (("accepted", True), ("quarantined", False)):
        messages = [Message(role="user", content="question")]
        if complete:
            messages.append(
                Message(role="assistant", content="answer", reasoning_content="")
            )
        node = enrich_trajectory(
            TrajectoryNode(
                messages=messages,
                tools=[],
                source=name,
                metadata=Metadata(source_file=name, session_id=name),
            )
        )
        output.write_trajectory(
            node,
            [
                {
                    "source_ref": name,
                    "sha256": "a" * 64,
                    "captured_at": "",
                }
            ],
        )
    for name in ("failed-one", "failed-two"):
        output.write_record(
            QuarantineRecord(
                source_ref=name,
                sha256="b" * 64,
                normalization_audit=NormalizationAudit(
                    tag=AuditTag.EXCLUDED,
                    reason_codes=["capture_parse_failed"],
                ),
            )
        )
    output.close(
        input_root="/synthetic",
        config_hash="config",
        started_at_ms=1000,
        parse_failure_count=2,
    )
    return orjson.loads((root / "manifest.json").read_bytes())


def test_feedback_counts_trajectories_and_excludes_lineage(tmp_path):
    manifest = write_batch(tmp_path)
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "diagnostics" / "input_diagnostics.ndjson").write_bytes(b"{}\n")
    trajectories = [
        item for item in manifest["files"] if item["path"] != "lineage.jsonl"
    ]
    assert manifest["status"] == "SUCCEEDED"
    assert manifest["record_count"] == 2
    assert manifest["accepted_trajectory_count"] == 1
    assert manifest["quarantined_trajectory_count"] == 1
    assert manifest["parse_failure_count"] == 2
    assert manifest["total_bytes"] == sum(
        (tmp_path / item["path"]).stat().st_size for item in trajectories
    )
    assert manifest["total_bytes"] < sum(item["bytes"] for item in manifest["files"])
    assert manifest["started_at_ms"] == 1000 <= manifest["completed_at_ms"] <= unix_ms()
    assert validate_output(tmp_path).valid


def test_empty_feedback_is_zero_even_with_lineage(tmp_path):
    output = OutputSet(tmp_path)
    output.close(input_root="/empty", config_hash="empty", started_at_ms=1000)
    manifest = orjson.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["record_count"] == manifest["total_bytes"] == 0
    assert (
        manifest["accepted_trajectory_count"]
        == manifest["quarantined_trajectory_count"]
        == 0
    )
    assert manifest["parse_failure_count"] == 0
    assert validate_output(tmp_path).valid


@pytest.mark.parametrize(
    "updates",
    [
        {"record_count": 3},
        {"accepted_trajectory_count": 0},
        {"quarantined_trajectory_count": 2},
        {"total_bytes": 0},
        {"started_at_ms": True},
        {"completed_at_ms": 999},
        {"parse_failure_count": -1},
        {"parse_failure_count": None},
        {"parse_failure_count": 999},
        {"status": None},
    ],
)
def test_invalid_feedback_is_rejected(tmp_path, updates):
    manifest = write_batch(tmp_path)
    manifest.update(updates)
    (tmp_path / "manifest.json").write_bytes(orjson.dumps(manifest))
    assert not validate_output(tmp_path).valid


def test_historical_manifest_without_feedback_remains_readable(tmp_path):
    manifest = write_batch(tmp_path)
    for key in FEEDBACK_KEYS:
        manifest.pop(key)
    (tmp_path / "manifest.json").write_bytes(orjson.dumps(manifest))
    assert validate_output(tmp_path).valid


@pytest.mark.parametrize("missing", sorted(FEEDBACK_KEYS))
def test_partial_feedback_is_rejected(tmp_path, missing):
    manifest = write_batch(tmp_path)
    manifest.pop(missing)
    (tmp_path / "manifest.json").write_bytes(orjson.dumps(manifest))
    assert not validate_output(tmp_path).valid


def test_status_without_feedback_is_rejected(tmp_path):
    manifest = write_batch(tmp_path)
    for key in FEEDBACK_KEYS - {"status"}:
        manifest.pop(key)
    (tmp_path / "manifest.json").write_bytes(orjson.dumps(manifest))
    assert not validate_output(tmp_path).valid


def test_feedback_rejects_noninteger_or_reversed_timestamps():
    from trajfoundry.export import ExportStats

    with pytest.raises(ValueError):
        feedback_fields(ExportStats(), total_bytes=True, started_at_ms=0)
    with pytest.raises(ValueError):
        feedback_fields(
            ExportStats(), total_bytes=0, started_at_ms=2, completed_at_ms=1
        )
