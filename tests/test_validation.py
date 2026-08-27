from pathlib import Path

import orjson
import pytest

from trajfoundry.export import OutputSet
from trajfoundry.io import file_sha256
from trajfoundry.models import (
    AuditTag,
    Message,
    Metadata,
    NormalizationAudit,
    QuarantineRecord,
    ServerToolCall,
    TrajectoryNode,
)
from trajfoundry.quality import enrich_trajectory
from trajfoundry.validation import validate_output


def _trajectory(source: str, *, complete: bool) -> TrajectoryNode:
    messages = [Message(role="user", content=f"request from {source}")]
    if complete:
        messages.append(Message(role="assistant", content="done", reasoning_content=""))
    return enrich_trajectory(
        TrajectoryNode(
            messages=messages,
            tools=[],
            source=source,
            metadata=Metadata(source_file=source),
        )
    )


def _origin(source: str) -> dict[str, object]:
    return {
        "source_ref": source,
        "sha256": "a" * 64,
        "captured_at": "2026-08-27T00:00:00Z",
        "disposition": "pass",
        "reason_codes": [],
    }


def _write_output(root: Path) -> None:
    accepted = _trajectory("accepted.json", complete=True)
    quarantined = _trajectory("quarantined.json", complete=False)
    record = QuarantineRecord(
        source_ref="invalid.json",
        sha256="b" * 64,
        normalization_audit=NormalizationAudit(
            tag=AuditTag.EXCLUDED,
            reason_codes=["capture_invalid"],
        ),
    )
    output = OutputSet(root)
    output.stats.input_files = 4
    output.write_trajectory(
        accepted,
        [_origin("accepted.json"), _origin("accepted-copy.json")],
    )
    output.write_trajectory(quarantined, [_origin("quarantined.json")])
    output.write_record(record)
    output.close(input_root="/input", config_hash="config")


def _refresh_manifest_entry(root: Path, relative: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    target = root / relative
    for entry in manifest["files"]:
        if entry["path"] == relative:
            entry["bytes"] = target.stat().st_size
            entry["sha256"] = file_sha256(target)
            break
    manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS))


def _manifest_path(root: Path, fragment: str) -> Path:
    manifest = orjson.loads((root / "manifest.json").read_bytes())
    relative = next(
        entry["path"] for entry in manifest["files"] if fragment in entry["path"]
    )
    return root / relative


def _mutate_accepted(root: Path, mutate: object) -> None:
    shard = _manifest_path(root, "/accepted/trajectories-")
    value = orjson.loads(shard.read_bytes())
    mutate(value)  # type: ignore[operator]
    shard.write_bytes(orjson.dumps(value, option=orjson.OPT_SORT_KEYS) + b"\n")
    _refresh_manifest_entry(root, shard.relative_to(root).as_posix())


def test_validate_output_accepts_complete_export(tmp_path: Path) -> None:
    _write_output(tmp_path)

    report = validate_output(tmp_path)

    assert report.valid
    assert report.errors == []
    assert report.counts["accepted"] == 1
    assert report.counts["quarantined_trajectories"] == 1
    assert report.counts["quarantined_records"] == 1
    assert report.counts["excluded_records"] == 1
    assert report.counts["duplicate_trajectories"] == 1
    assert report.counts["lineage_records"] == 2


def test_validate_output_reports_integrity_and_json_without_content(
    tmp_path: Path,
) -> None:
    _write_output(tmp_path)
    shard = _manifest_path(tmp_path, "/accepted/trajectories-")
    with shard.open("ab") as handle:
        handle.write(b'{"authorization":"TOP-SECRET"\n')

    report = validate_output(tmp_path)

    assert not report.valid
    assert any("byte count does not match" in error for error in report.errors)
    assert any("sha256 does not match" in error for error in report.errors)
    assert any("is not valid JSON" in error for error in report.errors)
    assert "TOP-SECRET" not in "\n".join(report.errors)


def test_validate_output_rejects_non_strict_accepted_sample(tmp_path: Path) -> None:
    _write_output(tmp_path)
    shard = _manifest_path(tmp_path, "/accepted/trajectories-")
    payload = orjson.loads(shard.read_bytes())
    payload["normalization_audit"]["tag"] = "quarantined"
    shard.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")
    _refresh_manifest_entry(tmp_path, shard.relative_to(tmp_path).as_posix())

    report = validate_output(tmp_path)

    assert not report.valid
    assert any("not a strict accepted sample" in error for error in report.errors)


def test_validate_output_rejects_type_correct_stale_derived_fields(
    tmp_path: Path,
) -> None:
    _write_output(tmp_path)
    _mutate_accepted(tmp_path, lambda row: row.__setitem__("total_rounds", 999))

    report = validate_output(tmp_path)

    assert not report.valid
    assert any("stale or inconsistent derived" in error for error in report.errors)


def test_validate_output_checks_lineage_contract(tmp_path: Path) -> None:
    _write_output(tmp_path)
    path = _manifest_path(tmp_path, "/lineage.jsonl")
    rows = [orjson.loads(line) for line in path.read_bytes().splitlines()]
    rows[0]["origin_count"] += 1
    path.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    _refresh_manifest_entry(tmp_path, path.relative_to(tmp_path).as_posix())

    report = validate_output(tmp_path)

    assert not report.valid
    assert any("violates the lineage contract" in error for error in report.errors)


def test_validate_output_checks_manifest_counts(tmp_path: Path) -> None:
    _write_output(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    manifest["counts"]["accepted"] = 99
    manifest_path.write_bytes(orjson.dumps(manifest))

    report = validate_output(tmp_path)

    assert not report.valid
    assert any("count accepted" in error for error in report.errors)


def test_validate_output_does_not_follow_manifest_path_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    _write_output(root)
    secret = tmp_path / "secret.jsonl"
    secret.write_bytes(b'{"authorization":"TOP-SECRET"}\n')
    manifest_path = root / "manifest.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    manifest["files"].append(
        {
            "path": "../secret.jsonl",
            "bytes": secret.stat().st_size,
            "sha256": file_sha256(secret),
        }
    )
    manifest_path.write_bytes(orjson.dumps(manifest))

    report = validate_output(root)

    assert not report.valid
    assert any("unsafe path" in error for error in report.errors)
    assert "TOP-SECRET" not in "\n".join(report.errors)


def test_validate_output_requires_complete_input_coverage(tmp_path: Path) -> None:
    _write_output(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    manifest["counts"]["input_files"] += 1
    manifest_path.write_bytes(orjson.dumps(manifest))

    report = validate_output(tmp_path)

    assert not report.valid
    assert any("input_files" in error for error in report.errors)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.pop("instructions"),
        lambda row: row.__setitem__("total_rounds", "1"),
        lambda row: row["messages"][0].__setitem__("reasoning_content", "leak"),
        lambda row: row["messages"][1].__setitem__("reasoning_details", ["bad"]),
        lambda row: row["messages"][1].__setitem__("tool_calls", []),
        lambda row: row["metadata"].__setitem__("source_file", "other.json"),
    ],
    ids=[
        "missing-required",
        "coerced-type",
        "role-field",
        "reasoning-details-type",
        "empty-tool-calls",
        "source-mismatch",
    ],
)
def test_validate_output_enforces_raw_trajectory_contract(
    tmp_path: Path, mutate: object
) -> None:
    _write_output(tmp_path)
    _mutate_accepted(tmp_path, mutate)

    report = validate_output(tmp_path)

    assert not report.valid
    assert any(
        "violates the TrajectoryNode contract" in error for error in report.errors
    )


def test_validate_output_rejects_nested_completeness_and_aggregate_tag(
    tmp_path: Path,
) -> None:
    _write_output(tmp_path)

    def mutate(row: dict[str, object]) -> None:
        child = dict(row)
        child["source"] = "child.json"
        child["metadata"] = {**row["metadata"], "source_file": "child.json"}  # type: ignore[arg-type]
        child["tool_defs_tag"] = "complete_with_incomplete_sub"
        row["sub_agent_trajectory"] = {"spawn-1": child}

    _mutate_accepted(tmp_path, mutate)
    report = validate_output(tmp_path)

    assert not report.valid
    assert any(
        "violates the TrajectoryNode contract" in error for error in report.errors
    )


def test_export_preserves_required_null_server_result(tmp_path: Path) -> None:
    node = _trajectory("server.json", complete=True)
    node.server_tool_calls = [
        ServerToolCall(
            name="web_search",
            id="server-1",
            arguments={"query": "x"},
            origin="response",
        )
    ]
    output = OutputSet(tmp_path)
    output.stats.input_files = 1
    output.write_trajectory(node, [_origin("server.json")])
    output.close(input_root="/input", config_hash="config")

    row = orjson.loads(_manifest_path(tmp_path, "/accepted/trajectories-").read_bytes())
    assert row["server_tool_calls"][0]["result"] is None
    assert validate_output(tmp_path).valid


def test_validate_output_rejects_missing_server_result_key(tmp_path: Path) -> None:
    node = _trajectory("server.json", complete=True)
    node.server_tool_calls = [
        ServerToolCall(
            name="web_search",
            id="server-1",
            arguments={},
            origin="response",
        )
    ]
    output = OutputSet(tmp_path)
    output.stats.input_files = 1
    output.write_trajectory(node, [_origin("server.json")])
    output.close(input_root="/input", config_hash="config")
    _mutate_accepted(tmp_path, lambda row: row["server_tool_calls"][0].pop("result"))

    report = validate_output(tmp_path)

    assert not report.valid
    assert any(
        "violates the TrajectoryNode contract" in error for error in report.errors
    )
