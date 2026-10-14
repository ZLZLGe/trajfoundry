"""Production feedback from synthetic end-to-end local and fake-S3 runs."""

from pathlib import Path
from typing import Any

import orjson
import pytest

from trajfoundry import jobs

from .test_s3_jobs import _capture, _MemoryS3Client


def _non_trajectory_capture(kind: str) -> bytes:
    capture = orjson.loads(_capture())
    capture["session_id"] = "non-trajectory-session"
    capture["request_id"] = "non-trajectory-turn"
    if kind == "count_tokens":
        capture["path"] = "/v1/messages/count_tokens"
        capture["request_body"] = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hello"}],
        }
        capture["response_body"] = {"input_tokens": 17}
    else:
        capture["status_code"] = 429
        capture["response_body"] = {
            "error": {"type": "rate_limit_error", "message": "synthetic failure"}
        }
    return orjson.dumps(capture)


def _quarantined_trajectory_capture() -> bytes:
    capture = orjson.loads(_capture())
    capture["session_id"] = "quarantined-session"
    capture["request_id"] = "quarantined-turn"
    capture["response_body"]["output"] = [
        {
            "type": "function_call",
            "call_id": "unanswered-call",
            "name": "undefined_tool",
            "arguments": "{}",
        }
    ]
    return orjson.dumps(capture)


def _assert_counts(
    manifest: dict[str, Any], *, quarantined_trajectories: int = 0, excluded: int = 1
) -> None:
    assert manifest["status"] == "SUCCEEDED"
    assert manifest["record_count"] == 1 + quarantined_trajectories
    assert manifest["accepted_trajectory_count"] == 1
    assert manifest["quarantined_trajectory_count"] == quarantined_trajectories
    assert manifest["record_count"] == (
        manifest["accepted_trajectory_count"] + manifest["quarantined_trajectory_count"]
    )
    assert manifest["parse_failure_count"] == 1
    assert manifest["counts"]["quarantined_records"] == 2
    assert manifest["counts"]["excluded_records"] == excluded
    assert manifest["parse_failure_count"] < manifest["counts"]["quarantined_records"]
    assert manifest["counts"]["input_files"] == 3 + quarantined_trajectories
    assert manifest["counts"]["skipped_inputs"] == 0


@pytest.mark.parametrize("record_kind", ["count_tokens", "api_error"])
def test_local_feedback_distinguishes_failed_and_parsed_records_on_resume(
    tmp_path: Path, record_kind: str
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    (input_root / "accepted.json").write_bytes(_capture())
    (input_root / "non-trajectory.json").write_bytes(
        _non_trajectory_capture(record_kind)
    )
    (input_root / "malformed.json").write_bytes(b"not valid JSON")

    first = jobs.run_job(input_root, output_root)
    first_manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    resumed = jobs.run_job(input_root, output_root, resume=True)
    resumed_manifest = orjson.loads((output_root / "manifest.json").read_bytes())

    assert first.validation.valid and resumed.validation.valid
    assert first.stats.parsed == 2
    assert first.stats.parse_failures == resumed.stats.parse_failures == 1
    assert resumed.stats.reused == 2
    for manifest in (first_manifest, resumed_manifest):
        _assert_counts(manifest, excluded=int(record_kind == "count_tokens"))
        trajectories = [
            item for item in manifest["files"] if item["path"] != "lineage.jsonl"
        ]
        assert len(trajectories) == 1
        assert manifest["total_bytes"] == sum(
            len((output_root / item["path"]).read_bytes()) for item in trajectories
        )
        assert manifest["total_bytes"] < sum(
            item["bytes"] for item in manifest["files"]
        )
    assert first_manifest["files"] == resumed_manifest["files"]


def test_s3_feedback_uses_trajectory_body_bytes_and_validated_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _MemoryS3Client()
    bucket = "synthetic-feedback"
    for filename, payload in (
        ("accepted.json", _capture()),
        ("quarantined.json", _quarantined_trajectory_capture()),
        ("non-trajectory.json", _non_trajectory_capture("count_tokens")),
        ("malformed.json", b"not valid JSON"),
    ):
        client.objects[(bucket, f"input/{filename}")] = payload
    diagnostic_key = "output/diagnostics/input_diagnostics.ndjson"
    client.objects[(bucket, diagnostic_key)] = b'{"synthetic":"diagnostic"}\n'
    monkeypatch.setattr(jobs, "create_s3_client", lambda **kwargs: client)
    clock = [1000]
    monkeypatch.setattr(jobs, "unix_ms", lambda: clock[0])
    original_validate = jobs.validate_output_backend
    validation_calls = []

    def validate_and_advance(manifest_bytes, backend, **kwargs):
        assert (bucket, "output/manifest.json") not in client.objects
        report = original_validate(manifest_bytes, backend, **kwargs)
        validation_calls.append(report.valid)
        clock[0] = 9000
        return report

    monkeypatch.setattr(jobs, "validate_output_backend", validate_and_advance)
    result = jobs.run_s3_job(
        f"s3://{bucket}/input/",
        f"s3://{bucket}/output/",
        "freerouter",
        "http://s3.example.invalid",
        workspace_parent=tmp_path,
        credentials_path=None,
        read_workers=1,
        parse_workers=1,
    )
    manifest = orjson.loads(client.objects[(bucket, "output/manifest.json")])
    assert result.validation.valid
    assert validation_calls == [True]
    _assert_counts(manifest, quarantined_trajectories=1)
    trajectories = [
        entry for entry in manifest["files"] if entry["path"] != "lineage.jsonl"
    ]
    assert len(trajectories) == 2
    bodies = [
        client.objects[(bucket, f"output/{entry['path']}")] for entry in trajectories
    ]
    assert {orjson.loads(body)["normalization_audit"]["tag"] for body in bodies} == {
        "pass",
        "quarantined",
    }
    assert manifest["total_bytes"] == sum(map(len, bodies))
    assert manifest["total_bytes"] < sum(
        len(client.objects[(bucket, f"output/{entry['path']}")])
        for entry in manifest["files"]
    )
    assert diagnostic_key in {key for _, key in client.objects}
    assert not any("diagnostic" in entry["path"] for entry in manifest["files"])
    assert manifest["started_at_ms"] == 1000
    assert manifest["completed_at_ms"] == 9000
    put_calls = [call for call in client.calls if call[0] == "put_object"]
    assert put_calls[-1] == ("put_object", "output/manifest.json")
