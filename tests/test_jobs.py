from pathlib import Path

import orjson
import pytest

from trajfoundry import jobs
from trajfoundry.pipeline import PipelineStats
from trajfoundry.validation import ValidationReport


def _write_capture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(
            {
                "path": "/v1/responses",
                "session_id": "session",
                "request_id": "turn-1",
                "captured_at": "2026-08-27T00:00:00Z",
                "status_code": 200,
                "is_stream": False,
                "request_body": {
                    "model": "gpt-test",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hello"}],
                        }
                    ],
                },
                "response_body": {
                    "id": "response-1",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "hello back"}],
                        }
                    ],
                },
            }
        )
    )


def test_run_job_validates_output_and_defaults_to_resume(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_capture(input_root / "capture.json")

    first = jobs.run_job(str(input_root), str(output_root))
    second = jobs.run_job(str(input_root), str(output_root))

    assert first.validation.valid
    assert first.accepted == 1
    assert first.stats.discovered == 1
    assert second.stats.reused == 1
    assert second.validation.valid


def test_run_job_warns_for_empty_input(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()

    with caplog.at_level("WARNING"):
        result = jobs.run_job(input_root, tmp_path / "output")

    assert result.validation.valid
    assert "no capture files" in caplog.text


def test_run_job_warns_when_every_capture_is_quarantined(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "broken.json").write_bytes(b"not JSON")

    with caplog.at_level("WARNING"):
        result = jobs.run_job(input_root, tmp_path / "output")

    assert result.validation.valid
    assert result.stats.parse_failures == 1
    assert result.quarantined_records == 1
    assert "quarantined 1 capture(s)" in caplog.text
    assert "no accepted trajectories" in caplog.text


def test_run_job_raises_with_validation_report(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = ValidationReport(
        valid=False,
        errors=["manifest files[0] is missing"],
        counts={},
    )

    monkeypatch.setattr(jobs, "normalize", lambda config: PipelineStats())
    monkeypatch.setattr(jobs, "validate_output", lambda root: expected)

    with pytest.raises(jobs.JobValidationError) as error:
        jobs.run_job("input", "output")

    assert error.value.report is expected
    assert "manifest files[0] is missing" in str(error.value)


def test_run_job_rejects_unknown_input_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported input format"):
        jobs.run_job(tmp_path / "input", tmp_path / "output", "unknown")  # type: ignore[arg-type]
