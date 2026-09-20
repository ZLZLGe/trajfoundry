from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from trajfoundry import pipeline
from trajfoundry.pipeline import PipelineConfig, normalize
from trajfoundry.validation import validate_output


def _envelope() -> dict[str, object]:
    request_body = {
        "model": "gpt-deepinfra",
        "client_metadata": {"session_id": "deepinfra-session"},
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }
    response_body = {
        "id": "response-deepinfra",
        "status": "completed",
        "model": "gpt-deepinfra",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
    }
    return {
        "request_id": "deepinfra-request",
        "request_time": "2026-09-14T01:02:03Z",
        "request": {
            "method": "POST",
            "path": "/v1/responses",
            "headers": {"x-user-id": "deepinfra-user"},
            "body": orjson.dumps(request_body).decode(),
            "body_truncated": False,
        },
        "response": {
            "status_code": 200,
            "headers": {},
            "body": orjson.dumps(response_body).decode(),
            "body_truncated": False,
        },
        "access_log": {
            "path": "/v1/responses",
            "request_id": "deepinfra-request",
            "response_code": 200,
        },
    }


def _manifest_file(root: Path, suffix: str) -> Path:
    manifest = orjson.loads((root / "manifest.json").read_bytes())
    if "trajectories-" in suffix:
        [entry] = [
            item for item in manifest["files"] if item["path"] != "lineage.jsonl"
        ]
    else:
        [entry] = [item for item in manifest["files"] if item["path"].endswith(suffix)]
    return root / entry["path"]


def test_deepinfra_pipeline_adapts_json_object_and_publishes_source_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "dt=2026-09-14" / "capture.json"
    source.parent.mkdir(parents=True)
    envelope = _envelope()
    source.write_bytes(orjson.dumps(envelope))
    (source.parent / "ignored.jsonl").write_bytes(b"{}\n")

    observed: list[object] = []
    real_adapter = pipeline.adapt_deepinfra_envelope

    def adapt(value: object) -> dict[str, object]:
        observed.append(value)
        return real_adapter(value)

    monkeypatch.setattr(pipeline, "adapt_deepinfra_envelope", adapt)

    stats = normalize(
        PipelineConfig(
            input_root=input_root,
            input_format="deepinfra",
            output_root=output_root,
        )
    )

    assert observed == [envelope]
    assert stats.discovered == 1
    assert stats.parsed == 1
    assert stats.parse_failures == 0

    manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    assert manifest["input_format"] == "deepinfra"
    assert manifest["counts"]["accepted"] == 1

    accepted_path = _manifest_file(output_root, "/accepted/trajectories-00000.jsonl")
    accepted = orjson.loads(accepted_path.read_bytes().splitlines()[0])
    assert accepted["metadata"] == {
        "source_file": "capture.json",
        "source_name": "deepinfra",
        "line_no": 0,
        "created_at": "2026-09-14T01:02:03Z",
        "model": "gpt-deepinfra",
        "user_id": "deepinfra-user",
        "session_id": "deepinfra-session",
        "sub_session_id": 0,
        "source_type": "api-router",
        "specific_source": "deep-infra",
    }
    assert validate_output(output_root).valid


@pytest.mark.parametrize(
    "reason_code",
    ["invalid_deepinfra_envelope", "deepinfra_incomplete_envelope"],
)
def test_deepinfra_adapter_errors_keep_code_and_outer_request_context(
    monkeypatch, tmp_path: Path, reason_code: str
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "capture.json"
    input_root.mkdir()
    source.write_bytes(
        orjson.dumps(
            {
                "request": {"path": "/v1/responses?beta=true"},
                "request_time": "2026-09-14T01:02:03Z",
            }
        )
    )

    def reject(value: object) -> dict[str, object]:
        del value
        raise pipeline.DeepInfraError(reason_code, "incomplete test envelope")

    monkeypatch.setattr(pipeline, "adapt_deepinfra_envelope", reject)

    stats = normalize(
        PipelineConfig(
            input_root=input_root,
            input_format="deepinfra",
            output_root=output_root,
        )
    )

    assert stats.discovered == 1
    assert stats.parsed == 0
    assert stats.parse_failures == 1
    manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    assert manifest["counts"]["quarantined_records"] == 1
    assert manifest["counts"]["reason_counts"] == {reason_code: 1}
    assert [entry["path"] for entry in manifest["files"]] == ["lineage.jsonl"]
    assert validate_output(output_root).valid
