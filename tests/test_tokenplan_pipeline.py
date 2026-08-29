from __future__ import annotations

from pathlib import Path

import orjson

from trajfoundry.pipeline import PipelineConfig, normalize
from trajfoundry.validation import validate_output


def _envelope(*, request_id: str, media: bool = False) -> dict:
    content: object = "hello"
    media_records: list[dict] = []
    if media:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "$media_ref:media_0"},
            },
            {"type": "text", "text": "describe"},
        ]
        media_records = [{"part_id": "media_0", "available": True}]
    return {
        "client_request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "protocol": "openai_chat",
            "public_model": "chat-test",
            "stream": False,
            "capture": {
                "body": {
                    "model": "chat-test",
                    "messages": [{"role": "user", "content": content}],
                },
                "status": "COMPLETE",
            },
            "media": media_records,
        },
        "client_response": {
            "capture": {
                "body": {
                    "id": "completion-1",
                    "object": "chat.completion",
                    "model": "chat-test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "done",
                            },
                        }
                    ],
                },
                "format": "ORIGINAL",
                "status": "COMPLETE",
            },
            "media": [],
        },
        "metadata": {
            "schema_version": "data_feedback_des.v1",
            "request_id": request_id,
            "session_id": "session-1",
            "task_id": "",
            "received_at_ms": 1000,
            "completed_at_ms": 1234,
            "http_status_code": 200,
            "data_quality": {
                "client_request_complete": True,
                "client_response_complete": True,
                "all_attachments_available": True,
                "truncated": False,
            },
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(value))


def _manifest_file(root: Path, suffix: str) -> Path:
    manifest = orjson.loads((root / "manifest.json").read_bytes())
    [entry] = [item for item in manifest["files"] if item["path"].endswith(suffix)]
    return root / entry["path"]


def test_tokenplan_pipeline_includes_test_and_accounts_for_media_skip(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_json(
        input_root / "dt=2026-08-30" / "session-1" / "req_good.json",
        _envelope(request_id="req_good"),
    )
    _write_json(
        input_root / "test" / "dt=2026-08-30" / "session-1" / "req_media.json",
        _envelope(request_id="req_media", media=True),
    )
    _write_json(
        input_root / "dt=2026-08-30" / "session-2" / "req_empty.json",
        {},
    )
    _write_json(input_root / "dt=2026-08-30" / "manifest.json", {"status": "ok"})

    stats = normalize(
        PipelineConfig(
            input_root=input_root,
            input_format="tokenplan",
            output_root=output_root,
        )
    )

    assert stats.discovered == 3
    assert stats.parsed == 1
    assert stats.parse_failures == 1
    assert stats.skipped_inputs == 1

    manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    assert manifest["schema_version"] == "trajfoundry-v3"
    assert manifest["input_format"] == "tokenplan"
    assert manifest["counts"]["input_files"] == 3
    assert manifest["counts"]["skipped_inputs"] == 1
    assert manifest["counts"]["skip_reason_counts"] == {"unsupported_media_capture": 1}

    accepted_path = _manifest_file(output_root, "/accepted/trajectories-00000.jsonl")
    accepted = orjson.loads(accepted_path.read_bytes().splitlines()[0])
    assert accepted["metadata"] == {
        "source_file": "req_good.json",
        "source_name": "tokenplan",
        "line_no": 0,
        "created_at": "1970-01-01T00:00:01.234Z",
    }

    records_path = _manifest_file(output_root, "/quarantine/records-00000.jsonl")
    record = orjson.loads(records_path.read_bytes().splitlines()[0])
    assert record["source_ref"].endswith("req_empty.json")
    assert record["normalization_audit"]["reason_codes"] == ["tokenplan_empty_envelope"]
    report = validate_output(output_root)
    assert report.valid
    assert report.counts["skipped_inputs"] == 1
