from __future__ import annotations

from pathlib import Path

import orjson

from trajfoundry.pipeline import PipelineConfig, normalize
from trajfoundry.state import StateStore
from trajfoundry.validation import validate_output


def _envelope(*, object_name: str, request_id: str = "req-media") -> dict:
    return {
        "client_request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "protocol": "openai_chat",
            "stream": False,
            "capture": {
                "body": {
                    "model": "chat-test",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "$media_ref:media_0"},
                                },
                                {"type": "text", "text": "describe"},
                            ],
                        }
                    ],
                },
                "status": "COMPLETE",
            },
            "media": [
                {
                    "part_id": "media_0",
                    "object_name": object_name,
                    "available": True,
                }
            ],
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
                            "message": {"role": "assistant", "content": "done"},
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
            "session_id": "session-media",
            "task_id": "task-media",
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


def _accepted_row(output_root: Path) -> dict:
    manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    path = next(
        output_root / item["path"]
        for item in manifest["files"]
        if item["path"].endswith("/accepted/trajectories-00000.jsonl")
    )
    return orjson.loads(path.read_bytes().splitlines()[0])


def test_tokenplan_mapping_survives_state_and_resume_reparse(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "dt=2026-09-03" / "session-media" / "req_media.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(orjson.dumps(_envelope(object_name="stored-v1.png")))
    config = PipelineConfig(
        input_root=input_root,
        input_format="tokenplan",
        output_root=output_root,
    )

    first = normalize(config)
    assert first.parsed == 1
    assert first.reused == 0
    assert first.skipped_inputs == 0
    assert _accepted_row(output_root)["multimodal_file_mapping"] == [
        {"part_id": "media_0", "object_name": "stored-v1.png"}
    ]
    with StateStore(output_root / ".state" / "trajfoundry.sqlite") as state:
        [snapshot] = list(state.iter_snapshots())
        assert snapshot.multimodal_file_mapping[0].object_name == "stored-v1.png"

    resume_config = PipelineConfig(
        input_root=input_root,
        input_format="tokenplan",
        output_root=output_root,
        resume=True,
    )
    unchanged = normalize(resume_config)
    assert unchanged.reused == 1
    assert unchanged.parsed == 0
    assert _accepted_row(output_root)["multimodal_file_mapping"] == [
        {"part_id": "media_0", "object_name": "stored-v1.png"}
    ]

    source.write_bytes(orjson.dumps(_envelope(object_name="stored-v2.png")))
    changed = normalize(resume_config)
    assert changed.reused == 0
    assert changed.parsed == 1
    assert _accepted_row(output_root)["multimodal_file_mapping"] == [
        {"part_id": "media_0", "object_name": "stored-v2.png"}
    ]
    assert validate_output(output_root).valid
