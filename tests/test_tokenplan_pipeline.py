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
        media_records = [
            {
                "part_id": "media_0",
                "object_name": "media_0-test.png",
                "available": True,
            }
        ]
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
            "source_user_id": "tokenplan-user",
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


def test_tokenplan_pipeline_includes_test_and_publishes_media_mapping(
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
    assert stats.parsed == 2
    assert stats.parse_failures == 1
    assert stats.skipped_inputs == 0

    manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    assert manifest["schema_version"] == "trajfoundry-v4"
    assert manifest["input_format"] == "tokenplan"
    assert manifest["counts"]["input_files"] == 3
    assert manifest["counts"]["accepted"] == 2
    assert manifest["counts"]["skipped_inputs"] == 0
    assert manifest["counts"]["skip_reason_counts"] == {}

    accepted_rows = [
        orjson.loads((output_root / entry["path"]).read_bytes())
        for entry in manifest["files"]
        if entry["path"] != "lineage.jsonl"
    ]
    good_row = next(
        row
        for row in accepted_rows
        if row["metadata"]["source_file"] == "req_good.json"
    )
    good_sub_session_id = good_row["metadata"].pop("sub_session_id")
    assert good_sub_session_id in {0, 1}
    assert good_row["metadata"] == {
        "source_file": "req_good.json",
        "source_name": "tokenplan",
        "line_no": 0,
        "created_at": "1970-01-01T00:00:01.234Z",
        "model": "chat-test",
        "user_id": "tokenplan-user",
        "session_id": "session-1",
        "source_type": "api-router",
        "specific_source": "token-plan",
    }

    media_row = next(
        row
        for row in accepted_rows
        if row["metadata"]["source_file"] == "req_media.json"
    )
    assert media_row["metadata"]["sub_session_id"] in {0, 1}
    assert media_row["metadata"]["sub_session_id"] != good_sub_session_id
    assert media_row["multimodal_file_mapping"] == [
        {"part_id": "media_0", "object_name": "media_0-test.png"}
    ]

    assert manifest["counts"]["quarantined_records"] == 1
    assert manifest["counts"]["reason_counts"]["tokenplan_empty_envelope"] == 1
    report = validate_output(output_root)
    assert report.valid
    assert report.counts["skipped_inputs"] == 0


def test_tokenplan_literal_missing_sessions_merge_only_within_user(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    for name, user, is_long, session in (
        ("alice-short", "alice", False, ""),
        ("alice-long", "alice", True, "no_session_id"),
        ("bob-short", "bob", False, "no_session_id"),
        ("alice-known", "alice", False, "real-session"),
    ):
        envelope = _envelope(request_id=name)
        envelope["metadata"]["source_user_id"] = user
        envelope["metadata"]["session_id"] = "no_session_id" if session else ""
        source_key = f"dt=2026-10-08/{user}/no_session_id/req_{name}.json"
        envelope["metadata"]["source_key"] = source_key
        request_body = envelope["client_request"]["capture"]["body"]
        request_body["client_metadata"] = {"session_id": session, "thread_id": name}
        if is_long:
            request_body["messages"].extend(
                [
                    {"role": "assistant", "content": "done"},
                    {"role": "user", "content": "next"},
                ]
            )
        _write_json(input_root / source_key, envelope)

    stats = normalize(
        PipelineConfig(
            input_root=input_root, input_format="tokenplan", output_root=output_root
        )
    )

    assert stats.parsed == 4
    assert stats.parse_failures == 0
    assert stats.prefix_intermediates == 1
    assert stats.stored_trajectories == 3
    manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    records = [
        orjson.loads((output_root / entry["path"]).read_bytes())
        for entry in manifest["files"]
        if entry["path"] != "lineage.jsonl"
    ]
    assert {row["metadata"]["source_file"] for row in records} == {
        "req_alice-long.json",
        "req_bob-short.json",
        "req_alice-known.json",
    }
    assert {
        (row["metadata"]["user_id"], row["metadata"]["session_id"]) for row in records
    } == {
        ("alice", "no_session_id"),
        ("bob", "no_session_id"),
        ("alice", "real-session"),
    }
    for row in records:
        synthesized = row["metadata"]["session_id"] == "no_session_id"
        assert row["metadata"]["sub_session_id"] == 0
        assert (
            "metadata_session_id_synthesized"
            in {issue["code"] for issue in row["normalization_audit"]["issues"]}
        ) == synthesized
    assert validate_output(output_root).valid
