from __future__ import annotations

from pathlib import Path
from typing import Literal

import orjson

from trajfoundry.models import CompactionRecord, Message, Snapshot
from trajfoundry.pipeline import PipelineConfig, _trajectory_from_leaf, normalize
from trajfoundry.validation import validate_output


def _snapshot(name: str, records: list[CompactionRecord]) -> Snapshot:
    return Snapshot(
        source_path=f"/{name}.json",
        source_sha256=name,
        session_id="session",
        thread_id="thread",
        turn_id=name,
        provider="openai",
        operation="responses",
        outcome="success",
        history=[Message(role="user", content="q")],
        response=[Message(role="assistant", content="done", reasoning_content="")],
        compaction_items=records,
        wire_complete=True,
    )


def _record(
    origin: Literal["history", "response"],
    item_index: int,
    compaction_id: str,
) -> CompactionRecord:
    return CompactionRecord(
        origin=origin,
        item_index=item_index,
        item={
            "type": "compaction",
            "id": compaction_id,
            "encrypted_content": f"opaque-{compaction_id}",
        },
    )


def test_contributor_compactions_are_deduplicated_and_sorted_deterministically() -> (
    None
):
    first_record = _record("history", 2, "cmp-a")
    second_record = _record("response", 0, "cmp-b")
    first = _snapshot("first", [first_record])
    leaf = _snapshot(
        "leaf",
        [second_record, first_record.model_copy(deep=True)],
    )

    forward = _trajectory_from_leaf(leaf, [first, leaf])
    backward = _trajectory_from_leaf(leaf, [leaf, first])

    assert forward.compaction_items == backward.compaction_items
    assert forward.compaction_items == [first_record, second_record]
    assert forward.compaction_items[0] is not first_record
    assert forward.compaction_items[1] is not second_record


def test_complete_compaction_capture_is_a_full_quarantined_trajectory(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    capture_path = input_root / "capture.json"
    capture_path.parent.mkdir(parents=True)
    raw_compaction = {
        "type": "compaction",
        "id": "cmp-1",
        "encrypted_content": "opaque-ciphertext",
        "future_provider_field": {"preserve": True},
    }
    capture_path.write_bytes(
        orjson.dumps(
            {
                "path": "/v1/responses",
                "session_id": "session",
                "request_id": "request",
                "captured_at": "2026-08-29T00:00:00Z",
                "status_code": 200,
                "request_body": {
                    "model": "gpt-test",
                    "client_metadata": {
                        "session_id": "session",
                        "thread_id": "thread",
                        "turn_id": "turn",
                    },
                    "input": [
                        {"type": "message", "role": "user", "content": "go"},
                        raw_compaction,
                    ],
                },
                "response_body": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": "done",
                        }
                    ],
                },
            }
        )
    )

    stats = normalize(PipelineConfig(input_root=input_root, output_root=output_root))
    manifest = orjson.loads((output_root / "manifest.json").read_bytes())
    trajectory_file = next(
        output_root / entry["path"]
        for entry in manifest["files"]
        if entry["path"].endswith("/quarantine/trajectories-00000.jsonl")
    )
    trajectory = orjson.loads(trajectory_file.read_bytes().splitlines()[0])

    assert stats.parsed == 1
    assert manifest["schema_version"] == "trajfoundry-v3"
    assert manifest["counts"]["accepted"] == 0
    assert manifest["counts"]["quarantined_trajectories"] == 1
    assert manifest["counts"]["quarantined_records"] == 0
    assert trajectory["compaction_items"] == [
        {"origin": "history", "item_index": 1, "item": raw_compaction}
    ]
    assert (
        "opaque_compaction_context" in trajectory["normalization_audit"]["reason_codes"]
    )
    assert validate_output(output_root).valid is True
