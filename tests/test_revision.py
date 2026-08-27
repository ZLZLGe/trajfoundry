from pathlib import Path

import orjson

from trajfoundry import pipeline


def test_resume_reparses_when_normalizer_revision_changes(
    tmp_path: Path, monkeypatch: object
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    capture_path = input_root / "partition" / "session" / "one.json"
    capture_path.parent.mkdir(parents=True)
    capture_path.write_bytes(
        orjson.dumps(
            {
                "path": "/v1/responses",
                "session_id": "session",
                "status_code": 200,
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
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        }
                    ],
                },
            }
        )
    )

    first = pipeline.normalize(
        pipeline.PipelineConfig(input_root=input_root, output_root=output_root)
    )
    assert first.parsed == 1

    monkeypatch.setattr(
        pipeline,
        "NORMALIZER_REVISION",
        f"{pipeline.NORMALIZER_REVISION}-next",
    )
    resumed = pipeline.normalize(
        pipeline.PipelineConfig(
            input_root=input_root,
            output_root=output_root,
            resume=True,
        )
    )

    assert resumed.reused == 0
    assert resumed.parsed == 1
