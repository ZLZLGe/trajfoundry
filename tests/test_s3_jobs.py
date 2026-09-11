from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import orjson
import pytest

from trajfoundry import jobs
from trajfoundry.credentials import S3Credentials
from trajfoundry.validation import ValidationReport


class _Paginator:
    def __init__(self, client: _MemoryS3Client) -> None:
        self._client = client

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
        contents = [
            {
                "Key": key,
                "Size": len(payload),
                "ETag": self._client.etag(payload),
            }
            for (bucket, key), payload in sorted(self._client.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        self._client.calls.append(("list_objects_v2", Prefix))
        return [{"Contents": contents}]


class _MemoryS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    @staticmethod
    def etag(payload: bytes) -> str:
        return f'"{hashlib.md5(payload, usedforsecurity=False).hexdigest()}"'

    def get_paginator(self, operation: str) -> _Paginator:
        assert operation == "list_objects_v2"
        return _Paginator(self)

    def get_object(self, *, Bucket: str, Key: str, **kwargs: str) -> dict[str, Any]:
        self.calls.append(("get_object", Key))
        payload = self.objects[(Bucket, Key)]
        etag = self.etag(payload)
        if_match = kwargs.get("IfMatch")
        if if_match is not None and if_match != etag:
            raise OSError("precondition failed")
        return {
            "Body": io.BytesIO(payload),
            "ContentLength": len(payload),
            "ETag": etag,
        }

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        **kwargs: str,
    ) -> dict[str, str]:
        del kwargs
        payload = bytes(Body)
        self.calls.append(("put_object", Key))
        self.objects[(Bucket, Key)] = payload
        return {"ETag": self.etag(payload)}

    def close(self) -> None:
        self.calls.append(("close", ""))
        self.closed = True


def _capture() -> bytes:
    return orjson.dumps(
        {
            "path": "/v1/responses",
            "session_id": "session",
            "request_id": "turn-1",
            "captured_at": "2026-09-09T00:00:00Z",
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


def _tokenplan_capture() -> bytes:
    return orjson.dumps(
        {
            "client_request": {
                "method": "POST",
                "path": "/v1/chat/completions",
                "protocol": "openai_chat",
                "public_model": "chat-test",
                "stream": False,
                "capture": {
                    "body": {
                        "model": "chat-test",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    "status": "COMPLETE",
                },
                "media": [],
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
                "request_id": "request-1",
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
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    client: _MemoryS3Client,
) -> jobs.JobResult:
    monkeypatch.setattr(jobs, "create_s3_client", lambda **kwargs: client)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = jobs.run_s3_job(
        "s3://agent-trajectory/lakehouse/free-router/masked-raw/v001/dt=2026-09-09/",
        "s3://agent-trajectory/lakehouse/free-router/normalized/v002/dt=2026-09-09/",
        "freerouter",
        "http://s3.example.invalid",
        workspace_parent=workspace,
        credentials_path=None,
    )
    assert list(workspace.iterdir()) == []
    return result


def test_run_s3_job_streams_valid_generation_then_publishes_manifest_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _MemoryS3Client()
    input_key = "lakehouse/free-router/masked-raw/v001/dt=2026-09-09/capture.json"
    client.objects[("agent-trajectory", input_key)] = _capture()

    result = _run(monkeypatch, tmp_path, client)

    assert result.validation.valid
    assert result.accepted == 1
    assert result.stats.discovered == 1
    manifest_key = "lakehouse/free-router/normalized/v002/dt=2026-09-09/manifest.json"
    manifest = orjson.loads(client.objects[("agent-trajectory", manifest_key)])
    assert manifest["input_root"].endswith("masked-raw/v001/dt=2026-09-09/")
    assert manifest["input_format"] == "freerouter"
    assert len(manifest["files"]) == 2
    assert all(path["path"].startswith("generations/") for path in manifest["files"])
    put_calls = [call for call in client.calls if call[0] == "put_object"]
    assert put_calls[-1] == ("put_object", manifest_key)
    assert client.calls[-2] == ("get_object", manifest_key)
    assert client.calls[-1] == ("close", "")
    assert client.closed


def test_run_s3_job_normalizes_tokenplan_directly_between_s3_prefixes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _MemoryS3Client()
    input_key = (
        "lakehouse/token-plan/masked-raw/v001/dt=2026-09-09/session/req_capture.json"
    )
    client.objects[("agent-trajectory", input_key)] = _tokenplan_capture()
    monkeypatch.setattr(jobs, "create_s3_client", lambda **kwargs: client)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = jobs.run_s3_job(
        "s3://agent-trajectory/lakehouse/token-plan/masked-raw/v001/dt=2026-09-09/",
        "s3://agent-trajectory/lakehouse/token-plan/normalized/v002/dt=2026-09-09/",
        "tokenplan",
        "http://s3.example.invalid",
        workspace_parent=workspace,
        credentials_path=None,
    )

    assert result.validation.valid
    assert result.accepted == 1
    assert result.stats.discovered == 1
    assert result.stats.parsed == 1
    manifest_key = "lakehouse/token-plan/normalized/v002/dt=2026-09-09/manifest.json"
    manifest = orjson.loads(client.objects[("agent-trajectory", manifest_key)])
    assert manifest["input_format"] == "tokenplan"
    assert list(workspace.iterdir()) == []
    assert client.closed


def test_run_s3_job_does_not_publish_manifest_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _MemoryS3Client()
    input_key = "lakehouse/free-router/masked-raw/v001/dt=2026-09-09/capture.json"
    client.objects[("agent-trajectory", input_key)] = _capture()
    manifest_key = "lakehouse/free-router/normalized/v002/dt=2026-09-09/manifest.json"
    previous_manifest = b"previous manifest"
    client.objects[("agent-trajectory", manifest_key)] = previous_manifest
    monkeypatch.setattr(
        jobs,
        "validate_output_backend",
        lambda manifest, backend: ValidationReport(
            valid=False,
            errors=["candidate generation is invalid"],
            counts={},
        ),
    )

    with pytest.raises(jobs.JobValidationError, match="candidate generation"):
        _run(monkeypatch, tmp_path, client)

    assert client.objects[("agent-trajectory", manifest_key)] == previous_manifest
    assert any("/generations/" in key for _, key in client.objects)
    assert client.closed


def test_run_s3_job_rejects_empty_input_without_writing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _MemoryS3Client()

    with pytest.raises(ValueError, match="capture source is empty"):
        _run(monkeypatch, tmp_path, client)

    assert not any(call[0] == "put_object" for call in client.calls)
    assert client.closed


def test_run_s3_job_loads_default_credentials_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _MemoryS3Client()
    credentials = S3Credentials(
        aws_access_key_id="fake-access-key",
        aws_secret_access_key="fake-secret-key",
        aws_session_token="fake-session-token",
    )
    loaded_paths: list[Path] = []
    client_kwargs: list[dict[str, Any]] = []

    def _load(path: Path) -> S3Credentials:
        loaded_paths.append(path)
        return credentials

    def _create(**kwargs: Any) -> _MemoryS3Client:
        client_kwargs.append(kwargs)
        return client

    monkeypatch.setattr(jobs, "load_s3_credentials", _load)
    monkeypatch.setattr(jobs, "create_s3_client", _create)

    with pytest.raises(ValueError, match="capture source is empty"):
        jobs.run_s3_job(
            "s3://bucket/input/",
            "s3://bucket/output/",
            "freerouter",
            "http://s3.example.invalid",
            workspace_parent=tmp_path,
        )

    assert loaded_paths == [jobs.DEFAULT_S3_CREDENTIALS_PATH]
    assert client_kwargs == [
        {
            "endpoint_url": "http://s3.example.invalid",
            "region_name": "us-east-1",
            "credentials": credentials,
        }
    ]
    assert client.closed


@pytest.mark.parametrize(
    ("input_uri", "output_uri"),
    [
        ("s3://bucket/root/", "s3://bucket/root/"),
        ("s3://bucket/root/", "s3://bucket/root/output/"),
        ("s3://bucket/root/input/", "s3://bucket/root/"),
    ],
)
def test_run_s3_job_rejects_overlapping_locations_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_uri: str,
    output_uri: str,
) -> None:
    monkeypatch.setattr(
        jobs,
        "create_s3_client",
        lambda **kwargs: pytest.fail("client must not be created"),
    )
    monkeypatch.setattr(
        jobs,
        "load_s3_credentials",
        lambda path: pytest.fail("credentials must not be loaded"),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        jobs.run_s3_job(
            input_uri,
            output_uri,
            "freerouter",
            "http://s3.example.invalid",
            workspace_parent=tmp_path,
        )


def test_run_s3_job_rejects_missing_workspace_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        jobs,
        "create_s3_client",
        lambda **kwargs: pytest.fail("client must not be created"),
    )
    monkeypatch.setattr(
        jobs,
        "load_s3_credentials",
        lambda path: pytest.fail("credentials must not be loaded"),
    )

    with pytest.raises(NotADirectoryError, match="workspace parent"):
        jobs.run_s3_job(
            "s3://bucket/input/",
            "s3://bucket/output/",
            "freerouter",
            "http://s3.example.invalid",
            workspace_parent=tmp_path / "missing",
        )
