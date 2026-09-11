import hashlib
import re
from typing import Any

import orjson
import pytest

from trajfoundry.export import SCHEMA_VERSION
from trajfoundry.models import (
    AuditTag,
    Message,
    Metadata,
    NormalizationAudit,
    QuarantineRecord,
    TrajectoryNode,
)
from trajfoundry.quality import enrich_trajectory
from trajfoundry.s3 import S3Location
from trajfoundry.s3_output import S3OutputSet


class FakeS3Client:
    def __init__(
        self,
        *,
        fail_part: bool = False,
        fail_abort_attempts: int = 0,
    ) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.uploads: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.aborted: list[tuple[str, str, str]] = []
        self._next_upload = 1
        self._fail_part = fail_part
        self._fail_abort_attempts = fail_abort_attempts

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict[str, str]:
        self.calls.append(("put_object", Key))
        self.objects[(Bucket, Key)] = bytes(Body)
        return {"ETag": '"put"'}

    def create_multipart_upload(self, *, Bucket: str, Key: str) -> dict[str, str]:
        self.calls.append(("create_multipart_upload", Key))
        upload_id = f"upload-{self._next_upload}"
        self._next_upload += 1
        self.uploads[upload_id] = {
            "bucket": Bucket,
            "key": Key,
            "parts": {},
            "etags": {},
        }
        return {"UploadId": upload_id}

    def upload_part(
        self,
        *,
        Bucket: str,
        Key: str,
        UploadId: str,
        PartNumber: int,
        Body: bytes,
    ) -> dict[str, str]:
        self.calls.append(("upload_part", Key))
        if self._fail_part:
            raise OSError("injected upload failure")
        upload = self.uploads[UploadId]
        assert (upload["bucket"], upload["key"]) == (Bucket, Key)
        etag = f'"part-{PartNumber}"'
        upload["parts"][PartNumber] = bytes(Body)
        upload["etags"][PartNumber] = etag
        return {"ETag": etag}

    def complete_multipart_upload(
        self,
        *,
        Bucket: str,
        Key: str,
        UploadId: str,
        MultipartUpload: dict[str, list[dict[str, str | int]]],
    ) -> dict[str, str]:
        self.calls.append(("complete_multipart_upload", Key))
        upload = self.uploads.pop(UploadId)
        assert (upload["bucket"], upload["key"]) == (Bucket, Key)
        parts = MultipartUpload["Parts"]
        assert parts == [
            {"ETag": upload["etags"][part_number], "PartNumber": part_number}
            for part_number in sorted(upload["parts"])
        ]
        self.objects[(Bucket, Key)] = b"".join(
            upload["parts"][part_number] for part_number in sorted(upload["parts"])
        )
        return {"ETag": '"complete"'}

    def abort_multipart_upload(
        self, *, Bucket: str, Key: str, UploadId: str
    ) -> dict[str, Any]:
        self.calls.append(("abort_multipart_upload", Key))
        if self._fail_abort_attempts:
            self._fail_abort_attempts -= 1
            raise OSError("injected abort failure")
        self.aborted.append((Bucket, Key, UploadId))
        self.uploads.pop(UploadId, None)
        return {}


def _trajectory(source: str, *, complete: bool = True) -> TrajectoryNode:
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


def _origin(source: str, **extra: object) -> dict[str, object]:
    return {
        "source_ref": source,
        "sha256": "a" * 64,
        "captured_at": "2026-09-09T00:00:00Z",
        "disposition": "pass",
        "reason_codes": [],
        **extra,
    }


def _location() -> S3Location:
    return S3Location("agent-trajectory", "lakehouse/source/normalized/v002/")


def test_small_generation_uses_put_and_returns_unpublished_manifest() -> None:
    client = FakeS3Client()
    output = S3OutputSet(client, _location())
    output.stats.input_files = 4
    output.write_trajectory(
        _trajectory("accepted.json"),
        [_origin("accepted.json"), _origin("accepted-copy.json")],
    )
    output.write_trajectory(
        _trajectory("quarantined.json", complete=False),
        [_origin("quarantined.json")],
    )
    output.write_record(
        QuarantineRecord(
            source_ref="invalid.json",
            sha256="b" * 64,
            normalization_audit=NormalizationAudit(
                tag=AuditTag.EXCLUDED,
                reason_codes=["capture_invalid"],
            ),
        )
    )
    output.write_skipped("test_input")

    manifest_bytes = output.close(
        input_root="s3://agent-trajectory/lakehouse/source/masked-raw/v001/",
        config_hash="config",
        input_format="freerouter",
    )
    manifest = orjson.loads(manifest_bytes)

    assert re.fullmatch(r"[0-9a-f]{32}", output.generation_id)
    assert manifest["schema_version"] == SCHEMA_VERSION == "trajfoundry-v3"
    assert manifest["counts"] == {
        "accepted": 1,
        "duplicate_trajectories": 1,
        "excluded_records": 1,
        "input_files": 4,
        "quarantined_records": 1,
        "quarantined_trajectories": 1,
        "reason_counts": {"capture_invalid": 1, "no_final_assistant_turn": 1},
        "skip_reason_counts": {"test_input": 1},
        "skipped_inputs": 1,
    }
    assert manifest["files"] == [
        item.manifest_entry() for item in output.written_objects
    ]
    assert len(manifest["files"]) == 4
    assert all(
        item["path"].startswith(f"generations/{output.generation_id}/")
        for item in manifest["files"]
    )
    assert all(name == "put_object" for name, _ in client.calls)
    assert (
        _location().bucket,
        _location().key("manifest.json"),
    ) not in client.objects
    for item in output.written_objects:
        body = client.objects[(item.bucket, item.key)]
        assert item.bytes == len(body)
        assert item.sha256 == hashlib.sha256(body).hexdigest()


def test_empty_generation_creates_only_empty_lineage() -> None:
    client = FakeS3Client()
    output = S3OutputSet(client, _location())

    manifest = orjson.loads(
        output.close(input_root="s3://bucket/input/", config_hash="config")
    )

    assert [item["path"] for item in manifest["files"]] == [
        f"generations/{output.generation_id}/lineage.jsonl"
    ]
    description = output.written_objects[0]
    assert description.bytes == 0
    assert client.objects[(description.bucket, description.key)] == b""
    assert not any(
        "/accepted/" in key or "/quarantine/" in key for _, key in client.objects
    )


def test_jsonl_rollover_does_not_split_lines() -> None:
    client = FakeS3Client()
    output = S3OutputSet(client, _location(), max_shard_bytes=1)
    output.write_trajectory(_trajectory("one.json"), [_origin("one.json")])
    output.write_trajectory(_trajectory("two.json"), [_origin("two.json")])

    output.close(input_root="s3://bucket/input/", config_hash="config")

    accepted = [
        item
        for item in output.written_objects
        if "/accepted/trajectories-" in item.path
    ]
    assert [item.path.rsplit("/", 1)[-1] for item in accepted] == [
        "trajectories-00000.jsonl",
        "trajectories-00001.jsonl",
    ]
    assert all(
        client.objects[(item.bucket, item.key)].count(b"\n") == 1 for item in accepted
    )


def test_large_object_uses_multipart_and_records_streaming_digest() -> None:
    client = FakeS3Client()
    output = S3OutputSet(client, _location())
    output.write_trajectory(
        _trajectory("large.json"),
        [_origin("large.json", evidence="x" * (8 * 1024 * 1024))],
    )

    output.close(input_root="s3://bucket/input/", config_hash="config")

    lineage = next(
        item for item in output.written_objects if item.path.endswith("lineage.jsonl")
    )
    body = client.objects[(lineage.bucket, lineage.key)]
    lineage_calls = [name for name, key in client.calls if key == lineage.key]
    assert lineage_calls == [
        "create_multipart_upload",
        "upload_part",
        "upload_part",
        "complete_multipart_upload",
    ]
    assert lineage.bytes == len(body)
    assert lineage.sha256 == hashlib.sha256(body).hexdigest()


def test_multipart_failure_is_aborted_without_deleting_objects() -> None:
    client = FakeS3Client(fail_part=True)
    output = S3OutputSet(client, _location())

    with pytest.raises(OSError, match="injected upload failure"), output:
        output.write_trajectory(
            _trajectory("large.json"),
            [_origin("large.json", evidence="x" * (8 * 1024 * 1024))],
        )

    assert len(client.aborted) == 1
    assert client.uploads == {}
    assert client.objects == {}


def test_failed_multipart_abort_keeps_upload_id_for_outer_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeS3Client(fail_part=True, fail_abort_attempts=1)
    output = S3OutputSet(client, _location())

    with (
        caplog.at_level("WARNING"),
        pytest.raises(OSError, match="injected upload failure"),
        output,
    ):
        output.write_trajectory(
            _trajectory("large.json"),
            [_origin("large.json", evidence="x" * (8 * 1024 * 1024))],
        )

    abort_calls = [name for name, _ in client.calls if name == "abort_multipart_upload"]
    assert abort_calls == ["abort_multipart_upload", "abort_multipart_upload"]
    assert len(client.aborted) == 1
    assert client.uploads == {}
    assert "will retry" in caplog.text
    assert "injected abort failure" not in caplog.text
    assert not any(name.startswith("delete") for name, _ in client.calls)
