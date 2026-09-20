"""Direct S3 JSONL export and unpublished run manifest creation."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Self

import orjson

from .canonical import trajectory_id
from .export import SCHEMA_VERSION, ExportStats, trajectory_filename
from .models import QuarantineRecord, TrajectoryNode
from .output_contract import project_trajectory
from .quality import is_strict_sample, validate_derived_fields
from .s3 import S3Location

_BUFFER_BYTES = 8 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class S3ObjectDescription:
    """A completed immutable object in a manifest-backed flat output run."""

    path: str
    bucket: str
    key: str
    sha256: str
    bytes: int

    def manifest_entry(self) -> dict[str, str | int]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


class _S3ObjectWriter:
    """Buffer a small object for PUT and stream a large object as multipart."""

    def __init__(
        self,
        client: Any,
        location: S3Location,
        relative_path: str,
        *,
        buffer_bytes: int = _BUFFER_BYTES,
    ) -> None:
        if buffer_bytes < 5 * 1024 * 1024:
            raise ValueError("multipart buffer must be at least 5 MiB")
        self._client = client
        self._location = location
        self._relative_path = relative_path
        self._key = location.key(relative_path)
        self._buffer_bytes = buffer_bytes
        self._buffer = bytearray()
        self._digest = hashlib.sha256()
        self._size = 0
        self._upload_id: str | None = None
        self._parts: list[dict[str, str | int]] = []
        self._closed = False
        self._failed = False
        self._description: S3ObjectDescription | None = None

    @property
    def description(self) -> S3ObjectDescription | None:
        return self._description

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("cannot write to a closed S3 object")
        if self._failed:
            raise ValueError("cannot write to a failed S3 object")

    def _start_multipart(self) -> None:
        response = self._client.create_multipart_upload(
            Bucket=self._location.bucket,
            Key=self._key,
        )
        upload_id = response.get("UploadId")
        if not isinstance(upload_id, str) or not upload_id:
            raise RuntimeError("CreateMultipartUpload returned no UploadId")
        self._upload_id = upload_id

    def _upload_part(self, body: bytes) -> None:
        assert self._upload_id is not None
        part_number = len(self._parts) + 1
        response = self._client.upload_part(
            Bucket=self._location.bucket,
            Key=self._key,
            UploadId=self._upload_id,
            PartNumber=part_number,
            Body=body,
        )
        etag = response.get("ETag")
        if not isinstance(etag, str) or not etag:
            raise RuntimeError("UploadPart returned no ETag")
        self._parts.append({"ETag": etag, "PartNumber": part_number})

    def _flush_full_parts(self) -> None:
        while len(self._buffer) >= self._buffer_bytes:
            body = bytes(self._buffer[: self._buffer_bytes])
            self._upload_part(body)
            del self._buffer[: self._buffer_bytes]

    def _abort_multipart(self) -> None:
        if self._upload_id is None:
            return
        upload_id = self._upload_id
        try:
            self._client.abort_multipart_upload(
                Bucket=self._location.bucket,
                Key=self._key,
                UploadId=upload_id,
            )
        except Exception:  # noqa: BLE001 - preserve the triggering failure
            # Keep the ID so the outer OutputSet.abort() can retry.  Do not log
            # the SDK exception because it may contain endpoint details.
            LOGGER.warning("S3 multipart upload could not be aborted; will retry")
        else:
            self._upload_id = None

    def write(self, payload: bytes) -> None:
        self._ensure_open()
        if not payload:
            return
        self._digest.update(payload)
        self._size += len(payload)
        self._buffer.extend(payload)
        try:
            if self._upload_id is None and len(self._buffer) > self._buffer_bytes:
                self._start_multipart()
            if self._upload_id is not None:
                self._flush_full_parts()
        except Exception:
            self._failed = True
            self._abort_multipart()
            raise

    def close(self) -> S3ObjectDescription:
        if self._description is not None:
            return self._description
        self._ensure_open()
        try:
            if self._upload_id is None:
                self._client.put_object(
                    Bucket=self._location.bucket,
                    Key=self._key,
                    Body=bytes(self._buffer),
                )
            else:
                if self._buffer:
                    self._upload_part(bytes(self._buffer))
                    self._buffer.clear()
                upload_id = self._upload_id
                self._client.complete_multipart_upload(
                    Bucket=self._location.bucket,
                    Key=self._key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": self._parts},
                )
                self._upload_id = None
        except Exception:
            self._failed = True
            self._abort_multipart()
            raise
        self._buffer.clear()
        self._closed = True
        self._description = S3ObjectDescription(
            path=self._relative_path,
            bucket=self._location.bucket,
            key=self._key,
            sha256=self._digest.hexdigest(),
            bytes=self._size,
        )
        return self._description

    def abort(self) -> None:
        self._abort_multipart()
        self._buffer.clear()
        if not self._closed:
            self._failed = True


class S3OutputSet:
    """Write flat trajectory objects without publishing the root manifest."""

    def __init__(
        self,
        client: Any,
        location: S3Location,
        *,
        max_shard_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if max_shard_bytes <= 0:
            raise ValueError("max shard bytes must be positive")
        self.max_shard_bytes = max_shard_bytes
        self._client = client
        self.location = location
        self._trajectory_objects: list[S3ObjectDescription] = []
        self._trajectory_names: set[str] = set()
        self._lineage = _S3ObjectWriter(client, location, "lineage.jsonl")
        self.stats = ExportStats()
        self._manifest_bytes: bytes | None = None
        self._aborted = False

    @property
    def written_objects(self) -> tuple[S3ObjectDescription, ...]:
        objects = [*self._trajectory_objects]
        if self._lineage.description is not None:
            objects.append(self._lineage.description)
        return tuple(sorted(objects, key=lambda item: item.path))

    @property
    def objects(self) -> tuple[S3ObjectDescription, ...]:
        """Alias for callers that do not need to distinguish staged objects."""

        return self.written_objects

    def write_trajectory(
        self, node: TrajectoryNode, origins: list[dict[str, Any]]
    ) -> str:
        projected = project_trajectory(node)
        validate_derived_fields(node)
        identifier = trajectory_id(node)
        filename = trajectory_filename(node, identifier)
        if filename in self._trajectory_names:
            raise ValueError(f"duplicate trajectory output filename: {filename}")
        payload = orjson.dumps(projected, option=orjson.OPT_SORT_KEYS) + b"\n"
        if len(payload) > self.max_shard_bytes:
            raise ValueError(
                "trajectory JSONL object exceeds max_shard_bytes: "
                f"{len(payload)} > {self.max_shard_bytes}"
            )
        self._lineage.write(
            orjson.dumps(
                {
                    "trajectory_id": identifier,
                    "representative": node.metadata.source_file,
                    "origin_count": len(origins),
                    "origins": origins,
                },
                option=orjson.OPT_SORT_KEYS,
            )
            + b"\n"
        )
        writer = _S3ObjectWriter(self._client, self.location, filename)
        try:
            writer.write(payload)
            description = writer.close()
        except Exception:
            writer.abort()
            raise
        self._trajectory_names.add(filename)
        self._trajectory_objects.append(description)
        strict = is_strict_sample(node)
        if strict:
            self.stats.accepted += 1
        else:
            self.stats.quarantined_trajectories += 1
            if node.normalization_audit:
                self.stats.add_reasons(node.normalization_audit.reason_codes)
        self.stats.duplicate_trajectories += max(0, len(origins) - 1)
        return identifier

    def write_record(self, record: QuarantineRecord) -> None:
        self.stats.quarantined_records += 1
        if record.normalization_audit.tag.value == "excluded":
            self.stats.excluded_records += 1
        self.stats.add_reasons(record.normalization_audit.reason_codes)

    def write_skipped(self, reason: str) -> None:
        self.stats.add_skipped(reason)

    def close(
        self,
        *,
        input_root: str,
        config_hash: str,
        input_format: Literal[
            "freerouter", "tokenplan", "sxf", "deepinfra"
        ] = "freerouter",
    ) -> bytes:
        if self._manifest_bytes is not None:
            return self._manifest_bytes
        if self._aborted:
            raise ValueError("cannot close an aborted S3 output set")
        try:
            self._lineage.close()
        except Exception:
            self.abort()
            raise

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "input_root": input_root,
            "input_format": input_format,
            "config_hash": config_hash,
            "token_estimator": "unicode-word-v1",
            "counts": {
                **self.stats.__dict__,
                "reason_counts": dict(sorted(self.stats.reason_counts.items())),
                "skip_reason_counts": dict(
                    sorted(self.stats.skip_reason_counts.items())
                ),
            },
            "files": [item.manifest_entry() for item in self.written_objects],
        }
        self._manifest_bytes = orjson.dumps(
            manifest, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
        )
        return self._manifest_bytes

    def abort(self) -> None:
        self._lineage.abort()
        if self._manifest_bytes is None:
            self._aborted = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is not None:
            self.abort()
