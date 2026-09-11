"""Scheduler-facing entry points for repeatable TrajFoundry runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from .pipeline import PipelineConfig, PipelineStats, normalize, normalize_source
from .s3 import S3CaptureSource, S3Location, create_s3_client, parse_s3_uri
from .s3_output import S3OutputSet
from .s3_validation import S3ValidationBackend
from .validation import ValidationReport, validate_output, validate_output_backend

LOGGER = logging.getLogger(__name__)

InputFormat = Literal["freerouter", "tokenplan"]
DEFAULT_MAX_SHARD_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class JobResult:
    """The run statistics and validation report for the selected generation."""

    stats: PipelineStats
    validation: ValidationReport

    @property
    def accepted(self) -> int:
        """Number of trajectories admitted to the accepted output."""

        return self.validation.counts.get("accepted", 0)

    @property
    def quarantined_trajectories(self) -> int:
        """Number of materialized trajectories kept in quarantine."""

        return self.validation.counts.get("quarantined_trajectories", 0)

    @property
    def quarantined_records(self) -> int:
        """Number of source captures represented by quarantine records."""

        return self.validation.counts.get("quarantined_records", 0)


class JobValidationError(RuntimeError):
    """Raised when an output generation does not satisfy its contract."""

    def __init__(self, output_root: str | Path, report: ValidationReport) -> None:
        self.output_root = output_root
        self.report = report
        details = "; ".join(report.errors[:5]) or "unknown validation error"
        if len(report.errors) > 5:
            details += f"; {len(report.errors) - 5} more error(s)"
        super().__init__(f"output validation failed for {output_root}: {details}")


def _locations_overlap(left: S3Location, right: S3Location) -> bool:
    return left.bucket == right.bucket and (
        left.prefix.startswith(right.prefix) or right.prefix.startswith(left.prefix)
    )


def _read_s3_response_bytes(response: dict[str, Any]) -> bytes:
    """Read and close a small S3 response without exposing its contents."""

    body = response.get("Body")
    if body is None or not callable(getattr(body, "read", None)):
        raise OSError("S3 response has no readable body")
    payload = bytearray()
    try:
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise TypeError("S3 response body must yield bytes")
            payload.extend(chunk)
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()

    content_length = response.get("ContentLength")
    if (
        not isinstance(content_length, int)
        or isinstance(content_length, bool)
        or content_length < 0
        or content_length != len(payload)
    ):
        raise OSError("S3 response byte count does not match")
    return bytes(payload)


def _close_s3_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # noqa: BLE001 - cleanup must not mask the task result
        LOGGER.warning("TrajFoundry could not close the S3 client cleanly")


def _run_s3_job_with_client(
    client: Any,
    *,
    input_location: S3Location,
    output_location: S3Location,
    input_format: InputFormat,
    workspace: Path,
    max_shard_bytes: int,
) -> JobResult:
    source = S3CaptureSource(client, input_location)
    with TemporaryDirectory(prefix="trajfoundry-state-", dir=workspace) as directory:
        stats, manifest_bytes = normalize_source(
            source,
            input_format=input_format,
            state_path=Path(directory) / "trajfoundry.sqlite",
            output_factory=lambda: S3OutputSet(
                client,
                output_location,
                max_shard_bytes=max_shard_bytes,
            ),
            max_shard_bytes=max_shard_bytes,
        )

    report = validate_output_backend(
        manifest_bytes,
        S3ValidationBackend(client, output_location),
    )
    if not report.valid:
        raise JobValidationError(output_location.uri, report)

    manifest_key = output_location.key("manifest.json")
    client.put_object(
        Bucket=output_location.bucket,
        Key=manifest_key,
        Body=manifest_bytes,
        ContentType="application/json",
    )
    published = _read_s3_response_bytes(
        client.get_object(Bucket=output_location.bucket, Key=manifest_key)
    )
    if published != manifest_bytes:
        raise OSError("published S3 manifest verification failed")

    if stats.parse_failures:
        LOGGER.warning(
            "TrajFoundry quarantined %d capture(s) during parsing",
            stats.parse_failures,
        )
    if report.counts.get("accepted", 0) == 0:
        LOGGER.warning(
            "TrajFoundry produced no accepted trajectories; inspect quarantine output"
        )
    LOGGER.info(
        "TrajFoundry S3 job complete: discovered=%d parsed=%d "
        "trajectories=%d accepted=%d quarantined_trajectories=%d "
        "quarantined_records=%d validation=passed",
        stats.discovered,
        stats.parsed,
        stats.stored_trajectories,
        report.counts.get("accepted", 0),
        report.counts.get("quarantined_trajectories", 0),
        report.counts.get("quarantined_records", 0),
    )
    return JobResult(stats=stats, validation=report)


def run_job(
    input_root: str | Path,
    output_root: str | Path,
    input_format: InputFormat = "freerouter",
    resume: bool = True,
    *,
    state_path: str | Path | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> JobResult:
    """Run normalization and verify the published output.

    This is the small public boundary intended for scheduler workers.  The
    underlying pipeline still quarantines individual bad captures so one
    malformed source does not discard the rest of a batch.  A pipeline-level
    exception or an invalid published output propagates as a task failure.
    """

    if input_format not in {"freerouter", "tokenplan"}:
        raise ValueError(f"unsupported input format: {input_format!r}")

    input_path = Path(input_root).expanduser()
    output_path = Path(output_root).expanduser()
    configured_state = Path(state_path).expanduser() if state_path is not None else None
    LOGGER.info(
        "starting TrajFoundry job: input=%s output=%s format=%s resume=%s",
        input_path,
        output_path,
        input_format,
        resume,
    )

    stats = normalize(
        PipelineConfig(
            input_root=input_path,
            input_format=input_format,
            output_root=output_path,
            state_path=configured_state,
            resume=resume,
            max_shard_bytes=max_shard_bytes,
        )
    )
    report = validate_output(output_path)

    if stats.discovered == 0:
        LOGGER.warning("TrajFoundry input contains no capture files: %s", input_path)
    if stats.parse_failures:
        LOGGER.warning(
            "TrajFoundry quarantined %d capture(s) during parsing",
            stats.parse_failures,
        )
    if stats.discovered and report.counts.get("accepted", 0) == 0:
        LOGGER.warning(
            "TrajFoundry produced no accepted trajectories; inspect quarantine output"
        )

    if not report.valid:
        raise JobValidationError(output_path, report)

    LOGGER.info(
        "TrajFoundry job complete: discovered=%d parsed=%d reused=%d "
        "trajectories=%d accepted=%d quarantined_trajectories=%d "
        "quarantined_records=%d validation=passed",
        stats.discovered,
        stats.parsed,
        stats.reused,
        stats.stored_trajectories,
        report.counts.get("accepted", 0),
        report.counts.get("quarantined_trajectories", 0),
        report.counts.get("quarantined_records", 0),
    )
    return JobResult(stats=stats, validation=report)


def run_s3_job(
    input_uri: str,
    output_uri: str,
    input_format: InputFormat,
    endpoint_url: str,
    *,
    region_name: str = "us-east-1",
    workspace_parent: str | Path | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> JobResult:
    """Normalize an S3 prefix directly into an unpublished S3 generation.

    Raw captures and JSONL output never pass through local staging files.  The
    aggregation database is created in a unique workspace-local directory and is
    deleted when this call returns or raises.  The root manifest is published
    only after the complete candidate generation passes remote validation.
    """

    if input_format not in {"freerouter", "tokenplan"}:
        raise ValueError(f"unsupported input format: {input_format!r}")
    if not isinstance(endpoint_url, str) or not endpoint_url:
        raise ValueError("endpoint_url must not be empty")
    if not isinstance(region_name, str) or not region_name:
        raise ValueError("region_name must not be empty")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")

    input_location = parse_s3_uri(input_uri)
    output_location = parse_s3_uri(output_uri)
    if _locations_overlap(input_location, output_location):
        raise ValueError("S3 input and output locations must not overlap")

    workspace = (
        Path.cwd() if workspace_parent is None else Path(workspace_parent).expanduser()
    ).resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace parent does not exist: {workspace}")

    LOGGER.info(
        "starting TrajFoundry S3 job: input=%s output=%s format=%s",
        input_location.uri,
        output_location.uri,
        input_format,
    )
    client = create_s3_client(
        endpoint_url=endpoint_url,
        region_name=region_name,
    )
    try:
        return _run_s3_job_with_client(
            client,
            input_location=input_location,
            output_location=output_location,
            input_format=input_format,
            workspace=workspace,
            max_shard_bytes=max_shard_bytes,
        )
    finally:
        _close_s3_client(client)


__all__ = [
    "DEFAULT_MAX_SHARD_BYTES",
    "InputFormat",
    "JobResult",
    "JobValidationError",
    "run_job",
    "run_s3_job",
]
