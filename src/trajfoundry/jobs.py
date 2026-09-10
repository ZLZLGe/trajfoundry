"""Scheduler-facing entry points for repeatable TrajFoundry runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .pipeline import PipelineConfig, PipelineStats, normalize
from .validation import ValidationReport, validate_output

LOGGER = logging.getLogger(__name__)

InputFormat = Literal["freerouter", "tokenplan"]
DEFAULT_MAX_SHARD_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class JobResult:
    """The durable run statistics and post-publication validation report."""

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
    """Raised when the published output does not satisfy its contract."""

    def __init__(self, output_root: Path, report: ValidationReport) -> None:
        self.output_root = output_root
        self.report = report
        details = "; ".join(report.errors[:5]) or "unknown validation error"
        if len(report.errors) > 5:
            details += f"; {len(report.errors) - 5} more error(s)"
        super().__init__(f"output validation failed for {output_root}: {details}")


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


__all__ = [
    "DEFAULT_MAX_SHARD_BYTES",
    "InputFormat",
    "JobResult",
    "JobValidationError",
    "run_job",
]
