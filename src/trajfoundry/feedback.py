"""Common production-feedback counters for Python and Spark manifests."""

from __future__ import annotations

import time
from typing import Any


def unix_ms() -> int:
    """Unix wall-clock milliseconds, not the monotonic duration clock."""
    return time.time_ns() // 1_000_000


def is_trajectory_data_file(path: str) -> bool:
    """Only flat trajectory JSONL files contribute to production data volume."""
    return "/" not in path and path.endswith(".jsonl") and path != "lineage.jsonl"


def feedback_fields(
    stats: Any,
    *,
    total_bytes: int,
    started_at_ms: int,
    parse_failure_count: int = 0,
    completed_at_ms: int | None = None,
) -> dict[str, str | int]:
    """Count persisted top-level trajectories, not origins or nested agents.

    Keep legacy internal counts intact. Parse failures count input records;
    quarantined_records may also contain parsed but non-trajectory captures and
    must never be used as an alias for parse failures.
    """
    completed = (
        max(started_at_ms, unix_ms()) if completed_at_ms is None else completed_at_ms
    )
    fields = {
        "started_at_ms": started_at_ms,
        "completed_at_ms": completed,
        "total_bytes": total_bytes,
        "record_count": stats.accepted + stats.quarantined_trajectories,
        "accepted_trajectory_count": stats.accepted,
        "quarantined_trajectory_count": stats.quarantined_trajectories,
        "parse_failure_count": parse_failure_count,
    }
    if any(type(value) is not int or value < 0 for value in fields.values()):
        raise ValueError(
            "feedback counters and timestamps must be nonnegative integers"
        )
    if started_at_ms > completed:
        raise ValueError("started_at_ms must not exceed completed_at_ms")
    return {"status": "SUCCEEDED", **fields}
