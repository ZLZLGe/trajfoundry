"""Low-overhead runtime resource diagnostics for scheduler logs.

The scheduler kills the whole cgroup when its memory limit is exceeded.  A
parent-process RSS value alone is therefore misleading: this module reports
both the current process and the cgroup totals when the Linux cgroup files are
available.  All reads are best-effort and never affect the normalization job.
"""

from __future__ import annotations

import os
import resource
from pathlib import Path


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:  # noqa: BLE001 - diagnostics must never fail a job
        return "unavailable"


def _mib(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value / 1024**2:.1f}MiB"


def _bytes_from_text(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _memory_events() -> tuple[str, str]:
    values: dict[str, str] = {}
    raw = _read_text("/sys/fs/cgroup/memory.events")
    if raw != "unavailable":
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) == 2:
                values[parts[0]] = parts[1]
    return values.get("oom", "unavailable"), values.get("oom_kill", "unavailable")


def memory_summary() -> str:
    """Return one compact, log-safe snapshot of process and cgroup memory."""

    try:
        # Linux reports ru_maxrss in KiB; keep it separate from current RSS.
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (AttributeError, ValueError, OSError):
        peak_rss = None

    rss_bytes: int | None = None
    status = _read_text("/proc/self/status")
    if status != "unavailable":
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rss_bytes = int(parts[1]) * 1024
                    except ValueError:
                        pass
                break

    current = _bytes_from_text(_read_text("/sys/fs/cgroup/memory.current"))
    peak = _bytes_from_text(_read_text("/sys/fs/cgroup/memory.peak"))
    limit = _read_text("/sys/fs/cgroup/memory.max")
    oom, oom_kill = _memory_events()
    return (
        f"pid={os.getpid()} rss={_mib(rss_bytes)} rss_peak={_mib(peak_rss)} "
        f"cgroup_current={_mib(current)} cgroup_peak={_mib(peak)} "
        f"cgroup_limit={limit} oom={oom} oom_kill={oom_kill}"
    )


__all__ = ["memory_summary"]
