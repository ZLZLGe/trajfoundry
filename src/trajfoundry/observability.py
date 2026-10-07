"""Best-effort diagnostics separating process memory from shared cgroup usage.

RSS covers one process, whereas cgroup accounting also includes other tasks,
page cache and kernel memory. Process-tree PSS avoids counting shared pages once
per child. Missing diagnostics remain unknown, never an invented zero.
"""

from __future__ import annotations

import os
import resource
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath


def _read_text(path: str | Path) -> str:
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


def _key_values(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in value.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            number = _bytes_from_text(parts[1])
            if number is not None:
                result[parts[0].rstrip(":")] = number
    return result


def _unescape_mount(value: str) -> str:
    for encoded, decoded in (("\\040", " "), ("\\011", "\t"), ("\\134", "\\")):
        value = value.replace(encoded, decoded)
    return value


@lru_cache(maxsize=1)
def _cgroup_directory() -> Path:
    """Resolve v2 membership for host and container cgroup namespaces."""

    membership: str | None = None
    for line in _read_text("/proc/self/cgroup").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and not parts[1]:
            membership = parts[2]
            break
    if membership is not None:
        for line in _read_text("/proc/self/mountinfo").splitlines():
            before, separator, after = line.partition(" - ")
            fields = before.split()
            if not separator or not after.startswith("cgroup2 ") or len(fields) < 5:
                continue
            root = PurePosixPath(_unescape_mount(fields[3]))
            mount = Path(_unescape_mount(fields[4]))
            try:
                relative = PurePosixPath(membership).relative_to(root)
            except ValueError:
                # Namespace-relative membership may accompany host-side roots.
                relative = PurePosixPath(membership.lstrip("/"))
            candidate = mount / str(relative)
            if (candidate / "memory.current").is_file():
                return candidate
            if membership == "/" and (mount / "memory.current").is_file():
                return mount
    return Path("/sys/fs/cgroup")


def _descendant_pids(root_pid: int) -> set[int]:
    """Include children created by any thread, without scanning every host PID."""

    found: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        try:
            tasks = list(Path(f"/proc/{pid}/task").iterdir())
        except OSError:
            continue
        for task in tasks:
            for value in _read_text(task / "children").split():
                if value.isdigit():
                    pending.append(int(value))
    return found


def _process_tree_memory(
    root_pid: int,
) -> tuple[int | None, int | None, int, bool, int | None, int | None]:
    rss = pss = count = anon = shmem = 0
    pss_complete = True
    breakdown_complete = True
    for pid in _descendant_pids(root_pid):
        values = _key_values(_read_text(f"/proc/{pid}/smaps_rollup"))
        pid_rss = values.get("Rss")
        pid_pss = values.get("Pss")
        if pid_rss is None:
            pid_rss = _key_values(_read_text(f"/proc/{pid}/status")).get("VmRSS")
        if pid_rss is None:
            pss_complete = False
            breakdown_complete = False
            continue
        count += 1
        rss += pid_rss * 1024
        if pid_pss is None:
            pss_complete = False
        else:
            pss += pid_pss * 1024
        if "Pss_Anon" not in values or "Pss_Shmem" not in values:
            breakdown_complete = False
        else:
            anon += values["Pss_Anon"] * 1024
            shmem += values["Pss_Shmem"] * 1024
    return (
        rss if count else None,
        pss if count and pss_complete else None,
        count,
        bool(count and pss_complete),
        anon if count and pss_complete and breakdown_complete else None,
        shmem if count and pss_complete and breakdown_complete else None,
    )


@dataclass(frozen=True)
class MemorySnapshot:
    pid: int
    rss_bytes: int | None = None
    rss_peak_bytes: int | None = None
    cgroup_current_bytes: int | None = None
    cgroup_peak_bytes: int | None = None
    cgroup_limit_bytes: int | None = None
    cgroup_anon_bytes: int | None = None
    cgroup_file_bytes: int | None = None
    cgroup_dirty_bytes: int | None = None
    cgroup_writeback_bytes: int | None = None
    cgroup_shmem_bytes: int | None = None
    cgroup_kernel_bytes: int | None = None
    oom: int | None = None
    oom_kill: int | None = None
    tree_rss_bytes: int | None = None
    tree_pss_bytes: int | None = None
    tree_process_count: int = 0
    tree_pss_complete: bool = False
    tree_pss_anon_bytes: int | None = None
    tree_pss_shmem_bytes: int | None = None

    @property
    def effective_tree_bytes(self) -> int | None:
        if self.tree_pss_complete and self.tree_pss_bytes is not None:
            return self.tree_pss_bytes
        return self.tree_rss_bytes

    @property
    def tree_nonreclaimable_bytes(self) -> int | None:
        """Conservative own-memory credit when estimating other tasks' usage.

        Never subtract all PSS from cgroup usage after subtracting clean file
        cache: mapped file PSS would be credited twice. On older kernels,
        subtracting *all* cgroup file usage from complete PSS yields a lower
        bound. An RSS sum is not a safe credit because shared pages repeat.
        """

        if not self.tree_pss_complete or self.tree_pss_bytes is None:
            return None
        if (
            self.tree_pss_anon_bytes is not None
            and self.tree_pss_shmem_bytes is not None
        ):
            return self.tree_pss_anon_bytes + self.tree_pss_shmem_bytes
        if self.cgroup_file_bytes is not None:
            return max(0, self.tree_pss_bytes - self.cgroup_file_bytes)
        return None


def memory_snapshot(*, include_process_tree: bool = False) -> MemorySnapshot:
    """Read a snapshot; opt into the more expensive PSS walk sparingly."""

    try:
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (AttributeError, ValueError, OSError):
        peak_rss = None
    rss_kib = _key_values(_read_text("/proc/self/status")).get("VmRSS")
    cgroup = _cgroup_directory()
    stats = _key_values(_read_text(cgroup / "memory.stat"))
    events = _key_values(_read_text(cgroup / "memory.events"))
    tree = (
        _process_tree_memory(os.getpid())
        if include_process_tree
        else (None, None, 0, False, None, None)
    )
    return MemorySnapshot(
        pid=os.getpid(),
        rss_bytes=None if rss_kib is None else rss_kib * 1024,
        rss_peak_bytes=peak_rss,
        cgroup_current_bytes=_bytes_from_text(_read_text(cgroup / "memory.current")),
        cgroup_peak_bytes=_bytes_from_text(_read_text(cgroup / "memory.peak")),
        cgroup_limit_bytes=_bytes_from_text(_read_text(cgroup / "memory.max")),
        cgroup_anon_bytes=stats.get("anon"),
        cgroup_file_bytes=stats.get("file"),
        cgroup_dirty_bytes=stats.get("file_dirty"),
        cgroup_writeback_bytes=stats.get("file_writeback"),
        cgroup_shmem_bytes=stats.get("shmem"),
        cgroup_kernel_bytes=stats.get("kernel"),
        oom=events.get("oom"),
        oom_kill=events.get("oom_kill"),
        tree_rss_bytes=tree[0],
        tree_pss_bytes=tree[1],
        tree_process_count=tree[2],
        tree_pss_complete=tree[3],
        tree_pss_anon_bytes=tree[4],
        tree_pss_shmem_bytes=tree[5],
    )


def memory_summary(*, include_process_tree: bool = False) -> str:
    """Return log-safe diagnostics without trajectory data or credentials."""

    value = memory_snapshot(include_process_tree=include_process_tree)
    limit = (
        value.cgroup_limit_bytes
        if value.cgroup_limit_bytes is not None
        else "unlimited/unknown"
    )
    summary = (
        f"pid={value.pid} rss={_mib(value.rss_bytes)} rss_peak={_mib(value.rss_peak_bytes)} "
        f"cgroup_current={_mib(value.cgroup_current_bytes)} "
        f"cgroup_peak={_mib(value.cgroup_peak_bytes)} cgroup_limit={limit} "
        f"anon={_mib(value.cgroup_anon_bytes)} file={_mib(value.cgroup_file_bytes)} "
        f"dirty={_mib(value.cgroup_dirty_bytes)} writeback={_mib(value.cgroup_writeback_bytes)} "
        f"shmem={_mib(value.cgroup_shmem_bytes)} kernel={_mib(value.cgroup_kernel_bytes)} "
        f"oom={value.oom} oom_kill={value.oom_kill}"
    )
    if include_process_tree:
        summary += (
            f" tree_processes={value.tree_process_count} "
            f"tree_rss={_mib(value.tree_rss_bytes)} tree_pss={_mib(value.tree_pss_bytes)}"
            f" tree_pss_anon={_mib(value.tree_pss_anon_bytes)}"
            f" tree_pss_shmem={_mib(value.tree_pss_shmem_bytes)}"
        )
    return summary


__all__ = ["MemorySnapshot", "memory_snapshot", "memory_summary"]
