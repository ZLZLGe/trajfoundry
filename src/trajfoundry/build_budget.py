"""Byte-weighted build admission, not a replacement for a cgroup limit.

Estimates reserve space for decoding, validation and projection copies.  The
observed process tree and shared cgroup headroom provide additional brakes;
neither estimate nor polling is an OS-enforced memory guarantee.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .observability import MemorySnapshot, memory_snapshot

LOGGER = logging.getLogger(__name__)
GIB = 1024**3
DEFAULT_BUILD_BUDGET = 80 * GIB
DEFAULT_BUILD_ELASTIC_BUDGET = 96 * GIB


class BuildMemoryError(RuntimeError):
    """An operation cannot safely be admitted with the available budget."""


class BuildBudget:
    def __init__(
        self,
        *,
        target_bytes: int = DEFAULT_BUILD_BUDGET,
        elastic_bytes: int = DEFAULT_BUILD_ELASTIC_BUDGET,
        sample: Callable[[], MemorySnapshot] | None = None,
    ) -> None:
        if target_bytes <= 0 or elastic_bytes < target_bytes:
            raise ValueError("build memory budgets must be positive and ordered")
        self.target_bytes = target_bytes
        self.elastic_bytes = elastic_bytes
        self.sample = sample or (lambda: memory_snapshot(include_process_tree=True))
        self._snapshot: MemorySnapshot | None = None
        self._sample_at = float("-inf")
        self._log_at = float("-inf")
        self._baseline_tree: int | None = None

    def snapshot(self) -> MemorySnapshot:
        now = time.monotonic()
        if self._snapshot is None or now - self._sample_at >= 2.0:
            self._snapshot = self.sample()
            self._sample_at = now
        return self._snapshot

    def can_submit(self, reserved: int, estimate: int, *, pending: bool) -> bool:
        snapshot = self.snapshot()
        limit = snapshot.cgroup_limit_bytes
        # Keep at least 12.5% of a finite container for fluctuations.  The
        # configured 80/96 GiB budgets scale DOWN on smaller containers.
        ceiling = self.elastic_bytes
        if limit is not None and limit > 0:
            ceiling = min(ceiling, int(limit * 0.75))
        if estimate > ceiling:
            raise BuildMemoryError(
                "root estimated working set exceeds build budget: "
                f"estimate={estimate / GIB:.2f}GiB ceiling={ceiling / GIB:.2f}GiB; "
                "no sources were marked skipped"
            )

        tree = snapshot.effective_tree_bytes
        if self._baseline_tree is None:
            self._baseline_tree = tree or snapshot.rss_bytes or 0
        current = snapshot.cgroup_current_bytes
        available: int | None = None
        if limit is not None and current is not None:
            # Clean file cache is reclaimable in normal operation.  Do not
            # wait forever for page cache to vanish from memory.current.
            clean_file = max(
                0,
                (snapshot.cgroup_file_bytes or 0)
                - (snapshot.cgroup_shmem_bytes or 0)
                - (snapshot.cgroup_dirty_bytes or 0)
                - (snapshot.cgroup_writeback_bytes or 0),
            )
            # Only credit our anonymous/shmem PSS here. File PSS is already
            # included in clean_file; subtracting the whole process tree
            # would let pending reservations consume the same headroom twice.
            own_nonreclaimable = snapshot.tree_nonreclaimable_bytes or 0
            other = max(0, current - clean_file - own_nonreclaimable)
            available = max(0, limit - other - int(limit * 0.125))
        nominal = min(self.target_bytes, ceiling)
        projected = max(
            (tree or 0) + estimate,
            self._baseline_tree + reserved + estimate,
        )
        if available is None or available >= projected:
            nominal = ceiling
        admitted = projected <= nominal
        if available is not None:
            admitted = admitted and projected <= available
        if not admitted:
            if time.monotonic() - self._log_at >= 30:
                LOGGER.warning(
                    "【内存调度】暂缓新根轨迹：已预留=%.2fGiB 新任务估计=%.2fGiB "
                    "软预算=%.2fGiB 弹性上限=%.2fGiB 任务进程树=%s 可用估计=%s",
                    reserved / GIB,
                    estimate / GIB,
                    self.target_bytes / GIB,
                    ceiling / GIB,
                    "unknown" if tree is None else f"{tree / GIB:.2f}GiB",
                    "unknown" if available is None else f"{available / GIB:.2f}GiB",
                )
                self._log_at = time.monotonic()
            if not pending:
                # Nothing in this job can drain to release the budget.  Do
                # not hang indefinitely behind another scheduler task.
                raise BuildMemoryError(
                    "insufficient shared-worker memory headroom for the next root; "
                    "retry after other tasks release memory"
                )
        return admitted


__all__ = [
    "DEFAULT_BUILD_BUDGET",
    "DEFAULT_BUILD_ELASTIC_BUDGET",
    "BuildBudget",
    "BuildMemoryError",
]
