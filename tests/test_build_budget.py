import pytest

from trajfoundry.build_budget import GIB, BuildBudget, BuildMemoryError
from trajfoundry.observability import MemorySnapshot


def sample(
    *,
    current: int,
    tree: int,
    file: int = 0,
    dirty: int = 0,
    writeback: int = 0,
    shmem: int = 0,
) -> MemorySnapshot:
    return MemorySnapshot(
        pid=1,
        cgroup_limit_bytes=128 * GIB,
        cgroup_current_bytes=current * GIB,
        cgroup_file_bytes=file * GIB,
        cgroup_dirty_bytes=dirty * GIB,
        cgroup_writeback_bytes=writeback * GIB,
        cgroup_shmem_bytes=shmem * GIB,
        tree_pss_bytes=tree * GIB,
        tree_rss_bytes=tree * GIB,
        tree_pss_complete=True,
        tree_process_count=17,
    )


def test_outstanding_reservations_consume_shared_headroom_before_materialization() -> (
    None
):
    # Other job=80 GiB, our idle processes=2 GiB; after a 16 GiB reserve,
    # only 30 GiB remains. Pending work has not yet inflated memory.current.
    budget = BuildBudget(sample=lambda: sample(current=82, tree=2))
    assert budget.can_submit(0, 4 * GIB, pending=False)
    assert budget.can_submit(24 * GIB, 4 * GIB, pending=True)
    assert not budget.can_submit(28 * GIB, 4 * GIB, pending=True)


def test_materialized_reservations_are_not_double_counted(monkeypatch) -> None:
    values = [sample(current=82, tree=2), sample(current=98, tree=18)]
    budget = BuildBudget(sample=lambda: values.pop(0))
    assert budget.can_submit(0, 4 * GIB, pending=False)
    # Existing tasks now use 16 GiB of their 24 GiB reservation. Remaining
    # reservations=8 GiB plus new=4 GiB still fit the 14 GiB live headroom.
    monkeypatch.setattr(budget, "_sample_at", float("-inf"))
    assert budget.can_submit(24 * GIB, 4 * GIB, pending=True)


def test_clean_page_cache_does_not_block_when_anonymous_working_set_fits() -> None:
    budget = BuildBudget(sample=lambda: sample(current=120, tree=2, file=100))
    assert budget.can_submit(64 * GIB, 8 * GIB, pending=True)


def test_dirty_writeback_and_shmem_are_not_counted_as_clean_headroom() -> None:
    budget = BuildBudget(
        sample=lambda: sample(
            current=120,
            tree=2,
            file=100,
            dirty=10,
            writeback=5,
            shmem=30,
        )
    )
    assert not budget.can_submit(44 * GIB, 8 * GIB, pending=True)


def test_actual_process_tree_limit_is_respected_even_if_estimates_are_small() -> None:
    budget = BuildBudget(sample=lambda: sample(current=95, tree=95))
    assert not budget.can_submit(0, 2 * GIB, pending=True)


def test_no_pending_work_fails_instead_of_waiting_forever_for_other_job() -> None:
    budget = BuildBudget(sample=lambda: sample(current=120, tree=2))
    with pytest.raises(BuildMemoryError, match="headroom"):
        budget.can_submit(0, 4 * GIB, pending=False)


def test_single_oversized_root_fails_before_submission() -> None:
    budget = BuildBudget(sample=lambda: sample(current=2, tree=2))
    with pytest.raises(BuildMemoryError, match="exceeds"):
        budget.can_submit(0, 97 * GIB, pending=False)


def test_unknown_resource_metrics_still_obey_configured_reservations() -> None:
    budget = BuildBudget(sample=lambda: MemorySnapshot(pid=1))
    assert budget.can_submit(80 * GIB, 8 * GIB, pending=True)
    assert not budget.can_submit(92 * GIB, 8 * GIB, pending=True)


def test_invalid_budgets_are_rejected() -> None:
    with pytest.raises(ValueError):
        BuildBudget(target_bytes=0)
    with pytest.raises(ValueError):
        BuildBudget(target_bytes=97 * GIB, elastic_bytes=96 * GIB)


def test_own_mapped_file_growth_does_not_pay_for_pending_anonymous_work(
    monkeypatch,
) -> None:
    # Our PSS grew from 2 to 60 GiB entirely due to clean mappings, while
    # 60 GiB of anonymous work is still only reserved. Counting the mapping
    # as both clean cache and our nonreclaimable usage would over-admit.
    initial = sample(current=67, tree=2)
    inflated = MemorySnapshot(
        pid=1,
        cgroup_limit_bytes=128 * GIB,
        cgroup_current_bytes=125 * GIB,
        cgroup_file_bytes=58 * GIB,
        cgroup_dirty_bytes=0,
        cgroup_writeback_bytes=0,
        cgroup_shmem_bytes=0,
        tree_pss_bytes=60 * GIB,
        tree_pss_complete=True,
        tree_pss_anon_bytes=2 * GIB,
        tree_pss_shmem_bytes=0,
    )
    values = [initial, inflated]
    budget = BuildBudget(sample=lambda: values.pop(0))
    assert budget.can_submit(0, 4 * GIB, pending=False)
    monkeypatch.setattr(budget, "_sample_at", float("-inf"))
    assert not budget.can_submit(60 * GIB, 4 * GIB, pending=True)
