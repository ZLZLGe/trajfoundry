from pathlib import Path

from trajfoundry import observability as module


def test_resolve_host_cgroup_membership(monkeypatch, tmp_path: Path) -> None:
    group = tmp_path / "pod" / "container"
    group.mkdir(parents=True)
    (group / "memory.current").touch()
    values = {
        "/proc/self/cgroup": "0::/pod/container",
        "/proc/self/mountinfo": f"1 2 0:1 / {tmp_path} rw - cgroup2 cgroup rw",
    }
    monkeypatch.setattr(
        module, "_read_text", lambda path: values.get(str(path), "unavailable")
    )
    module._cgroup_directory.cache_clear()
    assert module._cgroup_directory() == group
    module._cgroup_directory.cache_clear()


def test_resolve_namespace_root(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "memory.current").touch()
    values = {
        "/proc/self/cgroup": "0::/",
        "/proc/self/mountinfo": f"1 2 0:1 /host/pod/container {tmp_path} rw - cgroup2 cgroup rw",
    }
    monkeypatch.setattr(
        module, "_read_text", lambda path: values.get(str(path), "unavailable")
    )
    module._cgroup_directory.cache_clear()
    assert module._cgroup_directory() == tmp_path
    module._cgroup_directory.cache_clear()


def test_snapshot_exposes_memory_breakdown_and_unknowns(monkeypatch) -> None:
    values = {
        "/proc/self/status": "VmRSS: 1024 kB",
        "/cg/memory.current": "100000",
        "/cg/memory.peak": "120000",
        "/cg/memory.max": "max",
        "/cg/memory.stat": "anon 40000\nfile 50000\nfile_dirty 1000\nfile_writeback 2000\nshmem 4000\nkernel 10000",
        "/cg/memory.events": "oom 2\noom_kill 1",
    }
    monkeypatch.setattr(module, "_cgroup_directory", lambda: Path("/cg"))
    monkeypatch.setattr(
        module, "_read_text", lambda path: values.get(str(path), "unavailable")
    )
    snapshot = module.memory_snapshot()
    assert snapshot.rss_bytes == 1024 * 1024
    assert snapshot.cgroup_current_bytes == 100000
    assert snapshot.cgroup_limit_bytes is None
    assert snapshot.cgroup_file_bytes == 50000
    assert snapshot.cgroup_dirty_bytes == 1000
    assert snapshot.oom == 2 and snapshot.oom_kill == 1
    assert snapshot.effective_tree_bytes is None
    assert "oom=2 oom_kill=1" in module.memory_summary()


def test_process_tree_uses_pss_and_marks_partial_fallback(monkeypatch) -> None:
    monkeypatch.setattr(module, "_descendant_pids", lambda pid: {1, 2})
    values = {
        "/proc/1/smaps_rollup": "Rss: 100 kB\nPss: 70 kB",
        "/proc/2/smaps_rollup": "Rss: 200 kB\nPss: 150 kB",
    }
    monkeypatch.setattr(
        module, "_read_text", lambda path: values.get(str(path), "unavailable")
    )
    assert module._process_tree_memory(1) == (
        300 * 1024,
        220 * 1024,
        2,
        True,
        None,
        None,
    )
    values["/proc/2/smaps_rollup"] = "unavailable"
    values["/proc/2/status"] = "VmRSS: 200 kB"
    assert module._process_tree_memory(1) == (
        300 * 1024,
        None,
        2,
        False,
        None,
        None,
    )
    assert (
        module.MemorySnapshot(
            pid=1, tree_pss_bytes=1, tree_rss_bytes=2
        ).effective_tree_bytes
        == 2
    )


def test_process_tree_anon_and_shmem_exclude_mapped_file_credit(monkeypatch) -> None:
    monkeypatch.setattr(module, "_descendant_pids", lambda pid: {1, 2})
    values = {
        "/proc/1/smaps_rollup": "Rss: 100 kB\nPss: 70 kB\nPss_Anon: 20 kB\nPss_Shmem: 10 kB\nPss_File: 40 kB",
        "/proc/2/smaps_rollup": "Rss: 200 kB\nPss: 150 kB\nPss_Anon: 50 kB\nPss_Shmem: 30 kB\nPss_File: 70 kB",
    }
    monkeypatch.setattr(
        module, "_read_text", lambda path: values.get(str(path), "unavailable")
    )
    assert module._process_tree_memory(1) == (
        300 * 1024,
        220 * 1024,
        2,
        True,
        70 * 1024,
        40 * 1024,
    )
    snapshot = module.MemorySnapshot(
        pid=1,
        tree_pss_bytes=220,
        tree_pss_complete=True,
        tree_pss_anon_bytes=70,
        tree_pss_shmem_bytes=40,
        cgroup_file_bytes=200,
    )
    assert snapshot.tree_nonreclaimable_bytes == 110


def test_legacy_pss_breakdown_uses_lower_bound_not_rss_sum() -> None:
    snapshot = module.MemorySnapshot(
        pid=1,
        tree_pss_bytes=220,
        tree_pss_complete=True,
        cgroup_file_bytes=200,
    )
    assert snapshot.tree_nonreclaimable_bytes == 20
    assert (
        module.MemorySnapshot(
            pid=1,
            tree_rss_bytes=500,
            cgroup_file_bytes=200,
        ).tree_nonreclaimable_bytes
        is None
    )
