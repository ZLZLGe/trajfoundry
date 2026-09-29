# 2026-09-11 FreeRouter 调度失败复盘：NFS 上的 SQLite WAL 损坏

## 摘要

2026-09-11 的 FreeRouter 首次全量标准化任务在完成输入读取后、构建轨迹时失败。直接错误是：

```text
sqlite3.DatabaseError: database disk image is malformed
```

这不是某条 FreeRouter JSON 解析失败，也没有证据表明是前缀去重算法生成了错误数据。SQLite 在遍历 `snapshots` 表时报告数据库物理结构损坏。

最可能的根因是：任务将 WAL 模式的 SQLite 临时数据库放在 NFSv3 `/share` 上。当前挂载还使用了 `nolock,local_lock=all`，而 SQLite 官方明确说明 WAL 不适用于网络文件系统。

本次运行没有发布任何 FreeRouter 输出。失败后临时数据库已自动清理，输入 S3 数据和仓库代码均未被修改。

## 事故信息

| 项目 | 值 |
| --- | --- |
| 任务类型 | DolphinScheduler Python task |
| Worker | `data-producer-dolphin-worker-1` |
| Task app ID | `44483_72329` |
| 开始时间 | `2026-09-11 14:43:34 +0800` |
| 失败时间 | `2026-09-11 15:33:09 +0800` |
| 运行时长 | 约 49 分 35 秒 |
| 输入格式 | `freerouter` |
| 输入对象数 | 28,390 个 JSON |
| 输入总大小 | 13,759,859,795 bytes，约 12.815 GiB |
| Python | 3.11.14 |
| SQLite | 3.51.1（开发机同一共享 Python 环境检测值；Worker 日志未单独打印） |
| 代码提交 | `305dbe4` |

输入和输出分别为：

```text
s3://agent-trajectory/lakehouse/free-router/masked-raw/v001/dt=2026-09-09/
s3://agent-trajectory/lakehouse/free-router/normalized/v002/dt=2026-09-09/
```

## 失败位置

关键调用链为：

```text
run_s3_job
  -> normalize_source
    -> _build_trajectories
      -> streaming_prefix_leaves
        -> StateStore.snapshots_for_thread
          -> sqlite3.DatabaseError: database disk image is malformed
```

对应源码位置：

- `src/trajfoundry/jobs.py:120`
- `src/trajfoundry/pipeline.py:1237`
- `src/trajfoundry/pipeline.py:894`
- `src/trajfoundry/streaming.py:231`
- `src/trajfoundry/state.py:377`

`_build_trajectories()` 只会在输入扫描完成后执行。因此，该任务已经完成输入对象的读取和快照入库，随后在按 session/thread 读取数据库、构建前缀树时发现数据库损坏。

如果损坏的是单条压缩 snapshot payload，错误通常会发生在 zlib 解压或模型校验阶段；本次错误由 SQLite cursor 直接抛出，说明问题位于 SQLite 数据库层。

## 影响范围

事故后进行了只读核查，结果如下：

- `/share/gezhilong/trajfoundry-state` 下的本次临时目录已经自动删除；
- FreeRouter 目标前缀中没有对象；
- 没有 `manifest.json`；
- 没有 `generations/<run-id>/`；
- 没有未完成的 multipart upload；
- 输入 S3 前缀只被读取，没有被写入；
- Git 工作区保持干净。

失败发生在 `_export()` 之前，S3 output factory 尚未实例化。因此本次运行没有生成需要清理的半成品，也没有覆盖其他 manifest。

## 根因分析

### 已确认的直接原因

SQLite 状态库在持续写入后出现物理损坏，并在轨迹构建阶段首次被读取到损坏页时失败。

### 高可信的首要根因

状态库位于 `/share`，该目录实际是 NFSv3，挂载参数包含：

```text
nolock,local_lock=all
```

与此同时，`StateStore` 无条件启用了：

```python
PRAGMA journal_mode=WAL
PRAGMA synchronous=NORMAL
```

WAL 使用 `-shm` wal-index 协调读者、写者和 checkpoint。SQLite 官方说明，WAL 要求所有访问进程位于同一主机，不能在网络文件系统上使用。SQLite 对远程数据库文件的说明也指出，网络文件系统的锁和同步行为可能导致数据库损坏，rollback journal 只能降低风险，不能让所有远程文件系统组合都变成受支持配置。

本次 FreeRouter 数据量明显大于此前的 TokenPlan。长时间写入、频繁事务和 WAL 自动 checkpoint 更容易暴露短探针无法覆盖的 NFS 一致性问题。此前 SQLite WAL 权限探针和较小的 TokenPlan 成功，只能证明基本操作能够执行，不能证明该组合适合长时间、大规模任务。

由于失败数据库已随临时目录清理，事后无法再对现场执行 `PRAGMA integrity_check`，也没有 NFS 服务端日志可以还原首次损坏发生的时刻。因此，NFS/WAL 是与现有证据最吻合的首要根因假设，但不能写成已经得到法证证明的唯一原因。

### 跨主机只读检查的潜在影响

FreeRouter 运行期间，为检查前缀去重进度，曾从开发机使用 SQLite `mode=ro` 和 `PRAGMA query_only=ON` 对活动数据库执行短时统计查询。

该检查没有执行 INSERT、UPDATE、DELETE、DDL 或显式 checkpoint，也没有复制、移动或删除数据库、WAL、SHM 文件。从 SQL 层面没有主动写入数据库内容。

但是，只读 WAL 连接仍需要参与 WAL/SHM 的读者协调。写端位于 Dolphin Worker，读端位于另一台主机；在 `nolock,local_lock=all` 的 NFS 上，两端不能可靠共享锁和 wal-index 状态。因此：

- 没有证据证明只读查询直接写坏了主数据库；
- 不能排除跨主机读取干扰或放大了 WAL/NFS 的一致性问题；
- 不能把该检查认定为唯一原因；
- 以后禁止从其他主机打开运行中的 SQLite/WAL 数据库。

运行期间只读端还曾出现 `file is not a database`，这说明跨主机读取当时已经无法获得稳定一致的数据库视图。该检查方式不应再次使用。进度与去重统计必须由任务进程自身输出。

### 次要风险

当前 Python 运行时链接 SQLite 3.51.1。SQLite 官方记录了一项影响 WAL 的罕见 WAL-reset corruption bug，并在 3.51.3 中修复。该缺陷需要多连接并发写入或 checkpoint 等特定条件，与本项目正常的单连接设计并不完全吻合，因此不能将它认定为本次首要根因。

升级 SQLite 可以作为防御性措施，但不能替代“将 WAL 移出 NFS”或“停止在 NFS 上使用 WAL”。

## 已排除或不支持的解释

- **不是磁盘空间耗尽**：事故时 `/share` 仍约有 7 TiB 可用，inode 使用率约 1%。
- **不是 Python 脚本语法错误**：任务已正常启动并运行约 50 分钟。
- **不是环境或凭证读取失败**：任务已经开始读取并处理 S3 输入。
- **不是 workspace 权限失败**：SQLite 文件已持续创建和增长。
- **不是普通坏 JSON 导致**：单个输入内容错误会形成 quarantine record 或增加 `parse_failures`，不会产生 SQLite 的 `database disk image is malformed`。
- **没有证据表明前缀算法误删或写坏数据**：错误发生在从 SQLite 读取 snapshot 的过程中，前缀聚合没有完成。

## 修复建议

### 首选方案：使用 Worker 本地临时盘

为 Dolphin Worker 提供容量明确的本地 scratch、Kubernetes `emptyDir` 或其他本地文件系统挂载，并将 `workspace_parent` 指向该目录。

建议至少预留 20–30 GiB，并通过一次完整 FreeRouter 运行测量峰值后再确定正式配额。该目录不一定是 `/tmp`。

这是最可靠且通常性能最好的方案：

- SQLite 与数据库引擎位于同一 Worker；
- 可以继续使用 WAL；
- 不依赖 NFS 的跨主机锁和 mmap 语义；
- 任务状态本来就是临时状态，失败重试会从 S3 重建，不要求跨 Worker 共享。

### 备选方案：必须使用 `/share` 时改用 rollback journal

如果暂时无法提供 Worker 本地盘，第一版应优先使用：

```text
journal_mode=DELETE
synchronous=FULL
```

同时保证每次任务使用唯一临时目录、只有任务进程能够打开数据库，并禁止外部实时查询。

完整 FreeRouter 跑通后，如果 DELETE 模式的 NFS 元数据开销不可接受，再使用同一批数据对 `TRUNCATE + FULL` 做 A/B 测试。不要一开始使用 `PERSIST`、`MEMORY` 或 `OFF`，也不要只把 `synchronous` 改成 FULL 而继续保留普通 WAL 模式。

由于代码当前几乎按每个 capture/trajectory 提交一次事务，rollback journal 可能明显变慢。性能优化应优先考虑安全地批量提交事务，而不是降低 journal 或同步级别。

### 代码防护与可观测性

建议同时增加：

1. 检查 `PRAGMA journal_mode` 返回的实际模式，不要忽略返回值；
2. 输入入库后、轨迹构建前执行 `PRAGMA quick_check`，尽早失败并输出明确阶段；
3. 在任务进程中打印解析进度，替代外部读取活动 SQLite；
4. 任务结束时打印：

   ```text
   eligible_snapshots
   prefix_intermediates
   leaf_snapshots
   stored_trajectories
   ```

5. 验证：

   ```text
   eligible_snapshots = prefix_intermediates + leaf_snapshots
   ```

6. 可考虑将前三项统计写入 manifest，避免任务结束、临时数据库删除后无法复核精确前缀去重数量。

## 重跑前检查清单

- [ ] SQLite 状态库已经迁移到 Worker 本地文件系统，或代码已切换到经过验证的 rollback journal；
- [ ] 不再从开发机或其他 Worker 打开活动数据库；
- [ ] 状态库所在文件系统有足够容量和 inode；
- [ ] 任务内启用了数据库 `quick_check`；
- [ ] 任务日志包含解析和前缀聚合统计；
- [ ] 同一 format/date 的最大并发数为 1；
- [ ] FreeRouter 输出前缀仍为空，或已经明确处理旧 generation/manifest；
- [ ] 完整测试通过后确认 `validation=True`、任务退出码为 0；
- [ ] 成功后确认临时状态目录已自动清理；
- [ ] 检查 S3 `manifest.json`、generation shards 和 `lineage.jsonl`。

## 参考资料

- SQLite Write-Ahead Logging：<https://www.sqlite.org/wal.html>
- SQLite Over a Network：<https://www.sqlite.org/useovernet.html>
- SQLite How To Corrupt Your Database Files：<https://www.sqlite.org/howtocorrupt.html>
- SQLite PRAGMA journal_mode：<https://www.sqlite.org/pragma.html#pragma_journal_mode>
