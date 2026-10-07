# Disk-backed normalization

Normalizer revision: `2026-10-07.1`.

## What changed

- Exact prefix matching uses a temporary SQLite trie instead of retaining the
  scope's full snapshots and message bodies. Hash collisions still require
  exact canonical-byte equality. Source membership is queried on demand.
- Contributor metadata is stored separately from message history. Metadata
  merging is single-pass, and workers no longer cache every full source.
- A reusable build process pool starts before scope planning, avoiding copies
  of large coordinator graphs in newly forked workers.
- The local state database switches to NORMAL locking and WAL before build
  workers start. Read-only workers see committed inputs while the coordinator
  writes results; completed roots are stored in submission order to preserve
  source-lineage precedence.
- Build admission reserves estimated bytes as well as limiting job count.
  Defaults are an 80 GiB nominal soft budget and a 96 GiB elastic soft ceiling
  for the entire normalization task, not per worker. Finite container limits
  and other non-reclaimable usage can reduce these allowances. They are not
  cgroup reservations or hard memory limits.
- Worker artifacts use strict JSON-mode validation. Internal build failures
  now fail the task instead of dropping trajectories and reporting success.
- Upload futures retain local spool files, not complete serialized payloads;
  multipart buffers are bounded. Validation admission is weighted by file
  bytes, with an optional row-size limit.
- Branch numbering uses disk-backed sorting and rewrites one payload at a
  time. Export accounting reads compact source metadata, not all histories.

Use Worker-local disk for the normalization workspace. Intermediate indexes
and spool files are temporary; no intermediate snapshot or work directory is
added to the output bucket. These temporary workspaces do **not** implement
cross-run or cross-Worker build checkpoint recovery.

## Compatibility and limits

The published trajectory contract, canonical identity rules, audit rules,
prefix semantics and source lineage are unchanged. The normalizer revision
changes cache compatibility. Classification policy and model calls are not
changed by this release.

This release removes the largest duplicated-history and queued-payload
amplifiers; it is **not** an arbitrary-record-size streaming engine. A single
input snapshot, flat candidate or connected root is still materialized.
Routing descriptors and mount planning are scope-wide; source strings may
still be collected for a root. The standard output record size limit remains
in effect. Oversized or un-estimable build jobs fail explicitly rather than
silently omitting data. A record-size or available-memory rejection is not a
successful normalization run.

Memory estimates include decoding/projection multipliers but cannot prove a
peak bound. Shared-worker jobs, allocator behavior, filesystem cache and
single unusually large records still affect memory. Logs distinguish task
process-tree PSS, anonymous memory, file cache, dirty pages and cgroup events.

## First real run

Functional regression tests are required; a production-sized pressure test
is deliberately not part of this change. Run the first full partition with
an independent output prefix and inspect counts, errors, memory and timings
before directing downstream consumers to it. Do not overwrite a partition
that a classification job is currently reading.

Flat S3 objects are not a transaction: an existing object can be replaced
before the new manifest is published. A code rollback does not restore S3
object bodies. No source data is deleted by this upgrade.

The pre-change code is recorded in the GitHub tag
`rollback-pre-tokenplan-disk-20261007` at commit
`6e8fd20352b732aa922ef851d43cd7da651494a3`.
