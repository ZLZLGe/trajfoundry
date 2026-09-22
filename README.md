# TrajFoundry

TrajFoundry normalizes desensitized Freerouter and DeepInfra captures, TokenPlan
feedback envelopes, and SXF `.jsonl.zst` capture streams into deterministic,
auditable trajectory JSONL. A separate classification job can add scenario,
capability, model, and harness labels to already-normalized trajectories. It is
an independent project: it does not import, write to, or depend on AutoData or
DataHarness.

The first phase implements trajectory standardization. The data contract is
based on `/root/ailab文档/轨迹标准化/轨迹数据的标准化格式.docx`, with these
confirmed project-level extensions and overrides:

- native `developer` messages remain `developer` messages;
- Responses `instructions` remain an independent string field on the final
  trajectory and are not converted into a message;
- every node includes `instructions`, `normalization_audit`, and metadata for
  model, user/session identity, and normalized source classification;
- Responses reasoning items are preserved in `assistant.reasoning`, while
  Anthropic thinking blocks remain in `assistant.reasoning_details`.
- Responses `agent_message` items are not promoted to a conversational role;
  each raw item is preserved losslessly in `agent_messages` with its origin and
  provider item index, while routing uses only `id`, `author`, and `recipient`.
- Responses `compaction` items are preserved losslessly as opaque
  `compaction_items` records with their provider origin and item index. They are
  never converted into messages, decrypted, summarized, or interpreted.

## Quick start

```bash
uv sync --python 3.11 --dev --locked

uv run trajfoundry normalize \
  --input-format freerouter \
  --input /data/回流轨迹/data_feedback_des \
  --output /data/trajfoundry

uv run trajfoundry validate --output /data/trajfoundry
uv run trajfoundry inspect --output /data/trajfoundry
```

TokenPlan uses an explicit input format so its manifests and media assets are
not mistaken for request captures:

```bash
uv run trajfoundry normalize \
  --input-format tokenplan \
  --input /data/回流轨迹/lakehouse/token-plan/raw/v001 \
  --output /data/trajfoundry-tokenplan
```

SXF partitions are newline-delimited compressed streams.  They are decompressed
and parsed one row at a time, so the complete partition is never staged locally:

```bash
uv run trajfoundry normalize \
  --input-format sxf \
  --input /data/sxf/partitions \
  --output /data/trajfoundry-sxf
```

DeepInfra partitions contain one envelope per JSON object:

```bash
uv run trajfoundry normalize \
  --input-format deepinfra \
  --input /data/deep-infra/masked-raw/v001 \
  --output /data/trajfoundry-deepinfra
```

Scheduler jobs can also normalize directly from one S3 prefix to another with
`trajfoundry.jobs.run_s3_job`. The S3 mode lists and reads source objects one at
a time and writes one JSONL object per final trajectory; it does not copy the
complete input partition or normalized output onto `/share`. Global aggregation still
uses one temporary local SQLite database in the task workspace. The state is
deleted when the task finishes and is rebuilt from the source inventory on a
retry, so S3 jobs do not resume across workers.

S3 publication lists existing top-level JSONL objects, writes and validates the
new flat JSONL objects, then uploads the top-level `manifest.json` last. The
manifest is a write-only publication pointer; stale JSONL is removed only after
that commit. A cleanup failure fails the job, and the next retry lists the
residue again and removes it after the replacement manifest is published. See
[`docs/dolphinscheduler.md`](docs/dolphinscheduler.md) for normalization and
classification workflow configuration.

The input tree is read-only. A non-empty output directory requires `--resume`:

```bash
uv run trajfoundry normalize --resume \
  --input /data/回流轨迹/data_feedback_des \
  --output /data/trajfoundry
```

Resume mode hashes the current inventory, reuses unchanged captures, reparses
changed captures, removes deleted sources from state, rebuilds global
deduplication, and republishes the output. Trajectory JSONL files and lineage are
deterministic for the same inputs and configuration; only the manifest creation
timestamp changes.

## Normalization rules

- Supported endpoints: OpenAI-compatible Responses and Anthropic Messages,
  plus OpenAI-compatible Chat Completions. Streamed and non-streamed responses
  are supported; SXF SSE bodies are decoded before provider parsing. TokenPlan's
  normalized-final stream objects are parsed as final objects without changing
  the original request's `stream` value.
  Anthropic `count_tokens` captures are recorded as excluded non-trajectory
  records.
- TokenPlan discovery selects only `req_*.json`; partition manifests and media
  files are not capture inputs. Media-bearing envelopes are normalized without
  copying media bytes. When a body contains an explicit `$media_ref:<part_id>`
  reference, the published trajectory includes the optional
  `multimodal_file_mapping` entry for that part and its stored `object_name`;
  unused attachments are omitted.
- DeepInfra discovery selects every `*.json` object and adapts each envelope
  independently before provider parsing.
- Only a successful, structurally complete terminal response has
  `wire_complete=true`. API errors, transport failures, SSE gaps, truncation,
  and invalid captures are quarantined. Explicit Anthropic
  `stop_reason=max_tokens` and Chat Completions
  `finish_reason=length|content_filter` are treated as truncation without
  inventing a `termination` value.
- New runs use manifest schema `trajfoundry-v4`, which records `input_format`,
  `skipped_inputs`, and `skip_reason_counts` and publishes one trajectory per
  flat JSONL file. Validation remains backward compatible with v2/v3 manifests.
- `max_shard_bytes` (CLI: `--max-shard-mib`) is retained as a compatibility
  name, but in the flat v4 layout it limits the serialized size of one complete
  trajectory object. An oversized trajectory fails publication instead of
  being split across files.
- `--input` defines the dataset boundary. Storage directories are provenance,
  not trajectory identity. Captures with a real `session_id` are aggregated in
  a session scope while keeping `(session_id, thread_id)` as the prefix
  boundary. Captures without a session are aggregated by `user_id` across
  request/thread labels; missing users use the shared `no_user_id` scope.
  `request_id` is never promoted to a session. Published metadata projects
  missing identities as `session_id="no_session_id"` and
  `user_id="no_user_id"`; audit markers keep those synthesized values distinct
  from provider IDs that literally use either sentinel string.
- Every root and mounted child contains `metadata.sub_session_id`. Divergent
  final branches of a real session are numbered `0..N` in stable creation-time
  order. A trajectory whose session was synthesized always uses `0`; its
  filename also contains the stable semantic trajectory hash so unrelated
  user-scoped branches cannot collide.
- A snapshot is suppressed only when its complete transcript is an exact prefix
  of a later request history in the same scope. Every divergent maximal leaf is
  retained.
- Prefix matching compares stable conversational fields: role and content for
  system/developer/user messages; role, content, and tool calls for assistant
  messages; and role, call ID, name, and content for tool results. Reasoning
  payloads are preserved in output but deliberately excluded from prefix
  identity because response and replay envelopes can differ.
- Compaction evidence is position-sensitive prefix identity. Snapshots can be
  folded only when their complete compaction records are identical; differing
  contents, positions, origins, or presence form separate branches.
- Prefix aggregation uses a canonical-message trie with separate postings for
  active history extenders and complete transcript endpoints. A current full
  transcript probes extenders; each current history prefix probes endpoints.
  SHA-256 is only a lookup accelerator; canonical token bytes remain the
  equality authority. This avoids scanning unrelated active leaves and keeps
  memory tied to unique transcript structure and active branches.
- Final messages, instructions, model, harness, and termination are taken from
  the selected leaf capture. Prefix contributors never backfill or rewrite the
  leaf conversation; only tool definitions, server-tool records, agent-message
  evidence, audit issues, and lineage are merged.
- Equal terminal retransmissions are merged only when their complete flat
  semantics and routing identity match; their lineage is unioned, while tool,
  termination, and linkage variants remain separate.
- Client tool definitions are merged across proven prefix contributors. The
  richer compatible definition wins deterministically; incompatible same-name
  schemas quarantine the trajectory.
- Tool definitions are merged only among prefix contributors to the same
  aggregation scope and compaction signature. There is no cross-session or
  cross-user tool registry: a tool seen in a different trajectory cannot
  backfill this one.
- Server tools remain in `server_tool_calls` and never become client tools or
  tool messages. Their provider result blocks are preserved verbatim, and a
  missing result is represented explicitly as `"result": null`.
- Sub-agents mount only from structured parent thread, parent turn, and concrete
  `spawn_agent`/`Agent` call evidence. When one parent turn has multiple spawn
  calls, a unique inbound child `agent_message` may disambiguate them by exact
  canonical recipient; task-name basename matching is allowed only for a spawn
  without a canonical result name. Explicit marker/recipient conflicts remain
  orphaned. Time, file order, and free-form text are never routing signals.
- When one child thread has divergent maximal leaves, the first version mounts
  only the leaf with the largest `history + response` message count. Stable leaf
  order breaks ties; unselected branches remain explicit orphan trajectories.
  Child context-compaction semantics are not interpreted yet.
- `thread_source` is ignored completely for sub-agent detection and mounting;
  only explicit marker/kind, parent, fork, and spawn evidence participate.
- Responses relay mounts additionally require the canonical agent name returned
  by `spawn_agent`, a matching child recipient, and one unique parent-side
  `agent_message` after the corresponding call/result pair has completed.
  Missing or conflicting relay evidence never removes an otherwise proven
  parent/child mount. Raw relay content is retained but never parsed for routing.
- Semantic hashes deduplicate completed trajectory trees within their normalized
  identity boundary (`user_id` and `session_id`); source file, timestamp, and
  other provenance remain outside the hash. `lineage.jsonl` retains every
  contributing source capture.
- Session-wide sub-agent planning retains only compact routing evidence. Full
  nodes are reloaded and materialized one connected root at a time.

Request and response header objects are never written to output. Only a small
identity-header allowlist is read during parsing. Authorization, forwarded IP,
proxy credentials, and response headers are discarded before provider parsing.
Message content, reasoning details, tool arguments, and tool results are not
redacted or rewritten beyond format normalization.

Published trajectory rows use one strict projection shared by hashing, durable
state, export, and validation. Required empty strings/arrays remain present;
optional empty fields are omitted. `termination=""` is a normal current value
and is not inferred from a provider status or stop reason.

## Output layout

```text
/data/trajfoundry/
├── manifest.json
├── <session_id>_sub_<n>.jsonl
├── no_session_id_<trajectory_hash>_sub_0.jsonl
├── lineage.jsonl
└── .state/trajfoundry.sqlite
```

- each trajectory file contains exactly one complete root trajectory JSON row;
  both strict and quarantined complete trajectories are materialized, with
  `normalization_audit.tag` retaining their quality status;
- captures that cannot form a trajectory do not produce a trajectory file;
  their counts and reasons remain in `manifest.json`;
- `lineage.jsonl`: semantic trajectory IDs and all source origins;
- `manifest.json`: schema/config versions, counts, reason frequencies, file
  sizes, and SHA-256 checksums;
- `.state`: compressed parsing and deduplication state used by `--resume`.

Trajectories containing opaque compaction context are materialized in full with
reason `opaque_compaction_context`; compaction alone never produces a
metadata-only quarantine record.

Data files are finalized first and `manifest.json` is published last as the
authoritative file list. A flat object layout cannot atomically replace many
same-named files: during a rerun, an old manifest can briefly point to a newly
overwritten file. Do not run two writers for the same output partition, and
make readers verify the manifest checksums. `validate` checks checksums,
contracts, input coverage, derived quality fields, strict admission, lineage
IDs, and manifest counts. Top-level JSONL files not listed by the current
manifest are treated as interrupted-run residue and removed after the next
replacement manifest is published; inability to remove them fails the run.

## Trajectory classification

Classification is independent from normalization. `run_s3_label_job()` reads a
normalized root manifest, processes every complete trajectory it references
(strict, quarantined, and orphan roots), and writes a second flat dataset. It
never invokes or changes the normalization pipeline. v004 shard input remains
readable; current v4 flat input is also supported.

Each output row is the complete normalized row plus one top-level
`classification` object. Successful results contain the exact full taxonomy
objects selected by the model, the fixed capability labels, and model/harness
labels copied from the trajectory. A model or request failure still writes the
complete row with `classification.status="failed"` and a bounded reason code.
No `trajectory_id` is added to a trajectory row.

The bundled scenario taxonomy is versioned by content hash. The model returns
only taxonomy IDs and capability names; TrajFoundry validates them and expands
IDs from the bundled JSON. OpenAI-compatible Chat Completions configuration is
read from `CLASSIFIER_API_URL`, `CLASSIFIER_MODEL`, and `CLASSIFIER_API_KEY`.
The URL must include `/chat/completions`. Keep the key in a worker-side secret
injection mechanism, never in source, task parameters, or scheduler scripts.

Classification uses bounded concurrency and a persistent SQLite cache under
`workspace_parent/.trajfoundry-label-cache/` by default. Pass `state_path` to
place that cache on a worker-local persistent volume. HTTP 429 and 5xx responses
are retried with bounded exponential backoff; HTTP 404 and 422 fail the job as
configuration errors. Candidate rows are completed locally before any output
object is changed, and `manifest.json` is published last. Classification reads
and validates the normalized input manifest, but treats its own output manifest
as write-only. It lists existing top-level JSONL before model calls and removes
unlisted JSONL only after the replacement manifest is published; a cleanup
failure is recoverable on retry.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
python -m compileall -q src tests
```

For scheduler deployment with a shared Python environment and editable
checkout, see [`docs/dolphinscheduler.md`](docs/dolphinscheduler.md).

The implementation has also been smoke-tested on 56 real captures (43
Anthropic and 13 Responses), including successful SSE, API failures,
`count_tokens`, prefix aggregation, validation, and byte-stable resume output.
The prefix rules were additionally checked on the first five complete session
directories in deterministic path order: 599 captures parsed successfully,
545 intermediate snapshots collapsed to 24 leaves, validation passed, and a
resume run reproduced byte-identical trajectory and lineage shards. This
acceptance run is not a substitute for rerunning the earlier 5,000-capture
audit or for a full production run.

## Project ledger

The Feishu ledger [TrajFoundry 项目台账](https://aicarrier.feishu.cn/docx/LYTpdbTAQoPiFGxVnqQcq9e6nCb)
is the single project record for plans, decisions, implementation changes,
verification evidence, and future quality evaluation/classification work.
