# TrajFoundry

TrajFoundry normalizes desensitized freerouter HTTP captures into deterministic,
auditable trajectory JSONL. It is an independent project: it does not import,
write to, or depend on AutoData or DataHarness.

The first phase implements trajectory standardization. The data contract is
based on `/root/ailab文档/轨迹标准化/轨迹数据的标准化格式.docx`, with these
confirmed project-level extensions and overrides:

- native `developer` messages remain `developer` messages;
- Responses `instructions` remain an independent string field on the final
  trajectory and are not converted into a message;
- every node includes `instructions`, `metadata.source_name`, and
  `normalization_audit`;
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
  --input /data/回流轨迹/data_feedback_des \
  --output /data/trajfoundry

uv run trajfoundry validate --output /data/trajfoundry
uv run trajfoundry inspect --output /data/trajfoundry
```

The input tree is read-only. A non-empty output directory requires `--resume`:

```bash
uv run trajfoundry normalize --resume \
  --input /data/回流轨迹/data_feedback_des \
  --output /data/trajfoundry
```

Resume mode hashes the current inventory, reuses unchanged captures, reparses
changed captures, removes deleted sources from state, rebuilds global
deduplication, and republishes the output. JSONL shards and lineage are
deterministic for the same inputs and configuration; only the manifest creation
timestamp changes.

## Normalization rules

- Supported endpoints: OpenAI-compatible Responses and Anthropic Messages,
  including streamed and non-streamed responses. Anthropic `count_tokens`
  captures are recorded as excluded non-trajectory records.
- Only a successful, structurally complete terminal response has
  `wire_complete=true`. API errors, transport failures, SSE gaps, truncation,
  and invalid captures are quarantined.
- `--input` defines the dataset boundary. Cumulative snapshots are grouped only
  by `(session_id, thread_id)`; storage directories are provenance, not
  trajectory identity. A snapshot is suppressed only when its complete
  transcript is an exact prefix of a later request history. Every divergent
  maximal leaf is retained.
- Prefix matching compares stable conversational fields: role and content for
  system/developer/user messages; role, content, and tool calls for assistant
  messages; and role, call ID, name, and content for tool results. Reasoning
  payloads are preserved in output but deliberately excluded from prefix
  identity because response and replay envelopes can differ.
- Compaction evidence is position-sensitive prefix identity. Snapshots can be
  folded only when their complete compaction records are identical; differing
  contents, positions, origins, or presence form separate branches.
- Prefix aggregation uses a canonical-message trie. SHA-256 is only a lookup
  accelerator; canonical token bytes remain the equality authority. Memory
  grows with the unique transcript and active branches, not with all repeated
  cumulative histories.
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
- Cross-`session_id` tool-definition aggregation is intentionally deferred;
  the current aggregation boundary remains `(session_id, thread_id)`.
- Server tools remain in `server_tool_calls` and never become client tools or
  tool messages. Their provider result blocks are preserved verbatim, and a
  missing result is represented explicitly as `"result": null`.
- Sub-agents mount only from structured parent thread, parent turn, and concrete
  `spawn_agent`/`Agent` call evidence. When one parent turn has multiple spawn
  calls, a unique inbound child `agent_message` may disambiguate them by exact
  canonical recipient; task-name basename matching is allowed only for a spawn
  without a canonical result name. Explicit marker/recipient conflicts remain
  orphaned. Time, file order, and free-form text are never routing signals.
- `thread_source` is ignored completely for sub-agent detection and mounting;
  only explicit marker/kind, parent, fork, and spawn evidence participate.
- Responses relay mounts additionally require the canonical agent name returned
  by `spawn_agent`, a matching child recipient, and one unique parent-side
  `agent_message` after the corresponding call/result pair has completed.
  Missing or conflicting relay evidence never removes an otherwise proven
  parent/child mount. Raw relay content is retained but never parsed for routing.
- Global semantic hashes deduplicate completed trajectory trees while
  `lineage.jsonl` retains every contributing source capture.
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
├── generations/<run-id>/
│   ├── accepted/trajectories-00000.jsonl
│   ├── quarantine/trajectories-00000.jsonl
│   ├── quarantine/records-00000.jsonl
│   └── lineage.jsonl
└── .state/trajfoundry.sqlite
```

- `accepted`: trajectories passing tool, completion, provider, and sub-agent
  quality gates;
- `quarantine/trajectories`: materialized trajectories that fail a strict gate;
- `quarantine/records`: failed, truncated, invalid, or non-trajectory captures;
- `lineage.jsonl`: semantic trajectory IDs and all source origins;
- `manifest.json`: schema/config versions, counts, reason frequencies, file
  sizes, and SHA-256 checksums;
- `.state`: compressed parsing and deduplication state used by `--resume`.

Trajectories containing opaque compaction context are materialized in full and
written to `quarantine/trajectories` with reason
`opaque_compaction_context`; compaction alone never produces a metadata-only
quarantine record.

Publishing uses immutable generation directories. Data shards are finalized and
fsynced first, then top-level `manifest.json` is atomically replaced as the
single current-generation pointer. The prior generation remains available to
readers during the next publication. `validate` checks checksums, contracts,
input coverage, derived quality fields, strict admission, lineage IDs, and
manifest counts.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
python -m compileall -q src tests
```

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
