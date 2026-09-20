# DolphinScheduler deployment

TrajFoundry can run from a shared Python environment and a shared editable
checkout. In the examples below, replace these placeholders with deployment
paths:

```text
environment: /share/envs/ENV_NAME
source:      /share/OWNER/trajfoundry/src
```

The development host and every selected scheduler worker must mount the same
shared filesystem and have read and traverse permission for both paths.

## Environment installation

The following commands create or update the environment. Run them as the
environment owner, not as `root`:

```bash
PROJECT_ROOT=/share/OWNER/trajfoundry
PYTHON311=/path/to/python3.11
ENV_DIR=/share/envs/ENV_NAME
UV=/path/to/uv
REQUIREMENTS=/tmp/trajfoundry-runtime-requirements.txt

"$UV" venv "$ENV_DIR" \
  --python "$PYTHON311" \
  --prompt trajfoundry

cd "$PROJECT_ROOT"
"$UV" export \
  --frozen \
  --no-dev \
  --no-emit-project \
  --format requirements-txt \
  --no-hashes \
  --python "$PYTHON311" > "$REQUIREMENTS"
"$UV" pip install \
  --python "$ENV_DIR/bin/python" \
  --index-url https://pypi.org/simple \
  --requirements "$REQUIREMENTS"
"$UV" pip install \
  --python "$ENV_DIR/bin/python" \
  --index-url https://pypi.org/simple \
  --no-deps \
  --editable "$PROJECT_ROOT"

chmod -R a+rX "$ENV_DIR"
```

The public index is specified explicitly so a host-specific package mirror
configuration cannot change the installation source. `--frozen` preserves the
versions in `uv.lock`.

If the shared environment parent directory is not writable, use `sudo` only
to create and hand over the target directory, then run `uv` as the owning
user:

```bash
sudo install -d \
  -o "$(id -un)" \
  -g "$(id -gn)" \
  -m 2755 \
  /share/envs/ENV_NAME
```

Do not install packages with `sudo`; doing so leaves mixed ownership inside
the environment.

Verify the interpreter, editable source, and dependencies from outside the
checkout:

```bash
cd /tmp
"$ENV_DIR/bin/python" -c \
  'import sys, trajfoundry; print(sys.executable); print(trajfoundry.__file__)'
"$UV" pip check --python "$ENV_DIR/bin/python"
```

The output should point to the configured environment and checkout:

```text
/share/envs/ENV_NAME/bin/python
/share/OWNER/trajfoundry/src/trajfoundry/__init__.py
```

The imported package path must not point to an older checkout.

## DolphinScheduler configuration

Create a scheduler environment and use configuration equivalent to:

```bash
export TRAJFOUNDRY_ENV=/share/envs/ENV_NAME
export VIRTUAL_ENV=$TRAJFOUNDRY_ENV
export PYTHON_HOME=$TRAJFOUNDRY_ENV/bin/python
export PATH=$TRAJFOUNDRY_ENV/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
```

The tested scheduler deployment renders Python tasks as
`${PYTHON_HOME} <generated-script>`, so `PYTHON_HOME` points to the Python
executable rather than the environment directory. This variable is
deployment-specific; verify the generated command in a task log before using
the configuration on another installation.

### Local filesystem workflow

On the workflow and task definitions:

1. Select a worker group whose workers mount the shared `/share` filesystem.
2. Select the environment created for TrajFoundry.
3. Add the workflow parameters `input_root`, `output_root`, and `input_format`.
4. Paste [the Python node template](../examples/dolphinscheduler_node.py) into
   the Python task.

`input_root` and `output_root` must be absolute paths. `input_format` must be
`freerouter`, `tokenplan`, `sxf`, or `deepinfra`; SXF input roots contain
`.jsonl.zst` objects, while DeepInfra input roots contain individual `.json`
envelopes. The DolphinScheduler tenant must be able to read
and traverse the environment, project, and input paths, and must be able to
create or write `output_root`.

`run_job()` defaults to `resume=True`, validates the published manifest and
shards, and raises an error when validation fails. Individual malformed
captures are retained as quarantine records; empty input, parse failures, and
an all-quarantine batch are logged as warnings for inspection.

### Direct S3 workflows

S3 mode streams each SXF `.jsonl.zst` object through a bounded decompressor and
handles one JSONL row at a time; other formats read one capture object at a
time. One flat JSONL object is written for each final trajectory. The complete
input or output is never staged under `/share`, and AWS CLI is not required.
`/share` is still used for the shared Python environment and editable source
checkout.

S3 credentials are loaded from this fixed path by default:

```text
/share/gezhilong/trajfoundry-secrets/s3_credentials.json
```

Do not put an AK/SK in DolphinScheduler environment variables, workflow or task
parameters, or `rawScript`. The Python node does not accept credential values or
a credential-file parameter. In particular, do not configure
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, or `AWS_SESSION_TOKEN` for this
workflow. This keeps credential material out of generated scripts, scheduler
metadata, and task logs.

The credential file contains one JSON object. The following values are
placeholders only:

```json
{
  "aws_access_key_id": "<ACCESS_KEY_ID>",
  "aws_secret_access_key": "<SECRET_ACCESS_KEY>",
  "aws_session_token": "<SESSION_TOKEN>"
}
```

Omit the entire `aws_session_token` member when the credentials are not
temporary. Never paste real values into this repository, documentation, Python
node, workflow configuration, command line, or logs.

The loader accepts the credential path only when all of these conditions hold:

- it is a regular file and not a symbolic link;
- it is no larger than 64 KiB;
- its mode is exactly `0400` or `0600`, with no group or other access; and
- its owner is the operating-system identity that runs the Dolphin task.

Use mode `0700` for the containing directory. The current task identity has
been verified as UID:GID `1007:1007`, so the initial deployment can be created
as follows. The guarded block refuses to replace any existing credential path.
`sudoedit` keeps secret values out of shell arguments and shell history:

```bash
CREDENTIAL_DIR=/share/gezhilong/trajfoundry-secrets
CREDENTIAL_FILE=$CREDENTIAL_DIR/s3_credentials.json

sudo sh -c '
  set -eu
  credential_dir=$1
  credential_file=$2
  test ! -L "$credential_dir"
  test ! -e "$credential_file"
  test ! -L "$credential_file"
  install -d -o 1007 -g 1007 -m 0700 "$credential_dir"
  install -o 1007 -g 1007 -m 0600 /dev/null "$credential_file"
' sh "$CREDENTIAL_DIR" "$CREDENTIAL_FILE"
sudoedit "$CREDENTIAL_FILE"
sudo chown 1007:1007 "$CREDENTIAL_FILE"
sudo chmod 0400 "$CREDENTIAL_FILE"
sudo stat -c 'type=%F mode=%a owner=%u:%g bytes=%s path=%n' "$CREDENTIAL_FILE"
```

The final `stat` command prints metadata only; do not use `cat`, `jq`, shell
tracing, or another command that writes the file contents to a terminal or
task log. UID/GID mappings can change when the DolphinScheduler tenant or worker
configuration changes. Re-check the task's effective identity after such a
change and adjust the directory and file owner before running the workflow.
The example node calls `run_s3_job()` without credential arguments; the client
loads this fixed file internally.

Create one independent workflow per source, each containing one Python task based on
[`dolphinscheduler_s3_node.py`](../examples/dolphinscheduler_s3_node.py). For
an initial one-day run, configure the task parameters as follows:

| Workflow | `input_format` | `input_uri` | `output_uri` |
| --- | --- | --- | --- |
| `trajfoundry_tokenplan_v005` | `tokenplan` | `s3://agent-trajectory/lakehouse/token-plan/masked-raw/v001/dt=2026-09-09/` | `s3://agent-trajectory/lakehouse/token-plan/normalized/v005/dt=2026-09-09/` |
| `trajfoundry_freerouter_v005` | `freerouter` | `s3://agent-trajectory/lakehouse/free-router/masked-raw/v001/dt=2026-09-09/` | `s3://agent-trajectory/lakehouse/free-router/normalized/v005/dt=2026-09-09/` |
| `trajfoundry_sxf_v005` | `sxf` | `s3://agent-trajectory/lakehouse/SXF/mul-agent-sxf/guixu-data/260821/gpt-5.6-sol/` | `s3://agent-trajectory/lakehouse/SXF/mul-agent-sxf/normalized/v005/dt=2026-08-21/` |
| `trajfoundry_deepinfra_v005` | `deepinfra` | `s3://agent-trajectory/lakehouse/deep-infra/masked-raw/v001/dt=2026-09-14/` | `s3://agent-trajectory/lakehouse/deep-infra/normalized/v005/dt=2026-09-14/` |

Set the shared `endpoint_url` parameter to:

```text
http://d-ceph-ssd-inside.pjlab.org.cn
```

This supplied endpoint uses plain HTTP. Confirm that transmitting trajectory
content over that internal network is approved; use the service's HTTPS
endpoint instead if one is available. Do not guess an HTTPS URL without
confirming it with the storage administrator.

Both input and output values are directory URIs and must end in `/`. Run the
smaller TokenPlan workflow first, inspect its log and manifest, then run the
FreeRouter workflow. Do not run two instances for the same format and date at
the same time because both would compete to publish the same manifest. Enforce
maximum concurrency `1` for each format/date in the scheduler; the application
does not create a distributed S3 lock.

Add a shared workflow parameter named `workspace_parent`. For production S3
jobs, point it at a worker-local writable directory such as `/tmp` (or another
local filesystem with sufficient free space), not `/share`: SQLite aggregation
state is read/write and concurrent NFS access can corrupt it. The directory
must already exist and be writable by the Dolphin tenant. Size the local disk
for the largest expected partition; the state is removed when the task exits.

Only a unique temporary SQLite state directory is created below
`workspace_parent`; source captures and normalized shards are not persisted
locally. `run_s3_job()` removes this state on success and ordinary failure. A
retry rebuilds it from the current S3 inventory and may run on a different
worker. An abrupt worker termination can leave one stale `trajfoundry-state-*`
directory, which can be reviewed and removed after confirming no task uses it.

Publication follows this order:

1. List top-level JSONL objects and remove any object not referenced by the
   currently published manifest.
2. Write one top-level JSONL object per trajectory plus `lineage.jsonl`.
3. Stream those remote objects through the full output validator, ignoring only
   files referenced exclusively by the previous manifest.
4. Upload top-level `manifest.json` as the final publication pointer.
5. Read back and verify the uploaded manifest.
6. Delete files referenced only by the previous manifest.

If upload or validation fails before step 4, an existing manifest remains
unchanged. Because flat keys can reuse names, a same-named object may already
have been overwritten even while the old manifest is still visible. This is an
inherent limitation of the required flat layout. Do not run concurrent writers
for one output partition, and make downstream readers verify manifest hashes.
An empty source prefix, S3 permission failure, or network failure fails the task
without publishing a new manifest.

Deletion failures are not ignored. A failure before publication leaves the old
manifest authoritative; a failure after publication reports the task as failed
even though the new manifest is already authoritative. In either case, the next
retry removes the unlisted residue during step 1 before writing new objects.
Top-level JSONL objects under an output prefix are therefore managed entirely by
TrajFoundry; do not place unrelated JSONL files there.

`max_shard_bytes` is a compatibility parameter name. In the flat v4 contract it
sets the maximum serialized size of one complete trajectory JSONL object; a
trajectory above the limit fails the task and is never split across files.

Once the final manifest PUT begins, a lost network response can make the task
result indeterminate even though S3 accepted the already validated manifest.
If publication or read-back fails at that stage, inspect the current
`manifest.json` before retrying; do not start another same-date instance in
parallel.

### Independent classification workflow

Classification is a separate task and must use a normalized root as its input;
it never calls normalization. Use
[`dolphinscheduler_label_s3_node.py`](../examples/dolphinscheduler_label_s3_node.py)
with parameters such as:

| Parameter | Example |
| --- | --- |
| `input_uri` | `s3://agent-trajectory/lakehouse/token-plan/normalized/v004/dt=2026-09-14/` |
| `output_uri` | `s3://agent-trajectory/lakehouse/token-plan/classified/v001/dt=2026-09-14/` |
| `endpoint_url` | `http://d-ceph-ssd-inside.pjlab.org.cn` |
| `workspace_parent` | `/tmp` |

The input must be the directory containing `manifest.json`; do not pass an
`accepted`, `quarantine`, or `generations` child. Both strict and quarantined
complete trajectories are classified. v004 generation shards and the current
flat v4 contract are supported. Output is flat and contains each complete
normalized row plus its top-level `classification` object.

The worker process must provide these settings without embedding values in the
Python task:

```text
CLASSIFIER_API_URL=https://<approved-host>/v1/chat/completions
CLASSIFIER_MODEL=<available-model-id>
CLASSIFIER_API_KEY=<secret>
```

Inject `CLASSIFIER_API_KEY` through the platform's protected secret mechanism
or the worker service environment. Never put it in workflow parameters,
`rawScript`, source control, or task logs. A key pasted into chat or another
uncontrolled channel must be rotated before use.

The default cache is a deterministic SQLite file below
`workspace_parent/.trajfoundry-label-cache/`. For recovery across worker
restarts, point `state_path` at a worker-local persistent volume and pin retries
to that worker or volume. Do not place a live SQLite cache on NFS and do not run
two tasks against the same cache/output prefix concurrently. Set the scheduler
maximum concurrency to `1` per source and date.

The classifier uses bounded concurrency (four requests in the example). HTTP
429 and 5xx responses are retried; HTTP 404 and 422 indicate a bad endpoint or
missing model configuration and fail the job. Per-trajectory exhausted retries
or invalid model output still produce the original trajectory with
`classification.status="failed"`, allowing a later rerun to inspect and retry
the failed population. Classification also removes unlisted top-level JSONL
before output publication and fails on any cleanup error, so a retry can recover
an interrupted post-manifest cleanup.

## Updating the project

Treat the configured `PROJECT_ROOT` as the active checkout. Ordinary changes
under `src/` take effect on the next task run because the shared environment
uses an editable install. They do not require rebuilding a wheel or
reinstalling the package. Avoid modifying source files while a task is
running, and run the project tests before the next scheduled execution.

Rerun the dependency export and `uv pip install --requirements` commands when
`uv.lock` or project dependencies change. Rerun the final editable-install
command when packaging metadata or console entry points change.
