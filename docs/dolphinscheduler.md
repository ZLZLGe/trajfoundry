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
`freerouter` or `tokenplan`. The DolphinScheduler tenant must be able to read
and traverse the environment, project, and input paths, and must be able to
create or write `output_root`.

`run_job()` defaults to `resume=True`, validates the published manifest and
shards, and raises an error when validation fails. Individual malformed
captures are retained as quarantine records; empty input, parse failures, and
an all-quarantine batch are logged as warnings for inspection.

### Direct S3 workflows

S3 mode reads each source object into memory only while it is being normalized
and streams normalized JSONL shards back to S3. It does not stage the complete
input or output under `/share`, and it does not require AWS CLI. `/share` is
still used for the shared Python environment and editable source checkout.

Configure credentials only through the standard AWS SDK environment variables.
The values below are placeholders; never paste real credentials into the
repository, Python node, workflow parameters, or task logs:

```bash
export AWS_ACCESS_KEY_ID="<ACCESS_KEY_ID_FROM_PLATFORM_SECRET>"
export AWS_SECRET_ACCESS_KEY="<SECRET_ACCESS_KEY_FROM_PLATFORM_SECRET>"
# Set this only when the platform issues temporary credentials:
export AWS_SESSION_TOKEN="<SESSION_TOKEN_FROM_PLATFORM_SECRET>"
```

The application does not accept AK/SK arguments and does not log them. Access
to a scheduler environment may still reveal variables configured directly in
that environment, so use the platform's secret facility when one is available.

Create two independent workflows, each containing one Python task based on
[`dolphinscheduler_s3_node.py`](../examples/dolphinscheduler_s3_node.py). For
the first `2026-09-09` run, configure the task parameters as follows:

| Workflow | `input_format` | `input_uri` | `output_uri` |
| --- | --- | --- | --- |
| `trajfoundry_tokenplan_v002` | `tokenplan` | `s3://agent-trajectory/lakehouse/token-plan/masked-raw/v001/dt=2026-09-09/` | `s3://agent-trajectory/lakehouse/token-plan/normalized/v002/dt=2026-09-09/` |
| `trajfoundry_freerouter_v002` | `freerouter` | `s3://agent-trajectory/lakehouse/free-router/masked-raw/v001/dt=2026-09-09/` | `s3://agent-trajectory/lakehouse/free-router/normalized/v002/dt=2026-09-09/` |

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

Add a shared workflow parameter named `workspace_parent`, for example
`/share/OWNER/trajfoundry-state`. Create that directory once and make it
writable by the Dolphin tenant before the first run. Do not set it to
`Path.cwd()` or `/tmp`: the scheduler task directory is under `/tmp` in this
deployment, and the aggregation state can grow with the input partition.

Only a unique temporary SQLite state directory is created below
`workspace_parent`; source captures and normalized shards are not persisted
locally. `run_s3_job()` removes this state on success and ordinary failure. A
retry rebuilds it from the current S3 inventory and may run on a different
worker. An abrupt worker termination can leave one stale `trajfoundry-state-*`
directory, which can be reviewed and removed after confirming no task uses it.

Publication follows this order:

1. Write immutable JSONL objects under `generations/<run-id>/`.
2. Stream those remote objects through the full output validator.
3. Upload top-level `manifest.json` as the final publication pointer.
4. Read back and verify the uploaded manifest.

If upload or validation fails before step 3, an existing manifest remains
unchanged. Completed generation objects from a failed attempt can remain as
unreferenced orphans and are not automatically deleted. An empty source prefix,
S3 permission failure, or network failure fails the task without publishing a
new manifest; malformed individual captures continue to be represented in
quarantine.

Once the final manifest PUT begins, a lost network response can make the task
result indeterminate even though S3 accepted the already validated manifest.
If publication or read-back fails at that stage, inspect the current
`manifest.json` before retrying; do not start another same-date instance in
parallel.

## Updating the project

Treat the configured `PROJECT_ROOT` as the active checkout. Ordinary changes
under `src/` take effect on the next task run because the shared environment
uses an editable install. They do not require rebuilding a wheel or
reinstalling the package. Avoid modifying source files while a task is
running, and run the project tests before the next scheduled execution.

Rerun the dependency export and `uv pip install --requirements` commands when
`uv.lock` or project dependencies change. Rerun the final editable-install
command when packaging metadata or console entry points change.
