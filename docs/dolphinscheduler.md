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

The public index is specified because the committed lock file records the
company package mirror, which may be unavailable from this host. `--frozen`
still preserves the versions in `uv.lock`.

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

## Updating the project

Treat the configured `PROJECT_ROOT` as the active checkout. Ordinary changes
under `src/` take effect on the next task run because the shared environment
uses an editable install. They do not require rebuilding a wheel or
reinstalling the package. Avoid modifying source files while a task is
running, and run the project tests before the next scheduled execution.

Rerun the dependency export and `uv pip install --requirements` commands when
`uv.lock` or project dependencies change. Rerun the final editable-install
command when packaging metadata or console entry points change.
