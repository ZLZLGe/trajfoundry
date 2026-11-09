# Repository working instructions

## Canonical checkout

- Treat `/share/gezhilong/trajfoundry` as the canonical checkout for the ordinary TrajFoundry application and library.
- The ordinary DolphinScheduler deployment uses `/share/envs/gzl_trajfoundry_env/bin/python` and imports TrajFoundry from `/share/gezhilong/trajfoundry/src/trajfoundry`; the shared virtual environment uses an editable install, so source changes in this checkout take effect on the next task run.
- The production TokenPlan Spark Cluster scheduler is a separate deployment using `/share/gezhilong/trajfoundry-spark`. Its production config is `/share/gezhilong/trajfoundry-spark-runtime/production/tokenplan.json`; verify the task log and that config's `project_root` before changing Spark scheduler, submit, or normalizer code. Do not assume edits to the ordinary checkout affect this Spark deployment.
- Do not edit, commit, or push from `/home/gezhilong/TrajFoundry/trajfoundry`. It is an older, separate clone and is not the code used by DolphinScheduler.
- Before changing code, verify `pwd`, `git rev-parse --show-toplevel`, and the remote match the intended checkout: `/share/gezhilong/trajfoundry` (`https://github.com/ZLZLGe/trajfoundry.git`) for ordinary TrajFoundry, or `/share/gezhilong/trajfoundry-spark` (`https://github.com/ZLZLGe/trajfoundry-spark.git`) for the Spark deployment.
- Do not commit or push unless explicitly requested; never push `main` from the Spark checkout.

## Scheduler and storage safety

- Jupyter and DolphinScheduler execute independently; `/share` is their shared filesystem.
- Do not modify source files while a DolphinScheduler task is running.
- In the current DolphinScheduler deployment, use the verified Worker-local `/tmp` for S3 normalization state by passing `workspace_parent=Path("/tmp")`. This supersedes the older shared-workspace guidance in `docs/dolphinscheduler.md`; do not place SQLite WAL state under `/share`.
- Keep credentials outside the repository. Never print, inspect in task logs, commit, or push credential values.

## Verification and Git workflow

- Preserve unrelated user changes and inspect `git status` before staging files.
- Run `.venv/bin/python -m pytest -q` before pushing Python changes.
- Push `main` only from the canonical checkout and verify the remote branch commit afterward.
- Review incident and deployment documents for internal hostnames, task identifiers, storage paths, personal paths, and secrets before publishing them to a public repository.
