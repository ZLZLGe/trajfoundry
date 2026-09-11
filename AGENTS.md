# Repository working instructions

## Canonical checkout

- Treat `/share/gezhilong/trajfoundry` as the only active checkout for development, testing, commits, and pushes.
- DolphinScheduler uses `/share/envs/gzl_trajfoundry_env/bin/python` and imports TrajFoundry from `/share/gezhilong/trajfoundry/src/trajfoundry`; the shared virtual environment uses an editable install, so source changes in this checkout take effect on the next task run.
- Do not edit, commit, or push from `/home/gezhilong/TrajFoundry/trajfoundry`. It is an older, separate clone and is not the code used by DolphinScheduler.
- Before changing or pushing code, verify both `pwd` and `git rev-parse --show-toplevel` resolve to `/share/gezhilong/trajfoundry`, and verify `git remote get-url origin` is `https://github.com/ZLZLGe/trajfoundry.git`.

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
