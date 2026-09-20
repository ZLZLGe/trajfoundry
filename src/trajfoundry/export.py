"""Deterministic flat JSONL export, lineage, and run manifest creation."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

import orjson

from .canonical import trajectory_id
from .io import file_sha256
from .models import QuarantineRecord, TrajectoryNode
from .output_contract import project_trajectory
from .quality import is_strict_sample, validate_derived_fields

SCHEMA_VERSION = "trajfoundry-v4"

_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_SYNTHESIZED_SESSION_CODE = "metadata_session_id_synthesized"


@dataclass
class ExportStats:
    accepted: int = 0
    quarantined_trajectories: int = 0
    quarantined_records: int = 0
    excluded_records: int = 0
    duplicate_trajectories: int = 0
    input_files: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    skipped_inputs: int = 0
    skip_reason_counts: dict[str, int] = field(default_factory=dict)

    def add_reasons(self, reasons: list[str]) -> None:
        for reason in reasons:
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def add_skipped(self, reason: str) -> None:
        if not reason:
            raise ValueError("skip reason must not be empty")
        self.skipped_inputs += 1
        self.skip_reason_counts[reason] = self.skip_reason_counts.get(reason, 0) + 1


def _session_was_synthesized(node: TrajectoryNode) -> bool:
    if not node.metadata.session_id:
        return True
    audit = node.normalization_audit
    return bool(
        audit and any(issue.code == _SYNTHESIZED_SESSION_CODE for issue in audit.issues)
    )


def trajectory_filename(node: TrajectoryNode, identifier: str | None = None) -> str:
    """Return the stable root filename for one materialized trajectory."""

    sub_session_id = node.metadata.sub_session_id
    if type(sub_session_id) is not int or sub_session_id < 0:
        raise ValueError("sub_session_id must be a non-negative integer")
    if _session_was_synthesized(node):
        if sub_session_id != 0:
            raise ValueError("sessionless trajectories must use sub_session_id 0")
        digest = identifier or trajectory_id(node)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("trajectory identifier must be a sha256 digest")
        return f"no_session_id_{digest}_sub_0.jsonl"

    session_id = node.metadata.session_id
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        raise ValueError(
            "session_id must contain only ASCII letters, digits, '.', '_' or '-' "
            "and must be at most 200 characters"
        )
    return f"{session_id}_sub_{sub_session_id}.jsonl"


def _manifest_paths(root: Path) -> set[str]:
    try:
        manifest = orjson.loads((root / "manifest.json").read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return set()
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        return set()
    paths: set[str] = set()
    for entry in files:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            continue
        pure = PurePosixPath(relative)
        if (
            not pure.is_absolute()
            and pure.as_posix() == relative
            and all(part not in {"", ".", ".."} for part in pure.parts)
            and relative.endswith(".jsonl")
        ):
            paths.add(relative)
    return paths


def _published_flat_manifest_paths(root: Path) -> set[str]:
    """Return top-level JSONL paths from the currently published manifest."""

    return {
        relative
        for relative in _manifest_paths(root)
        if len(PurePosixPath(relative).parts) == 1
    }


def _top_level_jsonl_paths(root: Path) -> set[str]:
    """List every top-level JSONL object managed by this output root."""

    try:
        entries = root.iterdir()
        paths: set[str] = set()
        for path in entries:
            if not path.name.endswith(".jsonl"):
                continue
            if path.is_dir() and not path.is_symlink():
                raise OSError(f"managed JSONL path is a directory: {path.name}")
            paths.add(path.name)
        return paths
    except OSError as error:
        raise OSError("could not enumerate managed output JSONL files") from error


def _remove_top_level_jsonl_paths(root: Path, paths: set[str], *, phase: str) -> None:
    """Remove stale top-level JSONL files, failing on any deletion error."""

    for relative in sorted(paths):
        path = root / relative
        try:
            _unlink_local_output_path(path)
        except OSError as error:
            raise OSError(
                f"could not remove stale output JSONL during {phase} cleanup: "
                f"{relative}"
            ) from error
    if paths:
        OutputSet._fsync_directory(root)


def _unlink_local_output_path(path: Path) -> None:
    """Unlink one managed output path without following symlinks."""

    if path.is_dir() and not path.is_symlink():
        raise OSError("managed output path is a directory")
    path.unlink(missing_ok=True)


def _remove_empty_parents(path: Path, stop: Path) -> None:
    parent = path.parent
    while parent != stop:
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


class OutputSet:
    """Stage a flat output set and publish its manifest last.

    ``max_shard_bytes`` is retained as the public option name for backwards
    compatibility.  In the flat v4 layout it is the maximum serialized byte
    length (including the trailing newline) of one trajectory JSONL object.
    """

    def __init__(self, root: Path, *, max_shard_bytes: int = 512 * 1024 * 1024):
        if max_shard_bytes <= 0:
            raise ValueError("max shard bytes must be positive")
        self.max_shard_bytes = max_shard_bytes
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        # Remove leftovers from an interrupted/failed publication before any
        # new flat object can replace a published file.  The manifest remains
        # authoritative for the files that must be retained.
        published_paths = _published_flat_manifest_paths(self.root)
        managed_paths = _top_level_jsonl_paths(self.root)
        _remove_top_level_jsonl_paths(
            self.root,
            managed_paths - published_paths,
            phase="preflight",
        )
        self._run_id = uuid.uuid4().hex
        self._stage = root / f".staging-{self._run_id}"
        if self._stage.exists():
            raise FileExistsError(f"output staging path already exists: {self._stage}")
        self._stage.mkdir()
        self._manifest_tmp = root / f".manifest-{self._run_id}.tmp"
        self._published = False
        self._trajectory_paths: list[Path] = []
        self._trajectory_names: set[str] = set()
        self._lineage_tmp = self._stage / "lineage.jsonl.tmp"
        self._lineage = self._lineage_tmp.open("wb")
        self.stats = ExportStats()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write_trajectory(
        self, node: TrajectoryNode, origins: list[dict[str, Any]]
    ) -> str:
        projected = project_trajectory(node)
        validate_derived_fields(node)
        identifier = trajectory_id(node)
        filename = trajectory_filename(node, identifier)
        if filename in self._trajectory_names:
            raise ValueError(f"duplicate trajectory output filename: {filename}")

        path = self._stage / filename
        pending = path.with_suffix(path.suffix + ".tmp")
        payload = orjson.dumps(projected, option=orjson.OPT_SORT_KEYS) + b"\n"
        if len(payload) > self.max_shard_bytes:
            raise ValueError(
                "trajectory JSONL object exceeds max_shard_bytes: "
                f"{len(payload)} > {self.max_shard_bytes}"
            )
        with pending.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        pending.replace(path)
        self._trajectory_names.add(filename)
        self._trajectory_paths.append(path)

        strict = is_strict_sample(node)
        if strict:
            self.stats.accepted += 1
        else:
            self.stats.quarantined_trajectories += 1
            if node.normalization_audit:
                self.stats.add_reasons(node.normalization_audit.reason_codes)
        self._lineage.write(
            orjson.dumps(
                {
                    "trajectory_id": identifier,
                    "representative": node.metadata.source_file,
                    "origin_count": len(origins),
                    "origins": origins,
                },
                option=orjson.OPT_SORT_KEYS,
            )
            + b"\n"
        )
        self.stats.duplicate_trajectories += max(0, len(origins) - 1)
        return identifier

    def write_record(self, record: QuarantineRecord) -> None:
        """Account for an input that could not form a trajectory."""

        self.stats.quarantined_records += 1
        if record.normalization_audit.tag.value == "excluded":
            self.stats.excluded_records += 1
        self.stats.add_reasons(record.normalization_audit.reason_codes)

    def write_skipped(self, reason: str) -> None:
        self.stats.add_skipped(reason)

    def close(
        self,
        *,
        input_root: str,
        config_hash: str,
        input_format: Literal[
            "freerouter", "tokenplan", "sxf", "deepinfra"
        ] = "freerouter",
    ) -> None:
        if self._published:
            return
        self._lineage.flush()
        os.fsync(self._lineage.fileno())
        self._lineage.close()
        staged_lineage = self._stage / "lineage.jsonl"
        self._lineage_tmp.replace(staged_lineage)
        files = sorted(
            [*self._trajectory_paths, staged_lineage],
            key=lambda path: path.name,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "input_root": input_root,
            "input_format": input_format,
            "config_hash": config_hash,
            "token_estimator": "unicode-word-v1",
            "counts": {
                **self.stats.__dict__,
                "reason_counts": dict(sorted(self.stats.reason_counts.items())),
                "skip_reason_counts": dict(
                    sorted(self.stats.skip_reason_counts.items())
                ),
            },
            "files": [
                {
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        }
        manifest_bytes = orjson.dumps(
            manifest, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
        )
        with self._manifest_tmp.open("wb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())

        self._fsync_directory(self._stage)
        previous_paths = _manifest_paths(self.root)
        current_paths = {path.name for path in files}
        for path in files:
            path.replace(self.root / path.name)
        self._fsync_directory(self.root)
        self._manifest_tmp.replace(self.root / "manifest.json")
        self._published = True
        self._fsync_directory(self.root)

        # A flat layout cannot atomically swap every data file with the manifest.
        # Publish the new manifest last, then remove only files declared by the
        # previous manifest.  A failed cleanup is surfaced so a retry can use
        # the newly published manifest during its preflight pass.
        for relative in sorted(previous_paths - current_paths):
            old_path = self.root.joinpath(*PurePosixPath(relative).parts)
            try:
                _unlink_local_output_path(old_path)
                _remove_empty_parents(old_path, self.root)
            except OSError as error:
                raise OSError(
                    "could not remove stale output JSONL during "
                    f"post-publication cleanup: {relative}"
                ) from error
        if previous_paths - current_paths:
            self._fsync_directory(self.root)
        shutil.rmtree(self._stage, ignore_errors=True)

    def abort(self) -> None:
        if not self._lineage.closed:
            self._lineage.close()
        shutil.rmtree(self._stage, ignore_errors=True)
        self._manifest_tmp.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is not None:
            self.abort()


__all__ = [
    "SCHEMA_VERSION",
    "ExportStats",
    "OutputSet",
    "trajectory_filename",
]
