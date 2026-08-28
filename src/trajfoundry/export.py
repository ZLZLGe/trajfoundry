"""Deterministic JSONL export, lineage, and run manifest creation."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import orjson

from .canonical import trajectory_id
from .io import JsonlShardWriter, file_sha256
from .models import QuarantineRecord, TrajectoryNode
from .output_contract import project_quarantine_record, project_trajectory
from .quality import is_strict_sample, validate_derived_fields

SCHEMA_VERSION = "trajfoundry-v2"


@dataclass
class ExportStats:
    accepted: int = 0
    quarantined_trajectories: int = 0
    quarantined_records: int = 0
    excluded_records: int = 0
    duplicate_trajectories: int = 0
    input_files: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)

    def add_reasons(self, reasons: list[str]) -> None:
        for reason in reasons:
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1


class OutputSet:
    def __init__(self, root: Path, *, max_shard_bytes: int = 512 * 1024 * 1024):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        for reserved in ("accepted", "quarantine", "generations"):
            if (root / reserved).is_symlink():
                raise ValueError(f"output path must not be a symlink: {reserved}")
        self._generation_id = uuid.uuid4().hex
        self._stage = root / f".staging-{self._generation_id}"
        self._generation = root / "generations" / self._generation_id
        self._manifest_tmp = root / f".manifest-{self._generation_id}.tmp"
        self._published = False
        self.accepted = JsonlShardWriter(
            self._stage / "accepted", "trajectories", max_bytes=max_shard_bytes
        )
        self.quarantined = JsonlShardWriter(
            self._stage / "quarantine", "trajectories", max_bytes=max_shard_bytes
        )
        self.records = JsonlShardWriter(
            self._stage / "quarantine", "records", max_bytes=max_shard_bytes
        )
        self.lineage_path = self._generation / "lineage.jsonl"
        self._lineage_tmp = self._stage / "lineage.jsonl.tmp"
        self._lineage_tmp.parent.mkdir(parents=True, exist_ok=True)
        self._lineage = self._lineage_tmp.open("wb")
        self.stats = ExportStats()

    @staticmethod
    def _clear_legacy_files(root: Path) -> None:
        targets: list[Path] = []
        if (root / "accepted").is_dir() and not (root / "accepted").is_symlink():
            targets.extend((root / "accepted").glob("trajectories-*.jsonl"))
            targets.extend((root / "accepted").glob("trajectories-*.jsonl.tmp"))
        if (root / "quarantine").is_dir() and not (root / "quarantine").is_symlink():
            targets.extend((root / "quarantine").glob("trajectories-*.jsonl"))
            targets.extend((root / "quarantine").glob("records-*.jsonl"))
            targets.extend((root / "quarantine").glob("trajectories-*.jsonl.tmp"))
            targets.extend((root / "quarantine").glob("records-*.jsonl.tmp"))
        for path in [
            *targets,
            root / "lineage.jsonl",
            root / "lineage.jsonl.tmp",
        ]:
            if path.is_file() and not path.is_symlink():
                path.unlink()

    @staticmethod
    def _manifest_generation(root: Path) -> str | None:
        try:
            manifest = orjson.loads((root / "manifest.json").read_bytes())
        except (OSError, orjson.JSONDecodeError):
            return None
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, list):
            return None
        for entry in files:
            relative = entry.get("path") if isinstance(entry, dict) else None
            if not isinstance(relative, str):
                continue
            parts = Path(relative).parts
            if len(parts) >= 3 and parts[0] == "generations":
                return parts[1]
        return None

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _cleanup_old_generations(self, keep: set[str]) -> None:
        generations = self.root / "generations"
        if not generations.is_dir() or generations.is_symlink():
            return
        for candidate in generations.iterdir():
            if (
                candidate.name not in keep
                and len(candidate.name) == 32
                and all(character in "0123456789abcdef" for character in candidate.name)
                and candidate.is_dir()
                and not candidate.is_symlink()
            ):
                shutil.rmtree(candidate)

    def write_trajectory(
        self, node: TrajectoryNode, origins: list[dict[str, Any]]
    ) -> str:
        projected = project_trajectory(node)
        validate_derived_fields(node)
        identifier = trajectory_id(node)
        strict = is_strict_sample(node)
        destination = self.accepted if strict else self.quarantined
        destination.write(projected)
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
        self.records.write(project_quarantine_record(record))
        self.stats.quarantined_records += 1
        if record.normalization_audit.tag.value == "excluded":
            self.stats.excluded_records += 1
        self.stats.add_reasons(record.normalization_audit.reason_codes)

    def close(self, *, input_root: str, config_hash: str) -> None:
        self.accepted.close()
        self.quarantined.close()
        self.records.close()
        self._lineage.flush()
        os.fsync(self._lineage.fileno())
        self._lineage.close()
        staged_lineage = self._stage / "lineage.jsonl"
        self._lineage_tmp.replace(staged_lineage)
        files = sorted(
            [
                *self.accepted.paths,
                *self.quarantined.paths,
                *self.records.paths,
                staged_lineage,
            ],
            key=lambda path: str(path.relative_to(self._stage)),
        )
        generation_prefix = Path("generations") / self._generation_id
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "input_root": input_root,
            "config_hash": config_hash,
            "token_estimator": "unicode-word-v1",
            "counts": {
                **self.stats.__dict__,
                "reason_counts": dict(sorted(self.stats.reason_counts.items())),
            },
            "files": [
                {
                    "path": (
                        generation_prefix / path.relative_to(self._stage)
                    ).as_posix(),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
                if path.exists()
            ],
        }
        manifest_bytes = orjson.dumps(
            manifest, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
        )
        with self._manifest_tmp.open("wb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())

        for directory in (
            self._stage / "accepted",
            self._stage / "quarantine",
            self._stage,
        ):
            if directory.is_dir():
                self._fsync_directory(directory)

        previous_generation = self._manifest_generation(self.root)
        self._generation.parent.mkdir(parents=True, exist_ok=True)
        self._stage.replace(self._generation)
        self._fsync_directory(self._generation.parent)
        self._manifest_tmp.replace(self.root / "manifest.json")
        self._published = True
        self._fsync_directory(self.root)

        try:
            self._clear_legacy_files(self.root)
            keep = {self._generation_id}
            if previous_generation:
                keep.add(previous_generation)
            self._cleanup_old_generations(keep)
        except OSError:
            # Cleanup is not part of the committed generation. A later run can
            # safely retry it without invalidating the new manifest.
            pass

    def abort(self) -> None:
        self.accepted.abort()
        self.quarantined.abort()
        self.records.abort()
        if not self._lineage.closed:
            self._lineage.close()
        shutil.rmtree(self._stage, ignore_errors=True)
        self._manifest_tmp.unlink(missing_ok=True)
        if not self._published:
            shutil.rmtree(self._generation, ignore_errors=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is not None:
            self.abort()
