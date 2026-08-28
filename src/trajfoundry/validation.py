"""Integrity and contract validation for a completed TrajFoundry output set."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .canonical import trajectory_id
from .export import SCHEMA_VERSION
from .io import file_sha256
from .models import AuditTag
from .output_contract import (
    OutputContractError,
    parse_quarantine_record,
    parse_trajectory_record,
)
from .quality import (
    StaleDerivedFieldsError,
    is_strict_sample,
    validate_derived_fields,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SHARD_PATTERN = re.compile(r"^(?:trajectories|records)-[0-9]+\.jsonl$")
_MAX_REPORTED_ERRORS = 100

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
Sha256 = Annotated[str, Field(pattern=_SHA256_PATTERN)]


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _ManifestFile(_StrictContract):
    path: str = Field(min_length=1)
    sha256: Sha256
    bytes: NonNegativeInt


class _ManifestCounts(_StrictContract):
    accepted: NonNegativeInt
    quarantined_trajectories: NonNegativeInt
    quarantined_records: NonNegativeInt
    excluded_records: NonNegativeInt
    duplicate_trajectories: NonNegativeInt
    input_files: NonNegativeInt
    reason_counts: dict[str, NonNegativeInt]


class _Manifest(_StrictContract):
    schema_version: Literal["trajfoundry-v2"]
    created_at: str
    input_root: str
    config_hash: str
    token_estimator: str
    counts: _ManifestCounts
    files: list[_ManifestFile]


class _LineageOrigin(_StrictContract):
    source_ref: str = Field(min_length=1)
    sha256: Sha256
    captured_at: str = ""
    disposition: Literal["pass", "quarantined", "excluded"] | None = None
    reason_codes: list[str] = Field(default_factory=list)


class _LineageRecord(_StrictContract):
    trajectory_id: Sha256
    representative: str = Field(min_length=1)
    origin_count: PositiveInt
    origins: list[_LineageOrigin]

    @model_validator(mode="after")
    def validate_origin_count(self) -> _LineageRecord:
        if self.origin_count != len(self.origins):
            raise ValueError("origin_count must match origins")
        return self


class ValidationReport(BaseModel):
    """A safe-to-display summary of output validation."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: list[str]
    counts: dict[str, int]


@dataclass
class _ErrorCollector:
    messages: list[str] = field(default_factory=list)
    total: int = 0

    def add(self, message: str) -> None:
        self.total += 1
        if len(self.messages) < _MAX_REPORTED_ERRORS:
            self.messages.append(message)

    def finish(self) -> list[str]:
        omitted = self.total - len(self.messages)
        if omitted:
            self.messages.append(f"{omitted} additional validation error(s) omitted")
        return self.messages


@dataclass
class _Observed:
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "manifest_files": 0,
            "checked_files": 0,
            "jsonl_rows": 0,
            "accepted": 0,
            "quarantined_trajectories": 0,
            "quarantined_records": 0,
            "excluded_records": 0,
            "lineage_records": 0,
            "duplicate_trajectories": 0,
            "input_files_declared": 0,
        }
    )
    reason_counts: Counter[str] = field(default_factory=Counter)
    trajectory_ids: Counter[str] = field(default_factory=Counter)
    lineage_ids: Counter[str] = field(default_factory=Counter)
    covered_sources: set[str] = field(default_factory=set)
    trajectory_contract_valid: bool = True
    quarantine_contract_valid: bool = True
    records_contract_valid: bool = True
    lineage_contract_valid: bool = True

    def mark_invalid(self, kind: str | None) -> None:
        if kind in {"accepted", "quarantined_trajectories"}:
            self.trajectory_contract_valid = False
        if kind == "quarantined_trajectories":
            self.quarantine_contract_valid = False
        elif kind == "quarantined_records":
            self.records_contract_valid = False
        elif kind == "lineage":
            self.lineage_contract_valid = False


def _safe_manifest_path(root: Path, relative: str) -> Path | None:
    if "\\" in relative or "\x00" in relative:
        return None
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.split("/")
    ):
        return None
    if pure.as_posix() != relative:
        return None
    try:
        resolved_root = root.resolve()
        candidate = (root / Path(*pure.parts)).resolve()
        candidate.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _file_kind(relative: str) -> str | None:
    pure = PurePosixPath(relative)
    parts = pure.parts
    if len(parts) >= 3 and parts[0] == "generations":
        if not re.fullmatch(r"[0-9a-f]{32}", parts[1]):
            return None
        parts = parts[2:]
        pure = PurePosixPath(*parts)
    if pure.as_posix() == "lineage.jsonl":
        return "lineage"
    if len(pure.parts) != 2 or not _SHARD_PATTERN.fullmatch(pure.name):
        return None
    directory, name = pure.parts
    if directory == "accepted" and name.startswith("trajectories-"):
        return "accepted"
    if directory == "quarantine" and name.startswith("trajectories-"):
        return "quarantined_trajectories"
    if directory == "quarantine" and name.startswith("records-"):
        return "quarantined_records"
    return None


def _row_context(file_index: int, line_no: int, kind: str | None) -> str:
    label = kind or "unsupported"
    return f"manifest files[{file_index}] ({label}) line {line_no}"


def _validate_row(
    value: object,
    *,
    kind: str | None,
    file_index: int,
    line_no: int,
    observed: _Observed,
    errors: _ErrorCollector,
) -> None:
    context = _row_context(file_index, line_no, kind)
    if not isinstance(value, dict):
        observed.mark_invalid(kind)
        errors.add(f"{context} must contain a JSON object")
        return

    if kind in {"accepted", "quarantined_trajectories"}:
        try:
            node = parse_trajectory_record(value)
        except (OutputContractError, ValidationError):
            observed.mark_invalid(kind)
            errors.add(f"{context} violates the TrajectoryNode contract")
            return
        observed.trajectory_ids[trajectory_id(node)] += 1
        try:
            validate_derived_fields(node)
        except StaleDerivedFieldsError:
            errors.add(f"{context} has stale or inconsistent derived quality fields")
        if kind == "accepted":
            if not is_strict_sample(node):
                errors.add(f"{context} is not a strict accepted sample")
        else:
            if is_strict_sample(node):
                errors.add(f"{context} is strict and must not be quarantined")
            if node.normalization_audit is not None:
                observed.reason_counts.update(node.normalization_audit.reason_codes)
        return

    if kind == "quarantined_records":
        try:
            record = parse_quarantine_record(value)
        except (OutputContractError, ValidationError):
            observed.mark_invalid(kind)
            errors.add(f"{context} violates the QuarantineRecord contract")
            return
        if record.normalization_audit.tag == AuditTag.EXCLUDED:
            observed.counts["excluded_records"] += 1
        elif record.normalization_audit.tag == AuditTag.PASS:
            errors.add(f"{context} has an invalid pass disposition")
        observed.covered_sources.add(record.source_ref)
        observed.reason_counts.update(record.normalization_audit.reason_codes)
        return

    if kind == "lineage":
        try:
            lineage = _LineageRecord.model_validate(value)
        except ValidationError:
            observed.mark_invalid(kind)
            errors.add(f"{context} violates the lineage contract")
            return
        observed.lineage_ids[lineage.trajectory_id] += 1
        observed.counts["duplicate_trajectories"] += lineage.origin_count - 1
        observed.covered_sources.update(origin.source_ref for origin in lineage.origins)


def _validate_jsonl(
    path: Path,
    *,
    kind: str | None,
    file_index: int,
    observed: _Observed,
    errors: _ErrorCollector,
) -> None:
    try:
        handle = path.open("rb")
    except OSError:
        errors.add(f"manifest files[{file_index}] could not be opened")
        observed.mark_invalid(kind)
        return

    try:
        with handle:
            for line_no, raw_line in enumerate(handle, start=1):
                observed.counts["jsonl_rows"] += 1
                if kind in {
                    "accepted",
                    "quarantined_trajectories",
                    "quarantined_records",
                }:
                    observed.counts[kind] += 1
                elif kind == "lineage":
                    observed.counts["lineage_records"] += 1
                try:
                    value = orjson.loads(raw_line)
                except orjson.JSONDecodeError:
                    observed.mark_invalid(kind)
                    context = _row_context(file_index, line_no, kind)
                    errors.add(f"{context} is not valid JSON")
                    continue
                _validate_row(
                    value,
                    kind=kind,
                    file_index=file_index,
                    line_no=line_no,
                    observed=observed,
                    errors=errors,
                )
    except OSError:
        observed.mark_invalid(kind)
        errors.add(f"manifest files[{file_index}] could not be read completely")


def _generated_files(root: Path, listed: set[str]) -> set[str]:
    files: set[str] = set()
    patterns = (
        (root / "accepted", "trajectories-*.jsonl"),
        (root / "quarantine", "trajectories-*.jsonl"),
        (root / "quarantine", "records-*.jsonl"),
    )
    for directory, pattern in patterns:
        if directory.is_dir():
            files.update(
                path.relative_to(root).as_posix()
                for path in directory.glob(pattern)
                if path.is_file()
            )
    if (root / "lineage.jsonl").is_file():
        files.add("lineage.jsonl")
    generation_ids = {
        parts[1]
        for relative in listed
        if len(parts := PurePosixPath(relative).parts) >= 3
        and parts[0] == "generations"
    }
    for generation_id in generation_ids:
        generation = root / "generations" / generation_id
        if generation.is_dir() and not generation.is_symlink():
            files.update(
                path.relative_to(root).as_posix()
                for path in generation.rglob("*.jsonl")
                if path.is_file()
            )
    return files


def _compare_manifest_counts(
    manifest: _Manifest,
    observed: _Observed,
    errors: _ErrorCollector,
) -> None:
    expected = manifest.counts
    directly_observed = {
        "accepted": observed.counts["accepted"],
        "quarantined_trajectories": observed.counts["quarantined_trajectories"],
        "quarantined_records": observed.counts["quarantined_records"],
    }
    for name, actual in directly_observed.items():
        if getattr(expected, name) != actual:
            errors.add(f"manifest count {name} does not match the JSONL rows")

    if (
        observed.records_contract_valid
        and expected.excluded_records != observed.counts["excluded_records"]
    ):
        errors.add("manifest count excluded_records does not match the records")
    if (
        observed.lineage_contract_valid
        and expected.duplicate_trajectories != observed.counts["duplicate_trajectories"]
    ):
        errors.add("manifest count duplicate_trajectories does not match lineage")
    if observed.quarantine_contract_valid and observed.records_contract_valid:
        actual_reasons = dict(sorted(observed.reason_counts.items()))
        if expected.reason_counts != actual_reasons:
            errors.add("manifest reason_counts does not match quarantined output")


def validate_output(root: Path) -> ValidationReport:
    """Validate a completed output directory without exposing record contents.

    Validation is streaming at the JSONL level. Error messages identify only a
    manifest entry and line number; values and validation exception text are
    deliberately omitted because trajectory rows can contain sensitive data.
    """

    root = Path(root)
    errors = _ErrorCollector()
    observed = _Observed()
    manifest_path = root / "manifest.json"
    try:
        raw_manifest = orjson.loads(manifest_path.read_bytes())
    except FileNotFoundError:
        errors.add("manifest.json is missing")
        return ValidationReport(
            valid=False, errors=errors.finish(), counts=observed.counts
        )
    except (OSError, orjson.JSONDecodeError):
        errors.add("manifest.json could not be read as valid JSON")
        return ValidationReport(
            valid=False, errors=errors.finish(), counts=observed.counts
        )

    try:
        manifest = _Manifest.model_validate(raw_manifest)
    except ValidationError:
        errors.add("manifest.json violates the manifest contract")
        return ValidationReport(
            valid=False, errors=errors.finish(), counts=observed.counts
        )

    if manifest.schema_version != SCHEMA_VERSION:
        errors.add("manifest.json has an unsupported schema version")

    observed.counts["manifest_files"] = len(manifest.files)
    observed.counts["input_files_declared"] = manifest.counts.input_files
    seen_paths: set[str] = set()
    lineage_entries = 0

    for file_index, entry in enumerate(manifest.files):
        path = _safe_manifest_path(root, entry.path)
        if path is None:
            errors.add(f"manifest files[{file_index}] has an unsafe path")
            continue
        if entry.path in seen_paths:
            errors.add(f"manifest files[{file_index}] duplicates an earlier path")
            continue
        seen_paths.add(entry.path)

        kind = _file_kind(entry.path)
        if kind is None:
            errors.add(f"manifest files[{file_index}] is not a supported output file")
        elif kind == "lineage":
            lineage_entries += 1

        if not path.is_file():
            errors.add(f"manifest files[{file_index}] is missing or not a regular file")
            observed.mark_invalid(kind)
            continue

        observed.counts["checked_files"] += 1
        try:
            actual_bytes = path.stat().st_size
            actual_sha256 = file_sha256(path)
        except OSError:
            errors.add(f"manifest files[{file_index}] could not be inspected")
            observed.mark_invalid(kind)
            continue
        if actual_bytes != entry.bytes:
            errors.add(f"manifest files[{file_index}] byte count does not match")
        if actual_sha256 != entry.sha256:
            errors.add(f"manifest files[{file_index}] sha256 does not match")
        _validate_jsonl(
            path,
            kind=kind,
            file_index=file_index,
            observed=observed,
            errors=errors,
        )

    if lineage_entries != 1:
        errors.add("manifest must list lineage.jsonl exactly once")

    generation_ids = {
        parts[1]
        for relative in seen_paths
        if len(parts := PurePosixPath(relative).parts) >= 3
        and parts[0] == "generations"
    }
    if len(generation_ids) > 1:
        errors.add("manifest files must belong to one immutable generation")

    unlisted_count = len(_generated_files(root, seen_paths) - seen_paths)
    if unlisted_count:
        errors.add(
            f"found {unlisted_count} generated JSONL file(s) not listed in manifest"
        )

    trajectory_count = (
        observed.counts["accepted"] + observed.counts["quarantined_trajectories"]
    )
    if observed.counts["lineage_records"] != trajectory_count:
        errors.add("lineage row count does not match trajectory row count")
    if (
        observed.trajectory_contract_valid
        and observed.lineage_contract_valid
        and observed.trajectory_ids != observed.lineage_ids
    ):
        errors.add("lineage trajectory IDs do not match trajectory content")
    if any(count != 1 for count in observed.trajectory_ids.values()):
        errors.add("trajectory IDs must be globally unique")
    if any(count != 1 for count in observed.lineage_ids.values()):
        errors.add("lineage trajectory IDs must be globally unique")
    if len(observed.covered_sources) != manifest.counts.input_files:
        errors.add("input_files does not match unique lineage and quarantine sources")

    _compare_manifest_counts(manifest, observed, errors)
    messages = errors.finish()
    return ValidationReport(
        valid=errors.total == 0, errors=messages, counts=observed.counts
    )
