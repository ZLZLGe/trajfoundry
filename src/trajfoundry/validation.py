"""Integrity and contract validation for a completed TrajFoundry output set."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated, BinaryIO, Literal, Protocol

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .canonical import trajectory_id
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
    skipped_inputs: NonNegativeInt = 0
    skip_reason_counts: dict[str, NonNegativeInt] = Field(default_factory=dict)


class _Manifest(_StrictContract):
    schema_version: Literal["trajfoundry-v2", "trajfoundry-v3"]
    created_at: str
    input_root: str
    input_format: Literal["freerouter", "tokenplan"] = "freerouter"
    config_hash: str
    token_estimator: str
    counts: _ManifestCounts
    files: list[_ManifestFile]

    @model_validator(mode="after")
    def require_v3_skip_fields(self) -> _Manifest:
        """Keep v2 readable while requiring the new accounting in v3."""

        if self.schema_version == "trajfoundry-v3":
            if "input_format" not in self.model_fields_set:
                raise ValueError("v3 manifest requires input_format")
            if "skipped_inputs" not in self.counts.model_fields_set:
                raise ValueError("v3 manifest requires counts.skipped_inputs")
            if "skip_reason_counts" not in self.counts.model_fields_set:
                raise ValueError("v3 manifest requires counts.skip_reason_counts")
        return self


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


@dataclass(frozen=True)
class ValidationFile:
    """A backend object and the byte size reported for its binary stream."""

    size: int
    stream: BinaryIO


class ValidationBackend(Protocol):
    """Storage-neutral access required to validate one output generation."""

    def open_file(self, relative_path: str) -> ValidationFile | None:
        """Open a manifest file, or return ``None`` when it does not exist."""
        ...

    def iter_generated_jsonl_paths(
        self, generation_ids: frozenset[str]
    ) -> Iterable[str]:
        """Yield generated JSONL paths that could have been omitted from manifest."""
        ...


@dataclass
class _ErrorCollector:
    messages: list[str] = field(default_factory=list)
    total: int = 0

    def add(self, message: str) -> None:
        self.total += 1
        if len(self.messages) < _MAX_REPORTED_ERRORS:
            self.messages.append(message)

    def extend(self, other: _ErrorCollector) -> None:
        available = _MAX_REPORTED_ERRORS - len(self.messages)
        if available > 0:
            self.messages.extend(other.messages[:available])
        self.total += other.total

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
            "skipped_inputs": 0,
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


def _is_safe_manifest_path(relative: str) -> bool:
    if "\\" in relative or "\x00" in relative:
        return False
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.split("/")
    ):
        return False
    return pure.as_posix() == relative


def _safe_manifest_path(root: Path, relative: str) -> Path | None:
    if not _is_safe_manifest_path(relative):
        return None
    pure = PurePosixPath(relative)
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


@dataclass(frozen=True)
class _StreamResult:
    bytes_read: int
    sha256: str
    complete: bool


def _validate_jsonl_stream(
    stream: BinaryIO,
    *,
    kind: str | None,
    file_index: int,
    observed: _Observed,
    errors: _ErrorCollector,
) -> _StreamResult:
    digest = hashlib.sha256()
    bytes_read = 0
    pending = bytearray()
    line_no = 0
    scan_from = 0
    complete = False
    try:
        while True:
            try:
                chunk = stream.read(1024 * 1024)
            # Backends may surface SDK-specific transport exceptions here.
            except Exception:  # noqa: BLE001
                observed.mark_invalid(kind)
                errors.add(f"manifest files[{file_index}] could not be read completely")
                break
            if not chunk:
                complete = True
                break
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                observed.mark_invalid(kind)
                errors.add(f"manifest files[{file_index}] could not be read completely")
                break
            digest.update(chunk)
            bytes_read += len(chunk)
            pending.extend(chunk)

            start = 0
            while (newline := pending.find(b"\n", scan_from)) >= 0:
                line_no += 1
                _validate_jsonl_row(
                    bytes(pending[start : newline + 1]),
                    kind=kind,
                    file_index=file_index,
                    line_no=line_no,
                    observed=observed,
                    errors=errors,
                )
                start = newline + 1
                scan_from = start
            if start:
                del pending[:start]
            scan_from = len(pending)

        if complete and pending:
            line_no += 1
            _validate_jsonl_row(
                bytes(pending),
                kind=kind,
                file_index=file_index,
                line_no=line_no,
                observed=observed,
                errors=errors,
            )
    finally:
        try:
            stream.close()
        # Closing a remote response can also raise an SDK-specific exception.
        except Exception:  # noqa: BLE001
            observed.mark_invalid(kind)
            errors.add(f"manifest files[{file_index}] could not be read completely")

    return _StreamResult(
        bytes_read=bytes_read,
        sha256=digest.hexdigest(),
        complete=complete,
    )


def _validate_jsonl_row(
    raw_line: bytes,
    *,
    kind: str | None,
    file_index: int,
    line_no: int,
    observed: _Observed,
    errors: _ErrorCollector,
) -> None:
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
        return
    _validate_row(
        value,
        kind=kind,
        file_index=file_index,
        line_no=line_no,
        observed=observed,
        errors=errors,
    )


class _UnsafeBackendPathError(Exception):
    """A local manifest path resolved outside the validation root."""


class _LocalValidationBackend:
    def __init__(self, root: Path) -> None:
        self._root = root

    def open_file(self, relative_path: str) -> ValidationFile | None:
        path = _safe_manifest_path(self._root, relative_path)
        if path is None:
            raise _UnsafeBackendPathError
        if not path.is_file():
            return None
        size = path.stat().st_size
        return ValidationFile(size=size, stream=path.open("rb"))

    def iter_generated_jsonl_paths(
        self, generation_ids: frozenset[str]
    ) -> Iterable[str]:
        files: set[str] = set()
        patterns = (
            (self._root / "accepted", "trajectories-*.jsonl"),
            (self._root / "quarantine", "trajectories-*.jsonl"),
            (self._root / "quarantine", "records-*.jsonl"),
        )
        for directory, pattern in patterns:
            if directory.is_dir():
                files.update(
                    path.relative_to(self._root).as_posix()
                    for path in directory.glob(pattern)
                    if path.is_file()
                )
        if (self._root / "lineage.jsonl").is_file():
            files.add("lineage.jsonl")
        for generation_id in generation_ids:
            generation = self._root / "generations" / generation_id
            if generation.is_dir() and not generation.is_symlink():
                files.update(
                    path.relative_to(self._root).as_posix()
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


def validate_output_backend(
    manifest_bytes: bytes, backend: ValidationBackend
) -> ValidationReport:
    """Validate output through a storage-neutral streaming backend.

    The caller supplies the exact manifest bytes that are candidates for
    publication. Backend exceptions and validation details are deliberately
    omitted from errors because storage responses and rows can contain secrets.
    Every stream returned by the backend is closed before this function returns.
    """

    errors = _ErrorCollector()
    observed = _Observed()
    try:
        raw_manifest = orjson.loads(manifest_bytes)
    except (TypeError, orjson.JSONDecodeError):
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

    observed.counts["manifest_files"] = len(manifest.files)
    observed.counts["input_files_declared"] = manifest.counts.input_files
    observed.counts["skipped_inputs"] = manifest.counts.skipped_inputs
    seen_paths: set[str] = set()
    lineage_entries = 0

    for file_index, entry in enumerate(manifest.files):
        if not _is_safe_manifest_path(entry.path):
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

        try:
            source = backend.open_file(entry.path)
        except _UnsafeBackendPathError:
            errors.add(f"manifest files[{file_index}] has an unsafe path")
            continue
        # This is the storage boundary; exception details may contain secrets.
        except Exception:  # noqa: BLE001
            errors.add(f"manifest files[{file_index}] could not be opened")
            observed.mark_invalid(kind)
            continue
        if source is None:
            errors.add(f"manifest files[{file_index}] is missing or not a regular file")
            observed.mark_invalid(kind)
            continue

        observed.counts["checked_files"] += 1
        file_errors = _ErrorCollector()
        result = _validate_jsonl_stream(
            source.stream,
            kind=kind,
            file_index=file_index,
            observed=observed,
            errors=file_errors,
        )
        if source.size != entry.bytes or (
            result.complete and result.bytes_read != entry.bytes
        ):
            errors.add(f"manifest files[{file_index}] byte count does not match")
        if result.complete and result.sha256 != entry.sha256:
            errors.add(f"manifest files[{file_index}] sha256 does not match")
        errors.extend(file_errors)

    if lineage_entries != 1:
        errors.add("manifest must list lineage.jsonl exactly once")

    generation_ids = frozenset(
        parts[1]
        for relative in seen_paths
        if len(parts := PurePosixPath(relative).parts) >= 3
        and parts[0] == "generations"
    )
    if len(generation_ids) > 1:
        errors.add("manifest files must belong to one immutable generation")

    try:
        generated_files = set(backend.iter_generated_jsonl_paths(generation_ids))
    # Remote listing failures are SDK-specific and must remain safe to display.
    except Exception:  # noqa: BLE001
        errors.add("generated JSONL files could not be enumerated")
    else:
        unlisted_count = len(generated_files - seen_paths)
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
    if (
        len(observed.covered_sources) + manifest.counts.skipped_inputs
        != manifest.counts.input_files
    ):
        errors.add(
            "input_files does not match unique lineage, quarantine, and skipped sources"
        )
    if manifest.counts.skipped_inputs != sum(
        manifest.counts.skip_reason_counts.values()
    ):
        errors.add("skipped_inputs does not match skip_reason_counts")

    _compare_manifest_counts(manifest, observed, errors)
    messages = errors.finish()
    return ValidationReport(
        valid=errors.total == 0, errors=messages, counts=observed.counts
    )


def validate_output(root: Path) -> ValidationReport:
    """Validate a completed local output directory without exposing contents.

    Validation is streaming at the JSONL level. Error messages identify only a
    manifest entry and line number; values and validation exception text are
    deliberately omitted because trajectory rows can contain sensitive data.
    """

    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except FileNotFoundError:
        observed = _Observed()
        return ValidationReport(
            valid=False,
            errors=["manifest.json is missing"],
            counts=observed.counts,
        )
    except OSError:
        observed = _Observed()
        return ValidationReport(
            valid=False,
            errors=["manifest.json could not be read as valid JSON"],
            counts=observed.counts,
        )
    return validate_output_backend(manifest_bytes, _LocalValidationBackend(root))
