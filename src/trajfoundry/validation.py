"""Integrity and contract validation for a completed TrajFoundry output set."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter, deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated, BinaryIO, Literal, Protocol

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .canonical import trajectory_id
from .export import trajectory_filename
from .json_codec import loads
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
_PROGRESS_INTERVAL_SECONDS = 60.0
_PROGRESS_ITEM_INTERVAL = 10_000
DEFAULT_VALIDATION_PENDING_BYTES = 1024 * 1024**2

LOGGER = logging.getLogger(__name__)

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
    schema_version: Literal["trajfoundry-v2", "trajfoundry-v3", "trajfoundry-v4"]
    created_at: str
    input_root: str
    input_format: Literal["freerouter", "tokenplan", "sxf", "deepinfra"] = "freerouter"
    config_hash: str
    token_estimator: str
    counts: _ManifestCounts
    files: list[_ManifestFile]

    @model_validator(mode="after")
    def require_current_accounting_fields(self) -> _Manifest:
        """Keep v2 readable while requiring accounting fields in v3 and v4."""

        if self.schema_version in {"trajfoundry-v3", "trajfoundry-v4"}:
            if "input_format" not in self.model_fields_set:
                raise ValueError("v3 manifest requires input_format")
            if "skipped_inputs" not in self.counts.model_fields_set:
                raise ValueError("v3 manifest requires counts.skipped_inputs")
            if "skip_reason_counts" not in self.counts.model_fields_set:
                raise ValueError("v3 manifest requires counts.skip_reason_counts")
        if self.counts.excluded_records > self.counts.quarantined_records:
            raise ValueError("excluded_records cannot exceed quarantined_records")
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
        if kind in {"trajectory", "accepted", "quarantined_trajectories"}:
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


def _file_kind(relative: str, schema_version: str) -> str | None:
    pure = PurePosixPath(relative)
    if schema_version == "trajfoundry-v4":
        if len(pure.parts) != 1:
            return None
        if pure.name == "lineage.jsonl":
            return "lineage"
        if pure.name.endswith(".jsonl"):
            return "trajectory"
        return None
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
    relative_path: str,
    allow_legacy_metadata: bool,
    observed: _Observed,
    errors: _ErrorCollector,
) -> None:
    context = _row_context(file_index, line_no, kind)
    if not isinstance(value, dict):
        observed.mark_invalid(kind)
        errors.add(f"{context} must contain a JSON object")
        return

    if kind in {"trajectory", "accepted", "quarantined_trajectories"}:
        try:
            node = parse_trajectory_record(
                value, allow_legacy_metadata=allow_legacy_metadata
            )
        except (OutputContractError, ValidationError):
            observed.mark_invalid(kind)
            errors.add(f"{context} violates the TrajectoryNode contract")
            return
        observed.trajectory_ids[trajectory_id(node)] += 1
        strict = is_strict_sample(node)
        if kind == "trajectory":
            try:
                expected_filename = trajectory_filename(node)
            except ValueError:
                errors.add(f"{context} cannot produce a safe trajectory filename")
            else:
                if relative_path != expected_filename:
                    errors.add(f"{context} filename does not match trajectory metadata")
            observed.counts["accepted" if strict else "quarantined_trajectories"] += 1
            if not strict and node.normalization_audit is not None:
                observed.reason_counts.update(node.normalization_audit.reason_codes)
        try:
            validate_derived_fields(node)
        except StaleDerivedFieldsError:
            errors.add(f"{context} has stale or inconsistent derived quality fields")
        if kind == "accepted":
            if not strict:
                errors.add(f"{context} is not a strict accepted sample")
        elif kind == "quarantined_trajectories":
            if strict:
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
    relative_path: str,
    allow_legacy_metadata: bool,
    observed: _Observed,
    errors: _ErrorCollector,
    max_jsonl_row_bytes: int | None = None,
) -> _StreamResult:
    digest = hashlib.sha256()
    bytes_read = 0
    pending = bytearray()
    line_no = 0
    oversized = False
    complete = False

    def finish_row() -> None:
        nonlocal line_no, oversized
        line_no += 1
        if oversized:
            _count_jsonl_row(kind, observed)
            observed.mark_invalid(kind)
            errors.add(
                f"{_row_context(file_index, line_no, kind)} exceeds "
                f"the configured JSONL row byte limit ({max_jsonl_row_bytes})"
            )
        else:
            # orjson accepts bytearray directly; avoid creating another full
            # raw-line bytes copy while decoding a large trajectory.
            _validate_jsonl_row(
                pending,
                kind=kind,
                file_index=file_index,
                line_no=line_no,
                relative_path=relative_path,
                allow_legacy_metadata=allow_legacy_metadata,
                observed=observed,
                errors=errors,
            )
        pending.clear()
        oversized = False

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
            if isinstance(chunk, memoryview):
                chunk = chunk.tobytes()
            view = memoryview(chunk)
            start = 0
            while start < len(chunk):
                newline = chunk.find(b"\n", start)
                end = len(chunk) if newline < 0 else newline + 1
                if not oversized:
                    if (
                        max_jsonl_row_bytes is not None
                        and len(pending) + end - start > max_jsonl_row_bytes
                    ):
                        # Drain the rest of this row without retaining it.
                        # Checksums, byte counts and subsequent rows must still
                        # be checked; rejecting a row is never successful output.
                        oversized = True
                        pending.clear()
                    else:
                        pending.extend(view[start:end])
                if newline >= 0:
                    finish_row()
                start = end

        if complete and (pending or oversized):
            finish_row()
    finally:
        try:
            stream.close()
        # Closing a remote response can also raise an SDK-specific exception.
        except Exception:  # noqa: BLE001
            observed.mark_invalid(kind)
            errors.add(f"manifest files[{file_index}] could not be read completely")

    if complete and kind == "trajectory" and line_no != 1:
        observed.mark_invalid(kind)
        errors.add(f"manifest files[{file_index}] must contain exactly one trajectory")

    return _StreamResult(
        bytes_read=bytes_read,
        sha256=digest.hexdigest(),
        complete=complete,
    )


def _merge_observed(destination: _Observed, source: _Observed) -> None:
    """Merge one independent file validation result into the aggregate."""

    for name, value in source.counts.items():
        destination.counts[name] = destination.counts.get(name, 0) + value
    destination.reason_counts.update(source.reason_counts)
    destination.trajectory_ids.update(source.trajectory_ids)
    destination.lineage_ids.update(source.lineage_ids)
    destination.covered_sources.update(source.covered_sources)
    destination.trajectory_contract_valid &= source.trajectory_contract_valid
    destination.quarantine_contract_valid &= source.quarantine_contract_valid
    destination.records_contract_valid &= source.records_contract_valid
    destination.lineage_contract_valid &= source.lineage_contract_valid


def _validate_manifest_file(
    backend: ValidationBackend,
    *,
    entry: _ManifestFile,
    file_index: int,
    kind: str | None,
    allow_legacy_metadata: bool,
    max_jsonl_row_bytes: int | None = None,
) -> tuple[_Observed, _ErrorCollector]:
    """Validate one manifest object in isolation for bounded parallelism."""

    observed = _Observed()
    errors = _ErrorCollector()
    try:
        source = backend.open_file(entry.path)
    except _UnsafeBackendPathError:
        errors.add(f"manifest files[{file_index}] has an unsafe path")
        return observed, errors
    # This is the storage boundary; exception details may contain secrets.
    except Exception:  # noqa: BLE001
        errors.add(f"manifest files[{file_index}] could not be opened")
        observed.mark_invalid(kind)
        return observed, errors
    if source is None:
        errors.add(f"manifest files[{file_index}] is missing or not a regular file")
        observed.mark_invalid(kind)
        return observed, errors

    observed.counts["checked_files"] += 1
    result = _validate_jsonl_stream(
        source.stream,
        kind=kind,
        file_index=file_index,
        relative_path=entry.path,
        allow_legacy_metadata=allow_legacy_metadata,
        observed=observed,
        errors=errors,
        max_jsonl_row_bytes=max_jsonl_row_bytes,
    )
    if source.size != entry.bytes or (
        result.complete and result.bytes_read != entry.bytes
    ):
        errors.add(f"manifest files[{file_index}] byte count does not match")
    if result.complete and result.sha256 != entry.sha256:
        errors.add(f"manifest files[{file_index}] sha256 does not match")
    return observed, errors


def _count_jsonl_row(kind: str | None, observed: _Observed) -> None:
    observed.counts["jsonl_rows"] += 1
    if kind in {"accepted", "quarantined_trajectories", "quarantined_records"}:
        observed.counts[kind] += 1
    elif kind == "lineage":
        observed.counts["lineage_records"] += 1


def _validate_jsonl_row(
    raw_line: bytes | bytearray,
    *,
    kind: str | None,
    file_index: int,
    line_no: int,
    relative_path: str,
    allow_legacy_metadata: bool,
    observed: _Observed,
    errors: _ErrorCollector,
) -> None:
    _count_jsonl_row(kind, observed)
    try:
        value = loads(raw_line)
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
        relative_path=relative_path,
        allow_legacy_metadata=allow_legacy_metadata,
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
        files.update(
            path.name
            for path in self._root.glob("*.jsonl")
            if path.is_file() and not path.is_symlink()
        )
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
    directly_observed: dict[str, int] = {
        "accepted": observed.counts["accepted"],
        "quarantined_trajectories": observed.counts["quarantined_trajectories"],
    }
    if manifest.schema_version != "trajfoundry-v4":
        directly_observed["quarantined_records"] = observed.counts[
            "quarantined_records"
        ]
    for name, actual in directly_observed.items():
        if getattr(expected, name) != actual:
            errors.add(f"manifest count {name} does not match the JSONL rows")

    if manifest.schema_version != "trajfoundry-v4" and (
        observed.records_contract_valid
        and expected.excluded_records != observed.counts["excluded_records"]
    ):
        errors.add("manifest count excluded_records does not match the records")
    if (
        observed.lineage_contract_valid
        and expected.duplicate_trajectories != observed.counts["duplicate_trajectories"]
    ):
        errors.add("manifest count duplicate_trajectories does not match lineage")
    if (
        manifest.schema_version != "trajfoundry-v4"
        and observed.quarantine_contract_valid
        and observed.records_contract_valid
    ):
        actual_reasons = dict(sorted(observed.reason_counts.items()))
        if expected.reason_counts != actual_reasons:
            errors.add("manifest reason_counts does not match quarantined output")
    elif manifest.schema_version == "trajfoundry-v4":
        declared_reason_total = sum(expected.reason_counts.values())
        observed_reason_total = sum(observed.reason_counts.values())
        required_reason_total = observed_reason_total + expected.quarantined_records
        if declared_reason_total < required_reason_total:
            errors.add(
                "manifest reason_counts does not account for quarantined records"
            )
        for reason, count in observed.reason_counts.items():
            if expected.reason_counts.get(reason, 0) < count:
                errors.add(
                    "manifest reason_counts does not cover quarantined trajectories"
                )
                break


def validate_output_backend(
    manifest_bytes: bytes,
    backend: ValidationBackend,
    *,
    max_workers: int = 1,
    max_pending_bytes: int = DEFAULT_VALIDATION_PENDING_BYTES,
    max_jsonl_row_bytes: int | None = None,
) -> ValidationReport:
    """Validate output through a storage-neutral streaming backend.

    The caller supplies the exact manifest bytes that are candidates for
    publication. Backend exceptions and validation details are deliberately
    omitted from errors because storage responses and rows can contain secrets.
    Every stream returned by the backend is closed before this function returns.
    Pending file bytes bound concurrent raw-data exposure, not Python model
    expansion. A file larger than the budget is scanned alone, never rejected
    by total file size. ``max_jsonl_row_bytes`` optionally bounds individual row
    buffers (including their newline); ``None`` retains legacy compatibility.
    """

    if max_workers <= 0:
        raise ValueError("max workers must be positive")
    if max_pending_bytes <= 0:
        raise ValueError("max pending bytes must be positive")
    if max_jsonl_row_bytes is not None and max_jsonl_row_bytes <= 0:
        raise ValueError("max JSONL row bytes must be positive")
    errors = _ErrorCollector()
    observed = _Observed()
    try:
        raw_manifest = loads(manifest_bytes)
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
    if manifest.schema_version == "trajfoundry-v4":
        # v4 deliberately retains failed-record accounting only in the manifest.
        observed.counts["quarantined_records"] = manifest.counts.quarantined_records
        observed.counts["excluded_records"] = manifest.counts.excluded_records
    seen_paths: set[str] = set()
    lineage_entries = 0
    started_at = time.monotonic()
    last_progress_at = started_at
    processed_files = 0
    pending_bytes = 0

    pending: list[tuple[int, _ManifestFile, str | None]] = []
    for file_index, entry in enumerate(manifest.files):
        if not _is_safe_manifest_path(entry.path):
            errors.add(f"manifest files[{file_index}] has an unsafe path")
            continue
        if entry.path in seen_paths:
            errors.add(f"manifest files[{file_index}] duplicates an earlier path")
            continue
        seen_paths.add(entry.path)

        kind = _file_kind(entry.path, manifest.schema_version)
        if kind is None:
            errors.add(f"manifest files[{file_index}] is not a supported output file")
        elif kind == "lineage":
            lineage_entries += 1
        pending.append((file_index, entry, kind))

    # Each file is validated with an isolated accumulator.  Merging in manifest
    # order keeps error ordering deterministic while S3 GET and JSONL scanning
    # happen concurrently.  The default remains serial for local callers and
    # backwards compatibility; S3 jobs can opt into a bounded worker count.
    def validate_one(
        item: tuple[int, _ManifestFile, str | None],
    ) -> tuple[_Observed, _ErrorCollector]:
        file_index, entry, kind = item
        return _validate_manifest_file(
            backend,
            entry=entry,
            file_index=file_index,
            kind=kind,
            allow_legacy_metadata=manifest.schema_version
            in {
                "trajfoundry-v2",
                "trajfoundry-v3",
            },
            max_jsonl_row_bytes=max_jsonl_row_bytes,
        )

    def log_progress(*, force: bool = False) -> None:
        nonlocal last_progress_at
        now = time.monotonic()
        if not force and (
            processed_files % _PROGRESS_ITEM_INTERVAL != 0
            and now - last_progress_at < _PROGRESS_INTERVAL_SECONDS
        ):
            return
        LOGGER.info(
            "【校验阶段】已校验文件=%d/%d，JSONL行数=%d，接受轨迹=%d，"
            "隔离轨迹=%d，错误数=%d，在途原始字节=%.1fMiB，耗时=%.1f秒",
            processed_files,
            len(manifest.files),
            observed.counts["jsonl_rows"],
            observed.counts["accepted"],
            observed.counts["quarantined_trajectories"],
            errors.total,
            pending_bytes / 1024**2,
            now - started_at,
        )
        last_progress_at = now

    LOGGER.info(
        "【校验阶段】开始校验输出：文件总数=%d，并发线程数=%d，"
        "在途原始字节上限=%.1fMiB，单行字节上限=%s",
        len(manifest.files),
        max_workers,
        max_pending_bytes / 1024**2,
        "兼容模式（不限制）"
        if max_jsonl_row_bytes is None
        else str(max_jsonl_row_bytes),
    )

    if max_workers == 1 or len(pending) <= 1:
        # Merge each file as soon as it has been scanned.  Keeping this path
        # streaming is important for v4 outputs with a very large number of
        # one-trajectory files.
        for item in pending:
            file_observed, file_errors = validate_one(item)
            _merge_observed(observed, file_observed)
            errors.extend(file_errors)
            processed_files += 1
            log_progress()
    else:
        # Keep only a small ordered window in memory.  Submitting the complete
        # manifest at once would create one Future and one _Observed accumulator
        # per output file, defeating the bounded-concurrency guarantee at the
        # million-file scale.
        window = max_workers * 2
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="trajfoundry-s3-get",
        ) as executor:
            futures: deque[tuple[Future[tuple[_Observed, _ErrorCollector]], int]] = (
                deque()
            )
            next_index = 0
            while next_index < len(pending) or futures:
                while next_index < len(pending) and len(futures) < window:
                    item = pending[next_index]
                    weight = max(1, item[1].bytes)
                    if item[2] != "trajectory" and max_jsonl_row_bytes is not None:
                        # Lineage and legacy shards can legitimately contain
                        # many small rows. Their total size is not a row limit.
                        weight = min(weight, max_jsonl_row_bytes)
                    if futures and pending_bytes + weight > max_pending_bytes:
                        break
                    futures.append((executor.submit(validate_one, item), weight))
                    pending_bytes += weight
                    next_index += 1
                future, weight = futures[0]
                while True:
                    try:
                        file_observed, file_errors = future.result(
                            timeout=_PROGRESS_INTERVAL_SECONDS
                        )
                        break
                    except TimeoutError:
                        if future.done():
                            raise
                        log_progress(force=True)
                futures.popleft()
                pending_bytes -= weight
                _merge_observed(observed, file_observed)
                errors.extend(file_errors)
                processed_files += 1
                log_progress()

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
    accounted_sources = len(observed.covered_sources) + manifest.counts.skipped_inputs
    if manifest.schema_version == "trajfoundry-v4":
        accounted_sources += manifest.counts.quarantined_records
    if accounted_sources != manifest.counts.input_files:
        errors.add(
            "input_files does not match unique lineage, quarantine, and skipped sources"
        )
    if manifest.counts.skipped_inputs != sum(
        manifest.counts.skip_reason_counts.values()
    ):
        errors.add("skipped_inputs does not match skip_reason_counts")

    _compare_manifest_counts(manifest, observed, errors)
    messages = errors.finish()
    log_progress(force=True)
    LOGGER.info(
        "【校验阶段】完成：已校验文件=%d/%d，JSONL行数=%d，错误数=%d，"
        "结果=%s，耗时=%.1f秒",
        processed_files,
        len(manifest.files),
        observed.counts["jsonl_rows"],
        errors.total,
        "通过" if errors.total == 0 else "失败",
        time.monotonic() - started_at,
    )
    return ValidationReport(
        valid=errors.total == 0, errors=messages, counts=observed.counts
    )


def validate_output(
    root: Path,
    *,
    max_workers: int = 1,
    max_pending_bytes: int = DEFAULT_VALIDATION_PENDING_BYTES,
    max_jsonl_row_bytes: int | None = None,
) -> ValidationReport:
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
    return validate_output_backend(
        manifest_bytes,
        _LocalValidationBackend(root),
        max_workers=max_workers,
        max_pending_bytes=max_pending_bytes,
        max_jsonl_row_bytes=max_jsonl_row_bytes,
    )
