"""Independent scheduler-facing S3 trajectory classification job."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO

import orjson

from .canonical import trajectory_id
from .classification.cache import ClassificationCache
from .classification.classifier import (
    CAPABILITY_LABELS,
    CLASSIFIER_REVISION,
    DEFAULT_MAX_CONTEXT_CHARS,
    PROMPT_VERSION,
    TrajectoryClassifier,
)
from .classification.client import ChatCompletionsClient
from .classification.taxonomy import DEFAULT_TAXONOMY_PATH, ScenarioTaxonomy
from .credentials import DEFAULT_S3_CREDENTIALS_PATH, load_s3_credentials
from .output_contract import parse_trajectory_record
from .s3 import S3Location, create_s3_client, parse_s3_uri
from .s3_validation import S3ValidationBackend
from .validation import validate_output_backend

LOGGER = logging.getLogger(__name__)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FLAT_TRAJECTORY = re.compile(r"^.+_sub_[0-9]+\.jsonl$")
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_SYNTHESIZED_SESSION_CODE = "metadata_session_id_synthesized"
_READ_CHUNK_BYTES = 1024 * 1024


class LabelJobError(RuntimeError):
    """The classification input or published result is invalid."""


@dataclass(frozen=True, slots=True)
class LabelJobResult:
    """Safe summary returned to scheduler code."""

    input_trajectories: int
    classified: int
    failed: int
    cache_hits: int
    validation_valid: bool = True


@dataclass(frozen=True, slots=True)
class _ManifestEntry:
    path: str
    sha256: str
    bytes: int
    kind: str | None


@dataclass(frozen=True, slots=True)
class _StagedTrajectory:
    path: Path
    stable_hash: str
    session_id: str
    session_synthesized: bool
    created_at: str
    source: str
    input_order: int
    existing_sub_session_id: int | None
    sub_session_id: int | None = None
    output_path: str = ""


def _locations_overlap(left: S3Location, right: S3Location) -> bool:
    return left.bucket == right.bucket and (
        left.prefix.startswith(right.prefix) or right.prefix.startswith(left.prefix)
    )


def _close_s3_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            LOGGER.warning("classification S3 client could not be closed cleanly")


def _read_response_bytes(response: object) -> bytes:
    if not isinstance(response, dict):
        raise LabelJobError("S3 response is not an object")
    body = response.get("Body")
    if body is None or not callable(getattr(body, "read", None)):
        raise LabelJobError("S3 response has no readable body")
    result = bytearray()
    try:
        while True:
            chunk = body.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise LabelJobError("S3 response body yielded non-bytes")
            result.extend(chunk)
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    size = response.get("ContentLength")
    if type(size) is not int or size != len(result):
        raise LabelJobError("S3 response byte count does not match")
    return bytes(result)


def _get_object_bytes(client: Any, location: S3Location, relative: str) -> bytes:
    response = client.get_object(Bucket=location.bucket, Key=location.key(relative))
    return _read_response_bytes(response)


def _list_flat_jsonl_paths(client: Any, location: S3Location) -> set[str]:
    """List top-level JSONL objects, including objects absent from a manifest."""

    paginator = client.get_paginator("list_objects_v2")
    paths: set[str] = set()
    observed_keys: set[str] = set()
    for page in paginator.paginate(Bucket=location.bucket, Prefix=location.prefix):
        contents = page.get("Contents", [])
        if contents is None:
            contents = []
        if not isinstance(contents, list):
            raise LabelJobError("S3 output listing is invalid")
        for item in contents:
            if not isinstance(item, dict):
                raise LabelJobError("S3 output listing is invalid")
            key = item.get("Key")
            if not isinstance(key, str) or not key.startswith(location.prefix):
                raise LabelJobError("S3 output listing escaped its prefix")
            if key in observed_keys:
                raise LabelJobError("S3 output listing contains a duplicate object")
            observed_keys.add(key)
            relative = key[len(location.prefix) :]
            if not relative or "/" in relative or not relative.endswith(".jsonl"):
                continue
            try:
                if location.key(relative) != key:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise LabelJobError(
                    "S3 output listing contains an unsafe object"
                ) from error
            paths.add(relative)
    return paths


def _delete_flat_jsonl_paths(
    client: Any,
    location: S3Location,
    paths: set[str],
    *,
    phase: str,
) -> None:
    """Delete a known set and fail safely if any deletion cannot be confirmed."""

    for relative in sorted(paths):
        try:
            client.delete_object(
                Bucket=location.bucket,
                Key=location.key(relative),
            )
        except Exception as error:
            raise LabelJobError(
                f"could not remove stale classification object during {phase} cleanup"
            ) from error


def _manifest_entries(manifest_bytes: bytes) -> list[_ManifestEntry]:
    try:
        manifest = orjson.loads(manifest_bytes)
    except orjson.JSONDecodeError as error:
        raise LabelJobError("input manifest is not valid JSON") from error
    if type(manifest) is not dict or type(manifest.get("files")) is not list:
        raise LabelJobError("input manifest must contain a files array")
    entries: list[_ManifestEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(manifest["files"]):
        if type(item) is not dict:
            raise LabelJobError(f"input manifest files[{index}] is invalid")
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("bytes")
        kind = item.get("kind")
        if (
            type(path) is not str
            or not path
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
            or type(size) is not int
            or size < 0
            or (kind is not None and type(kind) is not str)
        ):
            raise LabelJobError(f"input manifest files[{index}] is invalid")
        if path in seen:
            raise LabelJobError("input manifest contains duplicate paths")
        seen.add(path)
        entries.append(_ManifestEntry(path, digest, size, kind))
    return entries


def _is_trajectory_entry(entry: _ManifestEntry) -> bool:
    if entry.kind == "trajectory":
        return True
    path = entry.path
    if path.endswith("/accepted/trajectories-00000.jsonl"):
        return True
    if "/accepted/trajectories-" in path and path.endswith(".jsonl"):
        return True
    if "/quarantine/trajectories-" in path and path.endswith(".jsonl"):
        return True
    if path.startswith("accepted/trajectories-") and path.endswith(".jsonl"):
        return True
    if path.startswith("quarantine/trajectories-") and path.endswith(".jsonl"):
        return True
    return "/" not in path and _FLAT_TRAJECTORY.fullmatch(path) is not None


def _is_lineage_entry(entry: _ManifestEntry) -> bool:
    return (
        entry.kind == "lineage"
        or entry.path == "lineage.jsonl"
        or entry.path.endswith("/lineage.jsonl")
    )


def _is_record_entry(entry: _ManifestEntry) -> bool:
    return (
        entry.kind == "quarantined_record"
        or entry.path.startswith("quarantine/records-")
        or "/quarantine/records-" in entry.path
    ) and entry.path.endswith(".jsonl")


def _session_is_synthesized(value: dict[str, Any]) -> bool:
    audit = value.get("normalization_audit")
    if type(audit) is not dict:
        return False
    issues = audit.get("issues")
    if type(issues) is not list:
        return False
    return any(
        type(issue) is dict and issue.get("code") == _SYNTHESIZED_SESSION_CODE
        for issue in issues
    )


def _validate_manifest_object(
    *, entry: _ManifestEntry, bytes_read: int, digest: str
) -> None:
    if bytes_read != entry.bytes:
        raise LabelJobError(f"input object byte count does not match: {entry.path}")
    if digest != entry.sha256:
        raise LabelJobError(f"input object sha256 does not match: {entry.path}")


def _parse_trajectory(value: object) -> Any:
    """Use the strict contract while allowing a v004 row to omit the new field."""

    try:
        return parse_trajectory_record(value, allow_legacy_metadata=True)
    except TypeError as error:
        # Compatibility while an older editable installation is being upgraded.
        if "allow_legacy_metadata" not in str(error):
            raise
        return parse_trajectory_record(value)


def _stage_manifest_entry(
    client: Any,
    input_location: S3Location,
    entry: _ManifestEntry,
    stage_root: Path,
    start_order: int,
) -> list[_StagedTrajectory]:
    response = client.get_object(
        Bucket=input_location.bucket,
        Key=input_location.key(entry.path),
    )
    if not isinstance(response, dict):
        raise LabelJobError("S3 trajectory response is not an object")
    body: BinaryIO | Any = response.get("Body")
    if body is None or not callable(getattr(body, "read", None)):
        raise LabelJobError("S3 trajectory response has no readable body")
    reported_size = response.get("ContentLength")
    if type(reported_size) is not int or reported_size != entry.bytes:
        close = getattr(body, "close", None)
        if callable(close):
            close()
        raise LabelJobError(f"input object byte count does not match: {entry.path}")

    staged: list[_StagedTrajectory] = []
    digest = hashlib.sha256()
    bytes_read = 0
    pending = bytearray()
    line_no = 0

    def consume(raw_line: bytes) -> None:
        nonlocal line_no
        line_no += 1
        if not raw_line.strip():
            raise LabelJobError(f"trajectory JSONL contains an empty row: {entry.path}")
        try:
            value = orjson.loads(raw_line)
        except orjson.JSONDecodeError as error:
            raise LabelJobError(
                f"trajectory JSONL contains invalid JSON: {entry.path}"
            ) from error
        if type(value) is not dict:
            raise LabelJobError("trajectory row must be a JSON object")
        metadata = value.get("metadata")
        if type(metadata) is not dict:
            raise LabelJobError("trajectory row has no metadata object")
        session_id = metadata.get("session_id")
        if type(session_id) is not str or not session_id:
            raise LabelJobError("trajectory session_id must be a non-empty string")
        existing_sub = metadata.get("sub_session_id")
        if existing_sub is not None and (
            type(existing_sub) is not int or existing_sub < 0
        ):
            raise LabelJobError("trajectory sub_session_id must be non-negative")
        node = _parse_trajectory(value)
        created_at = metadata.get("created_at")
        source = value.get("source")
        if type(created_at) is not str or type(source) is not str:
            raise LabelJobError("trajectory ordering metadata is invalid")
        order = start_order + len(staged)
        staged_path = stage_root / f"trajectory-{order:09d}.json"
        staged_path.write_bytes(orjson.dumps(value, option=orjson.OPT_SORT_KEYS))
        staged.append(
            _StagedTrajectory(
                path=staged_path,
                stable_hash=trajectory_id(node),
                session_id=session_id,
                session_synthesized=_session_is_synthesized(value),
                created_at=created_at,
                source=source,
                input_order=order,
                existing_sub_session_id=existing_sub,
            )
        )

    try:
        while True:
            chunk = body.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise LabelJobError("S3 trajectory response yielded non-bytes")
            digest.update(chunk)
            bytes_read += len(chunk)
            pending.extend(chunk)
            while (newline := pending.find(b"\n")) >= 0:
                raw_line = bytes(pending[:newline])
                del pending[: newline + 1]
                consume(raw_line)
        if pending:
            consume(bytes(pending))
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    _validate_manifest_object(
        entry=entry, bytes_read=bytes_read, digest=digest.hexdigest()
    )
    return staged


def _validate_session_filename(session_id: str) -> None:
    if (
        session_id in {".", ".."}
        or len(session_id) > 200
        or _SAFE_SESSION_ID.fullmatch(session_id) is None
    ):
        raise LabelJobError(
            "real session_id contains characters unsafe for a trajectory filename"
        )


def _assign_output_paths(
    trajectories: list[_StagedTrajectory],
) -> list[_StagedTrajectory]:
    present = [item.existing_sub_session_id is not None for item in trajectories]
    if any(present) and not all(present):
        raise LabelJobError("input mixes legacy and sub-session trajectory schemas")
    modern = bool(present) and all(present)
    real_sessions: dict[str, list[_StagedTrajectory]] = defaultdict(list)
    sessionless: list[_StagedTrajectory] = []
    for item in trajectories:
        if item.session_synthesized:
            if item.session_id != "no_session_id":
                raise LabelJobError("synthesized session must use no_session_id")
            sessionless.append(item)
        else:
            _validate_session_filename(item.session_id)
            real_sessions[item.session_id].append(item)

    assigned: list[_StagedTrajectory] = []
    for session_id in sorted(real_sessions):
        group = sorted(
            real_sessions[session_id],
            key=lambda item: (item.created_at, item.stable_hash, item.source),
        )
        for index, item in enumerate(group):
            if modern and item.existing_sub_session_id != index:
                raise LabelJobError(
                    "existing sub_session_id does not match stable session ordering"
                )
            sub_session_id = item.existing_sub_session_id if modern else index
            assert sub_session_id is not None
            assigned.append(
                replace(
                    item,
                    sub_session_id=sub_session_id,
                    output_path=f"{session_id}_sub_{sub_session_id}.jsonl",
                )
            )

    for item in sessionless:
        if modern and item.existing_sub_session_id != 0:
            raise LabelJobError("sessionless trajectories must use sub_session_id 0")
        assigned.append(
            replace(
                item,
                sub_session_id=0,
                output_path=(f"no_session_id_{item.stable_hash}_sub_0.jsonl"),
            )
        )

    paths = [item.output_path for item in assigned]
    if len(paths) != len(set(paths)):
        raise LabelJobError("trajectory output filenames are not unique")
    return sorted(assigned, key=lambda item: item.output_path)


def _put_and_verify(
    client: Any,
    location: S3Location,
    relative_path: str,
    payload: bytes,
    *,
    content_type: str,
    verify_readback: bool = True,
) -> dict[str, str | int]:
    client.put_object(
        Bucket=location.bucket,
        Key=location.key(relative_path),
        Body=payload,
        ContentType=content_type,
    )
    if verify_readback:
        published = _get_object_bytes(client, location, relative_path)
        if published != payload:
            raise LabelJobError(
                f"published object verification failed: {relative_path}"
            )
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _classification_cache_key(
    trajectory_hash: str,
    *,
    input_manifest_sha256: str,
    config_hash: str,
) -> str:
    return hashlib.sha256(
        f"{trajectory_hash}:{input_manifest_sha256}:{config_hash}".encode()
    ).hexdigest()


def _set_sub_session_id(value: dict[str, Any], sub_session_id: int) -> None:
    metadata = value.get("metadata")
    if type(metadata) is not dict:
        raise LabelJobError("trajectory node has no metadata")
    metadata["sub_session_id"] = sub_session_id
    children = value.get("sub_agent_trajectory")
    if children is None:
        return
    if type(children) is not dict:
        raise LabelJobError("sub_agent_trajectory must be an object")
    for child in children.values():
        if type(child) is not dict:
            raise LabelJobError("sub-agent trajectory must be an object")
        _set_sub_session_id(child, sub_session_id)


def _validate_sub_session_inheritance(value: dict[str, Any], expected: int) -> None:
    metadata = value.get("metadata")
    if type(metadata) is not dict or metadata.get("sub_session_id") != expected:
        raise LabelJobError("nested trajectory does not inherit sub_session_id")
    children = value.get("sub_agent_trajectory")
    if children is None:
        return
    if type(children) is not dict:
        raise LabelJobError("sub_agent_trajectory must be an object")
    for child in children.values():
        if type(child) is not dict:
            raise LabelJobError("sub-agent trajectory must be an object")
        _validate_sub_session_inheritance(child, expected)


def _validate_classification(
    value: dict[str, Any],
    *,
    item: _StagedTrajectory,
    classifier: TrajectoryClassifier,
) -> None:
    assert item.sub_session_id is not None
    _validate_sub_session_inheritance(value, item.sub_session_id)
    classification = value.get("classification")
    if type(classification) is not dict:
        raise LabelJobError("classified trajectory has no classification object")
    common = {
        "status",
        "model_label",
        "harness_label",
        "classifier_revision",
        "prompt_version",
        "taxonomy_sha256",
        "input_manifest_sha256",
        "context_truncated",
    }
    status = classification.get("status")
    expected_keys = (
        common | {"scenario_labels", "capability_labels"}
        if status == "accepted"
        else common | {"reason"}
    )
    if status not in {"accepted", "failed"} or set(classification) != expected_keys:
        raise LabelJobError("classification object violates its status contract")
    if classification["classifier_revision"] != classifier.classifier_revision:
        raise LabelJobError("classification revision does not match the job")
    if classification["prompt_version"] != classifier.prompt_version:
        raise LabelJobError("classification prompt version does not match the job")
    if classification["taxonomy_sha256"] != classifier.taxonomy.sha256:
        raise LabelJobError("classification taxonomy does not match the job")
    if classification["input_manifest_sha256"] != classifier.input_manifest_sha256:
        raise LabelJobError("classification input manifest does not match the job")
    if type(classification["context_truncated"]) is not bool:
        raise LabelJobError("classification context_truncated must be a boolean")
    expected_model = value.get("model") or "unknown"
    expected_harness = value.get("harness") or "unknown"
    if classification["model_label"] != expected_model:
        raise LabelJobError("classification model_label does not match trajectory")
    if classification["harness_label"] != expected_harness:
        raise LabelJobError("classification harness_label does not match trajectory")
    if status == "failed":
        if type(classification["reason"]) is not str or not classification["reason"]:
            raise LabelJobError("failed classification must contain a reason")
        return
    labels = classification["scenario_labels"]
    capabilities = classification["capability_labels"]
    if type(labels) is not list or not labels:
        raise LabelJobError("accepted classification must contain scenario labels")
    if any(
        type(label) is not dict or type(label.get("id")) is not int for label in labels
    ):
        raise LabelJobError("classification scenario labels are invalid")
    identifiers = [label["id"] for label in labels]
    if classifier.taxonomy.expand(identifiers) != labels:
        raise LabelJobError("classification scenario labels do not match taxonomy")
    if (
        type(capabilities) is not list
        or not capabilities
        or any(type(label) is not str for label in capabilities)
        or len(set(capabilities)) != len(capabilities)
        or any(label not in CAPABILITY_LABELS for label in capabilities)
    ):
        raise LabelJobError("classification capability labels are invalid")


def _load_and_classify(
    item: _StagedTrajectory,
    *,
    classifier: TrajectoryClassifier,
    cache: ClassificationCache,
) -> tuple[bytes, str, bool]:
    value = orjson.loads(item.path.read_bytes())
    if type(value) is not dict:
        raise LabelJobError("staged trajectory is not an object")
    if "classification" in value:
        raise LabelJobError("normalized input must not contain classification")
    assert item.sub_session_id is not None
    _set_sub_session_id(value, item.sub_session_id)
    cache_key = _classification_cache_key(
        item.stable_hash,
        input_manifest_sha256=classifier.input_manifest_sha256,
        config_hash=classifier.config_hash,
    )
    classification = cache.get(cache_key)
    cache_hit = classification is not None
    if classification is not None:
        value["classification"] = classification
        try:
            _validate_classification(value, item=item, classifier=classifier)
        except (LabelJobError, TypeError, ValueError):
            value.pop("classification", None)
            classification = None
            cache_hit = False
    if classification is None:
        attempt = classifier.classify(value)
        classification = attempt.classification
        if attempt.cacheable:
            cache.put(cache_key, classification)
    value["classification"] = classification
    _validate_classification(value, item=item, classifier=classifier)
    payload = orjson.dumps(value, option=orjson.OPT_SORT_KEYS) + b"\n"
    return payload, str(classification["status"]), cache_hit


def _run_s3_label_job_with_client(
    client: Any,
    *,
    input_location: S3Location,
    output_location: S3Location,
    classifier: TrajectoryClassifier,
    workspace: Path,
    state_path: Path | None,
    max_workers: int,
) -> LabelJobResult:
    manifest_bytes = _get_object_bytes(client, input_location, "manifest.json")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != classifier.input_manifest_sha256:
        raise LabelJobError("classifier was initialized for a different input manifest")
    input_report = validate_output_backend(
        manifest_bytes, S3ValidationBackend(client, input_location)
    )
    if not input_report.valid:
        detail = input_report.errors[0] if input_report.errors else "unknown error"
        raise LabelJobError(f"normalized input failed validation: {detail}")
    # Verify output listing access before any model calls. Existing flat objects
    # remain in place until the replacement manifest has been published. Do not
    # read the old output manifest: some S3-compatible stores return AccessDenied
    # for a missing object, which must not block a first classification run.
    initial_output_paths = _list_flat_jsonl_paths(client, output_location)
    entries = _manifest_entries(manifest_bytes)
    trajectory_entries = [entry for entry in entries if _is_trajectory_entry(entry)]
    lineage_entries = [entry for entry in entries if _is_lineage_entry(entry)]
    if len(lineage_entries) != 1:
        raise LabelJobError("input manifest must list exactly one lineage object")
    if any(
        not (
            _is_trajectory_entry(entry)
            or _is_lineage_entry(entry)
            or _is_record_entry(entry)
        )
        for entry in entries
    ):
        raise LabelJobError("input manifest contains an unsupported output object")

    with TemporaryDirectory(prefix="trajfoundry-label-", dir=workspace) as directory:
        stage_root = Path(directory)
        staged: list[_StagedTrajectory] = []
        for entry in trajectory_entries:
            staged.extend(
                _stage_manifest_entry(
                    client,
                    input_location,
                    entry,
                    stage_root,
                    len(staged),
                )
            )
        assigned = _assign_output_paths(staged)
        observed_trajectories = input_report.counts.get("accepted", 0) + (
            input_report.counts.get("quarantined_trajectories", 0)
        )
        if observed_trajectories != len(assigned):
            raise LabelJobError("input manifest trajectory count does not match rows")
        if state_path is None:
            raise ValueError(
                "state_path must identify a persistent classification cache"
            )
        cache_file = state_path
        candidate_root = stage_root / "candidate"
        candidate_root.mkdir()
        files: list[dict[str, Any]] = []
        classified = 0
        failed = 0
        cache_hits = 0
        candidates: list[tuple[_StagedTrajectory, Path, str]] = []
        with ClassificationCache(cache_file) as cache:

            def worker(item: _StagedTrajectory) -> tuple[bytes, str, bool]:
                return _load_and_classify(item, classifier=classifier, cache=cache)

            with ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="trajfoundry-label",
            ) as executor:
                window = max_workers * 2
                for offset in range(0, len(assigned), window):
                    batch = assigned[offset : offset + window]
                    for item, result in zip(batch, executor.map(worker, batch)):
                        payload, status, cache_hit = result
                        candidate = candidate_root / item.output_path
                        candidate.write_bytes(payload)
                        candidates.append((item, candidate, status))
                        cache_hits += int(cache_hit)
                        if status == "accepted":
                            classified += 1
                        else:
                            failed += 1

        lineage_entry = lineage_entries[0]
        lineage_payload = _get_object_bytes(client, input_location, lineage_entry.path)
        _validate_manifest_object(
            entry=lineage_entry,
            bytes_read=len(lineage_payload),
            digest=hashlib.sha256(lineage_payload).hexdigest(),
        )
        lineage_candidate = candidate_root / "lineage.jsonl"
        lineage_candidate.write_bytes(lineage_payload)

        # No S3 object is changed until every model call and local contract check
        # has completed. In particular, a late fatal 404/422 cannot corrupt the
        # previously published flat output set.
        for item, candidate, status in candidates:
            payload = candidate.read_bytes()
            if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
                raise LabelJobError("classified trajectory file must contain one row")
            value = orjson.loads(payload)
            if type(value) is not dict:
                raise LabelJobError("classified trajectory row must be an object")
            _validate_classification(value, item=item, classifier=classifier)
            files.append(
                {
                    "path": item.output_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "kind": "trajectory",
                    "classification_status": status,
                }
            )
        files.append(
            {
                "path": "lineage.jsonl",
                "sha256": hashlib.sha256(lineage_payload).hexdigest(),
                "bytes": len(lineage_payload),
                "kind": "lineage",
            }
        )
        if len(candidates) != classified + failed:
            raise LabelJobError("classification candidate counts do not reconcile")
        output_manifest = {
            "schema_version": "trajfoundry-classification-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "input_root": input_location.uri,
            "input_manifest_sha256": manifest_sha256,
            "classifier_revision": classifier.classifier_revision,
            "prompt_version": classifier.prompt_version,
            "taxonomy_sha256": classifier.taxonomy.sha256,
            "classifier_model": classifier.client.model,
            "config_hash": classifier.config_hash,
            "counts": {
                "input_trajectories": len(assigned),
                "classified": classified,
                "failed": failed,
                "cache_hits": cache_hits,
            },
            "files": sorted(files, key=lambda entry: str(entry["path"])),
        }
        output_manifest_bytes = orjson.dumps(
            output_manifest, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
        )

        files_by_path = {str(entry["path"]): entry for entry in files}
        for item, candidate, _ in candidates:
            uploaded = _put_and_verify(
                client,
                output_location,
                item.output_path,
                candidate.read_bytes(),
                content_type="application/x-ndjson",
            )
            expected = files_by_path[item.output_path]
            if uploaded["sha256"] != expected["sha256"]:
                raise LabelJobError("uploaded classification hash changed")
        _put_and_verify(
            client,
            output_location,
            "lineage.jsonl",
            lineage_payload,
            content_type="application/x-ndjson",
        )
        _put_and_verify(
            client,
            output_location,
            "manifest.json",
            output_manifest_bytes,
            content_type="application/json",
            verify_readback=False,
        )
        # Treat the output manifest as write-only. Trajectory JSONL and lineage
        # objects are still read back above, while a successful PutObject is the
        # publication boundary for the manifest itself.
        published_paths = {str(entry["path"]) for entry in files}
        listed_output_paths = initial_output_paths | _list_flat_jsonl_paths(
            client, output_location
        )
        _delete_flat_jsonl_paths(
            client,
            output_location,
            listed_output_paths - published_paths,
            phase="post-publication",
        )

    LOGGER.info(
        "TrajFoundry classification complete: input=%d classified=%d failed=%d "
        "cache_hits=%d validation=passed",
        len(assigned),
        classified,
        failed,
        cache_hits,
    )
    return LabelJobResult(
        input_trajectories=len(assigned),
        classified=classified,
        failed=failed,
        cache_hits=cache_hits,
    )


def run_s3_label_job(
    input_uri: str,
    output_uri: str,
    endpoint_url: str,
    *,
    classifier_api_url: str | None = None,
    classifier_model: str | None = None,
    classifier_api_key: str | None = None,
    region_name: str = "us-east-1",
    workspace_parent: str | Path | None = None,
    state_path: str | Path | None = None,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    credentials_path: str | Path | None = DEFAULT_S3_CREDENTIALS_PATH,
    max_workers: int = 4,
    max_retries: int = 5,
    request_timeout_seconds: float = 120.0,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> LabelJobResult:
    """Classify every complete normalized trajectory under one S3 root.

    API configuration may be passed explicitly or through ``CLASSIFIER_API_URL``,
    ``CLASSIFIER_MODEL``, and ``CLASSIFIER_API_KEY``. The key is never included in
    logging, manifests, cache keys, or exception messages.
    """

    if not isinstance(endpoint_url, str) or not endpoint_url:
        raise ValueError("endpoint_url must not be empty")
    if not isinstance(region_name, str) or not region_name:
        raise ValueError("region_name must not be empty")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    input_location = parse_s3_uri(input_uri)
    output_location = parse_s3_uri(output_uri)
    if _locations_overlap(input_location, output_location):
        raise ValueError("S3 input and output locations must not overlap")
    workspace = (
        Path.cwd() if workspace_parent is None else Path(workspace_parent).expanduser()
    ).resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace parent does not exist: {workspace}")
    if state_path is None:
        cache_directory = workspace / ".trajfoundry-label-cache"
        cache_directory.mkdir(parents=True, exist_ok=True)
        cache_name = hashlib.sha256(
            f"{input_location.uri}\0{output_location.uri}".encode()
        ).hexdigest()
        configured_state = cache_directory / f"{cache_name}.sqlite"
    else:
        configured_state = Path(state_path).expanduser()

    api_url = classifier_api_url or os.environ.get("CLASSIFIER_API_URL", "")
    model = classifier_model or os.environ.get("CLASSIFIER_MODEL", "")
    api_key = classifier_api_key or os.environ.get("CLASSIFIER_API_KEY", "")
    taxonomy = ScenarioTaxonomy.load(taxonomy_path)
    completion_client = ChatCompletionsClient(
        api_url=api_url,
        model=model,
        api_key=api_key,
        timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
    )
    credentials = (
        None
        if credentials_path is None
        else load_s3_credentials(Path(credentials_path).expanduser())
    )
    s3_client = create_s3_client(
        endpoint_url=endpoint_url,
        region_name=region_name,
        credentials=credentials,
    )
    try:
        manifest_bytes = _get_object_bytes(s3_client, input_location, "manifest.json")
        classifier = TrajectoryClassifier(
            client=completion_client,
            taxonomy=taxonomy,
            input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            classifier_revision=CLASSIFIER_REVISION,
            prompt_version=PROMPT_VERSION,
            max_context_chars=max_context_chars,
        )
        return _run_s3_label_job_with_client(
            s3_client,
            input_location=input_location,
            output_location=output_location,
            classifier=classifier,
            workspace=workspace,
            state_path=configured_state,
            max_workers=max_workers,
        )
    finally:
        _close_s3_client(s3_client)


__all__ = [
    "LabelJobError",
    "LabelJobResult",
    "run_s3_label_job",
]
