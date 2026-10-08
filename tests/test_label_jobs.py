from __future__ import annotations

import hashlib
import http.client
import io
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

import orjson
import pytest

from trajfoundry.canonical import trajectory_id
from trajfoundry.classification.classifier import TrajectoryClassifier
from trajfoundry.classification.client import (
    ChatCompletionsClient,
    ClassificationConfigurationError,
)
from trajfoundry.classification.taxonomy import ScenarioTaxonomy
from trajfoundry.label_jobs import (
    LabelJobError,
    _run_s3_label_job_with_client,
    _set_sub_session_id,
)
from trajfoundry.models import (
    AuditIssue,
    Message,
    Metadata,
    Severity,
    TrajectoryNode,
)
from trajfoundry.output_contract import parse_trajectory_record, project_trajectory
from trajfoundry.quality import enrich_trajectory
from trajfoundry.s3 import S3Location


class _MemoryS3:
    def __init__(self, *, delete_failures: int = 0) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.delete_failures = delete_failures

    class _Paginator:
        def __init__(self, parent: _MemoryS3) -> None:
            self.parent = parent

        def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
            return [
                {
                    "Contents": [
                        {"Key": key, "Size": len(payload)}
                        for (bucket, key), payload in sorted(
                            self.parent.objects.items()
                        )
                        if bucket == Bucket and key.startswith(Prefix)
                    ]
                }
            ]

    def get_paginator(self, _operation: str) -> _Paginator:
        return self._Paginator(self)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.calls.append(("get_object", Key))
        payload = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(payload), "ContentLength": len(payload)}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_kwargs: Any) -> None:
        self.calls.append(("put_object", Key))
        self.objects[(Bucket, Key)] = bytes(Body)

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.calls.append(("delete_object", Key))
        if self.delete_failures:
            self.delete_failures -= 1
            raise OSError("injected delete failure")
        self.objects.pop((Bucket, Key), None)


class _Completion:
    model = "classifier-test"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _messages: list[dict[str, str]]) -> str:
        self.calls += 1
        return orjson.dumps(
            {
                "scenario_label_key": "toc|Shopping|购物",
                "capability_labels": ["Tool Use"],
            }
        ).decode()


def _taxonomy(tmp_path: Path) -> ScenarioTaxonomy:
    path = tmp_path / "taxonomy.json"
    path.write_bytes(
        orjson.dumps(
            [
                {
                    "id": 10026,
                    "split": "toc",
                    "domain_l1_en": "Shopping",
                    "domain_l1_zh": "购物",
                    "domain_l2_en": "General E-commerce",
                    "domain_l2_zh": "综合电商",
                    "domain_path_en": "Shopping > General E-commerce",
                    "domain_path_zh": "购物->综合电商",
                }
            ]
        )
    )
    return ScenarioTaxonomy.load(path)


def _trajectory(
    source: str,
    *,
    session_id: str,
    created_at: str,
    synthesized: bool = False,
    complete: bool = True,
) -> dict[str, Any]:
    messages = [Message(role="user", content=f"request {source}")]
    if complete:
        messages.append(Message(role="assistant", content="done", reasoning_content=""))
    node = enrich_trajectory(
        TrajectoryNode(
            messages=messages,
            tools=[],
            model="model-under-test",
            harness="codex",
            source=source,
            metadata=Metadata(
                source_file=source,
                source_name="tokenplan",
                created_at=created_at,
                model="model-under-test",
                user_id="user-1",
                session_id=session_id,
                source_type="api-router",
                specific_source="token-plan",
            ),
        )
    )
    assert node.normalization_audit is not None
    if synthesized:
        node.normalization_audit.issues.append(
            AuditIssue(
                code="metadata_session_id_synthesized",
                stage="aggregation",
                severity=Severity.WARNING,
                path="/metadata/session_id",
                detail="missing session id was published as no_session_id",
            )
        )
        node = enrich_trajectory(node)
    value = project_trajectory(node)
    value["metadata"].pop("sub_session_id")
    return value


def _entry(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _install_v004_input(client: _MemoryS3, location: S3Location) -> bytes:
    later = _trajectory(
        "later.json", session_id="session-A", created_at="2026-09-14T02:00:00Z"
    )
    sessionless = _trajectory(
        "missing.json",
        session_id="no_session_id",
        created_at="2026-09-14T03:00:00Z",
        synthesized=True,
        complete=False,
    )
    earlier = _trajectory(
        "earlier.json", session_id="session-A", created_at="2026-09-14T01:00:00Z"
    )
    accepted_path = "generations/" + "a" * 32 + "/accepted/trajectories-00000.jsonl"
    quarantine_path = "generations/" + "a" * 32 + "/quarantine/trajectories-00000.jsonl"
    lineage_path = "generations/" + "a" * 32 + "/lineage.jsonl"
    accepted = orjson.dumps(later) + b"\n" + orjson.dumps(earlier) + b"\n"
    quarantine = orjson.dumps(sessionless) + b"\n"
    trajectories = [later, earlier, sessionless]
    lineage = b"".join(
        orjson.dumps(
            {
                "trajectory_id": trajectory_id(
                    parse_trajectory_record(value, allow_legacy_metadata=True)
                ),
                "representative": value["source"],
                "origin_count": 1,
                "origins": [
                    {
                        "source_ref": value["source"],
                        "sha256": hashlib.sha256(value["source"].encode()).hexdigest(),
                        "captured_at": value["metadata"]["created_at"],
                        "disposition": value["normalization_audit"]["tag"],
                        "reason_codes": value["normalization_audit"]["reason_codes"],
                    }
                ],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        + b"\n"
        for value in trajectories
    )
    objects = {
        accepted_path: accepted,
        quarantine_path: quarantine,
        lineage_path: lineage,
    }
    for path, payload in objects.items():
        client.objects[(location.bucket, location.key(path))] = payload
    manifest = {
        "schema_version": "trajfoundry-v3",
        "created_at": "2026-09-14T04:00:00Z",
        "input_root": "s3://bucket/raw/",
        "input_format": "tokenplan",
        "config_hash": "config",
        "token_estimator": "unicode-word-v1",
        "counts": {
            "accepted": 2,
            "quarantined_trajectories": 1,
            "quarantined_records": 0,
            "excluded_records": 0,
            "duplicate_trajectories": 0,
            "input_files": 3,
            "reason_counts": {
                reason: 1
                for reason in sessionless["normalization_audit"]["reason_codes"]
            },
            "skipped_inputs": 0,
            "skip_reason_counts": {},
        },
        "files": [_entry(path, payload) for path, payload in objects.items()],
    }
    manifest_bytes = orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS)
    client.objects[(location.bucket, location.key("manifest.json"))] = manifest_bytes
    return manifest_bytes


def _install_v4_input(client: _MemoryS3, location: S3Location) -> bytes:
    legacy_manifest = orjson.loads(_install_v004_input(client, location))
    rows: list[dict[str, Any]] = []
    lineage = b""
    for entry in legacy_manifest["files"]:
        payload = client.objects[(location.bucket, location.key(entry["path"]))]
        if entry["path"].endswith("/lineage.jsonl"):
            lineage = payload
        else:
            rows.extend(orjson.loads(line) for line in payload.splitlines())
    sub_sessions = {"earlier.json": 0, "later.json": 1, "missing.json": 0}
    new_objects: dict[str, bytes] = {"lineage.jsonl": lineage}
    for row in rows:
        sub_session_id = sub_sessions[row["source"]]
        row["metadata"]["sub_session_id"] = sub_session_id
        if row["source"] == "missing.json":
            identifier = trajectory_id(parse_trajectory_record(row))
            filename = f"no_session_id_{identifier}_sub_0.jsonl"
        else:
            filename = f"session-A_sub_{sub_session_id}.jsonl"
        new_objects[filename] = orjson.dumps(row) + b"\n"

    for key in [
        key
        for bucket, key in client.objects
        if bucket == location.bucket and key.startswith(location.prefix)
    ]:
        del client.objects[(location.bucket, key)]
    for path, payload in new_objects.items():
        client.objects[(location.bucket, location.key(path))] = payload
    manifest = {
        **legacy_manifest,
        "schema_version": "trajfoundry-v4",
        "files": [_entry(path, payload) for path, payload in new_objects.items()],
    }
    manifest_bytes = orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS)
    client.objects[(location.bucket, location.key("manifest.json"))] = manifest_bytes
    return manifest_bytes


def test_v004_job_classifies_accepted_and_quarantined_into_flat_files(
    tmp_path: Path,
) -> None:
    client = _MemoryS3()
    input_location = S3Location("bucket", "normalized/v004/dt=2026-09-14/")
    output_location = S3Location("bucket", "classified/v001/dt=2026-09-14/")
    manifest_bytes = _install_v004_input(client, input_location)
    completion = _Completion()
    classifier = TrajectoryClassifier(
        client=completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    state_path = tmp_path / "state.sqlite"
    stale_path = "old-session_sub_0.jsonl"
    old_manifest = orjson.dumps(
        {
            "schema_version": "trajfoundry-classification-v1",
            "files": [{"path": stale_path}],
        }
    )
    client.objects[(output_location.bucket, output_location.key("manifest.json"))] = (
        old_manifest
    )
    client.objects[(output_location.bucket, output_location.key(stale_path))] = b"old\n"

    result = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=classifier,
        workspace=tmp_path,
        state_path=state_path,
        max_workers=2,
    )

    assert result.input_trajectories == 3
    assert result.classified == 3
    assert result.failed == 0
    assert completion.calls == 3
    assert (
        output_location.bucket,
        output_location.key(stale_path),
    ) not in client.objects
    manifest_put = client.calls.index(
        ("put_object", output_location.key("manifest.json"))
    )
    stale_delete = client.calls.index(
        ("delete_object", output_location.key(stale_path))
    )
    assert manifest_put < stale_delete
    output_manifest = orjson.loads(
        client.objects[(output_location.bucket, output_location.key("manifest.json"))]
    )
    for field in (
        "catalog_sha256",
        "policy_sha256",
        "prompt_sha256",
        "strategy_fingerprint",
        "config_hash",
    ):
        assert output_manifest[field] == getattr(classifier, field)
    assert output_manifest["classification_policy"]["scenario_selection"] == (
        "exactly_one_l1"
    )
    assert output_manifest["classification_policy"]["capability_max"] == 2
    trajectory_files = [
        entry for entry in output_manifest["files"] if entry["kind"] == "trajectory"
    ]
    paths = [entry["path"] for entry in trajectory_files]
    assert "session-A_sub_0.jsonl" in paths
    assert "session-A_sub_1.jsonl" in paths
    sessionless_path = next(
        path
        for path in paths
        if path.startswith("no_session_id_") and path.endswith("_sub_0.jsonl")
    )
    assert len(sessionless_path.split("_")[-3]) == 64
    rows = {
        path: orjson.loads(
            client.objects[(output_location.bucket, output_location.key(path))]
        )
        for path in paths
    }
    assert rows["session-A_sub_0.jsonl"]["source"] == "earlier.json"
    assert rows["session-A_sub_1.jsonl"]["source"] == "later.json"
    assert rows["session-A_sub_0.jsonl"]["metadata"]["sub_session_id"] == 0
    assert rows["session-A_sub_1.jsonl"]["metadata"]["sub_session_id"] == 1
    assert rows[sessionless_path]["metadata"]["sub_session_id"] == 0
    assert all(row["classification"]["status"] == "accepted" for row in rows.values())
    assert all("trajectory_id" not in row for row in rows.values())

    rerun = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=classifier,
        workspace=tmp_path,
        state_path=state_path,
        max_workers=2,
    )

    assert rerun.cache_hits == 3
    assert completion.calls == 3


def test_invalid_cached_l2_shape_is_rejected_and_reclassified(
    tmp_path: Path,
) -> None:
    """A stale accepted cache row must not bypass the L1 output contract."""

    client = _MemoryS3()
    input_location = S3Location("bucket", "normalized/v004/dt=2026-09-14/")
    output_location = S3Location("bucket", "classified/v001/dt=2026-09-14/")
    manifest_bytes = _install_v004_input(client, input_location)
    completion = _Completion()
    classifier = TrajectoryClassifier(
        client=completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    state_path = tmp_path / "stale-l2-state.sqlite"

    first = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=classifier,
        workspace=tmp_path,
        state_path=state_path,
        max_workers=1,
    )
    assert first.cache_hits == 0
    assert completion.calls == 3

    with sqlite3.connect(state_path) as connection:
        cache_key, result = connection.execute(
            "SELECT cache_key, result FROM classifications ORDER BY cache_key LIMIT 1"
        ).fetchone()
        stale = orjson.loads(result)
        stale["scenario_labels"] = [
            stale["scenario_labels"][0],
            stale["scenario_labels"][0],
        ]
        connection.execute(
            "UPDATE classifications SET result = ? WHERE cache_key = ?",
            (orjson.dumps(stale), cache_key),
        )
        connection.commit()

    rerun = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=classifier,
        workspace=tmp_path,
        state_path=state_path,
        max_workers=1,
    )
    assert rerun.cache_hits == 2
    assert completion.calls == 4


def test_strategy_fingerprint_is_part_of_cache_namespace(tmp_path: Path) -> None:
    client = _MemoryS3()
    input_location = S3Location("bucket", "normalized/v004/dt=2026-09-14/")
    output_location = S3Location("bucket", "classified/v001/dt=2026-09-14/")
    manifest_bytes = _install_v004_input(client, input_location)
    first_completion = _Completion()
    first_classifier = TrajectoryClassifier(
        client=first_completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    state_path = tmp_path / "strategy-state.sqlite"

    first = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=first_classifier,
        workspace=tmp_path,
        state_path=state_path,
        max_workers=1,
    )
    assert first.cache_hits == 0
    assert first_completion.calls == 3

    class AlternateStrategyClassifier(TrajectoryClassifier):
        @property
        def strategy_fingerprint(self) -> str:
            return "b" * 64

    second_completion = _Completion()
    second_classifier = AlternateStrategyClassifier(
        client=second_completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    assert second_classifier.config_hash != first_classifier.config_hash
    rerun = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=second_classifier,
        workspace=tmp_path,
        state_path=state_path,
        max_workers=1,
    )
    assert rerun.cache_hits == 0
    assert second_completion.calls == 3


def test_output_manifest_is_write_only_on_first_classification_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _MemoryS3()
    input_location = S3Location("bucket", "normalized/v004/dt=2026-09-14/")
    output_location = S3Location("bucket", "classified/v001/dt=2026-09-14/")
    manifest_bytes = _install_v004_input(client, input_location)
    stale_path = "orphan.jsonl"
    client.objects[(output_location.bucket, output_location.key(stale_path))] = (
        b"orphan\n"
    )
    previous_manifest = b'{"previous":true}'
    output_manifest_key = output_location.key("manifest.json")
    client.objects[(output_location.bucket, output_manifest_key)] = previous_manifest

    original_get_object = client.get_object
    manifest_reads: list[str] = []

    def deny_output_manifest_read(*, Bucket: str, Key: str) -> dict[str, Any]:
        if Key == output_manifest_key:
            manifest_reads.append(Key)
            raise PermissionError("output manifest reads are forbidden")
        return original_get_object(Bucket=Bucket, Key=Key)

    monkeypatch.setattr(client, "get_object", deny_output_manifest_read)

    completion = _Completion()
    classifier = TrajectoryClassifier(
        client=completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )

    result = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=classifier,
        workspace=tmp_path,
        state_path=tmp_path / "manifest-write-only-state.sqlite",
        max_workers=1,
    )

    assert result.input_trajectories == 3
    assert result.classified == 3
    assert result.cache_hits == 0
    assert completion.calls == 3
    assert (
        output_location.bucket,
        output_location.key(stale_path),
    ) not in client.objects
    assert manifest_reads == []
    assert ("get_object", output_manifest_key) not in client.calls
    assert (
        client.objects[(output_location.bucket, output_manifest_key)]
        != previous_manifest
    )
    published_manifest = orjson.loads(
        client.objects[(output_location.bucket, output_manifest_key)]
    )
    assert published_manifest["schema_version"] == "trajfoundry-classification-v1"


def test_post_publication_delete_failure_is_recoverable_on_rerun(
    tmp_path: Path,
) -> None:
    client = _MemoryS3(delete_failures=1)
    input_location = S3Location("bucket", "normalized/v004/dt=2026-09-14/")
    output_location = S3Location("bucket", "classified/v001/dt=2026-09-14/")
    manifest_bytes = _install_v004_input(client, input_location)
    stale_path = "old-session_sub_0.jsonl"
    client.objects[(output_location.bucket, output_location.key(stale_path))] = b"old\n"
    old_manifest = orjson.dumps(
        {
            "schema_version": "trajfoundry-classification-v1",
            "files": [{"path": stale_path}],
        }
    )
    client.objects[(output_location.bucket, output_location.key("manifest.json"))] = (
        old_manifest
    )
    completion = _Completion()
    classifier = TrajectoryClassifier(
        client=completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    state_path = tmp_path / "post-publication-state.sqlite"

    with pytest.raises(LabelJobError, match="post-publication"):
        _run_s3_label_job_with_client(
            client,
            input_location=input_location,
            output_location=output_location,
            classifier=classifier,
            workspace=tmp_path,
            state_path=state_path,
            max_workers=1,
        )

    # The new manifest is already committed, while the stale object remains
    # because its post-publication deletion failed.
    published_manifest = orjson.loads(
        client.objects[(output_location.bucket, output_location.key("manifest.json"))]
    )
    assert published_manifest["schema_version"] == "trajfoundry-classification-v1"
    assert published_manifest != orjson.loads(old_manifest)
    assert (output_location.bucket, output_location.key(stale_path)) in client.objects
    assert completion.calls == 3

    client.delete_failures = 0
    result = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=classifier,
        workspace=tmp_path,
        state_path=state_path,
        max_workers=1,
    )

    assert result.cache_hits == 3
    assert completion.calls == 3
    assert (
        output_location.bucket,
        output_location.key(stale_path),
    ) not in client.objects


def test_v004_nested_subagent_inherits_assigned_sub_session_id() -> None:
    value = {
        "metadata": {"sub_session_id": 0},
        "sub_agent_trajectory": {
            "spawn-1": {
                "metadata": {"sub_session_id": 0},
                "sub_agent_trajectory": {
                    "spawn-2": {"metadata": {"sub_session_id": 0}}
                },
            }
        },
    }

    _set_sub_session_id(value, 3)

    assert value["metadata"]["sub_session_id"] == 3
    child = value["sub_agent_trajectory"]["spawn-1"]
    assert child["metadata"]["sub_session_id"] == 3
    assert child["sub_agent_trajectory"]["spawn-2"]["metadata"]["sub_session_id"] == 3


def test_job_rejects_corrupt_manifest_object_before_model_calls(tmp_path: Path) -> None:
    client = _MemoryS3()
    input_location = S3Location("bucket", "normalized/v004/dt=2026-09-14/")
    output_location = S3Location("bucket", "classified/v001/dt=2026-09-14/")
    manifest_bytes = _install_v004_input(client, input_location)
    lineage_key = next(
        key
        for bucket, key in client.objects
        if bucket == input_location.bucket and key.endswith("/lineage.jsonl")
    )
    client.objects[(input_location.bucket, lineage_key)] += b"corrupt"
    completion = _Completion()
    classifier = TrajectoryClassifier(
        client=completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )

    with pytest.raises(LabelJobError, match="normalized input failed validation"):
        _run_s3_label_job_with_client(
            client,
            input_location=input_location,
            output_location=output_location,
            classifier=classifier,
            workspace=tmp_path,
            state_path=tmp_path / "state.sqlite",
            max_workers=2,
        )

    assert completion.calls == 0


def test_late_configuration_error_does_not_change_published_output(
    tmp_path: Path,
) -> None:
    class FatalCompletion(_Completion):
        def complete(self, messages: list[dict[str, str]]) -> str:
            if self.calls == 1:
                raise ClassificationConfigurationError("HTTP 422")
            return super().complete(messages)

    client = _MemoryS3()
    input_location = S3Location("bucket", "normalized/v004/dt=2026-09-14/")
    output_location = S3Location("bucket", "classified/v001/dt=2026-09-14/")
    manifest_bytes = _install_v004_input(client, input_location)
    previous_manifest = b'{"previous":true}'
    client.objects[(output_location.bucket, output_location.key("manifest.json"))] = (
        previous_manifest
    )
    completion = FatalCompletion()
    classifier = TrajectoryClassifier(
        client=completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )

    with pytest.raises(ClassificationConfigurationError, match="HTTP 422"):
        _run_s3_label_job_with_client(
            client,
            input_location=input_location,
            output_location=output_location,
            classifier=classifier,
            workspace=tmp_path,
            state_path=tmp_path / "state.sqlite",
            max_workers=1,
        )

    assert completion.calls == 1
    assert (
        client.objects[(output_location.bucket, output_location.key("manifest.json"))]
        == previous_manifest
    )
    assert not any(
        operation == "put_object" and key.startswith(output_location.prefix)
        for operation, key in client.calls
    )


def test_v4_flat_input_is_classified_without_renumbering(tmp_path: Path) -> None:
    client = _MemoryS3()
    input_location = S3Location("bucket", "normalized/v005/dt=2026-09-14/")
    output_location = S3Location("bucket", "classified/v001/dt=2026-09-14/")
    manifest_bytes = _install_v4_input(client, input_location)
    completion = _Completion()
    classifier = TrajectoryClassifier(
        client=completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )

    result = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=classifier,
        workspace=tmp_path,
        state_path=tmp_path / "v4-state.sqlite",
        max_workers=2,
    )

    assert result.input_trajectories == 3
    manifest = orjson.loads(
        client.objects[(output_location.bucket, output_location.key("manifest.json"))]
    )
    paths = {
        entry["path"] for entry in manifest["files"] if entry["kind"] == "trajectory"
    }
    assert {"session-A_sub_0.jsonl", "session-A_sub_1.jsonl"} <= paths
    for path in paths:
        row = orjson.loads(
            client.objects[(output_location.bucket, output_location.key(path))]
        )
        if path.startswith("session-A_sub_"):
            assert row["metadata"]["sub_session_id"] == int(
                path.removesuffix(".jsonl").rsplit("_", 1)[1]
            )


def test_invalid_model_output_still_writes_every_complete_trajectory(
    tmp_path: Path,
) -> None:
    class InvalidCompletion(_Completion):
        def complete(self, _messages: list[dict[str, str]]) -> str:
            self.calls += 1
            return orjson.dumps(
                {
                    "scenario_label_ids": [99999],
                    "capability_labels": ["Tool Use"],
                }
            ).decode()

    client = _MemoryS3()
    input_location = S3Location("bucket", "normalized/v005/dt=2026-09-14/")
    output_location = S3Location("bucket", "classified/v001/dt=2026-09-14/")
    manifest_bytes = _install_v4_input(client, input_location)
    completion = InvalidCompletion()
    classifier = TrajectoryClassifier(
        client=completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )

    result = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=classifier,
        workspace=tmp_path,
        state_path=tmp_path / "failed-state.sqlite",
        max_workers=2,
    )

    assert result.input_trajectories == result.failed == 3
    assert result.classified == 0
    manifest = orjson.loads(
        client.objects[(output_location.bucket, output_location.key("manifest.json"))]
    )
    trajectory_entries = [
        entry for entry in manifest["files"] if entry["kind"] == "trajectory"
    ]
    assert len(trajectory_entries) == 3
    for entry in trajectory_entries:
        row = orjson.loads(
            client.objects[(output_location.bucket, output_location.key(entry["path"]))]
        )
        assert row["messages"]
        assert row["metadata"]["session_id"]
        assert row["normalization_audit"]
        assert row["classification"]["status"] == "failed"
        assert row["classification"]["reason"] == "invalid_model_output"


def test_network_exhaustion_publishes_failed_row_and_retries_only_it_on_rerun(
    tmp_path: Path,
) -> None:
    client = _MemoryS3()
    input_location = S3Location("bucket", "normalized/v005/dt=2026-09-14/")
    output_location = S3Location("bucket", "classified/v001/dt=2026-09-14/")
    manifest_bytes = _install_v4_input(client, input_location)
    calls = {"earlier.json": 0, "later.json": 0, "missing.json": 0}
    calls_lock = Lock()
    recovered = False
    decision = orjson.dumps(
        {
            "scenario_label_key": "toc|Shopping|购物",
            "capability_labels": ["Tool Use"],
        }
    ).decode()

    def opener(request, **_kwargs):
        messages = orjson.loads(request.data)["messages"]
        source = next(
            source for source in calls if f"request {source}" in messages[-1]["content"]
        )
        with calls_lock:
            calls[source] += 1
        if source == "earlier.json" and not recovered:
            raise http.client.RemoteDisconnected("private transport detail")
        return io.BytesIO(
            orjson.dumps({"choices": [{"message": {"content": decision}}]})
        )

    completion = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="classifier-test",
        api_key="test-key",
        max_retries=1,
        opener=opener,
        sleeper=lambda _delay: None,
    )
    classifier = TrajectoryClassifier(
        client=completion,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    state_path = tmp_path / "network-state.sqlite"

    def published_rows(expected_counts: dict[str, int]) -> dict[str, Any]:
        manifest = orjson.loads(
            client.objects[
                (output_location.bucket, output_location.key("manifest.json"))
            ]
        )
        assert manifest["schema_version"] == "trajfoundry-classification-v1"
        assert manifest["counts"] == expected_counts
        rows = {}
        for entry in manifest["files"]:
            payload = client.objects[
                (output_location.bucket, output_location.key(entry["path"]))
            ]
            assert entry["bytes"] == len(payload)
            assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
            if entry["kind"] == "trajectory":
                row = orjson.loads(payload)
                assert entry["classification_status"] == row["classification"]["status"]
                rows[row["source"]] = row
        assert set(rows) == set(calls)
        return rows

    result = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=classifier,
        workspace=tmp_path,
        state_path=state_path,
        max_workers=2,
    )

    assert result.validation_valid
    assert (
        result.input_trajectories,
        result.classified,
        result.failed,
        result.cache_hits,
    ) == (3, 2, 1, 0)
    assert calls == {"earlier.json": 2, "later.json": 1, "missing.json": 1}
    first_rows = published_rows(
        {"input_trajectories": 3, "classified": 2, "failed": 1, "cache_hits": 0}
    )
    failed_row = first_rows["earlier.json"]
    assert failed_row["messages"]
    assert failed_row["metadata"]["session_id"] == "session-A"
    assert failed_row["normalization_audit"]
    assert failed_row["classification"]["status"] == "failed"
    assert failed_row["classification"]["reason"] == "network_retry_exhausted"
    assert b"private transport detail" not in orjson.dumps(failed_row)
    for source in ("later.json", "missing.json"):
        assert first_rows[source]["classification"]["status"] == "accepted"

    recovered = True
    rerun = _run_s3_label_job_with_client(
        client,
        input_location=input_location,
        output_location=output_location,
        classifier=classifier,
        workspace=tmp_path,
        state_path=state_path,
        max_workers=2,
    )

    assert rerun.validation_valid
    assert (
        rerun.input_trajectories,
        rerun.classified,
        rerun.failed,
        rerun.cache_hits,
    ) == (3, 3, 0, 2)
    assert calls == {"earlier.json": 3, "later.json": 1, "missing.json": 1}
    recovered_rows = published_rows(
        {"input_trajectories": 3, "classified": 3, "failed": 0, "cache_hits": 2}
    )
    assert recovered_rows["earlier.json"]["classification"]["status"] == "accepted"
    for source in ("later.json", "missing.json"):
        assert recovered_rows[source] == first_rows[source]
