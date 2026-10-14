from __future__ import annotations

from hashlib import sha256
from io import BytesIO

import pytest

from trajfoundry.canonical import canonical_json, semantic_payload, trajectory_id
from trajfoundry.export import ExportStats, trajectory_filename
from trajfoundry.feedback import feedback_fields
from trajfoundry.json_codec import dumps
from trajfoundry.models import Message, Metadata, TrajectoryNode
from trajfoundry.output_contract import project_trajectory
from trajfoundry.quality import enrich_trajectory, is_strict_sample
from trajfoundry.validation import ValidationFile, validate_output_backend

from .test_total_tokens import _tree


class MemoryBackend:
    def __init__(self, objects):
        self.objects = objects

    def open_file(self, relative_path):
        value = self.objects[relative_path]
        return ValidationFile(size=len(value), stream=BytesIO(value))

    def iter_generated_jsonl_paths(self, generation_ids):
        return self.objects.keys()


def _timed_node(*, mounted=False, real_root=False):
    node = (
        _tree()
        if mounted
        else enrich_trajectory(
            TrajectoryNode(
                messages=[
                    Message(role="assistant", content="done", reasoning_content="")
                ],
                tools=[],
                source="sample.json",
                metadata=Metadata(source_file="sample.json"),
            )
        )
    )
    node.metadata.created_at = "2026-10-12T23:59:59Z"
    if real_root:
        node.metadata.session_id = "real-session"
    for child in (node.sub_agent_trajectory or {}).values():
        child.metadata.created_at = "2026-10-13T08:00:00Z"
    return enrich_trajectory(node)


def _remove_tokens(record):
    record.pop("total_tokens")
    for child in (record.get("sub_agent_trajectory") or {}).values():
        _remove_tokens(child)


def _output(node, *, hash_policy=0, declared_policy=0, historical=False):
    identifier = trajectory_id(node, sessionless_window_hours=hash_policy)
    record = project_trajectory(node)
    if historical:
        _remove_tokens(record)
    filename = trajectory_filename(node, identifier)
    trajectory_bytes = dumps(record, sort_keys=True) + b"\n"
    lineage_bytes = (
        dumps(
            {
                "trajectory_id": identifier,
                "representative": "sample.json",
                "origin_count": 1,
                "origins": [
                    {
                        "source_ref": "sample.json",
                        "sha256": "a" * 64,
                        "captured_at": node.metadata.created_at,
                        "disposition": node.normalization_audit.tag.value,
                        "reason_codes": node.normalization_audit.reason_codes,
                    }
                ],
            },
            sort_keys=True,
        )
        + b"\n"
    )
    objects = {filename: trajectory_bytes, "lineage.jsonl": lineage_bytes}
    strict = is_strict_sample(node)
    stats = ExportStats(
        accepted=int(strict),
        quarantined_trajectories=int(not strict),
        input_files=1,
    )
    if not strict:
        stats.add_reasons(node.normalization_audit.reason_codes)
    manifest = {
        "schema_version": "trajfoundry-v4",
        "created_at": "2026-10-13T08:00:00Z",
        "input_root": "/synthetic",
        "input_format": "freerouter",
        "config_hash": "synthetic",
        "token_estimator": "unicode-word-v1",
        "counts": stats.__dict__,
        "files": [
            {"path": path, "bytes": len(payload), "sha256": sha256(payload).hexdigest()}
            for path, payload in objects.items()
        ],
    }
    if declared_policy:
        manifest["sessionless_window_hours"] = declared_policy
    if not historical:
        manifest.update(
            feedback_fields(
                stats,
                total_bytes=len(trajectory_bytes),
                started_at_ms=1000,
                completed_at_ms=2000,
            )
        )
    return manifest, MemoryBackend(objects)


@pytest.mark.parametrize(
    "mounted,real_root", [(False, False), (True, False), (True, True)]
)
def test_historical_timed_root_and_child_identity_still_validates(mounted, real_root):
    node = _timed_node(mounted=mounted, real_root=real_root)
    assert trajectory_id(node, sessionless_window_hours=0) != trajectory_id(
        node, sessionless_window_hours=8
    )
    manifest, backend = _output(node, historical=True)
    report = validate_output_backend(dumps(manifest), backend)
    assert report.valid, report.errors


@pytest.mark.parametrize("written,declared", [(0, 0), (8, 8), (0, 8), (8, 0)])
@pytest.mark.parametrize("mounted", [False, True])
def test_manifest_policy_selects_one_exact_identity_with_new_statistics(
    written, declared, mounted
):
    manifest, backend = _output(
        _timed_node(mounted=mounted), hash_policy=written, declared_policy=declared
    )
    report = validate_output_backend(dumps(manifest), backend)
    assert report.valid is (written == declared), report.errors
    if written != declared:
        assert "lineage trajectory IDs do not match trajectory content" in report.errors
        assert any("filename does not match" in error for error in report.errors)


def test_window_policy_is_recursive_even_when_root_has_real_session():
    node = _timed_node(mounted=True, real_root=True)
    legacy = semantic_payload(node, sessionless_window_hours=0)
    current = semantic_payload(node, sessionless_window_hours=8)
    assert "sessionless_window" not in current["aggregation_identity"]
    assert (
        "sessionless_window"
        not in legacy["sub_agent_trajectory"]["spawn-1"]["aggregation_identity"]
    )
    assert (
        current["sub_agent_trajectory"]["spawn-1"]["aggregation_identity"][
            "sessionless_window"
        ]
        == "2026-10-13T16:00:00+08:00"
    )
    # The old algorithm is exactly the current semantic payload without the
    # newly introduced window keys, including nested children.
    current["sub_agent_trajectory"]["spawn-1"]["aggregation_identity"].pop(
        "sessionless_window"
    )
    assert sha256(canonical_json(current)).hexdigest() == trajectory_id(
        node, sessionless_window_hours=0
    )


def test_default_identity_preserves_this_deployments_policy():
    node = _timed_node(mounted=True)
    assert trajectory_id(node) == trajectory_id(node, sessionless_window_hours=0)
    assert semantic_payload(node) == semantic_payload(node, sessionless_window_hours=0)


@pytest.mark.parametrize("policy", [0, 4, 16, "8", True])
def test_unknown_or_illtyped_manifest_policy_fails_closed(policy):
    manifest, backend = _output(_timed_node())
    manifest["sessionless_window_hours"] = policy
    assert not validate_output_backend(dumps(manifest), backend).valid
