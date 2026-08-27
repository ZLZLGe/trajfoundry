import zlib
from pathlib import Path
from typing import Any

import orjson
import pytest

from trajfoundry.audit_codes import RESPONSES_UNSUPPORTED_CALL_EVIDENCE
from trajfoundry.canonical import trajectory_id
from trajfoundry.export import OutputSet
from trajfoundry.models import (
    AgentMessageRecord,
    AuditIssue,
    AuditTag,
    FunctionCall,
    Message,
    Metadata,
    NormalizationAudit,
    ServerToolCall,
    Severity,
    ToolCall,
    ToolDefinition,
    TrajectoryNode,
)
from trajfoundry.output_contract import (
    OutputContractError,
    project_message,
    project_trajectory,
)
from trajfoundry.quality import StaleDerivedFieldsError, enrich_trajectory
from trajfoundry.state import StateStore


def _trajectory(source: str = "source.json") -> TrajectoryNode:
    return enrich_trajectory(
        TrajectoryNode(
            messages=[
                Message(
                    role="assistant",
                    content="",
                    reasoning_content="think",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=FunctionCall(
                                name="read", arguments={"path": "/tmp/a"}
                            ),
                        )
                    ],
                ),
                Message(
                    role="tool",
                    content="ok",
                    tool_call_id="call-1",
                    name="read",
                ),
                Message(role="assistant", content="done", reasoning_content=""),
            ],
            tools=[
                ToolDefinition(
                    name="read",
                    parameters={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                )
            ],
            server_tool_calls=[
                ServerToolCall(
                    name="web_search",
                    id="server-1",
                    arguments={"query": "x"},
                    origin="response",
                    result={"items": []},
                )
            ],
            source=source,
            metadata=Metadata(source_file=source),
        )
    )


def _trajectory_with_relay_mount() -> TrajectoryNode:
    child = TrajectoryNode(
        messages=[Message(role="assistant", content="done", reasoning_content="")],
        tools=[],
        source="child.json",
        metadata=Metadata(source_file="child.json"),
    )
    return enrich_trajectory(
        TrajectoryNode(
            messages=[
                Message(
                    role="assistant",
                    content="",
                    reasoning_content="",
                    tool_calls=[
                        ToolCall(
                            id="spawn-1",
                            function=FunctionCall(name="spawn_agent", arguments={}),
                        )
                    ],
                ),
                Message(
                    role="tool",
                    content="spawned",
                    tool_call_id="spawn-1",
                    name="spawn_agent",
                ),
                Message(role="assistant", content="done", reasoning_content=""),
            ],
            tools=[ToolDefinition(name="spawn_agent")],
            agent_messages=[
                AgentMessageRecord(
                    origin="history",
                    item_index=3,
                    item={
                        "type": "agent_message",
                        "id": "agent-message-1",
                        "author": "/root/child",
                        "recipient": "/root",
                        "content": [
                            {"type": "input_text", "text": "complete"},
                            {
                                "type": "encrypted_content",
                                "encrypted_content": "opaque",
                            },
                        ],
                        "unknown": {"preserved": True},
                    },
                )
            ],
            source="parent.json",
            metadata=Metadata(source_file="parent.json"),
            sub_agent_trajectory={"spawn-1": child},
            sub_agent_relay_mounts={"spawn-1": "agent-message-1"},
        )
    )


def _put_non_json_value(node: TrajectoryNode, target: str, value: Any) -> None:
    if target == "tool_arguments":
        assert node.messages[0].tool_calls
        node.messages[0].tool_calls[0].function.arguments = value
    elif target == "reasoning":
        node.messages[0].reasoning = value
    elif target == "reasoning_details":
        node.messages[0].reasoning_details = [{"raw": value}]
    elif target == "tool_parameters":
        node.tools[0].parameters = {"raw": value}
    elif target == "server_arguments":
        node.server_tool_calls[0].arguments = value
    elif target == "server_result":
        node.server_tool_calls[0].result = {"raw": value}
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(target)


def _put_stale_derived_value(node: TrajectoryNode, target: str) -> None:
    assert node.normalization_audit is not None
    if target == "total_rounds":
        node.total_rounds = 999
    elif target == "tool_call_check":
        node.tool_call_check.total_calls = 777
    elif target == "tool_defs_tag":
        node.tool_defs_tag = "incomplete"
    elif target == "audit_tag":
        node.normalization_audit.tag = AuditTag.QUARANTINED
    elif target == "reason_codes":
        node.normalization_audit.reason_codes = ["made_up"]
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(target)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("tool_arguments", (1, 2)),
        ("tool_arguments", b"bytes"),
        ("tool_arguments", {1: "non-string key"}),
        ("tool_arguments", float("nan")),
        ("tool_arguments", float("inf")),
        ("tool_arguments", float("-inf")),
        ("reasoning", (1, 2)),
        ("reasoning_details", (1, 2)),
        ("tool_parameters", (1, 2)),
        ("server_arguments", (1, 2)),
        ("server_result", b"bytes"),
    ],
    ids=[
        "tuple-arguments",
        "bytes-arguments",
        "non-string-object-key",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "tuple-reasoning",
        "tuple-reasoning-detail",
        "tuple-tool-parameter",
        "tuple-server-arguments",
        "bytes-server-result",
    ],
)
def test_projection_rejects_non_json_python_values_before_serialization(
    target: str, value: Any
) -> None:
    node = _trajectory()
    _put_non_json_value(node, target, value)

    with pytest.raises(OutputContractError):
        project_trajectory(node)


def test_export_rejects_value_orjson_would_silently_coerce(tmp_path: Path) -> None:
    node = _trajectory()
    _put_non_json_value(node, "tool_arguments", (1, 2))
    output = OutputSet(tmp_path)

    with pytest.raises(OutputContractError):
        output.write_trajectory(node, [])

    output.abort()
    assert not list(tmp_path.rglob("*.jsonl"))


def test_export_rejects_type_correct_stale_derived_fields(tmp_path: Path) -> None:
    node = _trajectory()
    node.total_rounds = 999
    output = OutputSet(tmp_path)

    with pytest.raises(StaleDerivedFieldsError):
        output.write_trajectory(node, [])

    output.abort()
    assert not list(tmp_path.rglob("*.jsonl"))


def test_message_projection_rejects_role_fields_added_after_validation() -> None:
    message = Message(role="user", content="hello")
    message.reasoning_content = "must not be silently dropped"

    with pytest.raises(OutputContractError):
        project_message(message)


def test_trajectory_projection_rejects_invalid_assignment() -> None:
    node = _trajectory()
    node.total_rounds = "3"  # type: ignore[assignment]

    with pytest.raises(OutputContractError):
        project_trajectory(node)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spawn_calls", 1),
        ("mounted_subs", 1),
        ("subtree_complete", False),
        ("unkeyed_mounts", 1),
        ("relay_mounts", 1),
        ("trailing_unanswered_call", True),
        ("no_final_assistant_turn", True),
    ],
)
def test_projection_rejects_stale_completeness_fields(
    field: str, value: object
) -> None:
    node = _trajectory()
    assert node.completeness is not None
    setattr(node.completeness, field, value)

    with pytest.raises(OutputContractError, match=field):
        project_trajectory(node)


def test_projection_rejects_completeness_tag_inconsistent_with_messages() -> None:
    node = _trajectory()
    node.completeness_tag = "incomplete_main_no_sub"

    with pytest.raises(OutputContractError, match="completeness_tag"):
        project_trajectory(node)


def test_projection_validates_relay_mount_cross_fields() -> None:
    node = _trajectory_with_relay_mount()
    projected = project_trajectory(node)
    assert projected["completeness"]["relay_mounts"] == 1
    assert projected["agent_messages"][0]["item"] == node.agent_messages[0].item

    wrong_child = node.model_copy(deep=True)
    wrong_child.sub_agent_relay_mounts = {"not-mounted": "agent-message-1"}
    with pytest.raises(OutputContractError, match="mounted child"):
        project_trajectory(wrong_child)

    same_call = node.model_copy(deep=True)
    same_call.sub_agent_relay_mounts = {"spawn-1": "spawn-1"}
    with pytest.raises(OutputContractError, match="differ from spawn"):
        project_trajectory(same_call)

    missing_evidence = node.model_copy(deep=True)
    missing_evidence.agent_messages = []
    with pytest.raises(OutputContractError, match="exactly one local agent_message"):
        project_trajectory(missing_evidence)


def test_nested_projection_rejects_completeness_from_source_model() -> None:
    child = _trajectory("child.json")

    with pytest.raises(OutputContractError, match="nested trajectory source"):
        project_trajectory(child, top_level=False)


def test_state_rejects_mismatched_trajectory_id(tmp_path: Path) -> None:
    node = _trajectory()

    with StateStore(tmp_path / "state.sqlite") as state:
        with pytest.raises(ValueError, match="does not match trajectory content"):
            state.put_trajectory(
                "0" * 64,
                node,
                representative_key="source.json",
                source_ref="source.json",
                sha256="a" * 64,
                captured_at="",
                disposition="pass",
                reason_codes=[],
            )
        assert state.trajectory_count() == 0


@pytest.mark.parametrize(
    "target",
    [
        "total_rounds",
        "tool_call_check",
        "tool_defs_tag",
        "audit_tag",
        "reason_codes",
    ],
)
def test_state_rejects_type_correct_stale_derived_fields(
    tmp_path: Path, target: str
) -> None:
    node = _trajectory()
    _put_stale_derived_value(node, target)
    identifier = trajectory_id(node)
    assert node.normalization_audit is not None

    with StateStore(tmp_path / "state.sqlite") as state:
        with pytest.raises(StaleDerivedFieldsError):
            state.put_trajectory(
                identifier,
                node,
                representative_key="source.json",
                source_ref="source.json",
                sha256="a" * 64,
                captured_at="",
                disposition=node.normalization_audit.tag.value,
                reason_codes=node.normalization_audit.reason_codes,
            )
        assert state.trajectory_count() == 0


def test_state_trajectory_round_trip_preserves_projection_and_hash(
    tmp_path: Path,
) -> None:
    node = _trajectory()
    identifier = trajectory_id(node)

    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_trajectory(
            identifier,
            node,
            representative_key="source.json",
            source_ref="source.json",
            sha256="a" * 64,
            captured_at="",
            disposition="pass",
            reason_codes=[],
        )
        restored_id, restored, _ = next(state.iter_trajectories())

    assert restored_id == identifier
    assert trajectory_id(restored) == identifier
    assert project_trajectory(restored) == project_trajectory(node)


def test_state_batches_origins_behind_one_trajectory_validation(tmp_path: Path) -> None:
    node = _trajectory()
    identifier = trajectory_id(node)

    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_trajectory_with_origins(
            identifier,
            node,
            representative_key="source.json",
            origins=(
                {
                    "source_ref": "source-a.json",
                    "sha256": "a" * 64,
                    "captured_at": "",
                },
                {
                    "source_ref": "source-b.json",
                    "sha256": "b" * 64,
                    "captured_at": "",
                },
            ),
            disposition="pass",
            reason_codes=[],
        )
        _, _, origins = next(state.iter_trajectories())

    assert {origin["source_ref"] for origin in origins} == {
        "source-a.json",
        "source-b.json",
    }


def test_provider_evidence_separates_semantically_different_trajectories(
    tmp_path: Path,
) -> None:
    messages = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            content="",
            reasoning_content="",
            tool_calls=[
                ToolCall(
                    id="call-ghost",
                    function=FunctionCall(name="ghost", arguments={}),
                )
            ],
        ),
        Message(
            role="tool",
            content="unsupported call: ghost",
            tool_call_id="call-ghost",
            name="ghost",
        ),
        Message(role="assistant", content="done", reasoning_content=""),
    ]
    tools = [ToolDefinition(name="ghost")]
    anthropic = enrich_trajectory(
        TrajectoryNode(
            messages=messages,
            tools=tools,
            source="anthropic.json",
            metadata=Metadata(source_file="anthropic.json"),
        )
    )
    responses = enrich_trajectory(
        TrajectoryNode(
            messages=messages,
            tools=tools,
            source="responses.json",
            metadata=Metadata(source_file="responses.json"),
            normalization_audit=NormalizationAudit(
                tag=AuditTag.PASS,
                issues=[
                    AuditIssue(
                        code=RESPONSES_UNSUPPORTED_CALL_EVIDENCE,
                        stage="responses",
                        severity=Severity.WARNING,
                        path="request_body.input[2]",
                        detail=(
                            "provider rejected client tool call "
                            "id='call-ghost' name='ghost' as unsupported"
                        ),
                    )
                ],
            ),
        )
    )

    anthropic_id = trajectory_id(anthropic)
    responses_id = trajectory_id(responses)
    assert anthropic.normalization_audit is not None
    assert anthropic.normalization_audit.tag == AuditTag.PASS
    assert responses.normalization_audit is not None
    assert responses.normalization_audit.tag == AuditTag.QUARANTINED
    assert anthropic_id != responses_id

    with StateStore(tmp_path / "state.sqlite") as state:
        for identifier, node, source_ref in (
            (anthropic_id, anthropic, "anthropic.json"),
            (responses_id, responses, "responses.json"),
        ):
            assert node.normalization_audit is not None
            state.put_trajectory(
                identifier,
                node,
                representative_key=source_ref,
                source_ref=source_ref,
                sha256=("a" if source_ref.startswith("anthropic") else "b") * 64,
                captured_at="",
                disposition=node.normalization_audit.tag.value,
                reason_codes=node.normalization_audit.reason_codes,
            )

        stored = list(state.iter_trajectories())

    assert len(stored) == 2
    assert {identifier for identifier, _, _ in stored} == {
        anthropic_id,
        responses_id,
    }


def test_state_rejects_corrupt_stored_trajectory_id(tmp_path: Path) -> None:
    node = _trajectory()
    identifier = trajectory_id(node)

    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_trajectory(
            identifier,
            node,
            representative_key="source.json",
            source_ref="source.json",
            sha256="a" * 64,
            captured_at="",
            disposition="pass",
            reason_codes=[],
        )
        corrupt_id = "0" * 64
        state.connection.execute(
            "UPDATE trajectories SET trajectory_id=? WHERE trajectory_id=?",
            (corrupt_id, identifier),
        )

        with pytest.raises(ValueError, match="stored trajectory_id"):
            next(state.iter_trajectories())


def test_state_rejects_stale_derived_fields_during_restore(tmp_path: Path) -> None:
    node = _trajectory()
    identifier = trajectory_id(node)

    with StateStore(tmp_path / "state.sqlite") as state:
        state.put_trajectory(
            identifier,
            node,
            representative_key="source.json",
            source_ref="source.json",
            sha256="a" * 64,
            captured_at="",
            disposition="pass",
            reason_codes=[],
        )
        (compressed,) = state.connection.execute(
            "SELECT payload FROM trajectories WHERE trajectory_id=?", (identifier,)
        ).fetchone()
        payload = orjson.loads(zlib.decompress(compressed))
        payload["tool_call_check"]["total_calls"] = 777
        state.connection.execute(
            "UPDATE trajectories SET payload=? WHERE trajectory_id=?",
            (zlib.compress(orjson.dumps(payload)), identifier),
        )

        with pytest.raises(StaleDerivedFieldsError):
            next(state.iter_trajectories())
