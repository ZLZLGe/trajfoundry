"""Own-node token estimates and compatibility with older trajectory records."""

from copy import deepcopy
from pathlib import Path

import pytest

from trajfoundry.canonical import trajectory_id
from trajfoundry.export import OutputSet
from trajfoundry.models import (
    AgentMessageRecord,
    AuditIssue,
    AuditTag,
    CompactionRecord,
    FunctionCall,
    MediaMapping,
    Message,
    Metadata,
    NormalizationAudit,
    ServerToolCall,
    ToolCall,
    ToolDefinition,
    TrajectoryNode,
)
from trajfoundry.output_contract import (
    OutputContractError,
    parse_trajectory_record,
    project_trajectory,
)
from trajfoundry.quality import (
    _TOKEN_PATTERN,
    StaleDerivedFieldsError,
    enrich_trajectory,
    estimate_tokens,
    validate_derived_fields,
)


def _node(*, content: str = "Done!", **kwargs) -> TrajectoryNode:
    return TrajectoryNode(
        messages=[Message(role="assistant", content=content, reasoning_content="")],
        tools=[],
        source="sample.json",
        metadata=Metadata(source_file="sample.json"),
        **kwargs,
    )


def _tree() -> TrajectoryNode:
    parent = _node()
    parent.messages = [
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
        Message(role="tool", name="spawn_agent", tool_call_id="spawn-1", content="ok"),
        *parent.messages,
    ]
    parent.tools = [ToolDefinition(name="spawn_agent")]
    parent.sub_agent_trajectory = {"spawn-1": _node(content="Child has four words")}
    return enrich_trajectory(parent)


def test_total_tokens_counts_local_content_tools_and_instructions_once() -> None:
    node = _node(instructions="Follow rules.")
    node.tools = [
        ToolDefinition(
            name="read",
            description="Read text.",
            parameters={"type": "object", "properties": {}},
        )
    ]
    node.messages = [
        Message(role="system", content="System text."),
        Message(role="developer", content="Developer text."),
        Message(role="user", content="查 a."),
        Message(
            role="assistant",
            content="Working.",
            reasoning_content="Think deeply.",
            tool_calls=[
                ToolCall(
                    id="call-id-is-not-content",
                    function=FunctionCall(
                        name="read", arguments={"path": "a.txt", "limit": 2}
                    ),
                )
            ],
        ),
        Message(
            role="tool",
            name="read",
            tool_call_id="call-id-is-not-content",
            content="File says hello.",
        ),
        Message(role="assistant", content="Done!", reasoning_content=""),
    ]
    node.server_tool_calls = [
        ServerToolCall(
            name="web_search",
            id="server-id-is-not-content",
            origin="response",
            arguments={"query": "x"},
            result={"text": "found words"},
        )
    ]
    expected = sum(
        estimate_tokens(text)
        for text in (
            "Follow rules.",
            "read",
            "Read text.",
            '{"properties":{},"type":"object"}',
            "System text.",
            "Developer text.",
            "查 a.",
            "Working.",
            "Think deeply.",
            "read",
            '{"limit":2,"path":"a.txt"}',
            "File says hello.",
            "read",
            "Done!",
            "web_search",
            '{"query":"x"}',
            '{"text":"found words"}',
        )
    )

    enriched = enrich_trajectory(node)
    assert enriched.total_tokens == expected
    assert enriched.reasoning_total_tokens == 3
    assert enrich_trajectory(enriched) == enriched
    assert project_trajectory(enriched)["total_tokens"] == expected


def test_total_tokens_is_zero_for_empty_local_content() -> None:
    assert enrich_trajectory(_node(content="")).total_tokens == 0


@pytest.mark.parametrize(
    "text",
    [
        "",
        "ASCII words, punctuation?!_123",
        "中文测试。日本語 한글 العربية café 👩‍💻",
        "line one\nline two\twith spaces",
        "a 中文。 " * 100_000,
    ],
)
def test_incremental_estimator_matches_original_unicode_word_v1(text: str) -> None:
    assert estimate_tokens(text) == len(_TOKEN_PATTERN.findall(text))


@pytest.mark.parametrize("balanced", [False, True])
def test_deep_json_like_prose_does_not_trigger_media_decoding(balanced: bool) -> None:
    text = "[" * 2000 + '{"type":"object","data":"ordinary text"}'
    if balanced:
        text += "]" * 2000
    assert enrich_trajectory(_node(content=text)).total_tokens == estimate_tokens(text)


def test_total_tokens_excludes_mounted_children_but_exports_each_count() -> None:
    parent = _tree()
    assert parent.sub_agent_trajectory is not None
    original_count = parent.total_tokens
    child = parent.sub_agent_trajectory["spawn-1"]
    assert child.total_tokens == 4
    child.messages[0].content = " ".join(["child"] * 500)
    updated = enrich_trajectory(parent)
    assert updated.total_tokens == original_count
    assert updated.sub_agent_trajectory["spawn-1"].total_tokens == 500

    projected = project_trajectory(updated)
    mounted = projected["sub_agent_trajectory"]["spawn-1"]
    assert mounted["total_tokens"] == 500
    assert "completeness" not in mounted
    assert "completeness_tag" not in mounted


@pytest.mark.parametrize(
    "payload",
    [
        '{"type":"input_image","image_url":"data:image/png;base64,AA+/BB=="}',
        '{"source":{"data":"opaque bytes / + =","type":"base64"},"type":"image"}',
        '{"type":"input_audio","data":"opaque bytes / + ="}',
        '{"type":"file","file_data":"opaque bytes / + ="}',
        '{"type":"redacted_thinking","data":"opaque bytes / + ="}',
        '{"encrypted_content":"opaque bytes / + ="}',
        "data:image/png;base64,AA+/BB==",
        "<image>opaque bytes / + =</image>",
    ],
)
def test_total_tokens_excludes_embedded_media_and_opaque_payload(payload: str) -> None:
    node = _node(content="hello" + payload + " world")
    assert enrich_trajectory(node).total_tokens == 2


def test_total_tokens_preserves_normal_json_prose_and_long_text() -> None:
    text = 'Use {"type":"object","data":"ordinary text"} ' + "word " * 2000
    assert enrich_trajectory(_node(content=text)).total_tokens == estimate_tokens(text)


def test_total_tokens_excludes_raw_reasoning_sidecars_audits_and_media_mapping() -> (
    None
):
    node = _node()
    node.messages[0].reasoning_content = "Reason once."
    expected = enrich_trajectory(node).total_tokens
    node.messages[0].reasoning = {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Reason once."}],
        "encrypted_content": "opaque bytes / + =" * 200,
    }
    node.messages[0].reasoning_details = [
        {"type": "thinking", "thinking": "Reason once.", "signature": "opaque"}
    ]
    node.compaction_items = [
        CompactionRecord(
            origin="history",
            item_index=0,
            item={"type": "compaction", "encrypted_content": "opaque " * 200},
        )
    ]
    node.agent_messages = [
        AgentMessageRecord(
            origin="response",
            item_index=1,
            item={"type": "agent_message", "content": "sidecar " * 200},
        )
    ]
    node.multimodal_file_mapping = [
        MediaMapping(part_id="part_0", object_name="large-image.png")
    ]
    node.metadata.created_at = "2026-10-13T12:34:56Z"
    node.metadata.user_id = "this metadata must not count"
    node.normalization_audit = NormalizationAudit(
        tag=AuditTag.QUARANTINED,
        issues=[AuditIssue(code="capture_issue", stage="test", detail="audit " * 50)],
    )
    assert enrich_trajectory(node).total_tokens == expected


def test_total_tokens_filters_media_inside_arguments_and_server_results() -> None:
    node = _node()
    node.server_tool_calls = [
        ServerToolCall(
            name="search",
            id="id",
            origin="response",
            arguments={"query": "abc", "n": 2**80},
            result={"text": "visible"},
        )
    ]
    expected = enrich_trajectory(node).total_tokens
    node.server_tool_calls[0].arguments["image"] = {
        "type": "input_image",
        "image_url": "data:image/png;base64,AA+/BB==",
    }
    node.server_tool_calls[0].result["encrypted_content"] = "opaque bytes / + =" * 200
    assert enrich_trajectory(node).total_tokens == expected


def test_total_tokens_omits_hosted_image_generation_base64_result() -> None:
    node = _node()
    node.server_tool_calls = [
        ServerToolCall(
            name="image_generation",
            id="id",
            origin="response",
            arguments={"prompt": "draw a cat"},
        )
    ]
    expected = enrich_trajectory(node).total_tokens
    node.server_tool_calls[0].result = {"result": "base64 bytes / + =" * 200}
    assert enrich_trajectory(node).total_tokens == expected


def test_legacy_records_backfill_only_missing_total_tokens_without_mutation() -> None:
    parent = _tree()
    current = project_trajectory(parent)
    legacy = deepcopy(current)
    del legacy["total_tokens"]
    del legacy["sub_agent_trajectory"]["spawn-1"]["total_tokens"]
    untouched = deepcopy(legacy)

    parsed = parse_trajectory_record(legacy)
    assert parsed == parent
    assert project_trajectory(parsed) == current
    assert legacy == untouched
    validate_derived_fields(parsed)


@pytest.mark.parametrize(
    "missing", ["messages", "reasoning_total_tokens", "tool_counts"]
)
def test_legacy_total_tokens_backfill_keeps_existing_required_fields(
    missing: str,
) -> None:
    record = project_trajectory(enrich_trajectory(_node()))
    del record["total_tokens"]
    del record[missing]
    with pytest.raises(OutputContractError, match="missing required"):
        parse_trajectory_record(record)


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "10", None])
@pytest.mark.parametrize("nested", [False, True])
def test_total_tokens_rejects_invalid_published_values(invalid, nested: bool) -> None:
    record = project_trajectory(_tree())
    target = record["sub_agent_trajectory"]["spawn-1"] if nested else record
    target["total_tokens"] = invalid
    with pytest.raises(OutputContractError, match="total_tokens"):
        parse_trajectory_record(record)


def test_stale_total_tokens_are_rejected_without_changing_trajectory_identity() -> None:
    node = _tree()
    identifier = trajectory_id(node)
    node.total_tokens += 1
    node.sub_agent_trajectory["spawn-1"].total_tokens += 1
    assert trajectory_id(node) == identifier
    with pytest.raises(StaleDerivedFieldsError):
        validate_derived_fields(node)


@pytest.mark.parametrize("nested", [False, True])
def test_explicit_stale_total_tokens_are_preserved_then_rejected_on_export(
    tmp_path: Path, nested: bool
) -> None:
    record = project_trajectory(_tree())
    target = record["sub_agent_trajectory"]["spawn-1"] if nested else record
    target["total_tokens"] += 1
    untouched = deepcopy(record)
    parsed = parse_trajectory_record(record)
    assert project_trajectory(parsed) == untouched
    assert record == untouched
    with OutputSet(tmp_path) as output:
        with pytest.raises(StaleDerivedFieldsError):
            output.write_trajectory(parsed, [])
        assert output.stats.accepted == 0
