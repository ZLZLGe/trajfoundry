import pytest

from trajfoundry.audit_codes import (
    OPAQUE_COMPACTION_CONTEXT,
    RESPONSES_UNSUPPORTED_CALL_EVIDENCE,
)
from trajfoundry.canonical import trajectory_id
from trajfoundry.models import (
    AuditIssue,
    AuditTag,
    CompactionRecord,
    FunctionCall,
    Message,
    Metadata,
    NormalizationAudit,
    Severity,
    ToolCall,
    ToolDefinition,
    TrajectoryNode,
)
from trajfoundry.quality import enrich_trajectory, is_strict_sample


def _complete_leaf(source: str) -> TrajectoryNode:
    return TrajectoryNode(
        messages=[Message(role="assistant", content="done", reasoning_content="")],
        tools=[],
        source=source,
        metadata=Metadata(source_file=source),
    )


def _completed_tool_call_trajectory(
    *,
    arguments: object,
    tools: list[ToolDefinition],
    tool_name: str = "read",
) -> TrajectoryNode:
    return TrajectoryNode(
        messages=[
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=FunctionCall(name=tool_name, arguments=arguments),
                    )
                ],
            ),
            Message(
                role="tool",
                tool_call_id="call-1",
                name=tool_name,
                content="ok",
            ),
            Message(role="assistant", content="done", reasoning_content=""),
        ],
        tools=tools,
        source="root.json",
        metadata=Metadata(source_file="root.json"),
    )


def _spawn_parent(
    source: str,
    call_id: str,
    child: TrajectoryNode,
) -> TrajectoryNode:
    return TrajectoryNode(
        messages=[
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        function=FunctionCall(name="spawn_agent", arguments={}),
                    )
                ],
            ),
            Message(
                role="tool",
                tool_call_id=call_id,
                name="spawn_agent",
                content="done",
            ),
            Message(role="assistant", content="done", reasoning_content=""),
        ],
        tools=[ToolDefinition(name="spawn_agent")],
        source=source,
        metadata=Metadata(source_file=source),
        sub_agent_trajectory={call_id: child},
    )


def test_complete_tool_trajectory_passes_strict_gate() -> None:
    node = TrajectoryNode(
        messages=[
            Message(role="user", content="read it"),
            Message(
                role="assistant",
                content="",
                reasoning_content="inspect",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(name="read", arguments={"path": "x"}),
                    )
                ],
            ),
            Message(role="tool", tool_call_id="call_1", name="read", content="ok"),
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
        source="1.json",
        metadata=Metadata(source_file="1.json"),
    )
    enriched = enrich_trajectory(node)
    assert enriched.total_rounds == 2
    assert enriched.total_tool_calls == 1
    assert enriched.completeness_tag == "complete_main_no_sub"
    assert is_strict_sample(enriched)


def test_missing_tool_definition_gets_reason_without_schema_mismatch() -> None:
    node = _completed_tool_call_trajectory(
        arguments={"path": "x"},
        tools=[],
        tool_name="unknown_read",
    )

    enriched = enrich_trajectory(node)
    without_derived_audit = enriched.model_copy(
        update={"normalization_audit": NormalizationAudit(tag=AuditTag.PASS)}
    )

    assert trajectory_id(enriched) == trajectory_id(without_derived_audit)
    assert enriched.missing_tool_defs == ["unknown_read"]
    assert enriched.normalization_audit is not None
    assert enriched.normalization_audit.reason_codes == ["missing_tool_definitions"]
    assert "tool_call_schema_mismatch" not in enriched.normalization_audit.reason_codes
    assert enriched.tool_call_check.undefined_tool_calls == 1
    assert enriched.tool_call_check.extra_arg_calls == 0
    assert enriched.tool_call_check.missing_required_calls == 0
    assert enriched.tool_call_check.type_mismatch_calls == 0


@pytest.mark.parametrize(
    ("arguments", "parameters", "counter"),
    [
        (
            {"path": "x", "unexpected": True},
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "extra_arg_calls",
        ),
        (
            {},
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "missing_required_calls",
        ),
        (
            {"path": 1},
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "type_mismatch_calls",
        ),
    ],
)
def test_schema_validation_failures_get_one_summary_reason(
    arguments: object,
    parameters: dict[str, object],
    counter: str,
) -> None:
    node = _completed_tool_call_trajectory(
        arguments=arguments,
        tools=[ToolDefinition(name="read", parameters=parameters)],
    )

    enriched = enrich_trajectory(node)

    assert getattr(enriched.tool_call_check, counter) == 1
    assert enriched.normalization_audit is not None
    assert enriched.normalization_audit.reason_codes == ["tool_call_schema_mismatch"]
    assert (
        sum(
            issue.code == "tool_call_schema_mismatch"
            for issue in enriched.normalization_audit.issues
        )
        == 1
    )


def test_no_final_assistant_turn_gets_reason_code() -> None:
    node = TrajectoryNode(
        messages=[Message(role="user", content="continue")],
        tools=[],
        source="root.json",
        metadata=Metadata(source_file="root.json"),
    )

    enriched = enrich_trajectory(node)

    assert enriched.completeness is not None
    assert enriched.completeness.no_final_assistant_turn
    assert enriched.normalization_audit is not None
    assert enriched.normalization_audit.reason_codes == ["no_final_assistant_turn"]


def test_quality_reason_codes_coexist_and_are_idempotent() -> None:
    node = TrajectoryNode(
        messages=[
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=FunctionCall(name="unknown_read", arguments={}),
                    ),
                    ToolCall(
                        id="call-2",
                        function=FunctionCall(name="read", arguments={"path": 1}),
                    ),
                ],
            )
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
        source="root.json",
        metadata=Metadata(source_file="root.json"),
    )

    first = enrich_trajectory(node)
    second = enrich_trajectory(first)

    assert first.normalization_audit is not None
    assert first.normalization_audit.reason_codes == [
        "missing_tool_definitions",
        "missing_tool_result",
        "no_final_assistant_turn",
        "tool_call_schema_mismatch",
    ]
    assert second == first


def test_child_no_final_assistant_reason_stays_local() -> None:
    child = TrajectoryNode(
        messages=[Message(role="user", content="continue")],
        tools=[],
        source="child.json",
        metadata=Metadata(source_file="child.json"),
    )
    parent = _spawn_parent("root.json", "spawn-child", child)

    enriched = enrich_trajectory(parent)

    assert enriched.normalization_audit is not None
    assert enriched.normalization_audit.reason_codes == ["subagent_quality_failure"]
    assert enriched.sub_agent_trajectory is not None
    enriched_child = enriched.sub_agent_trajectory["spawn-child"]
    assert enriched_child.normalization_audit is not None
    assert enriched_child.normalization_audit.reason_codes == [
        "no_final_assistant_turn"
    ]


def test_opaque_compaction_context_quarantines_complete_trajectory() -> None:
    compaction_id = "cmp-must-not-leak"
    ciphertext = "ciphertext-must-not-leak"
    node = TrajectoryNode(
        messages=[Message(role="assistant", content="done", reasoning_content="")],
        tools=[],
        compaction_items=[
            CompactionRecord(
                origin="history",
                item_index=36,
                item={
                    "type": "compaction",
                    "id": compaction_id,
                    "encrypted_content": ciphertext,
                },
            )
        ],
        source="compacted.json",
        metadata=Metadata(source_file="compacted.json"),
    )

    enriched = enrich_trajectory(node)

    assert enriched.normalization_audit is not None
    assert enriched.normalization_audit.tag == AuditTag.QUARANTINED
    assert enriched.normalization_audit.reason_codes == [OPAQUE_COMPACTION_CONTEXT]
    issue = next(
        issue
        for issue in enriched.normalization_audit.issues
        if issue.code == OPAQUE_COMPACTION_CONTEXT
    )
    assert issue.severity == Severity.ERROR
    assert compaction_id not in issue.detail
    assert ciphertext not in issue.detail
    assert not is_strict_sample(enriched)


def test_compaction_quality_issue_is_idempotent() -> None:
    node = TrajectoryNode(
        messages=[Message(role="assistant", content="done", reasoning_content="")],
        tools=[],
        compaction_items=[
            CompactionRecord(
                origin="response",
                item_index=0,
                item={"type": "compaction", "encrypted_content": "opaque"},
            )
        ],
        source="compacted.json",
        metadata=Metadata(source_file="compacted.json"),
    )

    first = enrich_trajectory(node)
    second = enrich_trajectory(first)

    assert second == first
    assert second.normalization_audit is not None
    assert (
        sum(
            issue.code == OPAQUE_COMPACTION_CONTEXT
            for issue in second.normalization_audit.issues
        )
        == 1
    )


def test_child_compaction_quarantines_the_complete_parent_tree() -> None:
    child = _complete_leaf("child.json")
    child.compaction_items = [
        CompactionRecord(
            origin="history",
            item_index=2,
            item={"type": "compaction", "encrypted_content": "opaque"},
        )
    ]
    parent = _spawn_parent("parent.json", "spawn-child", child)

    enriched = enrich_trajectory(parent)

    assert enriched.normalization_audit is not None
    assert enriched.normalization_audit.tag == AuditTag.QUARANTINED
    assert "subagent_quality_failure" in enriched.normalization_audit.reason_codes
    assert enriched.sub_agent_trajectory is not None
    enriched_child = enriched.sub_agent_trajectory["spawn-child"]
    assert enriched_child.normalization_audit is not None
    assert OPAQUE_COMPACTION_CONTEXT in (
        enriched_child.normalization_audit.reason_codes
    )


def test_developer_is_not_equivalent_to_system() -> None:
    developer = Message(role="developer", content="rule")
    system = Message(role="system", content="rule")
    assert developer != system


def _codex_rejection_messages(
    *,
    call_id: str = "call-1",
    result_name: str = "ghost",
    content: str = "unsupported call: ghost",
) -> list[Message]:
    return [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            content="",
            reasoning_content="",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    function=FunctionCall(name="ghost", arguments={}),
                )
            ],
        ),
        Message(
            role="tool",
            content=content,
            tool_call_id=call_id,
            name=result_name,
        ),
        Message(role="assistant", content="done", reasoning_content=""),
    ]


def _responses_rejection_audit(
    *, call_id: str = "call-1", name: str = "ghost"
) -> NormalizationAudit:
    return NormalizationAudit(
        tag=AuditTag.PASS,
        issues=[
            AuditIssue(
                code=RESPONSES_UNSUPPORTED_CALL_EVIDENCE,
                stage="responses",
                severity=Severity.WARNING,
                path="request_body.input[2]",
                detail=(
                    f"provider rejected client tool call id={call_id!r} "
                    f"name={name!r} as unsupported"
                ),
            )
        ],
    )


def test_responses_evidence_is_hallucinated_and_independently_undefined() -> None:
    node = TrajectoryNode(
        messages=_codex_rejection_messages(),
        tools=[],
        harness="unknown",
        source="1.json",
        metadata=Metadata(source_file="1.json"),
        normalization_audit=_responses_rejection_audit(),
    )

    enriched = enrich_trajectory(node)

    assert enriched.tool_call_check.hallucinated_calls == 1
    assert enriched.tool_call_check.undefined_tool_calls == 1
    assert enriched.tool_call_check.mismatch_calls == 1
    assert enriched.normalization_audit is not None
    assert "hallucinated_tool_call" in enriched.normalization_audit.reason_codes


def test_responses_evidence_is_hallucinated_without_schema_mismatch() -> None:
    node = TrajectoryNode(
        messages=_codex_rejection_messages(
            content="unsupported custom tool call: ghost"
        ),
        tools=[ToolDefinition(name="ghost")],
        harness="unknown",
        source="1.json",
        metadata=Metadata(source_file="1.json"),
        normalization_audit=_responses_rejection_audit(),
    )

    enriched = enrich_trajectory(node)

    assert enriched.tool_call_check.hallucinated_calls == 1
    assert enriched.tool_call_check.undefined_tool_calls == 0
    assert enriched.tool_call_check.mismatch_calls == 0
    assert enriched.tool_call_tag == "consistent"
    assert not is_strict_sample(enriched)


def test_responses_hallucination_evidence_survives_repeated_enrichment() -> None:
    node = TrajectoryNode(
        messages=_codex_rejection_messages(),
        tools=[ToolDefinition(name="ghost")],
        source="1.json",
        metadata=Metadata(source_file="1.json"),
        normalization_audit=_responses_rejection_audit(),
    )

    first = enrich_trajectory(node)
    second = enrich_trajectory(first)

    assert first.tool_call_check.hallucinated_calls == 1
    assert second.tool_call_check.hallucinated_calls == 1
    assert second == first
    assert second.normalization_audit is not None
    assert (
        sum(
            issue.code == RESPONSES_UNSUPPORTED_CALL_EVIDENCE
            for issue in second.normalization_audit.issues
        )
        == 1
    )


def test_redacted_call_ids_remain_distinct_provider_evidence() -> None:
    call_ids = ("sk-abcdefgh1111", "sk-abcdefgh2222")
    messages = [Message(role="user", content="go")]
    evidence: list[AuditIssue] = []
    for index, call_id in enumerate(call_ids, start=1):
        messages.extend(
            [
                Message(
                    role="assistant",
                    content="",
                    reasoning_content="",
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            function=FunctionCall(name="ghost", arguments={}),
                        )
                    ],
                ),
                Message(
                    role="tool",
                    content="unsupported call: ghost",
                    tool_call_id=call_id,
                    name="ghost",
                ),
            ]
        )
        evidence.append(
            AuditIssue(
                code=RESPONSES_UNSUPPORTED_CALL_EVIDENCE,
                stage="responses",
                severity=Severity.WARNING,
                path=f"request_body.input[{index * 2}]",
                detail=(
                    f"provider rejected client tool call id={call_id!r} "
                    "name='ghost' as unsupported"
                ),
            )
        )
    messages.append(Message(role="assistant", content="done", reasoning_content=""))

    assert evidence[0].detail == evidence[1].detail
    one_evidence = enrich_trajectory(
        TrajectoryNode(
            messages=messages,
            tools=[ToolDefinition(name="ghost")],
            source="one.json",
            metadata=Metadata(source_file="one.json"),
            normalization_audit=NormalizationAudit(
                tag=AuditTag.PASS,
                issues=evidence[:1],
            ),
        )
    )
    two_evidence = enrich_trajectory(
        TrajectoryNode(
            messages=messages,
            tools=[ToolDefinition(name="ghost")],
            source="two.json",
            metadata=Metadata(source_file="two.json"),
            normalization_audit=NormalizationAudit(
                tag=AuditTag.PASS,
                issues=evidence,
            ),
        )
    )

    assert one_evidence.tool_call_check.hallucinated_calls == 1
    assert two_evidence.tool_call_check.hallucinated_calls == 2
    assert trajectory_id(one_evidence) != trajectory_id(two_evidence)


@pytest.mark.parametrize(
    ("harness", "call_id", "result_name", "content"),
    [
        ("unknown", "call-1", "ghost", "unsupported call: ghost"),
        ("codex", "call-1", "ghost", "unsupported call: ghost"),
        ("codex", "missing:client-call:1", "ghost", "unsupported call: ghost"),
        ("codex", "call-1", "other", "unsupported call: ghost"),
        ("codex", "call-1", "ghost", " unsupported call: ghost"),
        ("codex", "call-1", "ghost", "unsupported call: ghost\n"),
        ("codex", "call-1", "ghost", '{"error":"unsupported call: ghost"}'),
    ],
)
def test_unsupported_text_without_provider_evidence_is_not_hallucinated(
    harness: str,
    call_id: str,
    result_name: str,
    content: str,
) -> None:
    node = TrajectoryNode(
        messages=_codex_rejection_messages(
            call_id=call_id,
            result_name=result_name,
            content=content,
        ),
        tools=[ToolDefinition(name="ghost")],
        harness=harness,
        source="1.json",
        metadata=Metadata(source_file="1.json"),
    )

    assert enrich_trajectory(node).tool_call_check.hallucinated_calls == 0


def test_quality_never_infers_provider_evidence_from_message_shape() -> None:
    ordinary = _codex_rejection_messages()
    duplicate_call = ordinary[:2] + [ordinary[1], *ordinary[2:]]
    duplicate_result = ordinary[:3] + [ordinary[2], ordinary[3]]
    result_before_call = [ordinary[0], ordinary[2], ordinary[1], ordinary[3]]
    for messages in (duplicate_call, duplicate_result, result_before_call):
        node = TrajectoryNode(
            messages=messages,
            tools=[ToolDefinition(name="ghost")],
            harness="codex",
            source="1.json",
            metadata=Metadata(source_file="1.json"),
        )
        assert enrich_trajectory(node).tool_call_check.hallucinated_calls == 0


def test_missing_tool_result_is_quarantined() -> None:
    node = TrajectoryNode(
        messages=[
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(name="read", arguments={}),
                    )
                ],
            ),
        ],
        tools=[
            ToolDefinition(name="read", parameters={"type": "object", "properties": {}})
        ],
        source="1.json",
        metadata=Metadata(source_file="1.json"),
    )
    enriched = enrich_trajectory(node)
    assert enriched.normalization_audit is not None
    assert "missing_tool_result" in enriched.normalization_audit.reason_codes
    assert not is_strict_sample(enriched)


def test_tool_result_order_and_name_are_checked() -> None:
    node = TrajectoryNode(
        messages=[
            Message(role="tool", tool_call_id="call-1", name="wrong", content="x"),
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=FunctionCall(name="read", arguments={}),
                    )
                ],
            ),
        ],
        tools=[ToolDefinition(name="read")],
        source="1.json",
        metadata=Metadata(source_file="1.json"),
    )
    enriched = enrich_trajectory(node)
    assert enriched.normalization_audit is not None
    assert "tool_result_before_call" in enriched.normalization_audit.reason_codes
    assert "tool_result_name_mismatch" in enriched.normalization_audit.reason_codes


def test_child_audit_failure_is_not_a_mount_failure() -> None:
    child = TrajectoryNode(
        messages=[Message(role="assistant", content="done", reasoning_content="")],
        tools=[],
        source="child.json",
        metadata=Metadata(source_file="child.json"),
        normalization_audit=NormalizationAudit(
            tag=AuditTag.QUARANTINED,
            issues=[AuditIssue(code="child_error", stage="test")],
        ),
    )
    parent = TrajectoryNode(
        messages=[
            Message(role="user", content="go"),
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
                tool_call_id="spawn-1",
                name="spawn_agent",
                content="done",
            ),
            Message(role="assistant", content="done", reasoning_content=""),
        ],
        tools=[ToolDefinition(name="spawn_agent")],
        source="parent.json",
        metadata=Metadata(source_file="parent.json"),
        sub_agent_trajectory={"spawn-1": child},
    )
    enriched = enrich_trajectory(parent)
    assert enriched.completeness_tag == "complete_main_with_complete_sub"
    assert enriched.completeness is not None
    assert enriched.completeness.subtree_complete
    assert enriched.normalization_audit is not None
    assert "subagent_quality_failure" in enriched.normalization_audit.reason_codes
    assert "incomplete_subagent_mount" not in enriched.normalization_audit.reason_codes
    assert not is_strict_sample(enriched)


def test_top_level_tool_defs_recursively_aggregate_grandchild_failure() -> None:
    grandchild = TrajectoryNode(
        messages=[
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        function=FunctionCall(name="read", arguments={}),
                    )
                ],
            ),
            Message(role="tool", tool_call_id="read-1", name="read", content="ok"),
            Message(role="assistant", content="done", reasoning_content=""),
        ],
        tools=[],
        source="grandchild.json",
        metadata=Metadata(source_file="grandchild.json"),
    )
    child = _spawn_parent("child.json", "spawn-grandchild", grandchild)
    root = _spawn_parent("root.json", "spawn-child", child)

    enriched = enrich_trajectory(root)
    assert enriched.tool_defs_tag == "complete_with_incomplete_sub"
    assert enriched.sub_agent_trajectory is not None
    nested = enriched.sub_agent_trajectory["spawn-child"]
    assert nested.tool_defs_tag == "complete"
    assert nested.sub_agent_trajectory is not None
    assert nested.sub_agent_trajectory["spawn-grandchild"].tool_defs_tag == "incomplete"


def test_top_level_tool_calls_recursively_aggregate_grandchild_failure() -> None:
    grandchild = TrajectoryNode(
        messages=[
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        function=FunctionCall(name="read", arguments={}),
                    )
                ],
            ),
            Message(role="tool", tool_call_id="read-1", name="read", content="ok"),
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
        source="grandchild.json",
        metadata=Metadata(source_file="grandchild.json"),
    )
    child = _spawn_parent("child.json", "spawn-grandchild", grandchild)
    root = _spawn_parent("root.json", "spawn-child", child)

    enriched = enrich_trajectory(root)
    assert enriched.tool_call_tag == "consistent_with_inconsistent_sub"
    assert enriched.sub_agent_trajectory is not None
    nested = enriched.sub_agent_trajectory["spawn-child"]
    assert nested.tool_call_tag == "consistent"
    assert nested.sub_agent_trajectory is not None
    assert (
        nested.sub_agent_trajectory["spawn-grandchild"].tool_call_tag == "inconsistent"
    )


def test_unmounted_spawn_marks_subtree_incomplete() -> None:
    node = _spawn_parent("root.json", "spawn-1", _complete_leaf("child.json"))
    node.sub_agent_trajectory = None

    enriched = enrich_trajectory(node)
    assert enriched.completeness is not None
    assert not enriched.completeness.subtree_complete
    assert enriched.completeness_tag == "complete_main_with_incomplete_sub"
    assert enriched.normalization_audit is not None
    assert "incomplete_subagent_mount" in enriched.normalization_audit.reason_codes


def test_duplicate_spawn_ids_count_each_call_and_are_not_unique_mounts() -> None:
    call = ToolCall(
        id="spawn-1",
        function=FunctionCall(name="spawn_agent", arguments={}),
    )
    node = TrajectoryNode(
        messages=[
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[call, call.model_copy(deep=True)],
            ),
            Message(
                role="tool",
                tool_call_id="spawn-1",
                name="spawn_agent",
                content="done",
            ),
            Message(role="assistant", content="done", reasoning_content=""),
        ],
        tools=[ToolDefinition(name="spawn_agent")],
        source="root.json",
        metadata=Metadata(source_file="root.json"),
        sub_agent_trajectory={"spawn-1": _complete_leaf("child.json")},
    )

    enriched = enrich_trajectory(node)
    assert enriched.completeness is not None
    assert enriched.completeness.spawn_calls == 2
    assert enriched.completeness.mounted_subs == 1
    assert not enriched.completeness.subtree_complete


def test_nested_unkeyed_mounts_are_summed_at_top_level() -> None:
    child = _complete_leaf("child.json")
    child.sub_agent_trajectory = {"not-a-spawn": _complete_leaf("grandchild.json")}
    root = _spawn_parent("root.json", "spawn-child", child)

    enriched = enrich_trajectory(root)
    assert enriched.completeness is not None
    assert enriched.completeness.unkeyed_mounts == 1
    assert not enriched.completeness.subtree_complete


def test_missing_result_with_later_assistant_is_not_trailing() -> None:
    node = TrajectoryNode(
        messages=[
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        function=FunctionCall(name="read", arguments={}),
                    )
                ],
            ),
            Message(role="assistant", content="continued", reasoning_content=""),
        ],
        tools=[ToolDefinition(name="read")],
        source="root.json",
        metadata=Metadata(source_file="root.json"),
    )

    enriched = enrich_trajectory(node)
    assert enriched.completeness is not None
    assert not enriched.completeness.trailing_unanswered_call
    assert enriched.normalization_audit is not None
    assert "missing_tool_result" in enriched.normalization_audit.reason_codes


def test_missing_result_without_later_assistant_is_trailing() -> None:
    node = TrajectoryNode(
        messages=[
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        function=FunctionCall(name="read", arguments={}),
                    )
                ],
            )
        ],
        tools=[ToolDefinition(name="read")],
        source="root.json",
        metadata=Metadata(source_file="root.json"),
    )

    enriched = enrich_trajectory(node)
    assert enriched.completeness is not None
    assert enriched.completeness.trailing_unanswered_call
    assert enriched.completeness.no_final_assistant_turn


def test_enrichment_discards_stale_derived_issues_after_message_repair() -> None:
    node = TrajectoryNode(
        messages=[
            Message(
                role="assistant",
                content="",
                reasoning_content="",
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        function=FunctionCall(name="read", arguments={}),
                    )
                ],
            )
        ],
        tools=[ToolDefinition(name="read")],
        source="root.json",
        metadata=Metadata(source_file="root.json"),
    )
    first = enrich_trajectory(node)
    assert first.normalization_audit is not None
    assert "missing_tool_result" in first.normalization_audit.reason_codes
    assert "no_final_assistant_turn" in first.normalization_audit.reason_codes

    repaired = first.model_copy(deep=True)
    repaired.messages.extend(
        [
            Message(
                role="tool",
                content="ok",
                tool_call_id="read-1",
                name="read",
            ),
            Message(role="assistant", content="done", reasoning_content=""),
        ]
    )
    second = enrich_trajectory(repaired)

    assert second.normalization_audit is not None
    assert second.normalization_audit.tag == AuditTag.PASS
    assert "missing_tool_result" not in second.normalization_audit.reason_codes
    assert "no_final_assistant_turn" not in second.normalization_audit.reason_codes
    assert second == enrich_trajectory(second)


def test_enrichment_preserves_primary_provider_evidence() -> None:
    node = _complete_leaf("root.json")
    node.normalization_audit = NormalizationAudit(
        tag=AuditTag.QUARANTINED,
        reason_codes=["provider_failure"],
        issues=[
            AuditIssue(
                code="provider_failure",
                stage="provider.responses",
                detail="primary evidence",
            )
        ],
    )

    enriched = enrich_trajectory(node)

    assert enriched.normalization_audit is not None
    assert enriched.normalization_audit.reason_codes == ["provider_failure"]
    assert enriched.normalization_audit.issues[0].stage == "provider.responses"


def test_stale_aggregate_mount_issue_cannot_create_an_impossible_subtree() -> None:
    node = _complete_leaf("root.json")
    node.normalization_audit = NormalizationAudit(
        tag=AuditTag.QUARANTINED,
        reason_codes=["incomplete_subagent_mount"],
        issues=[
            AuditIssue(
                code="incomplete_subagent_mount",
                stage="subagents",
                detail="stale derived issue",
            )
        ],
    )

    enriched = enrich_trajectory(node)

    assert enriched.completeness is not None
    assert enriched.completeness.spawn_calls == 0
    assert enriched.completeness.mounted_subs == 0
    assert enriched.completeness.subtree_complete
    assert enriched.completeness_tag == "complete_main_no_sub"
    assert enriched.normalization_audit is not None
    assert enriched.normalization_audit.tag == AuditTag.PASS


def test_mount_only_grandchild_failure_is_not_promoted_to_child_quality() -> None:
    grandchild = _complete_leaf("grandchild.json")
    grandchild.normalization_audit = NormalizationAudit(
        tag=AuditTag.QUARANTINED,
        reason_codes=["unmounted_spawn_call"],
        issues=[
            AuditIssue(
                code="unmounted_spawn_call",
                stage="subagents",
                detail="primary graph evidence",
            )
        ],
    )
    child = _spawn_parent("child.json", "spawn-grandchild", grandchild)
    root = _spawn_parent("root.json", "spawn-child", child)

    enriched = enrich_trajectory(root)

    assert enriched.completeness is not None
    assert not enriched.completeness.subtree_complete
    assert enriched.normalization_audit is not None
    assert "incomplete_subagent_mount" in enriched.normalization_audit.reason_codes
    assert "subagent_quality_failure" not in enriched.normalization_audit.reason_codes
