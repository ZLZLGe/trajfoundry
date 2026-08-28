"""Single projection and validation boundary for published trajectory records."""

from __future__ import annotations

import math
from pathlib import PurePath
from typing import Any, NoReturn

import orjson
from pydantic import BaseModel, ValidationError

from .audit_codes import PRIMARY_MOUNT_DIAGNOSTIC_CODES
from .models import (
    AuditIssue,
    CompactionRecord,
    Completeness,
    Message,
    NormalizationAudit,
    QuarantineRecord,
    ToolCallCheck,
    ToolCallMismatch,
    TrajectoryNode,
    TypeMismatch,
)
from .tool_names import is_spawn_tool_name


class OutputContractError(ValueError):
    """Raised when a JSON value does not satisfy the published contract."""


def _fail(path: str, detail: str) -> NoReturn:
    raise OutputContractError(f"{path}: {detail}")


def _revalidate_source_model(
    value: object, expected_type: type[BaseModel], path: str
) -> BaseModel:
    """Re-run validation before projecting a mutable model instance.

    Pydantic models are mutable by default and ``model_copy(update=...)`` does not
    validate updates. Revalidation prevents the projector from silently dropping
    role- or scope-invalid fields introduced after construction. JSON validation
    still runs on the projected Python object before a serializer can coerce it.
    """

    if not isinstance(value, expected_type):
        _fail(path, f"must be a {expected_type.__name__}")
    try:
        dumped = value.model_dump(mode="python", round_trip=True, warnings=False)
        return expected_type.model_validate(dumped, strict=True)
    except (TypeError, ValueError, ValidationError) as error:
        raise OutputContractError(f"{path}: source model is invalid") from error


def _object(value: object, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(path, "must be an object")
    if any(type(key) is not str for key in value):
        _fail(path, "object keys must be strings")
    return value


def _array(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        _fail(path, "must be an array")
    return value


def _string(value: object, path: str, *, nonempty: bool = False) -> str:
    if type(value) is not str:
        _fail(path, "must be a string")
    if nonempty and not value:
        _fail(path, "must not be empty")
    return value


def _integer(value: object, path: str, *, nonnegative: bool = True) -> int:
    if type(value) is not int:
        _fail(path, "must be an integer")
    if nonnegative and value < 0:
        _fail(path, "must be non-negative")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be a boolean")
    return value


def _keys(
    value: dict[str, Any],
    path: str,
    *,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    missing = required - value.keys()
    if missing:
        _fail(path, f"missing required field(s): {', '.join(sorted(missing))}")
    extra = value.keys() - required - optional
    if extra:
        _fail(path, f"contains unsupported field(s): {', '.join(sorted(extra))}")


def _json_value(value: object, path: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(path, "must be a finite JSON number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _json_value(item, f"{path}/{index}")
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _fail(path, "object keys must be strings")
        for key, item in value.items():
            _json_value(item, f"{path}/{key}")
        return
    _fail(path, "must be a JSON value")


def _string_array(value: object, path: str) -> list[str]:
    items = _array(value, path)
    for index, item in enumerate(items):
        _string(item, f"{path}/{index}")
    return items


def _validate_function_call(value: object, path: str) -> None:
    item = _object(value, path)
    _keys(item, path, required={"name", "arguments"})
    _string(item["name"], f"{path}/name")
    _json_value(item["arguments"], f"{path}/arguments")


def _validate_tool_call(value: object, path: str) -> None:
    item = _object(value, path)
    _keys(item, path, required={"type", "id", "function"})
    if item["type"] != "function":
        _fail(f"{path}/type", "must equal 'function'")
    _string(item["id"], f"{path}/id")
    _validate_function_call(item["function"], f"{path}/function")


def _validate_message(value: object, path: str) -> None:
    message = _object(value, path)
    _keys(
        message,
        path,
        required={"role", "content"},
        optional={
            "reasoning_content",
            "reasoning_details",
            "reasoning",
            "tool_calls",
            "tool_call_id",
            "name",
        },
    )
    role = _string(message["role"], f"{path}/role")
    _string(message["content"], f"{path}/content")
    if role in {"system", "developer", "user"}:
        _keys(message, path, required={"role", "content"})
        return
    if role == "tool":
        _keys(
            message,
            path,
            required={"role", "content", "tool_call_id", "name"},
        )
        _string(message["tool_call_id"], f"{path}/tool_call_id")
        _string(message["name"], f"{path}/name")
        return
    if role != "assistant":
        _fail(f"{path}/role", "has an unsupported value")
    _keys(
        message,
        path,
        required={"role", "content", "reasoning_content"},
        optional={"reasoning_details", "reasoning", "tool_calls"},
    )
    _string(message["reasoning_content"], f"{path}/reasoning_content")
    if "reasoning_details" in message:
        details = _array(message["reasoning_details"], f"{path}/reasoning_details")
        if not details:
            _fail(f"{path}/reasoning_details", "must be omitted when empty")
        for index, detail in enumerate(details):
            _object(detail, f"{path}/reasoning_details/{index}")
            _json_value(detail, f"{path}/reasoning_details/{index}")
    if "reasoning" in message:
        if message["reasoning"] is None:
            _fail(f"{path}/reasoning", "must be omitted when null")
        _json_value(message["reasoning"], f"{path}/reasoning")
    if "tool_calls" in message:
        calls = _array(message["tool_calls"], f"{path}/tool_calls")
        if not calls:
            _fail(f"{path}/tool_calls", "must contain at least one item")
        for index, call in enumerate(calls):
            _validate_tool_call(call, f"{path}/tool_calls/{index}")


def _validate_tool_definition(value: object, path: str) -> None:
    definition = _object(value, path)
    _keys(
        definition,
        path,
        required={"type", "name", "description", "parameters"},
    )
    if definition["type"] != "function":
        _fail(f"{path}/type", "must equal 'function'")
    _string(definition["name"], f"{path}/name")
    _string(definition["description"], f"{path}/description")
    parameters = _object(definition["parameters"], f"{path}/parameters")
    _json_value(parameters, f"{path}/parameters")


def _validate_type_mismatch(value: object, path: str) -> None:
    mismatch = _object(value, path)
    _keys(mismatch, path, required={"arg", "declared", "actual"})
    _string(mismatch["arg"], f"{path}/arg")
    _string_array(mismatch["declared"], f"{path}/declared")
    _string(mismatch["actual"], f"{path}/actual")


def _validate_mismatch(value: object, path: str) -> None:
    mismatch = _object(value, path)
    _keys(
        mismatch,
        path,
        required={"tool", "tool_call_id", "message_index", "reasons"},
        optional={"extra_args", "missing_required", "type_mismatch"},
    )
    _string(mismatch["tool"], f"{path}/tool")
    _string(mismatch["tool_call_id"], f"{path}/tool_call_id")
    _integer(mismatch["message_index"], f"{path}/message_index")
    reasons = _string_array(mismatch["reasons"], f"{path}/reasons")
    allowed_reasons = {
        "undefined_tool",
        "extra_args",
        "missing_required",
        "type_mismatch",
    }
    if any(reason not in allowed_reasons for reason in reasons):
        _fail(f"{path}/reasons", "contains an unsupported value")
    for field in ("extra_args", "missing_required"):
        if field in mismatch:
            items = _string_array(mismatch[field], f"{path}/{field}")
            if not items:
                _fail(f"{path}/{field}", "must be omitted when empty")
    if "type_mismatch" in mismatch:
        items = _array(mismatch["type_mismatch"], f"{path}/type_mismatch")
        if not items:
            _fail(f"{path}/type_mismatch", "must be omitted when empty")
        for index, item in enumerate(items):
            _validate_type_mismatch(item, f"{path}/type_mismatch/{index}")


_CHECK_FIELDS = {
    "total_calls",
    "hallucinated_calls",
    "undefined_tool_calls",
    "unverifiable_calls",
    "checked_calls",
    "extra_arg_calls",
    "missing_required_calls",
    "type_mismatch_calls",
    "defaulted_missing_calls",
    "mismatch_calls",
    "mismatches",
    "mismatches_truncated",
}


def _validate_tool_call_check(value: object, path: str) -> None:
    check = _object(value, path)
    _keys(check, path, required=_CHECK_FIELDS)
    for field in _CHECK_FIELDS - {"mismatches"}:
        _integer(check[field], f"{path}/{field}")
    mismatches = _array(check["mismatches"], f"{path}/mismatches")
    for index, mismatch in enumerate(mismatches):
        _validate_mismatch(mismatch, f"{path}/mismatches/{index}")


def _validate_server_call(value: object, path: str) -> None:
    call = _object(value, path)
    _keys(
        call,
        path,
        required={"name", "id", "arguments", "origin", "result"},
    )
    _string(call["name"], f"{path}/name")
    _string(call["id"], f"{path}/id")
    _json_value(call["arguments"], f"{path}/arguments")
    if call["origin"] not in {"history", "response"}:
        _fail(f"{path}/origin", "has an unsupported value")
    if call["result"] is not None:
        result = _object(call["result"], f"{path}/result")
        _json_value(result, f"{path}/result")


def _validate_agent_message(value: object, path: str) -> None:
    record = _object(value, path)
    _keys(record, path, required={"origin", "item_index", "item"})
    if record["origin"] not in {"history", "response"}:
        _fail(f"{path}/origin", "has an unsupported value")
    _integer(record["item_index"], f"{path}/item_index")
    item = _object(record["item"], f"{path}/item")
    _json_value(item, f"{path}/item")
    if item.get("type") != "agent_message":
        _fail(f"{path}/item/type", "must equal 'agent_message'")


def _validate_compaction(value: object, path: str) -> None:
    record = _object(value, path)
    _keys(record, path, required={"origin", "item_index", "item"})
    if record["origin"] not in {"history", "response"}:
        _fail(f"{path}/origin", "has an unsupported value")
    _integer(record["item_index"], f"{path}/item_index")
    item = _object(record["item"], f"{path}/item")
    _json_value(item, f"{path}/item")
    if item.get("type") != "compaction":
        _fail(f"{path}/item/type", "must equal 'compaction'")


def _validate_issue(value: object, path: str) -> None:
    issue = _object(value, path)
    _keys(
        issue,
        path,
        required={"code", "stage", "severity", "path", "detail"},
    )
    _string(issue["code"], f"{path}/code")
    _string(issue["stage"], f"{path}/stage")
    if issue["severity"] not in {"warning", "error"}:
        _fail(f"{path}/severity", "has an unsupported value")
    _string(issue["path"], f"{path}/path")
    _string(issue["detail"], f"{path}/detail")


def _validate_audit(value: object, path: str) -> None:
    audit = _object(value, path)
    _keys(audit, path, required={"tag", "reason_codes", "issues"})
    if audit["tag"] not in {"pass", "quarantined", "excluded"}:
        _fail(f"{path}/tag", "has an unsupported value")
    _string_array(audit["reason_codes"], f"{path}/reason_codes")
    issues = _array(audit["issues"], f"{path}/issues")
    for index, issue in enumerate(issues):
        _validate_issue(issue, f"{path}/issues/{index}")


_COMPLETENESS_FIELDS = {
    "is_subagent",
    "spawn_calls",
    "mounted_subs",
    "subtree_complete",
    "unkeyed_mounts",
    "relay_mounts",
    "trailing_unanswered_call",
    "no_final_assistant_turn",
}


def _validate_completeness(value: object, path: str) -> dict[str, Any]:
    completeness = _object(value, path)
    _keys(completeness, path, required=_COMPLETENESS_FIELDS)
    for field in (
        "is_subagent",
        "subtree_complete",
        "trailing_unanswered_call",
        "no_final_assistant_turn",
    ):
        _boolean(completeness[field], f"{path}/{field}")
    for field in ("spawn_calls", "mounted_subs", "unkeyed_mounts", "relay_mounts"):
        _integer(completeness[field], f"{path}/{field}")
    return completeness


_NODE_REQUIRED = {
    "messages",
    "tools",
    "instructions",
    "termination",
    "harness",
    "model",
    "source",
    "total_rounds",
    "total_tool_calls",
    "tool_counts",
    "reasoning_total_tokens",
    "tool_defs_tag",
    "missing_tool_defs",
    "tool_call_tag",
    "tool_call_check",
    "server_tool_calls",
    "agent_messages",
    "compaction_items",
    "metadata",
    "normalization_audit",
}
_NODE_OPTIONAL = {
    "sub_agent_trajectory",
    "sub_agent_relay_mounts",
    "completeness_tag",
    "completeness",
}
_COMPLETENESS_TAGS = {
    "orphan_sub",
    "complete_main_no_sub",
    "incomplete_main_no_sub",
    "complete_main_with_complete_sub",
    "complete_main_with_incomplete_sub",
    "incomplete_main_with_complete_sub",
    "incomplete_main_with_incomplete_sub",
}


def _message_completeness(
    messages: list[Any],
) -> tuple[bool, bool, bool, list[str]]:
    """Return local completion flags plus spawn call ids."""

    calls: dict[str, tuple[str, int]] = {}
    results: dict[str, list[tuple[int, str]]] = {}
    spawn_ids: list[str] = []
    pairing_clean = True
    missing_indices: list[int] = []
    for message_index, message in enumerate(messages):
        for call in message.get("tool_calls", []):
            call_id = call["id"]
            call_name = call["function"]["name"]
            if is_spawn_tool_name(call_name):
                spawn_ids.append(call_id)
            if call_id in calls:
                pairing_clean = False
            else:
                calls[call_id] = (call_name, message_index)
        if message["role"] == "tool":
            results.setdefault(message["tool_call_id"], []).append(
                (message_index, message["name"])
            )

    for call_id, (call_name, call_index) in calls.items():
        matched = results.get(call_id, [])
        if not matched:
            pairing_clean = False
            missing_indices.append(call_index)
        elif len(matched) > 1:
            pairing_clean = False
        if any(
            result_index < call_index or result_name != call_name
            for result_index, result_name in matched
        ):
            pairing_clean = False
    if any(call_id not in calls for call_id in results):
        pairing_clean = False

    trailing_unanswered = any(
        not any(
            message["role"] == "assistant" for message in messages[call_index + 1 :]
        )
        for call_index in missing_indices
    )
    no_final_assistant = (
        not messages
        or messages[-1]["role"] != "assistant"
        or bool(messages[-1].get("tool_calls"))
    )
    return (
        pairing_clean and not no_final_assistant,
        trailing_unanswered,
        no_final_assistant,
        spawn_ids,
    )


def _has_primary_mount_failure(audit: dict[str, Any]) -> bool:
    return any(
        issue["severity"] == "error" and issue["code"] in PRIMARY_MOUNT_DIAGNOSTIC_CODES
        for issue in audit["issues"]
    )


def _validate_node(value: object, path: str, *, top_level: bool) -> tuple[bool, int]:
    node = _object(value, path)
    required = _NODE_REQUIRED | (
        {"completeness_tag", "completeness"} if top_level else set()
    )
    optional = _NODE_OPTIONAL - (
        {"completeness_tag", "completeness"} if top_level else set()
    )
    _keys(node, path, required=required, optional=optional)
    if not top_level and ({"completeness_tag", "completeness"} & node.keys()):
        _fail(path, "nested nodes must omit completeness fields")

    messages = _array(node["messages"], f"{path}/messages")
    for index, message in enumerate(messages):
        _validate_message(message, f"{path}/messages/{index}")
    tools = _array(node["tools"], f"{path}/tools")
    for index, definition in enumerate(tools):
        _validate_tool_definition(definition, f"{path}/tools/{index}")
    for field in ("instructions", "termination", "harness", "model"):
        _string(node[field], f"{path}/{field}")
    source = _string(node["source"], f"{path}/source", nonempty=True)
    if PurePath(source).name != source or "/" in source or "\\" in source:
        _fail(f"{path}/source", "must be a non-empty basename")
    for field in ("total_rounds", "total_tool_calls", "reasoning_total_tokens"):
        _integer(node[field], f"{path}/{field}")
    tool_counts = _object(node["tool_counts"], f"{path}/tool_counts")
    for name, count in tool_counts.items():
        _string(name, f"{path}/tool_counts key")
        _integer(count, f"{path}/tool_counts/{name}")
    if node["tool_defs_tag"] not in (
        {"complete", "incomplete", "complete_with_incomplete_sub"}
        if top_level
        else {"complete", "incomplete"}
    ):
        _fail(f"{path}/tool_defs_tag", "has an invalid scope-specific value")
    _string_array(node["missing_tool_defs"], f"{path}/missing_tool_defs")
    if node["tool_call_tag"] not in (
        {"consistent", "inconsistent", "consistent_with_inconsistent_sub"}
        if top_level
        else {"consistent", "inconsistent"}
    ):
        _fail(f"{path}/tool_call_tag", "has an invalid scope-specific value")
    _validate_tool_call_check(node["tool_call_check"], f"{path}/tool_call_check")
    server_calls = _array(node["server_tool_calls"], f"{path}/server_tool_calls")
    for index, call in enumerate(server_calls):
        _validate_server_call(call, f"{path}/server_tool_calls/{index}")
    agent_messages = _array(node["agent_messages"], f"{path}/agent_messages")
    for index, message in enumerate(agent_messages):
        _validate_agent_message(message, f"{path}/agent_messages/{index}")
    compaction_items = _array(node["compaction_items"], f"{path}/compaction_items")
    for index, compaction in enumerate(compaction_items):
        _validate_compaction(compaction, f"{path}/compaction_items/{index}")

    metadata = _object(node["metadata"], f"{path}/metadata")
    _keys(
        metadata,
        f"{path}/metadata",
        required={"source_file", "source_name", "line_no", "created_at"},
    )
    source_file = _string(
        metadata["source_file"], f"{path}/metadata/source_file", nonempty=True
    )
    if (
        source_file != source
        or PurePath(source_file).name != source_file
        or "/" in source_file
        or "\\" in source_file
    ):
        _fail(f"{path}/metadata/source_file", "must equal source and be a basename")
    if metadata["source_name"] != "freerouter":
        _fail(f"{path}/metadata/source_name", "must equal 'freerouter'")
    _integer(metadata["line_no"], f"{path}/metadata/line_no")
    _string(metadata["created_at"], f"{path}/metadata/created_at")
    _validate_audit(node["normalization_audit"], f"{path}/normalization_audit")

    children: dict[str, Any] = {}
    child_mount_states: list[tuple[bool, int]] = []
    if "sub_agent_trajectory" in node:
        children = _object(node["sub_agent_trajectory"], f"{path}/sub_agent_trajectory")
        if not children:
            _fail(f"{path}/sub_agent_trajectory", "must be omitted when empty")
        for call_id, child in children.items():
            _string(call_id, f"{path}/sub_agent_trajectory key", nonempty=True)
            child_mount_states.append(
                _validate_node(
                    child,
                    f"{path}/sub_agent_trajectory/{call_id}",
                    top_level=False,
                )
            )

    relay_mounts: dict[str, Any] = {}
    if "sub_agent_relay_mounts" in node:
        relay_mounts = _object(
            node["sub_agent_relay_mounts"], f"{path}/sub_agent_relay_mounts"
        )
        if not relay_mounts:
            _fail(f"{path}/sub_agent_relay_mounts", "must be omitted when empty")
        relay_ids: set[str] = set()
        for call_id, relay_id_value in relay_mounts.items():
            _string(call_id, f"{path}/sub_agent_relay_mounts key", nonempty=True)
            relay_id = _string(
                relay_id_value,
                f"{path}/sub_agent_relay_mounts/{call_id}",
                nonempty=True,
            )
            if call_id not in children:
                _fail(
                    f"{path}/sub_agent_relay_mounts/{call_id}",
                    "key must identify a mounted child",
                )
            if relay_id == call_id:
                _fail(
                    f"{path}/sub_agent_relay_mounts/{call_id}",
                    "relay id must differ from spawn id",
                )
            if relay_id in relay_ids:
                _fail(
                    f"{path}/sub_agent_relay_mounts/{call_id}",
                    "relay id must be unique",
                )
            relay_ids.add(relay_id)

    (
        local_complete,
        trailing_unanswered,
        no_final_assistant,
        spawn_ids,
    ) = _message_completeness(messages)
    spawn_id_set = set(spawn_ids)
    child_id_set = set(children)
    for call_id, relay_id_value in relay_mounts.items():
        relay_id = str(relay_id_value)
        if call_id not in spawn_id_set:
            _fail(
                f"{path}/sub_agent_relay_mounts/{call_id}",
                "key must identify a spawn call",
            )
        matching_agent_messages = [
            message
            for message in agent_messages
            if message["item"].get("id") == relay_id
        ]
        if len(matching_agent_messages) != 1:
            _fail(
                f"{path}/sub_agent_relay_mounts/{call_id}",
                "relay id must identify exactly one local agent_message",
            )
    direct_mount_complete = (
        len(spawn_ids) == len(spawn_id_set)
        and child_id_set == spawn_id_set
        and not _has_primary_mount_failure(node["normalization_audit"])
    )
    subtree_complete = direct_mount_complete and all(
        complete for complete, _ in child_mount_states
    )
    unkeyed_mounts = len(child_id_set - spawn_id_set) + sum(
        count for _, count in child_mount_states
    )

    if top_level:
        tag = node["completeness_tag"]
        if tag not in _COMPLETENESS_TAGS:
            _fail(f"{path}/completeness_tag", "has an unsupported value")
        completeness = _validate_completeness(
            node["completeness"], f"{path}/completeness"
        )
        is_subagent = completeness["is_subagent"]
        if is_subagent != (tag == "orphan_sub"):
            _fail(
                f"{path}/completeness/is_subagent",
                "must agree with completeness_tag",
            )
        expected_subtree_complete = False if is_subagent else subtree_complete
        expected_counts = {
            "spawn_calls": len(spawn_ids),
            "mounted_subs": len(children),
            "unkeyed_mounts": unkeyed_mounts,
            "relay_mounts": len(relay_mounts),
        }
        for field, expected in expected_counts.items():
            if completeness[field] != expected:
                _fail(
                    f"{path}/completeness/{field}",
                    f"must equal recomputed value {expected}",
                )
        expected_flags = {
            "subtree_complete": expected_subtree_complete,
            "trailing_unanswered_call": trailing_unanswered,
            "no_final_assistant_turn": no_final_assistant,
        }
        for field, expected in expected_flags.items():
            if completeness[field] is not expected:
                _fail(
                    f"{path}/completeness/{field}",
                    f"must equal recomputed value {expected}",
                )

        if is_subagent:
            expected_tag = "orphan_sub"
        else:
            has_sub = bool(
                spawn_ids
                or children
                or _has_primary_mount_failure(node["normalization_audit"])
            )
            if local_complete and not has_sub:
                expected_tag = "complete_main_no_sub"
            elif not local_complete and not has_sub:
                expected_tag = "incomplete_main_no_sub"
            elif local_complete and subtree_complete:
                expected_tag = "complete_main_with_complete_sub"
            elif local_complete:
                expected_tag = "complete_main_with_incomplete_sub"
            elif subtree_complete:
                expected_tag = "incomplete_main_with_complete_sub"
            else:
                expected_tag = "incomplete_main_with_incomplete_sub"
        if tag != expected_tag:
            _fail(
                f"{path}/completeness_tag",
                f"must equal recomputed value {expected_tag!r}",
            )

    return subtree_complete, unkeyed_mounts


def _project_type_mismatch(value: TypeMismatch) -> dict[str, Any]:
    return {
        "arg": value.arg,
        "declared": list(value.declared),
        "actual": value.actual,
    }


def _project_mismatch(value: ToolCallMismatch) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tool": value.tool,
        "tool_call_id": value.tool_call_id,
        "message_index": value.message_index,
        "reasons": list(value.reasons),
    }
    if value.extra_args:
        result["extra_args"] = list(value.extra_args)
    if value.missing_required:
        result["missing_required"] = list(value.missing_required)
    if value.type_mismatch:
        result["type_mismatch"] = [
            _project_type_mismatch(item) for item in value.type_mismatch
        ]
    return result


def _project_check(value: ToolCallCheck) -> dict[str, Any]:
    return {
        "total_calls": value.total_calls,
        "hallucinated_calls": value.hallucinated_calls,
        "undefined_tool_calls": value.undefined_tool_calls,
        "unverifiable_calls": value.unverifiable_calls,
        "checked_calls": value.checked_calls,
        "extra_arg_calls": value.extra_arg_calls,
        "missing_required_calls": value.missing_required_calls,
        "type_mismatch_calls": value.type_mismatch_calls,
        "defaulted_missing_calls": value.defaulted_missing_calls,
        "mismatch_calls": value.mismatch_calls,
        "mismatches": [_project_mismatch(item) for item in value.mismatches],
        "mismatches_truncated": value.mismatches_truncated,
    }


def _project_issue(value: AuditIssue) -> dict[str, Any]:
    return {
        "code": value.code,
        "stage": value.stage,
        "severity": value.severity.value,
        "path": value.path,
        "detail": value.detail,
    }


def _project_audit(value: NormalizationAudit) -> dict[str, Any]:
    return {
        "tag": value.tag.value,
        "reason_codes": list(value.reason_codes),
        "issues": [_project_issue(issue) for issue in value.issues],
    }


def _project_completeness(value: Completeness) -> dict[str, Any]:
    return {
        "is_subagent": value.is_subagent,
        "spawn_calls": value.spawn_calls,
        "mounted_subs": value.mounted_subs,
        "subtree_complete": value.subtree_complete,
        "unkeyed_mounts": value.unkeyed_mounts,
        "relay_mounts": value.relay_mounts,
        "trailing_unanswered_call": value.trailing_unanswered_call,
        "no_final_assistant_turn": value.no_final_assistant_turn,
    }


def _project_compaction(value: CompactionRecord) -> dict[str, Any]:
    return {
        "origin": value.origin,
        "item_index": value.item_index,
        "item": value.item,
    }


def _project_message(message: Message) -> dict[str, Any]:
    result: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "assistant":
        result["reasoning_content"] = message.reasoning_content
        reasoning = getattr(message, "reasoning", None)
        if reasoning is not None:
            result["reasoning"] = reasoning
        if message.reasoning_details:
            result["reasoning_details"] = message.reasoning_details
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "type": call.type,
                    "id": call.id,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
    elif message.role == "tool":
        result["tool_call_id"] = message.tool_call_id
        result["name"] = message.name
    return result


def project_message(message: Message) -> dict[str, Any]:
    validated = _revalidate_source_model(message, Message, "$")
    assert isinstance(validated, Message)
    result = _project_message(validated)
    _validate_message(result, "$")
    return result


def _project_trajectory(
    node: TrajectoryNode, *, top_level: bool, path: str
) -> dict[str, Any]:
    if node.normalization_audit is None:
        _fail(path, "trajectory has no normalization_audit")
    if not top_level and (
        node.completeness_tag is not None or node.completeness is not None
    ):
        _fail(path, "nested trajectory source must omit completeness fields")
    result: dict[str, Any] = {
        "messages": [_project_message(message) for message in node.messages],
        "tools": [
            {
                "type": definition.type,
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.parameters,
            }
            for definition in node.tools
        ],
        "instructions": node.instructions,
        "termination": node.termination,
        "harness": node.harness,
        "model": node.model,
        "source": node.source,
        "total_rounds": node.total_rounds,
        "total_tool_calls": node.total_tool_calls,
        "tool_counts": dict(node.tool_counts),
        "reasoning_total_tokens": node.reasoning_total_tokens,
        "tool_defs_tag": node.tool_defs_tag,
        "missing_tool_defs": list(node.missing_tool_defs),
        "tool_call_tag": node.tool_call_tag,
        "tool_call_check": _project_check(node.tool_call_check),
        "server_tool_calls": [
            {
                "name": call.name,
                "id": call.id,
                "arguments": call.arguments,
                "origin": call.origin,
                "result": call.result,
            }
            for call in node.server_tool_calls
        ],
        "agent_messages": [
            {
                "origin": message.origin,
                "item_index": message.item_index,
                "item": message.item,
            }
            for message in node.agent_messages
        ],
        "compaction_items": [
            _project_compaction(compaction) for compaction in node.compaction_items
        ],
        "metadata": {
            "source_file": node.metadata.source_file,
            "source_name": node.metadata.source_name,
            "line_no": node.metadata.line_no,
            "created_at": node.metadata.created_at,
        },
        "normalization_audit": _project_audit(node.normalization_audit),
    }
    if node.sub_agent_trajectory:
        result["sub_agent_trajectory"] = {
            call_id: _project_trajectory(
                child,
                top_level=False,
                path=f"{path}/sub_agent_trajectory/{call_id}",
            )
            for call_id, child in node.sub_agent_trajectory.items()
        }
    if node.sub_agent_relay_mounts:
        result["sub_agent_relay_mounts"] = dict(node.sub_agent_relay_mounts)
    if top_level:
        if node.completeness_tag is None or node.completeness is None:
            _fail(path, "top-level trajectory has no completeness")
        result["completeness_tag"] = node.completeness_tag
        result["completeness"] = _project_completeness(node.completeness)
    return result


def project_trajectory(
    node: TrajectoryNode, *, top_level: bool = True
) -> dict[str, Any]:
    validated = _revalidate_source_model(node, TrajectoryNode, "$")
    assert isinstance(validated, TrajectoryNode)
    result = _project_trajectory(validated, top_level=top_level, path="$")
    _validate_node(result, "$", top_level=top_level)
    return result


def parse_trajectory_record(value: object) -> TrajectoryNode:
    """Validate raw required keys/types before Pydantic defaults can apply."""

    _validate_node(value, "$", top_level=True)
    try:
        node = TrajectoryNode.model_validate_json(orjson.dumps(value), strict=True)
    except (TypeError, orjson.JSONEncodeError, ValidationError) as error:
        raise OutputContractError("trajectory violates the typed model") from error
    if project_trajectory(node) != value:
        raise OutputContractError("trajectory is not in canonical output projection")
    return node


def project_quarantine_record(record: QuarantineRecord) -> dict[str, Any]:
    validated = _revalidate_source_model(record, QuarantineRecord, "$")
    assert isinstance(validated, QuarantineRecord)
    result = {
        "source_ref": validated.source_ref,
        "sha256": validated.sha256,
        "endpoint": validated.endpoint,
        "captured_at": validated.captured_at,
        "normalization_audit": _project_audit(validated.normalization_audit),
    }
    _validate_quarantine_value(result)
    return result


def _validate_quarantine_value(value: object) -> dict[str, Any]:
    item = _object(value, "$")
    _keys(
        item,
        "$",
        required={
            "source_ref",
            "sha256",
            "endpoint",
            "captured_at",
            "normalization_audit",
        },
    )
    _string(item["source_ref"], "$/source_ref", nonempty=True)
    _string(item["sha256"], "$/sha256")
    _string(item["endpoint"], "$/endpoint")
    _string(item["captured_at"], "$/captured_at")
    _validate_audit(item["normalization_audit"], "$/normalization_audit")
    return item


def parse_quarantine_record(value: object) -> QuarantineRecord:
    item = _validate_quarantine_value(value)
    try:
        record = QuarantineRecord.model_validate_json(orjson.dumps(item), strict=True)
    except (TypeError, orjson.JSONEncodeError, ValidationError) as error:
        raise OutputContractError(
            "quarantine record violates the typed model"
        ) from error
    if project_quarantine_record(record) != item:
        raise OutputContractError(
            "quarantine record is not in canonical output projection"
        )
    return record


__all__ = [
    "OutputContractError",
    "parse_quarantine_record",
    "parse_trajectory_record",
    "project_message",
    "project_quarantine_record",
    "project_trajectory",
]
