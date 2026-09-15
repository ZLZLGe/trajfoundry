from __future__ import annotations

import hashlib
import json
from pathlib import Path

import orjson
import zstandard

from trajfoundry.io import LocalCaptureSource
from trajfoundry.providers.responses import parse_responses_capture
from trajfoundry.sources.sxf import adapt_sxf_envelope, parse_sse_events


def _responses_sse() -> str:
    return (
        ": keepalive\n\n"
        "event: response.created\n"
        'data: {"type":"response.created","sequence_number":0}\n\n'
        "event: response.completed\n"
        'data: {"type":"response.completed","sequence_number":1,"response":{"status":"completed","output":[]}}\n\n'
    )


def _chat_sse() -> str:
    chunks = [
        {
            "id": "chat-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "he"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chat-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"content": "llo"}, "finish_reason": "stop"}
            ],
        },
    ]
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    )


def test_parse_sse_events_handles_comments_and_event_data() -> None:
    events = parse_sse_events(_responses_sse())
    assert [event["type"] for event in events] == [
        "response.created",
        "response.completed",
    ]
    assert events[-1]["response"]["status"] == "completed"


def test_adapt_sxf_envelope_maps_raw_capture_and_chat_chunks() -> None:
    raw = {
        "schema": "tap_raw_capture_record.v1",
        "capture_id": "cap-1",
        "started_at": "2026-08-21T00:00:00Z",
        "completed_at": "2026-08-21T00:00:01Z",
        "path": "/v1/responses",
        "status_code": 200,
        "request_headers": {"Session-Id": ["session-1"], "Cookie": ["secret"]},
        "request_body": {
            "model": "m",
            "input": [],
            "client_metadata": {"session_id": "session-1"},
        },
        "response_body": _responses_sse(),
    }
    adapted = adapt_sxf_envelope(raw)
    assert adapted["request_id"] == "cap-1"
    assert adapted["captured_at"] == "2026-08-21T00:00:00Z"
    assert adapted["request_headers"] == {"session_id": "session-1"}
    assert isinstance(adapted["response_body"], list)

    chat = {
        "capture_meta": {
            "agent_kind": "codex",
            "duration_ms": 1,
            "routed_model": "m",
            "source": "test",
            "status_code": 200,
            "stream": True,
            "user_agent": "test",
        },
        "conversation_session_id": "session-2",
        "user_id": "sxf-user",
        "created_at": "2026-08-21T00:00:00Z",
        "method": "POST",
        "model": "m",
        "path": "/v1/chat/completions",
        "request_body": {"model": "m", "messages": []},
        "response_body": _chat_sse(),
        "session_id_source": "capture",
    }
    adapted_chat = adapt_sxf_envelope(chat)
    assert adapted_chat["session_id"] == "session-2"
    assert adapted_chat["user_id"] == "sxf-user"
    assert adapted_chat["response_body"]["object"] == "chat.completion"
    assert adapted_chat["response_body"]["choices"][0]["message"]["content"] == "hello"


def test_sxf_promotes_identity_headers_and_ignores_transport_events() -> None:
    raw = {
        "schema": "tap_raw_capture_record.v1",
        "capture_id": "cap-identity",
        "started_at": "2026-08-21T00:00:00Z",
        "path": "/v1/responses",
        "status_code": 200,
        "request_headers": {
            "Session-Id": ["session-from-header"],
            "Thread-Id": ["thread-from-header"],
        },
        "request_body": {"model": "m", "input": []},
        "response_body": (
            "event: response.created\n"
            'data: {"type":"response.created","sequence_number":0}\n\n'
            'data: {"type":"response.metadata","sequence_number":1,"metadata":{"moderation":{}}}\n\n'
            'data: {"type":"keepalive","sequence_number":2}\n\n'
            "event: response.completed\n"
            'data: {"type":"response.completed","sequence_number":3,"response":{"status":"completed","output":[]}}\n\n'
        ),
    }

    adapted = adapt_sxf_envelope(raw)
    assert adapted["session_id"] == "session-from-header"
    assert adapted["thread_id"] == "thread-from-header"
    snapshot = parse_responses_capture(
        adapted,
        source_path="part-000001.jsonl.zst#L00000000",
        source_sha256="a" * 64,
    )
    assert snapshot.outcome == "success"
    assert snapshot.wire_complete is True
    assert not any(issue.code == "unknown_sse_event" for issue in snapshot.issues)
    assert snapshot.session_id == "session-from-header"
    assert snapshot.thread_id == "thread-from-header"


def test_local_sxf_source_streams_rows_with_line_refs(tmp_path: Path) -> None:
    rows = [
        {"schema": "tap_raw_capture_record.v1", "capture_id": "a"},
        {"schema": "tap_raw_capture_record.v1", "capture_id": "b"},
    ]
    destination = tmp_path / "part-000001.jsonl.zst"
    compressor = zstandard.ZstdCompressor()
    with destination.open("wb") as handle, compressor.stream_writer(handle) as writer:
        for row in rows:
            writer.write(orjson.dumps(row) + b"\n")

    source = LocalCaptureSource(tmp_path)
    payloads = list(source.iter_capture_payloads("sxf"))
    assert [item[0] for item in payloads] == [
        "part-000001.jsonl.zst#L00000000",
        "part-000001.jsonl.zst#L00000001",
    ]
    assert payloads[0][2] == hashlib.sha256(payloads[0][1]).hexdigest()
