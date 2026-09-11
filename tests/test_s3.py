from __future__ import annotations

import hashlib
import io
import sys
import types
from typing import Any

import pytest

from trajfoundry.s3 import (
    S3Capture,
    S3CaptureSource,
    S3Location,
    create_s3_client,
    parse_s3_uri,
)


class _Paginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, str]] = []

    def paginate(self, **kwargs: str) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return self.pages


class _Body(io.BytesIO):
    def __init__(self, payload: bytes, *, fail_after_first_read: bool = False) -> None:
        super().__init__(payload)
        self.fail_after_first_read = fail_after_first_read
        self.reads = 0
        self.was_closed = False

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self.fail_after_first_read and self.reads > 1:
            raise OSError("stream interrupted")
        return super().read(size)

    def close(self) -> None:
        self.was_closed = True
        super().close()


class _Client:
    def __init__(
        self,
        pages: list[dict[str, Any]] | None = None,
        response: dict[str, Any] | None = None,
    ) -> None:
        self.paginator = _Paginator(pages or [])
        self.response = response
        self.get_calls: list[dict[str, str]] = []

    def get_paginator(self, operation: str) -> _Paginator:
        assert operation == "list_objects_v2"
        return self.paginator

    def get_object(self, **kwargs: str) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        assert self.response is not None
        return self.response


def _listed(key: str, *, size: int = 1, etag: str | None = None) -> dict[str, Any]:
    return {"Key": key, "Size": size, "ETag": etag or f'"{key}"'}


def test_parse_s3_uri_preserves_a_canonical_directory_prefix() -> None:
    location = parse_s3_uri(
        "s3://agent-trajectory/lakehouse/free-router/dt=2026-09-09/"
    )

    assert location == S3Location(
        bucket="agent-trajectory",
        prefix="lakehouse/free-router/dt=2026-09-09/",
    )
    assert location.uri == (
        "s3://agent-trajectory/lakehouse/free-router/dt=2026-09-09/"
    )
    assert location.key("nested/capture.json") == (
        "lakehouse/free-router/dt=2026-09-09/nested/capture.json"
    )


@pytest.mark.parametrize(
    "uri",
    [
        "https://bucket/prefix/",
        "S3://bucket/prefix/",
        "s3:///prefix/",
        "s3://bucket",
        "s3://bucket/",
        "s3://user:password@bucket/prefix/",
        "s3://bucket:9000/prefix/",
        "s3://bucket/prefix/?versionId=secret",
        "s3://bucket/prefix/#fragment",
        "s3://bucket//prefix/",
        "s3://bucket/prefix//nested/",
        "s3://bucket/./prefix/",
        "s3://bucket/prefix/../other/",
        "s3://bucket/prefix",
        "s3://bucket/prefix\\nested/",
        "s3://bucket/prefix\x00/",
        "s3://bucket/prefix\nother/",
        "s3://bucket/prefix\tother/",
        "s3://bucket/prefix\x1bother/",
        "s3://bucket/prefix\x7fother/",
    ],
)
def test_parse_s3_uri_rejects_ambiguous_or_unsafe_locations(uri: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_s3_uri(uri)


@pytest.mark.parametrize(
    "relative",
    [
        "",
        "/capture.json",
        "../capture.json",
        "a/./capture.json",
        "a//b.json",
        "a/",
        "a/escape\x1b.json",
        "a/delete\x7f.json",
    ],
)
def test_location_key_rejects_unsafe_relative_paths(relative: str) -> None:
    location = S3Location("bucket", "prefix/")

    with pytest.raises(ValueError):
        location.key(relative)


def test_freerouter_listing_filters_markers_and_sorts_relative_keys() -> None:
    client = _Client(
        [
            {
                "Contents": [
                    _listed("input/z.json", size=3, etag='"z"'),
                    _listed("input/readme.txt"),
                    _listed("input/nested/", size=0),
                ]
            },
            {
                "Contents": [
                    _listed("input/nested/a.json", size=7, etag='"a"'),
                    _listed("input/B.JSON"),
                ]
            },
        ]
    )
    source = S3CaptureSource(client, S3Location("bucket", "input/"))

    captures = tuple(source.iter_captures("freerouter"))

    assert source.label == "s3://bucket/input/"
    assert captures == (
        S3Capture(
            source_ref="nested/a.json",
            key="input/nested/a.json",
            size=7,
            etag='"a"',
        ),
        S3Capture(source_ref="z.json", key="input/z.json", size=3, etag='"z"'),
    )
    assert client.paginator.calls == [{"Bucket": "bucket", "Prefix": "input/"}]


def test_tokenplan_listing_selects_only_request_envelopes_recursively() -> None:
    client = _Client(
        [
            {
                "Contents": [
                    _listed("input/run/req_002.json"),
                    _listed("input/req_001.json"),
                    _listed("input/run/not_req.json"),
                    _listed("input/run/req_003.jsonl"),
                    _listed("input/run/REQ_004.json"),
                ]
            }
        ]
    )
    source = S3CaptureSource(client, S3Location("bucket", "input/"))

    assert [item.source_ref for item in source.iter_captures("tokenplan")] == [
        "req_001.json",
        "run/req_002.json",
    ]


def test_listing_propagates_service_errors() -> None:
    class _FailingPaginator:
        def paginate(self, **kwargs: str) -> Any:
            raise PermissionError("access denied")

    client = _Client()
    client.paginator = _FailingPaginator()  # type: ignore[assignment]
    source = S3CaptureSource(client, S3Location("bucket", "input/"))

    with pytest.raises(PermissionError, match="access denied"):
        source.iter_captures("freerouter")


@pytest.mark.parametrize(
    "pages, message",
    [
        (
            [{"Contents": [_listed("other/capture.json")]}],
            "outside the requested prefix",
        ),
        (
            [
                {
                    "Contents": [
                        _listed("input/capture.json"),
                        _listed("input/capture.json"),
                    ]
                }
            ],
            "duplicate key",
        ),
    ],
)
def test_listing_rejects_unstable_results(
    pages: list[dict[str, Any]], message: str
) -> None:
    source = S3CaptureSource(_Client(pages), S3Location("bucket", "input/"))

    with pytest.raises(OSError, match=message):
        source.iter_captures("freerouter")


def test_read_capture_is_conditional_hashed_and_closed() -> None:
    payload = b"capture payload"
    body = _Body(payload)
    client = _Client(
        response={"Body": body, "ContentLength": len(payload), "ETag": '"etag"'}
    )
    source = S3CaptureSource(client, S3Location("bucket", "input/"))
    capture = S3Capture("a.json", "input/a.json", len(payload), '"etag"')

    actual, digest = source.read_capture_bytes(capture)

    assert actual == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    assert body.was_closed
    assert client.get_calls == [
        {"Bucket": "bucket", "Key": "input/a.json", "IfMatch": '"etag"'}
    ]


def test_read_capture_rejects_a_capture_from_another_location_without_requesting_it() -> (
    None
):
    client = _Client()
    source = S3CaptureSource(client, S3Location("bucket", "input/"))
    capture = S3Capture("a.json", "other/a.json", 7, '"etag"')

    with pytest.raises(ValueError, match="outside its source location"):
        source.read_capture_bytes(capture)

    assert client.get_calls == []


@pytest.mark.parametrize(
    "response_size,response_etag,error",
    [
        (6, '"etag"', "size changed"),
        (7, '"different"', "ETag changed"),
    ],
)
def test_read_capture_rejects_changed_metadata_and_closes_body(
    response_size: int, response_etag: str, error: str
) -> None:
    body = _Body(b"payload")
    client = _Client(
        response={
            "Body": body,
            "ContentLength": response_size,
            "ETag": response_etag,
        }
    )
    source = S3CaptureSource(client, S3Location("bucket", "input/"))
    capture = S3Capture("a.json", "input/a.json", 7, '"etag"')

    with pytest.raises(OSError, match=error):
        source.read_capture_bytes(capture)

    assert body.was_closed


def test_read_capture_closes_body_when_stream_fails() -> None:
    body = _Body(b"payload", fail_after_first_read=True)
    client = _Client(response={"Body": body, "ContentLength": 7, "ETag": '"etag"'})
    source = S3CaptureSource(client, S3Location("bucket", "input/"))
    capture = S3Capture("a.json", "input/a.json", 7, '"etag"')

    with pytest.raises(OSError, match="stream interrupted"):
        source.read_capture_bytes(capture)

    assert body.was_closed


def test_create_s3_client_uses_sigv4_path_style_and_standard_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Config:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_boto3 = types.ModuleType("boto3")

    def _client(service: str, **kwargs: Any) -> object:
        calls.append((service, kwargs))
        return object()

    fake_boto3.client = _client  # type: ignore[attr-defined]
    fake_botocore = types.ModuleType("botocore")
    fake_config = types.ModuleType("botocore.config")
    fake_config.Config = _Config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_config)

    created = create_s3_client("http://ceph.internal", "us-east-1")

    assert created is not None
    assert len(calls) == 1
    service, kwargs = calls[0]
    assert service == "s3"
    assert kwargs["endpoint_url"] == "http://ceph.internal"
    assert kwargs["region_name"] == "us-east-1"
    assert kwargs["config"].kwargs == {
        "signature_version": "s3v4",
        "s3": {"addressing_style": "path"},
        "retries": {"mode": "standard", "max_attempts": 10},
    }
