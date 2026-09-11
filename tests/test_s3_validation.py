from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from trajfoundry.s3 import S3Location
from trajfoundry.s3_validation import S3ValidationBackend

_GEN_A = "a" * 32
_GEN_B = "b" * 32


class _NoSuchKey(Exception):
    pass


class _ClientError(Exception):
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        super().__init__("storage request failed")


class _Paginator:
    def __init__(self, pages_by_prefix: dict[str, list[dict[str, Any]]]) -> None:
        self.pages_by_prefix = pages_by_prefix
        self.calls: list[dict[str, str]] = []

    def paginate(self, **kwargs: str) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return self.pages_by_prefix[kwargs["Prefix"]]


class _Client:
    exceptions = SimpleNamespace(NoSuchKey=_NoSuchKey)

    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        get_error: Exception | None = None,
        pages_by_prefix: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.response = response
        self.get_error = get_error
        self.get_calls: list[dict[str, str]] = []
        self.paginator = _Paginator(pages_by_prefix or {})

    def get_object(self, **kwargs: str) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        if self.get_error is not None:
            raise self.get_error
        assert self.response is not None
        return self.response

    def get_paginator(self, operation: str) -> _Paginator:
        assert operation == "list_objects_v2"
        return self.paginator


class _Body(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.read_count = 0
        self.was_closed = False

    def read(self, size: int = -1) -> bytes:
        self.read_count += 1
        return super().read(size)

    def close(self) -> None:
        self.was_closed = True
        super().close()


def _backend(client: _Client) -> S3ValidationBackend:
    return S3ValidationBackend(client, S3Location("bucket", "normalized/dt=day/"))


def test_open_file_returns_the_unconsumed_response_stream() -> None:
    body = _Body(b"jsonl payload")
    client = _Client(response={"ContentLength": 13, "Body": body})
    backend = _backend(client)

    opened = backend.open_file(f"generations/{_GEN_A}/lineage.jsonl")

    assert opened is not None
    assert opened.size == 13
    assert opened.stream is body
    assert body.read_count == 0
    assert not body.was_closed
    assert client.get_calls == [
        {
            "Bucket": "bucket",
            "Key": f"normalized/dt=day/generations/{_GEN_A}/lineage.jsonl",
        }
    ]
    opened.stream.close()


@pytest.mark.parametrize(
    "error",
    [
        _NoSuchKey("missing"),
        _ClientError({"Error": {"Code": "NoSuchKey"}}),
        _ClientError({"Error": {"Code": "404"}}),
        _ClientError({"ResponseMetadata": {"HTTPStatusCode": 404}}),
    ],
)
def test_open_file_returns_none_for_s3_missing_object_forms(error: Exception) -> None:
    backend = _backend(_Client(get_error=error))

    assert backend.open_file(f"generations/{_GEN_A}/lineage.jsonl") is None


def test_open_file_propagates_non_404_storage_errors() -> None:
    error = _ClientError(
        {
            "Error": {"Code": "AccessDenied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        }
    )
    backend = _backend(_Client(get_error=error))

    with pytest.raises(_ClientError) as raised:
        backend.open_file(f"generations/{_GEN_A}/lineage.jsonl")

    assert raised.value is error


@pytest.mark.parametrize("size", [-1, True, "13", None])
def test_open_file_rejects_invalid_content_length_and_releases_body(
    size: object,
) -> None:
    body = _Body(b"payload")
    backend = _backend(_Client(response={"ContentLength": size, "Body": body}))

    with pytest.raises(TypeError, match="ContentLength"):
        backend.open_file(f"generations/{_GEN_A}/lineage.jsonl")

    assert body.was_closed
    assert body.read_count == 0


@pytest.mark.parametrize(
    "relative_path",
    ["../manifest.json", "/manifest.json", "a//manifest.json", "a\\manifest.json"],
)
def test_open_file_rejects_unsafe_relative_paths_before_s3(
    relative_path: str,
) -> None:
    client = _Client()
    backend = _backend(client)

    with pytest.raises(ValueError):
        backend.open_file(relative_path)

    assert client.get_calls == []


def test_generated_paths_are_filtered_and_stably_sorted_across_generations() -> None:
    prefix_a = f"normalized/dt=day/generations/{_GEN_A}/"
    prefix_b = f"normalized/dt=day/generations/{_GEN_B}/"
    client = _Client(
        pages_by_prefix={
            prefix_a: [
                {
                    "Contents": [
                        {"Key": prefix_a},
                        {"Key": f"{prefix_a}lineage.jsonl"},
                        {"Key": f"{prefix_a}metadata.json"},
                    ]
                },
                {"Contents": [{"Key": f"{prefix_a}accepted/z.jsonl"}]},
            ],
            prefix_b: [
                {
                    "Contents": [
                        {"Key": f"{prefix_b}quarantine/a.jsonl"},
                    ]
                }
            ],
        }
    )
    backend = _backend(client)

    paths = tuple(backend.iter_generated_jsonl_paths(frozenset({_GEN_B, _GEN_A})))

    assert paths == (
        f"generations/{_GEN_A}/accepted/z.jsonl",
        f"generations/{_GEN_A}/lineage.jsonl",
        f"generations/{_GEN_B}/quarantine/a.jsonl",
    )
    assert client.paginator.calls == [
        {"Bucket": "bucket", "Prefix": prefix_a},
        {"Bucket": "bucket", "Prefix": prefix_b},
    ]


def test_generated_paths_accept_an_empty_generation_set_without_listing() -> None:
    client = _Client()
    backend = _backend(client)

    assert tuple(backend.iter_generated_jsonl_paths(frozenset())) == ()
    assert client.paginator.calls == []


@pytest.mark.parametrize("generation_id", ["", "A" * 32, "a" * 31, "../unsafe"])
def test_generated_paths_reject_invalid_generation_ids(generation_id: str) -> None:
    client = _Client()
    backend = _backend(client)

    with pytest.raises(ValueError, match="generation id"):
        backend.iter_generated_jsonl_paths(frozenset({generation_id}))

    assert client.paginator.calls == []


@pytest.mark.parametrize(
    "keys,error",
    [
        (
            [f"normalized/dt=day/generations/{_GEN_B}/lineage.jsonl"],
            "outside the requested generation",
        ),
        (
            [f"normalized/dt=day/generations/{_GEN_A}/a//b.jsonl"],
            "unsafe object key",
        ),
        (
            [
                f"normalized/dt=day/generations/{_GEN_A}/lineage.jsonl",
                f"normalized/dt=day/generations/{_GEN_A}/lineage.jsonl",
            ],
            "duplicate key",
        ),
    ],
)
def test_generated_paths_reject_outside_unsafe_and_duplicate_keys(
    keys: list[str], error: str
) -> None:
    prefix = f"normalized/dt=day/generations/{_GEN_A}/"
    client = _Client(
        pages_by_prefix={prefix: [{"Contents": [{"Key": key} for key in keys]}]}
    )
    backend = _backend(client)

    with pytest.raises(OSError, match=error):
        backend.iter_generated_jsonl_paths(frozenset({_GEN_A}))


def test_generated_paths_propagate_listing_errors() -> None:
    class _FailingPaginator:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def paginate(self, **kwargs: str) -> Any:
            raise PermissionError("access denied")

    client = _Client()
    client.paginator = _FailingPaginator()  # type: ignore[assignment]
    backend = _backend(client)

    with pytest.raises(PermissionError, match="access denied"):
        backend.iter_generated_jsonl_paths(frozenset({_GEN_A}))
