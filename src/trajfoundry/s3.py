"""Strict S3 locations and stable capture reads.

The module deliberately keeps the AWS SDK behind :func:`create_s3_client` so
importing TrajFoundry does not initialize an SDK session or require boto3 for
local-only jobs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from .credentials import S3Credentials

InputFormat = Literal["freerouter", "tokenplan"]
_READ_CHUNK_BYTES = 1024 * 1024


def _validate_bucket(bucket: str) -> None:
    if not isinstance(bucket, str):
        raise TypeError("S3 bucket must be a string")
    if not bucket:
        raise ValueError("S3 bucket must not be empty")
    if bucket in {".", ".."}:
        raise ValueError("S3 bucket is invalid")
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in bucket
    ):
        raise ValueError("S3 bucket must not contain whitespace or control characters")
    if any(character in bucket for character in "/\\@?#:"):
        raise ValueError("S3 bucket must not contain URI delimiters")


def _validate_path_segments(value: str, *, name: str, directory: bool) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if value.startswith("/"):
        raise ValueError(f"{name} must be relative")
    if "\\" in value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(f"{name} must not contain a backslash or control character")
    if directory:
        if not value.endswith("/"):
            raise ValueError(f"{name} must end with '/'")
        segment_value = value[:-1]
    else:
        if value.endswith("/"):
            raise ValueError(f"{name} must identify an object, not a directory")
        segment_value = value
    segments = segment_value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"{name} contains an unsafe path segment")


@dataclass(frozen=True, slots=True)
class S3Location:
    """A bucket and a canonical, non-root directory prefix."""

    bucket: str
    prefix: str

    def __post_init__(self) -> None:
        _validate_bucket(self.bucket)
        _validate_path_segments(self.prefix, name="S3 prefix", directory=True)

    @property
    def uri(self) -> str:
        """Return the location without embedding credentials or query data."""

        return f"s3://{self.bucket}/{self.prefix}"

    def key(self, relative: str) -> str:
        """Join a safe object-relative path beneath this location."""

        _validate_path_segments(relative, name="relative S3 key", directory=False)
        return f"{self.prefix}{relative}"


def parse_s3_uri(uri: str) -> S3Location:
    """Parse a strict ``s3://bucket/non-empty/prefix/`` directory URI.

    Valid key text is preserved byte-for-byte.  Ambiguous path spellings are
    rejected instead of being silently normalized to a different S3 key.
    """

    if not isinstance(uri, str):
        raise TypeError("S3 URI must be a string")
    if any(ord(character) < 32 or ord(character) == 127 for character in uri):
        raise ValueError("S3 URI must not contain control characters")
    if not uri.startswith("s3://"):
        raise ValueError("S3 URI must start with 's3://'")
    if "?" in uri or "#" in uri:
        raise ValueError("S3 URI must not contain a query or fragment")

    parsed = urlsplit(uri)
    if parsed.scheme != "s3":  # Defensive; the exact prefix check is authoritative.
        raise ValueError("S3 URI must use the s3 scheme")
    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
    ):
        raise ValueError("S3 URI must not contain credentials")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("S3 URI contains an invalid authority") from error
    if port is not None:
        raise ValueError("S3 URI must not contain a port")
    if parsed.netloc != parsed.hostname:
        raise ValueError("S3 URI authority must contain only a bucket")
    if not parsed.path.startswith("/"):
        raise ValueError("S3 URI must contain an absolute URI path")

    # Remove only the separator introduced by URI syntax.  A second leading
    # slash remains visible to validation and is rejected as an empty segment.
    prefix = parsed.path[1:]
    return S3Location(bucket=parsed.netloc, prefix=prefix)


def create_s3_client(
    endpoint_url: str,
    region_name: str,
    *,
    credentials: S3Credentials | None = None,
) -> Any:
    """Create the configured S3-compatible client, importing boto3 lazily."""

    import boto3
    from botocore.config import Config

    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        retries={"mode": "standard", "max_attempts": 10},
    )
    client_kwargs: dict[str, Any] = {
        "endpoint_url": endpoint_url,
        "region_name": region_name,
        "config": config,
    }
    if credentials is not None:
        client_kwargs["aws_access_key_id"] = credentials.aws_access_key_id
        client_kwargs["aws_secret_access_key"] = credentials.aws_secret_access_key
        if credentials.aws_session_token is not None:
            client_kwargs["aws_session_token"] = credentials.aws_session_token

    return boto3.client("s3", **client_kwargs)


@dataclass(frozen=True, slots=True)
class S3Capture:
    """Immutable identity metadata for one capture in an S3 input scan."""

    source_ref: str
    key: str
    size: int
    etag: str


class S3CaptureSource:
    """List and read stable capture objects under one S3 directory prefix."""

    def __init__(self, client: Any, location: S3Location) -> None:
        self.client = client
        self.location = location

    @property
    def label(self) -> str:
        return self.location.uri

    def iter_captures(self, input_format: InputFormat) -> Iterator[S3Capture]:
        """Return capture metadata in stable relative-key order."""

        if input_format not in {"freerouter", "tokenplan"}:
            raise ValueError(f"unsupported input format: {input_format!r}")

        captures: list[S3Capture] = []
        observed_keys: set[str] = set()
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.location.bucket,
            Prefix=self.location.prefix,
        )
        for page in pages:
            contents = page.get("Contents", [])
            if contents is None:
                contents = []
            if not isinstance(contents, list):
                raise TypeError("S3 listing Contents must be a list")
            for item in contents:
                if not isinstance(item, dict):
                    raise TypeError("S3 listing entry must be an object")
                key = item["Key"]
                if not isinstance(key, str) or not key.startswith(self.location.prefix):
                    raise OSError(
                        "S3 listing returned a key outside the requested prefix"
                    )
                if key in observed_keys:
                    raise OSError("S3 listing returned a duplicate key")
                observed_keys.add(key)
                if key.endswith("/"):
                    continue
                size = item["Size"]
                etag = item["ETag"]
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise TypeError("S3 listing Size must be a non-negative integer")
                if not isinstance(etag, str) or not etag:
                    raise TypeError("S3 listing ETag must be a non-empty string")

                source_ref = key[len(self.location.prefix) :]
                if not source_ref:
                    continue
                if self.location.key(source_ref) != key:  # Defensive exact join check.
                    raise OSError("S3 listing returned an unsafe relative key")
                basename = source_ref.rsplit("/", 1)[-1]
                if input_format == "freerouter":
                    selected = basename.endswith(".json")
                else:
                    selected = basename.startswith("req_") and basename.endswith(
                        ".json"
                    )
                if selected:
                    captures.append(
                        S3Capture(
                            source_ref=source_ref,
                            key=key,
                            size=size,
                            etag=etag,
                        )
                    )

        return iter(sorted(captures, key=lambda capture: capture.source_ref))

    def read_capture_bytes(self, capture: S3Capture) -> tuple[bytes, str]:
        """Conditionally read, close, length-check, and hash one S3 object."""

        if self.location.key(capture.source_ref) != capture.key:
            raise ValueError("S3 capture key is outside its source location")
        if (
            not isinstance(capture.size, int)
            or isinstance(capture.size, bool)
            or capture.size < 0
        ):
            raise ValueError("S3 capture size must be a non-negative integer")
        if not isinstance(capture.etag, str) or not capture.etag:
            raise ValueError("S3 capture ETag must be a non-empty string")
        response = self.client.get_object(
            Bucket=self.location.bucket,
            Key=capture.key,
            IfMatch=capture.etag,
        )
        body = response["Body"]
        payload = bytearray()
        digest = hashlib.sha256()
        try:
            while True:
                chunk = body.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise TypeError("S3 response body must yield bytes")
                payload.extend(chunk)
                digest.update(chunk)
        finally:
            body.close()

        content_length = response.get("ContentLength")
        response_etag = response.get("ETag")
        if (
            not isinstance(content_length, int)
            or isinstance(content_length, bool)
            or content_length != capture.size
            or len(payload) != capture.size
        ):
            raise OSError("S3 capture size changed while it was being read")
        if not isinstance(response_etag, str) or response_etag != capture.etag:
            raise OSError("S3 capture ETag changed while it was being read")
        return bytes(payload), digest.hexdigest()


__all__ = [
    "InputFormat",
    "S3Capture",
    "S3CaptureSource",
    "S3Location",
    "create_s3_client",
    "parse_s3_uri",
]
