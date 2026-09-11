"""S3-backed streaming access for output contract validation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from contextlib import suppress
from typing import Any

from .s3 import S3Location
from .validation import ValidationFile

_GENERATION_ID = re.compile(r"^[0-9a-f]{32}$")


def _is_not_found(error: Exception, client: Any) -> bool:
    """Recognize the two S3 missing-object forms without importing botocore."""

    exceptions = getattr(client, "exceptions", None)
    no_such_key = getattr(exceptions, "NoSuchKey", None)
    if isinstance(no_such_key, type) and isinstance(error, no_such_key):
        return True

    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return False
    error_details = response.get("Error")
    code = error_details.get("Code") if isinstance(error_details, Mapping) else None
    metadata = response.get("ResponseMetadata")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    return code in {"NoSuchKey", "404"} or status == 404


def _close_rejected_body(response: object) -> None:
    """Release a response that cannot safely be handed to the validator."""

    if not isinstance(response, Mapping):
        return
    body = response.get("Body")
    close = getattr(body, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


class S3ValidationBackend:
    """Expose one S3 output root through the validation backend protocol."""

    def __init__(self, client: Any, location: S3Location) -> None:
        self.client = client
        self.location = location

    def open_file(self, relative_path: str) -> ValidationFile | None:
        """Open one output object without consuming or closing its body."""

        key = self.location.key(relative_path)
        try:
            response = self.client.get_object(
                Bucket=self.location.bucket,
                Key=key,
            )
        except Exception as error:
            if _is_not_found(error, self.client):
                return None
            raise

        if not isinstance(response, Mapping):
            raise TypeError("S3 GetObject response must be an object")
        size = response.get("ContentLength")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            _close_rejected_body(response)
            raise TypeError("S3 ContentLength must be a non-negative integer")
        stream = response.get("Body")
        if stream is None:
            raise TypeError("S3 GetObject response must contain a body")
        return ValidationFile(size=size, stream=stream)

    def iter_generated_jsonl_paths(
        self, generation_ids: frozenset[str]
    ) -> Iterable[str]:
        """Enumerate every JSONL object under the requested generations."""

        for generation_id in generation_ids:
            if not isinstance(generation_id, str) or not _GENERATION_ID.fullmatch(
                generation_id
            ):
                raise ValueError(
                    "generation id must be 32 lowercase hexadecimal digits"
                )
        if not generation_ids:
            return ()

        paths: list[str] = []
        observed_keys: set[str] = set()
        paginator = self.client.get_paginator("list_objects_v2")
        for generation_id in sorted(generation_ids):
            generation_prefix = f"{self.location.prefix}generations/{generation_id}/"
            pages = paginator.paginate(
                Bucket=self.location.bucket,
                Prefix=generation_prefix,
            )
            for page in pages:
                contents = page.get("Contents", [])
                if contents is None:
                    contents = []
                if not isinstance(contents, list):
                    raise TypeError("S3 listing Contents must be a list")
                for item in contents:
                    if not isinstance(item, Mapping):
                        raise TypeError("S3 listing entry must be an object")
                    key = item.get("Key")
                    if not isinstance(key, str) or not key.startswith(
                        generation_prefix
                    ):
                        raise OSError(
                            "S3 listing returned a key outside the requested generation"
                        )
                    if key in observed_keys:
                        raise OSError("S3 listing returned a duplicate key")
                    observed_keys.add(key)
                    if key.endswith("/"):
                        continue

                    relative_path = key[len(self.location.prefix) :]
                    try:
                        expected_key = self.location.key(relative_path)
                    except (TypeError, ValueError) as error:
                        raise OSError(
                            "S3 listing returned an unsafe object key"
                        ) from error
                    if expected_key != key:
                        raise OSError("S3 listing returned an unsafe object key")
                    if relative_path.endswith(".jsonl"):
                        paths.append(relative_path)

        return tuple(sorted(paths))


__all__ = ["S3ValidationBackend"]
