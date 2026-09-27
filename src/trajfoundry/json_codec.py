"""JSON helpers that preserve integers larger than the orjson range."""

from __future__ import annotations

import json as _json
import math
import re
from typing import Any

import orjson

_MAX_ORJSON_INTEGER = 2**64 - 1
_MIN_ORJSON_INTEGER = -(2**63)
_HUGE_INTEGER_TOKEN = re.compile(rb"(?:^|[\[,:])\s*-?[0-9]{19,}\s*(?=[,\]}])")


def _may_contain_coerced_integer(value: object) -> bool:
    """Detect the float representation produced by orjson for huge integers."""

    if isinstance(value, float):
        return (
            math.isfinite(value)
            and value.is_integer()
            and (value > _MAX_ORJSON_INTEGER or value <= _MIN_ORJSON_INTEGER)
        )
    if isinstance(value, dict):
        return any(_may_contain_coerced_integer(item) for item in value.values())
    if isinstance(value, list):
        return any(_may_contain_coerced_integer(item) for item in value)
    return False


def loads(value: str | bytes | bytearray) -> Any:
    """Decode JSON quickly, reparsing only when orjson overflowed an integer."""

    try:
        decoded = orjson.loads(value)
    except orjson.JSONDecodeError as error:
        if "number is infinity when parsed as double" not in str(
            error
        ) or not _HUGE_INTEGER_TOKEN.search(
            value.encode("utf-8") if isinstance(value, str) else value
        ):
            raise
        try:
            return _json.loads(value)
        except _json.JSONDecodeError:
            raise error
    if _may_contain_coerced_integer(decoded):
        # The standard decoder keeps arbitrary precision JSON integers.  The
        # fast path above has already established that the payload is valid
        # strict JSON, so this fallback cannot admit NaN or Infinity.
        return _json.loads(value)
    return decoded


def dumps(
    value: Any,
    *,
    sort_keys: bool = False,
    indent: int | None = None,
) -> bytes:
    """Encode JSON with an stdlib fallback for integers beyond 64 bits."""

    option = 0
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS
    if indent == 2:
        option |= orjson.OPT_INDENT_2
    try:
        return orjson.dumps(value, option=option)
    except TypeError as error:
        if str(error) != "Integer exceeds 64-bit range":
            raise
        return _json.dumps(
            value,
            sort_keys=sort_keys,
            ensure_ascii=False,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        ).encode("utf-8")


__all__ = ["dumps", "loads"]
