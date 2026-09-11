"""Load S3 credentials from a fixed, local, permission-restricted file."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_S3_CREDENTIALS_PATH = Path(
    "/share/gezhilong/trajfoundry-secrets/s3_credentials.json"
)
MAX_CREDENTIAL_FILE_BYTES = 64 * 1024

_ALLOWED_FILE_MODES = {0o400, 0o600}
_ALLOWED_FIELDS = {
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
}
_REQUIRED_FIELDS = {"aws_access_key_id", "aws_secret_access_key"}


class CredentialFileError(ValueError):
    """Raised when the credential file cannot be read or validated safely."""


@dataclass(frozen=True, slots=True)
class S3Credentials:
    """S3 credentials whose representation never includes secret material."""

    aws_access_key_id: str = field(repr=False)
    aws_secret_access_key: str = field(repr=False)
    aws_session_token: str | None = field(default=None, repr=False)


def _stable_file_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_uid,
        file_stat.st_gid,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _read_secure_file(path: str | os.PathLike[str]) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise CredentialFileError("secure credential file access is unavailable")

    flags = os.O_RDONLY | nofollow
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)

    try:
        descriptor = os.open(path, flags)
    except (OSError, TypeError, ValueError):
        raise CredentialFileError(
            "credential file could not be opened safely"
        ) from None

    try:
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise CredentialFileError("credential path is not a regular file")
            if before.st_uid != os.geteuid():
                raise CredentialFileError(
                    "credential file is not owned by the current user"
                )
            if stat.S_IMODE(before.st_mode) not in _ALLOWED_FILE_MODES:
                raise CredentialFileError("credential file permissions are unsafe")
            if before.st_size > MAX_CREDENTIAL_FILE_BYTES:
                raise CredentialFileError("credential file is too large")

            chunks: list[bytes] = []
            byte_count = 0
            while byte_count <= MAX_CREDENTIAL_FILE_BYTES:
                chunk = os.read(
                    descriptor,
                    min(8192, MAX_CREDENTIAL_FILE_BYTES + 1 - byte_count),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                byte_count += len(chunk)
            if byte_count > MAX_CREDENTIAL_FILE_BYTES:
                raise CredentialFileError("credential file is too large")

            after = os.fstat(descriptor)
        except CredentialFileError:
            raise
        except OSError:
            raise CredentialFileError(
                "credential file could not be read safely"
            ) from None

        if _stable_file_identity(before) != _stable_file_identity(after):
            raise CredentialFileError("credential file changed while being read")
        if byte_count != before.st_size:
            raise CredentialFileError("credential file changed while being read")
        return b"".join(chunks)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError
        result[name] = value
    return result


def _reject_json_constant(_: str) -> None:
    raise ValueError


def load_s3_credentials(
    path: str | os.PathLike[str] = DEFAULT_S3_CREDENTIALS_PATH,
) -> S3Credentials:
    """Load and strictly validate an S3 credential JSON file.

    The file must be a non-symlink regular file owned by the current effective
    user, at most 64 KiB, and have mode ``0400`` or ``0600``. Error messages
    intentionally omit file contents and credential values.
    """

    payload = _read_secure_file(path)
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise CredentialFileError("credential file is not valid JSON") from None

    if not isinstance(document, dict):
        raise CredentialFileError("credential file schema is invalid")
    fields = set(document)
    if not _REQUIRED_FIELDS.issubset(fields) or not fields.issubset(_ALLOWED_FIELDS):
        raise CredentialFileError("credential file schema is invalid")
    if any(
        not isinstance(value, str) or not value.strip() for value in document.values()
    ):
        raise CredentialFileError("credential file schema is invalid")

    return S3Credentials(
        aws_access_key_id=document["aws_access_key_id"],
        aws_secret_access_key=document["aws_secret_access_key"],
        aws_session_token=document.get("aws_session_token"),
    )
