from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from trajfoundry.credentials import (
    DEFAULT_S3_CREDENTIALS_PATH,
    MAX_CREDENTIAL_FILE_BYTES,
    CredentialFileError,
    S3Credentials,
    load_s3_credentials,
)


def _write_credentials(path: Path, document: object, *, mode: int = 0o600) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")
    path.chmod(mode)


def test_default_path_is_fixed_outside_the_repository() -> None:
    assert DEFAULT_S3_CREDENTIALS_PATH == Path(
        "/share/gezhilong/trajfoundry-secrets/s3_credentials.json"
    )


@pytest.mark.parametrize("mode", [0o400, 0o600])
def test_load_credentials_accepts_safe_owner_only_modes(
    tmp_path: Path, mode: int
) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(
        path,
        {
            "aws_access_key_id": "test-access-id",
            "aws_secret_access_key": "test-secret-key",
        },
        mode=mode,
    )

    credentials = load_s3_credentials(path)

    assert credentials == S3Credentials(
        aws_access_key_id="test-access-id",
        aws_secret_access_key="test-secret-key",
    )
    with pytest.raises((AttributeError, TypeError)):
        credentials.aws_access_key_id = "replacement"  # type: ignore[misc]


def test_load_credentials_accepts_a_session_token(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(
        path,
        {
            "aws_access_key_id": "test-access-id",
            "aws_secret_access_key": "test-secret-key",
            "aws_session_token": "test-session-token",
        },
    )

    assert load_s3_credentials(path).aws_session_token == "test-session-token"


def test_credentials_repr_omits_every_value() -> None:
    values = ("distinct-access-id", "distinct-secret-key", "distinct-session-token")
    representation = repr(S3Credentials(*values))

    assert representation == "S3Credentials()"
    assert all(value not in representation for value in values)


@pytest.mark.parametrize("mode", [0o000, 0o200, 0o700, 0o640, 0o604, 0o444])
def test_load_credentials_rejects_unsafe_permissions(tmp_path: Path, mode: int) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(
        path,
        {
            "aws_access_key_id": "test-access-id",
            "aws_secret_access_key": "test-secret-key",
        },
        mode=mode,
    )

    with pytest.raises(CredentialFileError):
        load_s3_credentials(path)


def test_load_credentials_rejects_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    link = tmp_path / "credentials.json"
    _write_credentials(
        target,
        {
            "aws_access_key_id": "test-access-id",
            "aws_secret_access_key": "test-secret-key",
        },
    )
    link.symlink_to(target)

    with pytest.raises(CredentialFileError, match="opened safely"):
        load_s3_credentials(link)


def test_load_credentials_rejects_a_non_file(tmp_path: Path) -> None:
    directory = tmp_path / "credentials.json"
    directory.mkdir(mode=0o700)

    with pytest.raises(CredentialFileError, match="regular file"):
        load_s3_credentials(directory)


def test_load_credentials_requires_the_current_user_to_own_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(
        path,
        {
            "aws_access_key_id": "test-access-id",
            "aws_secret_access_key": "test-secret-key",
        },
    )
    monkeypatch.setattr(
        "trajfoundry.credentials.os.geteuid", lambda: path.stat().st_uid + 1
    )

    with pytest.raises(CredentialFileError, match="owned by the current user"):
        load_s3_credentials(path)


def test_load_credentials_rejects_an_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_bytes(b"x" * (MAX_CREDENTIAL_FILE_BYTES + 1))
    path.chmod(0o600)

    with pytest.raises(CredentialFileError, match="too large"):
        load_s3_credentials(path)


@pytest.mark.parametrize(
    "payload",
    [
        b"not JSON",
        b'\xff{"aws_access_key_id":"id","aws_secret_access_key":"secret"}',
        (
            b'{"aws_access_key_id":"id","aws_access_key_id":"other",'
            b'"aws_secret_access_key":"secret"}'
        ),
    ],
)
def test_load_credentials_rejects_invalid_json(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "credentials.json"
    path.write_bytes(payload)
    path.chmod(0o600)

    with pytest.raises(CredentialFileError, match="valid JSON"):
        load_s3_credentials(path)


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"aws_access_key_id": "id"},
        {"aws_secret_access_key": "secret"},
        {
            "aws_access_key_id": "id",
            "aws_secret_access_key": "secret",
            "unexpected": "value",
        },
    ],
)
def test_load_credentials_rejects_invalid_schema(
    tmp_path: Path, document: object
) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(path, document)

    with pytest.raises(CredentialFileError, match="schema"):
        load_s3_credentials(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("aws_access_key_id", 1),
        ("aws_access_key_id", ""),
        ("aws_secret_access_key", "   "),
        ("aws_secret_access_key", None),
        ("aws_session_token", False),
        ("aws_session_token", ""),
    ],
)
def test_load_credentials_rejects_non_string_or_empty_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    document: dict[str, object] = {
        "aws_access_key_id": "id",
        "aws_secret_access_key": "secret",
    }
    document[field] = value
    path = tmp_path / "credentials.json"
    _write_credentials(path, document)

    with pytest.raises(CredentialFileError, match="schema"):
        load_s3_credentials(path)


def test_errors_do_not_include_file_contents(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    secret = "never-show-this-secret"
    _write_credentials(
        path,
        {
            "aws_access_key_id": "id",
            "aws_secret_access_key": secret,
            "unexpected": "value",
        },
    )

    with pytest.raises(CredentialFileError) as error:
        load_s3_credentials(path)

    assert secret not in str(error.value)


def test_load_credentials_rejects_a_file_changed_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(
        path,
        {
            "aws_access_key_id": "id",
            "aws_secret_access_key": "secret",
        },
    )
    original_read = os.read
    changed = False

    def changing_read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        result = original_read(descriptor, count)
        if result and not changed:
            changed = True
            path.write_bytes(path.read_bytes() + b" ")
            path.chmod(0o600)
        return result

    monkeypatch.setattr("trajfoundry.credentials.os.read", changing_read)

    with pytest.raises(CredentialFileError, match="changed"):
        load_s3_credentials(path)
