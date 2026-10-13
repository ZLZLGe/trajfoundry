import pytest
from pydantic import ValidationError

from trajfoundry.models import Snapshot


def snapshot_fields(session_id: object) -> dict:
    return {
        "source_path": "capture.json",
        "source_sha256": "a" * 64,
        "session_id": session_id,
        "thread_id": "",
        "provider": "openai",
        "operation": "responses",
        "outcome": "success",
    }


@pytest.mark.parametrize(
    "session_id,expected",
    [("", ""), ("no_session_id", ""), ("real-session", "real-session")],
)
def test_snapshot_normalizes_missing_session_sentinel(session_id, expected):
    assert Snapshot.model_validate(snapshot_fields(session_id)).session_id == expected


@pytest.mark.parametrize("session_id", [None, 123, False, [], {}])
def test_snapshot_session_normalization_preserves_strict_type_validation(session_id):
    with pytest.raises(ValidationError):
        Snapshot.model_validate(snapshot_fields(session_id))
