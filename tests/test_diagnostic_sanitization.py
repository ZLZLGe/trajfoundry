import pytest

from trajfoundry.models import AuditIssue, Message, sanitize_diagnostic_detail


@pytest.mark.parametrize(
    "detail,expected",
    [
        ("password=fake", "password=[REDACTED]"),
        ('api_key="fake"', "api_key=[REDACTED]"),
        ("passwd='fake'", "passwd=[REDACTED]"),
        ("Authorization: Bearer fake", "Authorization: [REDACTED]"),
        ("proxy-authorization=Basic ZmFrZTpmYWtl", "proxy-authorization=[REDACTED]"),
        ("Cookie: session=fake; path=/", "Cookie: [REDACTED]; path=/"),
        ("https://user:fake@example.test/path", "https://[REDACTED]@example.test/path"),
        ("peer 192.0.2.25, [2001:db8::1]", "peer [REDACTED], [[REDACTED]]"),
        ("key sk-proj-fake0123456789", "key [REDACTED]"),
        ("jwt eyJhbGciOiJIUzI1NiJ9.fake.signature", "jwt [REDACTED]"),
        ("password=[REDACTED]", "password=[REDACTED]"),
        ("[password=fake]", "[password=[REDACTED]]"),
        ('{"password":"fake"}', '{"password":[REDACTED]}'),
        ("ordinary diagnostic detail", "ordinary diagnostic detail"),
    ],
)
def test_diagnostic_redaction_is_canonical_across_json_round_trips(
    detail: str, expected: str
) -> None:
    assert sanitize_diagnostic_detail(detail) == expected
    original = AuditIssue(code="synthetic", stage="test", detail=detail)
    restored = original
    for _ in range(4):
        assert sanitize_diagnostic_detail(restored.detail) == expected
        restored = AuditIssue.model_validate_json(restored.model_dump_json())
        assert restored == original


@pytest.mark.parametrize(
    "detail",
    [
        "password=[REDACTED]fake-new-value",
        "password=[REDACTED] fake-new-value",
        'password="[REDACTED] fake-new-value"',
        "password=[REDACTED], api_key=fake-new-value",
        "already [REDACTED]; password=fake-new-value",
    ],
)
def test_existing_redaction_marker_does_not_bypass_new_secret(detail: str) -> None:
    result = sanitize_diagnostic_detail(detail)
    assert "fake-new-value" not in result
    assert "[REDACTED]" in result
    assert sanitize_diagnostic_detail(result) == result


@pytest.mark.parametrize(
    "fragment",
    [
        "password=fake",
        'api_key="fake"',
        "password=[REDACTED] fake-new-value",
        "Authorization: Bearer fake",
        "https://user:fake@example.test/",
        "192.0.2.25",
        "2001:db8::1",
        "::abcxyz",
        "2001:db8::abcdef123",
        "sk-proj-fake0123456789",
        "eyJhbGciOiJIUzI1NiJ9.fake.signature",
    ],
)
def test_truncated_diagnostic_is_stable_at_token_boundaries(fragment: str) -> None:
    for offset in range(980, 1030):
        result = sanitize_diagnostic_detail(
            "x" * offset + " " + fragment + "; " + "y" * 200
        )
        assert len(result) <= 1024 + len("...[TRUNCATED]")
        for _ in range(3):
            assert sanitize_diagnostic_detail(result) == result, (offset, fragment)


def test_redaction_does_not_modify_trajectory_message_content() -> None:
    content = "password=fake; peer 192.0.2.25"
    message = Message(role="user", content=content)
    restored = Message.model_validate_json(message.model_dump_json())
    assert restored.content == content
