"""Pure timestamp policy shared by grouping and manifest identity validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

_BEIJING = timezone(timedelta(hours=8))


def sessionless_window(captured_at: str) -> str:
    """Return the containing fixed eight-hour Beijing window, if time is known.

    Only timezone-aware instants establish a window. Missing, malformed, or
    timezone-naive timestamps retain the historical unknown-time user scope.
    A date is part of the key, so the same clock block on different days is
    never aggregated together.
    """

    try:
        instant = datetime.fromisoformat(captured_at)
        if instant.tzinfo is None or instant.utcoffset() is None:
            return ""
        local = instant.astimezone(_BEIJING)
        return local.replace(
            hour=(local.hour // 8) * 8, minute=0, second=0, microsecond=0
        ).isoformat()
    except (TypeError, ValueError, OverflowError):
        return ""
