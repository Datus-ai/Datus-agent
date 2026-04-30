# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Time utility functions for the Datus Agent."""

from datetime import datetime, timezone
from typing import Optional, Union

TimestampLike = Union[None, str, int, float, datetime]


def get_default_current_date(current_date: Optional[str]) -> str:
    """Get current_date or default to today's date if not set.

    Args:
        current_date: Optional date string in format 'YYYY-MM-DD'

    Returns:
        The provided current_date or today's date in 'YYYY-MM-DD' format
    """
    if current_date:
        return current_date
    return datetime.now().strftime("%Y-%m-%d")


def format_duration_human(seconds: float) -> str:
    """Format seconds into human readable (h/m/s) format."""
    seconds = int(seconds)

    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return "".join(parts)


def now_utc_iso() -> str:
    """Return current UTC time as ISO-8601 with ``Z`` suffix, second precision.

    Example: ``2026-04-30T12:34:56Z``.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(value: TimestampLike) -> Optional[datetime]:
    """Parse a heterogeneous timestamp value into an aware UTC ``datetime``.

    Accepts:
    - ``None`` or empty string → ``None``
    - ``int`` / ``float`` (Unix epoch seconds, UTC) → aware datetime
    - aware ``datetime`` → converted to UTC
    - naive ``datetime`` → assumed UTC
    - ``str`` in SQLite ``CURRENT_TIMESTAMP`` form ``YYYY-MM-DD HH:MM:SS`` (UTC)
      or ISO-8601 (with optional ``Z`` / ``+HH:MM`` offset). Naive forms are
      assumed UTC.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Normalize trailing 'Z' so fromisoformat can handle it.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        # SQLite default uses a space separator between date and time.
        if "T" not in text and " " in text:
            text = text.replace(" ", "T", 1)
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    return None


def to_utc_iso(value: TimestampLike) -> Optional[str]:
    """Normalize a timestamp value to ISO-8601 UTC string with ``Z`` suffix.

    Returns ``None`` for empty inputs (``None``/empty string) and for inputs
    that cannot be parsed. Output is always second precision regardless of
    sub-second components in the input.
    """
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def format_local_time(value: TimestampLike, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a UTC timestamp value in the local timezone for display.

    Returns an empty string when the input cannot be parsed, so call sites
    can use ``format_local_time(...) or "N/A"`` for fallback.
    """
    parsed = _parse_timestamp(value)
    if parsed is None:
        return ""
    return parsed.astimezone().strftime(fmt)
