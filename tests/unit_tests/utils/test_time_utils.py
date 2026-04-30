# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/time_utils.py — CI tier, zero external deps."""

import re
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from datus.utils.time_utils import (
    format_duration_human,
    format_local_time,
    get_default_current_date,
    now_utc_iso,
    to_utc_iso,
)


class TestGetDefaultCurrentDate:
    """Tests for get_default_current_date."""

    def test_returns_provided_date_when_not_none(self):
        """When a date string is given it is returned unchanged."""
        assert get_default_current_date("2025-06-15") == "2025-06-15"

    def test_returns_provided_date_when_arbitrary_string(self):
        """Any truthy string is returned as-is."""
        assert get_default_current_date("tomorrow") == "tomorrow"

    def test_returns_today_when_none(self):
        """When None is passed the function returns today's date in YYYY-MM-DD format."""
        from datetime import datetime

        result = get_default_current_date(None)
        today = datetime.now().strftime("%Y-%m-%d")
        assert result == today

    def test_returns_today_when_empty_string(self):
        """An empty string is falsy so today's date is returned."""
        from datetime import datetime

        result = get_default_current_date("")
        today = datetime.now().strftime("%Y-%m-%d")
        assert result == today

    def test_result_format_is_yyyy_mm_dd(self):
        """The fallback value matches YYYY-MM-DD pattern."""
        import re

        result = get_default_current_date(None)
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", result)


class TestFormatDurationHuman:
    """Tests for format_duration_human."""

    def test_seconds_only(self):
        assert format_duration_human(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_duration_human(23 * 60 + 36) == "23m36s"

    def test_hours_minutes_seconds(self):
        assert format_duration_human(1 * 3600 + 24 * 60 + 30) == "1h24m30s"

    def test_hours_and_minutes_no_seconds(self):
        assert format_duration_human(2 * 3600 + 3 * 60) == "2h3m"

    def test_days_hours_minutes_seconds(self):
        # 1 day + 2 hours + 3 minutes + 4 seconds
        total = 86400 + 2 * 3600 + 3 * 60 + 4
        assert format_duration_human(total) == "1d2h3m4s"

    def test_days_only(self):
        assert format_duration_human(2 * 86400) == "2d"

    def test_zero_seconds(self):
        """Zero should return '0s' (the fallback branch)."""
        assert format_duration_human(0) == "0s"

    def test_float_is_truncated(self):
        """Float input is truncated to int before processing."""
        assert format_duration_human(61.9) == "1m1s"

    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (60, "1m"),
            (3600, "1h"),
            (86400, "1d"),
            (90, "1m30s"),
            (3661, "1h1m1s"),
        ],
    )
    def test_parametrized_durations(self, seconds, expected):
        assert format_duration_human(seconds) == expected


_ISO_Z_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class TestNowUtcIso:
    """Tests for now_utc_iso."""

    def test_format_matches_iso8601_with_z_suffix(self):
        result = now_utc_iso()
        assert _ISO_Z_PATTERN.match(result), result

    def test_returns_utc_time(self):
        fixed = datetime(2026, 4, 30, 12, 34, 56, tzinfo=timezone.utc)

        class _FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                assert tz is timezone.utc
                return fixed

        with patch("datus.utils.time_utils.datetime", _FakeDatetime):
            assert now_utc_iso() == "2026-04-30T12:34:56Z"


class TestToUtcIso:
    """Tests for to_utc_iso normalization."""

    def test_none_returns_none(self):
        assert to_utc_iso(None) is None

    def test_empty_string_returns_none(self):
        assert to_utc_iso("") is None
        assert to_utc_iso("   ") is None

    def test_unparseable_string_returns_none(self):
        assert to_utc_iso("not-a-timestamp") is None

    def test_sqlite_naive_string_treated_as_utc(self):
        # SQLite CURRENT_TIMESTAMP form: space separator, no offset.
        assert to_utc_iso("2026-04-30 12:34:56") == "2026-04-30T12:34:56Z"

    def test_iso_t_form_naive_is_utc(self):
        assert to_utc_iso("2026-04-30T12:34:56") == "2026-04-30T12:34:56Z"

    def test_iso_with_z_suffix(self):
        assert to_utc_iso("2026-04-30T12:34:56Z") == "2026-04-30T12:34:56Z"

    def test_iso_with_explicit_offset(self):
        assert to_utc_iso("2026-04-30T20:34:56+08:00") == "2026-04-30T12:34:56Z"

    def test_microseconds_are_truncated_to_seconds(self):
        assert to_utc_iso("2026-04-30T12:34:56.789012Z") == "2026-04-30T12:34:56Z"

    def test_unix_float_timestamp(self):
        # 1761835200.0 = 2025-10-30T14:40:00Z (verified via fromtimestamp(..., UTC))
        assert to_utc_iso(1761835200.0) == "2025-10-30T14:40:00Z"

    def test_unix_int_timestamp(self):
        assert to_utc_iso(1761835200) == "2025-10-30T14:40:00Z"

    def test_naive_datetime_treated_as_utc(self):
        dt = datetime(2026, 4, 30, 12, 34, 56)
        assert to_utc_iso(dt) == "2026-04-30T12:34:56Z"

    def test_aware_datetime_converted_to_utc(self):
        dt = datetime(2026, 4, 30, 20, 34, 56, tzinfo=timezone(timedelta(hours=8)))
        assert to_utc_iso(dt) == "2026-04-30T12:34:56Z"

    def test_already_utc_aware_datetime(self):
        dt = datetime(2026, 4, 30, 12, 34, 56, tzinfo=timezone.utc)
        assert to_utc_iso(dt) == "2026-04-30T12:34:56Z"

    def test_unsupported_type_returns_none(self):
        assert to_utc_iso([1, 2, 3]) is None
        assert to_utc_iso({"a": 1}) is None


class TestFormatLocalTime:
    """Tests for format_local_time (CLI display helper)."""

    def test_none_returns_empty(self):
        assert format_local_time(None) == ""

    def test_empty_string_returns_empty(self):
        assert format_local_time("") == ""

    def test_unparseable_returns_empty(self):
        assert format_local_time("not a date") == ""

    def test_utc_zulu_converted_to_local(self):
        # Force a known local timezone via patching astimezone result.
        # Easier: pass an aware datetime and assert the pre-conversion math
        # by comparing to manual astimezone() output.
        utc = datetime(2026, 4, 30, 12, 34, 56, tzinfo=timezone.utc)
        expected = utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        assert format_local_time("2026-04-30T12:34:56Z") == expected

    def test_sqlite_naive_string_converted_to_local(self):
        utc = datetime(2026, 4, 30, 12, 34, 56, tzinfo=timezone.utc)
        expected = utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        assert format_local_time("2026-04-30 12:34:56") == expected

    def test_custom_format(self):
        utc = datetime(2026, 4, 30, 12, 34, 56, tzinfo=timezone.utc)
        expected = utc.astimezone().strftime("%H:%M")
        assert format_local_time("2026-04-30T12:34:56Z", fmt="%H:%M") == expected


class TestGetDefaultCurrentDateExtended:
    """Tests for get_default_current_date (line 21)."""

    def test_returns_provided_date_when_not_none(self):
        assert get_default_current_date("2025-06-15") == "2025-06-15"

    def test_returns_provided_date_when_arbitrary_string(self):
        assert get_default_current_date("tomorrow") == "tomorrow"

    def test_returns_today_when_none(self):
        from datetime import datetime

        result = get_default_current_date(None)
        today = datetime.now().strftime("%Y-%m-%d")
        assert result == today

    def test_returns_today_when_empty_string(self):
        """Empty string is falsy so today's date is returned (line 21)."""
        from datetime import datetime

        result = get_default_current_date("")
        today = datetime.now().strftime("%Y-%m-%d")
        assert result == today

    def test_result_format_is_yyyy_mm_dd_when_none(self):
        result = get_default_current_date(None)
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", result)


class TestFormatDurationHumanExtended:
    """Additional tests covering line 35 (days branch)."""

    def test_days_only(self):
        assert format_duration_human(2 * 86400) == "2d"

    def test_days_hours_minutes_seconds(self):
        total = 86400 + 2 * 3600 + 3 * 60 + 4
        assert format_duration_human(total) == "1d2h3m4s"

    def test_zero_seconds(self):
        assert format_duration_human(0) == "0s"

    def test_float_truncated(self):
        assert format_duration_human(61.9) == "1m1s"

    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (60, "1m"),
            (3600, "1h"),
            (86400, "1d"),
            (90, "1m30s"),
            (3661, "1h1m1s"),
        ],
    )
    def test_parametrized_durations(self, seconds, expected):
        assert format_duration_human(seconds) == expected
