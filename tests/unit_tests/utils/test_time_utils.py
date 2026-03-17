# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/time_utils.py — CI tier, zero external deps."""

import re

import pytest

from datus.utils.time_utils import format_duration_human, get_default_current_date


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
