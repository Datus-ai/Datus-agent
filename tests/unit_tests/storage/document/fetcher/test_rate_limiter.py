# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.document.fetcher.rate_limiter."""

import threading
from unittest.mock import patch

import pytest

from datus.storage.document.fetcher.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitState,
    get_rate_limiter,
)

# ---------------------------------------------------------------------------
# RateLimitConfig
# ---------------------------------------------------------------------------


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_config_defaults(self):
        """Default values should be sensible for a general HTTP service."""
        cfg = RateLimitConfig()
        assert cfg.requests_per_hour == 60
        assert cfg.min_interval == 0.5
        assert cfg.burst_size == 10

    def test_config_custom_values(self):
        """Custom values should override defaults."""
        cfg = RateLimitConfig(requests_per_hour=5000, min_interval=0.05, burst_size=50)
        assert cfg.requests_per_hour == 5000
        assert cfg.min_interval == 0.05
        assert cfg.burst_size == 50


# ---------------------------------------------------------------------------
# RateLimitState
# ---------------------------------------------------------------------------


class TestRateLimitState:
    """Tests for RateLimitState dataclass."""

    def test_state_defaults(self):
        """Default state should reflect zero requests with no API info."""
        state = RateLimitState()
        assert state.request_count == 0
        assert state.last_request == 0.0
        assert state.remaining is None
        assert state.reset_time is None
        assert state.window_start > 0  # set via time.time()


# ---------------------------------------------------------------------------
# RateLimiter — configure
# ---------------------------------------------------------------------------


class TestRateLimiterConfigure:
    """Tests for RateLimiter.configure and configure_github_authenticated."""

    def test_configure_custom_domain(self):
        """configure() should store a custom config for the given domain."""
        limiter = RateLimiter()
        custom = RateLimitConfig(requests_per_hour=100, min_interval=1.0, burst_size=5)
        limiter.configure("example.com", custom)
        cfg = limiter._get_config("example.com")
        assert cfg.requests_per_hour == 100
        assert cfg.min_interval == 1.0
        assert cfg.burst_size == 5

    def test_configure_github_authenticated(self):
        """configure_github_authenticated() should set high rate limits for api.github.com."""
        limiter = RateLimiter()
        limiter.configure_github_authenticated()
        cfg = limiter._get_config("api.github.com")
        assert cfg.requests_per_hour == 5000
        assert cfg.min_interval == 0.05
        assert cfg.burst_size == 50

    def test_configure_github_authenticated_custom_rate(self):
        """configure_github_authenticated() should accept a custom rate."""
        limiter = RateLimiter()
        limiter.configure_github_authenticated(requests_per_hour=10000)
        cfg = limiter._get_config("api.github.com")
        assert cfg.requests_per_hour == 10000

    def test_unknown_domain_gets_default_config(self):
        """An unconfigured domain should fall back to the 'default' config."""
        limiter = RateLimiter()
        cfg = limiter._get_config("unknown.example.org")
        assert cfg.requests_per_hour == 3600
        assert cfg.min_interval == 0.5


# ---------------------------------------------------------------------------
# RateLimiter — _calculate_wait_time
# ---------------------------------------------------------------------------


class TestCalculateWaitTime:
    """Tests for RateLimiter._calculate_wait_time with controlled time."""

    def test_first_request_no_wait(self):
        """The very first request should not need to wait."""
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=60, min_interval=0.5, burst_size=10)
        state = RateLimitState(request_count=0, last_request=0.0)
        wait = limiter._calculate_wait_time(config, state)
        assert wait == 0.0

    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_min_interval_enforced(self, mock_time):
        """Should wait until min_interval has elapsed since last request."""
        mock_time.return_value = 100.0
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=3600, min_interval=1.0, burst_size=10)
        state = RateLimitState(
            request_count=1,
            last_request=99.5,  # 0.5s ago
            window_start=90.0,
        )
        wait = limiter._calculate_wait_time(config, state)
        assert pytest.approx(wait, abs=0.01) == 0.5

    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_no_wait_when_interval_elapsed(self, mock_time):
        """No wait needed when min_interval has fully elapsed."""
        mock_time.return_value = 102.0
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=3600, min_interval=1.0, burst_size=10)
        state = RateLimitState(
            request_count=1,
            last_request=100.0,  # 2s ago, well past min_interval=1.0
            window_start=90.0,
        )
        wait = limiter._calculate_wait_time(config, state)
        assert wait == 0.0

    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_window_reset_forces_wait(self, mock_time):
        """When hourly limit is hit and window is not over, should wait."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=10, min_interval=0.1, burst_size=100)
        state = RateLimitState(
            request_count=10,  # hit hourly limit
            last_request=999.9,
            window_start=800.0,  # 200s into window, 3400s remaining
        )
        wait = limiter._calculate_wait_time(config, state)
        # Should be capped at max_wait (60.0) since 3400 > 60
        assert wait == 60.0

    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_burst_limit_adds_delay(self, mock_time):
        """After a burst, a small delay should be added."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=3600, min_interval=0.0, burst_size=5)
        state = RateLimitState(
            request_count=10,  # 10 % 5 == 0 -> burst boundary
            last_request=0.0,
            window_start=900.0,
        )
        wait = limiter._calculate_wait_time(config, state)
        assert wait >= 1.0

    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_api_remaining_zero_forces_wait(self, mock_time):
        """When API reports remaining=0, should wait until reset_time."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=5000, min_interval=0.1, burst_size=50)
        state = RateLimitState(
            request_count=1,
            last_request=0.0,
            window_start=900.0,
            remaining=0,
            reset_time=1010.0,  # resets in 10s
        )
        wait = limiter._calculate_wait_time(config, state)
        # reset_time - now + 1 = 10 + 1 = 11
        assert pytest.approx(wait, abs=0.1) == 11.0

    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_max_wait_cap(self, mock_time):
        """Wait time should never exceed max_wait."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=5000, min_interval=0.1, burst_size=50)
        state = RateLimitState(
            request_count=1,
            last_request=0.0,
            window_start=900.0,
            remaining=0,
            reset_time=2000.0,  # 1000s from now, well beyond max_wait
        )
        wait = limiter._calculate_wait_time(config, state, max_wait=30.0)
        assert wait == 30.0

    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_no_burst_delay_when_burst_size_zero(self, mock_time):
        """burst_size=0 should skip burst logic entirely."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=3600, min_interval=0.0, burst_size=0)
        state = RateLimitState(
            request_count=100,
            last_request=0.0,
            window_start=900.0,
        )
        wait = limiter._calculate_wait_time(config, state)
        assert wait == 0.0


# ---------------------------------------------------------------------------
# RateLimiter — update_from_headers
# ---------------------------------------------------------------------------


class TestUpdateFromHeaders:
    """Tests for RateLimiter.update_from_headers."""

    def test_update_github_headers(self):
        """Should parse x-ratelimit-remaining and x-ratelimit-reset from headers."""
        limiter = RateLimiter()
        headers = {
            "x-ratelimit-remaining": "4500",
            "x-ratelimit-reset": "1700000000",
        }
        limiter.update_from_headers("api.github.com", headers)
        state = limiter._states["api.github.com"]
        assert state.remaining == 4500
        assert state.reset_time == 1700000000.0

    def test_update_partial_headers(self):
        """Should handle headers with only remaining, no reset."""
        limiter = RateLimiter()
        headers = {"x-ratelimit-remaining": "100"}
        limiter.update_from_headers("api.github.com", headers)
        state = limiter._states["api.github.com"]
        assert state.remaining == 100
        assert state.reset_time is None

    def test_update_invalid_header_values(self):
        """Invalid header values should be silently ignored."""
        limiter = RateLimiter()
        headers = {
            "x-ratelimit-remaining": "not-a-number",
            "x-ratelimit-reset": "invalid",
        }
        limiter.update_from_headers("api.github.com", headers)
        state = limiter._states["api.github.com"]
        assert state.remaining is None
        assert state.reset_time is None

    def test_update_empty_headers(self):
        """Empty headers should not modify state adversely."""
        limiter = RateLimiter()
        limiter.update_from_headers("api.github.com", {})
        state = limiter._states["api.github.com"]
        assert state.remaining is None
        assert state.reset_time is None


# ---------------------------------------------------------------------------
# RateLimiter — get_remaining
# ---------------------------------------------------------------------------


class TestGetRemaining:
    """Tests for RateLimiter.get_remaining."""

    def test_get_remaining_unknown_domain(self):
        """Should return None for a domain with no state."""
        limiter = RateLimiter()
        assert limiter.get_remaining("unknown.example.com") is None

    def test_get_remaining_after_header_update(self):
        """Should return the value set from headers."""
        limiter = RateLimiter()
        limiter.update_from_headers("api.github.com", {"x-ratelimit-remaining": "42"})
        assert limiter.get_remaining("api.github.com") == 42


# ---------------------------------------------------------------------------
# RateLimiter — wait (mocking time.sleep to avoid real sleep)
# ---------------------------------------------------------------------------


class TestWait:
    """Tests for RateLimiter.wait, mocking time.sleep to avoid actual delays."""

    @patch("datus.storage.document.fetcher.rate_limiter.time.sleep")
    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_wait_first_request_no_sleep(self, mock_time, mock_sleep):
        """First request should return 0 and not call sleep."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter()
        waited = limiter.wait("example.com")
        assert waited == 0.0
        mock_sleep.assert_not_called()

    @patch("datus.storage.document.fetcher.rate_limiter.time.sleep")
    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_wait_increments_request_count(self, mock_time, mock_sleep):
        """Each call to wait() should increment the request count."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter()
        limiter.wait("example.com")
        limiter.wait("example.com")
        state = limiter._states["example.com"]
        assert state.request_count == 2

    @patch("datus.storage.document.fetcher.rate_limiter.time.sleep")
    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_wait_calls_sleep_when_needed(self, mock_time, mock_sleep):
        """When wait_time > 0, time.sleep should be called."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter()
        # Pre-populate state to force a wait (last request is very recent)
        limiter._states["example.com"] = RateLimitState(
            request_count=1,
            last_request=1000.0,  # just now
            window_start=900.0,
        )
        waited = limiter.wait("example.com")
        assert waited > 0
        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# RateLimiter — thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Basic concurrent access test for RateLimiter."""

    @patch("datus.storage.document.fetcher.rate_limiter.time.sleep")
    @patch("datus.storage.document.fetcher.rate_limiter.time.time")
    def test_concurrent_wait_calls_do_not_crash(self, mock_time, mock_sleep):
        """Multiple threads calling wait() concurrently should not raise exceptions."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter()
        errors = []

        def worker():
            try:
                for _ in range(10):
                    limiter.wait("api.github.com")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert errors == [], f"Concurrent access errors: {errors}"


# ---------------------------------------------------------------------------
# get_rate_limiter (global singleton)
# ---------------------------------------------------------------------------


class TestGetRateLimiter:
    """Tests for the global get_rate_limiter function."""

    def test_get_rate_limiter_returns_instance(self):
        """get_rate_limiter should return a RateLimiter instance."""
        limiter = get_rate_limiter()
        assert isinstance(limiter, RateLimiter)

    def test_get_rate_limiter_singleton(self):
        """get_rate_limiter should return the same instance on repeated calls."""
        a = get_rate_limiter()
        b = get_rate_limiter()
        assert a is b
