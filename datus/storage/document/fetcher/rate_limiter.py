# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Rate Limiter for HTTP Requests

Manages request frequency to avoid hitting API rate limits.
Supports different rate limits for different services (GitHub API, websites).
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit.

    Attributes:
        requests_per_hour: Maximum requests per hour
        min_interval: Minimum seconds between requests
        burst_size: Maximum burst of requests before throttling
    """

    requests_per_hour: int = 60
    min_interval: float = 0.5
    burst_size: int = 10


@dataclass
class RateLimitState:
    """Current state of rate limiting for a domain.

    Attributes:
        request_count: Number of requests made in current window
        window_start: Start of the current window
        last_request: Timestamp of last request
        remaining: Remaining requests (from API headers)
        reset_time: When the rate limit resets (from API headers)
    """

    request_count: int = 0
    window_start: float = field(default_factory=time.time)
    last_request: float = 0.0
    remaining: Optional[int] = None
    reset_time: Optional[float] = None


class RateLimiter:
    """Rate limiter for HTTP requests.

    Manages request frequency to avoid hitting rate limits.
    Supports per-domain configuration and dynamic updates from API headers.

    Example:
        >>> limiter = RateLimiter()
        >>> limiter.configure("api.github.com", RateLimitConfig(requests_per_hour=5000))
        >>> await limiter.wait("api.github.com")  # Wait if needed
        >>> # Make request...
        >>> limiter.update_from_headers("api.github.com", headers)  # Update from response
    """

    # Default configurations for known services
    DEFAULT_CONFIGS: Dict[str, RateLimitConfig] = {
        "api.github.com": RateLimitConfig(
            requests_per_hour=60,  # Unauthenticated default
            min_interval=0.1,
            burst_size=10,
        ),
        "github.com": RateLimitConfig(
            requests_per_hour=60,
            min_interval=0.5,
            burst_size=5,
        ),
        "default": RateLimitConfig(
            requests_per_hour=3600,  # 1 req/sec on average
            min_interval=0.5,
            burst_size=10,
        ),
    }

    def __init__(self):
        """Initialize the rate limiter."""
        self._configs: Dict[str, RateLimitConfig] = dict(self.DEFAULT_CONFIGS)
        self._states: Dict[str, RateLimitState] = {}
        self._lock = threading.Lock()

    def configure(self, domain: str, config: RateLimitConfig):
        """Configure rate limit for a domain.

        Args:
            domain: Domain name (e.g., "api.github.com")
            config: Rate limit configuration
        """
        with self._lock:
            self._configs[domain] = config
            logger.debug(
                f"Configured rate limit for {domain}: "
                f"{config.requests_per_hour}/hour, min_interval={config.min_interval}s"
            )

    def configure_github_authenticated(self, requests_per_hour: int = 5000):
        """Configure rate limit for authenticated GitHub API.

        Args:
            requests_per_hour: Requests per hour (default 5000 for authenticated)
        """
        self.configure(
            "api.github.com",
            RateLimitConfig(
                requests_per_hour=requests_per_hour,
                min_interval=0.05,
                burst_size=50,
            ),
        )

    def wait(self, domain: str) -> float:
        """Wait if necessary before making a request.

        Args:
            domain: Domain to make request to

        Returns:
            Seconds waited
        """
        with self._lock:
            config = self._get_config(domain)
            state = self._get_state(domain)
            wait_time = self._calculate_wait_time(config, state)
            # Reserve the slot under lock to prevent concurrent bypass
            state.last_request = time.time() + wait_time
            state.request_count += 1
            # Reset window if needed
            if time.time() - state.window_start >= 3600:
                state.window_start = time.time()
                state.request_count = 1

        if wait_time > 0:
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
            time.sleep(wait_time)

        return wait_time

    async def wait_async(self, domain: str) -> float:
        """Async version of wait.

        Args:
            domain: Domain to make request to

        Returns:
            Seconds waited
        """
        import asyncio

        with self._lock:
            config = self._get_config(domain)
            state = self._get_state(domain)
            wait_time = self._calculate_wait_time(config, state)
            # Reserve the slot under lock to prevent concurrent bypass
            state.last_request = time.time() + wait_time
            state.request_count += 1
            # Reset window if needed
            if time.time() - state.window_start >= 3600:
                state.window_start = time.time()
                state.request_count = 1

        if wait_time > 0:
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
            await asyncio.sleep(wait_time)

        return wait_time

    def update_from_headers(self, domain: str, headers: Dict[str, str]):
        """Update rate limit state from response headers.

        Supports GitHub API headers:
        - X-RateLimit-Remaining
        - X-RateLimit-Reset

        Args:
            domain: Domain the response came from
            headers: Response headers
        """
        with self._lock:
            state = self._get_state(domain)

            # GitHub API headers
            if "x-ratelimit-remaining" in headers:
                try:
                    state.remaining = int(headers["x-ratelimit-remaining"])
                except (ValueError, TypeError):
                    pass

            if "x-ratelimit-reset" in headers:
                try:
                    state.reset_time = float(headers["x-ratelimit-reset"])
                except (ValueError, TypeError):
                    pass

            # Log if running low
            if state.remaining is not None and state.remaining < 100:
                reset_in = ""
                if state.reset_time:
                    reset_in = f", resets in {int(state.reset_time - time.time())}s"
                logger.warning(f"Rate limit running low for {domain}: " f"{state.remaining} remaining{reset_in}")

    def get_remaining(self, domain: str) -> Optional[int]:
        """Get remaining requests for a domain.

        Args:
            domain: Domain to check

        Returns:
            Remaining requests or None if unknown
        """
        with self._lock:
            state = self._states.get(domain)
            return state.remaining if state else None

    def _get_config(self, domain: str) -> RateLimitConfig:
        """Get configuration for a domain."""
        return self._configs.get(domain, self._configs["default"])

    def _get_state(self, domain: str) -> RateLimitState:
        """Get or create state for a domain."""
        if domain not in self._states:
            self._states[domain] = RateLimitState()
        return self._states[domain]

    def _calculate_wait_time(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        max_wait: float = 60.0,
    ) -> float:
        """Calculate how long to wait before the next request.

        Args:
            config: Rate limit configuration
            state: Current rate limit state
            max_wait: Maximum wait time in seconds (default 60s)

        Returns:
            Seconds to wait (0 if no wait needed, capped at max_wait)
        """
        now = time.time()
        wait_time = 0.0

        # Check minimum interval
        if state.last_request > 0:
            elapsed = now - state.last_request
            if elapsed < config.min_interval:
                wait_time = config.min_interval - elapsed

        # Check hourly rate limit
        if state.request_count >= config.requests_per_hour:
            # Wait until window resets, but cap the wait time
            window_elapsed = now - state.window_start
            if window_elapsed < 3600:
                wait_time = max(wait_time, min(3600 - window_elapsed, max_wait))

        # Check API-reported limits
        if state.remaining is not None and state.remaining <= 0:
            if state.reset_time and state.reset_time > now:
                wait_time = max(wait_time, min(state.reset_time - now + 1, max_wait))

        # Check burst limit (backoff if making too many rapid requests)
        if config.burst_size > 0 and state.request_count > 0 and state.request_count % config.burst_size == 0:
            # Add small delay after each burst
            wait_time = max(wait_time, 1.0)

        # Cap total wait time
        return min(wait_time, max_wait)


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance.

    Returns:
        Global RateLimiter instance
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter
