"""Tests for datus.api.auth.no_auth_provider — bypass authentication."""

from unittest.mock import MagicMock

import pytest

from datus.api.auth.context import AppContext
from datus.api.auth.no_auth_provider import NoAuthProvider


class TestNoAuthProviderInit:
    """Tests for NoAuthProvider construction."""

    def test_default_namespace(self):
        """Default namespace is 'default'."""
        provider = NoAuthProvider()
        assert provider._namespace == "default"

    def test_custom_namespace(self):
        """Custom namespace is respected."""
        provider = NoAuthProvider(namespace="production")
        assert provider._namespace == "production"

    def test_evict_callbacks_empty_on_init(self):
        """Evict callbacks list is empty after init."""
        provider = NoAuthProvider()
        assert provider._evict_callbacks == []


@pytest.mark.asyncio
class TestNoAuthProviderAuthenticate:
    """Tests for authenticate — bypass auth returning AppContext."""

    async def test_returns_app_context(self):
        """authenticate returns AppContext with correct fields."""
        provider = NoAuthProvider(namespace="test_ns")
        request = MagicMock()
        ctx = await provider.authenticate(request)

        assert isinstance(ctx, AppContext)
        assert ctx.user_id == "anonymous"
        assert ctx.project_id == "test_ns"
        assert ctx.config is None

    async def test_default_namespace_in_context(self):
        """Default namespace provider uses 'default' as project_id."""
        provider = NoAuthProvider()
        ctx = await provider.authenticate(MagicMock())
        assert ctx.project_id == "default"


class TestNoAuthProviderOnEvict:
    """Tests for on_evict — callback registration."""

    def test_registers_callback(self):
        """on_evict appends callback to the list."""
        provider = NoAuthProvider()
        callback = MagicMock()
        provider.on_evict(callback)
        assert len(provider._evict_callbacks) == 1
        assert provider._evict_callbacks[0] is callback

    def test_registers_multiple_callbacks(self):
        """Multiple on_evict calls append all callbacks."""
        provider = NoAuthProvider()
        cb1, cb2 = MagicMock(), MagicMock()
        provider.on_evict(cb1)
        provider.on_evict(cb2)
        assert len(provider._evict_callbacks) == 2
