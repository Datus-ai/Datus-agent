"""Tests for datus.api.auth.no_auth_provider — header-based identification."""

from unittest.mock import MagicMock

import pytest

from datus.api.auth.context import AppContext
from datus.api.auth.no_auth_provider import NoAuthProvider
from datus.api.constants import HEADER_USER_ID
from datus.utils.exceptions import DatusException


def _make_request(headers: dict | None = None) -> MagicMock:
    request = MagicMock()
    request.headers = headers or {}
    return request


class TestNoAuthProviderInit:
    def test_init_is_stateless(self):
        provider = NoAuthProvider()
        assert provider._evict_callbacks == []


@pytest.mark.asyncio
class TestNoAuthProviderAuthenticate:
    async def test_no_header_returns_none_user(self):
        """Missing header → user_id is None, project_id is None."""
        provider = NoAuthProvider()
        ctx = await provider.authenticate(_make_request({}))
        assert isinstance(ctx, AppContext)
        assert ctx.user_id is None
        assert ctx.project_id is None
        assert ctx.config is None

    async def test_valid_header_populates_user_id(self):
        """Valid header → user_id reflects the header value."""
        provider = NoAuthProvider()
        ctx = await provider.authenticate(_make_request({HEADER_USER_ID: "alice"}))
        assert ctx.user_id == "alice"
        assert ctx.project_id is None

    async def test_whitespace_header_treated_as_missing(self):
        provider = NoAuthProvider()
        ctx = await provider.authenticate(_make_request({HEADER_USER_ID: "   "}))
        assert ctx.user_id is None

    async def test_invalid_header_raises(self):
        """Header with disallowed characters → DatusException."""
        provider = NoAuthProvider()
        with pytest.raises(DatusException):
            await provider.authenticate(_make_request({HEADER_USER_ID: "bad user!"}))


class TestNoAuthProviderOnEvict:
    def test_registers_callback(self):
        provider = NoAuthProvider()
        callback = MagicMock()
        provider.on_evict(callback)
        assert provider._evict_callbacks == [callback]

    def test_registers_multiple_callbacks(self):
        provider = NoAuthProvider()
        cb1, cb2 = MagicMock(), MagicMock()
        provider.on_evict(cb1)
        provider.on_evict(cb2)
        assert provider._evict_callbacks == [cb1, cb2]
