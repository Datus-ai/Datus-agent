"""Tests for the default header-based request context provider."""

from unittest.mock import MagicMock

import pytest

from datus.api.auth.context import AppContext
from datus.api.auth.header_context_provider import HeaderContextProvider
from datus.api.constants import HEADER_POLICY_CONTEXT, HEADER_USER_ID
from datus.utils.exceptions import DatusException


def _make_request(headers: dict | None = None) -> MagicMock:
    request = MagicMock()
    request.headers = headers or {}
    return request


class TestHeaderContextProviderInit:
    def test_init_is_stateless(self):
        provider = HeaderContextProvider()
        assert provider._evict_callbacks == []


@pytest.mark.asyncio
class TestHeaderContextProviderAuthenticate:
    async def test_no_header_returns_none_user(self):
        """Missing header → user_id is None, project_id is None."""
        provider = HeaderContextProvider()
        ctx = await provider.authenticate(_make_request({}))
        assert isinstance(ctx, AppContext)
        assert ctx.user_id is None
        assert ctx.project_id is None
        assert ctx.config is None
        assert ctx.policy_context == {}

    async def test_valid_header_populates_user_id(self):
        """Valid header → user_id reflects the header value."""
        provider = HeaderContextProvider()
        ctx = await provider.authenticate(_make_request({HEADER_USER_ID: "alice"}))
        assert ctx.user_id == "alice"
        assert ctx.project_id is None
        assert ctx.policy_context == {}

    async def test_whitespace_header_treated_as_missing(self):
        provider = HeaderContextProvider()
        ctx = await provider.authenticate(_make_request({HEADER_USER_ID: "   "}))
        assert ctx.user_id is None
        assert ctx.policy_context == {}

    async def test_policy_context_header_populates_request_context(self):
        provider = HeaderContextProvider()
        ctx = await provider.authenticate(
            _make_request(
                {HEADER_POLICY_CONTEXT: ('{"row_filter":{"access_mode":"scoped","market_codes":["MKT300","MKT301"]}}')}
            )
        )
        assert ctx.user_id is None
        assert ctx.policy_context == {"row_filter": {"access_mode": "scoped", "market_codes": ["MKT300", "MKT301"]}}

    async def test_blank_policy_context_header_returns_empty_context(self):
        provider = HeaderContextProvider()
        ctx = await provider.authenticate(_make_request({HEADER_POLICY_CONTEXT: "   "}))
        assert ctx.policy_context == {}

    async def test_user_id_header_is_independent_from_policy_context(self):
        provider = HeaderContextProvider()
        ctx = await provider.authenticate(
            _make_request({HEADER_USER_ID: "alice", HEADER_POLICY_CONTEXT: '{"row_filter":{"access_mode":"denied"}}'})
        )
        assert ctx.user_id == "alice"
        assert ctx.policy_context == {"row_filter": {"access_mode": "denied"}}

    async def test_invalid_header_raises(self):
        """Header with disallowed characters → DatusException."""
        provider = HeaderContextProvider()
        with pytest.raises(DatusException):
            await provider.authenticate(_make_request({HEADER_USER_ID: "bad user!"}))

    async def test_invalid_policy_context_header_raises(self):
        provider = HeaderContextProvider()
        with pytest.raises(DatusException):
            await provider.authenticate(_make_request({HEADER_POLICY_CONTEXT: "not-json"}))

        with pytest.raises(DatusException):
            await provider.authenticate(_make_request({HEADER_POLICY_CONTEXT: '["MKT300"]'}))


class TestHeaderContextProviderOnEvict:
    def test_registers_callback(self):
        provider = HeaderContextProvider()
        callback = MagicMock()
        provider.on_evict(callback)
        assert provider._evict_callbacks == [callback]

    def test_registers_multiple_callbacks(self):
        provider = HeaderContextProvider()
        cb1, cb2 = MagicMock(), MagicMock()
        provider.on_evict(cb1)
        provider.on_evict(cb2)
        assert provider._evict_callbacks == [cb1, cb2]
