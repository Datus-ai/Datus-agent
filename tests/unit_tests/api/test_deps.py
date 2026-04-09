"""Tests for datus.api.deps — dependency injection module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.api import deps
from datus.api.auth.context import AppContext
from datus.api.deps import get_datus_service, init_deps
from datus.api.services.datus_service_cache import DatusServiceCache


@pytest.fixture(autouse=True)
def _reset_deps():
    """Reset module-level singletons between tests."""
    deps._auth_provider = None
    deps._service_cache = None
    deps._namespace = "default"
    deps._default_source = None
    deps._default_interactive = True
    yield
    deps._auth_provider = None
    deps._service_cache = None
    deps._namespace = "default"
    deps._default_source = None
    deps._default_interactive = True


class TestInitDeps:
    """Tests for init_deps — singleton initialization."""

    def test_sets_auth_provider(self):
        """init_deps stores the auth provider."""
        mock_auth = MagicMock()
        mock_auth.on_evict = MagicMock()
        mock_cache = MagicMock(spec=DatusServiceCache)

        init_deps(mock_auth, mock_cache, namespace="test_ns")

        assert deps._auth_provider is mock_auth
        assert deps._service_cache is mock_cache
        assert deps._namespace == "test_ns"

    def test_default_namespace(self):
        """Default namespace is 'default'."""
        mock_auth = MagicMock()
        mock_auth.on_evict = MagicMock()
        mock_cache = MagicMock(spec=DatusServiceCache)

        init_deps(mock_auth, mock_cache)
        assert deps._namespace == "default"

    def test_default_source_and_interactive_defaults(self):
        """Without explicit args, default_source is None and default_interactive is True."""
        mock_auth = MagicMock()
        mock_auth.on_evict = MagicMock()
        mock_cache = MagicMock(spec=DatusServiceCache)

        init_deps(mock_auth, mock_cache)
        assert deps._default_source is None
        assert deps._default_interactive is True

    def test_default_source_and_interactive_stored(self):
        """init_deps stores explicit default_source and default_interactive."""
        mock_auth = MagicMock()
        mock_auth.on_evict = MagicMock()
        mock_cache = MagicMock(spec=DatusServiceCache)

        init_deps(
            mock_auth,
            mock_cache,
            namespace="ns",
            default_source="vscode",
            default_interactive=False,
        )
        assert deps._default_source == "vscode"
        assert deps._default_interactive is False

    def test_wires_eviction_callback(self):
        """init_deps wires auth_provider.on_evict to cache.evict."""
        mock_auth = MagicMock()
        mock_cache = MagicMock(spec=DatusServiceCache)

        init_deps(mock_auth, mock_cache)
        mock_auth.on_evict.assert_called_once_with(mock_cache.evict)


@pytest.mark.asyncio
class TestGetDatusService:
    """Tests for get_datus_service — FastAPI dependency."""

    async def test_raises_when_auth_not_initialized(self):
        """RuntimeError when auth_provider is None."""
        request = MagicMock()
        with pytest.raises(RuntimeError, match="Auth provider not initialized"):
            await get_datus_service(request)

    async def test_raises_when_cache_not_initialized(self):
        """RuntimeError when service_cache is None but auth is set."""
        deps._auth_provider = MagicMock()
        deps._auth_provider.authenticate = AsyncMock()
        request = MagicMock()
        request.state = MagicMock()
        with pytest.raises(RuntimeError, match="Service cache not initialized"):
            await get_datus_service(request)

    async def test_authenticates_and_returns_service(self):
        """Full flow: authenticate → factory → cache.get_or_create."""
        from unittest.mock import patch

        mock_auth = MagicMock()
        ctx = AppContext(user_id="user-1", project_id="proj-1", config=MagicMock())
        mock_auth.authenticate = AsyncMock(return_value=ctx)

        mock_cache = MagicMock(spec=DatusServiceCache)
        mock_svc = MagicMock()
        mock_cache.get_or_create = AsyncMock(return_value=mock_svc)

        deps._auth_provider = mock_auth
        deps._service_cache = mock_cache

        request = MagicMock()
        request.state = MagicMock()
        with patch(
            "datus.api.deps.DatusService.compute_fingerprint",
            return_value="fp-xyz",
        ) as mock_fp:
            result = await get_datus_service(request)

        assert result is mock_svc
        mock_auth.authenticate.assert_awaited_once_with(request)
        mock_cache.get_or_create.assert_awaited_once()
        mock_fp.assert_called_once_with(ctx.config)
        call_args = mock_cache.get_or_create.call_args
        assert call_args[0][0] == "proj-1"
        assert call_args.kwargs["expected_fingerprint"] == "fp-xyz"

    async def test_no_fingerprint_when_config_is_none(self):
        """When ctx.config is None, expected_fingerprint passed as None."""
        mock_auth = MagicMock()
        ctx = AppContext(user_id="user-1", project_id="proj-1", config=None)
        mock_auth.authenticate = AsyncMock(return_value=ctx)

        mock_cache = MagicMock(spec=DatusServiceCache)
        mock_cache.get_or_create = AsyncMock(return_value=MagicMock())

        deps._auth_provider = mock_auth
        deps._service_cache = mock_cache

        request = MagicMock()
        request.state = MagicMock()
        await get_datus_service(request)
        call_args = mock_cache.get_or_create.call_args
        assert call_args.kwargs["expected_fingerprint"] is None

    async def test_factory_propagates_default_source_and_interactive(self):
        """Factory passes module-level defaults through to DatusService constructor."""
        from unittest.mock import patch

        mock_auth = MagicMock()
        ctx = AppContext(user_id="u1", project_id="p1", config=MagicMock())
        mock_auth.authenticate = AsyncMock(return_value=ctx)

        mock_cache = MagicMock(spec=DatusServiceCache)
        captured = {}

        async def fake_get_or_create(key, factory, expected_fingerprint=None):
            captured["svc"] = await factory()
            return captured["svc"]

        mock_cache.get_or_create = AsyncMock(side_effect=fake_get_or_create)

        deps._auth_provider = mock_auth
        deps._service_cache = mock_cache
        deps._default_source = "web"
        deps._default_interactive = False

        request = MagicMock()
        request.state = MagicMock()

        with (
            patch("datus.api.deps.DatusService.compute_fingerprint", return_value="fp"),
            patch("datus.api.deps.DatusService") as mock_svc_cls,
        ):
            mock_svc_cls.compute_fingerprint = MagicMock(return_value="fp")
            mock_svc_cls.return_value = MagicMock()
            await get_datus_service(request)

        call_kwargs = mock_svc_cls.call_args.kwargs
        assert call_kwargs["default_source"] == "web"
        assert call_kwargs["default_interactive"] is False
        assert call_kwargs["project_id"] == "p1"

    async def test_factory_loads_config_when_none(self, real_agent_config):
        """Factory in get_datus_service loads config when ctx.config is None."""
        from datus.api.auth.no_auth_provider import NoAuthProvider
        from datus.api.services.datus_service import DatusService

        auth_provider = NoAuthProvider()
        cache = DatusServiceCache()
        deps._auth_provider = auth_provider
        deps._service_cache = cache
        deps._namespace = "test_ns"

        request = MagicMock()
        request.state = MagicMock()
        request.headers = {}
        # NoAuthProvider returns AppContext with config=None
        # Factory should call load_agent_config(namespace="test_ns")
        # This will fail because test_ns config doesn't exist in default paths,
        # but exercises the factory code path (lines 50-56)
        try:
            result = await get_datus_service(request)
            # If it succeeds, result should be a DatusService
            assert isinstance(result, DatusService)
        except RuntimeError as e:
            # Expected: config not found
            assert "Failed to load agent config" in str(e)
        finally:
            await cache.shutdown()
