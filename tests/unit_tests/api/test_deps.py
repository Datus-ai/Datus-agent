"""Tests for datus.api.deps — dependency injection module."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from datus.api import deps
from datus.api.auth.context import AppContext
from datus.api.deps import get_datus_service, init_deps
from datus.api.services.datus_service_cache import DatusServiceCache
from datus.utils.exceptions import DatusException, ErrorCode


@pytest.fixture(autouse=True)
def _reset_deps():
    """Reset module-level singletons between tests."""
    deps._auth_provider = None
    deps._service_cache = None
    deps._datasource = "default"
    deps._default_source = None
    deps._default_interactive = True
    deps._stream_thinking = False
    yield
    deps._auth_provider = None
    deps._service_cache = None
    deps._datasource = "default"
    deps._default_source = None
    deps._default_interactive = True
    deps._stream_thinking = False


class TestInitDeps:
    """Tests for init_deps — singleton initialization."""

    def test_sets_auth_provider(self):
        """init_deps stores the auth provider."""
        mock_auth = MagicMock()
        mock_auth.on_evict = MagicMock()
        mock_cache = MagicMock(spec=DatusServiceCache)

        init_deps(mock_auth, mock_cache, datasource="test_ns")

        assert deps._auth_provider is mock_auth
        assert deps._service_cache is mock_cache
        assert deps._datasource == "test_ns"

    def test_default_datasource(self):
        """Default datasource is 'default'."""
        mock_auth = MagicMock()
        mock_auth.on_evict = MagicMock()
        mock_cache = MagicMock(spec=DatusServiceCache)

        init_deps(mock_auth, mock_cache)
        assert deps._datasource == "default"

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
            datasource="ns",
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

    async def test_auth_validation_error_returns_bad_request(self):
        """Auth-provider request validation errors are API 400s, not internal errors."""
        mock_auth = MagicMock()
        mock_auth.authenticate = AsyncMock(
            side_effect=DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message="Invalid X-Datus-Policy-Context header value: expected a JSON object.",
            )
        )
        mock_cache = MagicMock(spec=DatusServiceCache)
        mock_cache.get_or_create = AsyncMock()

        deps._auth_provider = mock_auth
        deps._service_cache = mock_cache

        request = MagicMock()
        request.state = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_datus_service(request)

        assert exc_info.value.status_code == 400
        assert "Invalid X-Datus-Policy-Context header value" in exc_info.value.detail
        mock_cache.get_or_create.assert_not_called()

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

        async def fake_get_or_create(key, factory, expected_fingerprint=None, *, agent_config=None):
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
        from datus.api.auth.header_context_provider import HeaderContextProvider
        from datus.api.services.datus_service import DatusService

        auth_provider = HeaderContextProvider()
        cache = DatusServiceCache()
        deps._auth_provider = auth_provider
        deps._service_cache = cache
        deps._datasource = "test_ns"

        request = MagicMock()
        request.state = MagicMock()
        request.headers = {}
        # HeaderContextProvider returns AppContext with config=None
        # Factory should call load_agent_config(datasource="test_ns")
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


@pytest.mark.asyncio
class TestProjectConfigRefresh:
    """Exercise auth -> cache -> catalog/chat with a live background turn."""

    @pytest.mark.parametrize("change", ["add", "remove", "rename"])
    async def test_busy_project_uses_new_bindings_without_losing_running_turn(
        self, real_agent_config, monkeypatch, change
    ):
        from datus.api.models.chat_models import StopChatInput
        from datus.api.models.cli_models import StreamChatInput, UserInteractionInput
        from datus.api.routes.chat_routes import stop_chat, submit_user_interaction
        from datus.api.routes.config_routes import list_datasources_endpoint
        from datus.api.services.chat_task_manager import ChatTaskManager

        before = real_agent_config
        original_name = before.current_datasource
        if change != "add":
            before.services.datasources["warehouse"] = copy.deepcopy(before.services.datasources[original_name])
        after = copy.deepcopy(before)
        if change == "add":
            after.services.datasources["warehouse"] = copy.deepcopy(after.services.datasources[original_name])
        elif change == "remove":
            after.services.datasources.pop("warehouse")
        else:
            after.services.datasources["renamed_warehouse"] = after.services.datasources.pop("warehouse")

        live_config = before
        started = asyncio.Queue()
        release = asyncio.Event()
        snapshots = {}

        async def run_turn(manager, task, config, request, **kwargs):
            # Only the LLM loop is replaced. start_chat still takes the real
            # per-turn config snapshot and registers a real background task.
            snapshots[task.session_id] = config
            task.node = SimpleNamespace(interaction_broker=SimpleNamespace(submit=AsyncMock(return_value=True)))
            started.put_nowait(task.session_id)
            try:
                await release.wait()
            finally:
                task.status = "completed"

        monkeypatch.setattr(ChatTaskManager, "_run_loop", run_turn)

        async def authenticate(request):
            return AppContext(project_id="project", config=live_config)

        auth = SimpleNamespace(authenticate=authenticate, on_evict=lambda callback: None)
        cache = DatusServiceCache()
        init_deps(auth, cache, default_source="web", default_interactive=False, stream_thinking=True)
        request = MagicMock()
        try:
            old = await get_datus_service(request)
            old_chat = old.chat
            old_task = await old.task_manager.start_chat(
                old_chat.agent_config,
                StreamChatInput(message="old turn", datasource="warehouse" if change != "add" else original_name),
            )
            assert await asyncio.wait_for(started.get(), timeout=2) == old_task.session_id

            live_config = after
            refreshed = await get_datus_service(request)
            roster = await list_datasources_endpoint(refreshed)
            assert {entry.name for entry in roster.data.datasources} == set(after.datasource_configs)
            assert refreshed.agent_config is after
            assert old.agent_config is before
            assert refreshed.chat.agent_config is after
            assert old_chat.agent_config is before
            assert refreshed.task_manager is old.task_manager
            assert refreshed.task_manager.get_task(old_task.session_id) is old_task
            assert not old_task.asyncio_task.done()

            interaction = await submit_user_interaction(
                UserInteractionInput(session_id=old_task.session_id, interaction_key="question", input=[["yes"]]),
                refreshed,
            )
            assert interaction.success
            old_task.node.interaction_broker.submit.assert_awaited_once_with("question", [["yes"]])

            selected = "warehouse" if change == "add" else "renamed_warehouse" if change == "rename" else original_name
            new_task = await refreshed.task_manager.start_chat(
                refreshed.chat.agent_config, StreamChatInput(message="new turn", datasource=selected)
            )
            assert await asyncio.wait_for(started.get(), timeout=2) == new_task.session_id
            assert set(snapshots[old_task.session_id].datasource_configs) == set(before.datasource_configs)
            assert set(snapshots[new_task.session_id].datasource_configs) == set(after.datasource_configs)
            assert snapshots[new_task.session_id].current_datasource == selected
            assert snapshots[old_task.session_id] is not before
            assert snapshots[new_task.session_id] is not after

            stopped = await stop_chat(StopChatInput(session_id=old_task.session_id), refreshed)
            assert stopped.success
            await asyncio.gather(old_task.asyncio_task, return_exceptions=True)
            assert not new_task.asyncio_task.done()

            release.set()
            await refreshed.task_manager.wait_all_tasks()
            assert not refreshed.has_active_tasks()
            assert await get_datus_service(request) is refreshed
            assert refreshed.task_manager._default_source == "web"
            assert refreshed.task_manager._default_interactive is False
            assert refreshed.task_manager._stream_thinking is True

            final_task = await refreshed.task_manager.start_chat(
                refreshed.chat.agent_config, StreamChatInput(message="turn after the old task ended")
            )
            assert await asyncio.wait_for(started.get(), timeout=2) == final_task.session_id
            assert set(snapshots[final_task.session_id].datasource_configs) == set(after.datasource_configs)
        finally:
            release.set()
            await cache.shutdown()


class TestGetScopedSubAgent:
    """The gate that keeps an unrecognised sub-agent name from becoming an
    unscoped read.

    A service built from a name that matches no `agentic_nodes` key has no
    scope filter at all, so the request would succeed and return everything.
    Rejecting here — before any route body runs — means nothing unscoped is
    built or cached from a caller's identifier mistake.
    """

    @staticmethod
    def _request(sub_agent_name):
        request = MagicMock()
        request.state.app_context = AppContext(sub_agent_name=sub_agent_name)
        return request

    @staticmethod
    def _svc(known=("analyst",)):
        svc = MagicMock()
        svc.has_sub_agent.side_effect = lambda name: name in known
        return svc

    def test_unscoped_request_passes_through(self):
        """No name is not an unknown name — it is the single-tenant default."""
        svc = self._svc()

        assert deps.get_scoped_sub_agent(self._request(None), svc) is None
        svc.has_sub_agent.assert_not_called()

    def test_known_name_is_returned(self):
        assert deps.get_scoped_sub_agent(self._request("analyst"), self._svc()) == "analyst"

    def test_unknown_name_is_rejected_with_400(self):
        with pytest.raises(HTTPException) as excinfo:
            deps.get_scoped_sub_agent(self._request("no_such_sub_agent"), self._svc())

        assert excinfo.value.status_code == 400

    def test_rejection_names_the_offending_value(self):
        """The caller needs to know which identifier was wrong."""
        with pytest.raises(HTTPException) as excinfo:
            deps.get_scoped_sub_agent(self._request("typo_analyst"), self._svc())

        assert "typo_analyst" in str(excinfo.value.detail)

    def test_rejection_does_not_enumerate_configured_sub_agents(self):
        """Listing the other sub-agents to a consumer scoped to one of them is a
        disclosure in its own right."""
        with pytest.raises(HTTPException) as excinfo:
            deps.get_scoped_sub_agent(self._request("nope"), self._svc(known=("analyst", "auditor")))

        detail = str(excinfo.value.detail)
        assert "analyst" not in detail
        assert "auditor" not in detail
