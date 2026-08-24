"""Tests for datus.api.services.datus_service — per-project service facade."""

import pytest

from datus.api.services.chat_service import ChatService
from datus.api.services.chat_task_manager import ChatTaskManager
from datus.api.services.cli_service import CLIService
from datus.api.services.database_service import DatasourceService
from datus.api.services.datus_service import DatusService
from datus.api.services.explorer_service import ExplorerService
from datus.api.services.kb_service import KbService
from datus.api.services.mcp_service import MCPService


class TestDatusServiceInit:
    """Tests for DatusService construction."""

    def test_init_stores_config_and_project_id(self, real_agent_config):
        """Constructor stores agent_config and project_id as properties."""
        svc = DatusService(agent_config=real_agent_config, project_id="test-proj")
        assert svc.agent_config is real_agent_config
        assert svc.project_id == "test-proj"

    def test_init_creates_task_manager(self, real_agent_config):
        """Constructor creates a ChatTaskManager instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        assert isinstance(svc.task_manager, ChatTaskManager)

    def test_init_default_source_and_interactive_forwarded(self, real_agent_config):
        """default_source / default_interactive are forwarded to ChatTaskManager."""
        svc = DatusService(
            agent_config=real_agent_config,
            project_id="p1",
            default_source="vscode",
            default_interactive=False,
        )
        assert svc.task_manager._default_source == "vscode"
        assert svc.task_manager._default_interactive is False

    def test_init_default_source_and_interactive_defaults(self, real_agent_config):
        """When not passed, ChatTaskManager defaults to source=None, interactive=True."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        assert svc.task_manager._default_source is None
        assert svc.task_manager._default_interactive is True

    def test_lazy_slots_are_none_on_init(self, real_agent_config):
        """All lazy service slots are None after construction."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        assert svc._chat is None
        assert svc._cli is None
        assert svc._datasource is None
        assert svc._explorer is None
        assert svc._mcp is None
        assert svc._kb is None


class TestDatusServiceLazyProperties:
    """Tests for lazy service property initialization."""

    def test_chat_property_creates_chat_service(self, real_agent_config):
        """Accessing .chat creates a ChatService instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        chat = svc.chat
        assert isinstance(chat, ChatService)
        # Second access returns same instance
        assert svc.chat is chat

    def test_datasource_property_creates_datasource_service(self, real_agent_config):
        """Accessing .datasource creates a DatasourceService instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        db = svc.datasource
        assert isinstance(db, DatasourceService)
        assert svc.datasource is db

    def test_explorer_property_creates_explorer_service(self, real_agent_config):
        """Accessing .explorer creates an ExplorerService instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        explorer = svc.explorer
        assert isinstance(explorer, ExplorerService)
        assert svc.explorer is explorer

    def test_mcp_property_creates_mcp_service(self, real_agent_config):
        """Accessing .mcp creates an MCPService instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        mcp = svc.mcp
        assert isinstance(mcp, MCPService)
        assert svc.mcp is mcp

    def test_kb_property_creates_kb_service(self, real_agent_config):
        """Accessing .kb creates a KbService instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        kb = svc.kb
        assert isinstance(kb, KbService)
        assert svc.kb is kb

    def test_cli_property_creates_cli_service(self, real_agent_config):
        """Accessing .cli creates a CLIService (also initializes .chat)."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        cli = svc.cli
        assert isinstance(cli, CLIService)
        # cli depends on chat, so chat should also be initialized
        assert isinstance(svc._chat, ChatService)


class TestDatusServiceBehavior:
    """Tests for has_active_tasks and shutdown."""

    def test_has_active_tasks_delegates_to_task_manager(self, real_agent_config):
        """has_active_tasks() returns task_manager's result."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        # No tasks started => should be False
        assert svc.has_active_tasks() is False

    @pytest.mark.asyncio
    async def test_shutdown_does_not_raise(self, real_agent_config):
        """Shutdown completes without error even with no running tasks."""
        from unittest.mock import AsyncMock

        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        svc._task_manager.shutdown = AsyncMock()
        await svc.shutdown()
        svc._task_manager.shutdown.assert_awaited_once()

    def test_config_fingerprint_is_stable(self, real_agent_config):
        """Same config yields the same fingerprint across instances."""
        svc1 = DatusService(agent_config=real_agent_config, project_id="p1")
        svc2 = DatusService(agent_config=real_agent_config, project_id="p2")
        assert svc1.config_fingerprint == svc2.config_fingerprint
        assert isinstance(svc1.config_fingerprint, str) and len(svc1.config_fingerprint) > 0

    def test_compute_fingerprint_detects_changes(self, real_agent_config):
        """Mutating a dataclass field changes the fingerprint."""
        import copy

        fp1 = DatusService.compute_fingerprint(real_agent_config)
        mutated = copy.deepcopy(real_agent_config)
        mutated.target = f"{mutated.target}-mutated"
        fp2 = DatusService.compute_fingerprint(mutated)
        assert fp1 != fp2

    def test_compute_fingerprint_fallback_for_non_dataclass(self):
        """Non-dataclass input falls back to id-based fingerprint."""
        obj = object()
        fp = DatusService.compute_fingerprint(obj)  # type: ignore[arg-type]
        assert fp.startswith("id:")

    @pytest.mark.asyncio
    async def test_shutdown_handles_exception(self, real_agent_config):
        """Shutdown handles exception in task_manager gracefully."""
        from unittest.mock import AsyncMock

        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        svc._task_manager.shutdown = AsyncMock(side_effect=RuntimeError("boom"))
        await svc.shutdown()
        svc._task_manager.shutdown.assert_awaited_once()


class TestFingerprintPluginState:
    """Plugin changes must move the fingerprint so the cached service rebuilds.

    None of this state is a declared ``AgentConfig`` dataclass field, so before
    the plugin snapshot was folded in, ``DatusServiceCache`` kept serving a
    ``DatusService`` built from the previous plugin set.
    """

    def test_plugin_profile_config_change(self, real_agent_config):
        real_agent_config.init_plugin_services({"hello": {"prod": {"api_base_url": "https://one"}}})
        before = DatusService.compute_fingerprint(real_agent_config)

        real_agent_config.init_plugin_services({"hello": {"prod": {"api_base_url": "https://two"}}})

        assert DatusService.compute_fingerprint(real_agent_config) != before

    def test_plugin_disabled_for_project(self, real_agent_config):
        before = DatusService.compute_fingerprint(real_agent_config)

        real_agent_config.set_plugin_activation("hello", enabled=False, persist=False)

        assert DatusService.compute_fingerprint(real_agent_config) != before

    def test_plugins_master_switch_toggle(self, real_agent_config):
        before = DatusService.compute_fingerprint(real_agent_config)

        real_agent_config.plugins_enabled = False

        assert DatusService.compute_fingerprint(real_agent_config) != before

    def test_active_profile_pin_change(self, real_agent_config):
        real_agent_config.set_plugin_activation("hello", enabled=True, active_profiles=["staging"], persist=False)
        before = DatusService.compute_fingerprint(real_agent_config)

        real_agent_config.set_plugin_activation("hello", active_profiles=["prod"], persist=False)

        assert DatusService.compute_fingerprint(real_agent_config) != before

    def test_plugin_paths_change(self, real_agent_config, tmp_path):
        before = DatusService.compute_fingerprint(real_agent_config)

        real_agent_config.plugin_paths = [str(tmp_path / "mounted-plugin")]

        assert DatusService.compute_fingerprint(real_agent_config) != before

    def test_plugin_snapshot_failure_stays_stable(self, real_agent_config, monkeypatch):
        """A snapshot that raises degrades to one constant marker, not churn."""

        def boom(self):
            raise RuntimeError("snapshot exploded")

        monkeypatch.setattr("datus.configuration.agent_config.AgentConfig.plugin_state_signature", boom, raising=True)

        first = DatusService.compute_fingerprint(real_agent_config)
        second = DatusService.compute_fingerprint(real_agent_config)

        assert first == second
        assert not first.startswith("id:")
<<<<<<< HEAD
=======


class TestSubAgentScopedServices:
    """Explorer/Tool services keyed by sub-agent.

    These two services carry a knowledge-base scope filter, but
    ``DatusServiceCache`` keys only on project. A single lazy slot would cache
    the first request's sub-agent and serve its scope to every later one — not
    "too wide" like the unscoped default, but *wrong*, and silently so.
    """

    @staticmethod
    def _with_sub_agents(agent_config):
        """Give the config two sub-agents with disjoint scopes.

        ``tables`` rather than ``metrics``: a metrics/sqls scope resolves its
        paths against the subject tree, so it yields no filter on an empty test
        KB. ``tables`` goes through ``build_table_filter``, which needs no
        stored data — the point here is the plumbing, not the resolver.
        """
        agent_config.agentic_nodes = {
            **(agent_config.agentic_nodes or {}),
            "analyst": {"scoped_context": {"tables": "finance.revenue"}},
            "auditor": {"scoped_context": {"tables": "finance.audit"}},
        }
        return agent_config

    def test_default_explorer_is_unscoped(self, real_agent_config):
        """The property keeps today's behaviour: no name, no filter."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")

        assert svc.explorer.sub_agent_name is None
        assert svc.explorer.metric_rag.sub_agent_name is None

    def test_default_tool_is_unscoped(self, real_agent_config):
        svc = DatusService(agent_config=real_agent_config, project_id="p1")

        assert svc.tool._sub_agent_name is None

    def test_property_and_explicit_none_are_the_same_instance(self, real_agent_config):
        """`.explorer` is exactly `explorer_for(None)`, so the 14 existing call
        sites keep hitting one cached instance instead of rebuilding per call."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")

        assert svc.explorer is svc.explorer_for(None)
        assert svc.tool is svc.tool_for(None)

    def test_same_sub_agent_is_cached(self, real_agent_config):
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        assert svc.explorer_for("analyst") is svc.explorer_for("analyst")
        assert svc.tool_for("analyst") is svc.tool_for("analyst")

    def test_different_sub_agents_get_different_instances(self, real_agent_config):
        """The regression a single slot would cause: one sub-agent's scope
        answering another sub-agent's request."""
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        assert svc.explorer_for("analyst") is not svc.explorer_for("auditor")
        assert svc.tool_for("analyst") is not svc.tool_for("auditor")

    def test_scoped_explorer_is_not_the_unscoped_one(self, real_agent_config):
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        assert svc.explorer_for("analyst") is not svc.explorer

    def test_named_sub_agent_reaches_the_rag_layer(self, real_agent_config):
        """All three RAGs take `sub_agent_name` as their SECOND POSITIONAL
        argument. Passing only `datasource_id=` by keyword skipped it, which is
        why these reads were unscoped no matter what the caller asked.

        `MetricRAG` is the one that keeps the name as an attribute; the other
        two only use it to build their filter, which the next test covers."""
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        explorer = svc.explorer_for("analyst")

        assert explorer.sub_agent_name == "analyst"
        assert explorer.metric_rag.sub_agent_name == "analyst"

    def test_an_unnamed_service_stays_unscoped(self, real_agent_config):
        """`_build_sub_agent_filter` returns None for a falsy name, so an
        unnamed service must not carry a sub-agent at all."""
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        assert svc.explorer.sub_agent_name is None
        assert svc.explorer.metric_rag.sub_agent_name is None

    def test_two_sub_agents_get_their_own_scope(self, real_agent_config):
        """Equal names would mean the second service was handed the first
        one's scope. Filter contents are covered by test_rag_scope, which can
        bind a `tables` scope to a store; the explorer's RAGs are subject-scoped
        and build no filter at all."""
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        analyst = svc.explorer_for("analyst")
        auditor = svc.explorer_for("auditor")

        assert analyst.sub_agent_name == "analyst"
        assert auditor.sub_agent_name == "auditor"
        assert analyst.metric_rag.sub_agent_name == "analyst"
        assert auditor.metric_rag.sub_agent_name == "auditor"

    def test_an_unknown_name_is_refused(self, real_agent_config):
        """The failure mode this must never have: `sub_agent_config` is a plain
        dict lookup, so a name that is not an `agentic_nodes` key — an entry's
        `id`, a typo, a name left over from a rename — resolves to {} and the
        filter degrades to None. Building that service would answer 200 with
        everything, turning an identifier mistake into a scope bypass. Fail
        closed.
        """
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        with pytest.raises(DatusException) as excinfo:
            svc.explorer_for("no_such_sub_agent")

        assert "no_such_sub_agent" in str(excinfo.value)

    def test_an_unknown_name_is_refused_for_tools_too(self, real_agent_config):
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        with pytest.raises(DatusException):
            svc.tool_for("no_such_sub_agent")

    def test_a_refused_name_is_not_cached(self, real_agent_config):
        """A rejected name must leave no slot behind — otherwise a later lookup
        could find a service that was never supposed to exist."""
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        with pytest.raises(DatusException):
            svc.explorer_for("no_such_sub_agent")

        assert "no_such_sub_agent" not in svc._explorers

    def test_the_error_does_not_enumerate_other_sub_agents(self, real_agent_config):
        """On a deployment that publishes one sub-agent to a consumer, naming
        the others in an error is a disclosure in its own right."""
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        with pytest.raises(DatusException) as excinfo:
            svc.explorer_for("no_such_sub_agent")

        message = str(excinfo.value)
        assert "analyst" not in message
        assert "auditor" not in message

    def test_has_sub_agent_distinguishes_configured_names(self, real_agent_config):
        svc = DatusService(agent_config=self._with_sub_agents(real_agent_config), project_id="p1")

        assert svc.has_sub_agent("analyst") is True
        assert svc.has_sub_agent("no_such_sub_agent") is False
>>>>>>> cfac0a4 ([Refactor] Key the semantic projection on datasets so a table can be modelled more than once (#1327))
