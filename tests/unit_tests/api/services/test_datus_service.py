"""Tests for datus.api.services.datus_service — per-project service facade."""

import pytest

from datus.api.services.datus_service import DatusService


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
        assert svc.task_manager is not None

    def test_lazy_slots_are_none_on_init(self, real_agent_config):
        """All lazy service slots are None after construction."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        assert svc._chat is None
        assert svc._cli is None
        assert svc._database is None
        assert svc._explorer is None
        assert svc._mcp is None
        assert svc._kb is None


class TestDatusServiceLazyProperties:
    """Tests for lazy service property initialization."""

    def test_chat_property_creates_chat_service(self, real_agent_config):
        """Accessing .chat creates a ChatService instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        chat = svc.chat
        assert chat is not None
        # Second access returns same instance
        assert svc.chat is chat

    def test_database_property_creates_database_service(self, real_agent_config):
        """Accessing .database creates a DatabaseService instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        db = svc.database
        assert db is not None
        assert svc.database is db

    def test_explorer_property_creates_explorer_service(self, real_agent_config):
        """Accessing .explorer creates an ExplorerService instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        explorer = svc.explorer
        assert explorer is not None
        assert svc.explorer is explorer

    def test_mcp_property_creates_mcp_service(self, real_agent_config):
        """Accessing .mcp creates an MCPService instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        mcp = svc.mcp
        assert mcp is not None
        assert svc.mcp is mcp

    def test_kb_property_creates_kb_service(self, real_agent_config):
        """Accessing .kb creates a KbService instance."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        kb = svc.kb
        assert kb is not None
        assert svc.kb is kb

    def test_cli_property_creates_cli_service(self, real_agent_config):
        """Accessing .cli creates a CLIService (also initializes .chat)."""
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        cli = svc.cli
        assert cli is not None
        # cli depends on chat, so chat should also be initialized
        assert svc._chat is not None


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
        svc = DatusService(agent_config=real_agent_config, project_id="p1")
        await svc.shutdown()  # should not raise

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
        await svc.shutdown()  # should not raise — exception is caught and logged
