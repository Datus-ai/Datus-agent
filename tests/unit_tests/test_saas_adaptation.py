# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
E2E-style integration tests for SaaS adaptation changes.

Covers:
- All async init functions can be awaited from an async context
- SessionManager with project-level session isolation
- Sync wrapper backward compatibility (same return types)
- init_ext_knowledge with the new string parameter directly
- Parameter decoupling: no argparse.Namespace dependency in any changed function
"""

import inspect
import os
import sqlite3
from typing import Union
from unittest.mock import MagicMock, patch

import pytest

from datus.models.session_manager import SessionManager
from datus.storage.ext_knowledge.ext_knowledge_init import (
    init_ext_knowledge,
    init_success_story_knowledge,
    init_success_story_knowledge_async,
)
from datus.storage.metric.metric_init import init_success_story_metrics, init_success_story_metrics_async
from datus.storage.reference_sql.reference_sql_init import init_reference_sql, init_reference_sql_async
from datus.storage.semantic_model.semantic_model_init import (
    init_success_story_semantic_model,
    init_success_story_semantic_model_async,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _init_vector_backend(tmp_path):
    """Initialize storage backends with tmp_path so data goes to a temp directory, not project root."""
    from datus.storage.backend_holder import init_backends
    from datus.storage.registry import clear_storage_registry

    init_backends(data_dir=str(tmp_path))
    yield
    clear_storage_registry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ext_knowledge_csv(tmp_path, filename="knowledge.csv"):
    """Write a minimal ext_knowledge CSV with required columns."""
    csv_path = tmp_path / filename
    csv_path.write_text(
        "subject_path,name,search_text,explanation\n"
        "Finance/Revenue,ARR,Annual Recurring Revenue,Total recurring revenue per year\n"
    )
    return str(csv_path)


# ===========================================================================
# TestAllAsyncFunctionsAreCoroutines
# ===========================================================================


@pytest.mark.ci
class TestAllAsyncFunctionsAreCoroutines:
    """Verify every async init function introduced by the SaaS adaptation is an async def."""

    def test_semantic_model_async_is_coroutine(self):
        assert inspect.iscoroutinefunction(init_success_story_semantic_model_async)

    def test_metrics_async_is_coroutine(self):
        assert inspect.iscoroutinefunction(init_success_story_metrics_async)

    def test_ext_knowledge_async_is_coroutine(self):
        assert inspect.iscoroutinefunction(init_success_story_knowledge_async)

    def test_reference_sql_async_is_coroutine(self):
        assert inspect.iscoroutinefunction(init_reference_sql_async)


# ===========================================================================
# TestNoArgparseNamespaceDependency
# ===========================================================================


@pytest.mark.ci
class TestNoArgparseNamespaceDependency:
    """Verify that all changed functions have removed argparse.Namespace (args) param."""

    def _assert_no_args_param(self, func):
        sig = inspect.signature(func)
        assert "args" not in sig.parameters, (
            f"{func.__name__} still has 'args' parameter — argparse.Namespace dependency not removed"
        )

    def test_semantic_model_async_no_args(self):
        self._assert_no_args_param(init_success_story_semantic_model_async)

    def test_semantic_model_sync_no_args(self):
        self._assert_no_args_param(init_success_story_semantic_model)

    def test_metrics_async_no_args(self):
        self._assert_no_args_param(init_success_story_metrics_async)

    def test_metrics_sync_no_args(self):
        self._assert_no_args_param(init_success_story_metrics)

    def test_ext_knowledge_knowledge_async_no_args(self):
        self._assert_no_args_param(init_success_story_knowledge_async)

    def test_ext_knowledge_knowledge_sync_no_args(self):
        self._assert_no_args_param(init_success_story_knowledge)

    def test_ext_knowledge_init_no_args(self):
        self._assert_no_args_param(init_ext_knowledge)

    def test_reference_sql_async_no_args(self):
        self._assert_no_args_param(init_reference_sql_async)

    def test_reference_sql_sync_no_args(self):
        self._assert_no_args_param(init_reference_sql)


# ===========================================================================
# TestAsyncFunctionsAwaitableInAsyncContext
# ===========================================================================


@pytest.mark.ci
class TestAsyncFunctionsAwaitableInAsyncContext:
    """Verify all async functions can be awaited from an async context without import errors."""

    @pytest.mark.asyncio
    async def test_semantic_model_async_awaitable_missing_file(self, tmp_path):
        """init_success_story_semantic_model_async can be awaited; returns (False, str) for missing file."""
        mock_config = MagicMock()
        success, error = await init_success_story_semantic_model_async(mock_config, str(tmp_path / "missing.csv"))
        assert isinstance(success, bool)
        assert isinstance(error, str)
        assert success is False

    @pytest.mark.asyncio
    async def test_ext_knowledge_async_awaitable_missing_file(self, tmp_path):
        """init_success_story_knowledge_async can be awaited; returns (False, str) for missing file."""
        mock_config = MagicMock()
        success, error = await init_success_story_knowledge_async(mock_config, str(tmp_path / "missing.csv"))
        assert isinstance(success, bool)
        assert isinstance(error, str)
        assert success is False

    @pytest.mark.asyncio
    async def test_reference_sql_async_awaitable_empty_dir(self):
        """init_reference_sql_async can be awaited; returns dict for empty sql_dir."""
        mock_storage = MagicMock()
        mock_storage.get_reference_sql_size.return_value = 0
        mock_config = MagicMock()

        result = await init_reference_sql_async(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir="",
        )
        assert isinstance(result, dict)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_reference_sql_async_awaitable_validate_only(self, tmp_path):
        """init_reference_sql_async can be awaited in validate_only mode."""
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT name FROM employees;")

        mock_storage = MagicMock()
        mock_config = MagicMock()

        result = await init_reference_sql_async(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir=str(sql_file),
            validate_only=True,
        )
        assert isinstance(result, dict)
        assert result["processed_entries"] == 0


# ===========================================================================
# TestSyncWrapperBackwardCompatibility
# ===========================================================================


@pytest.mark.ci
class TestSyncWrapperBackwardCompatibility:
    """Verify sync wrappers return the same types as before the SaaS adaptation."""

    def test_semantic_model_sync_returns_two_tuple(self, tmp_path):
        """init_success_story_semantic_model returns (bool, str)."""
        mock_config = MagicMock()
        result = init_success_story_semantic_model(mock_config, str(tmp_path / "missing.csv"))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_ext_knowledge_sync_returns_two_tuple(self, tmp_path):
        """init_success_story_knowledge returns (bool, str)."""
        mock_config = MagicMock()
        result = init_success_story_knowledge(mock_config, str(tmp_path / "missing.csv"))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_reference_sql_sync_returns_dict(self):
        """init_reference_sql returns a dict (backward compatible)."""
        mock_storage = MagicMock()
        mock_storage.get_reference_sql_size.return_value = 0
        mock_config = MagicMock()

        result = init_reference_sql(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir="",
        )
        assert isinstance(result, dict)
        assert "status" in result
        assert "valid_entries" in result
        assert "processed_entries" in result

    def test_metrics_sync_returns_three_tuple(self):
        """init_success_story_metrics returns (bool, str, Optional[dict]) — same as before."""
        mock_config = MagicMock()

        # Patch the async function itself to avoid creating an unawaited coroutine
        with patch(
            "datus.storage.metric.metric_init.init_success_story_metrics_async",
            return_value=(True, "", {"metrics": []}),
        ):
            result = init_success_story_metrics(mock_config, "dummy.csv")

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


# ===========================================================================
# TestInitExtKnowledgeStringParam
# ===========================================================================


@pytest.mark.ci
@pytest.mark.usefixtures("_init_vector_backend")
class TestInitExtKnowledgeStringParam:
    """Verify init_ext_knowledge works with a plain string parameter (no SimpleNamespace needed)."""

    def test_string_csv_path_accepted(self, tmp_path):
        """init_ext_knowledge accepts a plain string path directly (no SimpleNamespace needed)."""
        from datus.storage.embedding_models import get_db_embedding_model
        from datus.storage.ext_knowledge.store import ExtKnowledgeStore

        csv_path = _make_ext_knowledge_csv(tmp_path)
        store = ExtKnowledgeStore(embedding_model=get_db_embedding_model())

        # Invoking with a direct string must not raise any TypeError or AttributeError
        # (previously would have needed SimpleNamespace to mimic argparse.Namespace.ext_knowledge_csv)
        try:
            init_ext_knowledge(store, csv_path, build_mode="overwrite", pool_size=1)
        except (TypeError, AttributeError) as exc:
            raise AssertionError(
                f"init_ext_knowledge raised {type(exc).__name__} when called with a plain string: {exc}"
            ) from exc

        # The ARR entry should be present in the store (added now or already there from a prior call)
        results = store.search_all_knowledge()
        names = [r["name"] for r in results]
        assert "ARR" in names

    def test_none_string_returns_early(self, tmp_path):
        """init_ext_knowledge with None string parameter returns early without adding data."""
        from datus.storage.embedding_models import get_db_embedding_model
        from datus.storage.ext_knowledge.store import ExtKnowledgeStore

        store = ExtKnowledgeStore(embedding_model=get_db_embedding_model())
        count_before = len(store.search_all_knowledge())

        # Should not raise; should return without inserting anything new
        init_ext_knowledge(store, None)

        count_after = len(store.search_all_knowledge())
        assert count_after == count_before

    def test_empty_string_returns_early(self, tmp_path):
        """init_ext_knowledge with empty string parameter returns early without adding data."""
        from datus.storage.embedding_models import get_db_embedding_model
        from datus.storage.ext_knowledge.store import ExtKnowledgeStore

        store = ExtKnowledgeStore(embedding_model=get_db_embedding_model())
        count_before = len(store.search_all_knowledge())

        init_ext_knowledge(store, "")

        count_after = len(store.search_all_knowledge())
        assert count_after == count_before

    def test_ext_knowledge_csv_param_is_optional_str_type_annotated(self):
        """init_ext_knowledge's ext_knowledge_csv parameter has an Optional[str] annotation."""
        import typing

        sig = inspect.signature(init_ext_knowledge)
        param = sig.parameters.get("ext_knowledge_csv")
        assert param is not None
        annotation = param.annotation
        assert annotation is not inspect.Parameter.empty, "ext_knowledge_csv must have a type annotation"
        # Should be Optional[str] (i.e. Union[str, None])
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        assert origin is Union and str in args and type(None) in args


# ===========================================================================
# TestSessionManagerProjectIsolation
# ===========================================================================


@pytest.mark.ci
class TestSessionManagerProjectIsolation:
    """E2E tests simulating SaaS per-project session isolation via custom session_dir."""

    def test_two_projects_have_independent_sessions(self, tmp_path):
        """Two SessionManagers with different project dirs have fully isolated session stores."""
        project_a_dir = str(tmp_path / "project_a" / "sessions")
        project_b_dir = str(tmp_path / "project_b" / "sessions")

        mgr_a = SessionManager(session_dir=project_a_dir)
        mgr_b = SessionManager(session_dir=project_b_dir)

        try:
            mgr_a.get_session("shared-name-session")
            mgr_b.get_session("shared-name-session")

            # Each project has its own .db file in its own directory
            assert os.path.isfile(os.path.join(project_a_dir, "shared-name-session.db"))
            assert os.path.isfile(os.path.join(project_b_dir, "shared-name-session.db"))

            # Listing sessions shows the session in both, but they are stored in separate directories
            assert "shared-name-session" in mgr_a.list_sessions()
            assert "shared-name-session" in mgr_b.list_sessions()

            # Deleting from project A does not affect project B
            mgr_a.delete_session("shared-name-session")
            assert "shared-name-session" not in mgr_a.list_sessions()
            assert "shared-name-session" in mgr_b.list_sessions()
        finally:
            mgr_a.close_all_sessions()
            mgr_b.close_all_sessions()

    def test_project_session_dir_structure(self, tmp_path):
        """Simulates {home}/{project_id}/sessions directory structure used by SaaS backend."""
        home = str(tmp_path)

        for project_id in ["proj-001", "proj-002", "proj-003"]:
            session_dir = os.path.join(home, project_id, "sessions")
            mgr = SessionManager(session_dir=session_dir)
            mgr.get_session(f"user-session-{project_id}")
            mgr.close_all_sessions()

        # Verify all three project directories were created independently
        for project_id in ["proj-001", "proj-002", "proj-003"]:
            session_dir = os.path.join(home, project_id, "sessions")
            assert os.path.isdir(session_dir)
            db_file = os.path.join(session_dir, f"user-session-{project_id}.db")
            assert os.path.isfile(db_file)

    def test_custom_dir_session_roundtrip(self, tmp_path):
        """Session created in custom dir can be retrieved and deleted correctly."""
        custom_dir = str(tmp_path / "roundtrip_project" / "sessions")
        mgr = SessionManager(session_dir=custom_dir)

        try:
            session_id = "roundtrip-session"
            session = mgr.get_session(session_id)
            assert session.session_id == session_id

            db_path = os.path.join(custom_dir, f"{session_id}.db")
            assert os.path.isfile(db_path)

            # Write a session record so session_exists returns True
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO agent_sessions (session_id) VALUES (?)",
                    (session_id,),
                )
                conn.commit()

            assert mgr.session_exists(session_id) is True

            mgr.delete_session(session_id)
            assert not os.path.isfile(db_path)
            assert mgr.session_exists(session_id) is False
        finally:
            mgr.close_all_sessions()

    def test_session_dir_none_uses_default(self, real_agent_config):
        """SessionManager(session_dir=None) correctly falls back to the default path_manager path."""
        mgr_none = SessionManager(session_dir=None)
        mgr_default = SessionManager()

        try:
            # Both should resolve to the same directory
            assert mgr_none.session_dir == mgr_default.session_dir
        finally:
            mgr_none.close_all_sessions()
            mgr_default.close_all_sessions()


# ===========================================================================
# Section 8: Pydantic V2 compatibility — model_config = ConfigDict(...)
# ===========================================================================


class TestPydanticV2ConfigDict:
    """Verify that all specified Pydantic models use model_config instead of class Config."""

    def test_base_input_uses_model_config(self):
        """BaseInput uses model_config = ConfigDict(...)."""
        from datus.schemas.base import BaseInput

        assert hasattr(BaseInput, "model_config")
        assert BaseInput.model_config.get("extra") == "forbid"

    def test_base_result_uses_model_config(self):
        """BaseResult uses model_config = ConfigDict(...)."""
        from datus.schemas.base import BaseResult

        assert hasattr(BaseResult, "model_config")
        assert BaseResult.model_config.get("extra") == "forbid"

    def test_common_data_uses_model_config(self):
        """CommonData uses model_config = ConfigDict(...)."""
        from datus.schemas.base import CommonData

        assert hasattr(CommonData, "model_config")
        assert CommonData.model_config.get("extra") == "forbid"

    def test_execute_sql_result_uses_model_config(self):
        """ExecuteSQLResult uses model_config with arbitrary_types_allowed."""
        from datus.schemas.node_models import ExecuteSQLResult

        assert hasattr(ExecuteSQLResult, "model_config")
        assert ExecuteSQLResult.model_config.get("arbitrary_types_allowed") is True

    def test_reflection_result_uses_model_config(self):
        """ReflectionResult uses model_config with use_enum_values."""
        from datus.schemas.node_models import ReflectionResult

        assert hasattr(ReflectionResult, "model_config")
        assert ReflectionResult.model_config.get("use_enum_values") is True

    def test_chat_node_input_uses_model_config(self):
        """ChatNodeInput uses model_config with populate_by_name."""
        from datus.schemas.chat_agentic_node_models import ChatNodeInput

        assert hasattr(ChatNodeInput, "model_config")
        assert ChatNodeInput.model_config.get("populate_by_name") is True

    def test_gen_sql_node_input_uses_model_config(self):
        """GenSQLNodeInput uses model_config with populate_by_name."""
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        assert hasattr(GenSQLNodeInput, "model_config")
        assert GenSQLNodeInput.model_config.get("populate_by_name") is True

    def test_semantic_node_input_uses_model_config(self):
        """SemanticNodeInput uses model_config with populate_by_name."""
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        assert hasattr(SemanticNodeInput, "model_config")
        assert SemanticNodeInput.model_config.get("populate_by_name") is True

    def test_connection_config_uses_model_config(self):
        """ConnectionConfig uses model_config with extra=forbid."""
        from datus.tools.db_tools.config import ConnectionConfig

        assert hasattr(ConnectionConfig, "model_config")
        assert ConnectionConfig.model_config.get("extra") == "forbid"

    def test_no_class_config_in_changed_files(self):
        """None of the migrated classes still define a nested class Config."""
        from datus.schemas.base import BaseInput, BaseResult, CommonData
        from datus.schemas.chat_agentic_node_models import ChatNodeInput
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
        from datus.schemas.node_models import ExecuteSQLResult, ReflectionResult
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
        from datus.tools.db_tools.config import ConnectionConfig

        for cls in [
            BaseInput,
            BaseResult,
            CommonData,
            ExecuteSQLResult,
            ReflectionResult,
            ChatNodeInput,
            GenSQLNodeInput,
            SemanticNodeInput,
            ConnectionConfig,
        ]:
            # In Pydantic V2, model_config is the right way. If the class still defines
            # a nested Config class that is NOT inherited from BaseModel, it's a problem.
            own_config = cls.__dict__.get("Config")
            assert own_config is None, f"{cls.__name__} still has a nested class Config"

    def test_models_still_instantiate_correctly(self):
        """Migrated models can still be instantiated with valid data."""
        from datus.schemas.base import BaseInput, BaseResult, CommonData
        from datus.schemas.node_models import ExecuteSQLResult

        # BaseInput — extra="forbid" means no extra fields
        inp = BaseInput()
        assert inp.to_dict() == {}

        # BaseResult — requires success field
        res = BaseResult(success=True)
        assert res.success is True

        # CommonData
        cd = CommonData()
        assert cd.to_dict() == {}

        # ExecuteSQLResult — arbitrary_types_allowed
        exec_res = ExecuteSQLResult(success=True, sql_query="SELECT 1")
        assert exec_res.sql_query == "SELECT 1"


# ===========================================================================
# Section 9: DBManager injection — set_db_manager_factory()
# ===========================================================================


class TestDBManagerFactory:
    """Tests for set_db_manager_factory() and db_manager_instance() with factory injection."""

    def test_set_db_manager_factory_importable(self):
        """set_db_manager_factory can be imported."""
        from datus.tools.db_tools.db_manager import set_db_manager_factory

        assert callable(set_db_manager_factory)

    def test_factory_called_when_set(self):
        """When a factory is set, db_manager_instance() delegates to it."""
        from datus.tools.db_tools.db_manager import DBManager, db_manager_instance, set_db_manager_factory

        sentinel = DBManager({})
        factory_calls = []

        def mock_factory(configs):
            factory_calls.append(configs)
            return sentinel

        try:
            set_db_manager_factory(mock_factory)
            result = db_manager_instance({"ns": {}})
            assert result is sentinel
            assert len(factory_calls) == 1
        finally:
            set_db_manager_factory(None)  # reset

    def test_factory_none_restores_default(self):
        """Setting factory to None restores default DBManager creation."""
        from datus.tools.db_tools.db_manager import DBManager, db_manager_instance, set_db_manager_factory

        def mock_factory(configs):
            return DBManager({})

        try:
            set_db_manager_factory(mock_factory)
            set_db_manager_factory(None)
            result = db_manager_instance({})
            assert isinstance(result, DBManager)
        finally:
            set_db_manager_factory(None)

    def test_factory_receives_configs(self):
        """The factory receives the db_configs dict passed to db_manager_instance()."""
        from datus.tools.db_tools.db_manager import DBManager, db_manager_instance, set_db_manager_factory

        received = []

        def capture_factory(configs):
            received.append(configs)
            return DBManager(configs)

        try:
            set_db_manager_factory(capture_factory)
            test_configs = {"my_namespace": {}}
            db_manager_instance(test_configs)
            assert len(received) == 1
            assert received[0] is test_configs
        finally:
            set_db_manager_factory(None)

    def test_no_factory_caches_by_config(self):
        """Without a factory, db_manager_instance caches by namespace keys (avoids connection leak)."""
        from datus.tools.db_tools.db_manager import DBManager, db_manager_instance, set_db_manager_factory

        set_db_manager_factory(None)
        # Same config → same instance (cached)
        instance1 = db_manager_instance()
        instance2 = db_manager_instance()
        assert isinstance(instance1, DBManager)
        assert instance1 is instance2

        # Different config → different instance
        instance3 = db_manager_instance({"other_ns": {}})
        assert instance3 is not instance1


# ===========================================================================
# Section 10.1: AgentConfig skip_init_dirs
# ===========================================================================


class TestAgentConfigSkipInitDirs:
    """Tests for AgentConfig(skip_init_dirs=True) — SaaS mode without implicit global path-manager use."""

    def _make_config(self, tmp_path, skip_init_dirs=False):
        """Helper to create a minimal AgentConfig."""
        from datus.configuration.agent_config import AgentConfig, NodeConfig

        return AgentConfig(
            nodes={"test": NodeConfig(model="test-model", input=None)},
            home=str(tmp_path / "saas_home"),
            target="mock",
            models={
                "mock": {
                    "type": "openai",
                    "api_key": "mock-key",
                    "model": "mock-model",
                    "base_url": "http://localhost:0",
                },
            },
            skip_init_dirs=skip_init_dirs,
        )

    def test_skip_init_dirs_sets_rag_base_path(self, tmp_path):
        """With skip_init_dirs=True, rag_base_path is computed directly without path_manager."""
        config = self._make_config(tmp_path, skip_init_dirs=True)
        expected = str((tmp_path / "saas_home").resolve() / "data")
        assert config.rag_base_path == expected

    def test_skip_init_dirs_empty_save_dir(self, tmp_path):
        """With skip_init_dirs=True, _save_dir and _trajectory_dir are empty strings."""
        config = self._make_config(tmp_path, skip_init_dirs=True)
        assert config._save_dir == ""
        assert config._trajectory_dir == ""

    def test_skip_init_dirs_empty_benchmark_configs(self, tmp_path):
        """With skip_init_dirs=True, benchmark_configs is empty."""
        config = self._make_config(tmp_path, skip_init_dirs=True)
        assert config.benchmark_configs == {}

    def test_skip_init_dirs_does_not_call_path_manager(self, tmp_path):
        """With skip_init_dirs=True, legacy global path-manager helpers are not used."""
        with patch("datus.utils.path_manager.get_path_manager") as mock_pm:
            config = self._make_config(tmp_path, skip_init_dirs=True)
            mock_pm.assert_not_called()
            assert config.path_manager.datus_home == (tmp_path / "saas_home").resolve()

    def test_skip_init_dirs_does_not_call_init_embedding_models(self, tmp_path):
        """With skip_init_dirs=True, init_embedding_models() is not called."""
        with patch("datus.configuration.agent_config.init_embedding_models") as mock_init:
            self._make_config(tmp_path, skip_init_dirs=True)
            mock_init.assert_not_called()

    def test_skip_init_dirs_does_not_call_init_backends(self, tmp_path):
        """With skip_init_dirs=True, init_backends() is not called."""
        with patch("datus.storage.backend_holder.init_backends") as mock_init:
            self._make_config(tmp_path, skip_init_dirs=True)
            mock_init.assert_not_called()

    def test_default_skip_init_dirs_is_false(self, tmp_path):
        """By default, skip_init_dirs is False (CLI backward compatible)."""
        config = self._make_config(tmp_path, skip_init_dirs=False)
        assert config._skip_init_dirs is False
        # rag_base_path should still be set (by _init_dirs)
        assert config.rag_base_path != ""

    def test_skip_init_dirs_storage_configs_empty(self, tmp_path):
        """With skip_init_dirs=True and storage config, storage_configs is empty dict."""
        from datus.configuration.agent_config import AgentConfig, NodeConfig

        config = AgentConfig(
            nodes={"test": NodeConfig(model="test-model", input=None)},
            home=str(tmp_path / "saas_home"),
            target="mock",
            models={
                "mock": {
                    "type": "openai",
                    "api_key": "mock-key",
                    "model": "mock-model",
                    "base_url": "http://localhost:0",
                },
            },
            storage={"workspace_root": str(tmp_path / "workspace")},
            skip_init_dirs=True,
        )
        assert config.storage_configs == {}
        assert config.workspace_root == str(tmp_path / "workspace")


# ===========================================================================
# Section 10.2: Storage Registry (singleton + scoped view)
# ===========================================================================
