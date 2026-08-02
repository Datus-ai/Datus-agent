"""Tests for datus.api.services.database_service — datasource management."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus.api.models.base_models import Result
from datus.api.models.database_models import ListDatabasesInput
from datus.api.models.table_models import SemanticModelInput, ValidateSemanticModelData
from datus.api.services.database_service import DatasourceService
from datus.storage.semantic_model.artifact_file import artifact_revision, semantic_artifact_lock
from datus.tools.db_tools.db_manager import DBManager
from datus.tools.func_tool.metric_filesystem_tools import MetricFilesystemFuncTool
from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState


def _service_with_semantic_adapter(adapter: str = "metricflow") -> DatasourceService:
    svc = DatasourceService.__new__(DatasourceService)
    svc.agent_config = SimpleNamespace(
        home="/datus-home",
        current_datasource="warehouse",
        resolve_semantic_adapter=lambda: adapter,
    )
    return svc


def _osi_yaml(*, metric_name: str = "") -> str:
    metrics = f"\n    metrics:\n      - name: {metric_name}" if metric_name else "\n    metrics: []"
    return (
        "version: 1.0.0\n"
        "semantic_model:\n"
        "  - name: orders\n"
        "    datasets:\n"
        "      - name: orders\n"
        "        source: orders"
        f"{metrics}\n"
    )


class TestDatasourceServiceInit:
    """Tests for DatasourceService initialization."""

    def test_init_with_real_config(self, real_agent_config):
        """DatasourceService initializes with real agent config."""
        svc = DatasourceService(agent_config=real_agent_config)
        assert isinstance(svc, DatasourceService)
        assert svc.current_db_connector.get_type() == "sqlite"

    def test_init_sets_current_db_name(self, real_agent_config):
        """Init resolves the current database name from the datasource."""
        svc = DatasourceService(agent_config=real_agent_config)
        assert svc.current_db_name == "california_schools"

    def test_init_sets_datasource(self, real_agent_config):
        """Init stores current_datasource from config."""
        svc = DatasourceService(agent_config=real_agent_config)
        assert svc.current_datasource == real_agent_config.current_datasource

    def test_db_manager_created(self, real_agent_config):
        """Init creates DBManager."""
        svc = DatasourceService(agent_config=real_agent_config)
        assert isinstance(svc.db_manager, DBManager)

    def test_init_without_datasource_defers_semantic_rag(self, real_agent_config):
        """Init does not open datasource-scoped semantic storage before datasource selection."""
        real_agent_config.current_datasource = ""

        svc = DatasourceService(agent_config=real_agent_config)

        assert svc.current_datasource == ""
        assert svc.semantic_rag is None


class TestDatabaseServiceGetDatabaseType:
    """Tests for _get_database_type helper."""

    def test_known_database_returns_type(self, real_agent_config):
        """Known database returns its type string."""
        svc = DatasourceService(agent_config=real_agent_config)
        db_type, ds_id = svc._get_database_type("california_schools")
        assert db_type == "sqlite"

    def test_current_db_name_used_as_default(self, real_agent_config):
        """Without database_name arg, uses current_db_name."""
        svc = DatasourceService(agent_config=real_agent_config)
        db_type, ds_id = svc._get_database_type()
        assert db_type == "sqlite"
        assert ds_id == svc.current_db_name


class TestSemanticLayerServiceBranches:
    def test_active_semantic_adapter_normalizes_resolved_name(self):
        svc = _service_with_semantic_adapter(" OSI ")

        assert svc._active_semantic_adapter() == "osi"
        assert svc._is_osi_semantic_layer() is True

    def test_active_semantic_adapter_returns_empty_without_resolver(self):
        svc = DatasourceService.__new__(DatasourceService)
        svc.agent_config = SimpleNamespace()

        assert svc._active_semantic_adapter() == ""
        assert svc._is_osi_semantic_layer() is False

    def test_validate_osi_semantic_yaml_success(self, monkeypatch):
        calls = []
        package_mod = ModuleType("datus_semantic_osi")
        profile_mod = ModuleType("datus_semantic_osi.profile")

        def _load_osi_path(path, *, normalize):
            calls.append((path, normalize))

        profile_mod.load_osi_path = _load_osi_path
        package_mod.profile = profile_mod
        monkeypatch.setitem(sys.modules, "datus_semantic_osi", package_mod)
        monkeypatch.setitem(sys.modules, "datus_semantic_osi.profile", profile_mod)

        is_valid, errors = DatasourceService._validate_osi_semantic_yaml("kind: semantic_model\n", "orders.yml")

        assert is_valid is True
        assert errors == []
        assert len(calls) == 1
        assert calls[0][1] is True

    def test_validate_osi_semantic_yaml_reports_errors_and_ignores_cleanup_failure(self, monkeypatch):
        package_mod = ModuleType("datus_semantic_osi")
        profile_mod = ModuleType("datus_semantic_osi.profile")

        def _load_osi_path(path, *, normalize):
            raise ValueError("bad osi yaml")

        def _raise_os_error(path):
            raise OSError("busy")

        profile_mod.load_osi_path = _load_osi_path
        package_mod.profile = profile_mod
        monkeypatch.setitem(sys.modules, "datus_semantic_osi", package_mod)
        monkeypatch.setitem(sys.modules, "datus_semantic_osi.profile", profile_mod)
        monkeypatch.setattr("datus.api.services.database_service.os.unlink", _raise_os_error)

        is_valid, errors = DatasourceService._validate_osi_semantic_yaml("not: osi\n", "orders.yml")

        assert is_valid is False
        assert errors == ["bad osi yaml"]

    @pytest.mark.asyncio
    async def test_validate_semantic_model_uses_osi_validator(self):
        svc = _service_with_semantic_adapter("osi")
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": "/tmp/orders.yml"})
        svc._validate_osi_semantic_yaml = MagicMock(return_value=(False, ["missing semantic_models"]))
        request = SemanticModelInput(
            table="orders",
            yaml="version: 1.0.0\nsemantic_model:\n  - name: orders\n    datasets: []\n",
        )

        result = await svc.validate_semantic_model(request)

        assert result.success is True
        assert result.data == ValidateSemanticModelData(valid=False, invalid_message=["missing semantic_models"])
        svc._validate_osi_semantic_yaml.assert_called_once_with(request.yaml, str(Path("/tmp/orders.yml").resolve()))

    @pytest.mark.asyncio
    async def test_validate_semantic_model_uses_metricflow_validator(self):
        svc = _service_with_semantic_adapter("metricflow")
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": "/tmp/orders.yml"})
        request = SemanticModelInput(
            table="orders",
            yaml="semantic_model:\n  name: orders\n",
            catalog="cat",
            database="db",
            db_schema="schema",
        )

        with patch("datus.api.utils.semantic_validation.validate_semantic_yaml", return_value=(True, [])) as validate:
            result = await svc.validate_semantic_model(request)

        assert result.success is True
        assert result.data == ValidateSemanticModelData(valid=True, invalid_message=None)
        validate.assert_called_once_with(
            yaml_content=request.yaml,
            file_path=str(Path("/tmp/orders.yml").resolve()),
            datus_home="/datus-home",
            datasource="warehouse",
            catalog="cat",
            database="db",
            db_schema="schema",
        )

    @pytest.mark.asyncio
    async def test_save_semantic_model_uses_osi_sync_tool(self, tmp_path):
        svc = _service_with_semantic_adapter("osi")
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text("version: 1.0.0\nsemantic_model:\n  - name: orders\n    datasets: []\n")
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})
        svc._validate_osi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(return_value=(True, {"valid": True, "issues": []}, ""))
        request = SemanticModelInput(
            table="orders",
            yaml=(
                "version: 1.0.0\nsemantic_model:\n  - name: orders\n    datasets:\n"
                "      - name: orders\n        source: orders\n    metrics: []\n"
            ),
        )

        with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
            tools_cls.return_value.sync_osi_to_db.return_value = {
                "success": True,
                "semantic_objects": 1,
                "metric_names": [],
            }
            result = await svc.save_semantic_model(request)

        assert result.success is True
        assert yaml_file.read_text(encoding="utf-8") == request.yaml
        assert result.data.status == "synced"
        assert result.data.revision == artifact_revision(request.yaml.encode())
        tools_cls.assert_called_once_with(agent_config=svc.agent_config, authoring_format="osi")
        tools_cls.return_value.sync_osi_to_db.assert_called_once_with(
            str(yaml_file),
            include_semantic_objects=True,
            include_metrics=True,
        )

    @pytest.mark.asyncio
    async def test_save_semantic_model_uses_metricflow_sync(self, tmp_path):
        svc = _service_with_semantic_adapter("metricflow")
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text("semantic_model:\n  name: orders\n")
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})
        request = SemanticModelInput(table="orders", yaml="semantic_model:\n  name: updated_orders\n")

        with (
            patch("datus.api.utils.semantic_validation.validate_semantic_yaml", return_value=(True, [])),
            patch(
                "datus.api.services.database_service.GenerationHooks._sync_semantic_to_db",
                return_value={"success": True},
            ) as sync,
        ):
            result = await svc.save_semantic_model(request)

        assert result.success is True
        sync.assert_called_once_with(
            str(yaml_file),
            svc.agent_config,
            include_semantic_objects=True,
            include_metrics=False,
        )

    @pytest.mark.asyncio
    async def test_save_semantic_model_rejects_stale_revision(self, tmp_path):
        svc = _service_with_semantic_adapter("osi")
        yaml_file = tmp_path / "orders.yml"
        original = _osi_yaml()
        yaml_file.write_text(original)
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})

        result = await svc.save_semantic_model(
            SemanticModelInput(
                table="orders",
                yaml=_osi_yaml(metric_name="order_count"),
                expected_revision="sha256:stale",
            )
        )

        assert result.success is False
        assert result.errorCode == "SEMANTIC_MODEL_REVISION_CONFLICT"
        assert result.data.status == "conflict"
        assert result.data.revision == artifact_revision(original.encode())
        assert yaml_file.read_text() == original

    def test_api_save_serializes_against_agent_metric_mutation(self, tmp_path):
        svc = _service_with_semantic_adapter("osi")
        yaml_file = tmp_path / "orders.yml"
        original = _osi_yaml()
        updated = _osi_yaml(metric_name="api_metric")
        yaml_file.write_text(original)
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})
        svc._validate_osi_semantic_yaml = MagicMock(return_value=(True, []))

        validation_started = threading.Event()
        release_validation = threading.Event()

        def validate_live_candidate(*_args, **_kwargs):
            validation_started.set()
            assert release_validation.wait(timeout=5)
            return True, {"valid": True, "issues": []}, ""

        svc._full_osi_validation = MagicMock(side_effect=validate_live_candidate)
        target_state = OsiSemanticModelTargetState()
        target_state.select(
            {
                "semantic_model_name": "orders",
                "semantic_model_file": "subject/semantic_models/warehouse/orders.yml",
                "absolute_path": str(yaml_file.resolve()),
                "artifact_sha256": artifact_revision(original.encode()).removeprefix("sha256:"),
            },
            mode="bound",
        )
        agent_tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=target_state,
        )
        agent_lock_attempted = threading.Event()

        @contextmanager
        def observed_agent_lock(path):
            agent_lock_attempted.set()
            with semantic_artifact_lock(path):
                yield

        def mutate_from_agent():
            return agent_tool.upsert_osi_metrics("orders.yml", '[{"name":"agent_metric"}]')

        with (
            patch(
                "datus.tools.func_tool.metric_filesystem_tools.semantic_artifact_lock",
                observed_agent_lock,
            ),
            patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls,
        ):
            tools_cls.return_value.sync_osi_to_db.return_value = {"success": True}
            with ThreadPoolExecutor(max_workers=2) as executor:
                api_future = executor.submit(
                    svc._save_semantic_model_sync,
                    SemanticModelInput(
                        table="orders",
                        yaml=updated,
                        expected_revision=artifact_revision(original.encode()),
                    ),
                )
                assert validation_started.wait(timeout=5)
                agent_future = executor.submit(mutate_from_agent)
                assert agent_lock_attempted.wait(timeout=5)
                try:
                    assert not agent_future.done()
                finally:
                    release_validation.set()
                api_result = api_future.result(timeout=5)
                agent_result = agent_future.result(timeout=5)

        assert api_result.success is True
        assert agent_result.success == 0
        assert "changed after selection" in agent_result.error
        assert yaml_file.read_text() == updated

    @pytest.mark.asyncio
    async def test_save_semantic_model_restores_yaml_after_full_validation_failure(self, tmp_path):
        svc = _service_with_semantic_adapter("osi")
        yaml_file = tmp_path / "orders.yml"
        original = _osi_yaml()
        yaml_file.write_text(original)
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})
        svc._validate_osi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(
            return_value=(False, {"valid": False, "issues": [{"message": "bad metric"}]}, "bad metric")
        )

        result = await svc.save_semantic_model(
            SemanticModelInput(table="orders", yaml=_osi_yaml(metric_name="bad_metric"))
        )

        assert result.success is False
        assert result.errorCode == "SEMANTIC_MODEL_INVALID"
        assert result.data.yaml_saved is False
        assert yaml_file.read_text() == original

    @pytest.mark.asyncio
    async def test_save_semantic_model_restores_yaml_when_full_validation_raises(self, tmp_path):
        svc = _service_with_semantic_adapter("osi")
        yaml_file = tmp_path / "orders.yml"
        original = _osi_yaml()
        yaml_file.write_text(original)
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})
        svc._validate_osi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(side_effect=RuntimeError("validator unavailable"))

        result = await svc.save_semantic_model(
            SemanticModelInput(table="orders", yaml=_osi_yaml(metric_name="order_count"))
        )

        assert result.success is False
        assert result.errorCode == "INTERNAL_COMMAND_ERROR"
        assert result.data.retryable is True
        assert result.data.yaml_saved is False
        assert yaml_file.read_text() == original

    @pytest.mark.asyncio
    async def test_save_semantic_model_keeps_valid_yaml_when_sync_fails(self, tmp_path):
        svc = _service_with_semantic_adapter("osi")
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text(_osi_yaml())
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})
        svc._validate_osi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(return_value=(True, {"valid": True, "issues": []}, ""))
        updated = _osi_yaml(metric_name="order_count")

        with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
            tools_cls.return_value.sync_osi_to_db.return_value = {"success": False, "error": "storage down"}
            result = await svc.save_semantic_model(SemanticModelInput(table="orders", yaml=updated))

        assert result.success is False
        assert result.errorCode == "SEMANTIC_MODEL_SYNC_FAILED"
        assert result.data.status == "saved_not_synced"
        assert result.data.retryable is True
        assert result.data.revision == artifact_revision(updated.encode())
        assert yaml_file.read_text() == updated

    @pytest.mark.asyncio
    async def test_save_semantic_model_unchanged_yaml_still_repairs_kb(self, tmp_path):
        svc = _service_with_semantic_adapter("osi")
        yaml_file = tmp_path / "orders.yml"
        content = _osi_yaml()
        yaml_file.write_text(content)
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})
        svc._validate_osi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(return_value=(True, {"valid": True, "issues": []}, ""))

        with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
            tools_cls.return_value.sync_osi_to_db.return_value = {"success": True}
            result = await svc.save_semantic_model(
                SemanticModelInput(
                    table="orders",
                    yaml=content,
                    expected_revision=artifact_revision(content.encode()),
                )
            )

        assert result.success is True
        tools_cls.return_value.sync_osi_to_db.assert_called_once()


class TestGetSemanticModel:
    """Tests for get_semantic_model and validate_semantic_model."""

    def test_get_semantic_model_returns_stable_file_identity_and_revision(self, tmp_path):
        svc = _service_with_semantic_adapter("osi")
        svc.agent_config.path_manager = SimpleNamespace(semantic_model_path=lambda _datasource: tmp_path)
        yaml_file = tmp_path / "orders.yml"
        content = _osi_yaml()
        yaml_file.write_text(content)
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})

        result = svc.get_semantic_model("orders")

        assert result.success is True
        assert result.data.semantic_model_name == "orders"
        assert result.data.semantic_model_file == "subject/semantic_models/warehouse/orders.yml"
        assert result.data.revision == artifact_revision(content.encode())

    @pytest.mark.asyncio
    async def test_save_semantic_model_accepts_stable_file_without_kb_lookup(self, tmp_path):
        svc = _service_with_semantic_adapter("osi")
        svc.agent_config.path_manager = SimpleNamespace(semantic_model_path=lambda _datasource: tmp_path)
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text(_osi_yaml())
        svc._get_semantic_model = MagicMock(side_effect=AssertionError("KB lookup must not be used"))
        svc._validate_osi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(return_value=(True, {"valid": True, "issues": []}, ""))

        with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
            tools_cls.return_value.sync_osi_to_db.return_value = {"success": True}
            result = await svc.save_semantic_model(
                SemanticModelInput(
                    yaml=_osi_yaml(metric_name="order_count"),
                    semantic_model_file="subject/semantic_models/warehouse/orders.yml",
                    semantic_model_name="orders",
                    expected_revision=artifact_revision(yaml_file.read_bytes()),
                )
            )

        assert result.success is True
        svc._get_semantic_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_semantic_model_rejects_file_escape(self, tmp_path):
        svc = _service_with_semantic_adapter("osi")
        svc.agent_config.path_manager = SimpleNamespace(semantic_model_path=lambda _datasource: tmp_path)

        result = await svc.save_semantic_model(
            SemanticModelInput(
                yaml=_osi_yaml(),
                semantic_model_file="../outside.yml",
                semantic_model_name="orders",
            )
        )

        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"
        assert "escapes" in result.errorMessage

    def test_get_semantic_model_nonexistent(self, real_agent_config):
        """get_semantic_model for nonexistent table returns empty result."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_semantic_model("nonexistent_table_xyz")
        # Should return success=True with no data, or success=False
        assert isinstance(result, Result)

    def test_get_semantic_model_for_known_table(self, real_agent_config):
        """get_semantic_model for known table (may return empty if no semantic model built)."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_semantic_model("schools")
        # The table exists but may not have a semantic model file
        assert isinstance(result, Result)

    def test_get_semantic_model_prefers_runtime_db_context(self, real_agent_config):
        """Runtime catalog/database/schema context is used for semantic model lookup."""
        svc = DatasourceService(agent_config=real_agent_config)
        call = {}

        class FakeSemanticRag:
            def get_semantic_model(
                self,
                *,
                catalog_name: str,
                database_name: str,
                schema_name: str,
                table_name: str,
            ):
                call.update(
                    catalog_name=catalog_name,
                    database_name=database_name,
                    schema_name=schema_name,
                    table_name=table_name,
                )
                return None

        svc._ensure_semantic_rag = lambda: FakeSemanticRag()

        result = svc.get_semantic_model(
            "embedded_catalog.embedded_db.embedded_schema.schools",
            catalog="runtime_catalog",
            database="runtime_db",
            db_schema="runtime_schema",
        )

        assert isinstance(result, Result)
        assert call == {
            "catalog_name": "runtime_catalog",
            "database_name": "runtime_db",
            "schema_name": "runtime_schema",
            "table_name": "schools",
        }

    def test_get_semantic_model_passes_explicit_model_name(self, real_agent_config):
        svc = DatasourceService(agent_config=real_agent_config)
        semantic_rag = MagicMock()
        semantic_rag.get_semantic_model.return_value = None
        svc._ensure_semantic_rag = lambda: semantic_rag

        svc._get_semantic_model("schools", semantic_model_name="education")

        assert semantic_rag.get_semantic_model.call_args.kwargs["semantic_model_name"] == "education"

    @pytest.mark.asyncio
    async def test_validate_semantic_model_nonexistent(self, real_agent_config):
        """validate_semantic_model for nonexistent table returns error."""
        from datus.api.models.table_models import SemanticModelInput

        svc = DatasourceService(agent_config=real_agent_config)
        request = SemanticModelInput(table="nonexistent_xyz", yaml="metric:\n  name: test\n")
        result = await svc.validate_semantic_model(request)
        assert result.success is False


class TestListDatabases:
    """Tests for list_databases with real SQLite connection."""

    def test_list_databases_returns_success(self, real_agent_config):
        """list_databases returns success with database info."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        assert result.success is True
        assert result.data.total_count == len(result.data.databases)
        assert result.data.total_count >= 1

    def test_list_databases_has_entries(self, real_agent_config):
        """list_databases returns at least one database entry."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        assert len(result.data.databases) >= 1

    def test_list_databases_connection_status(self, real_agent_config):
        """Databases are connected."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        for db in result.data.databases:
            assert db.connection_status == "connected"

    def test_list_databases_has_tables(self, real_agent_config):
        """Connected databases report table count > 0."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        connected_databases = [db for db in result.data.databases if db.connection_status == "connected"]
        assert connected_databases
        assert all(db.tables_count > 0 for db in connected_databases)

    def test_list_databases_with_datasource_filter(self, real_agent_config):
        """list_databases with datasource_id filter."""
        svc = DatasourceService(agent_config=real_agent_config)
        # datasource_id is a datasource name
        request = ListDatabasesInput(datasource_id="california_schools")
        result = svc.list_databases(request)
        assert result.success is True

    def test_list_databases_with_database_name_filter(self, real_agent_config):
        """list_databases with database_name filter narrows results."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput(database_name="main")
        result = svc.list_databases(request)
        assert result.success is True

    def test_list_databases_has_tables_list(self, real_agent_config):
        """list_databases includes tables list in database info."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        databases_with_tables = [db for db in result.data.databases if db.tables is not None]
        assert databases_with_tables
        assert all(isinstance(db.tables, list) for db in databases_with_tables)

    def test_list_databases_has_type_field(self, real_agent_config):
        """list_databases includes database type."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        for db in result.data.databases:
            assert db.type == "sqlite"

    def test_list_databases_has_current_database(self, real_agent_config):
        """list_databases data includes current_database field."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        assert result.data.current_database == "california_schools"


class _FakeServerConnector:
    """No-schema (server-style) connector that distinguishes its configured
    database from every database reachable on the instance.

    ``get_databases`` mimics ``SHOW DATABASES`` (the whole server); a scoped
    listing must NOT call it when a database is configured.
    """

    dialect = "starrocks"
    catalog_name = "default_catalog"
    connection_string = "mysql+pymysql://u:p@host:9030/benchmark"

    def __init__(self, database_name: str):
        self.database_name = database_name
        self.get_databases_calls = 0

    def test_connection(self) -> bool:  # audit-noqa: zero_assert_test — connector API stub, not a test
        return True

    def get_databases(self, catalog_name: str = "", include_sys: bool = False):
        self.get_databases_calls += 1
        return ["benchmark", "ga4", "olist", "fund_poc"]

    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        return ["t2", "t1"]


@pytest.fixture
def _no_schema_dialect(monkeypatch):
    """Force the server-style (no per-database schema) code path."""
    from datus_db_core import connector_registry

    monkeypatch.setattr(connector_registry, "support_schema", lambda dialect: False)


class TestGetConnectionInfoScoping:
    """A datasource is a connection profile scoped to its configured database;
    listing must not leak every database on the server."""

    def test_configured_database_is_listed_without_enumerating_server(self, real_agent_config, _no_schema_dialect):
        """With a configured database, only that database is returned and the
        server-wide ``get_databases`` enumeration is never invoked."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeServerConnector(database_name="benchmark")

        infos = svc._get_connection_info(connector, "benchmark", ListDatabasesInput())

        assert [i.name for i in infos] == ["benchmark"]
        assert connector.get_databases_calls == 0
        assert infos[0].current is True
        # tables are surfaced (and sorted) for the scoped database
        assert infos[0].tables == ["t1", "t2"]

    def test_falls_back_to_server_enumeration_when_unconfigured(self, real_agent_config, _no_schema_dialect):
        """Only when no database is configured do we enumerate the server so the
        connection's reachable databases stay browsable."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeServerConnector(database_name="")

        infos = svc._get_connection_info(connector, "ds", ListDatabasesInput())

        assert connector.get_databases_calls == 1
        assert [i.name for i in infos] == ["benchmark", "ga4", "olist", "fund_poc"]

    def test_request_database_name_filter_takes_precedence(self, real_agent_config, _no_schema_dialect):
        """An explicit database_name filter wins over the configured database and
        still avoids the server-wide enumeration."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeServerConnector(database_name="benchmark")

        infos = svc._get_connection_info(connector, "benchmark", ListDatabasesInput(database_name="ga4"))

        assert [i.name for i in infos] == ["ga4"]
        assert connector.get_databases_calls == 0


class TestGetTableSchema:
    """Tests for get_table_schema with real SQLite connection."""

    def test_get_table_schema_returns_columns(self, real_agent_config):
        """get_table_schema returns column info for existing table."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_table_schema("schools")
        assert result.success is True
        assert result.data.table.name == "schools"
        assert [col.name for col in result.data.table.columns[:2]] == ["CDSCode", "NCESDist"]

    def test_get_table_schema_column_has_name_and_type(self, real_agent_config):
        """Each column has name and type fields."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_table_schema("schools")
        for col in result.data.table.columns:
            assert col.name != ""
            assert col.type != ""

    def test_get_table_schema_uses_connector_nullable_contract(self, real_agent_config):
        svc = DatasourceService(agent_config=real_agent_config)
        svc.current_db_connector.get_schema = MagicMock(
            return_value=[
                {
                    "name": "id",
                    "type": "BIGINT",
                    "nullable": False,
                    "default_value": None,
                    "pk": False,
                }
            ]
        )

        result = svc.get_table_schema("orders")

        assert result.success is True
        assert result.data.table.columns[0].nullable is False

    def test_get_table_schema_nonexistent_table(self, real_agent_config):
        """Nonexistent table returns failure."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_table_schema("totally_fake_table_xyz")
        assert result.success is False

    def test_get_table_schema_caches_columns(self, real_agent_config):
        """Second lookup is served from cache without re-hitting the connector."""
        svc = DatasourceService(agent_config=real_agent_config)
        spy = MagicMock(wraps=svc.current_db_connector.get_schema)
        svc.current_db_connector.get_schema = spy

        first = svc.get_table_schema("schools")
        second = svc.get_table_schema("schools")

        assert first.success is True and second.success is True
        assert [c.name for c in second.data.table.columns] == [c.name for c in first.data.table.columns]
        assert spy.call_count == 1


class TestGetTablesColumns:
    """Tests for the batch get_tables_columns (autocomplete prefetch)."""

    def test_returns_columns_for_known_tables(self, real_agent_config):
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_tables_columns(["schools"])
        assert result.success is True
        assert [t.table for t in result.data.tables] == ["schools"]
        col = result.data.tables[0].columns[0]
        assert col.name != "" and col.type != ""
        # Slim shape: no default_value in the prefetch payload.
        assert not hasattr(col, "default_value")

    def test_omits_unresolved_tables(self, real_agent_config):
        """A bad name is skipped rather than failing the whole batch."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_tables_columns(["schools", "totally_fake_table_xyz"])
        assert result.success is True
        assert [t.table for t in result.data.tables] == ["schools"]

    def test_populates_shared_cache(self, real_agent_config):
        """Columns fetched by the batch are reused by a later single-table detail."""
        svc = DatasourceService(agent_config=real_agent_config)
        spy = MagicMock(wraps=svc.current_db_connector.get_schema)
        svc.current_db_connector.get_schema = spy

        svc.get_tables_columns(["schools"])
        detail = svc.get_table_schema("schools")

        assert detail.success is True
        assert spy.call_count == 1

    def test_over_limit_returns_validation_error(self, real_agent_config):
        svc = DatasourceService(agent_config=real_agent_config)
        svc.agent_config.api_config = {"max_prefetch_tables": 1}
        result = svc.get_tables_columns(["schools", "frpm"])
        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"
