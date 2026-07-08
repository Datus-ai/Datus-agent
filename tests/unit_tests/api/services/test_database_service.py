"""Tests for datus.api.services.database_service — datasource management."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.api.models.base_models import Result
from datus.api.models.database_models import ListDatabasesInput
from datus.api.models.table_models import SemanticModelInput, ValidateSemanticModelData
from datus.api.services.database_service import DatasourceService
from datus.tools.db_tools.db_manager import DBManager


def _service_with_semantic_adapter(adapter: str = "metricflow") -> DatasourceService:
    svc = DatasourceService.__new__(DatasourceService)
    svc.agent_config = SimpleNamespace(
        home="/datus-home",
        current_datasource="warehouse",
        resolve_semantic_adapter=lambda: adapter,
    )
    return svc


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
        request = SemanticModelInput(table="orders", yaml="kind: semantic_model\n")

        result = await svc.validate_semantic_model(request)

        assert result.success is True
        assert result.data == ValidateSemanticModelData(valid=False, invalid_message=["missing semantic_models"])
        svc._validate_osi_semantic_yaml.assert_called_once_with(request.yaml, "/tmp/orders.yml")

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
            file_path="/tmp/orders.yml",
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
        svc.validate_semantic_model = AsyncMock(
            return_value=Result(
                success=True,
                data=ValidateSemanticModelData(valid=True, invalid_message=None),
            )
        )
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})
        request = SemanticModelInput(table="orders", yaml="kind: semantic_model\n")

        with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
            tools_cls.return_value.sync_osi_semantic_to_db.return_value = {"success": True}
            result = await svc.save_semantic_model(request)

        assert result.success is True
        assert yaml_file.read_text(encoding="utf-8") == request.yaml
        tools_cls.assert_called_once_with(agent_config=svc.agent_config, authoring_format="osi")
        tools_cls.return_value.sync_osi_semantic_to_db.assert_called_once_with(str(yaml_file))

    @pytest.mark.asyncio
    async def test_save_semantic_model_uses_metricflow_sync(self, tmp_path):
        svc = _service_with_semantic_adapter("metricflow")
        yaml_file = tmp_path / "orders.yml"
        svc.validate_semantic_model = AsyncMock(
            return_value=Result(
                success=True,
                data=ValidateSemanticModelData(valid=True, invalid_message=None),
            )
        )
        svc._get_semantic_model = MagicMock(return_value={"yaml_path": str(yaml_file)})
        request = SemanticModelInput(table="orders", yaml="semantic_model:\n  name: orders\n")

        with patch(
            "datus.api.services.database_service.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True},
        ) as sync:
            result = await svc.save_semantic_model(request)

        assert result.success is True
        sync.assert_called_once_with(
            str(yaml_file),
            svc.agent_config,
            include_semantic_objects=True,
            include_metrics=False,
        )


class TestGetSemanticModel:
    """Tests for get_semantic_model and validate_semantic_model."""

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

    def test_get_table_schema_nonexistent_table(self, real_agent_config):
        """Nonexistent table returns failure."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_table_schema("totally_fake_table_xyz")
        assert result.success is False
