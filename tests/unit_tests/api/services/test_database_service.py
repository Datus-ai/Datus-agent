"""Tests for datus.api.services.database_service — datasource management."""

import pytest

from datus.api.models.base_models import Result
from datus.api.models.database_models import ListDatabasesInput
from datus.api.services.database_service import DatasourceService
from datus.tools.db_tools.db_manager import DBManager


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
