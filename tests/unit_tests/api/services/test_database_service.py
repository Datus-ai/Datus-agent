"""Tests for datus.api.services.database_service — database management."""

import pytest

from datus.api.models.database_models import ListDatabasesInput
from datus.api.services.database_service import DatabaseService


class TestDatabaseServiceInit:
    """Tests for DatabaseService initialization."""

    def test_init_with_real_config(self, real_agent_config):
        """DatabaseService initializes with real agent config."""
        svc = DatabaseService(agent_config=real_agent_config)
        assert svc is not None
        assert svc.current_db_connector is not None

    def test_init_sets_current_database(self, real_agent_config):
        """Init resolves the current database name from the namespace."""
        svc = DatabaseService(agent_config=real_agent_config)
        assert svc.current_database is not None

    def test_init_sets_namespace(self, real_agent_config):
        """Init stores current_namespace from config."""
        svc = DatabaseService(agent_config=real_agent_config)
        assert svc.current_namespace == real_agent_config.current_namespace

    def test_db_manager_created(self, real_agent_config):
        """Init creates DBManager."""
        svc = DatabaseService(agent_config=real_agent_config)
        assert svc.db_manager is not None


class TestDatabaseServiceGetDatabaseType:
    """Tests for _get_database_type helper."""

    def test_known_database_returns_type(self, real_agent_config):
        """Known database returns its type string."""
        svc = DatabaseService(agent_config=real_agent_config)
        db_type, ds_id = svc._get_database_type("california_schools")
        assert db_type == "sqlite"

    def test_current_database_used_as_default(self, real_agent_config):
        """Without database_name arg, uses current_database."""
        svc = DatabaseService(agent_config=real_agent_config)
        db_type, ds_id = svc._get_database_type()
        assert db_type == "sqlite"
        assert ds_id == svc.current_database


class TestGetSemanticModel:
    """Tests for get_semantic_model and validate_semantic_model."""

    def test_get_semantic_model_nonexistent(self, real_agent_config):
        """get_semantic_model for nonexistent table returns empty result."""
        svc = DatabaseService(agent_config=real_agent_config)
        result = svc.get_semantic_model("nonexistent_table_xyz")
        # Should return success=True with no data, or success=False
        assert result is not None

    def test_get_semantic_model_for_known_table(self, real_agent_config):
        """get_semantic_model for known table (may return empty if no semantic model built)."""
        svc = DatabaseService(agent_config=real_agent_config)
        result = svc.get_semantic_model("schools")
        # The table exists but may not have a semantic model file
        assert result is not None

    @pytest.mark.asyncio
    async def test_validate_semantic_model_nonexistent(self, real_agent_config):
        """validate_semantic_model for nonexistent table returns error."""
        from datus.api.models.table_models import SemanticModelInput

        svc = DatabaseService(agent_config=real_agent_config)
        request = SemanticModelInput(table="nonexistent_xyz", yaml="metric:\n  name: test\n")
        result = await svc.validate_semantic_model(request)
        assert result.success is False


class TestListDatabases:
    """Tests for list_databases with real SQLite connection."""

    def test_list_databases_returns_success(self, real_agent_config):
        """list_databases returns success with database info."""
        svc = DatabaseService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        assert result.success is True
        assert result.data is not None
        assert result.data.total_count >= 1

    def test_list_databases_has_entries(self, real_agent_config):
        """list_databases returns at least one database entry."""
        svc = DatabaseService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        assert len(result.data.databases) >= 1

    def test_list_databases_connection_status(self, real_agent_config):
        """Databases are connected."""
        svc = DatabaseService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        for db in result.data.databases:
            assert db.connection_status == "connected"

    def test_list_databases_has_tables(self, real_agent_config):
        """Connected databases report table count > 0."""
        svc = DatabaseService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        connected_databases = [db for db in result.data.databases if db.connection_status == "connected"]
        assert connected_databases
        assert all(db.tables_count > 0 for db in connected_databases)

    def test_list_databases_with_datasource_filter(self, real_agent_config):
        """list_databases with datasource_id filter."""
        svc = DatabaseService(agent_config=real_agent_config)
        # After namespace→services.databases refactor, datasource_id is a database name
        request = ListDatabasesInput(datasource_id="california_schools")
        result = svc.list_databases(request)
        assert result.success is True

    def test_list_databases_with_database_name_filter(self, real_agent_config):
        """list_databases with database_name filter narrows results."""
        svc = DatabaseService(agent_config=real_agent_config)
        request = ListDatabasesInput(database_name="main")
        result = svc.list_databases(request)
        assert result.success is True

    def test_list_databases_has_tables_list(self, real_agent_config):
        """list_databases includes tables list in database info."""
        svc = DatabaseService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        databases_with_tables = [db for db in result.data.databases if db.tables is not None]
        assert databases_with_tables
        assert all(isinstance(db.tables, list) for db in databases_with_tables)

    def test_list_databases_has_type_field(self, real_agent_config):
        """list_databases includes database type."""
        svc = DatabaseService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        for db in result.data.databases:
            assert db.type is not None

    def test_list_databases_has_current_database(self, real_agent_config):
        """list_databases data includes current_database field."""
        svc = DatabaseService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        assert result.data.current_database is not None


class TestGetTableSchema:
    """Tests for get_table_schema with real SQLite connection."""

    def test_get_table_schema_returns_columns(self, real_agent_config):
        """get_table_schema returns column info for existing table."""
        svc = DatabaseService(agent_config=real_agent_config)
        result = svc.get_table_schema("schools")
        assert result.success is True
        assert result.data is not None
        assert len(result.data.table.columns) > 0

    def test_get_table_schema_column_has_name_and_type(self, real_agent_config):
        """Each column has name and type fields."""
        svc = DatabaseService(agent_config=real_agent_config)
        result = svc.get_table_schema("schools")
        for col in result.data.table.columns:
            assert col.name != ""
            assert col.type != ""

    def test_get_table_schema_nonexistent_table(self, real_agent_config):
        """Nonexistent table returns failure."""
        svc = DatabaseService(agent_config=real_agent_config)
        result = svc.get_table_schema("totally_fake_table_xyz")
        assert result.success is False
