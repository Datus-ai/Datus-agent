"""Unit tests for registry.py — ConnectorRegistry capabilities, handlers, aliases."""

from unittest.mock import MagicMock

import pytest

from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.db_tools.registry import ConnectorRegistry, connector_registry
from datus.utils.exceptions import DatusException


class TestResolveKey:
    def test_lowercase(self):
        assert ConnectorRegistry._resolve_key("MySQL") == "mysql"

    def test_postgres_alias(self):
        assert ConnectorRegistry._resolve_key("postgres") == "postgresql"

    def test_sqlserver_alias(self):
        assert ConnectorRegistry._resolve_key("sqlserver") == "mssql"

    def test_passthrough(self):
        assert ConnectorRegistry._resolve_key("snowflake") == "snowflake"


class TestSupportCapabilities:
    """Test that fallback and adapter-registered capabilities work."""

    def test_support_catalog_snowflake(self):
        assert connector_registry.support_catalog("snowflake") is True

    def test_support_database_mysql(self):
        assert connector_registry.support_database("mysql") is True

    def test_support_schema_postgresql(self):
        assert connector_registry.support_schema("postgresql") is True

    def test_support_schema_via_alias(self):
        assert connector_registry.support_schema("postgres") is True

    def test_no_catalog_for_mysql(self):
        assert connector_registry.support_catalog("mysql") is False

    def test_no_schema_for_clickhouse(self):
        assert connector_registry.support_schema("clickhouse") is False

    def test_hive_no_schema(self):
        assert connector_registry.support_schema("hive") is False

    def test_hive_no_database(self):
        assert connector_registry.support_database("hive") is False

    def test_unknown_dialect(self):
        assert connector_registry.support_catalog("unknown_db_xyz") is False
        assert connector_registry.support_database("unknown_db_xyz") is False
        assert connector_registry.support_schema("unknown_db_xyz") is False


class TestRegisterHandlers:
    def test_register_capabilities_only(self):
        connector_registry.register_handlers("test_dialect_abc", capabilities={"database"})
        assert connector_registry.support_database("test_dialect_abc") is True
        assert connector_registry.support_catalog("test_dialect_abc") is False
        # Cleanup
        ConnectorRegistry._capabilities.pop("test_dialect_abc", None)

    def test_register_with_uri_builder(self):
        mock_builder = MagicMock(return_value="test://uri")
        connector_registry.register_handlers("test_dialect_def", uri_builder=mock_builder)
        assert connector_registry.get_uri_builder("test_dialect_def") is mock_builder
        # Cleanup
        ConnectorRegistry._uri_builders.pop("test_dialect_def", None)

    def test_register_with_context_resolver(self):
        mock_resolver = MagicMock(return_value=("d", "c", "db", "s"))
        connector_registry.register_handlers("test_dialect_ghi", context_resolver=mock_resolver)
        assert connector_registry.get_context_resolver("test_dialect_ghi") is mock_resolver
        # Cleanup
        ConnectorRegistry._context_resolvers.pop("test_dialect_ghi", None)


class TestGetUriBuilder:
    def test_bigquery_has_builder(self):
        assert connector_registry.get_uri_builder("bigquery") is not None

    def test_mssql_has_builder(self):
        assert connector_registry.get_uri_builder("mssql") is not None

    def test_oracle_has_builder(self):
        assert connector_registry.get_uri_builder("oracle") is not None

    def test_mysql_no_builder(self):
        assert connector_registry.get_uri_builder("mysql") is None

    def test_sqlserver_alias_resolves(self):
        assert connector_registry.get_uri_builder("sqlserver") is not None


class TestGetContextResolver:
    def test_bigquery_has_resolver(self):
        assert connector_registry.get_context_resolver("bigquery") is not None

    def test_mssql_has_resolver(self):
        assert connector_registry.get_context_resolver("mssql") is not None

    def test_oracle_has_resolver(self):
        assert connector_registry.get_context_resolver("oracle") is not None

    def test_mysql_no_resolver(self):
        assert connector_registry.get_context_resolver("mysql") is None


class TestIsRegistered:
    def test_sqlite_registered(self):
        assert connector_registry.is_registered("sqlite") is True

    def test_duckdb_registered(self):
        assert connector_registry.is_registered("duckdb") is True

    def test_unknown_not_registered(self):
        assert connector_registry.is_registered("unknown_xyz_123") is False


class TestGetMetadata:
    def test_sqlite_metadata(self):
        meta = connector_registry.get_metadata("sqlite")
        assert meta is not None
        assert meta.db_type == "sqlite"

    def test_unknown_metadata_none(self):
        assert connector_registry.get_metadata("unknown_xyz_123") is None


class TestRegisterWithHandlers:
    """Test register() with uri_builder and context_resolver (lines 130, 132)."""

    def test_register_stores_uri_builder_and_context_resolver(self):
        mock_cls = MagicMock(spec=BaseSqlConnector)
        mock_cls.__name__ = "TestConn"
        mock_builder = MagicMock(return_value="test://uri")
        mock_resolver = MagicMock(return_value=("d", "", "db", "s"))
        ConnectorRegistry.register(
            "test_reg_full",
            mock_cls,
            uri_builder=mock_builder,
            context_resolver=mock_resolver,
        )
        assert ConnectorRegistry.get_uri_builder("test_reg_full") is mock_builder
        assert ConnectorRegistry.get_context_resolver("test_reg_full") is mock_resolver
        # Cleanup
        ConnectorRegistry._connectors.pop("test_reg_full", None)
        ConnectorRegistry._metadata.pop("test_reg_full", None)
        ConnectorRegistry._uri_builders.pop("test_reg_full", None)
        ConnectorRegistry._context_resolvers.pop("test_reg_full", None)


class TestCreateConnector:
    """Test create_connector() paths (lines 163, 176)."""

    def test_create_with_factory(self):
        mock_cls = MagicMock(spec=BaseSqlConnector)
        mock_cls.__name__ = "FactoryConn"
        mock_instance = MagicMock(spec=BaseSqlConnector)
        mock_factory = MagicMock(return_value=mock_instance)
        ConnectorRegistry.register("test_factory_db", mock_cls, factory=mock_factory)
        result = ConnectorRegistry.create_connector("test_factory_db", {"host": "h"})
        assert result is mock_instance
        mock_factory.assert_called_once_with({"host": "h"})
        # Cleanup
        ConnectorRegistry._connectors.pop("test_factory_db", None)
        ConnectorRegistry._factories.pop("test_factory_db", None)
        ConnectorRegistry._metadata.pop("test_factory_db", None)

    def test_create_triggers_dynamic_load(self):
        with pytest.raises(DatusException):
            ConnectorRegistry.create_connector("nonexistent_db_xyz_999", {})
