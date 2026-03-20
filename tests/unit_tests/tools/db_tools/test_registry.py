"""Unit tests for registry.py — ConnectorRegistry capabilities, handlers, aliases."""

from unittest.mock import MagicMock

import pytest
from datus_db_core import BaseSqlConnector, ConnectorRegistry, DatusDbException, connector_registry

import datus.tools.db_tools  # noqa: F401 — triggers builtin connector registration


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot and restore ConnectorRegistry class-level dicts around each test."""
    attrs = ("_connectors", "_factories", "_metadata", "_capabilities", "_uri_builders", "_context_resolvers")
    snapshots = {a: getattr(ConnectorRegistry, a).copy() for a in attrs}
    yield
    for a, snap in snapshots.items():
        setattr(ConnectorRegistry, a, snap)


class TestResolveKey:
    def test_lowercase(self):
        assert ConnectorRegistry._resolve_key("MySQL") == "mysql"

    def test_postgres_alias(self):
        assert ConnectorRegistry._resolve_key("postgres") == "postgresql"

    def test_passthrough(self):
        assert ConnectorRegistry._resolve_key("snowflake") == "snowflake"


class TestSupportCapabilities:
    """Test register_handlers + support_*() query mechanism."""

    def test_catalog_database_schema(self):
        connector_registry.register_handlers("test_full", capabilities={"catalog", "database", "schema"})
        assert connector_registry.support_catalog("test_full") is True
        assert connector_registry.support_database("test_full") is True
        assert connector_registry.support_schema("test_full") is True

    def test_database_only(self):
        connector_registry.register_handlers("test_db_only", capabilities={"database"})
        assert connector_registry.support_database("test_db_only") is True
        assert connector_registry.support_catalog("test_db_only") is False
        assert connector_registry.support_schema("test_db_only") is False

    def test_empty_capabilities(self):
        connector_registry.register_handlers("test_empty_caps", capabilities=set())
        assert connector_registry.support_catalog("test_empty_caps") is False
        assert connector_registry.support_database("test_empty_caps") is False
        assert connector_registry.support_schema("test_empty_caps") is False

    def test_alias_resolution(self):
        connector_registry.register_handlers("postgresql", capabilities={"database", "schema"})
        assert connector_registry.support_schema("postgres") is True

    def test_unknown_dialect_returns_false(self):
        assert connector_registry.support_catalog("unknown_db_xyz") is False
        assert connector_registry.support_database("unknown_db_xyz") is False
        assert connector_registry.support_schema("unknown_db_xyz") is False


class TestRegisterHandlers:
    def test_register_capabilities_only(self):
        connector_registry.register_handlers("test_dialect_abc", capabilities={"database"})
        assert connector_registry.support_database("test_dialect_abc") is True
        assert connector_registry.support_catalog("test_dialect_abc") is False

    def test_register_with_uri_builder(self):
        mock_builder = MagicMock(return_value="test://uri")
        connector_registry.register_handlers("test_dialect_def", uri_builder=mock_builder)
        assert connector_registry.get_uri_builder("test_dialect_def") is mock_builder

    def test_register_with_context_resolver(self):
        mock_resolver = MagicMock(return_value=("d", "c", "db", "s"))
        connector_registry.register_handlers("test_dialect_ghi", context_resolver=mock_resolver)
        assert connector_registry.get_context_resolver("test_dialect_ghi") is mock_resolver


class TestGetUriBuilder:
    def test_mysql_no_builder(self):
        assert connector_registry.get_uri_builder("mysql") is None


class TestGetContextResolver:
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

    def test_create_triggers_dynamic_load(self):
        with pytest.raises(DatusDbException):
            ConnectorRegistry.create_connector("nonexistent_db_xyz_999", {})
