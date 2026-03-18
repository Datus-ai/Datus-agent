"""Unit tests for db_manager.py — gen_uri, _resolve_connection_context, helpers, and DBManager."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from datus_db_core import BaseSqlConnector, DatusDbException
from datus_db_core import ErrorCode as DbErrorCode

from datus.tools.db_tools.db_manager import (
    DBManager,
    _clean_str,
    _normalize_dialect_name,
    _port_or_none,
    _resolve_connection_context,
    _value_or_none,
    db_config_name,
    gen_uri,
    get_connection,
)
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException


def _cfg(**kwargs):
    defaults = dict(
        type=None,
        host=None,
        port=None,
        username=None,
        password=None,
        database=None,
        schema=None,
        catalog=None,
        uri=None,
        logic_name="",
        extra=None,
        path_pattern=None,
    )
    defaults.update(kwargs)
    ns = SimpleNamespace(**defaults)
    ns.to_dict = lambda: {k: v for k, v in defaults.items()}
    return ns


# ---------------------------------------------------------------------------
# _normalize_dialect_name
# ---------------------------------------------------------------------------


class TestNormalizeDialectName:
    def test_string_lower(self):
        assert _normalize_dialect_name("MySQL") == "mysql"

    def test_postgres_alias(self):
        assert _normalize_dialect_name("postgres") == "postgresql"

    def test_none(self):
        assert _normalize_dialect_name(None) == ""

    def test_dbtype_enum(self):
        assert _normalize_dialect_name(DBType.SQLITE) == "sqlite"

    def test_dbtype_enum_duckdb(self):
        assert _normalize_dialect_name(DBType.DUCKDB) == "duckdb"

    def test_whitespace(self):
        assert _normalize_dialect_name("  mysql  ") == "mysql"


# ---------------------------------------------------------------------------
# _clean_str / _value_or_none / _port_or_none
# ---------------------------------------------------------------------------


class TestCleanStr:
    def test_none(self):
        assert _clean_str(None) == ""

    def test_string(self):
        assert _clean_str("  hello  ") == "hello"

    def test_int(self):
        assert _clean_str(3306) == "3306"

    def test_list_first_truthy(self):
        assert _clean_str(["", "val"]) == "val"

    def test_list_all_empty(self):
        assert _clean_str([None, ""]) == ""

    def test_set(self):
        result = _clean_str({"only"})
        assert result == "only"

    def test_tuple(self):
        assert _clean_str(("first",)) == "first"


class TestValueOrNone:
    def test_returns_value(self):
        assert _value_or_none("abc") == "abc"

    def test_returns_none_for_empty(self):
        assert _value_or_none("") is None

    def test_returns_none_for_none(self):
        assert _value_or_none(None) is None


class TestPortOrNone:
    def test_valid_int(self):
        assert _port_or_none(5432) == 5432

    def test_valid_string(self):
        assert _port_or_none("3306") == 3306

    def test_none(self):
        assert _port_or_none(None) is None

    def test_invalid(self):
        assert _port_or_none("abc") is None

    def test_empty(self):
        assert _port_or_none("") is None


# ---------------------------------------------------------------------------
# gen_uri
# ---------------------------------------------------------------------------


class TestGenUri:
    def test_returns_uri_if_set(self):
        cfg = _cfg(uri="sqlite:///test.db")
        assert gen_uri(cfg) == "sqlite:///test.db"

    def test_generic_fallback(self):
        cfg = _cfg(type="mysql", host="localhost", port=3306, username="root", password="pass", database="mydb")
        uri = gen_uri(cfg)
        assert "mysql" in uri
        assert "localhost" in uri

    def test_delegates_to_registered_builder(self):
        cfg = _cfg(type="bigquery", catalog="proj", database="ds")
        uri = gen_uri(cfg)
        assert "bigquery" in uri

    def test_postgres_alias_in_uri(self):
        cfg = _cfg(type="postgres", host="localhost", port=5432, username="u", password="p", database="db")
        uri = gen_uri(cfg)
        assert "postgresql" in uri

    def test_builder_reraises_db_exception(self, monkeypatch):
        """DatusDbException from a registered URI builder is re-raised as-is."""
        import datus_db_core.registry as reg_mod

        def _boom(_cfg):
            raise DatusDbException(code=DbErrorCode.COMMON_CONFIG_ERROR, message="builder boom")

        monkeypatch.setattr(reg_mod.ConnectorRegistry, "get_uri_builder", classmethod(lambda cls, dt: _boom))
        cfg = _cfg(type="mysql", host="localhost", database="db")
        with pytest.raises(DatusDbException, match="builder boom"):
            gen_uri(cfg)

    def test_builder_wraps_generic_exception(self, monkeypatch):
        """Generic exception from a registered URI builder is wrapped in DatusException."""
        import datus_db_core.registry as reg_mod

        def _boom(_cfg):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(reg_mod.ConnectorRegistry, "get_uri_builder", classmethod(lambda cls, dt: _boom))
        cfg = _cfg(type="mysql", host="localhost", database="db")
        with pytest.raises(DatusException, match="URI builder failed"):
            gen_uri(cfg)


# ---------------------------------------------------------------------------
# _resolve_connection_context
# ---------------------------------------------------------------------------


class TestResolveConnectionContext:
    def test_generic_fallback(self):
        cfg = _cfg(type="mysql", catalog="", database="mydb", schema="")
        uri = "mysql://root@localhost/mydb"
        dialect, catalog, database, schema = _resolve_connection_context(cfg, uri)
        assert dialect == "mysql"
        assert database == "mydb"

    def test_delegates_to_registered_resolver(self):
        cfg = _cfg(type="bigquery", catalog="proj", database="ds", schema="")
        uri = "bigquery://proj/ds"
        dialect, catalog, database, schema = _resolve_connection_context(cfg, uri)
        assert dialect == "bigquery"
        assert catalog == "proj"

    def test_invalid_uri_raises(self):
        cfg = _cfg(type="mysql")
        with pytest.raises(DatusException):
            _resolve_connection_context(cfg, "not-a-valid-uri://[[[")

    def test_catalog_from_config(self):
        cfg = _cfg(type="mysql", catalog="my_catalog", database="mydb", schema="myschema")
        uri = "mysql://root@localhost/mydb"
        dialect, catalog, database, schema = _resolve_connection_context(cfg, uri)
        assert catalog == "my_catalog"
        assert schema == "myschema"

    def test_resolver_reraises_db_exception(self, monkeypatch):
        """DatusDbException from a registered context resolver is re-raised as-is."""
        import datus_db_core.registry as reg_mod

        def _boom(_cfg, _uri):
            raise DatusDbException(code=DbErrorCode.COMMON_CONFIG_ERROR, message="resolver boom")

        monkeypatch.setattr(reg_mod.ConnectorRegistry, "get_context_resolver", classmethod(lambda cls, dt: _boom))
        cfg = _cfg(type="mysql", database="mydb")
        with pytest.raises(DatusDbException, match="resolver boom"):
            _resolve_connection_context(cfg, "mysql://root@localhost/mydb")

    def test_resolver_wraps_generic_exception(self, monkeypatch):
        """Generic exception from a registered context resolver is wrapped in DatusException."""
        import datus_db_core.registry as reg_mod

        def _boom(_cfg, _uri):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(reg_mod.ConnectorRegistry, "get_context_resolver", classmethod(lambda cls, dt: _boom))
        cfg = _cfg(type="mysql", database="mydb")
        with pytest.raises(DatusException, match="Context resolver failed"):
            _resolve_connection_context(cfg, "mysql://root@localhost/mydb")


# ---------------------------------------------------------------------------
# get_connection
# ---------------------------------------------------------------------------


class TestGetConnection:
    def test_returns_single_connector(self):
        mock_conn = MagicMock(spec=BaseSqlConnector)
        result = get_connection(mock_conn)
        assert result is mock_conn

    def test_returns_from_dict_single(self):
        mock_conn = MagicMock()
        result = get_connection({"db1": mock_conn})
        assert result is mock_conn

    def test_returns_first_when_no_logic_name(self):
        c1, c2 = MagicMock(), MagicMock()
        result = get_connection({"a": c1, "b": c2}, "")
        assert result is c1

    def test_returns_named(self):
        c1, c2 = MagicMock(), MagicMock()
        result = get_connection({"a": c1, "b": c2}, "b")
        assert result is c2

    def test_raises_for_missing_name(self):
        c1 = MagicMock()
        with pytest.raises(DatusException):
            get_connection({"a": c1, "b": MagicMock()}, "c")


# ---------------------------------------------------------------------------
# db_config_name
# ---------------------------------------------------------------------------


class TestDbConfigName:
    def test_sqlite(self):
        result = db_config_name("ns", "sqlite", "myfile")
        assert result == "ns::myfile"

    def test_duckdb(self):
        result = db_config_name("ns", "duckdb", "myfile")
        assert result == "ns::myfile"

    def test_other(self):
        result = db_config_name("ns", "mysql", "ignored")
        assert result == "ns::ns"


# ---------------------------------------------------------------------------
# DBManager
# ---------------------------------------------------------------------------


class TestDBManager:
    def test_context_manager(self):
        mgr = DBManager({})
        with mgr as m:
            assert m is mgr

    def test_missing_namespace_raises(self):
        mgr = DBManager({})
        with pytest.raises(DatusException):
            mgr.get_conn("nonexistent")

    def test_current_db_configs(self):
        configs = {"ns": {"db1": _cfg(type="sqlite", database="test.db")}}
        mgr = DBManager(configs)
        assert "db1" in mgr.current_db_configs("ns")

    def test_get_db_uris(self):
        configs = {"ns": {"db1": _cfg(type="sqlite", uri="sqlite:///test.db")}}
        mgr = DBManager(configs)
        uris = mgr.get_db_uris("ns")
        assert uris["db1"] == "sqlite:///test.db"


# ---------------------------------------------------------------------------
# DBManager._db_config_to_connection_config — adapter branch (lines 269-309)
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestDbConfigToConnectionConfigAdapterBranch:
    """Tests for the adapter (non-SQLite, non-DuckDB) branch of _db_config_to_connection_config."""

    def _make_manager(self):
        return DBManager({})

    def test_adapter_returns_dict_not_connection_config(self):
        """Non-SQLite/DuckDB type returns a plain dict, not a ConnectionConfig subclass."""
        mgr = self._make_manager()
        cfg = _cfg(type="postgresql", host="localhost", port="5432", username="user", password="pass", database="mydb")
        result = mgr._db_config_to_connection_config(cfg)
        assert isinstance(result, dict)

    def test_adapter_excluded_fields_removed(self):
        """Excluded fields (type, path_pattern, logic_name, extra) are not in the result dict."""
        mgr = self._make_manager()
        cfg = _cfg(
            type="postgresql",
            host="localhost",
            database="mydb",
            logic_name="pg_main",
            path_pattern="/some/pattern",
            extra=None,
        )
        result = mgr._db_config_to_connection_config(cfg)
        assert "type" not in result
        assert "logic_name" not in result
        assert "path_pattern" not in result
        assert "extra" not in result

    def test_adapter_port_converted_to_int(self):
        """Port value provided as string is converted to int in the result."""
        mgr = self._make_manager()
        cfg = _cfg(type="postgresql", host="localhost", port="5432", database="mydb")
        result = mgr._db_config_to_connection_config(cfg)
        assert result.get("port") == 5432
        assert isinstance(result.get("port"), int)

    def test_adapter_port_already_int_stays_int(self):
        """Port value provided as int remains an int."""
        mgr = self._make_manager()
        cfg = _cfg(type="mysql", host="localhost", port=3306, database="mydb")
        result = mgr._db_config_to_connection_config(cfg)
        assert result.get("port") == 3306
        assert isinstance(result.get("port"), int)

    def test_adapter_extra_fields_expanded(self):
        """Extra dict fields are merged into the result config."""
        mgr = self._make_manager()
        cfg = _cfg(
            type="snowflake",
            host="acct.snowflakecomputing.com",
            database="mydb",
            extra={"warehouse": "COMPUTE_WH", "role": "ANALYST"},
        )
        result = mgr._db_config_to_connection_config(cfg)
        assert result.get("warehouse") == "COMPUTE_WH"
        assert result.get("role") == "ANALYST"

    def test_adapter_none_values_removed(self):
        """None-valued fields are excluded from the result dict."""
        mgr = self._make_manager()
        cfg = _cfg(type="mysql", host="localhost", database="mydb", username=None, password=None)
        result = mgr._db_config_to_connection_config(cfg)
        assert "username" not in result
        assert "password" not in result

    def test_adapter_empty_string_values_removed(self):
        """Empty-string-valued fields (after strip) are excluded from the result dict."""
        mgr = self._make_manager()
        cfg = _cfg(type="mysql", host="localhost", database="mydb", schema="", catalog="  ")
        result = mgr._db_config_to_connection_config(cfg)
        assert "schema" not in result
        assert "catalog" not in result

    def test_adapter_timeout_seconds_added(self):
        """timeout_seconds is always added to the result dict for adapter configs."""
        mgr = self._make_manager()
        cfg = _cfg(type="postgresql", host="localhost", database="mydb")
        result = mgr._db_config_to_connection_config(cfg)
        assert "timeout_seconds" in result
        assert isinstance(result["timeout_seconds"], int)

    def test_adapter_extra_none_does_not_expand(self):
        """When extra is None, no extra fields are added and no error is raised."""
        mgr = self._make_manager()
        cfg = _cfg(type="mysql", host="localhost", database="mydb", extra=None)
        result = mgr._db_config_to_connection_config(cfg)
        assert isinstance(result, dict)
        # extra key itself should not be present
        assert "extra" not in result

    def test_adapter_invalid_port_string_not_converted(self):
        """Invalid port string that cannot be int-cast is left unchanged (no error)."""
        mgr = self._make_manager()
        cfg = _cfg(type="postgresql", host="localhost", database="mydb", port="not_a_port")
        # Should not raise
        result = mgr._db_config_to_connection_config(cfg)
        # Port stays as original value since conversion fails silently
        assert "port" in result
