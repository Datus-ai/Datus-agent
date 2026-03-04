"""Unit tests for db_manager.py — gen_uri, _resolve_connection_context, helpers, and DBManager."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.tools.db_tools.base import BaseSqlConnector
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

    def test_sqlserver_alias(self):
        assert _normalize_dialect_name("sqlserver") == "mssql"

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
