"""Unit tests for builtin_handlers.py — BigQuery, MSSQL, Oracle URI builders and context resolvers."""

from types import SimpleNamespace

import pytest

from datus.tools.db_tools.builtin_handlers import (
    _clean_str,
    _port_or_none,
    _value_or_none,
    build_bigquery_uri,
    build_mssql_uri,
    build_oracle_uri,
    resolve_bigquery_context,
    resolve_mssql_context,
    resolve_oracle_context,
)
from datus.utils.exceptions import DatusException


def _cfg(**kwargs):
    """Create a minimal config-like object with sensible defaults."""
    defaults = dict(
        host=None,
        port=None,
        username=None,
        password=None,
        database=None,
        schema=None,
        catalog=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Helpers
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
# BigQuery
# ---------------------------------------------------------------------------


class TestBuildBigqueryUri:
    def test_basic(self):
        cfg = _cfg(catalog="my-project", database="my_dataset")
        uri = build_bigquery_uri(cfg)
        assert "bigquery" in uri
        assert "my-project" in uri
        assert "my_dataset" in uri

    def test_host_fallback(self):
        cfg = _cfg(host="proj-from-host", database="ds")
        uri = build_bigquery_uri(cfg)
        assert "proj-from-host" in uri

    def test_schema_fallback_for_dataset(self):
        cfg = _cfg(catalog="proj", schema="ds-from-schema")
        uri = build_bigquery_uri(cfg)
        assert "ds-from-schema" in uri

    def test_missing_project_raises(self):
        cfg = _cfg(database="ds")
        with pytest.raises(DatusException):
            build_bigquery_uri(cfg)

    def test_missing_dataset_raises(self):
        cfg = _cfg(catalog="proj")
        with pytest.raises(DatusException):
            build_bigquery_uri(cfg)


class TestResolveBigqueryContext:
    def test_basic(self):
        cfg = _cfg(catalog="proj", database="ds")
        uri = build_bigquery_uri(cfg)
        dialect, catalog, database, schema = resolve_bigquery_context(cfg, uri)
        assert dialect == "bigquery"
        assert catalog == "proj"
        assert database == "ds"


# ---------------------------------------------------------------------------
# MSSQL
# ---------------------------------------------------------------------------


class TestBuildMssqlUri:
    def test_basic(self):
        cfg = _cfg(host="localhost", port=1433, username="sa", password="pass", database="mydb")
        uri = build_mssql_uri(cfg)
        assert "mssql+pyodbc" in uri
        assert "localhost" in uri
        assert "mydb" in uri

    def test_with_schema(self):
        cfg = _cfg(host="h", port=1433, username="u", password="p", database="db", schema="custom")
        uri = build_mssql_uri(cfg)
        assert "schema=custom" in uri

    def test_minimal(self):
        cfg = _cfg()
        uri = build_mssql_uri(cfg)
        assert "mssql+pyodbc" in uri


class TestResolveMssqlContext:
    def test_basic(self):
        cfg = _cfg(host="h", port=1433, username="u", password="p", database="mydb", schema="myschema")
        uri = build_mssql_uri(cfg)
        dialect, catalog, database, schema = resolve_mssql_context(cfg, uri)
        assert dialect == "mssql"
        assert catalog == ""
        assert database == "mydb"
        assert schema == "myschema"

    def test_default_schema(self):
        cfg = _cfg(host="h", username="u", password="p", database="db")
        uri = build_mssql_uri(cfg)
        dialect, catalog, database, schema = resolve_mssql_context(cfg, uri)
        assert schema == "dbo"


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


class TestBuildOracleUri:
    def test_with_service_name(self):
        cfg = _cfg(host="orahost", port=1521, username="sys", password="pass", database="ORCL")
        uri = build_oracle_uri(cfg)
        assert "oracle+cx_oracle" in uri
        assert "service_name=ORCL" in uri

    def test_with_sid(self):
        cfg = _cfg(host="orahost", port=1521, username="sys", password="pass", schema="XE")
        uri = build_oracle_uri(cfg)
        assert "sid=XE" in uri

    def test_minimal(self):
        cfg = _cfg()
        uri = build_oracle_uri(cfg)
        assert "oracle+cx_oracle" in uri


class TestResolveOracleContext:
    def test_basic(self):
        cfg = _cfg(host="h", port=1521, username="admin", password="p", database="ORCL", schema="HR")
        uri = build_oracle_uri(cfg)
        dialect, catalog, database, schema = resolve_oracle_context(cfg, uri)
        assert dialect == "oracle"
        assert catalog == ""
        assert database == "ORCL"

    def test_schema_fallback_to_username(self):
        cfg = _cfg(host="h", username="admin", password="p", database="SVC")
        uri = build_oracle_uri(cfg)
        dialect, catalog, database, schema = resolve_oracle_context(cfg, uri)
        assert schema == "admin"
