"""Unit tests for DuckdbConnector.connect() — guards the in-process config
alignment that lets datus and SQLAlchemy+duckdb_engine clients coexist."""

import sys
import types

import duckdb
import pytest
import sqlalchemy
from sqlalchemy.pool import StaticPool

from datus.tools.db_tools.config import DuckDBConfig
from datus.tools.db_tools.duckdb_connector import DuckdbConnector
from datus.utils.exceptions import DatusException


def _fake_duckdb_engine(monkeypatch):
    module = types.ModuleType("duckdb_engine")
    module.__version__ = "test"
    monkeypatch.setitem(sys.modules, "duckdb_engine", module)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "smoke.duckdb")


def test_connect_and_close(db_path):
    connector = DuckdbConnector(DuckDBConfig(db_path=db_path))
    connector.connect()
    assert isinstance(connector.connection, duckdb.DuckDBPyConnection)
    connector.connection.execute("SELECT 1").fetchone()

    connector.close()
    assert connector.connection is None


def test_coexist_with_sqlalchemy_duckdb_engine(db_path):
    """Regression: a second connection via SQLAlchemy+duckdb_engine on the same
    file in the same process must succeed. Pre-fix, DuckDB rejected it with
    'Can't open a connection to same database file with a different configuration'.
    """
    connector = DuckdbConnector(DuckDBConfig(db_path=db_path))
    connector.connect()
    connector.connection.execute("CREATE TABLE t(x INT)")
    connector.connection.execute("INSERT INTO t VALUES (1)")

    engine = sqlalchemy.create_engine(f"duckdb:///{db_path}", poolclass=StaticPool)
    try:
        raw = engine.raw_connection()
        cur = raw.cursor()
        cur.execute("SELECT COUNT(*) FROM t")
        assert cur.fetchone()[0] == 1
        raw.close()
    finally:
        engine.dispose()
        connector.close()


def test_falls_back_when_duckdb_engine_missing(db_path, monkeypatch):
    """If duckdb_engine isn't installed, connect() falls back to a bare
    duckdb.connect(self.db_path) without aligning custom_user_agent."""
    monkeypatch.setitem(sys.modules, "duckdb_engine", None)

    seen_kwargs: dict = {}
    original_connect = duckdb.connect

    def spy_connect(path, **kwargs):
        seen_kwargs.update(kwargs)
        return original_connect(path, **kwargs)

    monkeypatch.setattr(duckdb, "connect", spy_connect)

    connector = DuckdbConnector(DuckDBConfig(db_path=db_path))
    connector.connect()
    try:
        assert isinstance(connector.connection, duckdb.DuckDBPyConnection)
        # Fallback path must NOT pass a config dict.
        assert "config" not in seen_kwargs
    finally:
        connector.close()


def test_connect_passes_read_only(db_path, monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    seen_kwargs: dict = {}

    class FakeConnection:
        def execute(self, *_args, **_kwargs):
            return self

        def close(self):
            pass

    def spy_connect(path, **kwargs):
        seen_kwargs["path"] = path
        seen_kwargs.update(kwargs)
        return FakeConnection()

    monkeypatch.setattr(duckdb, "connect", spy_connect)

    connector = DuckdbConnector(DuckDBConfig(db_path=db_path, read_only=True))
    connector.connect()
    try:
        assert seen_kwargs["path"] == db_path
        assert seen_kwargs["read_only"] is True
    finally:
        connector.close()


def test_execute_query_arrow_counts_rows(db_path):
    connector = DuckdbConnector(DuckDBConfig(db_path=db_path))
    connector.execute_ddl("CREATE TABLE t AS SELECT 1 AS x UNION ALL SELECT 2 AS x")

    result = connector.execute_query("SELECT * FROM t ORDER BY x", result_format="arrow")

    try:
        assert result.success is True
        assert result.row_count == 2
        assert result.sql_return.num_rows == 2
    finally:
        connector.close()


def test_connect_sets_up_iceberg_catalog(monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    executed: list[str] = []

    class FakeConnection:
        def execute(self, sql):
            executed.append(sql)
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "http://127.0.0.1:8181",
                "warehouse": "s3://warehouse/",
                "s3_endpoint": "http://127.0.0.1:9000",
                "s3_access_key_id": "admin",
                "s3_secret_access_key": "password",
                "s3_url_style": "path",
            },
        )
    )
    connector.connect()
    try:
        assert "LOAD httpfs" in executed
        assert "LOAD iceberg" in executed
        assert any("CREATE OR REPLACE SECRET datus_s3" in sql for sql in executed)
        assert any("ATTACH 's3://warehouse/' AS lake" in sql for sql in executed)
        assert any("READ_ONLY false" in sql for sql in executed)
    finally:
        connector.close()


def test_connect_sets_up_iceberg_catalog_infers_ssl_from_https_endpoint(monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    executed: list[str] = []

    class FakeConnection:
        def execute(self, sql):
            executed.append(sql)
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "http://127.0.0.1:8181",
                "warehouse": "s3://warehouse/",
                "s3_endpoint": "https://s3.example.com",
                "s3_access_key_id": "admin",
                "s3_secret_access_key": "password",
            },
        )
    )
    connector.connect()
    try:
        secret_sql = next(sql for sql in executed if "CREATE OR REPLACE SECRET datus_s3" in sql)
        assert "ENDPOINT 's3.example.com'" in secret_sql
        assert "USE_SSL true" in secret_sql
    finally:
        connector.close()


def test_iceberg_execute_ddl_rewrites_create_or_replace_table(monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    executed: list[str] = []

    class FakeConnection:
        def execute(self, sql):
            executed.append(sql)
            if sql.startswith("CREATE OR REPLACE TABLE"):
                raise RuntimeError("CREATE OR REPLACE not supported in DuckDB-Iceberg")
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "http://127.0.0.1:8181",
                "warehouse": "s3://warehouse/",
            },
        )
    )

    result = connector.execute_ddl("CREATE OR REPLACE TABLE lake.ws.table AS SELECT 1 AS x")

    try:
        assert result.success is True
        assert executed[-3:] == [
            "BEGIN",
            "DROP TABLE IF EXISTS lake.ws.table;\nCREATE TABLE lake.ws.table AS SELECT 1 AS x",
            "COMMIT",
        ]
    finally:
        connector.close()


def test_iceberg_execute_ddl_rewrites_quoted_create_or_replace_table(monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    executed: list[str] = []

    class FakeConnection:
        def execute(self, sql):
            executed.append(sql)
            if sql.startswith("CREATE OR REPLACE TABLE"):
                raise RuntimeError("CREATE OR REPLACE not supported in DuckDB-Iceberg")
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "http://127.0.0.1:8181",
                "warehouse": "s3://warehouse/",
            },
        )
    )

    result = connector.execute_ddl('CREATE OR REPLACE TABLE "lake"."raw"."my-table" AS SELECT 1 AS x')

    try:
        assert result.success is True
        assert executed[-3:] == [
            "BEGIN",
            'DROP TABLE IF EXISTS "lake"."raw"."my-table";\nCREATE TABLE "lake"."raw"."my-table" AS SELECT 1 AS x',
            "COMMIT",
        ]
    finally:
        connector.close()


def test_iceberg_execute_ddl_rewrite_rolls_back_on_create_failure(monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    executed: list[str] = []

    class FakeConnection:
        def execute(self, sql):
            executed.append(sql)
            if sql.startswith("CREATE OR REPLACE TABLE"):
                raise RuntimeError("CREATE OR REPLACE not supported in DuckDB-Iceberg")
            if sql.startswith("DROP TABLE IF EXISTS"):
                raise RuntimeError("create failed")
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "http://127.0.0.1:8181",
                "warehouse": "s3://warehouse/",
            },
        )
    )

    result = connector.execute_ddl("CREATE OR REPLACE TABLE lake.ws.table AS SELECT invalid")

    try:
        assert result.success is False
        assert "original error: CREATE OR REPLACE not supported in DuckDB-Iceberg" in result.error
        assert "rewrite error: create failed" in result.error
        assert executed[-1] == "ROLLBACK"
    finally:
        connector.close()


def test_load_extension_reports_initial_load_error_when_retry_fails():
    class FakeConnection:
        def execute(self, sql):
            if sql == "LOAD httpfs":
                raise RuntimeError("read-only extension directory")
            if sql == "INSTALL httpfs":
                raise RuntimeError("install failed")
            return self

    connector = DuckdbConnector(DuckDBConfig(db_path=":memory:"))
    connector.connection = FakeConnection()

    with pytest.raises(RuntimeError, match="initial LOAD error: read-only extension directory"):
        connector._load_extension("httpfs")


def test_connect_sets_up_iceberg_catalog_with_credential_chain(monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    executed: list[str] = []

    class FakeConnection:
        def execute(self, sql):
            executed.append(sql)
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "http://127.0.0.1:8181",
                "warehouse": "s3://example-datus-demo/warehouse/",
                "s3_provider": "credential_chain",
                "s3_region": "us-east-1",
            },
        )
    )
    connector.connect()
    try:
        secret_sql = next(sql for sql in executed if "CREATE OR REPLACE SECRET datus_s3" in sql)
        assert "PROVIDER credential_chain" in secret_sql
        assert "REGION 'us-east-1'" in secret_sql
        assert "USE_SSL true" in secret_sql
        assert "KEY_ID" not in secret_sql
        assert "SECRET '" not in secret_sql
        assert any("ATTACH 's3://example-datus-demo/warehouse/' AS lake" in sql for sql in executed)
    finally:
        connector.close()


def test_iceberg_rejects_invalid_s3_provider(monkeypatch):
    _fake_duckdb_engine(monkeypatch)

    class FakeConnection:
        def execute(self, _sql):
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "http://127.0.0.1:8181",
                "warehouse": "s3://warehouse/",
                "s3_provider": "credential_chain; DROP SECRET datus_s3",
            },
        )
    )

    with pytest.raises(DatusException, match="s3_provider"):
        connector.connect()


def test_connect_sets_up_iceberg_catalog_with_oauth_secret(monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    executed: list[str] = []

    class FakeConnection:
        def execute(self, sql):
            executed.append(sql)
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "https://catalog.example.com",
                "warehouse": "public_catalog",
                "client_id": "lakehouse_user",
                "client_secret": "user-secret",
                "oauth2_server_uri": "https://catalog.example.com/oauth/tokens",
                "access_delegation_mode": "vended_credentials",
            },
        )
    )
    connector.connect()
    try:
        secret_sql = next(sql for sql in executed if "CREATE OR REPLACE SECRET datus_iceberg" in sql)
        assert "TYPE iceberg" in secret_sql
        assert "CLIENT_ID 'lakehouse_user'" in secret_sql
        assert "CLIENT_SECRET 'user-secret'" in secret_sql
        assert "OAUTH2_SERVER_URI 'https://catalog.example.com/oauth/tokens'" in secret_sql
        attach_sql = next(sql for sql in executed if "ATTACH 'public_catalog' AS lake" in sql)
        assert "SECRET datus_iceberg" in attach_sql
        assert "ACCESS_DELEGATION_MODE 'vended_credentials'" in attach_sql
        assert "AUTHORIZATION_TYPE none" not in attach_sql
    finally:
        connector.close()


def test_iceberg_create_secret_false_without_secret_name_does_not_attach_uncreated_secret(monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    executed: list[str] = []

    class FakeConnection:
        def execute(self, sql):
            executed.append(sql)
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "https://catalog.example.com",
                "warehouse": "public_catalog",
                "client_id": "lakehouse_user",
                "client_secret": "user-secret",
                "create_iceberg_secret": False,
            },
        )
    )
    connector.connect()
    try:
        assert not any("CREATE OR REPLACE SECRET datus_iceberg" in sql for sql in executed)
        attach_sql = next(sql for sql in executed if "ATTACH 'public_catalog' AS lake" in sql)
        assert "SECRET datus_iceberg" not in attach_sql
        assert "AUTHORIZATION_TYPE none" in attach_sql
    finally:
        connector.close()


def test_iceberg_create_secret_false_uses_explicit_existing_secret(monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    executed: list[str] = []

    class FakeConnection:
        def execute(self, sql):
            executed.append(sql)
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "https://catalog.example.com",
                "warehouse": "public_catalog",
                "secret_name": "existing_iceberg_secret",
                "create_iceberg_secret": False,
            },
        )
    )
    connector.connect()
    try:
        assert not any("CREATE OR REPLACE SECRET existing_iceberg_secret" in sql for sql in executed)
        attach_sql = next(sql for sql in executed if "ATTACH 'public_catalog' AS lake" in sql)
        assert "SECRET existing_iceberg_secret" in attach_sql
    finally:
        connector.close()


def test_iceberg_explicit_secret_name_without_credentials_does_not_attach_uncreated_secret(monkeypatch):
    _fake_duckdb_engine(monkeypatch)
    executed: list[str] = []

    class FakeConnection:
        def execute(self, sql):
            executed.append(sql)
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "lake",
                "catalog_uri": "https://catalog.example.com",
                "warehouse": "public_catalog",
                "secret_name": "uncreated_secret",
            },
        )
    )
    connector.connect()
    try:
        assert not any("CREATE OR REPLACE SECRET uncreated_secret" in sql for sql in executed)
        attach_sql = next(sql for sql in executed if "ATTACH 'public_catalog' AS lake" in sql)
        assert "SECRET uncreated_secret" not in attach_sql
        assert "AUTHORIZATION_TYPE none" in attach_sql
    finally:
        connector.close()


def test_iceberg_rejects_invalid_catalog_alias(monkeypatch):
    _fake_duckdb_engine(monkeypatch)

    class FakeConnection:
        def execute(self, _sql):
            return self

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FakeConnection())

    connector = DuckdbConnector(
        DuckDBConfig(
            db_path=":memory:",
            iceberg={
                "catalog_alias": "bad-name",
                "catalog_uri": "http://127.0.0.1:8181",
                "warehouse": "s3://warehouse/",
            },
        )
    )
    with pytest.raises(Exception, match="Failed to establish connection"):
        connector.connect()
