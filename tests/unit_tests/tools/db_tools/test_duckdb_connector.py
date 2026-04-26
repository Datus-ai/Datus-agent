"""Unit tests for DuckdbConnector.connect() — guards the in-process config
alignment that lets datus and SQLAlchemy+duckdb_engine clients coexist."""

import sys

import duckdb
import pytest
import sqlalchemy
from sqlalchemy.pool import StaticPool

from datus.tools.db_tools.config import DuckDBConfig
from datus.tools.db_tools.duckdb_connector import DuckdbConnector


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "smoke.duckdb")


def test_connect_and_close(db_path):
    connector = DuckdbConnector(DuckDBConfig(db_path=db_path))
    connector.connect()
    assert connector.connection is not None
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
        assert connector.connection is not None
        # Fallback path must NOT pass a config dict.
        assert "config" not in seen_kwargs
    finally:
        connector.close()
