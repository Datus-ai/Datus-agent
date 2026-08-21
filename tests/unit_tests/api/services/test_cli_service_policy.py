"""SQL policy on the IDE's SQL console (``POST /api/v1/sql/execute``).

Red before the fix: ``_execute_sql_sync`` called ``connector.execute`` directly,
so a project with a row filter on ``orders`` still answered a hand-written
``SELECT * FROM orders`` with every row.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.api.models.cli_models import ExecuteSQLInput
from datus.api.services.cli_service import CLIService


class _Connector:
    dialect = "postgres"

    def __init__(self):
        self.executed = []

    def execute(self, input_params, result_format="list"):
        self.executed.append(input_params["sql_query"])
        return SimpleNamespace(
            success=True,
            sql_return=[],
            row_count=0,
            error=None,
            sql_query=input_params["sql_query"],
        )


def _service(monkeypatch, connector, db_tool):
    svc = CLIService.__new__(CLIService)
    svc.agent_config = SimpleNamespace()
    svc.current_db_connector = connector
    svc.current_datasource = "warehouse"
    svc._db_tool_cache = db_tool
    svc._sql_tasks = {}
    svc._sql_tasks_lock = MagicMock()
    svc._sql_tasks_lock.__enter__ = lambda *_: None
    svc._sql_tasks_lock.__exit__ = lambda *_: None
    return svc


def test_a_read_goes_through_the_enforced_path(monkeypatch):
    """The whole point: the console cannot be the one unfiltered door."""
    connector = _Connector()
    db_tool = MagicMock()
    db_tool.execute_read_enforced.return_value = SimpleNamespace(
        success=True, sql_return=[], row_count=0, error=None, sql_query="rewritten"
    )
    svc = _service(monkeypatch, connector, db_tool)

    svc._execute_sql_sync(
        ExecuteSQLInput(sql_query="SELECT * FROM orders"),
        "task-1",
        {"row_filter": {"access_mode": "scoped", "store_ids": ["S001"]}},
    )

    db_tool.execute_read_enforced.assert_called_once()
    kwargs = db_tool.execute_read_enforced.call_args.kwargs
    # The caller's context, not the service's: DatusService is cached per
    # project and shared, so its AgentConfig carries none.
    assert kwargs["policy_context"] == {"row_filter": {"access_mode": "scoped", "store_ids": ["S001"]}}
    assert connector.executed == []


def test_a_write_keeps_the_connector_path(monkeypatch):
    """The console is also where people write DDL.

    The enforced path admits only SELECT / SHOW / DESCRIBE / EXPLAIN, so routing
    everything through it would silently turn the console read-only. Row
    policies constrain reads by definition, so the split costs nothing.
    """
    connector = _Connector()
    db_tool = MagicMock()
    svc = _service(monkeypatch, connector, db_tool)

    svc._execute_sql_sync(ExecuteSQLInput(sql_query="CREATE VIEW v AS SELECT 1"), "task-2", {})

    db_tool.execute_read_enforced.assert_not_called()
    assert connector.executed == ["CREATE VIEW v AS SELECT 1"]


@pytest.mark.parametrize("sql", ["SHOW TABLES", "DESCRIBE orders", "EXPLAIN SELECT 1"])
def test_metadata_reads_are_enforced_too(monkeypatch, sql):
    connector = _Connector()
    db_tool = MagicMock()
    db_tool.execute_read_enforced.return_value = SimpleNamespace(success=True, sql_return=[], error=None, sql_query=sql)
    svc = _service(monkeypatch, connector, db_tool)

    svc._execute_sql_sync(ExecuteSQLInput(sql_query=sql), "task-4", {})

    db_tool.execute_read_enforced.assert_called_once()


def test_the_tool_is_built_once(monkeypatch):
    """Its constructor stands up a DBManager and three RAG indexes.

    Paying that per click on Run is what the memo avoids.
    """
    svc = _service(monkeypatch, _Connector(), None)
    built = []

    class _Tool:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr("datus.tools.func_tool.DBFuncTool", _Tool)

    assert svc._db_tool() is svc._db_tool()
    assert len(built) == 1
    # No sub-agent name: `cli_sql` names nothing, and it feeds three RAG
    # constructors whose scoped context would silently apply if a project ever
    # had a sub-agent by that name.
    assert "sub_agent_name" not in built[0]


def test_multi_statement_does_not_reach_the_connector(monkeypatch):
    """One extra statement must not be a way around the filter.

    `SELECT 1; SELECT * FROM orders` is not a read to the read-only validator —
    it is `multi_statement` — so dispatching on "did the validator object" hands
    it to the connector, which executes both and returns the last result set.
    Dispatch is on the statement *type* instead, so this lands on the enforced
    path and is refused there. Losing the two-statement selection is the price;
    Split Execute already sends them one at a time.
    """
    connector = _Connector()
    db_tool = MagicMock()
    db_tool.execute_read_enforced.return_value = SimpleNamespace(
        success=False, sql_return=[], row_count=0, error="multi", sql_query="x"
    )
    svc = _service(monkeypatch, connector, db_tool)

    svc._execute_sql_sync(ExecuteSQLInput(sql_query="SELECT 1; SELECT * FROM orders"), "t", {})

    assert connector.executed == []
    db_tool.execute_read_enforced.assert_called_once()


def test_unparseable_sql_does_not_reach_the_connector(monkeypatch):
    """A deliberate typo must not turn into an unfiltered read.

    Anything sqlglot cannot place comes back as UNKNOWN, which the read-only
    validator also reports as `non_read`. Treating that as "a write" would make
    every dialect quirk a bypass.
    """
    connector = _Connector()
    db_tool = MagicMock()
    db_tool.execute_read_enforced.return_value = SimpleNamespace(
        success=False, sql_return=[], row_count=0, error="nope", sql_query="x"
    )
    svc = _service(monkeypatch, connector, db_tool)

    svc._execute_sql_sync(ExecuteSQLInput(sql_query="SELEKT * FROM orders"), "t", {})

    assert connector.executed == []


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE mine AS SELECT * FROM orders",
        "INSERT INTO mine SELECT * FROM orders",
        # Legal Postgres, equivalent to the CTAS above, but sqlglot cannot
        # parse it and hands back an opaque `Command` with no Select inside.
        "CREATE TABLE mine AS TABLE orders",
        # Parsed, but its source is a table reference rather than a Select —
        # and `TO '/path'` / `TO PROGRAM` on a self-hosted PG is a real export.
        "COPY orders TO '/tmp/orders.csv'",
    ],
)
def test_a_write_that_reads_is_refused_when_policies_exist(monkeypatch, sql):
    """Laundering: copy the filtered rows into a table no policy covers.

    The plugin cannot see this — it hooks reads, and these are writes — so the
    refusal lives here, and only on a project that has a policy context at all.
    """
    connector = _Connector()
    svc = _service(monkeypatch, connector, MagicMock())

    result = svc._execute_sql_sync(
        ExecuteSQLInput(sql_query=sql),
        "t",
        {"row_filter": {"access_mode": "scoped", "store_ids": ["S001"]}},
    )

    assert result.success is False
    assert connector.executed == []


def test_the_same_write_is_allowed_without_policies(monkeypatch):
    """A project with no policies keeps the console it always had."""
    connector = _Connector()
    svc = _service(monkeypatch, connector, MagicMock())

    svc._execute_sql_sync(ExecuteSQLInput(sql_query="CREATE TABLE mine AS SELECT * FROM orders"), "t", {})

    assert connector.executed == ["CREATE TABLE mine AS SELECT * FROM orders"]


@pytest.mark.parametrize(
    "sql,reads",
    [
        ("CREATE TABLE plain_t (id int)", False),
        ("SET search_path TO other", False),
        ("DELETE FROM mine WHERE id = 1", False),
        ("CREATE TABLE mine AS SELECT * FROM orders", True),
        ("INSERT INTO mine SELECT * FROM orders", True),
        # `error_level=IGNORE` stops sqlglot raising on syntax it cannot place
        # and returns a `Command` instead, so "no Select inside" says nothing
        # about the statement. Both of these are full copies of `orders`.
        ("CREATE TABLE mine AS TABLE orders", True),
        ("COPY orders TO STDOUT", True),
    ],
)
def test_read_detection_uses_a_dialect_sqlglot_knows(sql, reads):
    """Connectors report `postgresql`; sqlglot only knows `postgres`.

    Handing the raw name over raises, which this helper reads as "contains a
    read" — so every write was refused on a policy-enabled project, plain
    `CREATE TABLE` included. Caught end-to-end, not by any unit test.
    """
    from datus.api.services.cli_service import _write_reads_data

    assert _write_reads_data(sql, "postgresql") is reads
