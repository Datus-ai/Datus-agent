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


def test_a_multi_statement_selection_keeps_working(monkeypatch):
    """ "Execute Statement" over a selection holding two statements.

    The enforced path rejects those outright, which would have broken a normal
    editor gesture.
    """
    connector = _Connector()
    svc = _service(monkeypatch, connector, MagicMock())

    svc._execute_sql_sync(ExecuteSQLInput(sql_query="SELECT 1; SELECT 2"), "task-3", {})

    assert connector.executed == ["SELECT 1; SELECT 2"]


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
