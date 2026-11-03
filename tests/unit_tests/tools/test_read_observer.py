import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.database import DBFuncTool
from datus.tools.read_observer import ReadObservation, notify_read, observe_reads
from datus.utils.exceptions import DatusException, ErrorCode


def execute(*, observer, denied=False, result_denied=False, database_failed=False):
    connector = Mock()
    connector.dialect = "sqlite"
    connector.get_databases.return_value = []
    connector.execute_query.return_value = SimpleNamespace(success=not database_failed, sql_return=[{"n": 1}])
    with (
        patch("datus.tools.func_tool.database.SchemaWithValueRAG"),
        patch("datus.tools.func_tool.database.SemanticDatasetRAG"),
    ):
        tool = DBFuncTool(connector)

    class Runtime:
        def before_sql_read(self, sql, **kwargs):
            return SimpleNamespace(
                allowed=not denied, sql=sql + " LIMIT 1", reason="denied", applied_policies=["scope"]
            )

        def after_read_result(self, result, **kwargs):
            return SimpleNamespace(allowed=not result_denied, result=result, reason="denied", applied_policies=["mask"])

    with (
        patch(
            "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
            return_value={"policy": lambda _: Runtime()},
        ),
        observe_reads(observer),
    ):
        # The real policy runtime still composes and executes both hooks.
        tool.agent_config = SimpleNamespace(active_plugin_names=lambda: {"policy"}, get_plugin_profile=lambda _: {})
        result = tool.execute_read_enforced("SELECT 1 AS n", connector, policy_context={"scope": ["sentinel"]})
    return result, connector


def test_observations_describe_actual_rewrite_and_result_policy():
    events = []
    result, connector = execute(observer=events.append)
    assert result.success is True
    assert [event.phase for event in events] == ["before_sql", "after_sql"]
    assert events[-1].sql == connector.execute_query.call_args.args[0] == "SELECT 1 AS n LIMIT 1"
    assert events[-1].applied_policy_ids == ("scope", "mask")
    assert events[-1].runtime_names == ("policy",)
    assert events[-1].policy_context == {"scope": ["sentinel"]}


@pytest.mark.parametrize("failure,db_calls", [("denied", 0), ("result_denied", 1), ("database_failed", 1)])
def test_no_completion_evidence_for_failed_read(failure, db_calls):
    events = []
    result, connector = execute(observer=events.append, **{failure: True})
    assert result.success is False
    assert [event.phase for event in events] == ["before_sql"]
    assert connector.execute_query.call_count == db_calls


def test_observer_can_refuse_before_database_execution():
    def refuse(event):
        raise DatusException(ErrorCode.POLICY_DENIED, message="Execution configuration changed")

    result, connector = execute(observer=refuse)
    assert result.success is False
    connector.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_requests_and_threads_keep_separate_observers():
    async def request(name):
        events = []
        with observe_reads(events.append):
            await asyncio.sleep(0)
            await asyncio.to_thread(notify_read, ReadObservation("before_sql", None, {}, name))
        return events

    first, second = await asyncio.gather(request("first"), request("second"))
    assert [event.datasource for event in first] == ["first"]
    assert [event.datasource for event in second] == ["second"]


def test_nested_observer_restores_outer_even_after_failure():
    outer, inner = [], []
    event = ReadObservation("before_sql", None, {}, "db")
    with observe_reads(outer.append):
        with pytest.raises(ValueError), observe_reads(inner.append):
            notify_read(event)
            raise ValueError("test")
        notify_read(event)
    notify_read(event)
    assert outer == [event]
    assert inner == [event]
