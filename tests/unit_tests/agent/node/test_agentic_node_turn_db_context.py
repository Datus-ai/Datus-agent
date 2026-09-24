# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``AgenticNode._apply_turn_db_context``: the turn's schema reaches the connector.

The prompt names ``user_input.db_schema`` as the authoritative target, but the
connector used to keep the datasource's configured schema — so unqualified SQL
ran somewhere other than where the model was told it would. These tests use a
real ``BaseSqlConnector`` subclass, because the isolation guarantee lives in its
ContextVars, not in anything a mock would reproduce.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Optional

import pytest
from datus_db_core.base import BaseSqlConnector
from datus_db_core.config import ConnectionConfig

from datus.agent.node.agentic_node import AgenticNode
from tests.unit_tests.agent.node.test_agentic_node_template import FakeAgenticNode, FakeInput, _ok_action, _StreamModel


class _PooledConnector(BaseSqlConnector):
    """Shaped like the SQLAlchemy adapters: no persistent ``connection``."""

    def __init__(self, schema: str = "aviation", database: str = "app"):
        super().__init__(ConnectionConfig(), dialect="postgresql")
        self._default_schema = schema
        self._default_database = database

    def _not_used(self, *args, **kwargs):
        raise NotImplementedError

    execute_content_set = execute_csv = execute_ddl = execute_delete = execute_insert = _not_used
    execute_pandas = execute_queries = execute_query = execute_update = _not_used
    get_databases = get_tables = test_connection = _not_used


class _NativeConnector(_PooledConnector):
    """Shaped like Snowflake / Redshift / DuckDB: one persistent connection."""

    def __init__(self):
        super().__init__()
        self.connection = None  # declared, not yet opened
        self.switch_calls = []

    def switch_context(self, **kwargs):
        self.switch_calls.append(kwargs)


def _node(connector, db_schema: str = "", database: str = "", catalog: str = ""):
    return SimpleNamespace(
        db_func_tool=SimpleNamespace(connector=connector),
        input=SimpleNamespace(db_schema=db_schema, database=database, catalog=catalog),
    )


def test_applies_the_turn_schema_to_the_connector():
    connector = _PooledConnector()

    async def turn():
        AgenticNode._apply_turn_db_context(_node(connector, db_schema="public", database="app"))
        return connector.schema_name, connector.database_name

    assert asyncio.run(turn()) == ("public", "app")


@pytest.mark.asyncio
async def test_concurrent_turns_on_a_shared_connector_do_not_see_each_other():
    """DBManager hands every session the same connector instance."""
    connector = _PooledConnector()

    async def turn(schema: str):
        AgenticNode._apply_turn_db_context(_node(connector, db_schema=schema))
        await asyncio.sleep(0)  # let the other turn set its own schema first
        return connector.schema_name

    # gather wraps each coroutine in its own task, as each chat turn is.
    assert await asyncio.gather(turn("public"), turn("ossie_flights")) == ["public", "ossie_flights"]


@pytest.mark.asyncio
async def test_does_not_leak_past_the_turn_task():
    connector = _PooledConnector()

    async def turn():
        AgenticNode._apply_turn_db_context(_node(connector, db_schema="public"))

    await asyncio.create_task(turn())

    assert connector.schema_name == "aviation"


@pytest.mark.asyncio
async def test_tool_threads_see_the_turn_schema():
    """Sync tools run via ``asyncio.to_thread``, which copies the context."""
    connector = _PooledConnector()

    async def turn():
        AgenticNode._apply_turn_db_context(_node(connector, db_schema="public"))
        return await asyncio.to_thread(lambda: connector.schema_name)

    assert await asyncio.create_task(turn()) == "public"


@pytest.mark.asyncio
async def test_an_empty_turn_context_keeps_the_configured_defaults():
    connector = _PooledConnector()

    async def turn():
        AgenticNode._apply_turn_db_context(_node(connector))
        return connector.schema_name, connector.database_name

    assert await asyncio.create_task(turn()) == ("aviation", "app")


def test_skips_connectors_with_a_persistent_connection():
    """A live USE there would switch every session sharing the connection."""
    connector = _NativeConnector()

    AgenticNode._apply_turn_db_context(_node(connector, db_schema="public"))

    assert connector.switch_calls == []


def test_tolerates_a_node_without_a_db_tool():
    AgenticNode._apply_turn_db_context(SimpleNamespace(input=SimpleNamespace(db_schema="public")))


def test_a_failed_switch_does_not_fail_the_turn():
    class _Broken(_PooledConnector):
        def switch_context(self, **kwargs):
            raise RuntimeError("boom")

    AgenticNode._apply_turn_db_context(_node(_Broken(), db_schema="public"))


class _DbInput(FakeInput):
    db_schema: Optional[str] = None
    database: Optional[str] = None
    catalog: Optional[str] = None


class _RecordingModel(_StreamModel):
    def __init__(self, connector):
        super().__init__([[_ok_action("done")]])
        self._connector = connector
        self.schema_at_call: Optional[str] = None

    async def generate_with_tools_stream(self, *args, **kwargs):
        self.schema_at_call = self._connector.schema_name
        async for action in super().generate_with_tools_stream(*args, **kwargs):
            yield action


@pytest.mark.asyncio
async def test_execute_stream_applies_it_before_the_model_runs():
    connector = _PooledConnector()
    model = _RecordingModel(connector)
    node = FakeAgenticNode(model)
    node.db_func_tool = SimpleNamespace(connector=connector)
    node.input = _DbInput(user_message="hi", db_schema="public")

    async def turn():
        async for _ in node.execute_stream():
            pass

    await asyncio.create_task(turn())

    assert model.schema_at_call == "public"
