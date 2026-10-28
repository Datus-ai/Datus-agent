"""The core SQL questions with a real model and the same business oracle."""

import json
import os
import re
import sqlite3
from pathlib import Path

import pytest

from datus.utils.path_manager import reset_path_manager
from tests.helpers.product_acceptance import isolated_connectors, local_telemetry, product_config
from tests.integration.agent.test_core_acceptance import run_chat

pytestmark = [pytest.mark.nightly, pytest.mark.product_e2e]


@pytest.fixture
def live_product(tmp_path, monkeypatch):
    provider, model = os.getenv("DATUS_PRODUCT_MODEL", "deepseek/deepseek-v4-flash").split("/", 1)
    key = os.getenv("DATUS_PRODUCT_API_KEY") or os.getenv(f"{provider.upper()}_API_KEY")
    if not key:
        pytest.fail(f"Real product validation requires credentials for {provider}", pytrace=False)
    reset_path_manager()
    config = product_config(
        tmp_path, "", live_model={"type": provider, "model": model, "api_key": key, "max_tokens": 4096}
    )
    config.current_datasource = "sales"
    monkeypatch.chdir(config.project_root)
    with isolated_connectors(), local_telemetry():
        yield config
    reset_path_manager()


@pytest.mark.asyncio
async def test_question_is_answered_with_real_sql(live_product):
    node, actions = await run_chat(
        live_product, "Use the database tools to compute the total amount from orders. Report the numeric result."
    )
    successful_sql = [a for a in actions if a.action_type == "execute_sql" and a.status == "success"]
    assert successful_sql, "The answer must be backed by an actual SQL execution"
    assert any("350" in json.dumps(a.output) for a in successful_sql)
    assert re.search(r"\b350(?:\.0+)?\b", node.result.response)


@pytest.mark.asyncio
async def test_followup_keeps_the_original_question(live_product):
    node, _ = await run_chat(live_product, "Use SQL to compute the total order amount for all customers.")
    assert "350" in node.result.response
    node, actions = await run_chat(
        live_product,
        "Now compute that total for East customers only. Use the customers table to resolve the region.",
        node=node,
    )
    assert any(a.action_type == "execute_sql" and a.status == "success" for a in actions)
    assert any(a.action_type == "execute_sql" and "150" in json.dumps(a.output) for a in actions)
    assert re.search(r"\b150(?:\.0+)?\b", node.result.response)


@pytest.mark.asyncio
async def test_sql_node_returns_an_executable_query(live_product):
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
    from datus.schemas.action_history import ActionHistoryManager
    from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

    node = GenSQLAgenticNode(
        node_id="core_sql",
        description="Generate a verified regional revenue query",
        node_type="gen_sql",
        node_name="gen_sql",
        agent_config=live_product,
        execution_mode="workflow",
    )
    node.input = GenSQLNodeInput(
        user_message=(
            "Inspect the schema and generate a SELECT query returning a single column total: "
            "the sum of orders.amount for customers in the East region. Run it to verify the result, "
            "and include the complete SQL in a SQL code block in your final answer."
        )
    )
    actions = [a async for a in node.execute_stream(ActionHistoryManager())]
    assert node.result.success, node.result.error
    assert any(a.action_type == "execute_sql" and a.status == "success" for a in actions)
    assert node.result.sql, node.result.response
    with sqlite3.connect(Path(live_product.project_root).parent / "sales.sqlite") as db:
        assert db.execute(node.result.sql).fetchall() == [(150,)]
