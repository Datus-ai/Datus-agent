"""Core product journeys through production models, SDK, tools and storage."""

import ast
import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.utils.path_manager import reset_path_manager
from tests.helpers.product_acceptance import ModelServer, answer, isolated_connectors, product_config, tool_call

pytestmark = [pytest.mark.acceptance, pytest.mark.timeout(30)]


@pytest.fixture
def product(tmp_path, monkeypatch):
    reset_path_manager()
    with isolated_connectors(), ModelServer().running() as server:
        config = product_config(tmp_path, server.url)
        config.current_datasource = "sales"
        monkeypatch.chdir(config.project_root)
        yield config, server
    reset_path_manager()


async def run_chat(config, message, *, node=None):
    node = node or ChatAgenticNode(
        node_id="core",
        description="Core product acceptance",
        node_type="chat",
        agent_config=config,
        execution_mode="workflow",
    )
    node.input = ChatNodeInput(user_message=message, max_turns=12)
    actions = [action async for action in node.execute_stream(ActionHistoryManager())]
    assert node.result.success, node.result.error
    assert actions[-1].status == ActionStatus.SUCCESS
    return node, actions


@pytest.mark.asyncio
async def test_chat_stream_reaches_model_and_completes(product):
    config, server = product
    server.enqueue(answer("I can explore your data and answer questions."))
    node, _ = await run_chat(config, "What can you help me with?")
    assert "explore" in node.result.response
    assert server.requests[0]["stream"] is True
    offered = {tool["function"]["name"] for tool in server.requests[0]["tools"]}
    assert {"execute_sql", "list_tables", "describe_table"} <= offered


@pytest.mark.asyncio
async def test_sql_result_reaches_followup_request(product):
    config, server = product
    server.enqueue(
        tool_call("execute_sql", {"sql": "SELECT SUM(amount) AS total FROM orders"}), answer("The total is 350.")
    )
    node, _ = await run_chat(config, "What is the total order amount?")
    results = [m for m in server.requests[-1]["messages"] if m.get("role") == "tool"]
    assert results and "350" in json.dumps(results)
    assert "350" in node.result.response
    with sqlite3.connect(Path(config.project_root).parent / "sales.sqlite") as db:
        assert db.execute("SELECT SUM(amount) FROM orders").fetchone() == (350,)


@pytest.mark.asyncio
async def test_multiturn_retains_question_and_tool_results(product):
    config, server = product
    server.enqueue(tool_call("execute_sql", {"sql": "SELECT SUM(amount) AS total FROM orders"}), answer("Total: 350."))
    node, _ = await run_chat(config, "What is the total order amount?")
    server.enqueue(
        tool_call(
            "execute_sql",
            {
                "sql": "SELECT SUM(o.amount) AS total FROM orders o JOIN customers c ON c.id=o.customer_id WHERE c.region='East'"
            },
            "call_east",
        ),
        answer("East: 150."),
    )
    await run_chat(config, "Now show only the East region.", node=node)
    transcript = json.dumps(server.requests[-1]["messages"])
    assert "What is the total order amount?" in transcript and "East region" in transcript
    assert "350" in transcript and "150" in transcript


@pytest.mark.asyncio
async def test_tool_error_can_be_corrected(product):
    config, server = product
    server.enqueue(
        tool_call("execute_sql", {"sql": "SELECT SUM(missing_amount) FROM orders"}),
        tool_call("execute_sql", {"sql": "SELECT SUM(amount) AS total FROM orders"}, "call_retry"),
        answer("Total: 350."),
    )
    await run_chat(config, "What is the total order amount?")
    assert "missing_amount" in json.dumps(server.requests[1]["messages"])
    assert "350" in json.dumps(server.requests[2]["messages"])
    assert len(server.requests) == 3


@pytest.mark.asyncio
async def test_metadata_exploration_uses_real_schema(product):
    config, server = product
    server.enqueue(
        tool_call("list_tables", {}),
        tool_call("describe_table", {"table_name": "orders"}, "describe"),
        answer("Orders contain id, customer_id and amount."),
    )
    await run_chat(config, "Explore the tables and describe orders.")
    assert "orders" in json.dumps(server.requests[1]["messages"])
    assert "customer_id" in json.dumps(server.requests[2]["messages"])


@pytest.mark.asyncio
async def test_memory_written_in_one_session_is_loaded_in_another(product):
    config, server = product
    rule = "Revenue is the sum of orders.amount; amounts are in USD."
    server.enqueue(tool_call("add_memory", {"content": rule}), answer("Saved the business rule."))
    await run_chat(config, "Remember our revenue definition.")
    memory = Path(config.project_root) / ".datus/memory/chat/MEMORY.md"
    assert rule in memory.read_text()
    server.enqueue(answer("Revenue is the sum of order amounts in USD."))
    reader = ChatAgenticNode(
        node_id="new_session",
        description="Read persisted memory",
        node_type="chat",
        agent_config=config,
        execution_mode="workflow",
    )
    await run_chat(config, "What is our revenue definition?", node=reader)
    assert rule in json.dumps(server.requests[-1]["messages"])


@pytest.mark.asyncio
async def test_init_metadata_and_knowledge_are_reusable(product):
    from datus.cli.init_context import InitContext
    from datus.tools.permission.profiles import get_profile

    config, server = product
    config.permissions_config = get_profile("dangerous")
    config.active_profile_name = "dangerous"
    server.enqueue(
        tool_call("describe_table", {"table_name": "orders"}),
        tool_call("write_file", {"path": "knowledge/sales.md", "content": "Orders store amounts in USD."}, "write"),
        answer("Initialized sales knowledge."),
    )
    node = ChatAgenticNode(
        node_id="init",
        description="Initialize project",
        node_type="chat",
        agent_config=config,
        execution_mode="workflow",
    )
    context = InitContext(Path(config.project_root), {"datasource": "sales"}, "Initialize sales knowledge")
    try:
        with context.activate(node):
            await run_chat(config, "Initialize this project using the init skill.", node=node)
            context.finish_execution(node.result)
        assert (Path(config.project_root) / "knowledge/sales.md").read_text() == "Orders store amounts in USD."
        assert "execute_sql" not in {t["function"]["name"] for t in server.requests[0]["tools"]}
    finally:
        context.close()
    server.enqueue(tool_call("read_file", {"path": "knowledge/sales.md"}), answer("Order amounts are in USD."))
    await run_chat(config, "Read the sales knowledge.")
    assert "Orders store amounts in USD." in json.dumps(server.requests[-1]["messages"])


@pytest.mark.asyncio
async def test_read_only_refuses_a_write_without_changing_data(product):
    from datus.tools.permission.profiles import get_profile

    config, server = product
    config.permissions_config = get_profile("dangerous")
    config.active_profile_name = "dangerous"
    config.harden_sql_read_only()
    server.enqueue(tool_call("execute_sql", {"sql": "DELETE FROM orders"}), answer("This datasource is read only."))
    await run_chat(config, "Delete the orders.")
    with sqlite3.connect(Path(config.project_root).parent / "sales.sqlite") as db:
        assert db.execute("SELECT COUNT(*),SUM(amount) FROM orders").fetchone() == (3, 350)
    results = [m for m in server.requests[-1]["messages"] if m.get("role") == "tool"]
    assert results and ast.literal_eval(results[0]["content"])["success"] == 0


@pytest.mark.asyncio
async def test_large_tool_result_is_bounded_and_archived(product):
    config, server = product
    content = "A long source document line.\n" * 20000
    (Path(config.project_root) / "large.txt").write_text(content)
    server.enqueue(tool_call("read_file", {"path": "large.txt"}), answer("The source was read."))
    await run_chat(config, "Read large.txt.")
    followup = json.dumps(server.requests[-1]["messages"])
    assert len(followup) < len(content)
    assert "result_id" in followup


@pytest.mark.asyncio
async def test_plan_confirmation_resumes_the_real_agent_loop(product):
    from datus.tools.permission.profiles import get_profile

    config, server = product
    config.permissions_config = get_profile("dangerous")
    config.active_profile_name = "dangerous"
    server.enqueue(tool_call("confirm_plan", {}), answer("The approved plan is ready."))
    node = ChatAgenticNode(node_id="confirm", description="Confirm a plan", node_type="chat", agent_config=config)
    node.input = ChatNodeInput(user_message="Confirm the plan before proceeding.", plan_mode=True)
    node.activate_plan_mode()
    Path(node.plan_file_path).write_text("# Plan\n\nInspect orders and summarize revenue.\n")
    broker = node._get_or_create_broker()

    async def user_confirms():
        while not broker.has_pending:
            await asyncio.sleep(0.01)
        action_id = next(iter(broker._pending))
        await broker.submit(action_id, [["confirm"]])

    ui = asyncio.create_task(user_confirms())
    try:
        async with asyncio.timeout(15):
            actions = [a async for a in node.execute_stream_with_interactions(ActionHistoryManager())]
        await asyncio.wait_for(ui, timeout=5)
        assert node.result.success, node.result.error
        assert any(a.role == "interaction" and a.status == "processing" for a in actions)
        assert "confirm" in json.dumps(server.requests[-1]["messages"]).lower()
    finally:
        if not ui.done():
            ui.cancel()
        await asyncio.gather(ui, return_exceptions=True)


@pytest.mark.asyncio
async def test_table_node_creates_a_queryable_data_product(product):
    from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
    from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
    from datus.tools.permission.profiles import get_profile

    config, server = product
    config.permissions_config = get_profile("dangerous")
    config.active_profile_name = "dangerous"
    server.enqueue(
        tool_call("execute_sql", {"sql": "CREATE TABLE daily_revenue AS SELECT SUM(amount) AS total FROM orders"}),
        tool_call("execute_sql", {"sql": "SELECT total FROM daily_revenue"}, "verify_table"),
        answer("Created and verified daily_revenue with total 350."),
        tool_call("describe_table", {"table_name": "daily_revenue"}, "validator_schema"),
        answer(
            json.dumps(
                {"checks": [{"name": "total_column", "passed": True, "severity": "blocking"}], "blocking_issues": []}
            )
        ),
    )
    node = GenTableAgenticNode(agent_config=config, execution_mode="workflow")
    node.input = SemanticNodeInput(user_message="Create a revenue summary table and verify its total.")
    actions = [a async for a in node.execute_stream(ActionHistoryManager())]
    assert node.result.success, node.result.error
    assert actions[-1].status == ActionStatus.SUCCESS
    with sqlite3.connect(Path(config.project_root).parent / "sales.sqlite") as db:
        assert db.execute("SELECT total FROM daily_revenue").fetchall() == [(350,)]
    assert "350" in json.dumps(server.requests[2]["messages"])
    assert "table-validation" in json.dumps(server.requests[3]["messages"])
    assert "total" in json.dumps(server.requests[4]["messages"])
    report = node.result.validation_report
    assert report and not report["warnings"], report
    assert any(c["name"] == "total_column" and c["passed"] for c in report["checks"]), report
