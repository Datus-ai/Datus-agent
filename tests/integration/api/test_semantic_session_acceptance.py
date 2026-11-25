"""A09: real semantic authoring tools must settle the owning HTTP session."""

import ast
import asyncio
import csv
import io
import json
import sqlite3

import pytest

from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.utils.query_result_text import parse_query_result_text
from tests.helpers.product_acceptance import answer, tool_call
from tests.integration.agent.test_semantic_modeling_agentic import (
    _revenue_metric,
    _use_deterministic_embeddings,
)
from tests.integration.agent.test_semantic_modeling_agentic import (
    restore_storage_defaults as restore_storage_defaults,  # noqa: F401
)
from tests.integration.api.test_api_chat import find_event, parse_sse_body
from tests.integration.api.test_chat_acceptance import model_server as model_server  # noqa: F401
from tests.integration.api.test_chat_acceptance import resource_project as resource_project  # noqa: F401

pytestmark = pytest.mark.acceptance

# Well below the Core Acceptance group's 600 s thread timeout: a missing terminal
# event fails this test alone instead of killing the group's pytest process.
TERMINAL_DEADLINE_SECONDS = 240


@pytest.mark.asyncio
async def test_http_semantic_generation_finishes_and_releases_session(
    resource_project, model_server, monkeypatch, restore_storage_defaults
):
    """Script only the model; author, reconcile and query a real persisted metric."""
    _use_deterministic_embeddings(monkeypatch)
    config = resource_project.service.agent_config
    config.agentic_nodes["semantic_modeling"] = {"model": "product", "max_turns": 12}
    target = "subject/semantic_models/sales/a09_revenue.yml"
    dataset = {
        "name": "orders",
        "description": "A09 fixed order data",
        "source": "main.orders",
        "primary_key": ["id"],
        "fields": [
            {"name": name, "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": name}]}}
            for name in ("id", "customer_id", "amount")
        ],
    }
    offset = len(model_server.requests)
    model_server.enqueue(
        tool_call("list_existing_osi_semantic_models", {}, "a09_inventory"),
        tool_call(
            "plan_osi_semantic_model_target",
            {"semantic_model_name": "a09_revenue", "business_domain": "sales", "fact_tables": ["main.orders"]},
            "a09_target",
        ),
        tool_call("upsert_osi_datasets", {"path": target, "datasets_json": json.dumps([dataset])}, "a09_dataset"),
        tool_call(
            "upsert_osi_metrics", {"path": target, "metrics_json": json.dumps([_revenue_metric()])}, "a09_metric"
        ),
        answer(json.dumps({"status": "generated", "output": "A09 revenue metric saved and queryable."})),
    )
    sid = "a09_semantic_terminal"
    client, manager = resource_project.client, resource_project.service.task_manager
    response = await asyncio.wait_for(
        client.post(
            "/api/v1/chat/stream",
            json={
                "message": "Generate the revenue metric for orders, reconcile it and finish.",
                "session_id": sid,
                "subagent_id": "semantic_modeling",
                "interactive": False,
                "permission_mode": "dangerous",
            },
        ),
        TERMINAL_DEADLINE_SECONDS,
    )
    assert response.status_code == 200
    events = parse_sse_body(response.text)
    assert find_event(events, "error") is None, response.text
    assert sum(event["event"] == "end" for event in events) == 1, response.text
    assert find_event(events, "end")["data"]["session_id"] == sid
    assert find_event(events, "session")["data"]["session_id"] == sid
    task = manager.get_task(sid)
    await asyncio.wait_for(task.asyncio_task, 10)
    assert task.status == "completed", task.error
    assert task.node is None
    assert not manager.has_active_tasks()
    assert config.path_manager.project_root.joinpath(target).is_file()

    # These are the actual tool results fed back through the SDK to the model,
    # rather than a success phrase in the scripted final answer.
    replies = [
        ast.literal_eval(message["content"])
        for message in model_server.requests[offset:][-1]["messages"]
        if message["role"] == "tool"
    ]
    assert len(replies) == 4, replies
    assert all(reply["success"] == 1 for reply in replies), replies
    semantic_tools = SemanticTools(config)
    catalog = semantic_tools.list_metrics(limit=20, offset=0)
    assert catalog.success == 1, catalog.error
    assert "revenue" in catalog.result
    queried = semantic_tools.query_metrics(metrics=["revenue"])
    assert queried.success == 1, queried.error
    cached = semantic_tools.get_cached_query_metrics_result(parse_query_result_text(queried.result).result_id)
    rows = list(csv.DictReader(io.StringIO(cached["csv"])))
    with sqlite3.connect(resource_project.root / "default" / "sales.sqlite") as database:
        expected = database.execute("SELECT SUM(amount) FROM orders").fetchone()[0]
    assert expected == 350
    assert len(rows) == 1 and float(rows[0]["revenue"]) == expected

    # A new request on the same session proves authoring released its run lock.
    model_server.enqueue(answer("A09 session can continue after metric generation."))
    followup = await asyncio.wait_for(
        client.post(
            "/api/v1/chat/stream",
            json={"message": "Continue after generating revenue.", "session_id": sid, "interactive": False},
        ),
        TERMINAL_DEADLINE_SECONDS,
    )
    followup_events = parse_sse_body(followup.text)
    assert find_event(followup_events, "error") is None, followup.text
    assert find_event(followup_events, "end") is not None, followup.text
    next_task = manager.get_task(sid)
    await asyncio.wait_for(next_task.asyncio_task, 10)
    assert next_task is not task
    assert next_task.status == "completed"
    assert next_task.node is None
    assert not manager.has_active_tasks()
