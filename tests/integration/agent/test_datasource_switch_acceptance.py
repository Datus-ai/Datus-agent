"""A01: a continuing authoring session must follow the selected datasource."""

import csv
import io
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.storage.metric.store import MetricRAG
from datus.utils.path_manager import reset_path_manager
from datus.utils.query_result_text import parse_query_result_text
from tests.helpers.product_acceptance import ModelServer, answer, isolated_connectors, product_config, tool_call
from tests.integration.agent.test_semantic_modeling_agentic import (
    _catalog_names,
    _orders_dataset,
    _revenue_metric,
    _use_deterministic_embeddings,
    restore_storage_defaults,  # noqa: F401 - shared storage lifecycle fixture
)

pytestmark = [pytest.mark.acceptance, pytest.mark.timeout(60)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("restore_storage_defaults")
async def test_same_session_switches_source_before_querying_and_authoring_metrics(tmp_path, monkeypatch):
    """Same table names cannot hide a stale DB tool or semantic write scope."""
    reset_path_manager()
    _use_deterministic_embeddings(monkeypatch)
    sources = {}
    for name, amount in (("source_a", 111), ("source_b", 777)):
        database = tmp_path / f"{name}.sqlite"
        with sqlite3.connect(database) as db:
            db.executescript(
                "CREATE TABLE orders(order_id INTEGER PRIMARY KEY,region TEXT,amount REAL);"
                f"INSERT INTO orders VALUES(1,'east',{amount});"
            )
        sources[name] = {"type": "sqlite", "uri": str(database), "default": name == "source_a"}

    try:
        with isolated_connectors(), ModelServer().running() as server:
            config = product_config(tmp_path, server.url, datasources=sources)
            config.current_datasource = "source_a"
            monkeypatch.chdir(config.project_root)
            node = SemanticModelingAgenticNode(agent_config=config, execution_mode="workflow")
            query = "SELECT SUM(amount) AS total FROM orders"
            server.enqueue(
                tool_call("execute_sql", {"sql": query, "datasource": "source_a"}),
                answer(json.dumps({"status": "skipped", "skip_reason": "no_semantic_change", "output": "A:111"})),
            )
            node.input = SemanticNodeInput(user_message="Read source A total; do not change any semantic assets.")
            first = [a async for a in node.execute_stream(ActionHistoryManager())]
            assert node.result.success, node.result.error
            assert first[-1].status == ActionStatus.SUCCESS
            first_queries = [
                a
                for a in first
                if a.role == ActionRole.TOOL and a.action_type == "execute_sql" and a.status == ActionStatus.SUCCESS
            ]
            assert len(first_queries) == 1
            first_output = first_queries[0].output["raw_output"]
            assert first_output["success"] == 1
            assert first_output["execution_context"]["datasource"] == "source_a"
            first_rows = list(csv.DictReader(io.StringIO(first_output["result"]["compressed_data"])))
            assert len(first_rows) == 1
            assert float(first_rows[0]["total"]) == 111
            session_id = node.session_id
            assert session_id
            first_request_count = len(server.requests)

            # Exercise the production slash-command switch and tool refresh while
            # supplying only the CLI shell, not an interactive terminal.
            from datus.cli.chat_commands import ChatCommands
            from datus.cli.cli_context import CliContext
            from datus.cli.datasource_commands import DatasourceCommands
            from datus.cli.repl import DatusCLI

            cli = SimpleNamespace(
                agent_config=config,
                console=Console(file=io.StringIO()),
                db_manager=node.db_func_tool._db_manager,
                cli_context=CliContext(),
                at_completer=None,
            )
            chat = SimpleNamespace(current_node=node, cli=cli)
            chat.update_chat_node_tools = lambda: ChatCommands.update_chat_node_tools(chat)
            cli.chat_commands = chat
            cli.reset_session = lambda: DatusCLI.reset_session(cli)
            DatasourceCommands(cli).cmd("source_b")
            assert config.current_datasource == "source_b"
            assert node.db_func_tool._default_datasource == "source_b"
            target = "subject/semantic_models/source_b/switched_orders.yml"
            server.enqueue(
                tool_call("execute_sql", {"sql": query}, "query_b"),
                tool_call("list_existing_osi_semantic_models", {}, "inventory_b"),
                tool_call(
                    "plan_osi_semantic_model_target",
                    {
                        "semantic_model_name": "switched_orders",
                        "business_domain": "sales",
                        "fact_tables": ["main.orders"],
                    },
                    "plan_b",
                ),
                tool_call("upsert_osi_datasets", {"path": target, "datasets_json": [_orders_dataset()]}, "dataset_b"),
                tool_call("upsert_osi_metrics", {"path": target, "metrics_json": [_revenue_metric()]}, "metric_b"),
                answer(json.dumps({"status": "generated", "output": "B total:777; saved its revenue metric."})),
            )
            node.input = SemanticNodeInput(
                user_message="Now use source B, read its total and create its revenue metric."
            )
            second = [a async for a in node.execute_stream(ActionHistoryManager())]
            assert node.session_id == session_id
            assert node.result.success, node.result.error
            assert second[-1].status == ActionStatus.SUCCESS
            assert node.result.status == "generated"
            calls = [a for a in second if a.role == ActionRole.TOOL and a.status == ActionStatus.SUCCESS]
            sql_calls = [a for a in calls if a.action_type == "execute_sql"]
            assert len(sql_calls) == 1
            sql_arguments = sql_calls[0].input["arguments"]
            if isinstance(sql_arguments, str):
                sql_arguments = json.loads(sql_arguments)
            assert sql_arguments["sql"] == query
            assert "datasource" not in sql_arguments  # The selected source must supply the default.
            sql_output = sql_calls[0].output["raw_output"]
            assert sql_output["success"] == 1
            assert sql_output["execution_context"]["datasource"] == "source_b"
            query_rows = list(csv.DictReader(io.StringIO(sql_output["result"]["compressed_data"])))
            assert len(query_rows) == 1
            assert float(query_rows[0]["total"]) == 777
            assert {"upsert_osi_datasets", "upsert_osi_metrics"} <= {a.action_type for a in calls}
            assert len(server.requests) > first_request_count
            transcript = json.dumps(server.requests[-1]["messages"])
            assert "Read source A total" in transcript
            assert "Now use source B" in transcript

            assert node.result.semantic_models == [target]
            assert (Path(config.project_root) / target).is_file()
            assert not list(config.path_manager.semantic_model_path("source_a").glob("*.yml"))
            assert _catalog_names(node.semantic_tools.list_metrics().result) == {"revenue"}
            assert MetricRAG(config, datasource_id="source_a").get_metrics_size() == 0
            result = node.semantic_tools.query_metrics(metrics=["revenue"])
            assert result.success == 1, result.error
            cached = node.semantic_tools.get_cached_query_metrics_result(
                parse_query_result_text(result.result).result_id
            )
            rows = list(csv.DictReader(io.StringIO(cached["csv"])))
            assert len(rows) == 1
            assert float(rows[0]["revenue"]) == 777
    finally:
        reset_path_manager()
