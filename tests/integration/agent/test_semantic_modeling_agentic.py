# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Integration coverage for the unified Dosi semantic-modeling node."""

from __future__ import annotations

import json
import os
import sqlite3

import pytest
from lancedb.embeddings import register
from lancedb.embeddings.base import TextEmbeddingFunction

from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SourceQueryEvidence
from datus.storage import embedding_models
from datus.storage.embedding_models import EmbeddingModel
from datus.storage.metric.store import MetricRAG
from datus.storage.registry import clear_storage_registry, configure_storage_defaults
from datus.storage.semantic_dataset.store import SemanticDatasetRAG
from datus.utils.path_manager import DatusPathManager

pytestmark = [pytest.mark.nightly, pytest.mark.product_e2e]


def _use_dosi(agent_config, project_root):
    agent_config.semantic_layer_configs = {"dosi": {}}
    agent_config._project_root = project_root.resolve()
    agent_config.path_manager = DatusPathManager(
        agent_config.home,
        project_name=agent_config.project_name,
        project_root=str(project_root),
    )
    return agent_config


@register("nightly_dosi_test")
class _DeterministicEmbeddings(TextEmbeddingFunction):
    def ndims(self) -> int:
        return 4

    def generate_embeddings(self, texts, *args, **kwargs):
        del args, kwargs
        vectors = []
        for text in texts:
            buckets = [0.0, 0.0, 0.0, 0.0]
            for index, value in enumerate(str(text).encode("utf-8")):
                buckets[index % 4] += float(value)
            scale = sum(buckets) or 1.0
            vectors.append([value / scale for value in buckets])
        return vectors


def _use_deterministic_embeddings(monkeypatch) -> None:
    model = EmbeddingModel("nightly-dosi", 4, registry_name="nightly-dosi")
    model._model = _DeterministicEmbeddings.create()
    monkeypatch.setattr(
        embedding_models,
        "EMBEDDING_MODELS",
        {
            name: model
            for name in (
                "database",
                "document",
                "metric",
                "reference_sql",
                "semantic_model",
                "subject",
            )
        },
    )
    configure_storage_defaults()
    clear_storage_registry()


def _deterministic_dosi_config(tmp_path, monkeypatch) -> AgentConfig:
    _use_deterministic_embeddings(monkeypatch)
    database = tmp_path / "orders.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                region TEXT NOT NULL,
                amount REAL NOT NULL
            );
            INSERT INTO orders VALUES
                (1, 'east', 100.0),
                (2, 'west', 150.0),
                (3, 'east', 200.0);
            """
        )

    config = AgentConfig(
        nodes={"semantic": NodeConfig(model="mock", input=None)},
        home=str(tmp_path / "home"),
        project_name="nightly_dosi",
        project_root=str(tmp_path / "workspace"),
        target="mock",
        models={
            "mock": {
                "type": "openai",
                "api_key": "unused",
                "model": "unused",
                "base_url": "http://127.0.0.1:1",
            }
        },
        services={
            "datasources": {
                "orders": {
                    "type": "sqlite",
                    "uri": str(database),
                    "default": True,
                }
            },
            "semantic_layer": {"dosi": {"type": "dosi", "default": True}},
        },
    )
    config.current_datasource = "orders"
    return config


def _orders_dataset() -> dict:
    return {
        "name": "orders",
        "description": "Nightly Dosi orders",
        "source": "main.orders",
        "primary_key": ["order_id"],
        "fields": [
            {
                "name": "order_id",
                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "order_id"}]},
            },
            {
                "name": "region",
                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "region"}]},
                "dimension": {},
            },
            {
                "name": "amount",
                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "amount"}]},
            },
        ],
    }


def _revenue_metric() -> dict:
    return {
        "name": "revenue",
        "description": "Total order revenue",
        "expression": {
            "dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.amount)"}],
        },
        "custom_extensions": [
            {
                "vendor_name": "DATUS",
                "data": json.dumps(
                    {"v": "1.4", "dataset": "orders", "subject_path": ["sales", "orders"]},
                    separators=(",", ":"),
                ),
            }
        ],
    }


def test_dosi_authoring_validates_reconciles_and_queries_without_llm(tmp_path, monkeypatch):
    """Cover the documented Dosi authoring contract through Agent-owned tools."""
    config = _deterministic_dosi_config(tmp_path, monkeypatch)
    node = SemanticModelingAgenticNode(agent_config=config, execution_mode="workflow")

    inventory = node.list_existing_osi_semantic_models()
    assert inventory.success == 1
    planned = node.plan_osi_semantic_model_target(
        semantic_model_name="orders_model",
        business_domain="sales",
        fact_tables=["main.orders"],
    )
    assert planned.success == 1
    target = node.osi_target_state.selected_path
    assert target

    dataset_result = node.filesystem_func_tool.upsert_osi_datasets(target, json.dumps([_orders_dataset()]))
    metric_result = node.filesystem_func_tool.upsert_osi_metrics(target, json.dumps([_revenue_metric()]))
    assert dataset_result.success == 1, dataset_result.error
    assert metric_result.success == 1, metric_result.error

    public_path = node._finalize_selected_osi_artifact()
    assert public_path == "subject/semantic_models/orders/orders_model.yml"
    assert config.path_manager.project_root.joinpath(public_path).is_file()

    catalog = node.semantic_tools.list_metrics(limit=20, offset=0)
    assert catalog.success == 1, catalog.error
    assert {item["name"] for item in catalog.result["items"]} == {"revenue"}

    dry_run = node.semantic_tools.query_metrics(metrics=["revenue"], dry_run=True)
    assert dry_run.success == 1, dry_run.error
    assert "main.orders" in dry_run.result["metadata"]["sql"]

    live = node.semantic_tools.query_metrics(metrics=["revenue"])
    assert live.success == 1, live.error
    cached = node.semantic_tools.get_cached_query_metrics_result(live.result["result_id"])
    assert cached is not None
    assert cached["columns"] == ["revenue"]
    assert "450" in cached["csv"]

    semantic_rag = SemanticDatasetRAG(config)
    metric_rag = MetricRAG(config)
    assert semantic_rag.get_size() == 1
    assert {item["name"] for item in semantic_rag.list_fields("orders_model", "orders")} == {
        "amount",
        "order_id",
        "region",
    }
    assert metric_rag.get_metrics_size() == 1
    assert [item["name"] for item in metric_rag.search_all_metrics()] == ["revenue"]


def test_semantic_modeling_scopes_share_one_real_project_workspace(nightly_agent_config, tmp_path):
    config = _use_dosi(nightly_agent_config, tmp_path)

    full_node = SemanticModelingAgenticNode(
        agent_config=config,
        execution_mode="workflow",
        authoring_scope="full",
    )
    datasets_node = SemanticModelingAgenticNode(
        agent_config=config,
        execution_mode="workflow",
        authoring_scope="datasets",
    )

    assert full_node.get_node_name() == "semantic_modeling"
    assert full_node.filesystem_func_tool.config.root_path == str(config.path_manager.subject_dir.parent)
    assert config.path_manager.semantic_model_path(config.current_datasource).is_relative_to(
        config.path_manager.subject_dir / "semantic_models"
    )

    full_tools = {tool.name for tool in full_node.tools}
    datasets_tools = {tool.name for tool in datasets_node.tools}
    assert {"upsert_osi_datasets", "delete_osi_datasets", "upsert_osi_metrics", "delete_osi_metrics"} <= full_tools
    assert {"upsert_osi_datasets", "delete_osi_datasets"} <= datasets_tools
    assert {"upsert_osi_metrics", "delete_osi_metrics"}.isdisjoint(datasets_tools)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_semantic_modeling_authors_one_queryable_dosi_model_with_real_llm(
    nightly_agent_config,
    tmp_path,
    monkeypatch,
):
    """Smoke the complete LLM authoring loop after the deterministic contract."""
    assert os.environ.get("DEEPSEEK_API_KEY"), "P0 Dosi smoke requires DEEPSEEK_API_KEY"
    _use_deterministic_embeddings(monkeypatch)
    nightly_agent_config.home = str(tmp_path / "home")
    config = _use_dosi(nightly_agent_config, tmp_path / "workspace")
    config.semantic_layer_configs = {"dosi": {"type": "dosi", "default": True}}

    node = SemanticModelingAgenticNode(agent_config=config, execution_mode="workflow")
    node.input = SemanticNodeInput(
        user_message=(
            "Create one Dosi semantic model named nightly_school_counts for the main.schools table. "
            "Include the reusable dataset fields needed for a metric named school_count that counts schools. "
            "Validate it, reconcile it to the Knowledge Base, and return status generated."
        ),
        semantic_model_name="nightly_school_counts",
        business_domain="schools",
        fact_tables=["main.schools"],
        max_turns=30,
        source_queries=[
            SourceQueryEvidence(
                source_sql_name="school_count",
                question="How many schools are there?",
                sql="SELECT COUNT(*) AS school_count FROM schools",
            )
        ],
    )

    actions = [action async for action in node.execute_stream(ActionHistoryManager())]
    successful_tools = {
        action.action_type
        for action in actions
        if action.role == ActionRole.TOOL and action.status == ActionStatus.SUCCESS
    }
    assert {"upsert_osi_datasets", "upsert_osi_metrics"} <= successful_tools
    assert actions[-1].status == ActionStatus.SUCCESS, actions[-1].output
    assert actions[-1].output["success"] is True
    assert actions[-1].output["status"] == "generated"
    assert len(actions[-1].output["semantic_models"]) == 1

    model_path = config.path_manager.project_root / actions[-1].output["semantic_models"][0]
    assert model_path.is_file()
    metric_catalog = node.semantic_tools.list_metrics(limit=50, offset=0)
    assert metric_catalog.success == 1, metric_catalog.error
    assert "school_count" in {item["name"] for item in metric_catalog.result["items"]}
