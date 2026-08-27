# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Deterministic Agent semantic projection + PostgreSQL storage regression."""

from __future__ import annotations

import textwrap
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql

from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.storage import embedding_models
from datus.storage.embedding_models import EmbeddingModel
from datus.storage.metric.store import MetricRAG
from datus.storage.registry import clear_storage_registry, configure_storage_defaults, get_storage_defaults
from datus.storage.semantic_dataset.store import SemanticDatasetRAG
from datus.tools.func_tool.generation_tools import GenerationTools


class _DeterministicEmbeddings:
    """Small stable embedding function: no model download and no LLM call."""

    def ndims(self) -> int:
        return 4

    def generate_embeddings(self, texts, *args, **kwargs):
        del args, kwargs
        vectors = []
        for text in texts:
            encoded = str(text).encode("utf-8")
            buckets = [0.0, 0.0, 0.0, 0.0]
            for index, value in enumerate(encoded):
                buckets[index % 4] += float(value)
            scale = sum(buckets) or 1.0
            vectors.append([value / scale for value in buckets])
        return vectors


def _embedding_model() -> EmbeddingModel:
    model = EmbeddingModel("nightly-deterministic", 4, registry_name="nightly-deterministic")
    model._model = _DeterministicEmbeddings()
    return model


def _osi_document(description: str, metric_description: str) -> str:
    return textwrap.dedent(
        f"""\
        version: "0.2.0.dev0"
        semantic_model:
          - name: nightly_orders
            description: PostgreSQL reconcile regression model
            datasets:
              - name: orders
                description: {description}
                source: analytics.orders
                primary_key: [order_id]
                fields:
                  - name: order_id
                    expression:
                      dialects:
                        - dialect: ANSI_SQL
                          expression: order_id
                  - name: region
                    expression:
                      dialects:
                        - dialect: ANSI_SQL
                          expression: region
                    dimension: {{}}
                  - name: amount
                    expression:
                      dialects:
                        - dialect: ANSI_SQL
                          expression: amount
            metrics:
              - name: order_revenue
                description: {metric_description}
                expression:
                  dialects:
                    - dialect: ANSI_SQL
                      expression: SUM(orders.amount)
                custom_extensions:
                  - vendor_name: DATUS
                    data: '{{"v":"1.4","dataset":"orders","subject_path":["sales","orders"]}}'
        """
    )


def _connect(params: dict):
    return psycopg.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        dbname=params["dbname"],
    )


@pytest.fixture
def postgresql_agent_config(required_postgresql_storage, tmp_path, monkeypatch) -> Iterator[AgentConfig]:
    project_name = f"nightly_pg_{uuid.uuid4().hex[:10]}"
    datasource = "warehouse"
    deterministic_model = _embedding_model()
    storage_defaults = get_storage_defaults()
    monkeypatch.setitem(embedding_models.EMBEDDING_MODELS, "semantic_model", deterministic_model)
    monkeypatch.setitem(embedding_models.EMBEDDING_MODELS, "metric", deterministic_model)
    configure_storage_defaults()
    clear_storage_registry()

    rdb_params = dict(required_postgresql_storage.rdb_config.params)
    vector_params = dict(required_postgresql_storage.vector_config.params)
    config = AgentConfig(
        nodes={"test": NodeConfig(model="mock", input=None)},
        home=str(tmp_path / "home"),
        project_name=project_name,
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
                datasource: {
                    "type": "postgresql",
                    "host": "unused",
                    "database": "analytics",
                    "schema": "public",
                    "default": True,
                }
            }
        },
        storage={
            "rdb": {"type": "postgresql", **rdb_params},
            "vector": {"type": "postgresql", **vector_params},
        },
    )
    config.current_datasource = datasource

    try:
        yield config
    finally:
        clear_storage_registry()
        try:
            # The testing providers own cleanup semantics.  Drop only this
            # random project schema, never shared/public data.
            with _connect(rdb_params) as connection:
                connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(project_name)))
                connection.commit()
            with _connect(vector_params) as connection:
                connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(project_name)))
                connection.commit()
        finally:
            configure_storage_defaults(**storage_defaults)
            clear_storage_registry()


@pytest.mark.nightly
@pytest.mark.timeout(300)
def test_postgresql_semantic_kb_reconcile_is_idempotent(postgresql_agent_config, tmp_path):
    """Exercise insert, update, and no-op reconcile through the Agent seam."""
    config = postgresql_agent_config
    artifact = tmp_path / "nightly_orders.yml"
    artifact.write_text(_osi_document("Initial orders dataset", "Initial revenue metric"), encoding="utf-8")
    tools = GenerationTools(config, authoring_format="osi")

    first = tools.sync_osi_to_db(str(artifact))
    artifact.write_text(_osi_document("Reconciled orders dataset", "Reconciled revenue metric"), encoding="utf-8")
    second = tools.sync_osi_to_db(str(artifact))
    third = tools.sync_osi_to_db(str(artifact))

    for result in (first, second, third):
        assert result["success"] is True, result
        assert result["semantic_dataset_rows"] == 4
        assert result["metric_names"] == ["order_revenue"]

    semantic_rag = SemanticDatasetRAG(config)
    metric_rag = MetricRAG(config)
    datasets = semantic_rag.list_datasets(table_name="orders")
    metrics = metric_rag.search_all_metrics()
    assert [row["name"] for row in datasets] == ["orders"]
    assert datasets[0]["description"] == "Reconciled orders dataset"
    assert [row["name"] for row in metrics] == ["order_revenue"]
    assert metrics[0]["description"] == "Reconciled revenue metric"
    assert semantic_rag.list_objects("reconciled orders", kinds=["dataset"])[0]["name"] == "orders"
    assert metric_rag.search_metrics("order_revenue")[0]["name"] == "order_revenue"

    project = config.project_name
    datasource = config.current_datasource
    rdb_params = dict(config._backend_config.rdb.params)
    vector_params = dict(config._backend_config.vector.params)

    with _connect(vector_params) as connection:
        semantic_rows = connection.execute(
            sql.SQL(
                "SELECT id, kind, description, is_primary_key, is_time, is_dimension, "
                "vector IS NOT NULL AS has_vector FROM {}.semantic_dataset ORDER BY id"
            ).format(sql.Identifier(project))
        ).fetchall()
        metric_rows = connection.execute(
            sql.SQL(
                "SELECT id, name, description, subject_node_id, vector IS NOT NULL AS has_vector "
                "FROM {}.metrics ORDER BY id"
            ).format(sql.Identifier(project))
        ).fetchall()

    assert len(semantic_rows) == 4
    assert len({row[0] for row in semantic_rows}) == 4
    dataset_row = next(row for row in semantic_rows if row[1] == "dataset")
    assert dataset_row[2:] == ("Reconciled orders dataset", None, None, None, True)
    field_flags = {row[0].rsplit(".", 1)[-1]: row[3:6] for row in semantic_rows if row[1] == "field"}
    assert field_flags == {
        "amount": (False, False, False),
        "order_id": (True, False, False),
        "region": (False, False, True),
    }
    assert len(metric_rows) == 1
    assert metric_rows[0][0:3] == ("metric:order_revenue", "order_revenue", "Reconciled revenue metric")
    assert metric_rows[0][4] is True

    with _connect(rdb_params) as connection:
        subject_rows = connection.execute(
            sql.SQL(
                "SELECT node_id, parent_id, name FROM {}.subject_nodes WHERE datasource_id = %s ORDER BY node_id"
            ).format(sql.Identifier(project)),
            (datasource,),
        ).fetchall()

    assert [row[2] for row in subject_rows] == ["sales", "orders"]
    assert subject_rows[0][1] == -1
    assert subject_rows[1][1] == subject_rows[0][0]
    assert metric_rows[0][3] == subject_rows[1][0]
