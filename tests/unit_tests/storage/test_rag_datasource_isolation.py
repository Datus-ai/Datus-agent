# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for RAG classes in PHYSICAL isolation mode.

In Storage 3.0, datasource_id isolation is handled at the backend level:
- PHYSICAL mode (CLI default): each namespace gets its own directory via
  init_backends(namespace=...).  All RAG instances within one process share
  the same storage — isolation is per-process, not per-RAG.
- LOGICAL mode (SaaS): backend auto-injects datasource_id column for
  within-process multi-tenant filtering.

LOGICAL isolation is tested at the LanceDB backend level in
tests/unit_tests/storage/vector/test_lance_backend.py
(TestLanceVectorTableLogicalIsolation).

This file tests that RAG classes work correctly in PHYSICAL mode
(single shared storage, no cross-datasource filtering).
"""

from typing import Any, Dict

import pyarrow as pa

from datus.storage.metric.store import MetricRAG
from datus.storage.semantic_model.store import SemanticModelRAG

# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------


def _make_table_object(suffix: str = "a") -> Dict[str, Any]:
    """Return a minimal SemanticModelRAG-compatible table object."""
    return {
        "id": f"table:orders_{suffix}",
        "kind": "table",
        "name": f"orders_{suffix}",
        "fq_name": f"analytics.public.orders_{suffix}",
        "semantic_model_name": f"orders_{suffix}",
        "catalog_name": "default",
        "database_name": "analytics",
        "schema_name": "public",
        "table_name": f"orders_{suffix}",
        "description": f"Order table {suffix}",
        "is_dimension": False,
        "is_measure": False,
        "is_entity_key": False,
        "is_deprecated": False,
        "expr": "",
        "column_type": "",
        "agg": "",
        "create_metric": False,
        "agg_time_dimension": "",
        "is_partition": False,
        "time_granularity": "",
        "entity": "",
        "yaml_path": "",
        "updated_at": pa.scalar(0, type=pa.timestamp("ms")),
    }


def _make_metric(suffix: str = "a") -> Dict[str, Any]:
    """Return a minimal MetricRAG-compatible metric object."""
    return {
        "id": f"metric:total_revenue_{suffix}",
        "subject_path": ["Finance", "Revenue"],
        "name": f"total_revenue_{suffix}",
        "semantic_model_name": "orders",
        "description": f"Total revenue metric {suffix}",
    }


# ---------------------------------------------------------------------------
# PHYSICAL mode tests — shared storage within one process
# ---------------------------------------------------------------------------


class TestRAGPhysicalModeSharedStorage:
    """In PHYSICAL mode, all RAG instances share the same storage.

    This verifies that store_batch/search_all work correctly when
    multiple RAGs use the same underlying storage.
    """

    def test_semantic_model_store_and_search(self, real_agent_config):
        """SemanticModelRAG can store and retrieve data."""
        rag = SemanticModelRAG(real_agent_config)
        rag.store_batch([_make_table_object("x1"), _make_table_object("x2")])
        results = rag.search_all()
        assert len(results) == 2

    def test_semantic_model_truncate(self, real_agent_config):
        """SemanticModelRAG truncate clears data."""
        rag = SemanticModelRAG(real_agent_config)
        rag.store_batch([_make_table_object("trunc")])
        assert rag.get_size() >= 1
        rag.truncate()
        assert rag.get_size() == 0

    def test_metric_store_and_search(self, real_agent_config):
        """MetricRAG can store and retrieve metrics."""
        rag = MetricRAG(real_agent_config)
        rag.store_batch([_make_metric("m1"), _make_metric("m2")])
        results = rag.search_all_metrics()
        assert len(results) >= 2

    def test_metric_get_size(self, real_agent_config):
        """MetricRAG.get_metrics_size returns correct count."""
        rag = MetricRAG(real_agent_config)
        rag.store_batch([_make_metric("sz1"), _make_metric("sz2")])
        assert rag.get_metrics_size() >= 2

    def test_metric_upsert_batch(self, real_agent_config):
        """MetricRAG.upsert_batch updates existing and inserts new."""
        rag = MetricRAG(real_agent_config)
        m = _make_metric("ups")
        rag.store_batch([m])
        initial_size = rag.get_metrics_size()

        # Upsert same id — should not increase count
        rag.upsert_batch([m])
        assert rag.get_metrics_size() == initial_size
