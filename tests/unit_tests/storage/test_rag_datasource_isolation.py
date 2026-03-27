# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datasource_id isolation and write field injection across all RAG classes.

Covers:
- _inject_write_fields adds datasource_id/creator_id/updator_id to write data
- Reads from one RAG instance do not see data written by another with a different datasource_id
"""

from typing import Any, Dict, List

import pyarrow as pa

from datus.storage.ext_knowledge.store import ExtKnowledgeRAG
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql.store import ReferenceSqlRAG
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


def _make_knowledge(suffix: str = "a") -> Dict[str, Any]:
    """Return a minimal ExtKnowledgeRAG-compatible knowledge entry."""
    return {
        "subject_path": ["Finance", "Terms"],
        "name": f"APR_{suffix}",
        "search_text": f"Annual Percentage Rate {suffix}",
        "explanation": f"Yearly cost of a loan variant {suffix}",
    }


def _make_reference_sql(suffix: str = "a") -> Dict[str, Any]:
    """Return a minimal ReferenceSqlRAG-compatible SQL entry."""
    return {
        "id": f"sql:daily_sales_{suffix}",
        "subject_path": ["Analytics", "Reports"],
        "name": f"daily_sales_{suffix}",
        "sql": "SELECT date, SUM(amount) FROM orders GROUP BY date",
        "comment": "Daily sales report",
        "summary": f"Daily sales aggregation {suffix}",
        "search_text": f"daily sales report aggregation {suffix}",
        "filepath": "",
        "tags": "sales,daily",
    }


# ---------------------------------------------------------------------------
# TestRAGWriteInjection
# ---------------------------------------------------------------------------


class TestRAGWriteInjection:
    """Tests that _inject_write_fields correctly injects datasource_id/creator_id/updator_id."""

    def test_inject_write_fields_adds_datasource_id(self, real_agent_config):
        """store_batch via MetricRAG injects datasource_id into data rows."""
        rag = MetricRAG(real_agent_config, datasource_id="inject_test_ds")
        data = [_make_metric("inject")]
        rag._inject_write_fields(data)
        assert data[0]["datasource_id"] == "inject_test_ds"

    def test_inject_write_fields_default_creator_and_updator(self, real_agent_config):
        """Default creator_id and updator_id are 'datus_agent'."""
        rag = MetricRAG(real_agent_config, datasource_id="inject_ds")
        data = [_make_metric("defaults")]
        rag._inject_write_fields(data)
        assert data[0]["creator_id"] == "datus_agent"
        assert data[0]["updator_id"] == "datus_agent"

    def test_inject_write_fields_custom_creator_updator(self, real_agent_config):
        """Custom creator_id/updator_id are stored."""
        rag = MetricRAG(
            real_agent_config,
            datasource_id="inject_ds",
            creator_id="alice",
            updator_id="bob",
        )
        data = [_make_metric("custom")]
        rag._inject_write_fields(data)
        assert data[0]["creator_id"] == "alice"
        assert data[0]["updator_id"] == "bob"

    def test_inject_write_fields_force_sets_existing(self, real_agent_config):
        """Pre-existing field values are always overwritten (force-set semantics)."""
        rag = MetricRAG(
            real_agent_config,
            datasource_id="inject_ds",
            creator_id="system",
            updator_id="system",
        )
        data = [dict(_make_metric("preexist"), datasource_id="already_set", creator_id="original")]
        rag._inject_write_fields(data)
        assert data[0]["datasource_id"] == "inject_ds"
        assert data[0]["creator_id"] == "system"
        assert data[0]["updator_id"] == "system"

    def test_inject_write_fields_multiple_rows(self, real_agent_config):
        """All rows in a batch get datasource_id injected."""
        rag = MetricRAG(real_agent_config, datasource_id="batch_ds")
        data = [_make_metric("r1"), _make_metric("r2"), _make_metric("r3")]
        rag._inject_write_fields(data)
        for row in data:
            assert row["datasource_id"] == "batch_ds"

    # --- Injection via SemanticModelRAG ---

    def test_semantic_model_rag_inject_fields(self, real_agent_config):
        """SemanticModelRAG._inject_write_fields adds all three fields."""
        rag = SemanticModelRAG(real_agent_config, datasource_id="sm_ds", creator_id="sm_user", updator_id="sm_bot")
        data = [_make_table_object("inj")]
        rag._inject_write_fields(data)
        assert data[0]["datasource_id"] == "sm_ds"
        assert data[0]["creator_id"] == "sm_user"
        assert data[0]["updator_id"] == "sm_bot"

    # --- Injection via ReferenceSqlRAG ---

    def test_reference_sql_rag_inject_fields(self, real_agent_config):
        """ReferenceSqlRAG._inject_write_fields adds all three fields."""
        rag = ReferenceSqlRAG(real_agent_config, datasource_id="sql_ds", creator_id="etl", updator_id="pipeline")
        data = [_make_reference_sql("inj")]
        rag._inject_write_fields(data)
        assert data[0]["datasource_id"] == "sql_ds"
        assert data[0]["creator_id"] == "etl"
        assert data[0]["updator_id"] == "pipeline"

    # --- Injection via ExtKnowledgeRAG ---

    def test_ext_knowledge_rag_inject_fields(self, real_agent_config):
        """ExtKnowledgeRAG._inject_write_fields adds all three fields."""
        rag = ExtKnowledgeRAG(real_agent_config, datasource_id="ek_ds", creator_id="admin", updator_id="admin")
        data = [_make_knowledge("inj")]
        rag._inject_write_fields(data)
        assert data[0]["datasource_id"] == "ek_ds"
        assert data[0]["creator_id"] == "admin"
        assert data[0]["updator_id"] == "admin"


# ---------------------------------------------------------------------------
# TestRAGDatasourceIsolation
# ---------------------------------------------------------------------------


class TestRAGDatasourceIsolation:
    """Core isolation tests: data written with datasource_id A is not visible to datasource_id B."""

    # -----------------------------------------------------------------------
    # SemanticModelRAG
    # -----------------------------------------------------------------------

    def test_semantic_model_search_all_isolates_by_datasource(self, real_agent_config):
        """search_all() only returns rows for the RAG's own datasource_id."""
        rag_a = SemanticModelRAG(real_agent_config, datasource_id="ds_a")
        rag_b = SemanticModelRAG(real_agent_config, datasource_id="ds_b")

        rag_a.store_batch([_make_table_object("a1"), _make_table_object("a2")])
        rag_b.store_batch([_make_table_object("b1")])

        results_a = rag_a.search_all()
        results_b = rag_b.search_all()

        assert len(results_a) == 2
        assert len(results_b) == 1
        # Make sure ds_a rows don't bleed into ds_b
        assert all(r["datasource_id"] == "ds_a" for r in results_a)
        assert all(r["datasource_id"] == "ds_b" for r in results_b)

    def test_semantic_model_get_size_isolates_by_datasource(self, real_agent_config):
        """get_size() counts only rows for the RAG's own datasource_id."""
        rag_a = SemanticModelRAG(real_agent_config, datasource_id="size_ds_a")
        rag_b = SemanticModelRAG(real_agent_config, datasource_id="size_ds_b")

        rag_a.store_batch([_make_table_object("sz_a1"), _make_table_object("sz_a2")])
        rag_b.store_batch([_make_table_object("sz_b1")])

        assert rag_a.get_size() == 2
        assert rag_b.get_size() == 1

    def test_semantic_model_upsert_batch_isolates_by_datasource(self, real_agent_config):
        """upsert_batch() writes are visible only to the correct datasource_id."""
        rag_a = SemanticModelRAG(real_agent_config, datasource_id="ups_ds_a")
        rag_b = SemanticModelRAG(real_agent_config, datasource_id="ups_ds_b")

        obj_a = _make_table_object("ups_a")
        obj_b = _make_table_object("ups_b")

        rag_a.upsert_batch([obj_a])
        rag_b.upsert_batch([obj_b])

        assert rag_a.get_size() == 1
        assert rag_b.get_size() == 1

        results_a = rag_a.search_all()
        results_b = rag_b.search_all()
        assert results_a[0]["table_name"] == "orders_ups_a"
        assert results_b[0]["table_name"] == "orders_ups_b"

    def test_semantic_model_empty_datasource_sees_nothing(self, real_agent_config):
        """A RAG instance with a new datasource_id sees zero rows even after others write."""
        rag_writer = SemanticModelRAG(real_agent_config, datasource_id="writer_ds")
        rag_empty = SemanticModelRAG(real_agent_config, datasource_id="empty_ds")

        rag_writer.store_batch([_make_table_object("writer")])

        assert rag_empty.get_size() == 0
        assert rag_empty.search_all() == []

    # -----------------------------------------------------------------------
    # MetricRAG
    # -----------------------------------------------------------------------

    def test_metric_search_all_isolates_by_datasource(self, real_agent_config):
        """search_all_metrics() only returns rows for the RAG's own datasource_id."""
        rag_a = MetricRAG(real_agent_config, datasource_id="metric_ds_a")
        rag_b = MetricRAG(real_agent_config, datasource_id="metric_ds_b")

        rag_a.store_batch([_make_metric("m_a1"), _make_metric("m_a2")])
        rag_b.store_batch([_make_metric("m_b1")])

        results_a = rag_a.search_all_metrics()
        results_b = rag_b.search_all_metrics()

        assert len(results_a) == 2
        assert len(results_b) == 1
        assert all(r["datasource_id"] == "metric_ds_a" for r in results_a)
        assert all(r["datasource_id"] == "metric_ds_b" for r in results_b)

    def test_metric_get_metrics_size_isolates_by_datasource(self, real_agent_config):
        """get_metrics_size() counts only rows for the RAG's own datasource_id."""
        rag_a = MetricRAG(real_agent_config, datasource_id="msize_ds_a")
        rag_b = MetricRAG(real_agent_config, datasource_id="msize_ds_b")

        rag_a.store_batch([_make_metric("sz_m_a1"), _make_metric("sz_m_a2"), _make_metric("sz_m_a3")])
        rag_b.store_batch([_make_metric("sz_m_b1")])

        assert rag_a.get_metrics_size() == 3
        assert rag_b.get_metrics_size() == 1

    def test_metric_upsert_batch_isolates_by_datasource(self, real_agent_config):
        """upsert_batch() writes are visible only to the correct datasource_id."""
        rag_a = MetricRAG(real_agent_config, datasource_id="mups_ds_a")
        rag_b = MetricRAG(real_agent_config, datasource_id="mups_ds_b")

        rag_a.upsert_batch([_make_metric("ups_m_a")])
        rag_b.upsert_batch([_make_metric("ups_m_b")])

        assert rag_a.get_metrics_size() == 1
        assert rag_b.get_metrics_size() == 1

    def test_metric_empty_datasource_sees_nothing(self, real_agent_config):
        """A RAG with new datasource_id sees zero metrics written by another."""
        rag_writer = MetricRAG(real_agent_config, datasource_id="metric_writer_ds")
        rag_empty = MetricRAG(real_agent_config, datasource_id="metric_empty_ds")

        rag_writer.store_batch([_make_metric("mw")])

        assert rag_empty.get_metrics_size() == 0
        assert rag_empty.search_all_metrics() == []

    # -----------------------------------------------------------------------
    # ExtKnowledgeRAG
    # -----------------------------------------------------------------------

    def _write_knowledge(self, rag: ExtKnowledgeRAG, entries: List[Dict[str, Any]]):
        """Helper: inject write fields and write via underlying store."""
        rag._inject_write_fields(entries)
        rag.store.batch_store_knowledge(entries)

    def test_ext_knowledge_query_isolates_by_datasource(self, real_agent_config):
        """query_knowledge() only returns rows for the RAG's own datasource_id."""
        rag_a = ExtKnowledgeRAG(real_agent_config, datasource_id="ek_ds_a")
        rag_b = ExtKnowledgeRAG(real_agent_config, datasource_id="ek_ds_b")

        entries_a = [_make_knowledge("ek_a1"), _make_knowledge("ek_a2")]
        entries_b = [_make_knowledge("ek_b1")]

        self._write_knowledge(rag_a, entries_a)
        self._write_knowledge(rag_b, entries_b)

        results_a = rag_a.query_knowledge(top_n=10)
        results_b = rag_b.query_knowledge(top_n=10)

        assert len(results_a) == 2
        assert len(results_b) == 1
        assert all(r["datasource_id"] == "ek_ds_a" for r in results_a)
        assert all(r["datasource_id"] == "ek_ds_b" for r in results_b)

    def test_ext_knowledge_get_size_isolates_by_datasource(self, real_agent_config):
        """get_knowledge_size() counts only rows for the RAG's own datasource_id."""
        rag_a = ExtKnowledgeRAG(real_agent_config, datasource_id="eksize_ds_a")
        rag_b = ExtKnowledgeRAG(real_agent_config, datasource_id="eksize_ds_b")

        entries_a = [_make_knowledge("eksize_a1"), _make_knowledge("eksize_a2")]
        entries_b = [_make_knowledge("eksize_b1")]

        self._write_knowledge(rag_a, entries_a)
        self._write_knowledge(rag_b, entries_b)

        assert rag_a.get_knowledge_size() == 2
        assert rag_b.get_knowledge_size() == 1

    def test_ext_knowledge_empty_datasource_sees_nothing(self, real_agent_config):
        """A RAG with new datasource_id sees zero knowledge written by another."""
        rag_writer = ExtKnowledgeRAG(real_agent_config, datasource_id="ek_writer_ds")
        rag_empty = ExtKnowledgeRAG(real_agent_config, datasource_id="ek_empty_ds")

        self._write_knowledge(rag_writer, [_make_knowledge("ek_w")])

        assert rag_empty.get_knowledge_size() == 0
        assert rag_empty.query_knowledge(top_n=10) == []

    # -----------------------------------------------------------------------
    # ReferenceSqlRAG
    # -----------------------------------------------------------------------

    def test_reference_sql_search_all_isolates_by_datasource(self, real_agent_config):
        """search_all_reference_sql() only returns rows for the RAG's own datasource_id."""
        rag_a = ReferenceSqlRAG(real_agent_config, datasource_id="rsql_ds_a")
        rag_b = ReferenceSqlRAG(real_agent_config, datasource_id="rsql_ds_b")

        rag_a.store_batch([_make_reference_sql("rs_a1"), _make_reference_sql("rs_a2")])
        rag_b.store_batch([_make_reference_sql("rs_b1")])

        results_a = rag_a.search_all_reference_sql()
        results_b = rag_b.search_all_reference_sql()

        assert len(results_a) == 2
        assert len(results_b) == 1
        assert all(r["datasource_id"] == "rsql_ds_a" for r in results_a)
        assert all(r["datasource_id"] == "rsql_ds_b" for r in results_b)

    def test_reference_sql_get_size_isolates_by_datasource(self, real_agent_config):
        """get_reference_sql_size() counts only rows for the RAG's own datasource_id."""
        rag_a = ReferenceSqlRAG(real_agent_config, datasource_id="rsize_ds_a")
        rag_b = ReferenceSqlRAG(real_agent_config, datasource_id="rsize_ds_b")

        rag_a.store_batch([_make_reference_sql("rsize_a1"), _make_reference_sql("rsize_a2")])
        rag_b.store_batch([_make_reference_sql("rsize_b1")])

        assert rag_a.get_reference_sql_size() == 2
        assert rag_b.get_reference_sql_size() == 1

    def test_reference_sql_upsert_batch_isolates_by_datasource(self, real_agent_config):
        """upsert_batch() writes are visible only to the correct datasource_id."""
        rag_a = ReferenceSqlRAG(real_agent_config, datasource_id="rups_ds_a")
        rag_b = ReferenceSqlRAG(real_agent_config, datasource_id="rups_ds_b")

        rag_a.upsert_batch([_make_reference_sql("ups_rs_a")])
        rag_b.upsert_batch([_make_reference_sql("ups_rs_b")])

        assert rag_a.get_reference_sql_size() == 1
        assert rag_b.get_reference_sql_size() == 1

        results_a = rag_a.search_all_reference_sql()
        results_b = rag_b.search_all_reference_sql()
        assert results_a[0]["name"] == "daily_sales_ups_rs_a"
        assert results_b[0]["name"] == "daily_sales_ups_rs_b"

    def test_reference_sql_empty_datasource_sees_nothing(self, real_agent_config):
        """A RAG with new datasource_id sees zero SQL entries written by another."""
        rag_writer = ReferenceSqlRAG(real_agent_config, datasource_id="rsql_writer_ds")
        rag_empty = ReferenceSqlRAG(real_agent_config, datasource_id="rsql_empty_ds")

        rag_writer.store_batch([_make_reference_sql("rsql_w")])

        assert rag_empty.get_reference_sql_size() == 0
        assert rag_empty.search_all_reference_sql() == []

    # -----------------------------------------------------------------------
    # Cross-RAG: multiple datasource_ids coexist in same storage backend
    # -----------------------------------------------------------------------

    def test_multiple_datasources_coexist_metric_rag(self, real_agent_config):
        """Multiple datasource_ids coexist: counts add up correctly per datasource."""
        rags = [MetricRAG(real_agent_config, datasource_id=f"coexist_ds_{i}") for i in range(3)]

        for idx, rag in enumerate(rags):
            # Write idx+1 entries per RAG so totals are 1, 2, 3
            rag.store_batch([_make_metric(f"co_{idx}_{j}") for j in range(idx + 1)])

        for idx, rag in enumerate(rags):
            assert rag.get_metrics_size() == idx + 1

    def test_multiple_datasources_coexist_semantic_model_rag(self, real_agent_config):
        """Multiple datasource_ids coexist: SemanticModelRAG counts match per datasource."""
        rags = [SemanticModelRAG(real_agent_config, datasource_id=f"sm_coexist_{i}") for i in range(3)]

        for idx, rag in enumerate(rags):
            rag.store_batch([_make_table_object(f"sm_co_{idx}_{j}") for j in range(idx + 1)])

        for idx, rag in enumerate(rags):
            assert rag.get_size() == idx + 1
