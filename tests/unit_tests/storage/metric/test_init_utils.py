# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.storage.metric.init_utils module."""

import tempfile

import pytest

from datus.storage.embedding_models import get_db_embedding_model, get_metric_embedding_model
from datus.storage.metric.init_utils import existing_semantic_metrics, gen_metric_id, gen_semantic_model_id
from datus.storage.metric.store import MetricStorage
from datus.storage.semantic_model.store import SemanticModelStorage

# ============================================================
# Tests for gen_semantic_model_id
# ============================================================


class TestGenSemanticModelId:
    """Tests for gen_semantic_model_id function."""

    def test_all_parts_present(self):
        """ID with all four parts joined by underscores."""
        result = gen_semantic_model_id("catalog1", "db1", "schema1", "table1")
        assert result == "catalog1_db1_schema1_table1"

    def test_empty_catalog(self):
        """ID with empty catalog still includes underscore prefix."""
        result = gen_semantic_model_id("", "db1", "schema1", "table1")
        assert result == "_db1_schema1_table1"

    def test_empty_database(self):
        """ID with empty database."""
        result = gen_semantic_model_id("catalog1", "", "schema1", "table1")
        assert result == "catalog1__schema1_table1"

    def test_empty_schema(self):
        """ID with empty schema."""
        result = gen_semantic_model_id("catalog1", "db1", "", "table1")
        assert result == "catalog1_db1__table1"

    def test_all_empty_except_table(self):
        """ID with only table name populated."""
        result = gen_semantic_model_id("", "", "", "users")
        assert result == "___users"

    def test_all_empty(self):
        """ID with all parts empty."""
        result = gen_semantic_model_id("", "", "", "")
        assert result == "___"

    @pytest.mark.parametrize(
        "catalog,database,schema,table,expected",
        [
            ("c", "d", "s", "t", "c_d_s_t"),
            ("prod", "analytics", "public", "orders", "prod_analytics_public_orders"),
            ("", "main", "dbo", "users", "_main_dbo_users"),
        ],
    )
    def test_parametrized_id_generation(self, catalog, database, schema, table, expected):
        """Parametrized test for various combinations of catalog, database, schema, table."""
        result = gen_semantic_model_id(catalog, database, schema, table)
        assert result == expected

    def test_deterministic(self):
        """Same inputs always produce the same ID."""
        id1 = gen_semantic_model_id("cat", "db", "sch", "tbl")
        id2 = gen_semantic_model_id("cat", "db", "sch", "tbl")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        """Different inputs produce different IDs."""
        id1 = gen_semantic_model_id("cat", "db1", "sch", "tbl")
        id2 = gen_semantic_model_id("cat", "db2", "sch", "tbl")
        assert id1 != id2


# ============================================================
# Tests for gen_metric_id
# ============================================================


class TestGenMetricId:
    """Tests for gen_metric_id function."""

    def test_basic_id_generation(self):
        """Basic metric ID with subject path, model, and metric name."""
        result = gen_metric_id(["Finance", "Revenue"], "sales_model", "total_revenue")
        assert result == "Finance/Revenue/sales_model_total_revenue"

    def test_single_path_element(self):
        """Metric ID with single element subject path."""
        result = gen_metric_id(["Analytics"], "model1", "dau")
        assert result == "Analytics/model1_dau"

    def test_empty_subject_path(self):
        """Metric ID with empty subject path."""
        result = gen_metric_id([], "model1", "dau")
        assert result == "/model1_dau"

    def test_deep_subject_path(self):
        """Metric ID with deeply nested subject path."""
        result = gen_metric_id(["L1", "L2", "L3", "L4"], "model", "metric")
        assert result == "L1/L2/L3/L4/model_metric"

    def test_none_subject_path(self):
        """Metric ID with None subject path."""
        result = gen_metric_id(None, "model", "metric")
        assert result == "/model_metric"

    @pytest.mark.parametrize(
        "path,model,metric,expected",
        [
            (["A"], "m", "x", "A/m_x"),
            (["A", "B"], "m", "x", "A/B/m_x"),
            ([], "m", "x", "/m_x"),
            (["X", "Y", "Z"], "model", "metric", "X/Y/Z/model_metric"),
        ],
    )
    def test_parametrized_metric_id(self, path, model, metric, expected):
        """Parametrized test for various metric ID generation scenarios."""
        result = gen_metric_id(path, model, metric)
        assert result == expected

    def test_deterministic(self):
        """Same inputs produce the same metric ID."""
        id1 = gen_metric_id(["A", "B"], "model", "metric")
        id2 = gen_metric_id(["A", "B"], "model", "metric")
        assert id1 == id2

    def test_different_path_different_id(self):
        """Different subject paths produce different IDs."""
        id1 = gen_metric_id(["A", "B"], "model", "metric")
        id2 = gen_metric_id(["A", "C"], "model", "metric")
        assert id1 != id2

    def test_different_model_different_id(self):
        """Different semantic model names produce different IDs."""
        id1 = gen_metric_id(["A"], "model1", "metric")
        id2 = gen_metric_id(["A"], "model2", "metric")
        assert id1 != id2


# ============================================================
# Tests for existing_semantic_metrics
# ============================================================


class TestExistingSemanticMetrics:
    """Tests for existing_semantic_metrics function."""

    def _make_semantic_storage(self, scope_dir):
        """Create a SemanticModelStorage in a temporary directory."""
        return SemanticModelStorage(embedding_model=get_db_embedding_model())

    def _make_metric_storage(self, scope_dir):
        """Create a MetricStorage in a temporary directory."""
        return MetricStorage(embedding_model=get_metric_embedding_model())

    def test_empty_stores_return_empty_sets(self):
        """Empty stores return two empty sets."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            semantic_storage = self._make_semantic_storage(tmp_dir)
            metric_storage = self._make_metric_storage(tmp_dir)

            # We need to wrap storages in RAG-like objects that have the expected methods
            # existing_semantic_metrics expects SemanticModelRAG and MetricRAG
            # SemanticModelRAG.search_all calls storage.search_all which needs kind="table" filter
            # MetricRAG.search_all_metrics calls storage.search_all_metrics

            # Create minimal wrapper objects
            class FakeSemanticRAG:
                def __init__(self, storage):
                    self.storage = storage

                def search_all(self, database_name="", select_fields=None):
                    from datus.storage.conditions import And, build_where, eq

                    conditions = [eq("kind", "table")]
                    where = build_where(And(conditions))
                    return self.storage._search_all(where=where, select_fields=select_fields).to_pylist()

            class FakeMetricRAG:
                def __init__(self, storage):
                    self.storage = storage

                def search_all_metrics(self, select_fields=None):
                    return self.storage.search_all_metrics(select_fields=select_fields)

            fake_semantic = FakeSemanticRAG(semantic_storage)
            fake_metric = FakeMetricRAG(metric_storage)

            all_models, all_metrics = existing_semantic_metrics(fake_semantic, fake_metric)
            assert all_models == set()
            assert all_metrics == set()

    def test_with_stored_semantic_models(self):
        """Returns IDs of stored semantic models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            semantic_storage = self._make_semantic_storage(tmp_dir)

            # Store some semantic model objects (kind=table)
            semantic_storage.store_batch(
                [
                    {
                        "id": "table:users",
                        "kind": "table",
                        "name": "users",
                        "fq_name": "db.public.users",
                        "semantic_model_name": "users_model",
                        "catalog_name": "",
                        "database_name": "db",
                        "schema_name": "public",
                        "table_name": "users",
                        "description": "User accounts table",
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
                        "updated_at": None,
                    },
                ]
            )

            class FakeSemanticRAG:
                def __init__(self, storage):
                    self.storage = storage

                def search_all(self, database_name="", select_fields=None):
                    from datus.storage.conditions import And, build_where, eq

                    conditions = [eq("kind", "table")]
                    where = build_where(And(conditions))
                    return self.storage._search_all(where=where, select_fields=select_fields).to_pylist()

            fake_semantic = FakeSemanticRAG(semantic_storage)

            # Search only for IDs
            results = fake_semantic.search_all(select_fields=["id"])
            assert len(results) == 1
            assert results[0]["id"] == "table:users"

    def test_with_stored_metrics(self):
        """Returns IDs of stored metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            metric_storage = self._make_metric_storage(tmp_dir)

            metric_storage.batch_store_metrics(
                [
                    {
                        "subject_path": ["Finance", "Revenue"],
                        "id": "metric:total_revenue",
                        "name": "total_revenue",
                        "semantic_model_name": "revenue_model",
                        "description": "Total revenue metric",
                        "metric_type": "simple",
                        "measure_expr": "SUM(amount)",
                        "base_measures": ["revenue"],
                        "dimensions": ["country"],
                        "entities": ["order"],
                        "catalog_name": "",
                        "database_name": "db",
                        "schema_name": "public",
                        "sql": "",
                        "yaml_path": "",
                        "updated_at": None,
                    }
                ]
            )

            class FakeMetricRAG:
                def __init__(self, storage):
                    self.storage = storage

                def search_all_metrics(self, select_fields=None):
                    return self.storage.search_all_metrics(select_fields=select_fields)

            fake_metric = FakeMetricRAG(metric_storage)
            results = fake_metric.search_all_metrics(select_fields=["id"])
            assert len(results) == 1
            assert results[0]["id"] == "metric:total_revenue"
