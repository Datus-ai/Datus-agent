# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/semantic_model/store.py -- SemanticModelStorage and SemanticModelRAG."""

import pytest
from pandas import Timestamp

from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.semantic_model.store import SemanticModelRAG, SemanticModelStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table_object(
    table_name: str,
    description: str = "A table",
    catalog_name: str = "default",
    database_name: str = "analytics",
    schema_name: str = "public",
    semantic_model_name: str = "",
    yaml_path: str = "",
) -> dict:
    """Build a table-kind semantic object for storage."""
    return {
        "id": f"table:{table_name}",
        "kind": "table",
        "name": table_name,
        "fq_name": f"{database_name}.{schema_name}.{table_name}",
        "semantic_model_name": semantic_model_name or table_name,
        "catalog_name": catalog_name,
        "database_name": database_name,
        "schema_name": schema_name,
        "table_name": table_name,
        "description": description,
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
        "yaml_path": yaml_path,
        "updated_at": Timestamp.now().floor("ms"),
    }


def _make_column_object(
    table_name: str,
    column_name: str,
    description: str = "A column",
    is_dimension: bool = False,
    is_measure: bool = False,
    is_entity_key: bool = False,
    column_type: str = "",
    agg: str = "",
    create_metric: bool = False,
    agg_time_dimension: str = "",
    is_partition: bool = False,
    time_granularity: str = "",
    entity: str = "",
    expr: str = "",
    catalog_name: str = "default",
    database_name: str = "analytics",
    schema_name: str = "public",
) -> dict:
    """Build a column-kind semantic object for storage."""
    return {
        "id": f"column:{table_name}.{column_name}",
        "kind": "column",
        "name": column_name,
        "fq_name": f"{database_name}.{schema_name}.{table_name}.{column_name}",
        "semantic_model_name": table_name,
        "catalog_name": catalog_name,
        "database_name": database_name,
        "schema_name": schema_name,
        "table_name": table_name,
        "description": description,
        "is_dimension": is_dimension,
        "is_measure": is_measure,
        "is_entity_key": is_entity_key,
        "is_deprecated": False,
        "expr": expr or column_name,
        "column_type": column_type,
        "agg": agg,
        "create_metric": create_metric,
        "agg_time_dimension": agg_time_dimension,
        "is_partition": is_partition,
        "time_granularity": time_granularity,
        "entity": entity,
        "yaml_path": "",
        "updated_at": Timestamp.now().floor("ms"),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sem_storage(tmp_path) -> SemanticModelStorage:
    """Create a SemanticModelStorage with real vector backend."""
    return SemanticModelStorage(embedding_model=get_db_embedding_model())


@pytest.fixture
def sem_rag(real_agent_config) -> SemanticModelRAG:
    """Create a SemanticModelRAG with real AgentConfig."""
    return SemanticModelRAG(agent_config=real_agent_config)


# ============================================================
# SemanticModelStorage.__init__
# ============================================================


class TestSemanticModelStorageInit:
    """Tests for SemanticModelStorage initialization."""

    def test_table_name(self, sem_storage):
        """Table name should be 'semantic_model'."""
        assert sem_storage.table_name == "semantic_model"

    def test_vector_source_name(self, sem_storage):
        """Vector source should be 'description'."""
        assert sem_storage.vector_source_name == "description"

    def test_vector_column_name(self, sem_storage):
        """Vector column should be 'vector'."""
        assert sem_storage.vector_column_name == "vector"

    def test_schema_has_expected_fields(self, sem_storage):
        """Schema should contain all required fields."""
        expected = {
            "id",
            "kind",
            "name",
            "fq_name",
            "semantic_model_name",
            "catalog_name",
            "database_name",
            "schema_name",
            "table_name",
            "description",
            "vector",
            "is_dimension",
            "is_measure",
            "is_entity_key",
            "is_deprecated",
            "expr",
            "column_type",
            "agg",
            "create_metric",
            "agg_time_dimension",
            "is_partition",
            "time_granularity",
            "entity",
            "yaml_path",
            "updated_at",
        }
        schema_names = set(sem_storage._schema.names)
        for field in expected:
            assert field in schema_names, f"Field '{field}' missing from schema"


# ============================================================
# SemanticModelStorage.store_batch / search
# ============================================================


class TestSemanticModelStorageBatchOps:
    """Tests for store_batch, upsert_batch, and search operations."""

    def test_store_batch_single_table(self, sem_storage):
        """Storing a single table object should be retrievable."""
        table_obj = _make_table_object("orders", description="Customer orders table")
        sem_storage.store_batch([table_obj])
        results = sem_storage.search_all()
        assert len(results) >= 1
        names = [r["name"] for r in results]
        assert "orders" in names

    def test_store_batch_empty_noop(self, sem_storage):
        """Storing an empty list should be a no-op."""
        sem_storage.store_batch([])
        results = sem_storage.search_all()
        assert results == []

    def test_store_batch_multiple_objects(self, sem_storage):
        """Storing multiple objects should all be retrievable."""
        objs = [
            _make_table_object("orders", description="Orders table"),
            _make_table_object("customers", description="Customers table"),
            _make_column_object("orders", "order_id", description="Primary key for orders", is_entity_key=True),
        ]
        sem_storage.store_batch(objs)
        results = sem_storage.search_all()
        assert len(results) == 3

    def test_upsert_batch_insert(self, sem_storage):
        """Upserting new objects should insert them."""
        table_obj = _make_table_object("products", description="Products catalog")
        sem_storage.upsert_batch([table_obj], on_column="id")
        results = sem_storage.search_all()
        assert len(results) == 1
        assert results[0]["name"] == "products"

    def test_upsert_batch_update(self, sem_storage):
        """Upserting existing objects should update them."""
        table_obj = _make_table_object("products", description="Original description")
        sem_storage.store_batch([table_obj])

        updated_obj = _make_table_object("products", description="Updated description")
        sem_storage.upsert_batch([updated_obj], on_column="id")

        results = sem_storage.search_all()
        assert len(results) == 1
        assert results[0]["description"] == "Updated description"


# ============================================================
# SemanticModelStorage.create_indices
# ============================================================


class TestSemanticModelStorageIndices:
    """Tests for create_indices."""

    def test_create_indices_after_data(self, sem_storage):
        """Creating indices after storing data should not raise."""
        objs = [
            _make_table_object("orders", description="Orders"),
            _make_column_object("orders", "amount", description="Order amount", is_measure=True),
        ]
        sem_storage.store_batch(objs)
        sem_storage.create_indices()
        # Verify search still works after index creation
        results = sem_storage.search_all()
        assert len(results) == 2


# ============================================================
# SemanticModelStorage.search_objects
# ============================================================


class TestSemanticModelStorageSearchObjects:
    """Tests for search_objects with kind and table_name filters."""

    @pytest.fixture(autouse=True)
    def _populate(self, sem_storage):
        """Populate storage with test data."""
        objs = [
            _make_table_object("orders", description="Customer orders table"),
            _make_table_object("products", description="Product catalog table"),
            _make_column_object("orders", "order_id", description="Order identifier", is_entity_key=True),
            _make_column_object("orders", "amount", description="Order total amount", is_measure=True),
            _make_column_object("products", "product_name", description="Product name", is_dimension=True),
        ]
        sem_storage.store_batch(objs)
        self.storage = sem_storage

    def test_search_objects_no_filter(self):
        """Search without filters returns results."""
        results = self.storage.search_objects("orders", top_n=10)
        assert len(results) > 0

    def test_search_objects_filter_by_kind_table(self):
        """Filtering by kind='table' returns only table objects."""
        results = self.storage.search_objects("table", kinds=["table"], top_n=10)
        for r in results:
            assert r["kind"] == "table"

    def test_search_objects_filter_by_kind_column(self):
        """Filtering by kind='column' returns only column objects."""
        results = self.storage.search_objects("column", kinds=["column"], top_n=10)
        for r in results:
            assert r["kind"] == "column"

    def test_search_objects_filter_by_table_name(self):
        """Filtering by table_name returns only objects for that table."""
        results = self.storage.search_objects("order", table_name="orders", top_n=10)
        for r in results:
            assert r["table_name"] == "orders"

    def test_search_objects_combined_filters(self):
        """Combining kind and table_name filters narrows results."""
        results = self.storage.search_objects("amount", kinds=["column"], table_name="orders", top_n=10)
        for r in results:
            assert r["kind"] == "column"
            assert r["table_name"] == "orders"


# ============================================================
# SemanticModelRAG.get_semantic_model
# ============================================================


class TestSemanticModelRAGGetSemanticModel:
    """Tests for the get_semantic_model method with multi-level fallback logic."""

    def test_get_semantic_model_returns_none_without_table_name(self, sem_rag):
        """Calling without table_name should return None."""
        result = sem_rag.get_semantic_model(table_name="")
        assert result is None

    def test_get_semantic_model_returns_none_for_nonexistent(self, sem_rag):
        """Querying a non-existent table should return None."""
        result = sem_rag.get_semantic_model(table_name="nonexistent_table")
        assert result is None

    def test_get_semantic_model_basic(self, sem_rag):
        """Store a table and retrieve it via get_semantic_model."""
        objs = [_make_table_object("orders", description="Orders table")]
        sem_rag.storage.store_batch(objs)

        result = sem_rag.get_semantic_model(table_name="orders")
        assert result is not None
        assert result["table_name"] == "orders"
        assert result["description"] == "Orders table"

    def test_get_semantic_model_with_children(self, sem_rag):
        """Retrieve a table with dimension, measure, and identifier children."""
        objs = [
            _make_table_object("orders", description="Orders table"),
            _make_column_object(
                "orders",
                "region",
                description="Region dimension",
                is_dimension=True,
                column_type="CATEGORICAL",
            ),
            _make_column_object(
                "orders",
                "amount",
                description="Total amount",
                is_measure=True,
                agg="SUM",
                create_metric=True,
                agg_time_dimension="order_date",
            ),
            _make_column_object(
                "orders",
                "order_id",
                description="Primary key",
                is_entity_key=True,
                column_type="PRIMARY",
                entity="order",
            ),
        ]
        sem_rag.storage.store_batch(objs)

        result = sem_rag.get_semantic_model(table_name="orders")
        assert result is not None

        # Verify dimensions
        assert len(result["dimensions"]) == 1
        dim = result["dimensions"][0]
        assert dim["name"] == "region"
        assert dim["type"] == "CATEGORICAL"

        # Verify measures
        assert len(result["measures"]) == 1
        measure = result["measures"][0]
        assert measure["name"] == "amount"
        assert measure["agg"] == "SUM"
        assert measure["create_metric"] is True
        assert measure["agg_time_dimension"] == "order_date"

        # Verify identifiers
        assert len(result["identifiers"]) == 1
        ident = result["identifiers"][0]
        assert ident["name"] == "order_id"
        assert ident["type"] == "PRIMARY"
        assert ident["entity"] == "order"

    def test_get_semantic_model_with_full_filter(self, sem_rag):
        """Retrieve with catalog/database/schema filters matching exactly."""
        objs = [
            _make_table_object(
                "orders",
                description="Orders",
                catalog_name="prod",
                database_name="sales",
                schema_name="dbo",
            )
        ]
        sem_rag.storage.store_batch(objs)

        result = sem_rag.get_semantic_model(
            catalog_name="prod", database_name="sales", schema_name="dbo", table_name="orders"
        )
        assert result is not None
        assert result["table_name"] == "orders"

    def test_get_semantic_model_fallback_broad_match(self, sem_rag):
        """When full filter fails, fallback to table_name-only match."""
        objs = [
            _make_table_object(
                "orders",
                description="Orders",
                catalog_name="prod",
                database_name="sales",
                schema_name="dbo",
            )
        ]
        sem_rag.storage.store_batch(objs)

        # Use a different catalog_name to trigger fallback
        result = sem_rag.get_semantic_model(
            catalog_name="wrong_catalog", database_name="wrong_db", schema_name="wrong_schema", table_name="orders"
        )
        assert result is not None
        assert result["table_name"] == "orders"

    def test_get_semantic_model_fallback_case_insensitive(self, sem_rag):
        """When table is stored with lowercase, querying uppercase triggers case-insensitive fallback."""
        objs = [_make_table_object("orders", description="Orders table")]
        sem_rag.storage.store_batch(objs)

        # Query with uppercase -- will try exact match first, then broad, then lowercase fallback
        # Since "ORDERS" != "orders", exact match fails, broad match also uses "ORDERS",
        # then case-insensitive tries "orders" (lowercase) which should succeed
        result = sem_rag.get_semantic_model(table_name="ORDERS")
        assert result is not None
        assert result["table_name"] == "orders"

    def test_get_semantic_model_with_select_fields(self, sem_rag):
        """select_fields filters the returned dict."""
        objs = [_make_table_object("orders", description="Orders")]
        sem_rag.storage.store_batch(objs)

        result = sem_rag.get_semantic_model(table_name="orders", select_fields=["table_name", "description"])
        assert result is not None
        assert "table_name" in result
        assert "description" in result
        # Fields not in select_fields should not be present
        assert "dimensions" not in result
        assert "measures" not in result

    def test_get_semantic_model_dimension_with_partition(self, sem_rag):
        """Dimension with is_partition flag should include it in result."""
        objs = [
            _make_table_object("events", description="Events table"),
            _make_column_object(
                "events",
                "event_date",
                description="Date of event",
                is_dimension=True,
                column_type="TIME",
                is_partition=True,
                time_granularity="DAY",
            ),
        ]
        sem_rag.storage.store_batch(objs)

        result = sem_rag.get_semantic_model(table_name="events")
        assert result is not None
        assert len(result["dimensions"]) == 1
        dim = result["dimensions"][0]
        assert dim["is_partition"] is True
        assert dim["time_granularity"] == "DAY"

    def test_get_semantic_model_column_without_flags(self, sem_rag):
        """A column that is not dimension/measure/identifier should not appear in any list."""
        objs = [
            _make_table_object("orders", description="Orders"),
            _make_column_object("orders", "internal_col", description="Internal column"),
        ]
        sem_rag.storage.store_batch(objs)

        result = sem_rag.get_semantic_model(table_name="orders")
        assert result is not None
        assert len(result["dimensions"]) == 0
        assert len(result["measures"]) == 0
        assert len(result["identifiers"]) == 0


# ============================================================
# SemanticModelRAG.search_all
# ============================================================


class TestSemanticModelRAGSearchAll:
    """Tests for search_all method."""

    def test_search_all_empty(self, sem_rag):
        """Empty storage should return empty list."""
        results = sem_rag.search_all()
        assert results == []

    def test_search_all_returns_tables(self, sem_rag):
        """search_all returns table-level objects."""
        objs = [
            _make_table_object("orders", description="Orders"),
            _make_table_object("products", description="Products"),
            _make_column_object("orders", "id", description="Order ID", is_entity_key=True),
        ]
        sem_rag.storage.store_batch(objs)

        results = sem_rag.search_all()
        # search_all filters kind=table
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert names == {"orders", "products"}

    def test_search_all_with_database_filter(self, sem_rag):
        """search_all with database_name filter narrows results."""
        objs = [
            _make_table_object("orders", description="Orders", database_name="sales"),
            _make_table_object("products", description="Products", database_name="catalog"),
        ]
        sem_rag.storage.store_batch(objs)

        results = sem_rag.search_all(database_name="sales")
        assert len(results) == 1
        assert results[0]["name"] == "orders"


# ============================================================
# SemanticModelRAG.get_size
# ============================================================


class TestSemanticModelRAGGetSize:
    """Tests for get_size method."""

    def test_get_size_empty(self, sem_rag):
        """Empty storage should return 0."""
        assert sem_rag.get_size() == 0

    def test_get_size_counts_tables_only(self, sem_rag):
        """get_size counts only table-kind objects, not columns."""
        objs = [
            _make_table_object("orders", description="Orders table"),
            _make_table_object("products", description="Products table"),
            _make_column_object("orders", "amount", description="Amount", is_measure=True),
        ]
        sem_rag.storage.store_batch(objs)

        assert sem_rag.get_size() == 2


# ============================================================
# SemanticModelRAG.store_batch / upsert_batch
# ============================================================


class TestSemanticModelRAGStoreUpsert:
    """Tests for store_batch and upsert_batch via RAG."""

    def test_store_batch_via_rag(self, sem_rag):
        """Storing via RAG delegates to storage.store_batch."""
        objs = [_make_table_object("orders", description="Orders")]
        sem_rag.store_batch(objs)
        assert sem_rag.get_size() >= 1

    def test_upsert_batch_via_rag(self, sem_rag):
        """Upserting via RAG delegates to storage.upsert_batch."""
        objs = [_make_table_object("orders", description="Original")]
        sem_rag.store_batch(objs)

        updated = [_make_table_object("orders", description="Updated")]
        sem_rag.upsert_batch(updated)

        result = sem_rag.get_semantic_model(table_name="orders")
        assert result is not None
        assert result["description"] == "Updated"


# ============================================================
# SemanticModelRAG.truncate
# ============================================================


class TestSemanticModelRAGTruncate:
    """Tests for truncate method."""

    def test_truncate_clears_data(self, sem_rag):
        """Truncate should remove all data."""
        objs = [_make_table_object("orders", description="Orders")]
        sem_rag.store_batch(objs)
        assert sem_rag.get_size() >= 1

        sem_rag.truncate()
        assert sem_rag.get_size() == 0


# ============================================================
# SemanticModelRAG.create_indices
# ============================================================


class TestSemanticModelRAGCreateIndices:
    """Tests for create_indices via RAG."""

    def test_create_indices_via_rag(self, sem_rag):
        """Creating indices via RAG should not raise."""
        objs = [_make_table_object("orders", description="Orders table")]
        sem_rag.store_batch(objs)
        sem_rag.create_indices()
        # Verify data is still accessible
        assert sem_rag.get_size() >= 1
