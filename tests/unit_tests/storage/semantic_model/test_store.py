# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/semantic_model/store.py -- SemanticModelStorage and SemanticModelRAG."""

import os
import tempfile

import pytest
import yaml
from pandas import Timestamp

from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.semantic_model.store import SemanticModelRAG, SemanticModelStorage
from datus.utils.exceptions import DatusException

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
# SemanticModelStorage.update_entry / _sync_semantic_update_to_yaml
# ============================================================


class TestUpdateEntryYamlSync:
    """Tests for update_entry and _sync_semantic_update_to_yaml."""

    def _make_yaml_with_table_only(self, f, description="Original table description"):
        """Write a minimal single-table YAML (no columns) to file f."""
        docs = [
            {
                "data_source": {
                    "name": "orders",
                    "description": description,
                    "sql_table": "analytics.public.orders",
                }
            }
        ]
        yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)
        f.flush()

    def _make_yaml_with_columns(self, f, dim_description="Region dimension", dim_type="CATEGORICAL"):
        """Write a YAML with a data_source containing dimensions, measures, identifiers."""
        docs = [
            {
                "data_source": {
                    "name": "orders",
                    "description": "Original table description",
                    "sql_table": "analytics.public.orders",
                    "dimensions": [
                        {"name": "region", "type": dim_type, "description": dim_description, "expr": "region"}
                    ],
                    "measures": [{"name": "amount", "description": "Total amount", "agg": "SUM", "expr": "amount"}],
                    "identifiers": [
                        {"name": "order_id", "type": "PRIMARY", "description": "Primary key", "expr": "order_id"}
                    ],
                }
            }
        ]
        yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)
        f.flush()

    def test_update_table_description_syncs_to_yaml(self, sem_storage):
        """update_entry on a table entry syncs the description back to the YAML file."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
        try:
            self._make_yaml_with_table_only(tmp, description="Original")
            tmp.close()

            table_obj = _make_table_object("orders", description="Original", yaml_path=tmp.name)
            sem_storage.store_batch([table_obj])

            result = sem_storage.update_entry("table:orders", {"description": "Updated table description"})
            assert result is True

            with open(tmp.name, encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
            data_source = next(d["data_source"] for d in docs if d and "data_source" in d)
            assert data_source["description"] == "Updated table description"
        finally:
            os.unlink(tmp.name)

    def test_update_column_description_syncs_to_yaml(self, sem_storage):
        """update_entry on a column entry syncs the description to the matching dimension item."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
        try:
            self._make_yaml_with_columns(tmp, dim_description="Original region desc")
            tmp.close()

            table_obj = _make_table_object("orders", yaml_path=tmp.name)
            col_obj = _make_column_object(
                "orders", "region", description="Original region desc", is_dimension=True, column_type="CATEGORICAL"
            )
            col_obj["yaml_path"] = tmp.name
            sem_storage.store_batch([table_obj, col_obj])

            result = sem_storage.update_entry("column:orders.region", {"description": "Updated region desc"})
            assert result is True

            with open(tmp.name, encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
            data_source = next(d["data_source"] for d in docs if d and "data_source" in d)
            dim = next(item for item in data_source["dimensions"] if item["name"] == "region")
            assert dim["description"] == "Updated region desc"
        finally:
            os.unlink(tmp.name)

    def test_update_column_type_syncs_to_yaml(self, sem_storage):
        """update_entry with column_type maps to 'type' key in YAML."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
        try:
            self._make_yaml_with_columns(tmp, dim_type="CATEGORICAL")
            tmp.close()

            table_obj = _make_table_object("orders", yaml_path=tmp.name)
            col_obj = _make_column_object(
                "orders", "region", description="Region dimension", is_dimension=True, column_type="CATEGORICAL"
            )
            col_obj["yaml_path"] = tmp.name
            sem_storage.store_batch([table_obj, col_obj])

            result = sem_storage.update_entry("column:orders.region", {"column_type": "TIME"})
            assert result is True

            with open(tmp.name, encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
            data_source = next(d["data_source"] for d in docs if d and "data_source" in d)
            dim = next(item for item in data_source["dimensions"] if item["name"] == "region")
            assert dim["type"] == "TIME"
        finally:
            os.unlink(tmp.name)

    def test_update_measure_agg_syncs_to_yaml(self, sem_storage):
        """update_entry on a measure column syncs 'agg' back to the YAML file."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
        try:
            self._make_yaml_with_columns(tmp)
            tmp.close()

            table_obj = _make_table_object("orders", yaml_path=tmp.name)
            col_obj = _make_column_object("orders", "amount", description="Total amount", is_measure=True, agg="SUM")
            col_obj["yaml_path"] = tmp.name
            sem_storage.store_batch([table_obj, col_obj])

            result = sem_storage.update_entry("column:orders.amount", {"agg": "AVERAGE"})
            assert result is True

            with open(tmp.name, encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
            data_source = next(d["data_source"] for d in docs if d and "data_source" in d)
            measure = next(item for item in data_source["measures"] if item["name"] == "amount")
            assert measure["agg"] == "AVERAGE"
        finally:
            os.unlink(tmp.name)

    def test_update_identifier_entity_syncs_to_yaml(self, sem_storage):
        """update_entry on an identifier column syncs 'entity' back to the YAML file."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
        try:
            self._make_yaml_with_columns(tmp)
            tmp.close()

            table_obj = _make_table_object("orders", yaml_path=tmp.name)
            col_obj = _make_column_object(
                "orders", "order_id", description="Primary key", is_entity_key=True, entity="order"
            )
            col_obj["yaml_path"] = tmp.name
            sem_storage.store_batch([table_obj, col_obj])

            result = sem_storage.update_entry("column:orders.order_id", {"entity": "transaction"})
            assert result is True

            with open(tmp.name, encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
            data_source = next(d["data_source"] for d in docs if d and "data_source" in d)
            ident = next(item for item in data_source["identifiers"] if item["name"] == "order_id")
            assert ident["entity"] == "transaction"
        finally:
            os.unlink(tmp.name)

    def test_update_entry_no_yaml_path_still_succeeds(self, sem_storage):
        """update_entry succeeds and returns True when yaml_path is empty."""
        table_obj = _make_table_object("orders", description="A table", yaml_path="")
        sem_storage.store_batch([table_obj])

        result = sem_storage.update_entry("table:orders", {"description": "No yaml sync"})
        assert result is True

    def test_update_entry_nonexistent_raises(self, sem_storage):
        """update_entry raises DatusException when the entry_id does not exist."""
        with pytest.raises(DatusException, match="entry not found"):
            sem_storage.update_entry("table:nonexistent", {"description": "x"})

    def test_update_entry_empty_id_raises(self, sem_storage):
        """update_entry raises DatusException when entry_id is empty."""
        with pytest.raises(DatusException, match="entry_id must not be empty"):
            sem_storage.update_entry("", {"description": "x"})

    def test_update_entry_empty_values_raises(self, sem_storage):
        """update_entry raises DatusException when update_values is empty."""
        table_obj = _make_table_object("orders")
        sem_storage.store_batch([table_obj])

        with pytest.raises(DatusException, match="update_values must not be empty"):
            sem_storage.update_entry("table:orders", {})

    def test_update_entry_nonexistent_yaml_file(self, sem_storage):
        """update_entry succeeds even when yaml_path points to a missing file."""
        table_obj = _make_table_object("orders", yaml_path="/nonexistent/path.yml")
        sem_storage.store_batch([table_obj])

        result = sem_storage.update_entry("table:orders", {"description": "new"})
        assert result is True

    def test_sync_yaml_no_data_source_doc(self, sem_storage):
        """_sync_semantic_update_to_yaml returns silently when YAML has no data_source."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
        try:
            yaml.safe_dump_all([{"unrelated": "doc"}], tmp, allow_unicode=True, sort_keys=False)
            tmp.close()

            table_obj = _make_table_object("orders", yaml_path=tmp.name)
            sem_storage.store_batch([table_obj])

            result = sem_storage.update_entry("table:orders", {"description": "new"})
            assert result is True

            with open(tmp.name, encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
            assert docs[0] == {"unrelated": "doc"}
        finally:
            os.unlink(tmp.name)

    def test_sync_yaml_corrupt_file_does_not_raise(self, sem_storage):
        """_sync_semantic_update_to_yaml catches exceptions on corrupt YAML files."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
        try:
            tmp.write("{{invalid yaml content")
            tmp.close()

            # Call private method directly — should log error but not raise
            sem_storage._sync_semantic_update_to_yaml(tmp.name, "table", "orders", "orders", {"description": "new"})
        finally:
            os.unlink(tmp.name)

    def test_update_column_not_found_in_yaml(self, sem_storage):
        """_sync_semantic_update_to_yaml for a column not present in YAML leaves file unchanged."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
        try:
            docs = [
                {
                    "data_source": {
                        "name": "orders",
                        "description": "Table",
                        "dimensions": [{"name": "region", "description": "Region"}],
                    }
                }
            ]
            yaml.safe_dump_all(docs, tmp, allow_unicode=True, sort_keys=False)
            tmp.close()

            col_obj = _make_column_object("orders", "nonexistent_col", is_dimension=True)
            col_obj["yaml_path"] = tmp.name
            table_obj = _make_table_object("orders", yaml_path=tmp.name)
            sem_storage.store_batch([table_obj, col_obj])

            result = sem_storage.update_entry("column:orders.nonexistent_col", {"description": "new"})
            assert result is True

            with open(tmp.name, encoding="utf-8") as f:
                reloaded = list(yaml.safe_load_all(f))
            # Original region dimension should be untouched
            assert reloaded[0]["data_source"]["dimensions"][0]["description"] == "Region"
        finally:
            os.unlink(tmp.name)

    def test_sync_yaml_multi_data_source_targets_matching_doc(self, sem_storage):
        """When a YAML file holds multiple data_source docs, only the matching one is rewritten."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
        try:
            docs = [
                {"data_source": {"name": "customers", "description": "Customers original"}},
                {"data_source": {"name": "orders", "description": "Orders original"}},
            ]
            yaml.safe_dump_all(docs, tmp, allow_unicode=True, sort_keys=False)
            tmp.close()

            table_obj = _make_table_object("orders", description="Orders original", yaml_path=tmp.name)
            sem_storage.store_batch([table_obj])

            sem_storage.update_entry("table:orders", {"description": "Orders updated"})

            with open(tmp.name, encoding="utf-8") as f:
                reloaded = list(yaml.safe_load_all(f))

            ds_by_name = {d["data_source"]["name"]: d["data_source"] for d in reloaded}
            assert ds_by_name["orders"]["description"] == "Orders updated"
            # The unrelated data_source must not be touched
            assert ds_by_name["customers"]["description"] == "Customers original"
        finally:
            os.unlink(tmp.name)

    def test_sync_yaml_multi_data_source_no_match_does_not_mutate(self, sem_storage):
        """When no data_source name matches and >1 docs exist, no doc is rewritten."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
        try:
            docs = [
                {"data_source": {"name": "customers", "description": "Customers original"}},
                {"data_source": {"name": "products", "description": "Products original"}},
            ]
            yaml.safe_dump_all(docs, tmp, allow_unicode=True, sort_keys=False)
            tmp.close()

            # The vector-DB row points at this file, but its name "orders" matches no doc
            table_obj = _make_table_object("orders", description="Orders DB", yaml_path=tmp.name)
            sem_storage.store_batch([table_obj])

            sem_storage.update_entry("table:orders", {"description": "Should not appear"})

            with open(tmp.name, encoding="utf-8") as f:
                reloaded = list(yaml.safe_load_all(f))

            ds_by_name = {d["data_source"]["name"]: d["data_source"] for d in reloaded}
            # Both unrelated docs must remain pristine
            assert ds_by_name["customers"]["description"] == "Customers original"
            assert ds_by_name["products"]["description"] == "Products original"
        finally:
            os.unlink(tmp.name)


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
        sem_rag.store_batch(objs)

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
        sem_rag.store_batch(objs)

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
        sem_rag.store_batch(objs)

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
        sem_rag.store_batch(objs)

        # Use a different catalog_name to trigger fallback
        result = sem_rag.get_semantic_model(
            catalog_name="wrong_catalog", database_name="wrong_db", schema_name="wrong_schema", table_name="orders"
        )
        assert result is not None
        assert result["table_name"] == "orders"

    def test_get_semantic_model_fallback_case_insensitive(self, sem_rag):
        """When table is stored with lowercase, querying uppercase triggers case-insensitive fallback."""
        objs = [_make_table_object("orders", description="Orders table")]
        sem_rag.store_batch(objs)

        # Query with uppercase -- will try exact match first, then broad, then lowercase fallback
        # Since "ORDERS" != "orders", exact match fails, broad match also uses "ORDERS",
        # then case-insensitive tries "orders" (lowercase) which should succeed
        result = sem_rag.get_semantic_model(table_name="ORDERS")
        assert result is not None
        assert result["table_name"] == "orders"

    def test_get_semantic_model_with_select_fields(self, sem_rag):
        """select_fields filters the returned dict."""
        objs = [_make_table_object("orders", description="Orders")]
        sem_rag.store_batch(objs)

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
        sem_rag.store_batch(objs)

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
        sem_rag.store_batch(objs)

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
        sem_rag.store_batch(objs)

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
        sem_rag.store_batch(objs)

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
        sem_rag.store_batch(objs)

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
