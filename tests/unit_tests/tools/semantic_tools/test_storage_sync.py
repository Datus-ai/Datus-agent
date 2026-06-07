# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/semantic_tools/storage_sync.py"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.tools.semantic_tools.storage_sync import SemanticStorageManager


def _make_manager():
    mock_config = MagicMock()
    return SemanticStorageManager(agent_config=mock_config)


def _make_metric(name="revenue", path=None, description="Total revenue"):
    from datus.tools.semantic_tools.models import MetricDefinition

    return MetricDefinition(
        name=name,
        description=description,
        type="simple",
        dimensions=["region"],
        measures=["total"],
        unit="USD",
        format="currency",
        path=path,
    )


class TestSemanticStorageManagerInit:
    def test_init_sets_agent_config(self):
        mock_config = MagicMock()
        manager = SemanticStorageManager(agent_config=mock_config)
        assert manager.agent_config is mock_config
        assert manager.semantic_model_store is None
        assert manager.metric_store is None
        assert manager.subject_tree_store is None


class TestEnsureSemanticModelStore:
    def test_lazy_init_creates_store(self):
        """_ensure_semantic_model_store lazily creates store via SemanticModelRAG import."""
        manager = _make_manager()
        mock_rag_instance = MagicMock()
        mock_expected_store = MagicMock()
        mock_rag_instance.storage = mock_expected_store
        mock_rag_class = MagicMock(return_value=mock_rag_instance)

        with patch("datus.storage.semantic_model.store.SemanticModelRAG", mock_rag_class):
            store = manager._ensure_semantic_model_store()

        assert store is mock_expected_store
        assert manager.semantic_model_store is mock_expected_store

    def test_returns_existing_store_on_second_call(self):
        manager = _make_manager()
        mock_store = MagicMock()
        manager.semantic_model_store = mock_store

        result = manager._ensure_semantic_model_store()
        assert result is mock_store


class TestEnsureMetricStore:
    def test_lazy_init_creates_store(self):
        """_ensure_metric_store lazily creates store via MetricRAG import."""
        manager = _make_manager()
        mock_rag_instance = MagicMock()
        mock_expected_store = MagicMock()
        mock_rag_instance.storage = mock_expected_store
        mock_rag_class = MagicMock(return_value=mock_rag_instance)

        with patch("datus.storage.metric.store.MetricRAG", mock_rag_class):
            store = manager._ensure_metric_store()

        assert store is mock_expected_store
        assert manager.metric_store is mock_expected_store

    def test_returns_existing_store_on_second_call(self):
        manager = _make_manager()
        mock_store = MagicMock()
        manager.metric_store = mock_store
        result = manager._ensure_metric_store()
        assert result is mock_store


class TestEnsureSubjectTreeStore:
    def test_lazy_init_creates_store(self):
        manager = _make_manager()
        mock_store = MagicMock()

        with patch("datus.storage.registry.get_subject_tree_store", return_value=mock_store):
            store = manager._ensure_subject_tree_store()

        assert store is mock_store

    def test_returns_existing_on_second_call(self):
        manager = _make_manager()
        mock_store = MagicMock()
        manager.subject_tree_store = mock_store
        result = manager._ensure_subject_tree_store()
        assert result is mock_store


class TestStoreSemanticModel:
    def test_raises_for_missing_semantic_model_name(self):
        manager = _make_manager()
        with pytest.raises(ValueError, match="semantic_model_name"):
            manager.store_semantic_model({"table_name": "users"})

    def test_stores_table_object(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "user_model",
                    "table_name": "users",
                    "database_name": "test_db",
                    "schema_name": "public",
                    "description": "User table",
                    "dimensions": [],
                    "measures": [],
                }
            )

        # batch_store called at least once (for the table object)
        assert mock_store.batch_store.call_count >= 1
        first_call_args = mock_store.batch_store.call_args_list[0][0][0]
        table_obj = first_call_args[0]
        assert table_obj["kind"] == "table"
        assert table_obj["table_name"] == "users"

    def test_stores_dimensions(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "user_model",
                    "table_name": "users",
                    "dimensions": [
                        {"name": "region", "description": "Sales region"},
                        {"name": "product", "description": "Product name"},
                    ],
                    "measures": [],
                }
            )

        # Should have stored table + dimensions (2 calls or table + one batch of dims)
        all_stored = []
        for call in mock_store.batch_store.call_args_list:
            all_stored.extend(call[0][0])

        dim_objects = [obj for obj in all_stored if obj.get("is_dimension") is True]
        assert len(dim_objects) == 2
        dim_names = [d["name"] for d in dim_objects]
        assert "region" in dim_names
        assert "product" in dim_names

    def test_stores_measures(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "sales_model",
                    "table_name": "sales",
                    "dimensions": [],
                    "measures": [{"name": "total_sales", "description": "Total sales"}],
                }
            )

        all_stored = []
        for call in mock_store.batch_store.call_args_list:
            all_stored.extend(call[0][0])

        measure_objects = [obj for obj in all_stored if obj.get("is_measure") is True]
        assert len(measure_objects) == 1
        assert measure_objects[0]["name"] == "total_sales"

    def test_stores_identifiers(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "order_model",
                    "table_name": "orders",
                    "dimensions": [],
                    "measures": [],
                    "identifiers": [{"name": "order_id", "description": "Order identifier"}],
                }
            )

        all_stored = []
        for call in mock_store.batch_store.call_args_list:
            all_stored.extend(call[0][0])

        id_objects = [obj for obj in all_stored if obj.get("is_entity_key") is True]
        assert len(id_objects) == 1
        assert id_objects[0]["name"] == "order_id"

    def test_skips_dimensions_without_name(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "model",
                    "table_name": "table",
                    "dimensions": [
                        {"name": "valid_dim"},
                        {"description": "no name here"},  # missing name
                    ],
                    "measures": [],
                }
            )

        all_stored = []
        for call in mock_store.batch_store.call_args_list:
            all_stored.extend(call[0][0])

        dim_objects = [obj for obj in all_stored if obj.get("is_dimension") is True]
        assert len(dim_objects) == 1

    def test_builds_fully_qualified_name(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "fq_model",
                    "catalog_name": "my_catalog",
                    "database_name": "my_db",
                    "schema_name": "my_schema",
                    "table_name": "my_table",
                    "dimensions": [],
                    "measures": [],
                }
            )

        first_call = mock_store.batch_store.call_args_list[0][0][0]
        table_obj = first_call[0]
        assert table_obj["fq_name"] == "my_catalog.my_db.my_schema.my_table"

    def test_skips_empty_parts_in_fq_name(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "simple_model",
                    "table_name": "simple_table",
                    "dimensions": [],
                    "measures": [],
                }
            )

        first_call = mock_store.batch_store.call_args_list[0][0][0]
        table_obj = first_call[0]
        assert table_obj["fq_name"] == "simple_table"

    def test_stores_semantic_model_info_with_extra_metadata(self):
        from datus.tools.semantic_tools.models import DimensionInfo, SemanticModelInfo

        manager = _make_manager()
        mock_store = MagicMock()
        model = SemanticModelInfo(
            name="orders_cube",
            description="Orders",
            platform_type="cube",
            dimensions=[DimensionInfo(name="region", description="Region")],
            measures=["orders.count"],
            extra={"table_name": "orders", "database_name": "analytics", "connectedComponent": 1},
        )
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(model)

        all_stored = []
        for call in mock_store.batch_store.call_args_list:
            all_stored.extend(call[0][0])
        table_obj = [o for o in all_stored if o["kind"] == "table"][0]
        assert table_obj["table_name"] == "orders"  # from extra, not model.name
        assert table_obj["database_name"] == "analytics"
        assert table_obj["semantic_model_name"] == "orders_cube"
        dim_objects = [o for o in all_stored if o.get("is_dimension")]
        assert len(dim_objects) == 1
        assert dim_objects[0]["name"] == "region"

    def test_stores_semantic_model_info_with_direct_fields(self):
        from datus.tools.semantic_tools.models import DimensionInfo, SemanticModelInfo

        manager = _make_manager()
        mock_store = MagicMock()
        model = SemanticModelInfo(
            name="orders_cube",
            description="Orders",
            table_name="orders",
            database_name="analytics",
            schema_name="public",
            dimensions=[DimensionInfo(name="region", description="Region")],
            measures=["orders.count"],
        )
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(model)

        all_stored = []
        for call in mock_store.batch_store.call_args_list:
            all_stored.extend(call[0][0])
        table_obj = [o for o in all_stored if o["kind"] == "table"][0]
        assert table_obj["table_name"] == "orders"
        assert table_obj["database_name"] == "analytics"
        assert table_obj["schema_name"] == "public"
        assert table_obj["fq_name"] == "analytics.public.orders"

    def test_rejects_semantic_model_info_without_physical_table_name(self):
        from datus.tools.semantic_tools.models import SemanticModelInfo
        from datus.utils.exceptions import DatusException

        manager = _make_manager()
        with pytest.raises(DatusException, match="missing physical table_name"):
            manager.store_semantic_model(SemanticModelInfo(name="orders_cube"))

    def test_skips_non_dict_dimensions(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_semantic_model_store", return_value=mock_store):
            manager.store_semantic_model(
                {
                    "semantic_model_name": "model",
                    "table_name": "table",
                    "dimensions": ["just_a_string", None, {"name": "valid"}],
                    "measures": [],
                }
            )
        all_stored = []
        for call in mock_store.batch_store.call_args_list:
            all_stored.extend(call[0][0])
        dim_objects = [obj for obj in all_stored if obj.get("is_dimension") is True]
        assert len(dim_objects) == 1


class TestStoreMetric:
    def test_raises_for_missing_name(self):
        manager = _make_manager()
        with pytest.raises(ValueError, match="'name'"):
            manager.store_metric({"description": "no name"})

    def test_stores_metric_with_defaults(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_metric_store", return_value=mock_store):
            manager.store_metric({"name": "revenue"})

        mock_store.batch_store_metrics.assert_called_once()
        stored = mock_store.batch_store_metrics.call_args[0][0]
        assert len(stored) == 1
        assert stored[0]["name"] == "revenue"
        assert stored[0]["subject_path"] == ["Uncategorized"]

    def test_uses_provided_subject_path(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_metric_store", return_value=mock_store):
            manager.store_metric({"name": "revenue"}, subject_path=["Finance", "Q1"])

        stored = mock_store.batch_store_metrics.call_args[0][0]
        assert stored[0]["subject_path"] == ["Finance", "Q1"]

    def test_metric_id_includes_subject_path(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_metric_store", return_value=mock_store):
            manager.store_metric({"name": "revenue"}, subject_path=["Finance", "Revenue"])

        stored = mock_store.batch_store_metrics.call_args[0][0]
        assert "Finance/Revenue.revenue" in stored[0]["id"]

    def test_metric_type_defaults_to_simple(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_metric_store", return_value=mock_store):
            manager.store_metric({"name": "count"})

        stored = mock_store.batch_store_metrics.call_args[0][0]
        assert stored[0]["metric_type"] == "simple"

    def test_custom_metric_type(self):
        manager = _make_manager()
        mock_store = MagicMock()
        with patch.object(manager, "_ensure_metric_store", return_value=mock_store):
            manager.store_metric({"name": "ratio", "metric_type": "ratio"})

        stored = mock_store.batch_store_metrics.call_args[0][0]
        assert stored[0]["metric_type"] == "ratio"


class TestSyncFromAdapter:
    def test_sync_semantic_models(self):
        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_semantic_models.return_value = ["users", "orders"]
        mock_adapter.get_semantic_model.side_effect = lambda table_name: {
            "semantic_model_name": table_name,
            "table_name": table_name,
            "dimensions": [],
            "measures": [],
        }
        mock_adapter.list_metrics = AsyncMock(return_value=[])

        with patch.object(manager, "store_semantic_model") as mock_store_sm:
            stats = asyncio.run(
                manager.sync_from_adapter(
                    adapter=mock_adapter,
                    sync_semantic_models=True,
                    sync_metrics=False,
                )
            )

        assert stats["semantic_models_synced"] == 2
        assert mock_store_sm.call_count == 2

    def test_sync_metrics(self):
        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_semantic_models.return_value = []
        metrics = [_make_metric("revenue"), _make_metric("cost")]
        mock_adapter.list_metrics = AsyncMock(return_value=metrics)

        with patch.object(manager, "store_metric") as mock_store_metric:
            stats = asyncio.run(
                manager.sync_from_adapter(
                    adapter=mock_adapter,
                    sync_semantic_models=False,
                    sync_metrics=True,
                )
            )

        assert stats["metrics_synced"] == 2
        assert mock_store_metric.call_count == 2

    def test_sync_with_both_disabled(self):
        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_metrics = AsyncMock(return_value=[])

        stats = asyncio.run(
            manager.sync_from_adapter(
                adapter=mock_adapter,
                sync_semantic_models=False,
                sync_metrics=False,
            )
        )

        assert stats["semantic_models_synced"] == 0
        assert stats["metrics_synced"] == 0

    def test_sync_handles_list_semantic_models_error(self):
        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_semantic_models.side_effect = Exception("connection failed")
        mock_adapter.list_metrics = AsyncMock(return_value=[])

        # Should not raise
        stats = asyncio.run(
            manager.sync_from_adapter(
                adapter=mock_adapter,
                sync_semantic_models=True,
                sync_metrics=False,
            )
        )

        assert stats["semantic_models_synced"] == 0

    def test_sync_handles_get_semantic_model_error(self):
        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_semantic_models.return_value = ["failing_table"]
        mock_adapter.get_semantic_model.side_effect = Exception("fetch failed")
        mock_adapter.list_metrics = AsyncMock(return_value=[])

        stats = asyncio.run(
            manager.sync_from_adapter(
                adapter=mock_adapter,
                sync_semantic_models=True,
                sync_metrics=False,
            )
        )

        assert stats["semantic_models_synced"] == 0

    def test_sync_handles_list_metrics_error(self):
        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_semantic_models.return_value = []
        mock_adapter.list_metrics = AsyncMock(side_effect=Exception("metrics unavailable"))

        stats = asyncio.run(
            manager.sync_from_adapter(
                adapter=mock_adapter,
                sync_semantic_models=False,
                sync_metrics=True,
            )
        )

        assert stats["metrics_synced"] == 0

    def test_sync_handles_store_metric_error(self):
        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_semantic_models.return_value = []
        mock_adapter.list_metrics = AsyncMock(return_value=[_make_metric("revenue")])

        with patch.object(manager, "store_metric", side_effect=Exception("store failed")):
            stats = asyncio.run(
                manager.sync_from_adapter(
                    adapter=mock_adapter,
                    sync_semantic_models=False,
                    sync_metrics=True,
                )
            )

        assert stats["metrics_synced"] == 0

    def test_sync_uses_metric_path_when_available(self):
        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_semantic_models.return_value = []
        metric = _make_metric("revenue", path=["Finance", "Revenue"])
        mock_adapter.list_metrics = AsyncMock(return_value=[metric])

        stored_calls = []

        def capture_store(metric_data, subject_path=None):
            stored_calls.append(subject_path)

        with patch.object(manager, "store_metric", side_effect=capture_store):
            asyncio.run(
                manager.sync_from_adapter(
                    adapter=mock_adapter,
                    sync_semantic_models=False,
                    sync_metrics=True,
                    subject_path=["Default"],
                )
            )

        # The metric's own path should be used, not the default
        assert stored_calls[0] == ["Finance", "Revenue"]

    def test_sync_uses_provided_subject_path_when_metric_has_no_path(self):
        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_semantic_models.return_value = []
        metric = _make_metric("revenue", path=None)  # no path on the metric
        mock_adapter.list_metrics = AsyncMock(return_value=[metric])

        stored_calls = []

        def capture_store(metric_data, subject_path=None):
            stored_calls.append(subject_path)

        with patch.object(manager, "store_metric", side_effect=capture_store):
            asyncio.run(
                manager.sync_from_adapter(
                    adapter=mock_adapter,
                    sync_semantic_models=False,
                    sync_metrics=True,
                    subject_path=["Default", "Category"],
                )
            )

        assert stored_calls[0] == ["Default", "Category"]

    def test_sync_semantic_models_with_semantic_model_info_entries(self):
        from datus.tools.semantic_tools.models import SemanticModelInfo

        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        model_info = SemanticModelInfo(
            name="orders_cube",
            description="Orders cube",
            extra={"table_name": "orders"},
        )
        mock_adapter.list_semantic_models.return_value = [model_info]
        mock_adapter.list_metrics = AsyncMock(return_value=[])

        with patch.object(manager, "store_semantic_model") as mock_store_sm:
            stats = asyncio.run(
                manager.sync_from_adapter(adapter=mock_adapter, sync_semantic_models=True, sync_metrics=False)
            )
        assert stats["semantic_models_synced"] == 1
        mock_store_sm.assert_called_once_with(model_info)
        mock_adapter.get_semantic_model.assert_not_called()

    def test_sync_semantic_model_info_without_table_name_fetches_full_model(self):
        from datus.tools.semantic_tools.models import SemanticModelInfo

        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        model_info = SemanticModelInfo(name="orders_cube", description="Orders cube")
        hydrated_model = {
            "semantic_model_name": "orders_cube",
            "table_name": "orders",
            "dimensions": [],
            "measures": [],
        }
        mock_adapter.list_semantic_models.return_value = [model_info]
        mock_adapter.get_semantic_model.return_value = hydrated_model
        mock_adapter.list_metrics = AsyncMock(return_value=[])

        with patch.object(manager, "store_semantic_model") as mock_store_sm:
            stats = asyncio.run(
                manager.sync_from_adapter(adapter=mock_adapter, sync_semantic_models=True, sync_metrics=False)
            )

        assert stats["semantic_models_synced"] == 1
        mock_adapter.get_semantic_model.assert_called_once_with(table_name="orders_cube")
        mock_store_sm.assert_called_once_with(hydrated_model)

    def test_skip_semantic_model_info_without_table_name_when_fetch_returns_none(self):
        from datus.tools.semantic_tools.models import SemanticModelInfo

        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_semantic_models.return_value = [SemanticModelInfo(name="orders_cube")]
        mock_adapter.get_semantic_model.return_value = None
        mock_adapter.list_metrics = AsyncMock(return_value=[])

        with patch.object(manager, "store_semantic_model") as mock_store_sm:
            stats = asyncio.run(
                manager.sync_from_adapter(adapter=mock_adapter, sync_semantic_models=True, sync_metrics=False)
            )

        assert stats["semantic_models_synced"] == 0
        mock_store_sm.assert_not_called()
        mock_adapter.get_semantic_model.assert_called_once_with(table_name="orders_cube")

    def test_skip_none_semantic_model(self):
        manager = _make_manager()
        mock_adapter = MagicMock()
        mock_adapter.service_type = "test_service"
        mock_adapter.list_semantic_models.return_value = ["empty_model"]
        mock_adapter.get_semantic_model.return_value = None  # None model
        mock_adapter.list_metrics = AsyncMock(return_value=[])

        with patch.object(manager, "store_semantic_model") as mock_store_sm:
            stats = asyncio.run(
                manager.sync_from_adapter(
                    adapter=mock_adapter,
                    sync_semantic_models=True,
                    sync_metrics=False,
                )
            )

        mock_store_sm.assert_not_called()
        assert stats["semantic_models_synced"] == 0
