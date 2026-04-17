# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for storage registry LRU cache, preload, and backend_holder isolation config."""

from unittest.mock import MagicMock, call, patch

from datus.storage.base import BaseEmbeddingStore


class _FakeEmbeddingModel:
    dim_size = 384
    batch_size = 32
    model_name = "fake"
    is_model_failed = False
    model_error_message = ""
    device = None

    @property
    def model(self):
        return MagicMock()


class TestGetStorageLRUCache:
    """Tests for per-namespace LRU caching in get_storage()."""

    def test_same_namespace_returns_cached(self, reset_global_singletons):
        """get_storage with same factory+namespace returns the same instance."""
        from datus.storage.registry import get_storage

        def _factory(embedding_model, **kwargs):
            return BaseEmbeddingStore(table_name="test", embedding_model=embedding_model, **kwargs)

        with patch("datus.storage.registry.get_embedding_model", return_value=_FakeEmbeddingModel()):
            s1 = get_storage(_factory, "database", namespace="ns1")
            s2 = get_storage(_factory, "database", namespace="ns1")
            assert s1 is s2

    def test_different_namespace_returns_different(self, reset_global_singletons):
        """get_storage with different namespaces returns distinct instances."""
        from datus.storage.registry import get_storage

        def _factory(embedding_model, **kwargs):
            return BaseEmbeddingStore(table_name="test", embedding_model=embedding_model, **kwargs)

        with (
            patch("datus.storage.registry.get_embedding_model", return_value=_FakeEmbeddingModel()),
            patch("datus.storage.backend_holder.get_vector_backend") as mock_backend,
        ):
            mock_backend.return_value = MagicMock()
            s1 = get_storage(_factory, "database", namespace="ns_a")
            s2 = get_storage(_factory, "database", namespace="ns_b")
            assert s1 is not s2

    def test_empty_namespace_does_not_pass_db_kwarg(self, reset_global_singletons):
        """get_storage with empty namespace does not pass a 'db' kwarg to factory."""
        from datus.storage.registry import get_storage

        received_kwargs = {}

        def _factory(embedding_model, **kwargs):
            received_kwargs.update(kwargs)
            return BaseEmbeddingStore(table_name="test", embedding_model=embedding_model, **kwargs)

        with patch("datus.storage.registry.get_embedding_model", return_value=_FakeEmbeddingModel()):
            get_storage(_factory, "database", namespace="")
            assert "db" not in received_kwargs

    def test_clear_registry_clears_cache(self, reset_global_singletons):
        """clear_storage_registry() clears the LRU cache."""
        from datus.storage.registry import _get_storage_cached, clear_storage_registry, get_storage

        def _factory(embedding_model, **kwargs):
            return BaseEmbeddingStore(table_name="test", embedding_model=embedding_model, **kwargs)

        with patch("datus.storage.registry.get_embedding_model", return_value=_FakeEmbeddingModel()):
            get_storage(_factory, "database")
            assert _get_storage_cached.cache_info().currsize >= 1

            clear_storage_registry()
            assert _get_storage_cached.cache_info().currsize == 0

    def test_subject_store_binds_subject_tree_by_current_project(self, reset_global_singletons):
        """Subject stores created via get_storage() bind the active project's subject tree.

        Subject-tree isolation moved from datasource to project; the requested
        ``namespace`` (datasource_id) no longer drives subject-tree lookup.
        """
        from datus.storage.metric.store import MetricStorage
        from datus.storage.registry import get_storage

        project_tree = MagicMock(name="project_tree")

        with (
            patch("datus.storage.registry.get_embedding_model", return_value=_FakeEmbeddingModel()),
            patch("datus.storage.registry._get_subject_tree_cached", return_value=project_tree) as mock_tree,
            patch("datus.storage.backend_holder.get_current_project", return_value="my_project"),
            patch("datus.storage.backend_holder.get_vector_backend") as mock_backend,
        ):
            mock_backend.return_value = MagicMock()
            store = get_storage(MetricStorage, "metric", namespace="requested_ns")

        assert store.subject_tree is project_tree
        assert call("my_project", "my_project") in mock_tree.call_args_list


class TestPreloadAllStorages:
    """Tests for preload_all_storages() with project/datasource_id."""

    def test_preload_forwards_project_and_datasource_id(self, reset_global_singletons):
        """preload_all_storages forwards project to init_backends and datasource_id to get_storage."""
        from datus.storage.registry import preload_all_storages

        with (
            patch("datus.storage.registry.get_storage") as mock_get_storage,
            patch("datus.storage.backend_holder.init_backends") as mock_init,
            patch("datus.storage.registry.get_subject_tree_store"),
        ):
            preload_all_storages(data_dir="/tmp/test", project="my_project", datasource_id="ds_001")
            mock_init.assert_called_once_with(config=None, data_dir="/tmp/test", project="my_project")
            # All get_storage calls receive datasource_id as the ``namespace`` kwarg.
            for call in mock_get_storage.call_args_list:
                assert call.kwargs.get("namespace") == "ds_001"

    def test_preload_applies_defaults(self, reset_global_singletons):
        """preload_all_storages applies deployment defaults."""
        from datus.storage.registry import get_storage_defaults, preload_all_storages

        with (
            patch("datus.storage.registry.get_storage"),
            patch("datus.storage.backend_holder.init_backends"),
            patch("datus.storage.registry.get_subject_tree_store"),
        ):
            preload_all_storages(data_dir="/tmp/test", table_prefix="tb_")
            defaults = get_storage_defaults()
            assert defaults["table_prefix"] == "tb_"


class TestBackendHolderConfigPropagation:
    """Tests for config propagation in backend_holder.

    Project path isolation is backend-owned, so ``project`` is what flows to
    backends; ``isolation`` only matters for LOGICAL row-scoping in the vector
    backend.
    """

    def test_vector_backend_receives_isolation_and_project(self, reset_global_singletons):
        """get_vector_backend() passes both isolation and project to vector config."""
        from datus.storage.backend_holder import get_vector_backend, init_backends

        with patch("datus.storage.vector.VectorRegistry.create_backend") as mock_create:
            mock_create.return_value = MagicMock()
            init_backends(data_dir="/tmp/test", project="proj")
            get_vector_backend()
            call_config = mock_create.call_args[0][1]
            assert "isolation" in call_config
            assert call_config.get("project") == "proj"

    def test_rdb_backend_receives_project(self, reset_global_singletons):
        """_get_rdb_backend() passes project to rdb config; isolation is not needed."""
        from datus.storage.backend_holder import _get_rdb_backend, init_backends

        with patch("datus.storage.rdb.RdbRegistry.create_backend") as mock_create:
            mock_create.return_value = MagicMock()
            init_backends(data_dir="/tmp/test", project="proj")
            _get_rdb_backend()
            call_config = mock_create.call_args[0][1]
            assert call_config.get("project") == "proj"

    def test_create_vector_connection_default_has_empty_datasource_id(self, reset_global_singletons):
        """create_vector_connection() defaults datasource_id to '' (no LOGICAL filter)."""
        from datus.storage.backend_holder import create_vector_connection, init_backends

        with patch("datus.storage.vector.VectorRegistry.create_backend") as mock_create:
            mock_backend = MagicMock()
            mock_create.return_value = mock_backend
            init_backends(data_dir="/tmp/test", project="proj")
            create_vector_connection()
            mock_backend.connect.assert_called_once_with(namespace="")

    def test_create_vector_connection_explicit_datasource_id(self, reset_global_singletons):
        """create_vector_connection(datasource_id=...) forwards it as the vector-level namespace."""
        from datus.storage.backend_holder import create_vector_connection, init_backends

        with patch("datus.storage.vector.VectorRegistry.create_backend") as mock_create:
            mock_backend = MagicMock()
            mock_create.return_value = mock_backend
            init_backends(data_dir="/tmp/test", project="proj")
            create_vector_connection(datasource_id="ds_1")
            mock_backend.connect.assert_called_once_with(namespace="ds_1")
