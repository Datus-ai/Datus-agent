# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for storage registry LRU cache, preload, and backend_holder isolation config."""

from unittest.mock import MagicMock, patch

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


class TestPreloadAllStorages:
    """Tests for preload_all_storages() with namespace."""

    def test_preload_passes_namespace(self, reset_global_singletons):
        """preload_all_storages passes namespace to get_storage and init_backends."""
        from datus.storage.registry import preload_all_storages

        with (
            patch("datus.storage.registry.get_storage") as mock_get_storage,
            patch("datus.storage.backend_holder.init_backends") as mock_init,
            patch("datus.storage.registry.get_subject_tree_store"),
        ):
            preload_all_storages(data_dir="/tmp/test", namespace="my_ns")
            mock_init.assert_called_once_with(config=None, data_dir="/tmp/test", namespace="my_ns")
            # All get_storage calls should include namespace
            for call in mock_get_storage.call_args_list:
                assert call.kwargs.get("namespace") == "my_ns"

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


class TestBackendHolderIsolationConfig:
    """Tests for isolation config propagation in backend_holder."""

    def test_vector_backend_receives_isolation(self, reset_global_singletons):
        """get_vector_backend() passes isolation to vector config."""
        from datus.storage.backend_holder import get_vector_backend, init_backends

        with patch("datus.storage.vector.VectorRegistry.create_backend") as mock_create:
            mock_create.return_value = MagicMock()
            init_backends(data_dir="/tmp/test")
            get_vector_backend()
            call_config = mock_create.call_args[0][1]
            assert "isolation" in call_config

    def test_rdb_backend_receives_isolation(self, reset_global_singletons):
        """_get_rdb_backend() passes isolation to rdb config."""
        from datus.storage.backend_holder import _get_rdb_backend, init_backends

        with patch("datus.storage.rdb.RdbRegistry.create_backend") as mock_create:
            mock_create.return_value = MagicMock()
            init_backends(data_dir="/tmp/test")
            _get_rdb_backend()
            call_config = mock_create.call_args[0][1]
            assert "isolation" in call_config

    def test_create_vector_connection_uses_global_namespace(self, reset_global_singletons):
        """create_vector_connection() uses global namespace when none given."""
        from datus.storage.backend_holder import create_vector_connection, init_backends

        with patch("datus.storage.vector.VectorRegistry.create_backend") as mock_create:
            mock_backend = MagicMock()
            mock_create.return_value = mock_backend
            init_backends(data_dir="/tmp/test", namespace="global_ns")
            create_vector_connection()
            mock_backend.connect.assert_called_once_with(namespace="global_ns")

    def test_create_vector_connection_explicit_namespace(self, reset_global_singletons):
        """create_vector_connection() uses explicit namespace over global."""
        from datus.storage.backend_holder import create_vector_connection, init_backends

        with patch("datus.storage.vector.VectorRegistry.create_backend") as mock_create:
            mock_backend = MagicMock()
            mock_create.return_value = mock_backend
            init_backends(data_dir="/tmp/test", namespace="global_ns")
            create_vector_connection(namespace="explicit_ns")
            mock_backend.connect.assert_called_once_with(namespace="explicit_ns")
