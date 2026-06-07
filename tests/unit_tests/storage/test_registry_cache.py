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
    """Tests for per-project LRU caching in get_storage()."""

    def test_same_project_returns_cached(self, reset_global_singletons):
        """get_storage with same factory+project returns the same instance."""
        from datus.storage.registry import get_storage

        def _factory(embedding_model, **kwargs):
            return BaseEmbeddingStore(table_name="test", embedding_model=embedding_model, **kwargs)

        with patch("datus.storage.registry.get_embedding_model", return_value=_FakeEmbeddingModel()):
            s1 = get_storage(_factory, "database", project="proj_1")
            s2 = get_storage(_factory, "database", project="proj_1")
            assert s1 is s2

    def test_different_project_returns_different(self, reset_global_singletons):
        """get_storage with different projects returns distinct instances."""
        from datus.storage.registry import get_storage

        def _factory(embedding_model, **kwargs):
            return BaseEmbeddingStore(table_name="test", embedding_model=embedding_model, **kwargs)

        with (
            patch("datus.storage.registry.get_embedding_model", return_value=_FakeEmbeddingModel()),
            patch("datus.storage.backend_holder.get_vector_backend") as mock_backend,
        ):
            mock_backend.return_value = MagicMock()
            s1 = get_storage(_factory, "database", project="proj_a")
            s2 = get_storage(_factory, "database", project="proj_b")
            assert s1 is not s2

    def test_empty_project_raises(self, reset_global_singletons):
        """get_storage with empty project propagates the backend DatusException."""
        import pytest

        from datus.storage.registry import get_storage
        from datus.utils.exceptions import DatusException

        def _factory(embedding_model, **kwargs):
            return BaseEmbeddingStore(table_name="test", embedding_model=embedding_model, **kwargs)

        with patch("datus.storage.registry.get_embedding_model", return_value=_FakeEmbeddingModel()):
            with pytest.raises(DatusException):
                get_storage(_factory, "database", project="")

    def test_clear_registry_clears_cache(self, reset_global_singletons):
        """clear_storage_registry() clears the LRU cache."""
        from datus.storage.registry import _get_storage_cached, clear_storage_registry, get_storage

        def _factory(embedding_model, **kwargs):
            return BaseEmbeddingStore(table_name="test", embedding_model=embedding_model, **kwargs)

        with (
            patch("datus.storage.registry.get_embedding_model", return_value=_FakeEmbeddingModel()),
            patch("datus.storage.backend_holder.get_vector_backend") as mock_backend,
        ):
            mock_backend.return_value = MagicMock()
            get_storage(_factory, "database", project="my_project")
            assert _get_storage_cached.cache_info().currsize >= 1

            clear_storage_registry()
            assert _get_storage_cached.cache_info().currsize == 0

    def test_subject_store_binds_subject_tree_by_requested_project(self, reset_global_singletons):
        """Subject stores created via get_storage() bind the requested project's subject tree."""
        from datus.storage.metric.store import MetricStorage
        from datus.storage.registry import get_storage

        project_tree = MagicMock(name="project_tree")

        with (
            patch("datus.storage.registry.get_embedding_model", return_value=_FakeEmbeddingModel()),
            patch("datus.storage.registry._get_subject_tree_cached", return_value=project_tree) as mock_tree,
            patch("datus.storage.backend_holder.get_vector_backend") as mock_backend,
        ):
            mock_backend.return_value = MagicMock()
            store = get_storage(MetricStorage, "metric", project="my_project")

        assert store.subject_tree is project_tree
        assert call("my_project") in mock_tree.call_args_list


class TestPreloadAllStorages:
    """Tests for preload_all_storages() with project."""

    def test_preload_forwards_project(self, reset_global_singletons):
        """preload_all_storages forwards project to both init_backends and get_storage."""
        from datus.storage.registry import preload_all_storages

        with (
            patch("datus.storage.registry.get_storage") as mock_get_storage,
            patch("datus.storage.backend_holder.init_backends") as mock_init,
            patch("datus.storage.registry.get_subject_tree_store"),
        ):
            preload_all_storages(data_dir="/tmp/test", project="my_project")
            mock_init.assert_called_once_with(config=None, data_dir="/tmp/test")
            # All get_storage calls receive project as a kwarg.
            for call_args in mock_get_storage.call_args_list:
                assert call_args.kwargs.get("project") == "my_project"

    def test_preload_applies_defaults(self, reset_global_singletons):
        """preload_all_storages applies deployment defaults."""
        from datus.storage.registry import get_storage_defaults, preload_all_storages

        with (
            patch("datus.storage.registry.get_storage"),
            patch("datus.storage.backend_holder.init_backends"),
            patch("datus.storage.registry.get_subject_tree_store"),
        ):
            preload_all_storages("my_project", data_dir="/tmp/test", table_prefix="tb_")
            defaults = get_storage_defaults()
            assert defaults["table_prefix"] == "tb_"


class TestBackendHolderConfigPropagation:
    """Tests for config propagation in backend_holder.

    Backends are stateless w.r.t. project: ``initialize()`` only carries
    backend-wide settings (``data_dir``, ``isolation``). The project
    identifier is passed to ``connect()`` via ``create_*`` helpers, not
    injected into backend config.
    """

    def test_vector_backend_receives_isolation_but_not_project(self, reset_global_singletons):
        """get_vector_backend() passes isolation to vector config; project is NOT injected."""
        from datus.storage.backend_holder import get_vector_backend, init_backends

        with patch("datus.storage.vector.VectorRegistry.create_backend") as mock_create:
            mock_create.return_value = MagicMock()
            init_backends(data_dir="/tmp/test")
            get_vector_backend()
            call_config = mock_create.call_args[0][1]
            assert "isolation" in call_config
            assert "project" not in call_config

    def test_rdb_backend_receives_isolation_but_not_project(self, reset_global_singletons):
        """_get_rdb_backend() passes isolation to rdb config; project is NOT injected."""
        from datus.storage.backend_holder import _get_rdb_backend, init_backends

        with patch("datus.storage.rdb.RdbRegistry.create_backend") as mock_create:
            mock_create.return_value = MagicMock()
            init_backends(data_dir="/tmp/test")
            _get_rdb_backend()
            call_config = mock_create.call_args[0][1]
            assert "isolation" in call_config
            assert "project" not in call_config

    def test_create_vector_connection_forwards_project(self, reset_global_singletons):
        """create_vector_connection(project=...) forwards project to backend.connect()."""
        from datus.storage.backend_holder import create_vector_connection, init_backends

        with patch("datus.storage.vector.VectorRegistry.create_backend") as mock_create:
            mock_backend = MagicMock()
            mock_create.return_value = mock_backend
            init_backends(data_dir="/tmp/test")
            create_vector_connection("my_project")
            mock_backend.connect.assert_called_once_with("my_project")

    def test_create_rdb_for_store_forwards_project(self, reset_global_singletons):
        """create_rdb_for_store(store, project) forwards both to backend.connect()."""
        from datus.storage.backend_holder import create_rdb_for_store, init_backends

        with patch("datus.storage.rdb.RdbRegistry.create_backend") as mock_create:
            mock_backend = MagicMock()
            mock_create.return_value = mock_backend
            init_backends(data_dir="/tmp/test")
            create_rdb_for_store("task", "my_project")
            mock_backend.connect.assert_called_once_with("my_project", "task")
