# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.registry (singleton + scoped view)."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.registry import (
    clear_storage_registry,
    configure_storage_defaults,
    create_scoped_view,
    get_rag_storage,
    get_storage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEmbeddingModel:
    """Minimal stand-in for EmbeddingModel to avoid real model loading."""

    dim_size = 384
    batch_size = 32
    model_name = "fake"
    is_model_failed = False
    model_error_message = ""
    device = None

    @property
    def model(self):
        return MagicMock()


def _fake_get_embedding_model(_conf_name):
    return _FakeEmbeddingModel()


class _DummyStore:
    """Trivial 'storage' that records its init args."""

    def __init__(self, embedding_model, **kwargs):
        self.embedding_model = embedding_model
        self.init_kwargs = kwargs
        self._scope_filter = None
        self._default_values = {}
        from datus.storage.base import _SharedTableState

        self._shared = _SharedTableState()
        self._shared.initialized = True

    def _ensure_table_ready(self):
        pass

    def __copy__(self):

        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure a fresh registry and defaults for every test."""
    configure_storage_defaults()  # reset to empty
    clear_storage_registry()
    yield
    configure_storage_defaults()  # reset to empty
    clear_storage_registry()


class TestGetStorage:
    """Tests for get_storage singleton behaviour."""

    def test_same_factory_namespace_returns_same_instance(self):
        """Same (factory, namespace) key must return the identical instance."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            a = get_storage(_DummyStore, "metric", "ns1")
            b = get_storage(_DummyStore, "metric", "ns1")
        assert a is b

    def test_different_namespace_returns_different_instance(self):
        """Different namespaces produce different singletons."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            a = get_storage(_DummyStore, "metric", "ns1")
            b = get_storage(_DummyStore, "metric", "ns2")
        assert a is not b

    def test_clear_registry_invalidates_cache(self):
        """After clear_storage_registry, get_storage returns a new instance."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            a = get_storage(_DummyStore, "metric", "ns1")
            clear_storage_registry()
            b = get_storage(_DummyStore, "metric", "ns1")
        assert a is not b


class TestCreateScopedView:
    """Tests for create_scoped_view."""

    def test_none_filter_returns_original(self):
        """No filter → return the singleton directly (no copy)."""
        store = _DummyStore(_FakeEmbeddingModel())
        view = create_scoped_view(store, None)
        assert view is store

    def test_scoped_view_is_different_object(self):
        """With a filter the returned view must be a distinct object."""
        store = _DummyStore(_FakeEmbeddingModel())
        from datus_storage_base.conditions import eq

        view = create_scoped_view(store, eq("table_name", "orders"))
        assert view is not store

    def test_scoped_view_has_filter(self):
        """The view's _scope_filter must be the one we passed in."""
        store = _DummyStore(_FakeEmbeddingModel())
        from datus_storage_base.conditions import eq

        filt = eq("table_name", "orders")
        view = create_scoped_view(store, filt)
        assert view._scope_filter is filt

    def test_scoped_view_does_not_affect_singleton(self):
        """Setting a filter on the view must not mutate the singleton."""
        store = _DummyStore(_FakeEmbeddingModel())
        from datus_storage_base.conditions import eq

        create_scoped_view(store, eq("table_name", "orders"))
        assert store._scope_filter is None

    def test_scoped_view_shares_internal_state(self):
        """The view must share the same embedding_model reference as the singleton."""
        store = _DummyStore(_FakeEmbeddingModel())
        from datus_storage_base.conditions import eq

        view = create_scoped_view(store, eq("table_name", "orders"))
        assert view.embedding_model is store.embedding_model


class TestConfigureStorageDefaults:
    """Tests for configure_storage_defaults."""

    def test_defaults_forwarded_to_factory(self):
        """Global defaults should arrive as kwargs in the factory call."""
        configure_storage_defaults(table_prefix="tb_")
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric", "ns1")
        assert store.init_kwargs == {"table_prefix": "tb_"}

    def test_no_defaults_gives_empty_kwargs(self):
        """Without configure_storage_defaults, factory gets no extra kwargs."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric", "ns1")
        assert store.init_kwargs == {}

    def test_reconfigure_overwrites_previous(self):
        """Calling configure_storage_defaults again replaces old values."""
        configure_storage_defaults(table_prefix="old_")
        configure_storage_defaults(table_prefix="new_")
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric", "ns1")
        assert store.init_kwargs == {"table_prefix": "new_"}

    def test_clear_registry_preserves_defaults(self):
        """clear_storage_registry should NOT wipe defaults."""
        configure_storage_defaults(table_prefix="tb_")
        clear_storage_registry()
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric", "ns1")
        assert store.init_kwargs == {"table_prefix": "tb_"}


class TestScopedViewWriteDefaults:
    """Tests for create_scoped_view write_defaults parameter."""

    def test_write_defaults_set_on_view(self):
        """write_defaults should be merged into the view's _default_values."""
        store = _DummyStore(_FakeEmbeddingModel())
        store._default_values = {}
        view = create_scoped_view(store, None, write_defaults={"workspace_id": "ws_1"})
        assert view is not store
        assert view._default_values == {"workspace_id": "ws_1"}

    def test_write_defaults_do_not_mutate_singleton(self):
        """Setting write_defaults on view must not affect the singleton."""
        store = _DummyStore(_FakeEmbeddingModel())
        store._default_values = {"existing": "val"}
        create_scoped_view(store, None, write_defaults={"workspace_id": "ws_1"})
        assert store._default_values == {"existing": "val"}

    def test_write_defaults_merged_with_existing(self):
        """write_defaults should merge with deployment-level defaults."""
        store = _DummyStore(_FakeEmbeddingModel())
        store._default_values = {"tenant": "t1"}
        view = create_scoped_view(store, None, write_defaults={"workspace_id": "ws_1"})
        assert view._default_values == {"tenant": "t1", "workspace_id": "ws_1"}

    def test_no_write_defaults_no_filter_returns_original(self):
        """No filter + no write_defaults → return singleton directly."""
        store = _DummyStore(_FakeEmbeddingModel())
        view = create_scoped_view(store, None, write_defaults=None)
        assert view is store


class TestGetRagStorage:
    """Tests for the get_rag_storage one-stop helper."""

    def _mock_agent_config(self, namespace="ns1", request_context=None):
        config = MagicMock()
        config.current_namespace = namespace
        config.request_context = request_context
        config.db_type = ""
        config.sub_agent_config = MagicMock(return_value={})
        return config

    def test_returns_singleton_for_cli_mode(self):
        """CLI mode (no request_context, no sub-agent) → returns singleton."""
        config = self._mock_agent_config()
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            a = get_rag_storage(_DummyStore, "metric", config, None, "metrics")
            b = get_rag_storage(_DummyStore, "metric", config, None, "metrics")
        assert a is b  # no scoping → singleton returned directly

    def test_request_context_creates_scoped_view(self):
        """SaaS mode (request_context + scope_fields) → scoped view with read filter + write defaults."""
        configure_storage_defaults(scope_fields=["workspace_id"])
        config = self._mock_agent_config(
            request_context={"workspace_id": "ws_abc", "creator_id": "user_1", "updator_id": "user_1"},
        )
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            singleton = get_storage(_DummyStore, "metric", "ns1")
            view = get_rag_storage(_DummyStore, "metric", config, None, "metrics")
        assert view is not singleton
        # Write defaults include ALL request_context fields
        assert view._default_values == {"workspace_id": "ws_abc", "creator_id": "user_1", "updator_id": "user_1"}
        # Read filter only has scope_fields (workspace_id), NOT creator_id/updator_id
        from datus_storage_base.conditions import build_where

        clause = build_where(view._scope_filter)
        assert "workspace_id" in clause
        assert "creator_id" not in clause
