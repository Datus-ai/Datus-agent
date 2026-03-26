# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.registry (singleton storage)."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.registry import (
    clear_storage_registry,
    configure_storage_defaults,
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
        self._default_values = {}
        from datus.storage.base import _SharedTableState

        self._shared = _SharedTableState()
        self._shared.initialized = True

    def _ensure_table_ready(self):
        pass


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

    def test_same_factory_returns_same_instance(self):
        """Same factory must return the identical instance (true singleton)."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            a = get_storage(_DummyStore, "metric")
            b = get_storage(_DummyStore, "metric")
        assert a is b

    def test_clear_registry_invalidates_cache(self):
        """After clear_storage_registry, get_storage returns a new instance."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            a = get_storage(_DummyStore, "metric")
            clear_storage_registry()
            b = get_storage(_DummyStore, "metric")
        assert a is not b

    def test_different_namespaces_not_in_key(self):
        """get_storage ignores namespace — same factory always returns same instance."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            a = get_storage(_DummyStore, "metric")
            b = get_storage(_DummyStore, "metric")
        assert a is b


class TestConfigureStorageDefaults:
    """Tests for configure_storage_defaults."""

    def test_defaults_forwarded_to_factory(self):
        """Global defaults should arrive as kwargs in the factory call."""
        configure_storage_defaults(table_prefix="tb_")
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric")
        assert store.init_kwargs == {"table_prefix": "tb_"}

    def test_no_defaults_gives_empty_kwargs(self):
        """Without configure_storage_defaults, factory gets no extra kwargs."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric")
        assert store.init_kwargs == {}

    def test_reconfigure_overwrites_previous(self):
        """Calling configure_storage_defaults again replaces old values."""
        configure_storage_defaults(table_prefix="old_")
        configure_storage_defaults(table_prefix="new_")
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric")
        assert store.init_kwargs == {"table_prefix": "new_"}

    def test_clear_registry_preserves_defaults(self):
        """clear_storage_registry should NOT wipe defaults."""
        configure_storage_defaults(table_prefix="tb_")
        clear_storage_registry()
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric")
        assert store.init_kwargs == {"table_prefix": "tb_"}
