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
            a = get_storage(_DummyStore, "metric", project="test")
            b = get_storage(_DummyStore, "metric", project="test")
        assert a is b

    def test_clear_registry_invalidates_cache(self):
        """After clear_storage_registry, get_storage returns a new instance."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            a = get_storage(_DummyStore, "metric", project="test")
            clear_storage_registry()
            b = get_storage(_DummyStore, "metric", project="test")
        assert a is not b

    def test_different_datasources_get_distinct_wrappers(self):
        """Datasource participates in the wrapper cache key."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            a = get_storage(_DummyStore, "metric", project="test", datasource_id="ds_a")
            b = get_storage(_DummyStore, "metric", project="test", datasource_id="ds_b")
        assert a is not b


class TestConfigureStorageDefaults:
    """Tests for configure_storage_defaults."""

    def test_defaults_forwarded_to_factory(self):
        """Global defaults should arrive as kwargs in the factory call."""
        configure_storage_defaults(table_prefix="tb_")
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric", project="test")
        assert store.init_kwargs.get("table_prefix") == "tb_"
        assert "db" in store.init_kwargs  # backend connection is always injected now

    def test_no_defaults_gives_empty_kwargs(self):
        """Without configure_storage_defaults, factory gets only the injected backend db."""
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric", project="test")
        assert set(store.init_kwargs) == {"db"}

    def test_reconfigure_overwrites_previous(self):
        """Calling configure_storage_defaults again replaces old values."""
        configure_storage_defaults(table_prefix="old_")
        configure_storage_defaults(table_prefix="new_")
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric", project="test")
        assert store.init_kwargs.get("table_prefix") == "new_"

    def test_clear_registry_preserves_defaults(self):
        """clear_storage_registry should NOT wipe defaults."""
        configure_storage_defaults(table_prefix="tb_")
        clear_storage_registry()
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric", project="test")
        assert store.init_kwargs.get("table_prefix") == "tb_"


class TestRequestScopedTablePrefix:
    """``table_prefix_scope`` — one process serving two table sets.

    Datus-backend can host the Studio surface (``tb_*``) and the publication
    surface (``pub_tb_*``) together, and which one a request reads is a property
    of the request. These tests pin the two things that made that impossible
    before: the global-only prefix, and a store cache that did not key on it.
    """

    def test_scope_overrides_the_deployment_default(self):
        from datus.storage.registry import get_storage_defaults, table_prefix_scope

        configure_storage_defaults(table_prefix="tb_")
        assert get_storage_defaults()["table_prefix"] == "tb_"
        with table_prefix_scope("pub_tb_"):
            assert get_storage_defaults()["table_prefix"] == "pub_tb_"
        # ...and is restored, so a wrapped request cannot leak into the next one.
        assert get_storage_defaults()["table_prefix"] == "tb_"

    def test_scope_nests_and_restores(self):
        from datus.storage.registry import get_storage_defaults, table_prefix_scope

        configure_storage_defaults(table_prefix="tb_")
        with table_prefix_scope("pub_tb_"):
            with table_prefix_scope("other_"):
                assert get_storage_defaults()["table_prefix"] == "other_"
            assert get_storage_defaults()["table_prefix"] == "pub_tb_"
        assert get_storage_defaults()["table_prefix"] == "tb_"

    def test_the_store_built_inside_a_scope_carries_that_prefix(self):
        from datus.storage.registry import table_prefix_scope

        configure_storage_defaults(table_prefix="tb_")
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            studio = get_storage(_DummyStore, "metric", project="p1")
            with table_prefix_scope("pub_tb_"):
                published = get_storage(_DummyStore, "metric", project="p1")

        assert studio.init_kwargs["table_prefix"] == "tb_"
        assert published.init_kwargs["table_prefix"] == "pub_tb_"

    def test_two_table_sets_are_two_cache_entries_for_the_same_project(self):
        """THE regression this exists for.

        Same project, same factory, same datasource — only the prefix differs. If
        the prefix is not in the key, the second call returns the first instance
        and the process serves the wrong table set until it restarts.
        """
        from datus.storage.registry import table_prefix_scope

        configure_storage_defaults(table_prefix="tb_")
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            studio = get_storage(_DummyStore, "metric", project="same", datasource_id="ds")
            with table_prefix_scope("pub_tb_"):
                published = get_storage(_DummyStore, "metric", project="same", datasource_id="ds")
            studio_again = get_storage(_DummyStore, "metric", project="same", datasource_id="ds")

        assert studio is not published
        # Still a cache, not a factory: the same prefix returns the same instance.
        assert studio is studio_again

    def test_a_scope_does_not_leak_into_a_concurrent_task(self):
        """ContextVar, not a thread-local or a global: asyncio tasks must not see it.

        Each task inherits a COPY of the context at creation, so the publication
        task's override is invisible to the Studio task running beside it.
        """
        import asyncio

        from datus.storage.registry import get_storage_defaults, table_prefix_scope

        configure_storage_defaults(table_prefix="tb_")
        seen: dict[str, str] = {}
        started = asyncio.Event()

        async def publication():
            with table_prefix_scope("pub_tb_"):
                started.set()
                # Yield with the override in force, so the other task runs inside
                # this window rather than before or after it.
                await asyncio.sleep(0)
                seen["publication"] = get_storage_defaults()["table_prefix"]

        async def studio():
            await started.wait()
            seen["studio"] = get_storage_defaults()["table_prefix"]

        async def main():
            await asyncio.gather(publication(), studio())

        asyncio.run(main())
        assert seen == {"publication": "pub_tb_", "studio": "tb_"}

    def test_no_scope_leaves_the_kwargs_shape_alone(self):
        """With no defaults and no scope, the factory sees only ``db`` — as before.

        An unconditional ``table_prefix=""`` would be handing every factory a
        keyword it never used to receive.
        """
        with patch("datus.storage.registry.get_embedding_model", side_effect=_fake_get_embedding_model):
            store = get_storage(_DummyStore, "metric", project="test")
        assert set(store.init_kwargs) == {"db"}
