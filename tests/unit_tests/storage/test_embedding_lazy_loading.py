# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from unittest.mock import MagicMock

import pandas as pd

from datus.storage.base import BaseEmbeddingStore
from datus.storage.embedding_models import EmbeddingModel
from datus.utils.exceptions import DatusException


class _FakeEmbeddingFunction:
    def generate_embeddings(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def ndims(self):
        return 2


class _FakeTable:
    def __init__(self):
        self.rows = []

    def add(self, frame: pd.DataFrame):
        self.rows.extend(frame.to_dict("records"))

    def count_rows(self, where=None):
        return len(self.rows)


class _FakeVectorDb:
    def __init__(self):
        self.table = _FakeTable()
        self.created_with = {}

    def table_exists(self, table_name):
        return False

    def create_table(self, table_name, **kwargs):
        self.created_with = {"table_name": table_name, **kwargs}
        return self.table


def test_model_property_loads_lazily(monkeypatch):
    sentinel_model = _FakeEmbeddingFunction()
    init_calls = []

    def fake_init_model(self):
        init_calls.append(self.model_name)
        self._model = sentinel_model

    monkeypatch.setattr(EmbeddingModel, "init_model", fake_init_model)
    model = EmbeddingModel(model_name="unit-test-model", dim_size=2)

    assert model.model_initialization_attempted is False
    assert model.is_model_available() is True

    assert model.model is sentinel_model

    assert init_calls == ["unit-test-model"]
    assert model.model_initialization_attempted is True
    assert model.is_model_failed is False
    assert model.is_model_available() is True


def test_silent_initialization_updates_success_state(monkeypatch):
    sentinel_model = _FakeEmbeddingFunction()

    def fake_init_model(self):
        self._model = sentinel_model

    monkeypatch.setattr(EmbeddingModel, "init_model", fake_init_model)
    model = EmbeddingModel(model_name="unit-test-model", dim_size=2)

    assert model.try_init_model_silent() is True

    assert model._model is sentinel_model
    assert model.model_initialization_attempted is True
    assert model.is_model_failed is False


def test_silent_initialization_records_failure(monkeypatch):
    def fail_init_model(self):
        raise RuntimeError("download unavailable")

    monkeypatch.setattr(EmbeddingModel, "init_model", fail_init_model)
    model = EmbeddingModel(model_name="missing-model", dim_size=2)

    assert model.try_init_model_silent() is False

    assert model._model is None
    assert model.model_initialization_attempted is True
    assert model.is_model_failed is True
    assert model.model_error_message == "download unavailable"


def test_storage_defers_failed_model_error_until_first_use():
    failed_model = EmbeddingModel(model_name="missing-model", dim_size=2)
    failed_model.is_model_failed = True
    failed_model.model_error_message = "download unavailable"
    db = MagicMock()

    storage = BaseEmbeddingStore(table_name="test_table", embedding_model=failed_model, db=db)

    assert storage._shared.initialized is False
    db.create_table.assert_not_called()

    try:
        storage.table_size()
    except DatusException as exc:
        assert "missing-model" in str(exc)
        assert "download unavailable" in str(exc)
    else:
        raise AssertionError("table_size should fail when the embedding model is unavailable")

    assert storage._shared.initialized is False
    db.create_table.assert_not_called()


def test_store_initializes_table_with_loaded_model():
    model = EmbeddingModel(model_name="unit-test-model", dim_size=2)
    model._model = _FakeEmbeddingFunction()
    db = _FakeVectorDb()
    storage = BaseEmbeddingStore(table_name="test_table", embedding_model=model, db=db)

    storage.store([{"definition": "test data"}])

    assert storage.table_size() == 1
    assert storage._shared.initialized is True
    assert db.created_with["table_name"] == "test_table"
    assert db.created_with["embedding_function"] is model._model
