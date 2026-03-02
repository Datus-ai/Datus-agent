# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/embedding_models.py — EmbeddingModel and get_embedding_model."""

import pytest

from datus.storage.embedding_models import DEFAULT_MODEL_CONFIG, EMBEDDING_MODELS, EmbeddingModel, get_embedding_model
from datus.utils.constants import EmbeddingProvider
from datus.utils.exceptions import DatusException

# ---------------------------------------------------------------------------
# EmbeddingModel.to_dict round-trip
# ---------------------------------------------------------------------------


class TestEmbeddingModelToDict:
    """Tests for EmbeddingModel.to_dict serialization."""

    def test_to_dict_contains_required_keys(self):
        """to_dict output must contain registry_name, model_name, and dim_size."""
        model = EmbeddingModel(model_name="all-MiniLM-L6-v2", dim_size=384)
        d = model.to_dict()
        assert set(d.keys()) == {"registry_name", "model_name", "dim_size"}

    def test_to_dict_round_trip_values(self):
        """Values in to_dict must match the constructor arguments."""
        model = EmbeddingModel(
            model_name="test-model",
            dim_size=768,
            registry_name=EmbeddingProvider.OPENAI,
        )
        d = model.to_dict()
        assert d["model_name"] == "test-model"
        assert d["dim_size"] == 768
        assert d["registry_name"] == EmbeddingProvider.OPENAI

    def test_to_dict_default_registry(self):
        """Default registry should be sentence-transformers."""
        model = EmbeddingModel(model_name="some-model", dim_size=256)
        d = model.to_dict()
        assert d["registry_name"] == EmbeddingProvider.SENTENCE_TRANSFORMERS

    def test_to_dict_round_trip_reconstruct(self):
        """A new EmbeddingModel created from to_dict output should match the original."""
        original = EmbeddingModel(
            model_name="my-model",
            dim_size=512,
            registry_name=EmbeddingProvider.FASTEMBED,
        )
        d = original.to_dict()
        reconstructed = EmbeddingModel(
            model_name=d["model_name"],
            dim_size=d["dim_size"],
            registry_name=d["registry_name"],
        )
        assert reconstructed.to_dict() == d


# ---------------------------------------------------------------------------
# EmbeddingModel.is_model_available state transitions
# ---------------------------------------------------------------------------


class TestIsModelAvailable:
    """Tests for is_model_available reflecting internal state."""

    def test_available_initially(self):
        """Brand new model should report available (not attempted, not failed)."""
        model = EmbeddingModel(model_name="test-model", dim_size=384)
        assert model.is_model_available() is True

    def test_not_available_after_failure(self):
        """After marking model as failed, is_model_available should return False."""
        model = EmbeddingModel(model_name="test-model", dim_size=384)
        model.is_model_failed = True
        assert model.is_model_available() is False

    def test_available_with_loaded_model(self):
        """If _model is set (simulating successful load), should still be available."""
        model = EmbeddingModel(model_name="test-model", dim_size=384)
        model._model = object()  # Simulate a loaded model
        model.model_initialization_attempted = True
        assert model.is_model_available() is True

    def test_not_available_after_attempted_without_model(self):
        """If initialization was attempted but _model is still None (and not marked failed), unavailable."""
        model = EmbeddingModel(model_name="test-model", dim_size=384)
        model.model_initialization_attempted = True
        # _model is still None, not marked as failed either
        # is_model_available: not failed AND (_model is not None OR not attempted)
        # = True AND (False OR False) = False
        assert model.is_model_available() is False


# ---------------------------------------------------------------------------
# Error message preservation
# ---------------------------------------------------------------------------


class TestErrorMessagePreservation:
    """Tests for error state and message retention."""

    def test_error_message_preserved_on_failure(self):
        """model_error_message should retain the error string after failure."""
        model = EmbeddingModel(model_name="bad-model", dim_size=384)
        model.is_model_failed = True
        model.model_error_message = "Connection refused"
        assert model.model_error_message == "Connection refused"

    def test_model_property_raises_with_error_message(self):
        """Accessing .model on a failed model should raise DatusException containing the error message."""
        model = EmbeddingModel(model_name="bad-model", dim_size=384)
        model.is_model_failed = True
        model.model_error_message = "CUDA out of memory"

        with pytest.raises(DatusException) as exc_info:
            _ = model.model
        assert "CUDA out of memory" in str(exc_info.value)
        assert "bad-model" in str(exc_info.value)

    def test_model_property_raises_with_error_code_message(self):
        """Error messages containing 'error_code=' should still include the core message."""
        model = EmbeddingModel(model_name="bad-model", dim_size=384)
        model.is_model_failed = True
        model.model_error_message = "error_code=300019 error_message=Download timeout"

        with pytest.raises(DatusException) as exc_info:
            _ = model.model
        assert "bad-model" in str(exc_info.value)
        # The core message should be extracted
        assert "Download timeout" in str(exc_info.value)

    def test_error_message_empty_string_initially(self):
        """Initial model_error_message should be empty."""
        model = EmbeddingModel(model_name="test-model", dim_size=384)
        assert model.model_error_message == ""

    def test_is_model_failed_initially_false(self):
        """Initial is_model_failed should be False."""
        model = EmbeddingModel(model_name="test-model", dim_size=384)
        assert model.is_model_failed is False


# ---------------------------------------------------------------------------
# get_embedding_model with fallback to default
# ---------------------------------------------------------------------------


class TestGetEmbeddingModel:
    """Tests for get_embedding_model fallback logic."""

    @pytest.fixture(autouse=True)
    def _clean_embedding_models(self):
        """Clear the global EMBEDDING_MODELS dict before and after each test."""
        saved = dict(EMBEDDING_MODELS)
        EMBEDDING_MODELS.clear()
        yield
        EMBEDDING_MODELS.clear()
        EMBEDDING_MODELS.update(saved)

    def test_get_existing_model(self):
        """Requesting a known store name should return the registered model."""
        model = EmbeddingModel(model_name="custom-model", dim_size=512)
        EMBEDDING_MODELS["my_store"] = model

        result = get_embedding_model("my_store")
        assert result is model
        assert result.model_name == "custom-model"

    def test_get_unknown_store_fallback_to_default_name_match(self):
        """Unknown store with a default-named model available should reuse it."""
        default_name = DEFAULT_MODEL_CONFIG["model_name"]
        default_model = EmbeddingModel(model_name=default_name, dim_size=384)
        EMBEDDING_MODELS["database"] = default_model

        result = get_embedding_model("unknown_store")
        assert result is default_model
        # Should also be cached now
        assert EMBEDDING_MODELS["unknown_store"] is default_model

    def test_get_unknown_store_creates_new_default(self):
        """Unknown store with no default model should create a new default model."""
        result = get_embedding_model("brand_new_store")
        assert result.model_name == DEFAULT_MODEL_CONFIG["model_name"]
        assert result._dim_size == DEFAULT_MODEL_CONFIG["dim_size"]
        # Should be cached
        assert "brand_new_store" in EMBEDDING_MODELS

    def test_get_embedding_model_caching(self):
        """Second call with same store_name should return the cached model."""
        result1 = get_embedding_model("store_a")
        result2 = get_embedding_model("store_a")
        assert result1 is result2


# ---------------------------------------------------------------------------
# Device setting
# ---------------------------------------------------------------------------


class TestDeviceSetting:
    """Tests for device configuration on EmbeddingModel."""

    def test_default_device_is_global(self):
        """Device should be set from the global EMBEDDING_DEVICE_TYPE."""
        import datus.storage.embedding_models as em_mod

        saved = em_mod.EMBEDDING_DEVICE_TYPE
        try:
            em_mod.EMBEDDING_DEVICE_TYPE = "test_device"
            model = EmbeddingModel(model_name="test", dim_size=384)
            assert model.device == "test_device"
        finally:
            em_mod.EMBEDDING_DEVICE_TYPE = saved

    def test_device_empty_string_default(self):
        """When EMBEDDING_DEVICE_TYPE is empty, device should be empty string."""
        import datus.storage.embedding_models as em_mod

        saved = em_mod.EMBEDDING_DEVICE_TYPE
        try:
            em_mod.EMBEDDING_DEVICE_TYPE = ""
            model = EmbeddingModel(model_name="test", dim_size=384)
            assert model.device == ""
        finally:
            em_mod.EMBEDDING_DEVICE_TYPE = saved


# ---------------------------------------------------------------------------
# EmbeddingModel.dim_size property
# ---------------------------------------------------------------------------


class TestDimSize:
    """Tests for the dim_size property."""

    def test_dim_size_returns_constructor_value(self):
        """dim_size should return the value provided at construction."""
        model = EmbeddingModel(model_name="test", dim_size=768)
        assert model.dim_size == 768

    def test_dim_size_384_default_model(self):
        """The default model (all-MiniLM-L6-v2) has dim_size 384."""
        model = EmbeddingModel(model_name="all-MiniLM-L6-v2", dim_size=384)
        assert model.dim_size == 384


# ---------------------------------------------------------------------------
# EmbeddingModel.batch_size
# ---------------------------------------------------------------------------


class TestBatchSize:
    """Tests for batch_size configuration."""

    def test_default_batch_size(self):
        """Default batch_size should be 64."""
        model = EmbeddingModel(model_name="test", dim_size=384)
        assert model.batch_size == 64

    def test_custom_batch_size(self):
        """Custom batch_size should be preserved."""
        model = EmbeddingModel(model_name="test", dim_size=384, batch_size=128)
        assert model.batch_size == 128


# ---------------------------------------------------------------------------
# try_init_model_silent
# ---------------------------------------------------------------------------


class TestTryInitModelSilent:
    """Tests for try_init_model_silent method."""

    def test_returns_true_if_already_loaded(self):
        """If _model is already set, should return True without reinitializing."""
        model = EmbeddingModel(model_name="test", dim_size=384)
        model._model = object()
        assert model.try_init_model_silent() is True

    def test_returns_false_if_already_failed(self):
        """If model already marked as failed, should return False."""
        model = EmbeddingModel(model_name="test", dim_size=384)
        model.is_model_failed = True
        assert model.try_init_model_silent() is False
