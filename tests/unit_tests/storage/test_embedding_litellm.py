# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.embedding_litellm (pure logic only, no API calls)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from datus.storage.embedding_litellm import LiteLLMEmbeddings

# ---------------------------------------------------------------------------
# Initialization / defaults
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingsInit:
    """Tests for LiteLLMEmbeddings construction and defaults."""

    @pytest.mark.ci
    def test_default_dim_is_none(self):
        """Default dim should be None."""
        emb = LiteLLMEmbeddings(name="test-model")
        assert emb.dim is None

    @pytest.mark.ci
    def test_custom_dim(self):
        """Custom dim should be stored correctly."""
        emb = LiteLLMEmbeddings(name="test-model", dim=1024)
        assert emb.dim == 1024

    @pytest.mark.ci
    def test_name_stored(self):
        """Name should be stored as given."""
        emb = LiteLLMEmbeddings(name="bedrock/amazon.titan-embed-text-v2:0")
        assert emb.name == "bedrock/amazon.titan-embed-text-v2:0"


# ---------------------------------------------------------------------------
# create() factory
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingsCreate:
    """Tests for the create() class method."""

    @pytest.mark.ci
    def test_create_returns_instance(self):
        """create() should return a LiteLLMEmbeddings instance with given params."""
        emb = LiteLLMEmbeddings.create(name="test-model", dim=512)
        assert isinstance(emb, LiteLLMEmbeddings)
        assert emb.name == "test-model"
        assert emb.dim == 512


# ---------------------------------------------------------------------------
# ndims / _ndims
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingsNdims:
    """Tests for dimension calculation logic."""

    @pytest.mark.ci
    def test_ndims_with_explicit_dim(self):
        """When dim is set explicitly, ndims() should return it directly."""
        emb = LiteLLMEmbeddings(name="test-model", dim=1024)
        assert emb.ndims() == 1024

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_ndims_auto_detect(self, mock_litellm):
        """When dim is None, ndims() should auto-detect via litellm.embedding call."""
        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.1] * 768}]
        mock_litellm.embedding.return_value = mock_resp

        emb = LiteLLMEmbeddings(name="test-model")
        assert emb.ndims() == 768
        mock_litellm.embedding.assert_called_once_with(model="test-model", input=["test"])


# ---------------------------------------------------------------------------
# sensitive_keys()
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingsSensitiveKeys:
    """Tests for sensitive_keys() static method."""

    @pytest.mark.ci
    def test_sensitive_keys_empty(self):
        """sensitive_keys() should return an empty list."""
        assert LiteLLMEmbeddings.sensitive_keys() == []


# ---------------------------------------------------------------------------
# _max_input_tokens
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingsMaxInputTokens:
    """Tests for _max_input_tokens cached property."""

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_auto_detect_from_model_info(self, mock_litellm):
        """Should use max_input_tokens from litellm.get_model_info when available."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 8192}

        emb = LiteLLMEmbeddings(name="test-model", dim=256)
        assert emb._max_input_tokens == 8192
        mock_litellm.get_model_info.assert_called_once_with("test-model")

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_fallback_on_error(self, mock_litellm):
        """Should fall back to 8191 when get_model_info raises an exception."""
        mock_litellm.get_model_info.side_effect = Exception("API error")

        emb = LiteLLMEmbeddings(name="test-model", dim=256)
        assert emb._max_input_tokens == 8191

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_fallback_on_missing_key(self, mock_litellm):
        """Should fall back to 8191 when max_input_tokens is not in model info."""
        mock_litellm.get_model_info.return_value = {}

        emb = LiteLLMEmbeddings(name="test-model", dim=256)
        assert emb._max_input_tokens == 8191

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_fallback_on_none_value(self, mock_litellm):
        """Should fall back to 8191 when max_input_tokens is None."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": None}

        emb = LiteLLMEmbeddings(name="test-model", dim=256)
        assert emb._max_input_tokens == 8191


# ---------------------------------------------------------------------------
# _truncate()
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingsTruncate:
    """Tests for text truncation logic."""

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_short_text_unchanged(self, mock_litellm):
        """Text shorter than the limit should be returned as-is."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 100}

        emb = LiteLLMEmbeddings(name="test-model", dim=256)
        text = "short text"
        assert emb._truncate(text) == text

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_long_text_truncated(self, mock_litellm):
        """Text longer than max_tokens * 3 chars should be truncated."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 10}

        emb = LiteLLMEmbeddings(name="test-model", dim=256)
        text = "a" * 100  # 100 chars, limit is 10 * 3 = 30
        result = emb._truncate(text)
        assert len(result) == 30
        assert result == "a" * 30

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_exact_boundary(self, mock_litellm):
        """Text exactly at the limit should not be truncated."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 10}

        emb = LiteLLMEmbeddings(name="test-model", dim=256)
        text = "a" * 30  # exactly 10 * 3 = 30 chars
        assert emb._truncate(text) == text


# ---------------------------------------------------------------------------
# generate_embeddings()
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingsGenerateEmbeddings:
    """Tests for the generate_embeddings method."""

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_normal_generation(self, mock_litellm):
        """Should return list of np.arrays for valid texts."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 8192}
        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.1, 0.2, 0.3]}]
        mock_litellm.embedding.return_value = mock_resp

        emb = LiteLLMEmbeddings(name="test-model", dim=3)
        result = emb.generate_embeddings(["hello", "world"])

        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result[1], [0.1, 0.2, 0.3])

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_empty_texts_return_none(self, mock_litellm):
        """All-empty texts should return all Nones without calling the API."""
        emb = LiteLLMEmbeddings(name="test-model", dim=3)
        result = emb.generate_embeddings(["", "", ""])

        assert result == [None, None, None]
        mock_litellm.embedding.assert_not_called()

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_mixed_empty_and_valid(self, mock_litellm):
        """Mix of empty and valid texts should return None for empty, arrays for valid."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 8192}
        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.5, 0.6]}]
        mock_litellm.embedding.return_value = mock_resp

        emb = LiteLLMEmbeddings(name="test-model", dim=2)
        result = emb.generate_embeddings(["", "hello", ""])

        assert result[0] is None
        assert isinstance(result[1], np.ndarray)
        np.testing.assert_array_almost_equal(result[1], [0.5, 0.6])
        assert result[2] is None

    @pytest.mark.ci
    @patch("datus.storage.embedding_litellm.litellm")
    def test_single_failure_continues(self, mock_litellm):
        """A failure on one text should not prevent others from embedding."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 8192}
        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.1, 0.2]}]

        # First call raises, second call succeeds
        mock_litellm.embedding.side_effect = [Exception("API error"), mock_resp]

        emb = LiteLLMEmbeddings(name="test-model", dim=2)
        result = emb.generate_embeddings(["fail-text", "good-text"])

        assert result[0] is None
        assert isinstance(result[1], np.ndarray)
        np.testing.assert_array_almost_equal(result[1], [0.1, 0.2])
