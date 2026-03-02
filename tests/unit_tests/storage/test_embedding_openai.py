# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.embedding_openai (pure logic only, no API calls)."""

import pytest

from datus.storage.embedding_openai import OpenAIEmbeddings
from datus.utils.exceptions import DatusException

# ---------------------------------------------------------------------------
# Initialization / defaults
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingsInit:
    """Tests for OpenAIEmbeddings construction and defaults."""

    @pytest.mark.ci
    def test_default_values(self):
        """Default instance should use ada-002 with expected field defaults."""
        emb = OpenAIEmbeddings()
        assert emb.name == "text-embedding-ada-002"
        assert emb.dim is None
        assert emb.base_url is None
        assert emb.default_headers is None
        assert emb.organization is None
        assert emb.api_key is None
        assert emb.use_azure is False

    @pytest.mark.ci
    def test_custom_values(self):
        """All fields should be settable via constructor kwargs."""
        emb = OpenAIEmbeddings(
            name="text-embedding-3-large",
            dim=256,
            base_url="https://custom.endpoint/v1",
            default_headers={"X-Custom": "value"},
            organization="org-abc",
            api_key="sk-test",
            use_azure=True,
        )
        assert emb.name == "text-embedding-3-large"
        assert emb.dim == 256
        assert emb.base_url == "https://custom.endpoint/v1"
        assert emb.default_headers == {"X-Custom": "value"}
        assert emb.organization == "org-abc"
        assert emb.api_key == "sk-test"
        assert emb.use_azure is True


# ---------------------------------------------------------------------------
# create() factory
# ---------------------------------------------------------------------------


class TestCreate:
    """Tests for the create() class method."""

    @pytest.mark.ci
    def test_create_returns_instance(self):
        """create() should return an OpenAIEmbeddings instance with given params."""
        emb = OpenAIEmbeddings.create(name="text-embedding-3-small", dim=512)
        assert isinstance(emb, OpenAIEmbeddings)
        assert emb.name == "text-embedding-3-small"
        assert emb.dim == 512


# ---------------------------------------------------------------------------
# ndims / _ndims
# ---------------------------------------------------------------------------


class TestNdims:
    """Tests for dimension calculation logic."""

    @pytest.mark.ci
    def test_ada_002_default_dims(self):
        """text-embedding-ada-002 should always return 1536."""
        emb = OpenAIEmbeddings(name="text-embedding-ada-002")
        assert emb.ndims() == 1536

    @pytest.mark.ci
    def test_ada_002_ignores_dim_override(self):
        """text-embedding-ada-002 has fixed 1536 dimensions regardless of dim parameter."""
        emb = OpenAIEmbeddings(name="text-embedding-ada-002", dim=256)
        assert emb.ndims() == 1536

    @pytest.mark.ci
    def test_3_large_default_dims(self):
        """text-embedding-3-large should default to 3072."""
        emb = OpenAIEmbeddings(name="text-embedding-3-large")
        assert emb.ndims() == 3072

    @pytest.mark.ci
    def test_3_large_custom_dims(self):
        """text-embedding-3-large should respect custom dim when provided."""
        emb = OpenAIEmbeddings(name="text-embedding-3-large", dim=1024)
        assert emb.ndims() == 1024

    @pytest.mark.ci
    def test_3_small_default_dims(self):
        """text-embedding-3-small should default to 1536."""
        emb = OpenAIEmbeddings(name="text-embedding-3-small")
        assert emb.ndims() == 1536

    @pytest.mark.ci
    def test_3_small_custom_dims(self):
        """text-embedding-3-small should respect custom dim when provided."""
        emb = OpenAIEmbeddings(name="text-embedding-3-small", dim=512)
        assert emb.ndims() == 512

    @pytest.mark.ci
    def test_unknown_model_raises_value_error(self):
        """An unknown model name should raise DatusException."""
        emb = OpenAIEmbeddings(name="nonexistent-model")
        with pytest.raises(DatusException, match="Unknown embedding model name"):
            emb.ndims()


# ---------------------------------------------------------------------------
# model_names()
# ---------------------------------------------------------------------------


class TestModelNames:
    """Tests for model_names() static method."""

    @pytest.mark.ci
    def test_model_names_returns_expected_list(self):
        """model_names() should return the three known model names."""
        names = OpenAIEmbeddings.model_names()
        assert names == [
            "text-embedding-ada-002",
            "text-embedding-3-large",
            "text-embedding-3-small",
        ]


# ---------------------------------------------------------------------------
# sensitive_keys()
# ---------------------------------------------------------------------------


class TestSensitiveKeys:
    """Tests for sensitive_keys() static method."""

    @pytest.mark.ci
    def test_sensitive_keys_returns_api_key(self):
        """sensitive_keys() should return ['api_key']."""
        assert OpenAIEmbeddings.sensitive_keys() == ["api_key"]


# ---------------------------------------------------------------------------
# __setattr__ (attribute change tracking)
# ---------------------------------------------------------------------------


class TestSetattr:
    """Tests for the custom __setattr__ (attribute change logging)."""

    @pytest.mark.ci
    def test_setattr_updates_field(self):
        """Setting a tracked attribute should update the value."""
        emb = OpenAIEmbeddings()
        emb.name = "text-embedding-3-large"
        assert emb.name == "text-embedding-3-large"

    @pytest.mark.ci
    def test_setattr_non_tracked_field(self):
        """Setting an untracked attribute should still work."""
        emb = OpenAIEmbeddings()
        emb.organization = "org-new"
        assert emb.organization == "org-new"
