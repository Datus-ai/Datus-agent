# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.embedding_openai (pure logic only, no API calls)."""

import copy
import json

import httpx
import pytest
from openai import AzureOpenAI, OpenAI

from datus.configuration.agent_config import ModelConfig
from datus.storage import embedding_models
from datus.storage.embedding_openai import OpenAIEmbeddings
from datus.utils.constants import EmbeddingProvider
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


@pytest.fixture
def embedding_api(monkeypatch):
    """Use the real SDK with a local transport that rejects unsupported fields."""
    requests = []
    clients = []
    native_widths = {"text-embedding-ada-002": 1536, "text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        model = payload["model"]
        supports_dimensions = model in ("text-embedding-3-small", "text-embedding-3-large", "dimensions-capable-custom")
        if "dimensions" in payload and not supports_dimensions:
            return httpx.Response(
                400, json={"error": {"message": "dimensions is unsupported", "type": "invalid_request_error"}}
            )
        width = payload.get("dimensions", native_widths.get(model, 4))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": model,
                "data": [
                    {"object": "embedding", "index": index, "embedding": [float(index + 1)] * width}
                    for index, _ in enumerate(payload["input"])
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    def create_client(**kwargs):
        client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)), max_retries=0)
        clients.append(client)
        return client

    def create_azure_client(**kwargs):
        client = AzureOpenAI(
            **kwargs,
            api_version="2024-02-01",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
            max_retries=0,
        )
        clients.append(client)
        return client

    monkeypatch.setattr("datus.storage.embedding_openai.OpenAI", create_client)
    monkeypatch.setattr("datus.storage.embedding_openai.AzureOpenAI", create_azure_client)
    yield requests
    for client in clients:
        client.close()


class TestCustomEmbeddingCompatibility:
    def test_stores_with_different_provider_routes_do_not_share_a_model(self, embedding_api, monkeypatch):
        monkeypatch.setattr(embedding_models, "EMBEDDING_MODELS", {})
        first = ModelConfig(
            type="openai", model="chat-a", api_key="local-test-a", base_url="https://first.example.invalid/v1"
        )
        second = ModelConfig(
            type="openai", model="chat-b", api_key="local-test-b", base_url="https://second.example.invalid/v1"
        )
        shared = {"registry_name": "openai", "model_name": "vendor/custom-model", "dim_size": 4}
        models = embedding_models.init_embedding_models(
            {
                "embedding_device_type": "cpu",
                "database": {**shared, "target_model": "first"},
                "document": {**shared, "target_model": "second"},
                "metric": {**shared, "target_model": "second"},
            },
            {"first": first, "second": second},
            first,
        )

        assert models["database"] is not models["document"]
        assert models["document"] is models["metric"]
        for store, host in [("database", "first.example.invalid"), ("document", "second.example.invalid")]:
            assert models[store].model.generate_embeddings(["query"]) == [[1.0] * 4]
            assert models[store].model._openai_client.base_url.host == host

    @pytest.mark.parametrize("dim_size", [0, -1])
    def test_storage_initialization_rejects_nonpositive_custom_widths(self, dim_size, monkeypatch):
        monkeypatch.setattr(embedding_models, "EMBEDDING_MODELS", {})
        with pytest.raises(DatusException, match="positive dim_size"):
            embedding_models.init_embedding_models(
                {
                    "embedding_device_type": "cpu",
                    "database": {"registry_name": "openai", "model_name": "vendor/custom-model", "dim_size": dim_size},
                },
                {},
                None,
            )

    @pytest.mark.parametrize("send_dimensions", [False, True])
    def test_environment_credentials_keep_the_custom_request_policy(self, embedding_api, monkeypatch, send_dimensions):
        monkeypatch.setenv("OPENAI_API_KEY", "local-test-only")
        wrapper = embedding_models.EmbeddingModel(
            model_name="dimensions-capable-custom",
            dim_size=4,
            registry_name=EmbeddingProvider.OPENAI,
            send_dimensions=send_dimensions,
        )

        assert wrapper.model.generate_embeddings(["alpha"]) == [[1.0] * 4]
        assert wrapper.model.ndims() == wrapper.dim_size == 4
        if send_dimensions:
            assert embedding_api[-1]["dimensions"] == 4
        else:
            assert "dimensions" not in embedding_api[-1]

    def test_stores_with_identical_settings_initialize_once(self, embedding_api, monkeypatch):
        monkeypatch.setattr(embedding_models, "EMBEDDING_MODELS", {})
        monkeypatch.setattr(embedding_models, "EMBEDDING_DEVICE_TYPE", "cpu")
        provider = ModelConfig(
            type="openai", model="unused-chat-model", api_key="local-test-only", base_url="http://embedding.test/v1"
        )
        config = {
            "registry_name": EmbeddingProvider.OPENAI,
            "model_name": "dimensions-capable-custom",
            "dim_size": 4,
            "send_dimensions": False,
        }
        models = embedding_models.init_embedding_models(
            {"embedding_device_type": "cpu", "database": dict(config), "document": dict(config)}, {}, provider
        )

        for name, text in [("database", "alpha"), ("document", "beta")]:
            assert models[name].model.generate_embeddings([text]) == [[1.0] * 4]
            assert models[name].model.ndims() == models[name].dim_size == 4
        assert [request["input"] for request in embedding_api] == [["foo"], ["alpha"], ["beta"]]
        assert all("dimensions" not in request for request in embedding_api)

    @pytest.mark.parametrize("name", ["kinfra-text-embedding-0.6b", "vendor/custom-model", "deployment-name"])
    def test_custom_model_uses_configured_vector_width(self, name):
        embedding = OpenAIEmbeddings(name=name, dim=4)

        assert embedding.ndims() == 4
        embedding.dim = 8
        assert embedding.ndims() == 8

    @pytest.mark.parametrize("dim", [None, 0, -1])
    def test_custom_model_requires_a_positive_configured_width(self, dim):
        embedding = OpenAIEmbeddings(name="kinfra-text-embedding-0.6b", dim=dim)

        with pytest.raises(DatusException, match="positive dim_size"):
            embedding.ndims()

    @pytest.mark.parametrize(
        ("name", "send_dimensions", "expected_dimensions", "expected_width"),
        [
            ("kinfra-text-embedding-0.6b", None, None, 4),
            ("vendor/custom-model", None, None, 4),
            ("kinfra-text-embedding-0.6b", False, None, 4),
            ("dimensions-capable-custom", True, 4, 4),
            ("text-embedding-3-small", None, 4, 4),
            ("text-embedding-3-large", None, 4, 4),
            ("text-embedding-3-small", False, None, 1536),
            ("text-embedding-ada-002", True, None, 1536),
        ],
    )
    def test_request_policy_preserves_embeddings_and_empty_input_positions(
        self, embedding_api, name, send_dimensions, expected_dimensions, expected_width
    ):
        embedding = OpenAIEmbeddings(
            name=name,
            dim=expected_width if send_dimensions is False else 4,
            send_dimensions=send_dimensions,
            api_key="local-test-only",
            base_url="http://embedding.test/v1",
        )

        result = embedding.generate_embeddings(["alpha", "", "beta", None])

        assert result == [[1.0] * expected_width, None, [2.0] * expected_width, None]
        assert embedding.ndims() == expected_width
        assert embedding_api[-1]["input"] == ["alpha", "beta"]
        assert embedding_api[-1]["model"] == name
        if expected_dimensions is None:
            assert "dimensions" not in embedding_api[-1]
        else:
            assert embedding_api[-1]["dimensions"] == expected_dimensions

    def test_azure_deployment_can_explicitly_request_dimensions(self, embedding_api):
        embedding = OpenAIEmbeddings(
            name="dimensions-capable-custom",
            dim=4,
            send_dimensions=True,
            api_key="local-test-only",
            base_url="http://embedding.test/openai/deployments/test",
            use_azure=True,
        )

        assert embedding.generate_embeddings(["alpha"]) == [[1.0] * 4]
        assert embedding.ndims() == 4
        assert embedding_api[-1]["dimensions"] == 4

    @pytest.mark.parametrize("send_dimensions", [None, False, True])
    def test_storage_config_keeps_dimension_policy_through_copy_and_serialization(
        self, embedding_api, monkeypatch, send_dimensions
    ):
        monkeypatch.setattr(embedding_models, "EMBEDDING_MODELS", {})
        monkeypatch.setattr(embedding_models, "EMBEDDING_DEVICE_TYPE", "cpu")
        provider = ModelConfig(
            type="openai", model="unused-chat-model", api_key="local-test-only", base_url="http://embedding.test/v1"
        )
        config = {"registry_name": "openai", "model_name": "dimensions-capable-custom", "dim_size": 4}
        if send_dimensions is not None:
            config["send_dimensions"] = send_dimensions
        models = embedding_models.init_embedding_models(
            {"embedding_device_type": "cpu", "database": config}, {}, provider
        )
        wrapper = models["database"]
        copied = copy.deepcopy(wrapper)
        reconstructed = embedding_models.EmbeddingModel(**wrapper.to_dict(), openai_config=provider)

        for candidate in (wrapper, copied, reconstructed):
            assert candidate.model.generate_embeddings(["alpha"]) == [[1.0] * 4]
            assert candidate.model.ndims() == candidate.dim_size == 4
            if send_dimensions is True:
                assert embedding_api[-1]["dimensions"] == 4
            else:
                assert "dimensions" not in embedding_api[-1]

    def test_stores_with_different_request_policies_do_not_share_a_model(self, embedding_api, monkeypatch):
        monkeypatch.setattr(embedding_models, "EMBEDDING_MODELS", {})
        monkeypatch.setattr(embedding_models, "EMBEDDING_DEVICE_TYPE", "cpu")
        provider = ModelConfig(
            type="openai", model="unused-chat-model", api_key="local-test-only", base_url="http://embedding.test/v1"
        )
        shared = {"registry_name": EmbeddingProvider.OPENAI, "model_name": "dimensions-capable-custom", "dim_size": 4}
        models = embedding_models.init_embedding_models(
            {
                "embedding_device_type": "cpu",
                "database": {**shared, "send_dimensions": True},
                "document": {**shared, "send_dimensions": False},
            },
            {},
            provider,
        )

        assert models["database"].model.generate_embeddings(["alpha"]) == [[1.0] * 4]
        assert embedding_api[-1]["dimensions"] == 4
        assert models["document"].model.generate_embeddings(["alpha"]) == [[1.0] * 4]
        assert "dimensions" not in embedding_api[-1]

    def test_stores_with_different_vector_widths_keep_their_dimensions(self, embedding_api, monkeypatch):
        monkeypatch.setattr(embedding_models, "EMBEDDING_MODELS", {})
        monkeypatch.setattr(embedding_models, "EMBEDDING_DEVICE_TYPE", "cpu")
        provider = ModelConfig(
            type="openai", model="unused-chat-model", api_key="local-test-only", base_url="http://embedding.test/v1"
        )
        shared = {
            "registry_name": EmbeddingProvider.OPENAI,
            "model_name": "dimensions-capable-custom",
            "send_dimensions": True,
        }
        models = embedding_models.init_embedding_models(
            {
                "embedding_device_type": "cpu",
                "database": {**shared, "dim_size": 4},
                "document": {**shared, "dim_size": 8},
            },
            {},
            provider,
        )

        for name, width in [("database", 4), ("document", 8)]:
            assert models[name].model.generate_embeddings(["alpha"]) == [[1.0] * width]
            assert models[name].model.ndims() == models[name].dim_size == width
            assert embedding_api[-1]["dimensions"] == width
