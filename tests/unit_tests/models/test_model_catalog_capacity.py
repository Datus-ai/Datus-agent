# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Exercise automatic metadata resolution through the real model budget path."""

import asyncio
from unittest.mock import Mock

import litellm
import pytest

from datus.cli import provider_model_catalog as catalog
from datus.configuration.agent_config import ModelConfig
from datus.models.openai_model import OpenAIModel
from datus.models.request_capacity import DEFAULT_MAX_TOKENS


def _model(**overrides):
    return OpenAIModel(ModelConfig(type="openai", model="glm-5.3", api_key="test", **overrides))


def _glm_details(context=1048576, output=131072):
    return {"glm": [{"id": "glm-5.3", "context_length": context, "max_tokens": output}]}


@pytest.mark.asyncio
async def test_event_loop_capacity_uses_cache_after_threaded_refresh(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", {})
    fetch = Mock(return_value=_glm_details(context=524288, output=8192))
    monkeypatch.setattr(catalog, "fetch_openrouter_models", fetch)
    model = _model()
    assert model.context_length() == 1048576
    assert model.max_tokens() == DEFAULT_MAX_TOKENS
    fetch.assert_not_called()

    await asyncio.to_thread(catalog.resolve_model_details)

    assert model.context_length() == 524288
    assert model.max_tokens() == 8192
    fetch.assert_called_once()


def test_empty_home_fetches_both_limits_without_opening_model_picker(monkeypatch):
    fetch = Mock(return_value=_glm_details())
    monkeypatch.setattr(catalog, "fetch_openrouter_models", fetch)
    model = _model(tokenizer_ref="tiktoken:cl100k_base")
    capacity = model.request_capacity()
    assert capacity.context_length == 1048576
    assert capacity.default_output == 131072
    assert capacity.output_ceiling == 131072
    assert model.check_request({"messages": [{"role": "user", "content": "Review the skill"}]})["max_tokens"] == 131072
    assert model._build_agent("Review the skill", str, False, {}, []).model_settings.max_tokens == 131072
    fetch.assert_called_once()
    assert catalog.load_cached_model_details() == _glm_details()


def test_explicit_limits_win_over_catalog_metadata(monkeypatch):
    fetch = Mock(return_value=_glm_details())
    monkeypatch.setattr(catalog, "fetch_openrouter_models", fetch)
    model = _model(context_length=262144, max_tokens=8192)
    assert model.request_capacity().context_length == 262144
    assert model.request_capacity().default_output == 8192
    fetch.assert_called_once()  # Capacity checks still resolve the provider's output ceiling.


@pytest.mark.parametrize("field,value", [("context_length", 262144), ("max_tokens", 8192)])
def test_independent_override_keeps_other_limit_automatic(monkeypatch, field, value):
    monkeypatch.setattr(catalog, "fetch_openrouter_models", Mock(return_value=_glm_details()))
    model = _model(**{field: value})
    assert model.context_length() == (value if field == "context_length" else 1048576)
    # A smaller explicit window also constrains the automatically chosen output default.
    assert model.max_tokens() == (value if field == "max_tokens" else 124518)


def test_existing_model_observes_disk_cache_refresh(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", {})
    model = _model()
    assert model.context_length() == 1048576  # Unknown numeric version uses the fallback.
    assert model.max_tokens() == DEFAULT_MAX_TOKENS
    catalog.save_cached_models(_glm_details(context=524288, output=65536))
    assert model.context_length() == 524288
    assert model.max_tokens() == 65536


def test_switching_agent_homes_does_not_reuse_another_homes_specs(monkeypatch, tmp_path):
    path = [tmp_path / "first" / catalog.CACHE_FILE_NAME]
    monkeypatch.setattr(catalog, "_cache_file_path", lambda: path[0])
    catalog.save_cached_models(_glm_details(context=524288, output=65536))
    model = _model()
    assert model.context_length() == 524288
    path[0] = tmp_path / "second" / catalog.CACHE_FILE_NAME
    catalog.save_cached_models(_glm_details())
    assert model.context_length() == 1048576
    assert model.max_tokens() == 131072
    path[0] = tmp_path / "first" / catalog.CACHE_FILE_NAME
    assert model.context_length() == 524288
    assert model.max_tokens() == 65536


@pytest.mark.parametrize("model_name", ["glm-5.3", "glm-50", "glm-5.3-flashx"])
def test_new_numeric_version_does_not_inherit_glm5_specs(monkeypatch, model_name):
    monkeypatch.setattr(litellm, "model_cost", {})
    model = OpenAIModel(ModelConfig(type="openai", model=model_name, api_key="test"))
    assert model.context_length() == 1048576
    assert model.max_tokens() == DEFAULT_MAX_TOKENS


def test_exact_litellm_metadata_beats_family_prefix(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"glm-5-custom": {"max_input_tokens": 1048576, "max_output_tokens": 64000}},
    )
    model = OpenAIModel(ModelConfig(type="openai", model="glm-5-custom", api_key="test"))
    assert model.context_length() == 1048576
    assert model.max_tokens() == 64000


@pytest.mark.parametrize("value", [None, 0, -1, True, "131072"])
def test_invalid_remote_limits_do_not_replace_local_specs(monkeypatch, value):
    monkeypatch.setattr(
        catalog,
        "fetch_openrouter_models",
        Mock(return_value={"glm": [{"id": "glm-5", "context_length": value, "max_tokens": value}]}),
    )
    model = OpenAIModel(ModelConfig(type="openai", model="glm-5", api_key="test"))
    assert model.context_length() == 200000
    assert model.max_tokens() == 95000  # Local output default leaves input headroom.


def test_output_only_metadata_keeps_local_context_fallback(monkeypatch):
    monkeypatch.setattr(
        catalog,
        "fetch_openrouter_models",
        Mock(return_value={"glm": [{"id": "glm-5", "max_tokens": 8192}]}),
    )
    model = OpenAIModel(ModelConfig(type="openai", model="glm-5", api_key="test"))
    assert model.context_length() == 200000
    assert model.max_tokens() == 8192


def test_large_gateway_output_ceiling_does_not_expand_default_budget(monkeypatch):
    monkeypatch.setattr(catalog, "fetch_openrouter_models", Mock(return_value=_glm_details(output=943718)))
    model = _model(tokenizer_ref="tiktoken:cl100k_base")
    capacity = model.request_capacity()
    assert capacity.context_length == 1048576
    assert capacity.default_output == DEFAULT_MAX_TOKENS
    assert capacity.output_ceiling == 943718
    assert (
        model.check_request({"messages": [{"role": "user", "content": "Review the skill"}]})["max_tokens"]
        == DEFAULT_MAX_TOKENS
    )


def test_known_output_default_is_preserved_when_gateway_advertises_more(monkeypatch):
    monkeypatch.setattr(
        catalog,
        "fetch_openrouter_models",
        Mock(return_value={"deepseek": [{"id": "deepseek-v4-flash", "max_tokens": 943718}]}),
    )
    model = OpenAIModel(ModelConfig(type="deepseek", model="deepseek-v4-flash", api_key="test"))
    assert model.max_tokens() == 384000


@pytest.mark.parametrize("model_name,expected", [("glm-5.3", 256000), ("gpt-4o", 16384)])
@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_catalog_slugs_do_not_raise_output_defaults(monkeypatch, model_name, expected, reverse):
    monkeypatch.setattr(litellm, "model_cost", {})
    buckets = [
        ("openai", [{"id": model_name, "context_length": 1048576, "max_tokens": 256000}]),
        ("glm", [{"id": model_name, "context_length": 1048576, "max_tokens": 512000}]),
    ]
    details = dict(reversed(buckets) if reverse else buckets)
    monkeypatch.setattr(catalog, "fetch_openrouter_models", Mock(return_value=details))
    model = OpenAIModel(ModelConfig(type="openai", model=model_name, api_key="test"))
    assert model.max_tokens() == expected
