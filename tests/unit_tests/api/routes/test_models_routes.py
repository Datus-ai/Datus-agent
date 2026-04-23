# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/api/routes/models_routes.py."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
from unittest.mock import MagicMock

import pytest

from datus.api.routes import models_routes
from datus.api.routes.models_routes import list_models
from datus.configuration.agent_config import ModelConfig


def _make_svc(
    *,
    catalog: Dict[str, Any],
    available: Optional[Iterable[str]] = None,
    custom_models: Optional[Dict[str, Any]] = None,
    target_provider: Optional[str] = None,
    target_model: Optional[str] = None,
    target: str = "",
) -> MagicMock:
    """Build a MagicMock svc with provider_catalog + provider_available wired up.

    ``available`` is the whitelist of providers for which
    ``provider_available()`` returns True. None means all providers in the
    catalog are available.
    """
    allowed = set(available) if available is not None else set((catalog.get("providers") or {}).keys())
    svc = MagicMock()
    svc.agent_config.provider_catalog = catalog
    svc.agent_config.provider_available.side_effect = lambda p: p in allowed
    svc.agent_config.models = custom_models if custom_models is not None else {}
    svc.agent_config._target_provider = target_provider
    svc.agent_config._target_model = target_model
    svc.agent_config.target = target
    return svc


def _basic_catalog() -> Dict[str, Any]:
    return {
        "providers": {
            "openai": {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "models": ["gpt-4o", "gpt-4.1"],
                "default_model": "gpt-4.1",
            },
            "claude": {
                "type": "claude",
                "base_url": "https://api.anthropic.com",
                "api_key_env": "ANTHROPIC_API_KEY",
                "models": ["claude-sonnet-4-5"],
                "default_model": "claude-sonnet-4-5",
            },
            "deepseek": {
                "type": "deepseek",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "models": ["deepseek-chat"],
                "default_model": "deepseek-chat",
            },
        },
        "model_specs": {
            "gpt-4.1": {"context_length": 400000, "max_tokens": 128000},
            "gpt-4o": {"context_length": 128000, "max_tokens": 16384},
            "claude-sonnet-4-5": {"context_length": 1048576, "max_tokens": 65536},
            "deepseek-chat": {"context_length": 65535, "max_tokens": 8192},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Provider scoping: only configured providers appear
# ─────────────────────────────────────────────────────────────────────────────


class TestProviderScoping:
    @pytest.mark.asyncio
    async def test_only_configured_providers_are_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        svc = _make_svc(catalog=_basic_catalog(), available={"openai"})
        result = await list_models(svc)

        assert result.success is True
        assert result.data.providers == ["openai"]
        ids = {m.id for m in result.data.models}
        assert ids == {"gpt-4o", "gpt-4.1"}
        # Unavailable providers must not appear.
        assert all(m.provider == "openai" for m in result.data.models)

    @pytest.mark.asyncio
    async def test_no_configured_providers_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        svc = _make_svc(catalog=_basic_catalog(), available=set())
        result = await list_models(svc)

        assert result.success is True
        assert result.data.models == []
        assert result.data.providers == []

    @pytest.mark.asyncio
    async def test_all_available_returns_every_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        svc = _make_svc(catalog=_basic_catalog())
        result = await list_models(svc)

        assert sorted(result.data.providers) == ["claude", "deepseek", "openai"]
        ids = {m.id for m in result.data.models}
        assert ids == {"gpt-4o", "gpt-4.1", "claude-sonnet-4-5", "deepseek-chat"}


# ─────────────────────────────────────────────────────────────────────────────
# Cache metadata is surfaced in the response
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheMetadata:
    @pytest.mark.asyncio
    async def test_pricing_and_context_length_pass_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = {
            "openai": [
                {
                    "id": "gpt-4o",
                    "name": "GPT-4o",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
                }
            ]
        }
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: cache)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: "2026-04-21T07:15:48Z")

        svc = _make_svc(catalog=_basic_catalog(), available={"openai"})
        result = await list_models(svc)

        assert result.data.source == "cache"
        assert result.data.fetched_at == "2026-04-21T07:15:48Z"
        assert len(result.data.models) == 1
        model = result.data.models[0]
        assert model.provider == "openai"
        assert model.id == "gpt-4o"
        assert model.model == "gpt-4o"
        assert model.name == "GPT-4o"
        assert model.context_length == 128000
        assert model.max_tokens == 16384  # from model_specs fallback
        assert (model.pricing.prompt, model.pricing.completion) == ("0.0000025", "0.00001")

    @pytest.mark.asyncio
    async def test_cache_absent_uses_providers_yml_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        svc = _make_svc(catalog=_basic_catalog(), available={"claude"})
        result = await list_models(svc)

        assert result.data.source == "catalog"
        assert result.data.fetched_at is None
        assert len(result.data.models) == 1
        model = result.data.models[0]
        assert model.id == "claude-sonnet-4-5"
        assert model.model == "claude-sonnet-4-5"
        # context_length from model_specs, pricing unavailable without cache.
        assert model.context_length == 1048576
        assert model.pricing is None

    @pytest.mark.asyncio
    async def test_model_spec_prefix_matching_supplies_context_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Providers.yml uses prefix matching for model_specs — e.g. ``kimi-k2`` matches
        ``kimi-k2-0711-preview``. The endpoint must respect that when the cache
        lacks the field."""
        monkeypatch.setattr(
            models_routes,
            "load_cached_model_details",
            lambda: {"kimi": [{"id": "kimi-k2-0711-preview"}]},
        )
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        catalog = {
            "providers": {
                "kimi": {
                    "type": "kimi",
                    "models": ["kimi-k2-0711-preview"],
                    "default_model": "kimi-k2-0711-preview",
                }
            },
            "model_specs": {"kimi-k2": {"context_length": 256000, "max_tokens": 8192}},
        }
        svc = _make_svc(catalog=catalog, available={"kimi"})
        result = await list_models(svc)

        assert len(result.data.models) == 1
        assert result.data.models[0].context_length == 256000
        assert result.data.models[0].max_tokens == 8192

    @pytest.mark.asyncio
    async def test_unknown_slug_returns_null_context_and_pricing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            models_routes,
            "load_cached_model_details",
            lambda: {"openai": [{"id": "brand-new-slug"}]},
        )
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: "2026-04-21T00:00:00Z")

        svc = _make_svc(catalog=_basic_catalog(), available={"openai"})
        result = await list_models(svc)

        assert len(result.data.models) == 1
        model = result.data.models[0]
        assert model.id == "brand-new-slug"
        assert model.context_length is None
        assert model.max_tokens is None
        assert model.pricing is None

    @pytest.mark.asyncio
    async def test_pricing_with_only_prompt_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            models_routes,
            "load_cached_model_details",
            lambda: {"openai": [{"id": "gpt-4o", "pricing": {"prompt": "0.0000025"}}]},
        )
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        svc = _make_svc(catalog=_basic_catalog(), available={"openai"})
        result = await list_models(svc)

        pricing = result.data.models[0].pricing
        assert (pricing.prompt, pricing.completion) == ("0.0000025", None)


# ─────────────────────────────────────────────────────────────────────────────
# Cache-catalog interplay: cache drives primary data, model_specs fills gaps
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheCatalogInterplay:
    @pytest.mark.asyncio
    async def test_cache_entries_outside_available_providers_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A provider present in the cache but without configured credentials
        must not leak into the response."""
        monkeypatch.setattr(
            models_routes,
            "load_cached_model_details",
            lambda: {
                "openai": [{"id": "gpt-4o", "context_length": 128000}],
                "claude": [{"id": "claude-ghost"}],
            },
        )
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: "2026-04-21T00:00:00Z")

        svc = _make_svc(catalog=_basic_catalog(), available={"openai"})
        result = await list_models(svc)

        providers_in_response = {m.provider for m in result.data.models}
        assert providers_in_response == {"openai"}
        assert result.data.providers == ["openai"]

    @pytest.mark.asyncio
    async def test_available_provider_without_cache_falls_back_to_yaml_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If cache has openai but not claude, claude still appears via yaml models."""
        monkeypatch.setattr(
            models_routes,
            "load_cached_model_details",
            lambda: {"openai": [{"id": "gpt-4o"}]},
        )
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: "2026-04-21T00:00:00Z")

        svc = _make_svc(catalog=_basic_catalog(), available={"openai", "claude"})
        result = await list_models(svc)

        by_provider: Dict[str, list] = {}
        for model in result.data.models:
            by_provider.setdefault(model.provider, []).append(model.id)
        assert by_provider["openai"] == ["gpt-4o"]
        assert by_provider["claude"] == ["claude-sonnet-4-5"]


# ─────────────────────────────────────────────────────────────────────────────
# Defensive parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestDefensiveParsing:
    @pytest.mark.asyncio
    async def test_provider_catalog_not_a_dict_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        svc = MagicMock()
        svc.agent_config.provider_catalog = None
        svc.agent_config.provider_available.return_value = True
        svc.agent_config.models = {}
        svc.agent_config._target_provider = None
        svc.agent_config._target_model = None
        svc.agent_config.target = ""

        result = await list_models(svc)
        assert result.data.models == []
        assert result.data.providers == []

    @pytest.mark.asyncio
    async def test_provider_meta_without_models_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        catalog = {
            "providers": {
                "openai": {"type": "openai"},  # no `models` key at all
            }
        }
        svc = _make_svc(catalog=catalog, available={"openai"})
        result = await list_models(svc)

        assert result.data.models == []
        assert result.data.providers == []

    @pytest.mark.asyncio
    async def test_invalid_entry_in_cache_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            models_routes,
            "load_cached_model_details",
            lambda: {"openai": ["not-a-dict", {"id": ""}, {"id": "gpt-4o"}]},  # type: ignore[list-item]
        )
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: "2026-04-21T00:00:00Z")

        svc = _make_svc(catalog=_basic_catalog(), available={"openai"})
        result = await list_models(svc)

        assert [m.id for m in result.data.models] == ["gpt-4o"]


# ─────────────────────────────────────────────────────────────────────────────
# Custom models from agent.models
# ─────────────────────────────────────────────────────────────────────────────


def _custom_model(model: str, type_: str = "openai") -> ModelConfig:
    return ModelConfig(type=type_, api_key="sk-test", model=model)


class TestCustomModels:
    @pytest.mark.asyncio
    async def test_custom_models_are_appended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        custom = {
            "my-deepseek": _custom_model("deepseek-chat", "deepseek"),
            "my-gpt": _custom_model("gpt-4", "openai"),
        }
        svc = _make_svc(catalog=_basic_catalog(), available=set(), custom_models=custom)
        result = await list_models(svc)

        assert "custom" in result.data.providers
        assert len(result.data.models) == 2
        by_id = {m.id: m for m in result.data.models}
        assert by_id["my-deepseek"].provider == "custom"
        assert by_id["my-deepseek"].model == "deepseek-chat"
        assert by_id["my-gpt"].provider == "custom"
        assert by_id["my-gpt"].model == "gpt-4"

    @pytest.mark.asyncio
    async def test_custom_models_pick_up_model_specs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        custom = {"my-ds": _custom_model("deepseek-chat", "deepseek")}
        svc = _make_svc(catalog=_basic_catalog(), available=set(), custom_models=custom)
        result = await list_models(svc)

        m = result.data.models[0]
        assert m.id == "my-ds"
        assert m.model == "deepseek-chat"
        assert m.context_length == 65535
        assert m.max_tokens == 8192

    @pytest.mark.asyncio
    async def test_empty_custom_models_no_custom_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        svc = _make_svc(catalog=_basic_catalog(), available=set(), custom_models={})
        result = await list_models(svc)

        assert "custom" not in result.data.providers
        assert result.data.models == []

    @pytest.mark.asyncio
    async def test_custom_and_provider_models_coexist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        custom = {"my-ds": _custom_model("deepseek-chat", "deepseek")}
        svc = _make_svc(catalog=_basic_catalog(), available={"openai"}, custom_models=custom)
        result = await list_models(svc)

        providers = set(result.data.providers)
        assert providers == {"openai", "custom"}
        provider_models = [m for m in result.data.models if m.provider == "openai"]
        custom_models = [m for m in result.data.models if m.provider == "custom"]
        assert len(provider_models) == 2
        assert all(m.model == m.id for m in provider_models)
        assert len(custom_models) == 1
        assert custom_models[0].id == "my-ds"
        assert custom_models[0].model == "deepseek-chat"


# ─────────────────────────────────────────────────────────────────────────────
# Current model resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestCurrentModel:
    @pytest.mark.asyncio
    async def test_provider_target_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        custom = {"my-ds": _custom_model("deepseek-chat", "deepseek")}
        svc = _make_svc(
            catalog=_basic_catalog(),
            available={"openai"},
            custom_models=custom,
            target_provider="openai",
            target_model="gpt-4.1",
            target="my-ds",
        )
        result = await list_models(svc)

        assert result.data.current_model == "openai/gpt-4.1"

    @pytest.mark.asyncio
    async def test_custom_target_used_when_no_provider_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        custom = {"my-ds": _custom_model("deepseek-chat", "deepseek")}
        svc = _make_svc(
            catalog=_basic_catalog(),
            available=set(),
            custom_models=custom,
            target="my-ds",
        )
        result = await list_models(svc)

        assert result.data.current_model == "custom/my-ds"

    @pytest.mark.asyncio
    async def test_fallback_to_first_custom_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        custom = {"alpha": _custom_model("gpt-4", "openai")}
        svc = _make_svc(
            catalog=_basic_catalog(),
            available={"openai"},
            custom_models=custom,
        )
        result = await list_models(svc)

        assert result.data.current_model == "custom/alpha"

    @pytest.mark.asyncio
    async def test_fallback_to_first_provider_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        svc = _make_svc(catalog=_basic_catalog(), available={"openai"})
        result = await list_models(svc)

        assert result.data.current_model is not None
        assert result.data.current_model.startswith("openai/")

    @pytest.mark.asyncio
    async def test_no_models_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_routes, "load_cached_model_details", lambda: None)
        monkeypatch.setattr(models_routes, "load_cache_fetched_at", lambda: None)

        svc = _make_svc(catalog=_basic_catalog(), available=set())
        result = await list_models(svc)

        assert result.data.current_model is None
