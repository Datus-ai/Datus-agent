# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/provider_model_catalog.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock, patch

import httpx
import pytest

from datus.cli import provider_model_catalog as pmc

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_datus_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect `_cache_file_path()` to a tmp datus_home."""
    pm = MagicMock()
    pm.datus_home = tmp_path
    monkeypatch.setattr(pmc, "get_path_manager", lambda: pm)
    return tmp_path


def _payload(*ids: str) -> Dict[str, Any]:
    return {"data": [{"id": mid} for mid in ids]}


def _install_mock_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> patch:
    """Patch `httpx.Client` inside the helper so every instance uses MockTransport."""
    original = httpx.Client
    transport = httpx.MockTransport(handler)

    def _factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    return patch.object(pmc.httpx, "Client", side_effect=_factory)


def _raising_transport(exc: Exception) -> httpx.MockTransport:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.MockTransport(handler)


def _local_catalog() -> dict:
    return {
        "providers": {
            "openai": {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "models": ["gpt-4.1"],
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
            "codex": {
                "type": "codex",
                "auth_type": "oauth",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "models": ["codex-mini-latest"],
                "default_model": "codex-mini-latest",
            },
            "claude_subscription": {
                "type": "claude",
                "auth_type": "subscription",
                "base_url": "https://api.anthropic.com",
                "models": ["claude-sonnet-4-6"],
                "default_model": "claude-sonnet-4-6",
            },
        },
        "model_overrides": {"kimi-k2.5": {"temperature": 1.0}},
        "model_specs": {"gpt-4.1": {"context_length": 400000, "max_tokens": 128000}},
    }


def _write_cache(
    tmp_home: Path,
    models: Dict[str, List[Any]],
    *,
    version: int = 1,
    fetched_at: str = "2026-04-18T00:00:00Z",
) -> Path:
    cache = tmp_home / "cache" / pmc.CACHE_FILE_NAME
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "version": version,
                "source": "openrouter",
                "fetched_at": fetched_at,
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# _is_model_unfit
# ─────────────────────────────────────────────────────────────────────────────


class TestIsModelUnfit:
    @pytest.mark.parametrize(
        "slug",
        [
            "gpt-4o:free",
            "gpt-4o:extended",
            "claude-3-haiku:beta",
            "deepseek-chat:nitro",
            "gpt-4o:floor",
        ],
    )
    def test_routing_suffixes_are_rejected(self, slug: str) -> None:
        assert pmc._is_model_unfit(slug, None) is True

    @pytest.mark.parametrize(
        "slug",
        [
            "text-embedding-3-large",
            "text-embedding-ada-002",
            "dall-e-3",
            "tts-1",
            "tts-1-hd",
            "whisper-1",
            "text-moderation-latest",
            "gpt-4o-realtime-preview",
            "gpt-4o-transcribe",
            "gpt-audio",
            "gpt-audio-mini",
            "lyria-3-pro-preview",
            "lyria-3-clip-preview",
        ],
    )
    def test_non_chat_models_are_rejected(self, slug: str) -> None:
        assert pmc._is_model_unfit(slug, None) is True

    def test_small_context_is_rejected(self) -> None:
        assert pmc._is_model_unfit("some-model", 2048) is True
        assert pmc._is_model_unfit("some-model", 1024) is True
        assert pmc._is_model_unfit("some-model", 1) is True

    def test_context_at_threshold_passes(self) -> None:
        assert pmc._is_model_unfit("some-model", pmc._MIN_CONTEXT_LENGTH) is False

    def test_none_context_passes(self) -> None:
        assert pmc._is_model_unfit("gpt-4o", None) is False

    @pytest.mark.parametrize(
        "slug",
        [
            "gpt-4o",
            "gpt-4.1",
            "claude-sonnet-4-5",
            "deepseek-chat",
            "gemini-2.5-pro",
            "o3",
            "qwen3-max",
        ],
    )
    def test_legitimate_models_pass(self, slug: str) -> None:
        assert pmc._is_model_unfit(slug, 128000) is False

    def test_fragment_matching_is_case_insensitive(self) -> None:
        assert pmc._is_model_unfit("Text-Embedding-3-Large", None) is True


# ─────────────────────────────────────────────────────────────────────────────
# _bucket_by_vendor
# ─────────────────────────────────────────────────────────────────────────────


def _ids_only(buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    """Extract just the slug ids from the rich bucket output for equality checks."""
    return {k: [entry["id"] for entry in v] for k, v in buckets.items()}


class TestBucketByVendor:
    @pytest.mark.parametrize(
        "model_id,expected_provider,expected_slug",
        [
            ("openai/gpt-4o", "openai", "gpt-4o"),
            ("anthropic/claude-sonnet-4-5", "claude", "claude-sonnet-4-5"),
            ("deepseek/deepseek-chat", "deepseek", "deepseek-chat"),
            ("moonshotai/kimi-k2", "kimi", "kimi-k2"),
            ("moonshot/kimi-k2-turbo", "kimi", "kimi-k2-turbo"),
            ("qwen/qwen3-max", "qwen", "qwen3-max"),
            ("alibaba/qwen3-coder-plus", "qwen", "qwen3-coder-plus"),
            ("google/gemini-2.5-pro", "gemini", "gemini-2.5-pro"),
            ("minimax/MiniMax-M2.7", "minimax", "MiniMax-M2.7"),
            ("z-ai/glm-5", "glm", "glm-5"),
            ("zhipuai/glm-4.7", "glm", "glm-4.7"),
            ("thudm/glm-4.5-air", "glm", "glm-4.5-air"),
        ],
    )
    def test_each_known_vendor_maps_to_expected_provider(
        self, model_id: str, expected_provider: str, expected_slug: str
    ) -> None:
        buckets = pmc._bucket_by_vendor([{"id": model_id}])
        assert _ids_only(buckets) == {expected_provider: [expected_slug]}

    def test_unknown_vendor_is_dropped(self) -> None:
        assert pmc._bucket_by_vendor([{"id": "mystery/foo-1"}]) == {}

    def test_vendor_match_is_case_insensitive(self) -> None:
        buckets = pmc._bucket_by_vendor([{"id": "OpenAI/gpt-4o"}])
        assert _ids_only(buckets) == {"openai": ["gpt-4o"]}

    def test_slug_preserves_order_and_deduplicates(self) -> None:
        buckets = pmc._bucket_by_vendor(
            [
                {"id": "openai/gpt-4o"},
                {"id": "openai/gpt-4.1"},
                {"id": "openai/gpt-4o"},  # duplicate
                {"id": "openai/o3"},
            ]
        )
        assert _ids_only(buckets) == {"openai": ["gpt-4o", "gpt-4.1", "o3"]}

    def test_id_without_slash_is_dropped(self) -> None:
        assert pmc._bucket_by_vendor([{"id": "gpt-4o"}]) == {}

    def test_empty_slug_is_dropped(self) -> None:
        assert pmc._bucket_by_vendor([{"id": "openai/"}]) == {}

    def test_non_dict_entries_are_skipped(self) -> None:
        assert pmc._bucket_by_vendor(["openai/gpt-4o", None, 42]) == {}  # type: ignore[list-item]

    def test_missing_id_field_is_skipped(self) -> None:
        assert pmc._bucket_by_vendor([{"name": "No id here"}]) == {}

    def test_allowed_providers_drops_non_whitelisted(self) -> None:
        """providers.yml is the authoritative whitelist — vendors it does not
        declare must not enter the bucket even if OPENROUTER_VENDOR_MAP matches."""
        buckets = pmc._bucket_by_vendor(
            [
                {"id": "openai/gpt-4o"},
                {"id": "anthropic/claude-sonnet-4-5"},
                {"id": "deepseek/deepseek-chat"},
            ],
            allowed_providers={"openai"},
        )
        assert _ids_only(buckets) == {"openai": ["gpt-4o"]}

    def test_allowed_providers_none_means_no_filter(self) -> None:
        buckets = pmc._bucket_by_vendor(
            [{"id": "openai/gpt-4o"}, {"id": "anthropic/claude-x"}],
            allowed_providers=None,
        )
        assert sorted(_ids_only(buckets).keys()) == ["claude", "openai"]

    def test_pricing_and_context_length_are_preserved(self) -> None:
        buckets = pmc._bucket_by_vendor(
            [
                {
                    "id": "openai/gpt-4o",
                    "name": "GPT-4o",
                    "context_length": 128000,
                    "pricing": {
                        "prompt": "0.0000025",
                        "completion": "0.00001",
                        "request": "0",
                    },
                }
            ]
        )
        assert buckets == {
            "openai": [
                {
                    "id": "gpt-4o",
                    "name": "GPT-4o",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
                }
            ]
        }

    def test_context_length_falls_back_to_top_provider(self) -> None:
        buckets = pmc._bucket_by_vendor(
            [
                {
                    "id": "openai/gpt-4o",
                    "top_provider": {"context_length": 64000},
                }
            ]
        )
        assert buckets["openai"][0]["context_length"] == 64000

    def test_missing_pricing_produces_no_pricing_field(self) -> None:
        buckets = pmc._bucket_by_vendor([{"id": "openai/gpt-4o", "pricing": {"request": "0"}}])
        assert "pricing" not in buckets["openai"][0]

    def test_non_dict_pricing_is_ignored(self) -> None:
        buckets = pmc._bucket_by_vendor([{"id": "openai/gpt-4o", "pricing": "free"}])
        assert "pricing" not in buckets["openai"][0]

    def test_routing_suffix_models_are_filtered(self) -> None:
        buckets = pmc._bucket_by_vendor(
            [
                {"id": "openai/gpt-4o"},
                {"id": "openai/gpt-4o:free"},
                {"id": "openai/gpt-4o:extended"},
            ]
        )
        assert _ids_only(buckets) == {"openai": ["gpt-4o"]}

    def test_non_chat_models_are_filtered(self) -> None:
        buckets = pmc._bucket_by_vendor(
            [
                {"id": "openai/gpt-4o"},
                {"id": "openai/text-embedding-3-large"},
                {"id": "openai/dall-e-3"},
                {"id": "openai/tts-1"},
                {"id": "openai/gpt-audio"},
                {"id": "google/lyria-3-pro-preview"},
            ]
        )
        assert _ids_only(buckets) == {"openai": ["gpt-4o"]}

    def test_small_context_models_are_filtered(self) -> None:
        buckets = pmc._bucket_by_vendor(
            [
                {"id": "openai/gpt-4o", "context_length": 128000},
                {"id": "openai/old-model", "context_length": 2048},
            ]
        )
        assert _ids_only(buckets) == {"openai": ["gpt-4o"]}


# ─────────────────────────────────────────────────────────────────────────────
# Cache I/O
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheIO:
    def test_load_cached_models_returns_none_when_file_missing(self, fake_datus_home: Path) -> None:
        assert pmc.load_cached_models() is None
        assert pmc.load_cached_model_details() is None
        assert pmc.load_cache_fetched_at() is None

    def test_save_then_load_roundtrip_v2_details(self, fake_datus_home: Path) -> None:
        buckets = {
            "openai": [
                {
                    "id": "gpt-4o",
                    "name": "GPT-4o",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
                }
            ],
            "claude": [{"id": "claude-sonnet-4-5"}],
        }
        pmc.save_cached_models(buckets)

        cache_path = fake_datus_home / "cache" / pmc.CACHE_FILE_NAME
        assert cache_path.exists()

        assert pmc.load_cached_model_details() == buckets
        # Slug-only view collapses entries to their ids for legacy consumers.
        assert pmc.load_cached_models() == {"openai": ["gpt-4o"], "claude": ["claude-sonnet-4-5"]}
        assert pmc.load_cache_fetched_at() and pmc.load_cache_fetched_at().endswith("Z")

    def test_save_is_atomic_no_leftover_tmp(self, fake_datus_home: Path) -> None:
        pmc.save_cached_models({"openai": [{"id": "gpt-4o"}]})
        tmp = fake_datus_home / "cache" / (pmc.CACHE_FILE_NAME + ".tmp")
        assert not tmp.exists()

    def test_cache_file_contains_version_and_fetched_at(self, fake_datus_home: Path) -> None:
        pmc.save_cached_models({"openai": [{"id": "gpt-4o", "context_length": 128000}]})
        raw = json.loads((fake_datus_home / "cache" / pmc.CACHE_FILE_NAME).read_text())
        assert raw["version"] == pmc.CACHE_SCHEMA_VERSION == 2
        assert raw["source"] == "openrouter"
        assert raw["fetched_at"].endswith("Z")
        assert raw["models"] == {"openai": [{"id": "gpt-4o", "context_length": 128000}]}

    def test_load_v1_cache_is_backward_compatible(self, fake_datus_home: Path) -> None:
        """v1 caches (flat slug lists) are upgraded on read to avoid a cold cache."""
        _write_cache(fake_datus_home, {"openai": ["gpt-4o", "gpt-4.1"]}, version=1)

        assert pmc.load_cached_models() == {"openai": ["gpt-4o", "gpt-4.1"]}
        assert pmc.load_cached_model_details() == {"openai": [{"id": "gpt-4o"}, {"id": "gpt-4.1"}]}

    def test_load_rejects_wrong_version(self, fake_datus_home: Path) -> None:
        _write_cache(fake_datus_home, {"openai": [{"id": "gpt-4o"}]}, version=99)
        assert pmc.load_cached_models() is None
        assert pmc.load_cached_model_details() is None

    def test_load_rejects_missing_models_key(self, fake_datus_home: Path) -> None:
        cache = fake_datus_home / "cache" / pmc.CACHE_FILE_NAME
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"version": 2, "source": "openrouter"}))
        assert pmc.load_cached_models() is None
        assert pmc.load_cached_model_details() is None

    def test_load_rejects_corrupt_json(self, fake_datus_home: Path) -> None:
        cache = fake_datus_home / "cache" / pmc.CACHE_FILE_NAME
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("{not-json")
        assert pmc.load_cached_models() is None
        assert pmc.load_cached_model_details() is None

    def test_load_filters_non_string_values(self, fake_datus_home: Path) -> None:
        _write_cache(
            fake_datus_home,
            {"openai": ["gpt-4o", 123, None, "gpt-4.1"]},  # type: ignore[list-item]
            version=1,
        )
        assert pmc.load_cached_models() == {"openai": ["gpt-4o", "gpt-4.1"]}

    def test_load_mixed_v2_ignores_invalid_entries(self, fake_datus_home: Path) -> None:
        _write_cache(
            fake_datus_home,
            {"openai": [{"id": "gpt-4o"}, {"id": ""}, {"no_id": True}, 42, {"id": "gpt-4.1"}]},
            version=2,
        )
        assert pmc.load_cached_models() == {"openai": ["gpt-4o", "gpt-4.1"]}

    def test_save_swallows_os_error(self, fake_datus_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_a: Any, **_kw: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _boom)
        pmc.save_cached_models({"openai": [{"id": "gpt-4o"}]})
        assert not (fake_datus_home / "cache" / pmc.CACHE_FILE_NAME).exists()


# ─────────────────────────────────────────────────────────────────────────────
# fetch_openrouter_models
# ─────────────────────────────────────────────────────────────────────────────


class TestFetchOpenrouterModels:
    def test_success_returns_bucketed_models(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.host == "openrouter.ai"
            assert req.headers.get("accept") == "application/json"
            assert "authorization" not in {k.lower() for k in req.headers.keys()}
            return httpx.Response(
                200,
                json=_payload(
                    "openai/gpt-4o",
                    "anthropic/claude-sonnet-4-5",
                    "unknown/foo",
                ),
            )

        with _install_mock_transport(handler):
            result = pmc.fetch_openrouter_models()

        assert _ids_only(result or {}) == {"openai": ["gpt-4o"], "claude": ["claude-sonnet-4-5"]}

    def test_success_with_allowed_providers_filters(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_payload("openai/gpt-4o", "anthropic/claude-sonnet-4-5", "deepseek/deepseek-chat"),
            )

        with _install_mock_transport(handler):
            result = pmc.fetch_openrouter_models(allowed_providers={"openai"})

        assert _ids_only(result or {}) == {"openai": ["gpt-4o"]}

    def test_timeout_returns_none(self) -> None:
        transport = _raising_transport(httpx.TimeoutException("slow"))
        original = httpx.Client

        def _factory(*a: Any, **kw: Any) -> httpx.Client:
            kw["transport"] = transport
            return original(*a, **kw)

        with patch.object(pmc.httpx, "Client", side_effect=_factory):
            assert pmc.fetch_openrouter_models() is None

    def test_http_500_returns_none(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        with _install_mock_transport(handler):
            assert pmc.fetch_openrouter_models() is None

    def test_http_429_returns_none(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="rate limited")

        with _install_mock_transport(handler):
            assert pmc.fetch_openrouter_models() is None

    def test_non_json_body_returns_none(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>cloudflare challenge</html>")

        with _install_mock_transport(handler):
            assert pmc.fetch_openrouter_models() is None

    def test_missing_data_field_returns_none(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"other": []})

        with _install_mock_transport(handler):
            assert pmc.fetch_openrouter_models() is None

    def test_empty_data_returns_none(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        with _install_mock_transport(handler):
            assert pmc.fetch_openrouter_models() is None

    def test_all_unknown_vendors_returns_none(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_payload("mystery/m1", "unknown/x"))

        with _install_mock_transport(handler):
            assert pmc.fetch_openrouter_models() is None

    def test_request_error_returns_none(self) -> None:
        transport = _raising_transport(httpx.ConnectError("dns fail"))
        original = httpx.Client

        def _factory(*a: Any, **kw: Any) -> httpx.Client:
            kw["transport"] = transport
            return original(*a, **kw)

        with patch.object(pmc.httpx, "Client", side_effect=_factory):
            assert pmc.fetch_openrouter_models() is None


# ─────────────────────────────────────────────────────────────────────────────
# resolve_provider_models (three-tier fallback)
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveProviderModels:
    def test_remote_success_overlays_models_and_writes_cache(self, fake_datus_home: Path) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_payload(
                    "openai/gpt-5.2",
                    "openai/gpt-4o",
                    "anthropic/claude-opus-4-6",
                    "deepseek/deepseek-chat",
                ),
            )

        local = _local_catalog()
        with _install_mock_transport(handler):
            merged = pmc.resolve_provider_models(local)

        assert merged["providers"]["openai"]["models"] == ["gpt-5.2", "gpt-4o"]
        assert merged["providers"]["claude"]["models"] == ["claude-opus-4-6"]
        assert merged["providers"]["deepseek"]["models"] == ["deepseek-chat"]
        # Cache file was written
        cache_path = fake_datus_home / "cache" / pmc.CACHE_FILE_NAME
        assert cache_path.exists()

    def test_preserves_non_models_fields(self, fake_datus_home: Path) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_payload("openai/gpt-5.2"))

        local = _local_catalog()
        with _install_mock_transport(handler):
            merged = pmc.resolve_provider_models(local)

        openai_entry = merged["providers"]["openai"]
        assert openai_entry["type"] == "openai"
        assert openai_entry["base_url"] == "https://api.openai.com/v1"
        assert openai_entry["api_key_env"] == "OPENAI_API_KEY"
        assert openai_entry["default_model"] == "gpt-4.1"
        # Top-level fields untouched.
        assert merged["model_overrides"] == {"kimi-k2.5": {"temperature": 1.0}}
        assert merged["model_specs"]["gpt-4.1"]["context_length"] == 400000

    def test_protected_providers_keep_local_models(self, fake_datus_home: Path) -> None:
        """codex / claude_subscription must NOT be touched even if remote has matching vendor."""

        def handler(_req: httpx.Request) -> httpx.Response:
            # Remote returns a model under anthropic/, which would normally overlay `claude`.
            # claude_subscription uses provider_key "claude_subscription" (not "claude"),
            # so it must remain untouched regardless.
            return httpx.Response(200, json=_payload("anthropic/claude-opus-4-6"))

        local = _local_catalog()
        with _install_mock_transport(handler):
            merged = pmc.resolve_provider_models(local)

        assert merged["providers"]["codex"]["models"] == ["codex-mini-latest"]
        assert merged["providers"]["claude_subscription"]["models"] == ["claude-sonnet-4-6"]

    def test_cache_written_is_v2_with_metadata(self, fake_datus_home: Path) -> None:
        """The freshly-written cache must carry pricing/context_length for billing."""

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "openai/gpt-4o",
                            "name": "GPT-4o",
                            "context_length": 128000,
                            "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
                        }
                    ]
                },
            )

        local = _local_catalog()
        with _install_mock_transport(handler):
            pmc.resolve_provider_models(local)

        raw = json.loads((fake_datus_home / "cache" / pmc.CACHE_FILE_NAME).read_text())
        assert raw["version"] == 2
        assert raw["models"]["openai"][0] == {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "context_length": 128000,
            "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
        }

    def test_falls_back_to_cache_when_remote_fails(self, fake_datus_home: Path) -> None:
        _write_cache(fake_datus_home, {"openai": [{"id": "cached-gpt-x"}]}, version=2)
        transport = _raising_transport(httpx.ConnectError("offline"))
        original = httpx.Client

        def _factory(*a: Any, **kw: Any) -> httpx.Client:
            kw["transport"] = transport
            return original(*a, **kw)

        local = _local_catalog()
        with patch.object(pmc.httpx, "Client", side_effect=_factory):
            merged = pmc.resolve_provider_models(local)

        assert merged["providers"]["openai"]["models"] == ["cached-gpt-x"]
        # claude has no cache entry → keep local
        assert merged["providers"]["claude"]["models"] == ["claude-sonnet-4-5"]

    def test_cache_fallback_drops_vendors_absent_from_providers_yml(self, fake_datus_home: Path) -> None:
        """A stale cache containing a provider that was later removed from
        providers.yml must not bleed back into the overlay."""
        _write_cache(
            fake_datus_home,
            {
                "openai": [{"id": "cached-gpt-x"}],
                "removed_provider": [{"id": "ghost-model"}],
            },
            version=2,
        )
        transport = _raising_transport(httpx.ConnectError("offline"))
        original = httpx.Client

        def _factory(*a: Any, **kw: Any) -> httpx.Client:
            kw["transport"] = transport
            return original(*a, **kw)

        local = _local_catalog()
        with patch.object(pmc.httpx, "Client", side_effect=_factory):
            merged = pmc.resolve_provider_models(local)

        # Verified provider (in YAML) is overlaid, ghost provider is dropped.
        assert merged["providers"]["openai"]["models"] == ["cached-gpt-x"]
        assert "removed_provider" not in merged["providers"]

    def test_falls_back_to_local_when_no_cache_and_remote_fails(self, fake_datus_home: Path) -> None:
        transport = _raising_transport(httpx.ConnectError("offline"))
        original = httpx.Client

        def _factory(*a: Any, **kw: Any) -> httpx.Client:
            kw["transport"] = transport
            return original(*a, **kw)

        local = _local_catalog()
        with patch.object(pmc.httpx, "Client", side_effect=_factory):
            merged = pmc.resolve_provider_models(local)

        # Identical to local.
        assert merged == local

    def test_falls_back_to_local_on_corrupt_cache(self, fake_datus_home: Path) -> None:
        cache = fake_datus_home / "cache" / pmc.CACHE_FILE_NAME
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("{bad json")

        transport = _raising_transport(httpx.TimeoutException("slow"))
        original = httpx.Client

        def _factory(*a: Any, **kw: Any) -> httpx.Client:
            kw["transport"] = transport
            return original(*a, **kw)

        local = _local_catalog()
        with patch.object(pmc.httpx, "Client", side_effect=_factory):
            merged = pmc.resolve_provider_models(local)

        assert merged == local

    def test_remote_empty_provider_keeps_local(self, fake_datus_home: Path) -> None:
        """If remote returns only openai models, claude should keep its local list."""

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_payload("openai/gpt-5.2"))

        local = _local_catalog()
        with _install_mock_transport(handler):
            merged = pmc.resolve_provider_models(local)

        assert merged["providers"]["openai"]["models"] == ["gpt-5.2"]
        assert merged["providers"]["claude"]["models"] == ["claude-sonnet-4-5"]
        assert merged["providers"]["deepseek"]["models"] == ["deepseek-chat"]

    def test_does_not_mutate_input_catalog(self, fake_datus_home: Path) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_payload("openai/gpt-5.2"))

        local = _local_catalog()
        snapshot_models = list(local["providers"]["openai"]["models"])

        with _install_mock_transport(handler):
            pmc.resolve_provider_models(local)

        assert local["providers"]["openai"]["models"] == snapshot_models


# ─────────────────────────────────────────────────────────────────────────────
# Silence contract: never WARN/ERROR/console on any failure path
# ─────────────────────────────────────────────────────────────────────────────


class TestSilenceContract:
    def test_fetch_timeout_only_emits_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        transport = _raising_transport(httpx.TimeoutException("slow"))
        original = httpx.Client

        def _factory(*a: Any, **kw: Any) -> httpx.Client:
            kw["transport"] = transport
            return original(*a, **kw)

        caplog.set_level(logging.DEBUG, logger="datus.cli.provider_model_catalog")
        with patch.object(pmc.httpx, "Client", side_effect=_factory):
            assert pmc.fetch_openrouter_models() is None

        high_severity = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert high_severity == []

    def test_resolve_on_triple_failure_only_emits_debug(
        self, fake_datus_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        transport = _raising_transport(httpx.ConnectError("offline"))
        original = httpx.Client

        def _factory(*a: Any, **kw: Any) -> httpx.Client:
            kw["transport"] = transport
            return original(*a, **kw)

        caplog.set_level(logging.DEBUG, logger="datus.cli.provider_model_catalog")
        with patch.object(pmc.httpx, "Client", side_effect=_factory):
            pmc.resolve_provider_models(_local_catalog())

        high_severity = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert high_severity == []


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_models_from: inherit parent provider's model list
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveModelsFrom:
    def test_copies_parent_models(self) -> None:
        catalog = {
            "providers": {
                "qwen": {"models": ["qwen3-max", "qwen3-coder-plus"]},
                "alibaba_coding": {
                    "models_from": "qwen",
                    "default_model": "qwen3-coder-plus",
                    "models": ["qwen3-coder-plus"],
                },
            }
        }
        result = pmc._resolve_models_from(catalog)
        assert result["providers"]["alibaba_coding"]["models"] == ["qwen3-max", "qwen3-coder-plus"]
        assert result["providers"]["alibaba_coding"]["default_model"] == "qwen3-coder-plus"

    def test_unknown_parent_keeps_static(self) -> None:
        catalog = {
            "providers": {
                "alibaba_coding": {
                    "models_from": "nonexistent",
                    "models": ["qwen3-coder-plus"],
                },
            }
        }
        result = pmc._resolve_models_from(catalog)
        assert result["providers"]["alibaba_coding"]["models"] == ["qwen3-coder-plus"]

    def test_empty_parent_keeps_static(self) -> None:
        catalog = {
            "providers": {
                "qwen": {"models": []},
                "alibaba_coding": {
                    "models_from": "qwen",
                    "models": ["qwen3-coder-plus"],
                },
            }
        }
        result = pmc._resolve_models_from(catalog)
        assert result["providers"]["alibaba_coding"]["models"] == ["qwen3-coder-plus"]

    def test_no_models_from_is_noop(self) -> None:
        catalog = {"providers": {"openai": {"models": ["gpt-4.1"]}}}
        result = pmc._resolve_models_from(catalog)
        assert result["providers"]["openai"]["models"] == ["gpt-4.1"]

    def test_no_shared_reference(self) -> None:
        catalog = {
            "providers": {
                "qwen": {"models": ["qwen3-max"]},
                "alibaba_coding": {"models_from": "qwen", "models": ["old"]},
            }
        }
        result = pmc._resolve_models_from(catalog)
        result["providers"]["alibaba_coding"]["models"].append("extra")
        assert "extra" not in result["providers"]["qwen"]["models"]

    def test_resolve_after_remote_overlay(self, fake_datus_home: Path) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_payload("qwen/qwen3-max", "qwen/qwen3-new"))

        local: Dict[str, Any] = {
            "providers": {
                "qwen": {"type": "openai", "models": ["qwen3-max"]},
                "alibaba_coding": {
                    "type": "claude",
                    "models_from": "qwen",
                    "default_model": "qwen3-coder-plus",
                    "models": ["qwen3-coder-plus"],
                },
            },
        }
        with _install_mock_transport(handler):
            merged = pmc.resolve_provider_models(local)
        assert merged["providers"]["qwen"]["models"] == ["qwen3-max", "qwen3-new"]
        assert merged["providers"]["alibaba_coding"]["models"] == ["qwen3-max", "qwen3-new"]
        assert merged["providers"]["alibaba_coding"]["default_model"] == "qwen3-coder-plus"

    def test_l3_fallback_still_resolves_models_from(self, fake_datus_home: Path) -> None:
        transport = _raising_transport(httpx.ConnectError("offline"))
        original = httpx.Client

        def _factory(*a: Any, **kw: Any) -> httpx.Client:
            kw["transport"] = transport
            return original(*a, **kw)

        local: Dict[str, Any] = {
            "providers": {
                "openai": {"type": "openai", "models": ["gpt-4.1", "o3"]},
                "codex": {
                    "type": "codex",
                    "models_from": "openai",
                    "default_model": "codex-mini-latest",
                    "models": ["codex-mini-latest"],
                },
            },
        }
        with patch.object(pmc.httpx, "Client", side_effect=_factory):
            merged = pmc.resolve_provider_models(local)
        assert merged["providers"]["codex"]["models"] == ["gpt-4.1", "o3"]
        assert merged["providers"]["codex"]["default_model"] == "codex-mini-latest"
