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


def _write_cache(tmp_home: Path, models: Dict[str, List[str]], *, version: int = 1) -> Path:
    cache = tmp_home / "cache" / pmc.CACHE_FILE_NAME
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "version": version,
                "source": "openrouter",
                "fetched_at": "2026-04-18T00:00:00Z",
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# _bucket_by_vendor
# ─────────────────────────────────────────────────────────────────────────────


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
        assert buckets == {expected_provider: [expected_slug]}

    def test_unknown_vendor_is_dropped(self) -> None:
        assert pmc._bucket_by_vendor([{"id": "mystery/foo-1"}]) == {}

    def test_vendor_match_is_case_insensitive(self) -> None:
        buckets = pmc._bucket_by_vendor([{"id": "OpenAI/gpt-4o"}])
        assert buckets == {"openai": ["gpt-4o"]}

    def test_slug_preserves_order_and_deduplicates(self) -> None:
        buckets = pmc._bucket_by_vendor(
            [
                {"id": "openai/gpt-4o"},
                {"id": "openai/gpt-4.1"},
                {"id": "openai/gpt-4o"},  # duplicate
                {"id": "openai/o3"},
            ]
        )
        assert buckets == {"openai": ["gpt-4o", "gpt-4.1", "o3"]}

    def test_id_without_slash_is_dropped(self) -> None:
        assert pmc._bucket_by_vendor([{"id": "gpt-4o"}]) == {}

    def test_empty_slug_is_dropped(self) -> None:
        assert pmc._bucket_by_vendor([{"id": "openai/"}]) == {}

    def test_non_dict_entries_are_skipped(self) -> None:
        assert pmc._bucket_by_vendor(["openai/gpt-4o", None, 42]) == {}  # type: ignore[list-item]

    def test_missing_id_field_is_skipped(self) -> None:
        assert pmc._bucket_by_vendor([{"name": "No id here"}]) == {}


# ─────────────────────────────────────────────────────────────────────────────
# Cache I/O
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheIO:
    def test_load_cached_models_returns_none_when_file_missing(self, fake_datus_home: Path) -> None:
        assert pmc.load_cached_models() is None

    def test_save_then_load_roundtrip(self, fake_datus_home: Path) -> None:
        buckets = {"openai": ["gpt-4o", "gpt-4.1"], "claude": ["claude-sonnet-4-5"]}
        pmc.save_cached_models(buckets)

        cache_path = fake_datus_home / "cache" / pmc.CACHE_FILE_NAME
        assert cache_path.exists()
        loaded = pmc.load_cached_models()
        assert loaded == buckets

    def test_save_is_atomic_no_leftover_tmp(self, fake_datus_home: Path) -> None:
        pmc.save_cached_models({"openai": ["gpt-4o"]})
        tmp = fake_datus_home / "cache" / (pmc.CACHE_FILE_NAME + ".tmp")
        assert not tmp.exists()

    def test_cache_file_contains_version_and_fetched_at(self, fake_datus_home: Path) -> None:
        pmc.save_cached_models({"openai": ["gpt-4o"]})
        raw = json.loads((fake_datus_home / "cache" / pmc.CACHE_FILE_NAME).read_text())
        assert raw["version"] == pmc.CACHE_SCHEMA_VERSION
        assert raw["source"] == "openrouter"
        assert raw["fetched_at"].endswith("Z")
        assert raw["models"] == {"openai": ["gpt-4o"]}

    def test_load_rejects_wrong_version(self, fake_datus_home: Path) -> None:
        _write_cache(fake_datus_home, {"openai": ["gpt-4o"]}, version=99)
        assert pmc.load_cached_models() is None

    def test_load_rejects_missing_models_key(self, fake_datus_home: Path) -> None:
        cache = fake_datus_home / "cache" / pmc.CACHE_FILE_NAME
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"version": 1, "source": "openrouter"}))
        assert pmc.load_cached_models() is None

    def test_load_rejects_corrupt_json(self, fake_datus_home: Path) -> None:
        cache = fake_datus_home / "cache" / pmc.CACHE_FILE_NAME
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("{not-json")
        assert pmc.load_cached_models() is None

    def test_load_filters_non_string_values(self, fake_datus_home: Path) -> None:
        _write_cache(
            fake_datus_home,
            {"openai": ["gpt-4o", 123, None, "gpt-4.1"]},  # type: ignore[list-item]
        )
        assert pmc.load_cached_models() == {"openai": ["gpt-4o", "gpt-4.1"]}

    def test_save_swallows_os_error(self, fake_datus_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_a: Any, **_kw: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _boom)
        # Should not raise.
        pmc.save_cached_models({"openai": ["gpt-4o"]})


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

        assert result == {"openai": ["gpt-4o"], "claude": ["claude-sonnet-4-5"]}

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

    def test_falls_back_to_cache_when_remote_fails(self, fake_datus_home: Path) -> None:
        _write_cache(fake_datus_home, {"openai": ["cached-gpt-x"]})
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
