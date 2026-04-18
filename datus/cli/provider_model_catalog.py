# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Overlay provider model lists from OpenRouter's public catalog.

Used by `datus configure` to present up-to-date model choices without shipping
an ever-expanding local list. Failures silently fall back through:

  L1 Remote GET https://openrouter.ai/api/v1/models  (8s timeout, no auth)
  L2 Cached file at ~/.datus/cache/openrouter_models.json
  L3 Local catalog from datus/conf/providers.yml

The helper never raises and never writes to the console; only `logger.debug`
lines are emitted on failure paths.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_TIMEOUT_SEC = 8.0
CACHE_FILE_NAME = "openrouter_models.json"
CACHE_SCHEMA_VERSION = 1

# OpenRouter prefixes every model id as "<vendor>/<slug>". Map the vendor half
# onto the provider keys used in datus/conf/providers.yml.
OPENROUTER_VENDOR_MAP: Dict[str, str] = {
    "openai": "openai",
    "anthropic": "claude",
    "deepseek": "deepseek",
    "moonshotai": "kimi",
    "moonshot": "kimi",
    "qwen": "qwen",
    "alibaba": "qwen",
    "google": "gemini",
    "minimax": "minimax",
    "z-ai": "glm",
    "zhipuai": "glm",
    "thudm": "glm",
}

# These providers route through vendor-specific gateways whose model SKUs are
# not exposed in the public OpenRouter catalog. Keep their local `models` list.
PROTECTED_PROVIDERS = frozenset(
    {
        "alibaba_coding",
        "glm_coding",
        "minimax_coding",
        "kimi_coding",
        "claude_subscription",
        "codex",
    }
)


def _cache_file_path() -> Path:
    return get_path_manager().datus_home / "cache" / CACHE_FILE_NAME


def _bucket_by_vendor(raw_models: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group OpenRouter models into provider buckets, preserving first-seen order."""
    buckets: Dict[str, Dict[str, None]] = {}
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or "/" not in model_id:
            continue
        vendor, _, slug = model_id.partition("/")
        slug = slug.strip()
        if not slug:
            continue
        provider_key = OPENROUTER_VENDOR_MAP.get(vendor.strip().lower())
        if provider_key is None:
            continue
        buckets.setdefault(provider_key, {})[slug] = None
    return {k: list(v.keys()) for k, v in buckets.items()}


def fetch_openrouter_models(timeout: float = OPENROUTER_TIMEOUT_SEC) -> Optional[Dict[str, List[str]]]:
    """Fetch and bucket the OpenRouter model catalog. Returns None on any failure."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(OPENROUTER_MODELS_URL, headers={"Accept": "application/json"})
            resp.raise_for_status()
            payload = resp.json()
    except httpx.TimeoutException:
        logger.debug("openrouter fetch timeout")
        return None
    except httpx.HTTPStatusError as e:
        logger.debug(f"openrouter fetch http error: {e.response.status_code}")
        return None
    except httpx.RequestError as e:
        logger.debug(f"openrouter fetch request error: {e}")
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug(f"openrouter fetch decode error: {e}")
        return None
    except Exception as e:
        logger.debug(f"openrouter fetch unexpected error: {e}")
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        logger.debug("openrouter payload missing 'data' list")
        return None

    buckets = _bucket_by_vendor(data)
    if not buckets:
        logger.debug("openrouter payload produced no matching vendor buckets")
        return None
    return buckets


def load_cached_models() -> Optional[Dict[str, List[str]]]:
    """Return the cached `models` subdict, or None if missing/invalid."""
    path = _cache_file_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug(f"openrouter cache read error: {e}")
        return None

    if not isinstance(raw, dict) or raw.get("version") != CACHE_SCHEMA_VERSION:
        logger.debug("openrouter cache schema mismatch")
        return None
    models = raw.get("models")
    if not isinstance(models, dict):
        logger.debug("openrouter cache models not a dict")
        return None

    result: Dict[str, List[str]] = {}
    for provider_key, value in models.items():
        if not isinstance(provider_key, str) or not isinstance(value, list):
            continue
        result[provider_key] = [m for m in value if isinstance(m, str)]
    return result or None


def save_cached_models(buckets: Dict[str, List[str]]) -> None:
    """Atomically persist the bucketed model list. Swallows I/O errors."""
    path = _cache_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_SCHEMA_VERSION,
            "source": "openrouter",
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "models": buckets,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError as e:
        logger.debug(f"openrouter cache write error: {e}")
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:
            pass


def _overlay(local_catalog: dict, buckets: Dict[str, List[str]]) -> dict:
    """Return a new catalog with each non-protected provider's `models` replaced."""
    merged = copy.deepcopy(local_catalog)
    providers = merged.get("providers")
    if not isinstance(providers, dict):
        return merged
    for provider_key, provider_info in providers.items():
        if not isinstance(provider_info, dict):
            continue
        if provider_key in PROTECTED_PROVIDERS:
            continue
        remote_models = buckets.get(provider_key)
        if remote_models:
            provider_info["models"] = list(remote_models)
    return merged


def resolve_provider_models(local_catalog: dict) -> dict:
    """Three-tier fallback: remote → cache → local. Never raises.

    Only overlays per-provider `models` lists; `model_overrides`, `model_specs`,
    `default_model`, `base_url`, `api_key_env`, `type`, `auth_type` are preserved.
    PROTECTED_PROVIDERS keep their local `models` regardless of remote content.
    """
    remote_buckets = fetch_openrouter_models()
    if remote_buckets is not None:
        save_cached_models(remote_buckets)
        return _overlay(local_catalog, remote_buckets)

    cached_buckets = load_cached_models()
    if cached_buckets is not None:
        return _overlay(local_catalog, cached_buckets)

    return local_catalog
