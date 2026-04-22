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

Cache schema v2 stores the full metadata (name, context_length, pricing) so
downstream billing/statistics tools can read it without re-hitting OpenRouter.
v1 caches (flat slug lists) are still accepted via a one-way upgrade path.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import httpx

from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_TIMEOUT_SEC = 8.0
CACHE_FILE_NAME = "openrouter_models.json"
CACHE_SCHEMA_VERSION = 2
# Cache versions that load_cached_models / load_cached_model_details accept.
_SUPPORTED_CACHE_VERSIONS = frozenset({1, 2})

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

_NON_CHAT_SLUG_FRAGMENTS = frozenset(
    {
        "embed",
        "dall-e",
        "tts",
        "whisper",
        "moderat",
        "realtime",
        "transcri",
        "speech",
        "audio",
        "lyria",
    }
)
_MIN_CONTEXT_LENGTH = 4096


def _cache_file_path() -> Path:
    return get_path_manager().datus_home / "cache" / CACHE_FILE_NAME


def _extract_pricing(raw_pricing: Any) -> Optional[Dict[str, str]]:
    """Keep only input/output token prices; drop noisy per-request fields."""
    if not isinstance(raw_pricing, dict):
        return None
    out: Dict[str, str] = {}
    for key in ("prompt", "completion"):
        value = raw_pricing.get(key)
        if isinstance(value, (str, int, float)):
            out[key] = str(value)
    return out or None


def _extract_context_length(item: Dict[str, Any]) -> Optional[int]:
    """OpenRouter exposes context_length at the top level and under top_provider."""
    raw = item.get("context_length")
    if isinstance(raw, int):
        return raw
    top = item.get("top_provider")
    if isinstance(top, dict):
        top_ctx = top.get("context_length")
        if isinstance(top_ctx, int):
            return top_ctx
    return None


def _is_model_unfit(slug: str, context_length: Optional[int]) -> bool:
    if ":" in slug:
        return True
    slug_lower = slug.lower()
    if any(frag in slug_lower for frag in _NON_CHAT_SLUG_FRAGMENTS):
        return True
    if context_length is not None and context_length < _MIN_CONTEXT_LENGTH:
        return True
    return False


def _bucket_by_vendor(
    raw_models: List[Dict[str, Any]],
    allowed_providers: Optional[Iterable[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group OpenRouter models into provider buckets, preserving first-seen order.

    When ``allowed_providers`` is given, only provider keys present in that set
    are kept — this lets providers.yml act as the single whitelist.
    """
    allowed: Optional[Set[str]] = set(allowed_providers) if allowed_providers is not None else None
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {}
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
        if allowed is not None and provider_key not in allowed:
            continue
        ctx_len = _extract_context_length(item)
        if _is_model_unfit(slug, ctx_len):
            continue
        if provider_key == "claude":
            slug = slug.replace(".", "-")
        entry: Dict[str, Any] = {"id": slug}
        name = item.get("name")
        if isinstance(name, str) and name:
            entry["name"] = name
        if ctx_len is not None:
            entry["context_length"] = ctx_len
        pricing = _extract_pricing(item.get("pricing"))
        if pricing:
            entry["pricing"] = pricing
        buckets.setdefault(provider_key, {}).setdefault(slug, entry)
    return {k: list(v.values()) for k, v in buckets.items()}


def fetch_openrouter_models(
    timeout: float = OPENROUTER_TIMEOUT_SEC,
    allowed_providers: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
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

    buckets = _bucket_by_vendor(data, allowed_providers=allowed_providers)
    if not buckets:
        logger.debug("openrouter payload produced no matching vendor buckets")
        return None
    return buckets


def _read_cache_file() -> Optional[Dict[str, Any]]:
    """Read and shallowly validate the cache file. Returns None on any problem."""
    path = _cache_file_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug(f"openrouter cache read error: {e}")
        return None

    if not isinstance(raw, dict):
        logger.debug("openrouter cache schema mismatch")
        return None
    version = raw.get("version", 1)
    if version not in _SUPPORTED_CACHE_VERSIONS:
        logger.debug("openrouter cache schema mismatch")
        return None
    models = raw.get("models")
    if not isinstance(models, dict):
        logger.debug("openrouter cache models not a dict")
        return None
    return raw


def load_cached_models() -> Optional[Dict[str, List[str]]]:
    """Return the cached model slugs as ``Dict[provider, List[slug]]``.

    Accepts both v1 (list of strings) and v2 (list of dicts) caches. v2 entries
    contribute their ``id`` field. Returns None when the cache is missing,
    corrupt, or produces no valid slugs.
    """
    raw = _read_cache_file()
    if raw is None:
        return None

    result: Dict[str, List[str]] = {}
    for provider_key, value in raw["models"].items():
        if not isinstance(provider_key, str) or not isinstance(value, list):
            continue
        slugs: List[str] = []
        for entry in value:
            if isinstance(entry, str):
                slugs.append(entry)
            elif isinstance(entry, dict):
                slug = entry.get("id")
                if isinstance(slug, str) and slug:
                    slugs.append(slug)
        if slugs:
            result[provider_key] = slugs
    return result or None


def load_cached_model_details() -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Return the full per-model metadata from the cache, upgrading v1 on the fly.

    v1 caches (list of slug strings) are promoted to ``[{"id": slug}]`` so
    downstream consumers see a uniform shape. Returns None when the cache is
    missing or invalid.
    """
    raw = _read_cache_file()
    if raw is None:
        return None

    result: Dict[str, List[Dict[str, Any]]] = {}
    for provider_key, value in raw["models"].items():
        if not isinstance(provider_key, str) or not isinstance(value, list):
            continue
        entries: List[Dict[str, Any]] = []
        for entry in value:
            if isinstance(entry, str):
                if entry:
                    entries.append({"id": entry})
            elif isinstance(entry, dict):
                slug = entry.get("id")
                if isinstance(slug, str) and slug:
                    entries.append(dict(entry))
        if entries:
            result[provider_key] = entries
    return result or None


def load_cache_fetched_at() -> Optional[str]:
    """Return the cache file's ISO-8601 ``fetched_at`` timestamp, or None."""
    raw = _read_cache_file()
    if raw is None:
        return None
    value = raw.get("fetched_at")
    return value if isinstance(value, str) else None


def save_cached_models(buckets: Dict[str, List[Dict[str, Any]]]) -> None:
    """Atomically persist the bucketed model list as schema v2. Swallows I/O errors."""
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


def _overlay(local_catalog: dict, buckets: Dict[str, List[Dict[str, Any]]]) -> dict:
    """Return a new catalog with each non-protected provider's `models` replaced.

    The overlay only writes slug lists into ``providers.yml`` consumers — the
    richer per-model metadata stays in the cache file.
    """
    merged = copy.deepcopy(local_catalog)
    providers = merged.get("providers")
    if not isinstance(providers, dict):
        return merged
    for provider_key, provider_info in providers.items():
        if not isinstance(provider_info, dict):
            continue
        if provider_key in PROTECTED_PROVIDERS:
            continue
        remote_entries = buckets.get(provider_key)
        if remote_entries:
            provider_info["models"] = [entry["id"] for entry in remote_entries if "id" in entry]
    return merged


def _resolve_models_from(catalog: dict) -> dict:
    """Copy parent provider's models into children that declare ``models_from``."""
    providers = catalog.get("providers")
    if not isinstance(providers, dict):
        return catalog
    for _provider_key, provider_info in providers.items():
        if not isinstance(provider_info, dict):
            continue
        parent_key = provider_info.get("models_from")
        if not isinstance(parent_key, str) or not parent_key:
            continue
        parent_info = providers.get(parent_key)
        if not isinstance(parent_info, dict):
            continue
        parent_models = parent_info.get("models")
        if not isinstance(parent_models, list) or not parent_models:
            continue
        provider_info["models"] = list(parent_models)
    return catalog


def resolve_provider_models(local_catalog: dict) -> dict:
    """Three-tier fallback: remote → cache → local. Never raises.

    Only overlays per-provider `models` lists; `model_overrides`, `model_specs`,
    `default_model`, `base_url`, `api_key_env`, `type`, `auth_type` are preserved.
    PROTECTED_PROVIDERS keep their local `models` regardless of remote content.

    Providers not declared in ``providers.yml`` are dropped at fetch time — the
    YAML is the single source of truth for which vendors are considered verified.
    """
    providers_block = local_catalog.get("providers") if isinstance(local_catalog, dict) else None
    allowed_providers = set(providers_block.keys()) if isinstance(providers_block, dict) else None

    remote_buckets = fetch_openrouter_models(allowed_providers=allowed_providers)
    if remote_buckets is not None:
        save_cached_models(remote_buckets)
        return _resolve_models_from(_overlay(local_catalog, remote_buckets))

    cached_details = load_cached_model_details()
    if cached_details is not None:
        if allowed_providers is not None:
            cached_details = {k: v for k, v in cached_details.items() if k in allowed_providers}
        if cached_details:
            return _resolve_models_from(_overlay(local_catalog, cached_details))

    return _resolve_models_from(copy.deepcopy(local_catalog))
