"""Dynamic model-catalog resolver for the OpenCode Go subscription."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)

OPENCODE_GO_PROVIDER_KEY = "opencode_go"
OPENCODE_GO_MODELS_URL = "https://opencode.ai/zen/go/v1/models"
OPENCODE_GO_TIMEOUT_SEC = 8.0
OPENCODE_GO_CACHE_TTL_SEC = 600.0
OPENCODE_GO_CACHE_FILE = "opencode_go_models.json"
OPENCODE_GO_CACHE_VERSION = 1


def _cache_file_path() -> Path:
    return get_path_manager().datus_home / "cache" / OPENCODE_GO_CACHE_FILE


def _cache_is_fresh(max_age_sec: float = OPENCODE_GO_CACHE_TTL_SEC) -> bool:
    try:
        modified_at = _cache_file_path().stat().st_mtime
    except OSError:
        return False
    return (time.time() - modified_at) < max_age_sec


def _supported_models(catalog: dict[str, Any]) -> list[str]:
    providers = catalog.get("providers")
    if not isinstance(providers, dict):
        return []
    provider = providers.get(OPENCODE_GO_PROVIDER_KEY)
    if not isinstance(provider, dict):
        return []
    models = provider.get("models")
    if not isinstance(models, list):
        return []
    return [item for item in models if isinstance(item, str) and item]


def _filter_supported_models(payload: Any, supported_models: Iterable[str]) -> list[str] | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None

    remote_ids = {item.get("id") for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)}
    filtered = [model for model in supported_models if model in remote_ids]
    return filtered or None


def fetch_opencode_go_models(
    supported_models: Iterable[str],
    timeout: float = OPENCODE_GO_TIMEOUT_SEC,
) -> list[str] | None:
    """Return currently available supported models, or None on any failure."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(
                OPENCODE_GO_MODELS_URL,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            return _filter_supported_models(response.json(), supported_models)
    except httpx.TimeoutException:
        logger.debug("OpenCode Go model-catalog request timed out")
    except httpx.HTTPStatusError as exc:
        logger.debug("OpenCode Go model-catalog HTTP error: %s", exc.response.status_code)
    except httpx.RequestError as exc:
        logger.debug("OpenCode Go model-catalog request error: %s", exc)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.debug("OpenCode Go model-catalog decode error: %s", exc)
    except Exception as exc:  # noqa: BLE001 - catalog lookup must degrade safely
        logger.debug("OpenCode Go model-catalog unexpected error: %s", exc)
    return None


def load_cached_opencode_go_models(supported_models: Iterable[str]) -> list[str] | None:
    path = _cache_file_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(raw, dict):
        return None
    if raw.get("version") != OPENCODE_GO_CACHE_VERSION:
        return None
    if raw.get("source") != OPENCODE_GO_PROVIDER_KEY:
        return None

    cached = raw.get("models")
    if not isinstance(cached, list):
        return None
    supported = set(supported_models)
    filtered = [item for item in cached if isinstance(item, str) and item in supported]
    return filtered or None


def save_cached_opencode_go_models(models: list[str]) -> None:
    path = _cache_file_path()
    temporary_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": OPENCODE_GO_CACHE_VERSION,
            "source": OPENCODE_GO_PROVIDER_KEY,
            "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "models": models,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except OSError as exc:
        logger.debug("OpenCode Go model-cache write error: %s", exc)
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


def _overlay_models(catalog: dict[str, Any], models: list[str]) -> dict[str, Any]:
    merged = copy.deepcopy(catalog)
    providers = merged.get("providers")
    if not isinstance(providers, dict):
        return merged
    provider = providers.get(OPENCODE_GO_PROVIDER_KEY)
    if isinstance(provider, dict) and models:
        provider["models"] = list(models)
    return merged


def resolve_opencode_go_models(local_catalog: dict[str, Any]) -> dict[str, Any]:
    """Resolve fresh cache → remote → stale cache → local allowlist."""
    catalog = copy.deepcopy(local_catalog)
    supported = _supported_models(catalog)
    if not supported:
        return catalog

    if _cache_is_fresh():
        cached = load_cached_opencode_go_models(supported)
        if cached:
            return _overlay_models(catalog, cached)

    remote = fetch_opencode_go_models(supported)
    if remote:
        save_cached_opencode_go_models(remote)
        return _overlay_models(catalog, remote)

    cached = load_cached_opencode_go_models(supported)
    if cached:
        return _overlay_models(catalog, cached)

    return catalog
