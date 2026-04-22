"""API routes for the resolved LLM model catalog.

Surfaces the OpenRouter-derived model list, filtered down to the providers
whose credentials are actually configured in ``agent.yml``. The response
includes ``context_length`` and ``pricing`` so UIs can drive billing and
context-budget decisions without re-hitting OpenRouter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.config_models import ModelInfo, ModelPricing, ModelsData
from datus.cli.provider_model_catalog import load_cache_fetched_at, load_cached_model_details
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["models"])


def _model_spec(catalog: Dict[str, Any], slug: str) -> Dict[str, Any]:
    """Return the ``model_specs`` entry from providers.yml for a slug, or {}.

    Prefix matching mirrors the runtime convention documented in
    ``datus/conf/providers.yml:154-155``: the longest spec key that is a prefix
    of the slug wins. Exact matches always take priority.
    """
    specs = catalog.get("model_specs") if isinstance(catalog, dict) else None
    if not isinstance(specs, dict):
        return {}
    exact = specs.get(slug)
    if isinstance(exact, dict):
        return exact
    best_key: Optional[str] = None
    for key in specs.keys():
        if not isinstance(key, str):
            continue
        if slug.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    if best_key is None:
        return {}
    value = specs.get(best_key)
    return value if isinstance(value, dict) else {}


def _build_model_info(
    provider: str,
    entry: Dict[str, Any],
    catalog: Dict[str, Any],
) -> ModelInfo:
    """Merge cache metadata with providers.yml ``model_specs`` as a fallback."""
    slug = str(entry.get("id", ""))
    spec = _model_spec(catalog, slug)
    context_length = entry.get("context_length")
    if not isinstance(context_length, int):
        spec_ctx = spec.get("context_length")
        context_length = spec_ctx if isinstance(spec_ctx, int) else None
    spec_max = spec.get("max_tokens")
    max_tokens = spec_max if isinstance(spec_max, int) else None

    pricing_raw = entry.get("pricing")
    pricing = None
    if isinstance(pricing_raw, dict):
        pricing = ModelPricing(
            prompt=pricing_raw.get("prompt") if isinstance(pricing_raw.get("prompt"), str) else None,
            completion=pricing_raw.get("completion") if isinstance(pricing_raw.get("completion"), str) else None,
        )
        if pricing.prompt is None and pricing.completion is None:
            pricing = None

    name = entry.get("name") if isinstance(entry.get("name"), str) else None

    return ModelInfo(
        provider=provider,
        id=slug,
        name=name,
        context_length=context_length,
        max_tokens=max_tokens,
        pricing=pricing,
    )


@router.get(
    "/models",
    response_model=Result[ModelsData],
    summary="List Available Models",
    description="Return models for providers with credentials configured in agent.yml.",
)
async def list_models(svc: ServiceDep) -> Result[ModelsData]:
    """Return every model exposed by providers the current project has credentials for.

    Data priority per-model:
      1. OpenRouter cache (``~/.datus/cache/openrouter_models.json``) — richest.
      2. ``providers.yml`` model list with ``model_specs`` for context_length.
    """
    agent_config = svc.agent_config
    catalog = agent_config.provider_catalog if isinstance(agent_config.provider_catalog, dict) else {}
    providers_meta = catalog.get("providers", {}) if isinstance(catalog, dict) else {}
    if not isinstance(providers_meta, dict):
        providers_meta = {}

    cached_details = load_cached_model_details() or {}
    used_cache = False

    models: List[ModelInfo] = []
    seen_providers: List[str] = []

    for provider_key, meta in providers_meta.items():
        if not isinstance(provider_key, str) or not isinstance(meta, dict):
            continue
        if not agent_config.provider_available(provider_key):
            continue

        entries = cached_details.get(provider_key)
        provider_used_cache = bool(entries)
        if not entries:
            slugs = meta.get("models")
            if not isinstance(slugs, list):
                continue
            entries = [{"id": slug} for slug in slugs if isinstance(slug, str) and slug]

        if not entries:
            continue

        seen_providers.append(provider_key)
        used_cache = used_cache or provider_used_cache
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            slug = entry.get("id")
            if not isinstance(slug, str) or not slug:
                continue
            models.append(_build_model_info(provider_key, entry, catalog))

    return Result(
        success=True,
        data=ModelsData(
            models=models,
            providers=seen_providers,
            fetched_at=load_cache_fetched_at() if used_cache else None,
            source="cache" if used_cache else "catalog",
        ),
    )
