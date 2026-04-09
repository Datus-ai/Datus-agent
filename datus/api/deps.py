"""FastAPI dependency injection — plugin-based auth + DatusService cache."""

from typing import Annotated, Optional

from fastapi import Depends, Request

from datus.api.auth.context import AppContext
from datus.api.auth.provider import AuthProvider
from datus.api.services.datus_service import DatusService
from datus.api.services.datus_service_cache import DatusServiceCache
from datus.configuration.agent_config_loader import load_agent_config
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Module-level singletons (set during lifespan via init_deps)
_auth_provider: Optional[AuthProvider] = None
_service_cache: Optional[DatusServiceCache] = None
_namespace: str = "default"
_default_source: Optional[str] = None
_default_interactive: bool = True

_DEFAULT_PROJECT_KEY = "default"


def init_deps(
    auth_provider: AuthProvider,
    cache: DatusServiceCache,
    namespace: str = "default",
    default_source: Optional[str] = None,
    default_interactive: bool = True,
) -> None:
    """Initialize global auth provider and service cache.

    Called from main.py lifespan to inject dependencies.
    """
    global _auth_provider, _service_cache, _namespace, _default_source, _default_interactive
    _auth_provider = auth_provider
    _service_cache = cache
    _namespace = namespace
    _default_source = default_source
    _default_interactive = default_interactive
    # Wire eviction callback: auth config changes trigger cache eviction
    auth_provider.on_evict(cache.evict)


async def get_datus_service(request: Request) -> DatusService:
    """Primary dependency for all agent routes.

    Authenticates the request, caches the resulting ``AppContext`` on
    ``request.state`` for downstream dependencies (e.g. ``AppContextDep``),
    then returns a cached-per-project DatusService. If AppContext has no
    config, loads it on-demand from YAML.
    """
    if _auth_provider is None:
        raise RuntimeError("Auth provider not initialized. Call init_deps() in lifespan.")
    if _service_cache is None:
        raise RuntimeError("Service cache not initialized. Call init_deps() in lifespan.")

    ctx: AppContext = await _auth_provider.authenticate(request)
    request.state.app_context = ctx

    expected_fp = DatusService.compute_fingerprint(ctx.config) if ctx.config is not None else None
    cache_key = ctx.project_id or _DEFAULT_PROJECT_KEY

    async def _factory() -> DatusService:
        # Load config on-demand if not provided by auth provider
        agent_config = ctx.config
        if agent_config is None:
            try:
                agent_config = load_agent_config(namespace=_namespace)
            except Exception as e:
                logger.error(f"Failed to load agent config for namespace '{_namespace}': {e}")
                raise RuntimeError(f"Failed to load agent config: {e}") from e

        return DatusService(
            agent_config=agent_config,
            project_id=cache_key,
            default_source=_default_source,
            default_interactive=_default_interactive,
        )

    return await _service_cache.get_or_create(cache_key, _factory, expected_fingerprint=expected_fp)


def get_app_context(request: Request) -> AppContext:
    """Return the ``AppContext`` cached on the request by ``get_datus_service``.

    Must be used together with (and after) ``ServiceDep`` on the same route.
    """
    ctx = getattr(request.state, "app_context", None)
    if ctx is None:
        raise RuntimeError(
            "AppContext not found on request.state — ensure ServiceDep is declared before AppContextDep."
        )
    return ctx


ServiceDep = Annotated[DatusService, Depends(get_datus_service)]
AppContextDep = Annotated[AppContext, Depends(get_app_context)]
