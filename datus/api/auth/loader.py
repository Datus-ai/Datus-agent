"""Dynamic AuthProvider loader.

Resolves an ``AuthProvider`` implementation declared in ``agent.yml`` under the
``api.auth_provider`` section. Falls back to :class:`NoAuthProvider` when no
provider is configured.
"""

import importlib
from typing import Any, Dict, Optional

from datus.api.auth.no_auth_provider import NoAuthProvider
from datus.api.auth.provider import AuthProvider
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def load_auth_provider(api_config: Optional[Dict[str, Any]], namespace: str) -> AuthProvider:
    """Load an AuthProvider instance from the ``api.auth_provider`` config section.

    Args:
        api_config: The ``api`` section dict from agent.yml (may be ``None``).
        namespace: Default namespace passed to ``NoAuthProvider`` when no custom
            provider is declared.

    Returns:
        An ``AuthProvider`` instance — either the custom one declared in config
        or the default :class:`NoAuthProvider`.
    """
    spec = (api_config or {}).get("auth_provider") or {}
    class_path = spec.get("class")
    if not class_path:
        return NoAuthProvider()

    normalized = class_path.replace(":", ".")
    module_name, _, class_name = normalized.rpartition(".")
    if not module_name or not class_name:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"Invalid auth_provider class path: {class_path!r}. Expected 'module.Class' or 'module:Class'.",
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"Failed to import auth_provider module {module_name!r}: {e}",
        ) from e

    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"Auth provider class {class_name!r} not found in module {module_name!r}",
        ) from e

    kwargs = spec.get("kwargs") or {}
    try:
        instance = cls(**kwargs)
    except Exception as e:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"Failed to instantiate auth_provider {class_path!r}: {e}",
        ) from e

    if not isinstance(instance, AuthProvider):
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"{class_path} does not implement the AuthProvider protocol",
        )

    logger.info(f"Loaded custom AuthProvider: {class_path}")
    return instance
