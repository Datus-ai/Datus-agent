"""Default provider for request context carried in trusted headers."""

import json
from typing import Any

from fastapi import Request

from datus.api.auth.context import AppContext
from datus.api.auth.provider import EvictCallback
from datus.api.constants import HEADER_POLICY_CONTEXT, HEADER_USER_ID, USER_ID_PATTERN
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class HeaderContextProvider:
    """Read optional identity and execution-policy context headers.

    ``X-Datus-User-Id`` is optional caller identity for per-user session
    isolation. ``X-Datus-Policy-Context`` is an optional JSON object forwarded
    to active execution policies without interpreting user identity.

    Auth provider only handles identification, not config loading.
    Config is loaded on-demand by ``get_datus_service``.
    """

    def __init__(self) -> None:
        self._evict_callbacks: list[EvictCallback] = []

    async def authenticate(self, request: Request) -> AppContext:
        user_id = self._read_user_id(request)
        policy_context = self._read_policy_context(request)
        return AppContext(user_id=user_id, project_id=None, config=None, policy_context=policy_context)

    def on_evict(self, callback: EvictCallback) -> None:
        """Register an eviction callback for the provider lifecycle."""
        self._evict_callbacks.append(callback)

    @staticmethod
    def _read_user_id(request: Request) -> str | None:
        raw = request.headers.get(HEADER_USER_ID)
        if raw is None:
            return None
        candidate = raw.strip()
        if not candidate:
            return None
        if not USER_ID_PATTERN.match(candidate):
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=(
                    f"Invalid {HEADER_USER_ID} header value: {candidate!r}. "
                    "Only letters, digits, underscore and hyphen are allowed."
                ),
            )
        return candidate

    @staticmethod
    def _read_policy_context(request: Request) -> dict[str, Any]:
        raw = request.headers.get(HEADER_POLICY_CONTEXT)
        if raw is None or not raw.strip():
            return {}

        try:
            policy_context = json.loads(raw)
        except json.JSONDecodeError as e:
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=(
                    f"Invalid {HEADER_POLICY_CONTEXT} header value: expected a JSON object "
                    f"with policy inputs ({e.msg})."
                ),
            ) from e

        if not isinstance(policy_context, dict):
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=(f"Invalid {HEADER_POLICY_CONTEXT} header value: expected a JSON object with policy inputs."),
            )
        return policy_context
