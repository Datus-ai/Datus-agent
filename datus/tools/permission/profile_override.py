# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Apply a permission profile to a single node's manager, in place.

Both callers face the same constraint: ``AgentConfig`` is shared by every
concurrent request in a SaaS deployment, so a per-request profile can only
live on the node's own ``PermissionManager`` — never on
``agent_config.active_profile_name``. That is also why a subagent, built from
the same shared config, has to be told explicitly what its parent is running
under (see ``SubAgentTaskTool._inherit_permission_profile``).
"""

from typing import Any, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def apply_profile_override(
    permission_manager: Any,
    agent_config: Any,
    profile_name: Optional[str],
    *,
    subject: str = "",
) -> bool:
    """Switch ``permission_manager`` to ``profile_name``.

    No-ops when ``profile_name`` is falsy, there is no manager (e.g. workflow
    nodes that skip the skill setup), or the profile is already active.
    Failure handling is split deliberately:

    * Building ``user_overrides`` from ``agent.yml`` fails closed — raises, so
      the caller aborts rather than applying the bare profile base, which can
      be **broader** than the operator-configured posture (an explicit DENY in
      the yaml would be lost).
    * ``switch_profile`` failures (unknown profile, malformed merge result)
      are logged and swallowed: the manager still has its original,
      server-default profile installed, which is the safe side to land on.

    Args:
        permission_manager: The node's ``PermissionManager``, or ``None``.
        agent_config: Source of the raw ``permissions`` block from agent.yml.
        profile_name: Target profile (``normal`` / ``auto`` / ``dangerous``).
        subject: Free-form identifier for the log line (session, node name).

    Returns:
        True when the profile actually changed.

    Raises:
        DatusException: ``COMMON_CONFIG_ERROR`` if agent.yml's user rules
            cannot be rebuilt — the same code ``switch_profile`` reports an
            unusable permission configuration with.
    """
    if not profile_name or permission_manager is None:
        return False
    if getattr(permission_manager, "active_profile", None) == profile_name:
        return False

    # Imported here, not at module scope, so the yaml-parsing failure path
    # stays patchable in tests.
    from datus.tools.permission.profiles import build_user_overrides

    raw_permissions = getattr(agent_config, "_raw_permissions", {}) or {}
    raw_user = {k: v for k, v in raw_permissions.items() if k != "profile"}
    try:
        user_overrides = build_user_overrides(profile_name, raw_user)
    except Exception as exc:
        logger.error(
            "Cannot build user overrides for permission_mode=%r from agent.yml: %s; "
            "refusing to switch profile to avoid broadening permissions beyond the "
            "operator-configured rules",
            profile_name,
            exc,
            exc_info=True,
        )
        from datus.utils.exceptions import DatusException, ErrorCode

        raise DatusException(
            code=ErrorCode.COMMON_CONFIG_ERROR,
            message_args={
                "config_error": (
                    f"cannot apply permission_mode={profile_name!r}: agent.yml permissions.rules is malformed ({exc})"
                )
            },
        ) from exc

    try:
        permission_manager.switch_profile(profile_name, user_overrides=user_overrides)
    except Exception as e:
        logger.error(
            "Failed to switch permission profile to %r for %s: %s",
            profile_name,
            subject or "node",
            e,
        )
        return False

    logger.info("Applied permission profile %r for %s", profile_name, subject or "node")
    return True
