# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Project-level plugin activation resolution.

The ``plugins:`` section of ``./.datus/config.yml`` decides which installed
plugins are active for a project (see :class:`datus.configuration.
project_config.PluginActivation`). :func:`active_names_for_cwd` distils that
into the ``active_names`` filter consumed by the registry's collection
functions, for callers that do not hold an :class:`AgentConfig`
(``SkillConfig`` builds skill directories without one).

Semantics mirror ``AgentConfig.active_plugin_names``: section absent → ``None``
(no filter, every plugin active); section present → the set of names whose
``enabled`` flag is true (authoritative whitelist).
"""

from __future__ import annotations

from typing import Optional, Set

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def active_names_for_cwd(cwd: Optional[str] = None) -> Optional[Set[str]]:
    """Return the active-plugin whitelist for ``cwd``, or ``None`` for "all".

    Reads ``./.datus/config.yml`` directly (cheap, CWD-relative) so skill
    discovery — which may run without an ``AgentConfig`` in reach — applies the
    same activation gate as the agent runtime. Any failure resolves to ``None``
    (no filter) so a malformed override never blocks discovery.
    """
    try:
        from datus.configuration.project_config import load_project_override

        override = load_project_override(cwd)
    except Exception as exc:  # noqa: BLE001 - discovery must never crash
        logger.debug("plugin activation lookup failed: %s", exc)
        return None
    if override is None or override.plugins is None:
        return None
    return {name for name, activation in override.plugins.items() if getattr(activation, "enabled", True)}
