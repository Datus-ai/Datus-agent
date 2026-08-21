# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Coercion helpers for values that arrive from YAML, env vars or API payloads.

These live outside :mod:`datus.configuration.agent_config` so that modules which
need to interpret a config value do not have to reach into another package for a
private symbol.
"""

from typing import Any

__all__ = ["coerce_bool"]


def coerce_bool(value: Any, default: bool) -> bool:
    """Coerce a config value to ``bool`` accepting YAML's string booleans.

    ``bool("false")`` is ``True`` in Python, so a naive ``bool(...)`` cast on a
    YAML value like ``enabled: "false"`` silently flips the toggle on. This
    helper normalizes booleans and the common string spellings users actually
    write in agent.yml.

    ``None`` (the key is absent) yields ``default``. Anything else — an int, a
    Mock, an object a host handed us — falls through to ``bool(value)``, which
    for a security toggle means an unrecognised value reads as "on" rather than
    silently off.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)
