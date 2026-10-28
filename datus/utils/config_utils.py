# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Coercion helpers for values that arrive from YAML, env vars or API payloads.

These live outside :mod:`datus.configuration.agent_config` so that modules which
need to interpret a config value do not have to reach into another package for a
private symbol.
"""

import math
from typing import Any

__all__ = ["coerce_bool", "coerce_positive_int", "coerce_positive_seconds"]


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


def coerce_positive_int(value: Any, default: int) -> int:
    """Coerce a config value or caller-supplied count to a positive ``int``.

    Both operator YAML and direct (non-HTTP) callers are untrusted here: a
    missing key, ``null``, a non-numeric string, or a non-positive number all
    yield *default* rather than a bound that breaks the caller.

    Two things Python would coerce silently are treated as typos instead:

    - ``bool`` is an ``int`` subclass, and YAML reads ``true`` / ``yes`` / ``on``
      as ``True``, so ``int()`` would turn a mistyped toggle into a bound of 1 —
      a bound that looks deliberate and quietly rejects almost everything.
    - YAML ``.inf`` — what an operator writes for "no limit" — is a float
      infinity that ``int()`` refuses to convert, raising ``OverflowError``.

    A fractional number does still truncate (``3.5`` yields 3, ``0.5`` yields 0
    and so falls back for being non-positive). That is deliberate: callers pass
    request-supplied page sizes through here too, where truncating still bounds
    the request, whereas falling back would hand back a *larger* default than
    was asked for. A caller that must reject a fractional bound outright should
    check before calling.
    """
    # Checked before int(), which accepts bool via the int subclass.
    if isinstance(value, bool):
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return coerced if coerced > 0 else default


def coerce_positive_seconds(value: Any, default: float) -> float:
    """Coerce a config value to a positive, finite number of seconds.

    Like :func:`coerce_positive_int` but keeping fractions, since sub-second
    budgets are meaningful. Infinity is rejected rather than honoured: it passes
    a bare ``> 0`` test and would silently turn a deadline into no deadline —
    the opposite of what a timeout is for. ``bool`` is rejected for the same
    reason it is there: YAML ``true`` would otherwise buy a 1-second budget.
    """
    if isinstance(value, bool):
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if coerced > 0 and math.isfinite(coerced):
        return coerced
    return default
