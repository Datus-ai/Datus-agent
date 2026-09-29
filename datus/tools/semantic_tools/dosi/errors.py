# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Mapping from the engine's structured exceptions to core error shapes.

The engine's ``QueryError`` already carries stable codes and candidates —
this module only reshapes them into ``SemanticValidationError`` (agent-facing,
retryable) or ``SemanticCoreException`` (infrastructure, not retryable).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from datus.tools.semantic_tools.exceptions import SemanticCoreException
from datus.tools.semantic_tools.models import SemanticValidationError

# A ``QueryError`` is by construction a planner rejection: the engine raises
# it when a *request* cannot be planned, and it carries the names, candidates
# and retry hint an agent needs to revise it. Model loading, execution and
# configuration failures are separate exception types. So the default is
# retryable, and this names the exceptions — a denylist, because an allowlist
# of codes goes stale silently as the engine grows, and did: seven codes added
# after it was written (``window_exclude_not_grouped``,
# ``unconformed_dimension``, ``unknown_metric_param``,
# ``param_expansion_too_large``, ``unknown_dataset``, ``detail_fanout``,
# ``metric_in_detail_query``) reached the caller as flat
# ``SemanticCoreException`` text, dropping the structured payload they exist
# to carry. Failing the other way — a structured payload for something the
# agent cannot actually fix — costs it nothing, since either shape arrives as
# a failed tool call.
_NON_RETRYABLE_QUERY_CODES = {
    # The engine failed, not the request. Revising the query cannot help.
    "internal",
}


class SemanticValidationException(Exception):
    """A deterministic query rejection carrying a structured payload.

    ``payload`` is a ``SemanticValidationError`` so callers revise arguments
    from stable fields instead of parsing exception text.
    """

    def __init__(self, payload: SemanticValidationError):
        self.payload = payload
        super().__init__(payload.message or "semantic validation error")


def _message_with_context(exc: Any) -> str:
    parts = [str(getattr(exc, "message", "") or exc)]
    candidates = list(getattr(exc, "candidates", ()) or ())
    if candidates:
        parts.append(f"candidates: {', '.join(candidates)}")
    hint = getattr(exc, "hint", None)
    if hint:
        parts.append(str(hint))
    return " | ".join(parts)


def validation_error_from_query_error(
    exc: Any,
    *,
    requested_metrics: Optional[List[str]] = None,
    requested_dimensions: Optional[List[str]] = None,
    requested_time_granularity: Optional[str] = None,
) -> SemanticValidationError:
    """Reshape an engine ``QueryError`` into a ``SemanticValidationError``.

    The engine's ``candidates`` become a concrete ``suggested_retry`` only
    when the fix is unambiguous (exactly one candidate); otherwise they stay
    in the message for the agent to choose from.
    """
    code = str(getattr(exc, "code", "") or "validation_error")
    candidates = list(getattr(exc, "candidates", ()) or ())

    raw_retry = getattr(exc, "suggested_retry", None)
    suggested_retry: Optional[Dict[str, Any]] = dict(raw_retry) if isinstance(raw_retry, dict) else None
    if suggested_retry is None and len(candidates) == 1:
        if code == "unknown_metric":
            suggested_retry = {"metrics": candidates}
        elif code in {"unknown_dimension", "ambiguous_dimension"}:
            suggested_retry = {
                "metrics": list(requested_metrics or []),
                "dimensions": candidates,
            }
        elif code == "no_primary_time_dimension":
            dimensions = []
            replaced = False
            for dimension in requested_dimensions or []:
                if dimension == "metric_time":
                    dimensions.append(candidates[0])
                    replaced = True
                else:
                    dimensions.append(dimension)
            if not replaced:
                dimensions.append(candidates[0])
            suggested_retry = {
                "metrics": list(requested_metrics or []),
                "dimensions": dimensions,
            }
            if requested_time_granularity:
                suggested_retry["time_granularity"] = requested_time_granularity

    unsupported_dimensions: List[str] = []
    if code in {"unknown_dimension", "ambiguous_dimension"}:
        # The offending name is whichever requested dimension is not itself
        # a valid candidate; with no request context this stays empty.
        unsupported_dimensions = [d for d in (requested_dimensions or []) if d not in candidates]

    return SemanticValidationError(
        code=code,
        metrics=list(getattr(exc, "metrics", ()) or ()) or list(requested_metrics or []),
        unsupported_dimensions=unsupported_dimensions,
        required_dimensions=(candidates if code in {"no_primary_time_dimension", "time_range_needs_dimension"} else []),
        suggested_retry=suggested_retry,
        message=_message_with_context(exc),
    )


def raise_mapped(exc: Any, binding: Any, **request_context: Any) -> None:
    """Re-raise an engine exception in the Agent's stable error vocabulary.

    Retryable planner rejections become ``SemanticValidationException``;
    model/execution/config failures become ``SemanticCoreException``.
    """
    if isinstance(exc, binding.QueryError) and exc.code not in _NON_RETRYABLE_QUERY_CODES:
        raise SemanticValidationException(validation_error_from_query_error(exc, **request_context)) from exc
    not_computable_error = getattr(binding, "NotComputableError", None)
    if not_computable_error is not None and isinstance(exc, not_computable_error):
        raise SemanticValidationException(
            SemanticValidationError(
                code=str(getattr(exc, "code", "") or "not_computable"),
                metrics=list(request_context.get("requested_metrics") or []),
                message=_message_with_context(exc),
            )
        ) from exc
    if isinstance(exc, binding.OsiError):
        raise SemanticCoreException(f"Dosi {type(exc).__name__} [{exc.code}]: {_message_with_context(exc)}") from exc
    raise exc
