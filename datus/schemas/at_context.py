# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared @-context carrier for node inputs.

Every ``@Table`` / ``@Metric`` / ``@Sql`` / ``@Knowledge`` reference a user
attaches to a chat turn is resolved into structured objects and must reach the
node's prompt via :meth:`AgenticNode._build_enhanced_message`. Declaring these
fields on a single base (rather than per input model) is what keeps the set
leak-proof: a new reference kind is added here and to :func:`apply_at_context`
once, and every subagent input inherits it automatically.

Lives in its own module (not ``base.py``) because the field types come from
``node_models`` which already imports ``base`` — folding them into ``base``
would create a ``base`` <-> ``node_models`` import cycle.
"""

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, ConfigDict, Field

from datus.schemas.base import BaseInput
from datus.schemas.node_models import Metric, ReferenceSql, TableSchema


class AtContextInput(BaseInput):
    """Base input carrying resolved @-referenced context.

    Consumed uniformly by :meth:`AgenticNode._build_enhanced_message` (which
    reads each field via ``getattr``), so any node whose input inherits this
    renders the referenced context without node-specific wiring.
    """

    schemas: Optional[List[TableSchema]] = Field(default=None, description="@Table referenced table schemas")
    metrics: Optional[List[Metric]] = Field(default=None, description="@Metric referenced metrics")
    reference_sql: Optional[List[ReferenceSql]] = Field(
        default=None,
        description="@Sql referenced SQL snippets to reuse/adjust",
        validation_alias=AliasChoices("reference_sql", "historical_sql"),
    )
    external_knowledge: Optional[str] = Field(
        default="", description="@Knowledge / supplementary business evidence supplied with the question"
    )
    context_hints: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Referenced items whose full detail could not be pre-loaded. Each hint "
            "{kind, name, subject_path} tells the model to fetch it via the matching "
            "tool (get_metrics / get_reference_sql) instead of searching for it."
        ),
    )

    model_config = ConfigDict(populate_by_name=True)


def apply_at_context(
    node_input: BaseInput,
    *,
    schemas: Optional[List[TableSchema]] = None,
    metrics: Optional[List[Metric]] = None,
    reference_sql: Optional[List[ReferenceSql]] = None,
    external_knowledge: Optional[str] = None,
    context_hints: Optional[List[Dict[str, Any]]] = None,
) -> BaseInput:
    """Populate @-context fields on *node_input* in place, returning it.

    ``hasattr``-guarded per field so inputs that do not carry a given field
    (or don't inherit :class:`AtContextInput` at all) are silently skipped —
    this is the single choke point both node-input funnels call, so adding a
    reference kind never requires touching individual branches. ``None`` args
    leave the existing value untouched.
    """
    if schemas is not None and hasattr(node_input, "schemas"):
        node_input.schemas = schemas
    if metrics is not None and hasattr(node_input, "metrics"):
        node_input.metrics = metrics
    if reference_sql is not None and hasattr(node_input, "reference_sql"):
        node_input.reference_sql = reference_sql
    if external_knowledge is not None and hasattr(node_input, "external_knowledge"):
        node_input.external_knowledge = external_knowledge
    if context_hints is not None and hasattr(node_input, "context_hints"):
        node_input.context_hints = context_hints
    return node_input
