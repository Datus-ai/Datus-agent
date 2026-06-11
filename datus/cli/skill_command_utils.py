# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared helpers for skill-shortcut slash commands.

``/init``, ``/session-summarize`` and ``/memory-organize`` are thin wrappers
that inject a deterministic chat message asking the agent to follow a bundled
skill. Each accepts an optional free-text description after the command
(e.g. ``/init this is a sales analytics warehouse, focus on the orders domain``).

The description is not parsed — it is forwarded verbatim into the chat message
through a ``{user_context}`` placeholder the prompt template carries, so the
agent can fold it into its inferred goal / scope / manifest. Keeping the
rendering here means the three handlers stay decoupled yet format the appended
block identically.
"""

from __future__ import annotations

_USER_CONTEXT_LEAD = (
    "Additional context from the user (treat as goal / scope / refinement hints "
    "and fold them into your plan or manifest):"
)


def render_skill_prompt(template: str, args: str | None) -> str:
    """Fill a skill-shortcut prompt template's ``{user_context}`` placeholder.

    ``template`` MUST contain a single ``{user_context}`` slot. When ``args``
    carries a non-blank description it is appended as an explicit context block;
    when blank/whitespace/``None`` the slot collapses to an empty string so the
    canonical prompt is sent unchanged.
    """
    extra = (args or "").strip()
    block = f"\n\n{_USER_CONTEXT_LEAD}\n{extra}" if extra else ""
    return template.format(user_context=block)
