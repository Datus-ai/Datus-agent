# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Prompt helpers for the major compact pass.

Two responsibilities:

1. Render the structured summary prompt from a versioned j2 template
   (``compact_major_1.0.j2``) so the wording can be tuned without touching
   Python.
2. Append a single JSONL recovery pointer to the LLM's summary. The pointer
   is added by host code rather than asked of the LLM so the model cannot
   re-enumerate archive files or echo prompt scaffolding into the message.
"""

from __future__ import annotations

from typing import Optional

from datus.prompts.prompt_manager import get_prompt_manager

_TEMPLATE_NAME = "compact_major"
_TEMPLATE_VERSION = "1.0"


def render_major_compact_prompt(
    node_role: str,
    custom_instructions: Optional[str] = None,
) -> str:
    """Render the major-compact summarization prompt.

    Args:
        node_role: Name of the AgenticNode driving this session — surfaces in
            the prompt so the model maintains its node identity.
        custom_instructions: Optional extra steering appended after the
            constraints section. ``None`` or empty skips the block.
    """
    return get_prompt_manager().render_template(
        _TEMPLATE_NAME,
        _TEMPLATE_VERSION,
        node_role=node_role,
        custom_instructions=custom_instructions or "",
    )


def build_continuation_message(summary: str, history_jsonl_path: str) -> str:
    """Append the JSONL recovery pointer to the LLM summary.

    The pointer is appended verbatim rather than asked of the LLM so that
    (a) the model cannot enumerate individual archive files and inflate the
    summary, and (b) the path round-trip stays deterministic. The output is
    a plain markdown string suitable for an ``output_text`` assistant block.
    """
    body = summary.strip()
    if history_jsonl_path:
        return f"{body}\n\n---\nFull session history (every item, JSONL): `read_file({history_jsonl_path!r})`"
    return body
