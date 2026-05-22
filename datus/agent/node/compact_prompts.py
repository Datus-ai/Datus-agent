# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Prompt helpers for the major compact pass.

Two responsibilities:

1. Render the structured 10-section summary prompt from a versioned j2
   template (``compact_major_1.0.j2``) so the wording can be tuned without
   touching Python.
2. Wrap the LLM's summary into a continuation user-message the next turn
   sees. That message embeds the JSONL history dump path and the archive
   directory, so the model can ``read_file`` any specific detail it omitted
   from the summary.
"""

from __future__ import annotations

from typing import Optional

from datus.prompts.prompt_manager import get_prompt_manager

_TEMPLATE_NAME = "compact_major"
_TEMPLATE_VERSION = "1.0"


def render_major_compact_prompt(
    node_role: str,
    history_jsonl_path: str,
    archive_dir: str,
    custom_instructions: Optional[str] = None,
) -> str:
    """Render the major-compact summarization prompt.

    Args:
        node_role: Name of the AgenticNode driving this session — surfaces in
            the prompt so the model maintains its node identity.
        history_jsonl_path: Path to the on-disk JSONL dump of the full prior
            session, used as a recovery pointer.
        archive_dir: Directory holding per-message archived args/output blobs.
        custom_instructions: Optional extra steering appended after the
            constraints section. ``None`` or empty skips the block.
    """
    return get_prompt_manager().render_template(
        _TEMPLATE_NAME,
        _TEMPLATE_VERSION,
        node_role=node_role,
        history_jsonl_path=history_jsonl_path,
        archive_dir=archive_dir,
        custom_instructions=custom_instructions or "",
    )


def build_continuation_message(summary: str, history_jsonl_path: str, archive_dir: str) -> str:
    """Build the user message that seeds the next turn after a major compact.

    The message:
    - Tells the model the session was continued from a previous run.
    - Embeds the summary verbatim.
    - Surfaces the JSONL and archive paths so the model knows how to recover
      detail that the summary dropped.
    - Instructs the model to resume work without greetings or recaps.
    """
    return (
        "This session is being continued from a previous conversation that hit "
        "context limits. Below is the structured summary; full original content "
        "is preserved on disk.\n\n"
        f"{summary.strip()}\n\n"
        "If you need exact details not in the summary:\n"
        f"- Full session history (every item, JSONL): read_file({history_jsonl_path!r})\n"
        f"- Archived tool I/O: read_file files under {archive_dir}\n\n"
        "Continue from where you left off. Do not greet, do not recap — pick up "
        "the last task as if uninterrupted."
    )
