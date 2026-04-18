# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Utilities for composing feedback-agent user prompts from reaction context."""

from typing import Optional

_ELLIPSIS = "..."


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    if max_len <= len(_ELLIPSIS):
        return _ELLIPSIS[:max_len]
    keep = max_len - len(_ELLIPSIS)
    return text[:keep] + _ELLIPSIS


def build_reaction_feedback_prompt(
    reaction_emoji: str,
    reference_msg: str,
    reaction_msg: Optional[str] = None,
    max_ref_len: int = 500,
) -> str:
    """Build the canonical user prompt for reaction-triggered feedback.

    Format:
        [The user reacted to this message "{reference_msg}" with [{emoji}]] {reaction_msg}

    Args:
        reaction_emoji: Normalized emoji name (e.g. "thumbsup").
        reference_msg: The bot message the user reacted to.
        reaction_msg: Optional free-text comment the user attached to the reaction.
        max_ref_len: Truncate ``reference_msg`` to this length; ``<= 0`` disables truncation.
    """
    emoji = (reaction_emoji or "").strip()
    ref = _truncate((reference_msg or "").strip(), max_ref_len)
    prompt = f'[The user reacted to this message "{ref}" with [{emoji}]]'
    extra = (reaction_msg or "").strip()
    if extra:
        prompt = f"{prompt} {extra}"
    return prompt
