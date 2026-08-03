# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared helpers for the pinned response language.

Lives in ``utils`` rather than next to the agentic nodes because standalone
LLM calls outside the node stack (e.g. the visualization tool) need the same
code → name mapping without importing the node layer.
"""

from typing import Dict, Optional

LANGUAGE_NAME_MAP: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Traditional Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
}


def resolve_language_name(code: Optional[str]) -> str:
    """Map a language code (e.g. ``"zh"``) to a human-readable name.

    Unknown codes are returned as-is so operators can plug in custom values
    without a code change.
    """
    if not code:
        return "English"
    return LANGUAGE_NAME_MAP.get(code.strip().lower(), code)
