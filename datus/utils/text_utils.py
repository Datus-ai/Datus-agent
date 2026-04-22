# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import re
import unicodedata

LITELLM_EMPTY_PLACEHOLDER = "[System: Empty message content sanitised to satisfy protocol]"


def strip_litellm_placeholder(text: str) -> str:
    """Return empty string if text is only the LiteLLM sanitizer placeholder."""
    if not text or not isinstance(text, str):
        return text
    if text.strip() == LITELLM_EMPTY_PLACEHOLDER:
        return ""
    return text


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    text = unicodedata.normalize("NFKC", text)

    # 2. Replace common invisible whitespace
    # NBSP  # zero width  # BOM
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

    # 3.Remove control characters(retain \n \t)
    text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", text)

    # 4. Uniform line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text.strip()
