# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

LITELLM_EMPTY_PLACEHOLDER = "[System: Empty message content sanitised to satisfy protocol]"


def redact_uri(uri: str) -> str:
    """Redact any password in a database URI, keeping scheme/host/path visible."""
    if not uri:
        return uri
    try:
        parts = urlsplit(uri)
    except ValueError:
        return uri
    if parts.password is None:
        return uri
    username = parts.username or ""
    hostname = parts.hostname or ""
    # urlsplit strips IPv6 brackets from .hostname; restore them so the
    # reconstructed netloc remains parseable (e.g. "[::1]:5432").
    host = f"[{hostname}]" if ":" in hostname else hostname
    try:
        port_num = parts.port
    except ValueError:
        port_num = None
    port = f":{port_num}" if port_num is not None else ""
    netloc = f"{username}:***@{host}{port}" if username else f"***@{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def strip_litellm_placeholder(text: str) -> str:
    """Return empty string if text is only the LiteLLM sanitizer placeholder."""
    if not text or not isinstance(text, str):
        return text
    if text.strip() == LITELLM_EMPTY_PLACEHOLDER:
        return ""
    return text


class LitellmPlaceholderStreamFilter:
    """Incremental filter that suppresses the LiteLLM sanitizer placeholder
    while it is still being streamed token-by-token.

    Strategy: buffer chunks while they remain a strict prefix of the
    placeholder. Once the buffer either diverges from the placeholder prefix
    or fully consumes it, switch to pass-through and emit the appropriate
    tail. Subsequent placeholder occurrences in the same stream are not
    filtered (acceptable trade-off — production traces show the placeholder
    only ever appears at the very start of a content part).
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._passthrough = False

    def feed(self, delta: str) -> str:
        if not delta or not isinstance(delta, str):
            return ""
        if self._passthrough:
            return delta

        self._buffer += delta

        if LITELLM_EMPTY_PLACEHOLDER.startswith(self._buffer):
            return ""

        if self._buffer.startswith(LITELLM_EMPTY_PLACEHOLDER):
            tail = self._buffer[len(LITELLM_EMPTY_PLACEHOLDER) :]
            self._buffer = ""
            self._passthrough = True
            return tail

        flushed = self._buffer
        self._buffer = ""
        self._passthrough = True
        return flushed

    def finalize(self) -> str:
        if self._passthrough:
            self.reset()
            return ""
        tail = "" if self._buffer == LITELLM_EMPTY_PLACEHOLDER else self._buffer
        self.reset()
        return tail

    def reset(self) -> None:
        self._buffer = ""
        self._passthrough = False


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
