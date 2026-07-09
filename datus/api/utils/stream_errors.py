"""Turn a stream-breaking exception into a client-safe error payload.

The chat SSE stream can die on almost anything the agentic loop touches, but
the most common culprit is an upstream LLM failure surfaced by litellm. Its
exceptions stringify to blobs like::

    litellm.InternalServerError: AnthropicException - b'{"type":"error",
    "error":{"type":"api_error","code":"1234","message":"[1234][\\xe7\\xbd...]"}}'

Sending ``str(exc)`` straight to the browser (as ``SSEErrorData.error``) means
the user sees that raw, byte-escaped blob. ``humanize_stream_error`` maps the
exception to a stable machine ``error_type`` (so the frontend can localize) and
a clean human ``message`` — preferring the provider's own wording when it can
be decoded, falling back to a generic sentence otherwise. The original
exception is still logged server-side with a full traceback for debugging.
"""

import ast
import json
import re
from typing import Optional

# Stable ``error_type`` codes the frontend can map to localized copy. Keyed by
# litellm/openai exception class names (matched anywhere in the MRO, so
# provider-specific subclasses still resolve). Each maps to (code, fallback).
_CLASS_TO_ERROR: dict[str, tuple[str, str]] = {
    "RateLimitError": (
        "UPSTREAM_RATE_LIMITED",
        "The AI service is receiving too many requests right now. Please retry in a moment.",
    ),
    "Timeout": ("UPSTREAM_TIMEOUT", "The AI service timed out. Please try again."),
    "APITimeoutError": ("UPSTREAM_TIMEOUT", "The AI service timed out. Please try again."),
    "APIConnectionError": (
        "UPSTREAM_UNAVAILABLE",
        "Could not reach the AI service. Please check your connection and retry.",
    ),
    "ServiceUnavailableError": (
        "UPSTREAM_UNAVAILABLE",
        "The AI service is temporarily unavailable. Please try again shortly.",
    ),
    "InternalServerError": (
        "UPSTREAM_ERROR",
        "The AI service ran into a temporary error. Please try again.",
    ),
    "APIError": ("UPSTREAM_ERROR", "The AI service ran into a temporary error. Please try again."),
    "ContextWindowExceededError": (
        "CONTEXT_LENGTH_EXCEEDED",
        "This conversation is too long for the model. Please start a new session or compact it.",
    ),
    "AuthenticationError": (
        "UPSTREAM_AUTH_ERROR",
        "The AI service rejected the request credentials. Please contact your administrator.",
    ),
    "PermissionDeniedError": (
        "UPSTREAM_AUTH_ERROR",
        "The AI service rejected the request credentials. Please contact your administrator.",
    ),
    "ContentPolicyViolationError": (
        "CONTENT_POLICY_VIOLATION",
        "The request was blocked by the AI provider's content policy.",
    ),
    "BadRequestError": (
        "UPSTREAM_BAD_REQUEST",
        "The AI service rejected the request. Please try again or adjust your input.",
    ),
}

_DEFAULT_ERROR: tuple[str, str] = (
    "INTERNAL_ERROR",
    "Something went wrong while generating the response. Please try again.",
)

# Long opaque tokens (request ids, trace ids) embedded in a provider message —
# stripped so the surfaced sentence stays readable.
_ID_TOKEN = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_CJK = re.compile(r"[一-鿿]")
_BYTES_LITERAL = re.compile(r"b'(?:[^'\\]|\\.)*'|b\"(?:[^\"\\]|\\.)*\"")
_MAX_LEN = 300


def humanize_stream_error(exc: BaseException) -> tuple[str, str]:
    """Return ``(error_type, message)`` safe to send to the client.

    ``error_type`` is a stable code (see ``_CLASS_TO_ERROR``); ``message`` is a
    human-readable sentence, preferring the upstream provider's own wording.
    Never returns a raw byte-escaped blob.
    """
    error_type, fallback = _classify(exc)
    provider_message = _extract_provider_message(exc)
    message = provider_message or fallback
    return error_type, _clean(message)


def _classify(exc: BaseException) -> tuple[str, str]:
    for klass in type(exc).__mro__:
        mapped = _CLASS_TO_ERROR.get(klass.__name__)
        if mapped:
            return mapped
    return _DEFAULT_ERROR


def _extract_provider_message(exc: BaseException) -> Optional[str]:
    """Best-effort: pull the provider's human message out of ``str(exc)``.

    Handles the ``... - b'{...}'`` litellm shape (Python bytes-repr wrapping
    provider JSON) and bare embedded JSON. Returns None if nothing readable
    can be recovered.
    """
    raw = str(exc)
    if not raw:
        return None

    payload = _decode_embedded_payload(raw)
    if payload is None:
        return None

    message = payload.get("error", {}).get("message") if isinstance(payload.get("error"), dict) else None
    message = message or payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return None

    return _pick_readable_segment(message)


def _decode_embedded_payload(raw: str) -> Optional[dict]:
    # ``b'...\xe7...'`` — a Python bytes literal; eval it back to bytes and
    # decode as UTF-8 so escaped multibyte chars become real text.
    literal = _BYTES_LITERAL.search(raw)
    if literal:
        try:
            decoded = ast.literal_eval(literal.group(0))
            text = decoded.decode("utf-8", "replace") if isinstance(decoded, bytes) else str(decoded)
            return json.loads(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass

    # Bare embedded JSON object anywhere in the string.
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _pick_readable_segment(message: str) -> str:
    """Some providers pack the message as ``[code][human text][request_id]``.

    Pick the most human-readable bracketed segment (CJK first, then longest);
    fall back to the whole message when there are no brackets.
    """
    segments = re.findall(r"\[([^\[\]]+)\]", message)
    if segments:
        segments.sort(key=lambda s: (1 if _CJK.search(s) else 0, len(s)), reverse=True)
        return segments[0]
    return message


def _clean(message: str) -> str:
    message = _ID_TOKEN.sub("", message)
    message = re.sub(r"\s+", " ", message).strip(" ,;:")
    if len(message) > _MAX_LEN:
        message = message[:_MAX_LEN].rstrip() + "…"
    return message or _DEFAULT_ERROR[1]
