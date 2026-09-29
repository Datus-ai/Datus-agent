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
be decoded (``datus.models.model_error.provider_error_message``), falling back
to a generic sentence otherwise. The original exception is still logged
server-side with a full traceback for debugging.
"""

from datus.models.model_error import provider_error_message

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


def humanize_stream_error(exc: BaseException) -> tuple[str, str]:
    """Return ``(error_type, message)`` safe to send to the client.

    ``error_type`` is a stable code (see ``_CLASS_TO_ERROR``); ``message`` is a
    human-readable sentence, preferring the upstream provider's own wording.
    Never returns a raw byte-escaped blob.
    """
    error_type, fallback = _classify(exc)
    return error_type, provider_error_message(exc) or fallback


def _classify(exc: BaseException) -> tuple[str, str]:
    for klass in type(exc).__mro__:
        mapped = _CLASS_TO_ERROR.get(klass.__name__)
        if mapped:
            return mapped
    return _DEFAULT_ERROR
