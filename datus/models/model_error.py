"""Name the failure behind a model call that raised, for a host to act on.

A provider rejecting a chat turn reaches the client as a plain ``error``
content block (see ``action_sse_converter._build_error_content``), so without a
code the host can only pattern-match the provider's wording. This maps the
exception to the ``ErrorCode`` it amounts to, so the block can carry one.
"""

import ast
import json
import re
from typing import Any, Optional

from openai import APIError as OpenAIAPIError

from datus.models.openai_compatible import classify_openai_compatible_error
from datus.utils.exceptions import DatusException, ErrorCode

try:  # The native Anthropic path raises its own SDK's errors.
    from anthropic import APIError as AnthropicAPIError
except ImportError:  # pragma: no cover - anthropic is a hard dependency today
    AnthropicAPIError = None

# Model-layer codes (the 3xxxxx range) — the only ones this ever returns.
_MODEL_CODE_PREFIX = "3"

_BY_STATUS = {
    401: ErrorCode.MODEL_AUTHENTICATION_ERROR,
    403: ErrorCode.MODEL_PERMISSION_ERROR,
    404: ErrorCode.MODEL_NOT_FOUND,
    413: ErrorCode.MODEL_REQUEST_TOO_LARGE,
    500: ErrorCode.MODEL_API_ERROR,
    502: ErrorCode.MODEL_OVERLOADED,
    503: ErrorCode.MODEL_OVERLOADED,
    529: ErrorCode.MODEL_OVERLOADED,
}

# Some providers answer an unknown model name with HTTP 400 rather than 404 —
# DeepSeek: "The supported API model names are ..., but you passed ...".
_UNKNOWN_MODEL = re.compile(
    r"supported api model names are"
    r"|\bmodel_not_found\b"
    r"|\bmodel\b[^.\n]{0,60}\b(?:does not exist|not exist|not found)\b"
    r"|\b(?:invalid|unknown) model\b",
    re.IGNORECASE,
)

_QUOTA = re.compile(r"quota|billing", re.IGNORECASE)

# The SDKs stringify a rejection as ``Error code: 400 - {<body as a Python
# dict repr>}``; litellm as ``litellm.BadRequestError: AnthropicException -
# b'{<json>}'`` or with bare JSON.
_BYTES_LITERAL = re.compile(r"b'(?:[^'\\]|\\.)*'|b\"(?:[^\"\\]|\\.)*\"")
# Some gateways pack the whole message as ``[code][human text][request id]``.
# Anchored so a message that merely contains ``content[0]`` is left alone.
_BRACKETED_ONLY = re.compile(r"^\s*(?:\[[^\[\]]*\]\s*)+$")
_CJK = re.compile(r"[一-鿿]")
# Long opaque tokens (request ids, trace ids) embedded in a provider message.
# Hex-specific lookarounds (not ``\b``) so ids touching CJK text are stripped
# too: CJK chars are ``\w`` under Unicode, so ``\b`` finds no boundary there.
_ID_TOKEN = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{16,}(?![0-9a-fA-F])")
_MAX_MESSAGE_LEN = 300


def _is_provider_error(exc: BaseException) -> bool:
    if isinstance(exc, OpenAIAPIError):  # litellm's exceptions subclass these
        return True
    return AnthropicAPIError is not None and isinstance(exc, AnthropicAPIError)


def classify_model_error(exc: BaseException) -> Optional[ErrorCode]:
    """The model ``ErrorCode`` behind ``exc``, or None when it is not a model failure.

    Only provider SDK errors and model-range ``DatusException``s are
    classified; a tool or storage failure stays unlabelled rather than being
    passed off as the model's.
    """
    if isinstance(exc, DatusException):
        return exc.code if exc.code.code.startswith(_MODEL_CODE_PREFIX) else None

    if not _is_provider_error(exc):
        return None

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 400 and _UNKNOWN_MODEL.search(str(exc)):
            return ErrorCode.MODEL_NOT_FOUND
        if status == 429:
            return ErrorCode.MODEL_QUOTA_EXCEEDED if _QUOTA.search(str(exc)) else ErrorCode.MODEL_RATE_LIMIT
        # Any other status (a 400 for another reason, 409, 422, ...) says too
        # little to name: a host decides by the code alone, so a guessed one
        # would be worse than none.
        return _BY_STATUS.get(status)

    # No status (connection, timeout, TLS): the message-based rules the retry
    # path already uses.
    code, _ = classify_openai_compatible_error(exc)
    return code


def model_error_message(exc: BaseException) -> Optional[str]:
    """The provider's own sentence behind a failed model call, or None.

    ``str(exc)`` on a provider rejection is ``Error code: 400 - {...}`` with the
    whole response body inlined; a chat error card should carry only the
    ``error.message`` inside it. Returns None for anything that is not a
    provider error, or when no message can be recovered, so the caller keeps
    its usual rendering.
    """
    if not _is_provider_error(exc):
        return None
    return provider_error_message(exc)


def provider_error_message(exc: BaseException) -> Optional[str]:
    """Best-effort: the readable ``error.message`` carried by ``exc``, or None.

    Unlike ``model_error_message`` this does not check the exception type, for
    callers that already know the failure came from the model call.
    """
    message = _body_message(getattr(exc, "body", None)) or _body_message(_decode_embedded_body(str(exc)))
    if not message:
        return None
    message = _pick_readable_segment(message)
    message = _ID_TOKEN.sub("", message)
    message = re.sub(r"\s+", " ", message).strip(" ,;:")
    if len(message) > _MAX_MESSAGE_LEN:
        message = message[:_MAX_MESSAGE_LEN].rstrip() + "…"
    return message or None


def _body_message(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    # Anthropic nests ``{"type": "error", "error": {...}}``; OpenAI-compatible
    # servers send ``{"error": {...}}`` or, via the openai SDK, the inner object.
    message = error.get("message") if isinstance(error, dict) else error
    message = message or body.get("message")
    return message if isinstance(message, str) and message.strip() else None


def _decode_embedded_body(raw: str) -> Optional[dict]:
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

    start, end = raw.find("{"), raw.rfind("}")
    if not 0 <= start < end:
        return None
    candidate = raw[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(candidate)  # a Python dict repr
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _pick_readable_segment(message: str) -> str:
    """For ``[code][text][id]``, the most readable segment: CJK first, then longest."""
    if not _BRACKETED_ONLY.match(message):
        return message
    segments = [s for s in re.findall(r"\[([^\[\]]*)\]", message) if s.strip()]
    if not segments:
        return message
    return max(segments, key=lambda s: (1 if _CJK.search(s) else 0, len(s)))
