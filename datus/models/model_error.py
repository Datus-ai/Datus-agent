"""Name the failure behind a model call that raised, for a host to act on.

A provider rejecting a chat turn reaches the client as a plain ``error``
content block (see ``action_sse_converter._build_error_content``), so without a
code the host can only pattern-match the provider's wording. This maps the
exception to the ``ErrorCode`` it amounts to, so the block can carry one.
"""

import re
from typing import Optional

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
