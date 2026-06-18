# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers for resolving SSL/TLS verification configuration for LLM endpoints.

The ``ssl_verify`` model setting mirrors the ``verify`` argument of httpx and
litellm:

* ``True``  -> verify against the system / certifi CA bundle (default)
* ``False`` -> disable verification entirely (discouraged; MITM-exposed)
* ``str``   -> path to a CA bundle (PEM) to trust, e.g. a private gateway CA

These helpers normalize a user-supplied value into that ``bool | str`` shape and
render it back for the ``SSL_VERIFY`` environment variable so the litellm code
paths (which only accept SSL configuration via env / module globals) honor the
same setting as the native client path.
"""

from typing import Optional, Union

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# String spellings accepted as booleans (case-insensitive), matching litellm.
_TRUE_STRINGS = {"true"}
_FALSE_STRINGS = {"false"}


def normalize_ssl_verify(value: Union[bool, str]) -> Union[bool, str]:
    """Normalize a configured ``ssl_verify`` value into an httpx ``verify`` value.

    * ``bool`` is returned as-is.
    * ``"true"`` / ``"false"`` (any case) are coerced to ``bool``.
    * Any other non-empty string is treated as a CA bundle path.

    A warning is logged when verification is disabled.
    """
    if isinstance(value, bool):
        verify: Union[bool, str] = value
    elif isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if not stripped:
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message_args={
                    "field_name": "ssl_verify",
                    "except_values": "true/false or a CA bundle path",
                    "your_value": "(empty string)",
                },
            )
        if lowered in _TRUE_STRINGS:
            verify = True
        elif lowered in _FALSE_STRINGS:
            verify = False
        else:
            # Treat as a path to a CA bundle. Existence is not enforced here so
            # configuration errors surface as an explicit TLS error at call time.
            verify = stripped
    else:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message_args={
                "field_name": "ssl_verify",
                "except_values": "bool or str",
                "your_value": type(value).__name__,
            },
        )

    if verify is False:
        logger.warning(
            "ssl_verify is disabled — TLS certificate verification is OFF for this "
            "endpoint. This is insecure (MITM-exposed); prefer pointing ssl_verify at "
            "a CA bundle path instead."
        )
    elif isinstance(verify, str):
        logger.debug("ssl_verify resolved to custom CA bundle: %s", verify)

    return verify


# Substrings that identify a TLS certificate-verification failure, regardless of
# which layer (litellm, httpx, anthropic SDK, stdlib ssl) raised it.
_SSL_CERT_ERROR_MARKERS = (
    "certificate_verify_failed",
    "certificate verify failed",
    "self-signed certificate",
    "self signed certificate",
)


def is_ssl_cert_verification_error(exc: BaseException) -> bool:
    """Return True if ``exc`` (or anything in its cause/context chain) is a TLS
    certificate-verification failure.

    The chain is walked because higher layers wrap the original
    ``SSLCertVerificationError`` — e.g. the native Anthropic client surfaces only
    ``APIConnectionError("Connection error.")`` at the top level, with the SSL
    detail nested in ``__cause__``.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    # Bound the walk: real exception chains are short, and the depth cap also
    # protects against pathological/non-terminating chains (e.g. mock objects
    # whose ``__cause__`` lazily yields a fresh child on every access).
    for _ in range(20):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        text = str(cur).lower()
        if any(marker in text for marker in _SSL_CERT_ERROR_MARKERS):
            return True
        nxt = cur.__cause__ or cur.__context__
        cur = nxt if isinstance(nxt, BaseException) else None
    return False


def ssl_verify_to_env(value: Union[bool, str]) -> str:
    """Render a normalized verify value for the ``SSL_VERIFY`` environment variable.

    Booleans become ``"true"`` / ``"false"`` (litellm-compatible spellings); a CA
    bundle path is returned unchanged.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
