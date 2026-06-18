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

from typing import Union

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
        if lowered in _TRUE_STRINGS:
            verify = True
        elif lowered in _FALSE_STRINGS:
            verify = False
        else:
            # Treat as a path to a CA bundle. Existence is not enforced here so
            # configuration errors surface as an explicit TLS error at call time.
            verify = stripped
    else:
        raise TypeError(f"ssl_verify must be bool or str, got {type(value).__name__}")

    if verify is False:
        logger.warning(
            "ssl_verify is disabled — TLS certificate verification is OFF for this "
            "endpoint. This is insecure (MITM-exposed); prefer pointing ssl_verify at "
            "a CA bundle path instead."
        )
    elif isinstance(verify, str):
        logger.debug("ssl_verify resolved to custom CA bundle: %s", verify)

    return verify


def ssl_verify_to_env(value: Union[bool, str]) -> str:
    """Render a normalized verify value for the ``SSL_VERIFY`` environment variable.

    Booleans become ``"true"`` / ``"false"`` (litellm-compatible spellings); a CA
    bundle path is returned unchanged.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
