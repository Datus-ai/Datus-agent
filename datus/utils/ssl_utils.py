# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers for resolving SSL/TLS verification configuration for LLM endpoints.

The ``ssl_verify`` model setting mirrors the ``verify`` argument of httpx and
litellm:

* ``True``      -> verify against the system / certifi CA bundle (default)
* ``False``     -> disable verification entirely (discouraged; MITM-exposed)
* ``str`` path  -> path to a CA bundle (PEM) to trust, e.g. a private gateway CA
* ``str`` PEM   -> the CA certificate content itself (``-----BEGIN CERTIFICATE...``)

The PEM-content form lets a caller (e.g. Datus SaaS) forward a private CA inline
without materializing a file on disk. It is consumed differently per path:

* native httpx path -> an in-memory ``ssl.SSLContext`` (no file), via
  :func:`resolve_ssl_verify_for_httpx`.
* litellm path -> litellm only accepts a CA bundle via a file path (SSL_VERIFY /
  SSL_CERT_FILE), so :func:`materialize_ca_bundle` spills the content to a
  temp file under a private per-process directory. That is the single
  unavoidable file.
"""

import hashlib
import os
import ssl
import tempfile
from typing import Optional, Union

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# String spellings accepted as booleans (case-insensitive), matching litellm.
_TRUE_STRINGS = {"true"}
_FALSE_STRINGS = {"false"}

# Marks a value as inline PEM certificate content rather than a path/bool.
_PEM_CERT_MARKER = "-----BEGIN CERTIFICATE-----"


def is_pem_cert_content(value: object) -> bool:
    """Return True if ``value`` is an inline PEM certificate (vs a path/bool)."""
    return isinstance(value, str) and _PEM_CERT_MARKER in value


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


def resolve_ssl_verify_for_httpx(value: Union[bool, str]) -> Union[bool, str, ssl.SSLContext]:
    """Resolve a config ``ssl_verify`` into an httpx ``verify`` value.

    Inline PEM content becomes an in-memory ``ssl.SSLContext`` that trusts the
    system roots plus the supplied private CA — no file is written (mirrors how
    the Snowflake connector loads a ``private_key`` from content). Anything else
    is delegated to :func:`normalize_ssl_verify` (bool or path).
    """
    if is_pem_cert_content(value):
        context = ssl.create_default_context()
        context.load_verify_locations(cadata=value)
        logger.debug("ssl_verify resolved to in-memory CA context (PEM content, %d bytes)", len(value))
        return context
    return normalize_ssl_verify(value)


# Private per-process directory (0700, unguessable) that holds materialized CA
# bundles. Created lazily so processes that never use a custom CA pay nothing.
_ca_bundle_dir: Optional[str] = None


def _get_ca_bundle_dir() -> str:
    global _ca_bundle_dir
    if _ca_bundle_dir is None:
        # mkdtemp yields a 0700, uniquely-named dir owned by us — an attacker
        # cannot pre-plant a file or symlink at our target paths.
        _ca_bundle_dir = tempfile.mkdtemp(prefix="datus-ca-")
    return _ca_bundle_dir


def materialize_ca_bundle(pem: str) -> str:
    """Write inline PEM CA content to a temp file and return its path.

    The litellm path only accepts a CA bundle via a file path, so PEM content
    must be spilled to disk there. Files live in a private per-process directory
    and are content-addressed (sha256) so repeated calls and shared CAs reuse
    the same file. A CA certificate is public, so no encryption is needed; the
    file is intentionally not cleaned up (tiny, bounded by distinct CAs, and
    cleared when the process's temp dir goes away).
    """
    digest = hashlib.sha256(pem.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(_get_ca_bundle_dir(), f"{digest}.pem")
    try:
        # Atomic exclusive create: O_EXCL refuses to follow a symlink or reuse an
        # existing file, and dedups concurrent writers of the same content.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path
    try:
        os.write(fd, pem.encode("utf-8"))
    finally:
        os.close(fd)
    return path
