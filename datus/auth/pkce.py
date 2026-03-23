# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""PKCE (Proof Key for Code Exchange) utilities for OAuth 2.0 authorization."""

import base64
import hashlib
import secrets


def generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code_verifier and code_challenge pair.

    Returns:
        A tuple of (code_verifier, code_challenge) where:
        - code_verifier: base64url-encoded 64 random bytes
        - code_challenge: base64url(SHA256(code_verifier))
    """
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode("ascii")
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    )
    return code_verifier, code_challenge


def generate_state() -> str:
    """Generate a random state parameter for OAuth 2.0 authorization.

    Returns:
        A base64url-encoded 32-byte random string.
    """
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
