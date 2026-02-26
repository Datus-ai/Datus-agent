# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Token persistence for Town Skills Marketplace authentication.

Stores JWT tokens at ~/.datus/marketplace_auth.json so that
`datus skill` commands can authenticate with the Town backend.
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AUTH_FILE = Path.home() / ".datus" / "marketplace_auth.json"


def _read_auth_file() -> dict:
    """Read the auth file, returning an empty dict if missing or invalid."""
    if not AUTH_FILE.exists():
        return {}
    try:
        return json.loads(AUTH_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_auth_file(data: dict) -> None:
    """Write the auth file, creating parent dirs as needed."""
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _is_token_expired(token: str) -> bool:
    """Check if a JWT token is expired by decoding its payload.

    Returns True if the token is expired or cannot be decoded.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return True
        # Add padding for base64 decoding
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        if exp is None:
            return False  # No expiry claim — assume valid
        return time.time() >= exp
    except Exception:
        return True


def save_token(token: str, marketplace_url: str, email: str) -> None:
    """Save a JWT token for the given marketplace URL."""
    data = _read_auth_file()
    url_key = marketplace_url.rstrip("/")
    data[url_key] = {
        "token": token,
        "email": email,
    }
    _write_auth_file(data)
    logger.debug("Saved marketplace token for %s", url_key)


def load_token(marketplace_url: str) -> Optional[str]:
    """Load a saved JWT token for the given marketplace URL.

    Returns None if no token is stored or the token has expired.
    """
    data = _read_auth_file()
    url_key = marketplace_url.rstrip("/")
    entry = data.get(url_key)
    if not entry:
        return None
    token = entry.get("token")
    if not token:
        return None
    if _is_token_expired(token):
        logger.debug("Stored token for %s has expired", url_key)
        return None
    return token


def clear_token(marketplace_url: str) -> bool:
    """Remove saved credentials for a marketplace URL.

    Returns True if an entry was removed.
    """
    data = _read_auth_file()
    url_key = marketplace_url.rstrip("/")
    if url_key in data:
        del data[url_key]
        _write_auth_file(data)
        logger.debug("Cleared marketplace token for %s", url_key)
        return True
    return False
