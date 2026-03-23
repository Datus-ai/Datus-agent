# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Token storage for OAuth credentials with secure file permissions."""

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from datus.auth.oauth_config import TOKEN_REFRESH_INTERVAL_SECONDS
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Default storage path
_DEFAULT_AUTH_PATH = os.path.join(str(Path.home()), ".datus", "auth.json")


class TokenStorage:
    """Persist OAuth tokens to disk with secure file permissions (0o600)."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get("DATUS_AUTH_PATH", _DEFAULT_AUTH_PATH)

    def save(self, tokens: dict) -> None:
        """Save tokens to disk with restricted permissions.

        Args:
            tokens: Dictionary containing access_token, refresh_token, etc.
        """
        tokens = dict(tokens)
        tokens.setdefault("last_refresh", datetime.now(timezone.utc).isoformat())

        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2)

        # Set file permissions to owner-only read/write (0o600)
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        logger.debug("OAuth tokens saved to %s", self.path)

    def load(self) -> Optional[dict]:
        """Load tokens from disk.

        Returns:
            Token dictionary, or None if file does not exist or is invalid.
        """
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load OAuth tokens from %s: %s", self.path, e)
            return None

    def clear(self) -> None:
        """Remove the stored token file."""
        if os.path.exists(self.path):
            os.remove(self.path)
            logger.debug("OAuth tokens cleared from %s", self.path)

    def needs_refresh(self) -> bool:
        """Check whether the stored token needs to be refreshed.

        Returns:
            True if tokens are missing, have no last_refresh timestamp,
            or if more than TOKEN_REFRESH_INTERVAL_SECONDS have elapsed.
        """
        tokens = self.load()
        if not tokens:
            return True
        last_refresh_str = tokens.get("last_refresh")
        if not last_refresh_str:
            return True
        try:
            last_refresh = datetime.fromisoformat(last_refresh_str)
            if last_refresh.tzinfo is None:
                last_refresh = last_refresh.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last_refresh).total_seconds()
            return elapsed >= TOKEN_REFRESH_INTERVAL_SECONDS
        except (ValueError, TypeError):
            return True
