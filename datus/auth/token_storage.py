# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Token storage for OAuth credentials with secure file permissions."""

import json
import os
import tempfile
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

        # Compute expires_at from expires_in if available (prefer server metadata)
        if "expires_in" in tokens and "expires_at" not in tokens:
            try:
                expires_in = float(tokens["expires_in"])
                tokens["expires_at"] = datetime.now(timezone.utc).timestamp() + expires_in
            except (TypeError, ValueError):
                logger.warning("Invalid expires_in in OAuth tokens; skipping expires_at derivation")

        dir_path = os.path.dirname(self.path) or "."
        if dir_path:
            os.makedirs(dir_path, mode=0o700, exist_ok=True)

        # Atomic write: write to temp file then rename (prevents partial reads)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_path)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(tokens, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.path)
                os.chmod(self.path, 0o600)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        except OSError as e:
            logger.error("Failed to save OAuth tokens to %s: %s", self.path, e)
            raise
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
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
                logger.debug("OAuth tokens cleared from %s", self.path)
        except OSError as e:
            logger.error("Failed to clear OAuth tokens from %s: %s", self.path, e)

    def needs_refresh(self) -> bool:
        """Check whether the stored token needs to be refreshed (reads from disk)."""
        tokens = self.load()
        return self.is_expired(tokens)

    def is_expired(self, tokens: Optional[dict]) -> bool:
        """Check whether the given tokens dict indicates expiry.

        Uses expires_at (server-derived) with a 60-second safety buffer,
        falling back to last_refresh + TOKEN_REFRESH_INTERVAL_SECONDS.
        """
        if not tokens:
            return True

        # Prefer expires_at (computed from server's expires_in during save)
        expires_at = tokens.get("expires_at")
        if expires_at is not None:
            try:
                return datetime.now(timezone.utc).timestamp() >= (float(expires_at) - 60)
            except (ValueError, TypeError):
                pass

        # Fallback to last_refresh + fixed interval
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
