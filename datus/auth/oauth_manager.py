# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OAuth flow manager for OpenAI Codex authentication.

Uses browser-based PKCE flow for interactive authentication.
"""

import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import httpx

from datus.auth.oauth_config import (
    AUTHORIZE_URL,
    CALLBACK_PORT,
    CLIENT_ID,
    HTTP_TIMEOUT,
    REDIRECT_URI,
    SCOPES,
    TOKEN_URL,
)
from datus.auth.pkce import generate_pkce_pair, generate_state
from datus.auth.token_storage import TokenStorage
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OAuthManager:
    """Manage OAuth authentication lifecycle for Codex API access."""

    def __init__(self, token_storage: Optional[TokenStorage] = None):
        self.token_storage = token_storage or TokenStorage()
        self._refresh_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Browser PKCE flow
    # ------------------------------------------------------------------

    def login_browser(self) -> dict:
        """Authenticate via browser-based PKCE flow.

        1. Generate PKCE pair and state
        2. Start a local HTTP callback server on port 1455
        3. Open the authorization URL in the default browser
        4. Wait for the callback with the authorization code
        5. Exchange the code for tokens
        6. Store tokens

        Returns:
            Token dictionary with access_token, refresh_token, etc.
        """
        code_verifier, code_challenge = generate_pkce_pair()
        state = generate_state()

        # Build authorization URL
        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

        # Container for the authorization code received by the callback
        result = {"code": None, "error": None}

        class _CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                # Only accept requests to the OAuth callback path
                if parsed.path != "/auth/callback":
                    self.send_response(404)
                    self.end_headers()
                    return

                qs = urllib.parse.parse_qs(parsed.query)
                returned_state = qs.get("state", [None])[0]
                if returned_state != state:
                    result["error"] = "State mismatch"
                elif "error" in qs:
                    result["error"] = qs["error"][0]
                else:
                    result["code"] = qs.get("code", [None])[0]

                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                if result["code"]:
                    self.wfile.write(b"<h1>Authentication successful! You can close this tab.</h1>")
                else:
                    self.wfile.write(b"<h1>Authentication failed. Please try again.</h1>")

            def log_message(self, format, *args):  # noqa: A002
                # Suppress default HTTP server logging
                pass

        try:
            server = HTTPServer(("localhost", CALLBACK_PORT), _CallbackHandler)
        except OSError as e:
            raise DatusException(
                ErrorCode.OAUTH_AUTH_FAILED,
                message_args={"error_detail": f"Could not start callback server on port {CALLBACK_PORT}: {e}"},
            ) from e
        server.timeout = 10  # Short timeout per handle_request; loop controls overall deadline

        # Open browser in a background thread so we can serve the callback
        threading.Thread(target=webbrowser.open, args=(auth_url,), daemon=True).start()

        logger.info("Waiting for OAuth callback on port %d ...", CALLBACK_PORT)
        deadline = time.monotonic() + 120  # 2 minutes overall
        while result["code"] is None and result["error"] is None:
            if time.monotonic() > deadline:
                break
            server.handle_request()
        server.server_close()

        if result["error"]:
            raise DatusException(ErrorCode.OAUTH_AUTH_FAILED, message_args={"error_detail": result["error"]})
        if not result["code"]:
            raise DatusException(
                ErrorCode.OAUTH_AUTH_FAILED,
                message_args={"error_detail": "No authorization code received (timeout or cancelled)"},
            )

        tokens = self._exchange_code(result["code"], code_verifier)
        self.token_storage.save(tokens)
        logger.info("Browser OAuth login successful")
        return tokens

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def get_access_token(self) -> str:
        """Return a valid access token, refreshing if needed.

        Thread-safe: uses a lock to prevent concurrent refresh races.

        Raises:
            DatusException: If not authenticated or refresh fails.
        """
        tokens = self.token_storage.load()
        if not tokens or "access_token" not in tokens:
            raise DatusException(ErrorCode.OAUTH_NOT_AUTHENTICATED)

        if self.token_storage.is_expired(tokens):
            with self._refresh_lock:
                # Re-check after acquiring lock (another thread may have refreshed)
                tokens = self.token_storage.load()
                if not tokens or self.token_storage.is_expired(tokens):
                    tokens = self._refresh_tokens_unlocked()

        return tokens["access_token"]

    def refresh_tokens(self) -> dict:
        """Refresh the access token using the stored refresh token.

        Thread-safe: acquires the refresh lock to prevent concurrent refresh races.

        Returns:
            Updated token dictionary.
        """
        with self._refresh_lock:
            return self._refresh_tokens_unlocked()

    def _refresh_tokens_unlocked(self) -> dict:
        """Internal refresh implementation (caller must hold _refresh_lock)."""
        tokens = self.token_storage.load()
        if not tokens or "refresh_token" not in tokens:
            raise DatusException(ErrorCode.OAUTH_NO_REFRESH_TOKEN)

        try:
            resp = httpx.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": CLIENT_ID,
                    "refresh_token": tokens["refresh_token"],
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise DatusException(
                ErrorCode.OAUTH_AUTH_FAILED,
                message_args={"error_detail": f"Token refresh failed (HTTP {e.response.status_code})"},
            ) from e
        except httpx.TimeoutException as e:
            raise DatusException(ErrorCode.OAUTH_TIMEOUT) from e
        except httpx.RequestError as e:
            raise DatusException(
                ErrorCode.OAUTH_AUTH_FAILED,
                message_args={"error_detail": f"Token refresh failed (network error: {e})"},
            ) from e
        new_tokens = resp.json()

        # Preserve refresh_token if the server didn't rotate it
        if "refresh_token" not in new_tokens:
            new_tokens["refresh_token"] = tokens["refresh_token"]

        self.token_storage.save(new_tokens)
        logger.info("OAuth tokens refreshed successfully")
        return new_tokens

    def is_authenticated(self) -> bool:
        """Check if valid tokens are stored."""
        tokens = self.token_storage.load()
        return tokens is not None and "access_token" in tokens

    def logout(self) -> None:
        """Clear stored tokens."""
        with self._refresh_lock:
            self.token_storage.clear()
        logger.info("OAuth tokens cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exchange_code(self, code: str, code_verifier: str) -> dict:
        """Exchange an authorization code for tokens."""
        data = {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": REDIRECT_URI,
        }
        try:
            resp = httpx.post(
                TOKEN_URL,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise DatusException(
                ErrorCode.OAUTH_AUTH_FAILED,
                message_args={"error_detail": f"Code exchange failed (HTTP {e.response.status_code})"},
            ) from e
        except httpx.TimeoutException as e:
            raise DatusException(ErrorCode.OAUTH_TIMEOUT) from e
        except httpx.RequestError as e:
            raise DatusException(
                ErrorCode.OAUTH_AUTH_FAILED,
                message_args={"error_detail": f"Code exchange failed (network error: {e})"},
            ) from e
        return resp.json()
