# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OAuth flow manager for OpenAI Codex authentication.

Supports two authentication flows:
- Browser PKCE flow (interactive environments)
- Device Code flow (headless environments)
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
    DEVICE_CODE_POLL_INTERVAL,
    DEVICE_CODE_TIMEOUT,
    DEVICE_CODE_URL,
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

        server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
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
    # Device Code flow
    # ------------------------------------------------------------------

    def login_device(self) -> dict:
        """Authenticate via Device Code flow (headless environments).

        1. Request a device code and user code
        2. Prompt the user to visit the verification URL and enter the code
        3. Poll for token completion
        4. Store tokens

        Returns:
            Token dictionary with access_token, refresh_token, etc.
        """
        # Step 1: Request device code
        try:
            resp = httpx.post(
                DEVICE_CODE_URL,
                data={
                    "client_id": CLIENT_ID,
                    "scope": SCOPES,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise DatusException(
                ErrorCode.OAUTH_AUTH_FAILED,
                message_args={"error_detail": f"Device code request failed (HTTP {e.response.status_code})"},
            ) from e
        except httpx.TimeoutException as e:
            raise DatusException(ErrorCode.OAUTH_TIMEOUT) from e
        device_data = resp.json()

        user_code = device_data.get("user_code")
        verification_uri = device_data.get("verification_uri") or device_data.get("verification_url")
        device_code = device_data.get("device_code")
        interval = device_data.get("interval", DEVICE_CODE_POLL_INTERVAL)

        logger.info("Device code flow initiated. Visit the URL below to authenticate.")
        logger.info("Verification URL: %s", verification_uri)
        logger.info("User code: %s", user_code)

        # Step 2: Poll for completion
        deadline = time.monotonic() + DEVICE_CODE_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(interval)
            try:
                token_resp = httpx.post(
                    TOKEN_URL,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "client_id": CLIENT_ID,
                        "device_code": device_code,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=HTTP_TIMEOUT,
                )
            except httpx.TimeoutException:
                continue  # Retry on timeout during polling
            if token_resp.status_code == 200:
                tokens = token_resp.json()
                self.token_storage.save(tokens)
                logger.info("Device code OAuth login successful")
                return tokens

            try:
                error_body = token_resp.json()
            except Exception:
                raise DatusException(
                    ErrorCode.OAUTH_AUTH_FAILED,
                    message_args={"error_detail": f"Unexpected response (HTTP {token_resp.status_code})"},
                )
            error_code = error_body.get("error", "")
            if error_code == "authorization_pending":
                continue
            elif error_code == "slow_down":
                interval = min(interval + 5, 30)
                continue
            else:
                raise DatusException(ErrorCode.OAUTH_AUTH_FAILED, message_args={"error_detail": error_code})

        raise DatusException(ErrorCode.OAUTH_TIMEOUT)

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
                    tokens = self.refresh_tokens()

        return tokens["access_token"]

    def refresh_tokens(self) -> dict:
        """Refresh the access token using the stored refresh token.

        Returns:
            Updated token dictionary.
        """
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
        self.token_storage.clear()
        logger.info("OAuth tokens cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exchange_code(self, code: str, code_verifier: str) -> dict:
        """Exchange an authorization code for tokens."""
        try:
            resp = httpx.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": CLIENT_ID,
                    "code": code,
                    "redirect_uri": REDIRECT_URI,
                    "code_verifier": code_verifier,
                },
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
        return resp.json()
