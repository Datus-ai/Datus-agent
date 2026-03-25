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
    DEVICE_TOKEN_URL,
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
    # Device Code flow
    # ------------------------------------------------------------------

    def login_device(self) -> dict:
        """Authenticate via Device Code flow (headless environments).

        Convenience wrapper that calls request_device_code() then poll_device_token().

        Returns:
            Token dictionary with access_token, refresh_token, etc.
        """
        self.request_device_code()
        return self.poll_device_token()

    def request_device_code(self) -> dict:
        """Request a device code and user code from the auth server.

        After calling this, display _device_verification_uri and _device_user_code
        to the user, then call poll_device_token() to wait for completion.

        Returns:
            The raw device code response data.
        """
        try:
            resp = httpx.post(
                DEVICE_CODE_URL,
                json={"client_id": CLIENT_ID},
                headers={"Content-Type": "application/json"},
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
        except httpx.RequestError as e:
            raise DatusException(
                ErrorCode.OAUTH_AUTH_FAILED,
                message_args={"error_detail": f"Device code request failed (network error: {e})"},
            ) from e
        device_data = resp.json()

        self._device_user_code = device_data.get("user_code")
        # Codex server doesn't return verification_uri — it's constructed client-side
        self._device_verification_uri = (
            device_data.get("verification_uri")
            or device_data.get("verification_url")
            or device_data.get("verification_uri_complete")
            or "https://auth.openai.com/codex/device"
        )
        self._device_auth_id = device_data.get("device_auth_id") or device_data.get("device_code")
        self._device_poll_interval = int(device_data.get("interval", DEVICE_CODE_POLL_INTERVAL))

        logger.info("Device code flow initiated. Visit the URL below to authenticate.")
        logger.info("Verification URL: %s", self._device_verification_uri)
        logger.debug("Device code response fields: %s", list(device_data.keys()))
        logger.info("User code: %s", self._device_user_code)
        return device_data

    def poll_device_token(self) -> dict:
        """Poll the auth server until the user completes device code authentication.

        Must be called after request_device_code().

        Returns:
            Token dictionary with access_token, refresh_token, etc.
        """
        device_auth_id = self._device_auth_id
        user_code = self._device_user_code
        interval = self._device_poll_interval

        deadline = time.monotonic() + DEVICE_CODE_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(interval)
            try:
                token_resp = httpx.post(
                    DEVICE_TOKEN_URL,
                    json={
                        "device_auth_id": device_auth_id,
                        "user_code": user_code,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=HTTP_TIMEOUT,
                )
            except (httpx.TimeoutException, httpx.RequestError):
                continue  # Retry on timeout or network error during polling
            if token_resp.status_code == 200:
                device_data = token_resp.json()
                # Codex device code flow returns authorization_code + code_verifier,
                # not access_token directly. Exchange them for real tokens.
                auth_code = device_data.get("authorization_code")
                verifier = device_data.get("code_verifier")
                if auth_code and verifier:
                    tokens = self._exchange_code(auth_code, verifier, redirect_uri=None)
                else:
                    tokens = device_data
                self.token_storage.save(tokens)
                logger.info("Device code OAuth login successful")
                return tokens

            # Codex server returns 403/404 while user hasn't completed auth yet
            if token_resp.status_code in (403, 404):
                continue

            try:
                error_body = token_resp.json()
            except Exception as e:
                raise DatusException(
                    ErrorCode.OAUTH_AUTH_FAILED,
                    message_args={"error_detail": f"Unexpected response (HTTP {token_resp.status_code})"},
                ) from e
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

    def _exchange_code(self, code: str, code_verifier: str, redirect_uri: Optional[str] = REDIRECT_URI) -> dict:
        """Exchange an authorization code for tokens."""
        data = {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": code_verifier,
        }
        if redirect_uri:
            data["redirect_uri"] = redirect_uri
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
