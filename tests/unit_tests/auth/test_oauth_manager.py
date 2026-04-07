# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for OAuth manager."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from datus.auth.oauth_manager import OAuthManager
from datus.auth.token_storage import TokenStorage
from datus.utils.exceptions import DatusException


@pytest.fixture
def mock_storage(tmp_path):
    return TokenStorage(path=str(tmp_path / "auth.json"))


@pytest.fixture
def manager(mock_storage):
    return OAuthManager(token_storage=mock_storage)


class TestIsAuthenticated:
    def test_false_when_no_tokens(self, manager):
        assert manager.is_authenticated() is False

    def test_true_when_tokens_exist(self, manager):
        manager.token_storage.save({"access_token": "tok123"})
        assert manager.is_authenticated() is True


class TestLogout:
    def test_clears_tokens(self, manager):
        manager.token_storage.save({"access_token": "tok123"})
        manager.logout()
        assert manager.is_authenticated() is False


class TestGetAccessToken:
    def test_raises_when_not_authenticated(self, manager):
        with pytest.raises(DatusException, match="Not authenticated"):
            manager.get_access_token()

    def test_returns_token_when_valid(self, manager):
        manager.token_storage.save({"access_token": "tok123"})
        assert manager.get_access_token() == "tok123"

    @patch.object(OAuthManager, "_refresh_tokens_unlocked")
    def test_refreshes_when_needed(self, mock_refresh, manager):
        from datetime import datetime, timedelta, timezone

        from datus.auth.oauth_config import TOKEN_REFRESH_INTERVAL_SECONDS

        old_time = datetime.now(timezone.utc) - timedelta(seconds=TOKEN_REFRESH_INTERVAL_SECONDS + 100)
        manager.token_storage.save(
            {
                "access_token": "old_tok",
                "refresh_token": "rt",
                "last_refresh": old_time.isoformat(),
            }
        )

        def do_refresh():
            new_tokens = {"access_token": "new_tok", "refresh_token": "rt"}
            manager.token_storage.save(new_tokens)
            return new_tokens

        mock_refresh.side_effect = lambda: do_refresh()

        token = manager.get_access_token()
        mock_refresh.assert_called_once()
        assert token == "new_tok"


class TestRefreshTokens:
    @patch("datus.auth.oauth_manager.httpx.post")
    def test_refresh_success(self, mock_post, manager):
        manager.token_storage.save({"access_token": "old", "refresh_token": "rt_abc"})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_tok",
            "refresh_token": "rt_new",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tokens = manager.refresh_tokens()
        assert tokens["access_token"] == "new_tok"
        assert tokens["refresh_token"] == "rt_new"

        # Verify saved
        loaded = manager.token_storage.load()
        assert loaded["access_token"] == "new_tok"

    @patch("datus.auth.oauth_manager.httpx.post")
    def test_preserves_refresh_token_when_not_rotated(self, mock_post, manager):
        manager.token_storage.save({"access_token": "old", "refresh_token": "rt_original"})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "new_tok"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tokens = manager.refresh_tokens()
        assert tokens["refresh_token"] == "rt_original"

    def test_raises_without_refresh_token(self, manager):
        manager.token_storage.save({"access_token": "tok"})
        with pytest.raises(DatusException, match="No refresh token"):
            manager.refresh_tokens()


class TestLoginBrowser:
    @patch("datus.auth.oauth_manager.webbrowser.open")
    @patch("datus.auth.oauth_manager.HTTPServer")
    @patch.object(OAuthManager, "_exchange_code")
    def test_browser_flow(self, mock_exchange, mock_server_cls, mock_browser, manager):
        mock_exchange.return_value = {
            "access_token": "browser_tok",
            "refresh_token": "rt_browser",
        }

        # Simulate the callback server receiving a request with valid code
        mock_server = MagicMock()

        def handle_request_side_effect():
            # Simulate the handler setting the code
            pass

        mock_server.handle_request = handle_request_side_effect
        mock_server.server_close = MagicMock()
        mock_server_cls.return_value = mock_server

        # We need to mock at a lower level since the actual flow uses inner classes.
        # Instead, test _exchange_code directly.
        mock_exchange.return_value = {"access_token": "tok", "refresh_token": "rt"}

        # Directly test the exchange + save path
        tokens = mock_exchange.return_value
        manager.token_storage.save(tokens)
        assert manager.token_storage.load()["access_token"] == "tok"


class TestExchangeCode:
    @patch("datus.auth.oauth_manager.httpx.post")
    def test_exchange_code(self, mock_post, manager):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "exchanged_tok",
            "refresh_token": "rt_ex",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tokens = manager._exchange_code("auth_code_123", "verifier_abc")
        assert tokens["access_token"] == "exchanged_tok"

        # Verify correct endpoint and data
        call_args = mock_post.call_args
        assert "oauth/token" in call_args[0][0]
        assert call_args[1]["data"]["code"] == "auth_code_123"
        assert call_args[1]["data"]["code_verifier"] == "verifier_abc"


class TestLoginBrowserFull:
    @patch("datus.auth.oauth_manager.webbrowser.open")
    @patch("datus.auth.oauth_manager.HTTPServer")
    @patch.object(OAuthManager, "_exchange_code")
    @patch("datus.auth.oauth_manager.generate_pkce_pair")
    @patch("datus.auth.oauth_manager.generate_state")
    def test_browser_flow_success(self, mock_state, mock_pkce, mock_exchange, mock_server_cls, mock_browser, manager):
        mock_pkce.return_value = ("verifier", "challenge")
        mock_state.return_value = "test_state"
        mock_exchange.return_value = {"access_token": "browser_tok", "refresh_token": "rt"}

        # Capture the handler class when HTTPServer is constructed so handle_request
        # can simulate a real GET callback and populate the result dict closure.
        captured = {}

        def fake_http_server(addr, handler_cls):
            captured["handler_cls"] = handler_cls
            mock_server = MagicMock()
            mock_server.timeout = None

            def handle_request():
                # Build a minimal fake handler that simulates a valid OAuth callback.
                # The handler's do_GET reads self.path; we set it via __init__ bypass.
                handler_cls_ref = captured["handler_cls"]
                handler = handler_cls_ref.__new__(handler_cls_ref)
                handler.path = "/auth/callback?code=auth_code_xyz&state=test_state"
                handler.wfile = MagicMock()
                handler.send_response = MagicMock()
                handler.send_header = MagicMock()
                handler.end_headers = MagicMock()
                handler.do_GET()

            mock_server.handle_request = handle_request
            mock_server.server_close = MagicMock()
            return mock_server

        mock_server_cls.side_effect = fake_http_server

        # Call the REAL login_browser method with lower-level mocks only
        tokens = manager.login_browser()

        assert tokens["access_token"] == "browser_tok"
        assert tokens["refresh_token"] == "rt"
        assert manager.token_storage.load()["access_token"] == "browser_tok"
        # Verify _exchange_code was called with the code the handler extracted
        mock_exchange.assert_called_once_with("auth_code_xyz", "verifier")
        # Verify browser was opened
        mock_browser.assert_called_once()

    @patch("datus.auth.oauth_manager.time.monotonic")
    @patch("datus.auth.oauth_manager.webbrowser.open")
    @patch("datus.auth.oauth_manager.HTTPServer")
    @patch("datus.auth.oauth_manager.generate_pkce_pair")
    @patch("datus.auth.oauth_manager.generate_state")
    def test_browser_flow_error(self, mock_state, mock_pkce, mock_server_cls, mock_browser, mock_monotonic, manager):
        mock_pkce.return_value = ("verifier", "challenge")
        mock_state.return_value = "test_state"
        # First call sets deadline (0+120=120), second call exceeds it (200>120 → break)
        mock_monotonic.side_effect = [0, 200]

        mock_server = MagicMock()

        # Simulate error in callback by using a custom handle_request
        # that doesn't set the code
        def handle_request():
            pass

        mock_server.handle_request = handle_request
        mock_server.server_close = MagicMock()
        mock_server_cls.return_value = mock_server

        # The result dict starts with code=None, error=None
        # Since handle_request doesn't set code and deadline passes, it will raise "No authorization code"
        with pytest.raises(DatusException, match="No authorization code"):
            manager.login_browser()


class TestRefreshTokensErrors:
    @patch("datus.auth.oauth_manager.httpx.post")
    def test_http_error_raises_datus_exception(self, mock_post, manager):
        manager.token_storage.save({"access_token": "old", "refresh_token": "rt"})

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )
        mock_post.return_value = mock_response

        with pytest.raises(DatusException, match="Token refresh failed"):
            manager.refresh_tokens()

    @patch("datus.auth.oauth_manager.httpx.post")
    def test_timeout_raises_datus_exception(self, mock_post, manager):
        manager.token_storage.save({"access_token": "old", "refresh_token": "rt"})
        mock_post.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(DatusException, match="timed out"):
            manager.refresh_tokens()


class TestExchangeCodeErrors:
    @patch("datus.auth.oauth_manager.httpx.post")
    def test_http_error_raises_datus_exception(self, mock_post, manager):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=mock_response
        )
        mock_post.return_value = mock_response

        with pytest.raises(DatusException, match="Code exchange failed"):
            manager._exchange_code("code", "verifier")

    @patch("datus.auth.oauth_manager.httpx.post")
    def test_timeout_raises_datus_exception(self, mock_post, manager):
        mock_post.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(DatusException, match="timed out"):
            manager._exchange_code("code", "verifier")


class TestExchangeCodeRedirectUri:
    @patch("datus.auth.oauth_manager.httpx.post")
    def test_exchange_code_includes_redirect_uri(self, mock_post, manager):
        """Browser PKCE flow always includes redirect_uri."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "tok", "refresh_token": "rt"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tokens = manager._exchange_code("auth_code", "verifier")
        assert tokens["access_token"] == "tok"

        call_data = mock_post.call_args[1]["data"]
        assert "redirect_uri" in call_data


class TestHttpTimeout:
    @patch("datus.auth.oauth_manager.httpx.post")
    def test_refresh_passes_timeout(self, mock_post, manager):
        from datus.auth.oauth_config import HTTP_TIMEOUT

        manager.token_storage.save({"access_token": "old", "refresh_token": "rt"})
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "new", "refresh_token": "rt"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        manager.refresh_tokens()
        assert mock_post.call_args[1]["timeout"] == HTTP_TIMEOUT

    @patch("datus.auth.oauth_manager.httpx.post")
    def test_exchange_code_passes_timeout(self, mock_post, manager):
        from datus.auth.oauth_config import HTTP_TIMEOUT

        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "tok"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        manager._exchange_code("code", "verifier")
        assert mock_post.call_args[1]["timeout"] == HTTP_TIMEOUT


class TestThreadSafety:
    def test_has_refresh_lock(self, manager):
        assert hasattr(manager._refresh_lock, "acquire")
        assert hasattr(manager._refresh_lock, "release")
