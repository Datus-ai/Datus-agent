# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/auth/claude_credential.py.

CI-level: zero external dependencies. File system and env vars are mocked.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from datus.auth.claude_credential import (
    _extract_oauth_token,
    _read_keychain_credentials,
    get_claude_subscription_token,
)
from datus.utils.exceptions import DatusException, ErrorCode


class TestGetClaudeSubscriptionToken:
    @pytest.fixture(autouse=True)
    def _no_keychain(self):
        with patch("datus.auth.claude_credential._read_keychain_credentials", return_value=None):
            yield

    def test_returns_config_api_key(self):
        """Priority 1: config api_key takes precedence."""
        token, source = get_claude_subscription_token(api_key_from_config="sk-ant-oat01-config-token")
        assert token == "sk-ant-oat01-config-token"
        assert "config" in source

    def test_ignores_empty_config_key(self):
        """Empty string config key should be skipped."""
        with patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-env-token"}):
            token, source = get_claude_subscription_token(api_key_from_config="")
            assert token == "sk-ant-oat01-env-token"
            assert "CLAUDE_CODE_OAUTH_TOKEN" in source

    def test_ignores_whitespace_config_key(self):
        """Whitespace-only config key should be skipped."""
        with patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-env-token"}):
            token, source = get_claude_subscription_token(api_key_from_config="   ")
            assert token == "sk-ant-oat01-env-token"
            assert "CLAUDE_CODE_OAUTH_TOKEN" in source

    def test_returns_env_var(self):
        """Priority 2: CLAUDE_CODE_OAUTH_TOKEN env var."""
        with patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-env-token"}):
            token, source = get_claude_subscription_token(api_key_from_config=None)
            assert token == "sk-ant-oat01-env-token"
            assert "CLAUDE_CODE_OAUTH_TOKEN" in source

    def test_reads_credentials_file(self, tmp_path):
        """Priority 4: ~/.claude/.credentials.json."""
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        cred_file = cred_dir / ".credentials.json"
        cred_file.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-file-token"}}),
            encoding="utf-8",
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            token, source = get_claude_subscription_token(api_key_from_config=None)
            assert token == "sk-ant-oat01-file-token"
            assert ".credentials.json" in source

    def test_raises_when_not_found(self, tmp_path):
        """Raises DatusException when no token source is available."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            with pytest.raises(DatusException) as exc_info:
                get_claude_subscription_token(api_key_from_config=None)
            assert exc_info.value.code == ErrorCode.CLAUDE_SUBSCRIPTION_TOKEN_NOT_FOUND

    def test_ignores_malformed_credentials_file(self, tmp_path):
        """Malformed JSON in credentials file should be skipped gracefully."""
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        cred_file = cred_dir / ".credentials.json"
        cred_file.write_text("not valid json{{{", encoding="utf-8")

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            with pytest.raises(DatusException):
                get_claude_subscription_token(api_key_from_config=None)

    def test_ignores_credentials_file_without_token(self, tmp_path):
        """Credentials file exists but missing accessToken field."""
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        cred_file = cred_dir / ".credentials.json"
        cred_file.write_text(json.dumps({"otherField": "value"}), encoding="utf-8")

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            with pytest.raises(DatusException):
                get_claude_subscription_token(api_key_from_config=None)

    def test_config_key_takes_priority_over_env_and_file(self, tmp_path):
        """Config api_key wins even when env var and file are available."""
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        cred_file = cred_dir / ".credentials.json"
        cred_file.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-file"}}),
            encoding="utf-8",
        )

        with (
            patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-env"}),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            token, source = get_claude_subscription_token(api_key_from_config="sk-ant-oat01-config")
            assert token == "sk-ant-oat01-config"
            assert "config" in source

    def test_ignores_missing_placeholder(self):
        """<MISSING:...> placeholder from resolve_env should be skipped."""
        with patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-real-token"}):
            token, source = get_claude_subscription_token(api_key_from_config="<MISSING:CLAUDE_CODE_OAUTH_TOKEN>")
            assert token == "sk-ant-oat01-real-token"
            assert "CLAUDE_CODE_OAUTH_TOKEN" in source

    def test_missing_placeholder_without_fallback_raises(self, tmp_path):
        """<MISSING:...> placeholder with no env/file raises DatusException."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            with pytest.raises(DatusException) as exc_info:
                get_claude_subscription_token(api_key_from_config="<MISSING:CLAUDE_CODE_OAUTH_TOKEN>")
            assert exc_info.value.code == ErrorCode.CLAUDE_SUBSCRIPTION_TOKEN_NOT_FOUND

    def test_skips_expired_credentials_file_token(self, tmp_path):
        """Expired token in credentials file should be skipped."""
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        cred_file = cred_dir / ".credentials.json"
        # expiresAt is in milliseconds, set to a past time
        cred_file.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-expired", "expiresAt": 1000000}}),
            encoding="utf-8",
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            with pytest.raises(DatusException) as exc_info:
                get_claude_subscription_token(api_key_from_config=None)
            assert exc_info.value.code == ErrorCode.CLAUDE_SUBSCRIPTION_TOKEN_NOT_FOUND

    def test_returns_non_expired_credentials_file_token(self, tmp_path):
        """Non-expired token in credentials file should be returned."""
        import time

        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        cred_file = cred_dir / ".credentials.json"
        # expiresAt far in the future (in milliseconds)
        future_ms = int((time.time() + 3600) * 1000)
        cred_file.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-valid", "expiresAt": future_ms}}),
            encoding="utf-8",
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            token, source = get_claude_subscription_token(api_key_from_config=None)
            assert token == "sk-ant-oat01-valid"
            assert ".credentials.json" in source

    def test_returns_token_without_expiry_field(self, tmp_path):
        """Token without expiresAt field should still be returned (no expiry check)."""
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        cred_file = cred_dir / ".credentials.json"
        cred_file.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-no-expiry"}}),
            encoding="utf-8",
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            token, source = get_claude_subscription_token(api_key_from_config=None)
            assert token == "sk-ant-oat01-no-expiry"
            assert ".credentials.json" in source

    def test_ignores_unresolved_env_placeholder(self):
        """${VAR} placeholder (unresolved env substitution) should be skipped."""
        with patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-real-token"}):
            token, source = get_claude_subscription_token(api_key_from_config="${CLAUDE_CODE_OAUTH_TOKEN}")
            assert token == "sk-ant-oat01-real-token"
            assert "CLAUDE_CODE_OAUTH_TOKEN" in source

    def test_unresolved_env_placeholder_without_fallback_raises(self, tmp_path):
        """${VAR} placeholder with no env/file raises DatusException."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            with pytest.raises(DatusException) as exc_info:
                get_claude_subscription_token(api_key_from_config="${CLAUDE_CODE_OAUTH_TOKEN}")
            assert exc_info.value.code == ErrorCode.CLAUDE_SUBSCRIPTION_TOKEN_NOT_FOUND

    def test_env_var_takes_priority_over_file(self, tmp_path):
        """Env var wins when config is empty and file exists."""
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        cred_file = cred_dir / ".credentials.json"
        cred_file.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-file"}}),
            encoding="utf-8",
        )

        with (
            patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-env"}),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
        ):
            token, source = get_claude_subscription_token(api_key_from_config=None)
            assert token == "sk-ant-oat01-env"
            assert "CLAUDE_CODE_OAUTH_TOKEN" in source


class TestReadKeychainCredentials:
    def test_returns_none_on_non_darwin(self):
        with patch("datus.auth.claude_credential.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert _read_keychain_credentials() is None

    def test_returns_parsed_json_on_success(self):
        keychain_json = json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-kc"}})
        mock_result = MagicMock(returncode=0, stdout=keychain_json)
        with (
            patch("datus.auth.claude_credential.sys") as mock_sys,
            patch("datus.auth.claude_credential.subprocess.run", return_value=mock_result),
        ):
            mock_sys.platform = "darwin"
            data = _read_keychain_credentials()
            assert data == {"claudeAiOauth": {"accessToken": "sk-ant-oat01-kc"}}

    def test_returns_none_on_nonzero_returncode(self):
        mock_result = MagicMock(returncode=44, stdout="", stderr="not found")
        with (
            patch("datus.auth.claude_credential.sys") as mock_sys,
            patch("datus.auth.claude_credential.subprocess.run", return_value=mock_result),
        ):
            mock_sys.platform = "darwin"
            assert _read_keychain_credentials() is None

    def test_returns_none_on_malformed_json(self):
        mock_result = MagicMock(returncode=0, stdout="not-valid-json{{{")
        with (
            patch("datus.auth.claude_credential.sys") as mock_sys,
            patch("datus.auth.claude_credential.subprocess.run", return_value=mock_result),
        ):
            mock_sys.platform = "darwin"
            assert _read_keychain_credentials() is None

    def test_returns_none_on_timeout(self):
        with (
            patch("datus.auth.claude_credential.sys") as mock_sys,
            patch(
                "datus.auth.claude_credential.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="security", timeout=5),
            ),
        ):
            mock_sys.platform = "darwin"
            assert _read_keychain_credentials() is None

    def test_returns_none_when_security_not_found(self):
        with (
            patch("datus.auth.claude_credential.sys") as mock_sys,
            patch("datus.auth.claude_credential.subprocess.run", side_effect=FileNotFoundError),
        ):
            mock_sys.platform = "darwin"
            assert _read_keychain_credentials() is None


class TestExtractOauthToken:
    def test_returns_token_and_source(self):
        data = {"claudeAiOauth": {"accessToken": "sk-ant-oat01-test"}}
        result = _extract_oauth_token(data, "test-source")
        assert result == ("sk-ant-oat01-test", "test-source")

    def test_returns_none_for_missing_token(self):
        assert _extract_oauth_token({"claudeAiOauth": {}}, "src") is None
        assert _extract_oauth_token({}, "src") is None

    def test_returns_none_for_expired_token(self):
        data = {"claudeAiOauth": {"accessToken": "sk-ant-oat01-expired", "expiresAt": 1000000}}
        assert _extract_oauth_token(data, "src") is None

    def test_returns_token_when_not_expired(self):
        import time

        future_ms = int((time.time() + 3600) * 1000)
        data = {"claudeAiOauth": {"accessToken": "sk-ant-oat01-valid", "expiresAt": future_ms}}
        result = _extract_oauth_token(data, "src")
        assert result is not None
        assert result[0] == "sk-ant-oat01-valid"


class TestKeychainIntegration:
    """Tests for Keychain as Priority 3 in get_claude_subscription_token."""

    def test_keychain_returns_valid_token(self, tmp_path):
        keychain_json = json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-kc-token"}})
        mock_result = MagicMock(returncode=0, stdout=keychain_json)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
            patch("datus.auth.claude_credential.sys") as mock_sys,
            patch("datus.auth.claude_credential.subprocess.run", return_value=mock_result),
        ):
            mock_sys.platform = "darwin"
            token, source = get_claude_subscription_token(api_key_from_config=None)
            assert token == "sk-ant-oat01-kc-token"
            assert "Keychain" in source

    def test_keychain_takes_priority_over_file(self, tmp_path):
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        (cred_dir / ".credentials.json").write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-file"}}),
            encoding="utf-8",
        )
        keychain_json = json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-kc"}})
        mock_result = MagicMock(returncode=0, stdout=keychain_json)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
            patch("datus.auth.claude_credential.sys") as mock_sys,
            patch("datus.auth.claude_credential.subprocess.run", return_value=mock_result),
        ):
            mock_sys.platform = "darwin"
            token, source = get_claude_subscription_token(api_key_from_config=None)
            assert token == "sk-ant-oat01-kc"
            assert "Keychain" in source

    def test_env_var_takes_priority_over_keychain(self):
        with (
            patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-env"}),
            patch("datus.auth.claude_credential.subprocess.run") as mock_run,
        ):
            token, source = get_claude_subscription_token(api_key_from_config=None)
            assert token == "sk-ant-oat01-env"
            assert "CLAUDE_CODE_OAUTH_TOKEN" in source
            mock_run.assert_not_called()

    def test_keychain_not_found_falls_through_to_file(self, tmp_path):
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        (cred_dir / ".credentials.json").write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-file"}}),
            encoding="utf-8",
        )
        mock_result = MagicMock(returncode=44, stdout="", stderr="not found")
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
            patch("datus.auth.claude_credential.sys") as mock_sys,
            patch("datus.auth.claude_credential.subprocess.run", return_value=mock_result),
        ):
            mock_sys.platform = "darwin"
            token, source = get_claude_subscription_token(api_key_from_config=None)
            assert token == "sk-ant-oat01-file"
            assert ".credentials.json" in source

    def test_keychain_expired_token_falls_through_to_file(self, tmp_path):
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        (cred_dir / ".credentials.json").write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-file-valid"}}),
            encoding="utf-8",
        )
        keychain_json = json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-expired", "expiresAt": 1000000}})
        mock_result = MagicMock(returncode=0, stdout=keychain_json)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
            patch("datus.auth.claude_credential.sys") as mock_sys,
            patch("datus.auth.claude_credential.subprocess.run", return_value=mock_result),
        ):
            mock_sys.platform = "darwin"
            token, source = get_claude_subscription_token(api_key_from_config=None)
            assert token == "sk-ant-oat01-file-valid"
            assert ".credentials.json" in source

    def test_keychain_skipped_on_non_darwin(self, tmp_path):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("datus.auth.claude_credential.Path.home", return_value=tmp_path),
            patch("datus.auth.claude_credential.sys") as mock_sys,
            patch("datus.auth.claude_credential.subprocess.run") as mock_run,
        ):
            mock_sys.platform = "linux"
            with pytest.raises(DatusException):
                get_claude_subscription_token(api_key_from_config=None)
            mock_run.assert_not_called()
