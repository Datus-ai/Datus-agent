# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/auth/claude_credential.py.

CI-level: zero external dependencies. File system and env vars are mocked.
"""

import json
from unittest.mock import patch

import pytest

from datus.auth.claude_credential import get_claude_subscription_token
from datus.utils.exceptions import DatusException, ErrorCode


class TestGetClaudeSubscriptionToken:
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
        """Priority 3: ~/.claude/.credentials.json."""
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
