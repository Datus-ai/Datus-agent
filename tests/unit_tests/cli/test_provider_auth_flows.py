# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.provider_auth_flows``.

Subscription flow: auto-detection uses ``datus.auth.claude_credential``
and falls back to ``prompt_input`` on failure. OAuth flow: triggers a
browser login via ``datus.auth.oauth_manager.OAuthManager``.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.cli import provider_auth_flows

pytestmark = pytest.mark.ci


@pytest.fixture
def console():
    import io

    from rich.console import Console

    return Console(file=io.StringIO(), no_color=True)


@pytest.fixture
def provider_info_sub():
    return {
        "type": "claude",
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-6",
        "models": ["claude-sonnet-4-6", "claude-opus-4-6"],
        "auth_type": "subscription",
    }


@pytest.fixture
def provider_info_oauth():
    return {
        "type": "codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "default_model": "gpt-5.3-codex",
        "models": ["gpt-5.3-codex"],
        "auth_type": "oauth",
    }


class TestConfigureClaudeSubscription:
    def test_returns_dict_on_successful_auto_detect(self, console, provider_info_sub):
        with (
            patch(
                "datus.cli.provider_auth_flows.select_choice",
                return_value="claude-sonnet-4-6",
            ),
            patch(
                "datus.auth.claude_credential.get_claude_subscription_token",
                return_value=("sk-ant-oat01-auto", "credentials.json"),
            ),
        ):
            result = provider_auth_flows.configure_claude_subscription(
                console, "claude_subscription", provider_info_sub
            )
        assert isinstance(result, dict)
        assert result["model"] == "claude-sonnet-4-6"
        assert result["api_key"] == "sk-ant-oat01-auto"
        assert result["auth_type"] == "subscription"
        assert result["base_url"] == "https://api.anthropic.com"
        assert result["type"] == "claude"

    def test_falls_back_to_prompt_input_when_auto_detect_fails(self, console, provider_info_sub):
        with (
            patch(
                "datus.cli.provider_auth_flows.select_choice",
                return_value="claude-sonnet-4-6",
            ),
            patch(
                "datus.auth.claude_credential.get_claude_subscription_token",
                side_effect=Exception("missing creds"),
            ),
            patch("datus.cli.provider_auth_flows.prompt_input", return_value="sk-ant-oat01-manual"),
        ):
            result = provider_auth_flows.configure_claude_subscription(
                console, "claude_subscription", provider_info_sub
            )
        assert isinstance(result, dict)
        assert result["api_key"] == "sk-ant-oat01-manual"
        assert result["auth_type"] == "subscription"

    def test_returns_none_when_manual_input_is_empty(self, console, provider_info_sub):
        with (
            patch(
                "datus.cli.provider_auth_flows.select_choice",
                return_value="claude-sonnet-4-6",
            ),
            patch(
                "datus.auth.claude_credential.get_claude_subscription_token",
                side_effect=Exception("missing creds"),
            ),
            patch("datus.cli.provider_auth_flows.prompt_input", return_value=""),
        ):
            result = provider_auth_flows.configure_claude_subscription(
                console, "claude_subscription", provider_info_sub
            )
        assert result is None


class TestConfigureCodexOAuth:
    def test_successful_oauth_login(self, console, provider_info_oauth):
        with (
            patch("datus.cli.provider_auth_flows.select_choice", return_value="gpt-5.3-codex"),
            patch("datus.auth.oauth_manager.OAuthManager") as oauth_cls,
        ):
            oauth_cls.return_value = MagicMock()
            result = provider_auth_flows.configure_codex_oauth(console, "codex", provider_info_oauth)
        assert isinstance(result, dict)
        assert result["model"] == "gpt-5.3-codex"
        assert result["auth_type"] == "oauth"
        assert result["api_key"] == ""
        assert result["type"] == "codex"
        oauth_cls.return_value.login_browser.assert_called_once()

    def test_login_failure_returns_none(self, console, provider_info_oauth):
        with (
            patch("datus.cli.provider_auth_flows.select_choice", return_value="gpt-5.3-codex"),
            patch("datus.auth.oauth_manager.OAuthManager") as oauth_cls,
        ):
            oauth_cls.return_value.login_browser.side_effect = RuntimeError("redirect blocked")
            result = provider_auth_flows.configure_codex_oauth(console, "codex", provider_info_oauth)
        assert result is None

    def test_returns_none_when_model_selection_cancelled(self, console, provider_info_oauth):
        with patch("datus.cli.provider_auth_flows.select_choice", return_value=""):
            result = provider_auth_flows.configure_codex_oauth(console, "codex", provider_info_oauth)
        assert result is None
