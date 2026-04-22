# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Provider authentication helper flows shared by ``datus init`` and ``/model``.

Both call sites need to (a) pick a model name from the provider catalog,
(b) acquire a credential (subscription token or OAuth login), and
(c) hand the result back to the caller for persistence and connectivity
testing. Centralizing the logic here keeps ``interactive_init.py`` and
``model_commands.py`` from drifting on the credential-capture details.

Each flow returns a dict on success or ``None`` on cancellation / failure.
Connectivity verification is intentionally out of scope — callers handle
that via :meth:`datus.models.base.LLMBaseModel.test_connection`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rich.console import Console

from datus.cli._cli_utils import prompt_input, select_choice
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _select_model(console: Console, provider_config: Dict[str, Any]) -> str:
    """Render the provider's ``models`` list and return the chosen name.

    Falls back to a free-text prompt when the catalog does not ship a
    model list for this provider (e.g. a BYO endpoint).
    """
    models = provider_config.get("models", [])
    default_model = provider_config.get("default_model", "")
    if models:
        console.print("- Select your model:")
        return select_choice(
            console,
            {str(m): str(m) for m in models},
            default=str(default_model) if default_model else str(models[0]),
            allow_free_text=True,
        )
    return prompt_input(console, "Enter your model name", default=str(default_model)).strip()


def configure_claude_subscription(
    console: Console,
    provider: str,
    provider_config: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """Acquire a Claude Pro/Max subscription token and pick a model.

    Args:
        console: Rich console for user prompts.
        provider: Provider key (e.g. ``claude_subscription``) — kept for
            error messages so callers can log which entry failed.
        provider_config: The ``providers.yml`` entry describing the
            provider (``type``, ``base_url``, ``default_model``, ``models``).

    Returns:
        ``{"model": str, "api_key": str, "auth_type": "subscription",
        "base_url": str, "type": str}`` on success; ``None`` when the user
        cancels or the token cannot be obtained.
    """
    model_name = _select_model(console, provider_config)
    if not model_name:
        return None

    token = _get_subscription_token(console)
    if not token:
        return None

    return {
        "model": model_name,
        "api_key": token,
        "auth_type": "subscription",
        "base_url": str(provider_config.get("base_url", "")),
        "type": str(provider_config.get("type", provider)),
    }


def _get_subscription_token(console: Console) -> Optional[str]:
    """Auto-detect the Claude subscription token, else prompt the user.

    Returns the token string on success, ``None`` if the user provides no
    input. Auto-detection failures fall through to the manual prompt
    rather than aborting outright, since ``claude setup-token`` writes
    credentials to paths that may not exist in all environments.
    """
    console.print("  [dim]Detecting Claude subscription token...[/dim]")
    try:
        from datus.auth.claude_credential import get_claude_subscription_token

        token, source = get_claude_subscription_token()
        console.print(f"  ✅ Subscription token detected (from {source})")
        return token
    except Exception:
        console.print("  [yellow]⚠️  Could not auto-detect subscription token[/yellow]")
        console.print("  [dim]Run 'claude setup-token' to get your subscription token[/dim]")
        token = prompt_input(
            console,
            "Paste your subscription token (sk-ant-oat01-...)",
            is_password=True,
        )
        token = token.strip() if token else ""
        if not token:
            console.print("❌ Token cannot be empty")
            return None
        return token


def configure_codex_oauth(
    console: Console,
    provider: str,
    provider_config: Dict[str, Any],
    *,
    model_name: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Drive the Codex browser OAuth handshake and pick a model.

    When ``model_name`` is provided the interactive model picker is
    skipped — used by ``/model``, which drives model selection elsewhere
    so the user is never asked twice.
    """
    if not model_name:
        model_name = _select_model(console, provider_config)
        if not model_name:
            return None

    console.print("→ Opening browser for OAuth authentication...")
    try:
        from datus.auth.oauth_manager import OAuthManager

        OAuthManager().login_browser()
    except Exception as e:
        logger.error(f"OAuth authentication failed for provider `{provider}`: {e}")
        console.print(f"❌ OAuth authentication failed: {e}")
        return None

    console.print("✅ OAuth authentication successful\n")
    return {
        "model": model_name,
        "api_key": "",
        "auth_type": "oauth",
        "base_url": str(provider_config.get("base_url", "")),
        "type": str(provider_config.get("type", provider)),
    }
