# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.model_app.ModelApp`.

The Application itself is not exercised under a pty — instead each test
constructs a ``ModelApp`` and drives its state machine by calling the
action methods directly (``_on_provider_enter``, ``_submit_cred_form``,
``_apply_seed``, ...). :meth:`Application.exit` is patched so we can
capture what the app would have returned to its caller.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.model_app import ModelApp, ModelSelection, _Tab, _View

pytestmark = pytest.mark.ci


def _stub_agent_config(**overrides):
    cfg = MagicMock()
    cfg.provider_catalog = {
        "providers": {
            "openai": {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "default_model": "gpt-4.1",
                "models": ["gpt-4.1", "gpt-4o"],
            },
            "claude_subscription": {
                "type": "claude",
                "base_url": "https://api.anthropic.com",
                "default_model": "claude-sonnet-4-6",
                "models": ["claude-sonnet-4-6"],
                "auth_type": "subscription",
            },
            "codex": {
                "type": "codex",
                "base_url": "https://api.codex",
                "default_model": "code-1",
                "models": ["code-1"],
                "auth_type": "oauth",
            },
        },
    }
    cfg.providers = {}
    cfg.models = overrides.get("models", {"my-internal": MagicMock(type="openai", model="internal-gpt")})
    cfg.target = overrides.get("target", "")
    cfg._target_provider = overrides.get("target_provider")
    cfg._target_model = overrides.get("target_model")
    cfg.provider_available = MagicMock(side_effect=lambda name: name == "openai")
    return cfg


def _build(**overrides) -> ModelApp:
    cfg = _stub_agent_config(**overrides)
    return ModelApp(cfg, Console(file=io.StringIO(), no_color=True))


# ─────────────────────────────────────────────────────────────────────
# Provider listing
# ─────────────────────────────────────────────────────────────────────


class TestProviderList:
    def test_providers_tab_excludes_plan_entries(self):
        """``Providers`` tab hides subscription / OAuth / coding-plan providers."""
        app = _build()
        visible = app._providers_for_tab(_Tab.PROVIDERS)
        assert "openai" in visible
        assert "claude_subscription" not in visible
        assert "codex" not in visible

    def test_plans_tab_contains_plan_entries(self):
        app = _build()
        plans = app._providers_for_tab(_Tab.PLANS)
        assert set(plans) == {"claude_subscription", "codex"}

    def test_availability_is_cached(self):
        cfg = _stub_agent_config()
        ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        # One call per provider during __init__ — no extra disk hits per render.
        assert cfg.provider_available.call_count == len(cfg.provider_catalog["providers"])

    def test_available_provider_enter_drills_into_model_list(self):
        app = _build()
        visible = app._providers_for_tab(_Tab.PROVIDERS)
        app._list_cursor = visible.index("openai")
        app._on_provider_enter()
        assert app._view == _View.PROVIDER_MODELS
        assert app._active_provider == "openai"
        assert app._provider_models == ["gpt-4.1", "gpt-4o"]

    def test_unconfigured_api_key_provider_enter_opens_cred_form(self):
        cfg = _stub_agent_config()
        # Make openai look unavailable too (no credentials).
        cfg.provider_available = MagicMock(return_value=False)
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        visible = app._providers_for_tab(_Tab.PROVIDERS)
        app._list_cursor = visible.index("openai")
        with patch.object(app._app.layout, "focus"):
            app._on_provider_enter()
        assert app._view == _View.PROVIDER_CRED_FORM

    def test_unconfigured_subscription_prefers_auto_detected_token(self):
        cfg = _stub_agent_config()
        cfg.provider_available = MagicMock(return_value=False)
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        app._enter_provider_list(_Tab.PLANS)
        visible = app._providers_for_tab(_Tab.PLANS)
        app._list_cursor = visible.index("claude_subscription")
        with patch(
            "datus.auth.claude_credential.get_claude_subscription_token",
            return_value=("sk-ant-oat01-abc", "~/.claude/.credentials.json"),
        ):
            app._on_provider_enter()
        assert app._view == _View.PROVIDER_MODELS
        cfg.set_provider_config.assert_called_once()
        persist_kwargs = cfg.set_provider_config.call_args.kwargs
        assert persist_kwargs["auth_type"] == "subscription"
        assert persist_kwargs["api_key"] == "sk-ant-oat01-abc"

    def test_oauth_provider_enter_exits_with_needs_oauth(self):
        cfg = _stub_agent_config()
        cfg.provider_available = MagicMock(return_value=False)
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        app._enter_provider_list(_Tab.PLANS)
        visible = app._providers_for_tab(_Tab.PLANS)
        app._list_cursor = visible.index("codex")
        with patch.object(app._app, "exit") as mock_exit:
            app._on_provider_enter()
        mock_exit.assert_called_once()
        result = mock_exit.call_args.kwargs["result"]
        assert isinstance(result, ModelSelection)
        assert result.kind == "needs_oauth"
        assert result.provider == "codex"


# ─────────────────────────────────────────────────────────────────────
# Model list selection
# ─────────────────────────────────────────────────────────────────────


class TestEditCredentialsShortcut:
    def test_edit_opens_cred_form_even_when_provider_is_configured(self):
        """The ``e`` shortcut must bypass the "available → drill in" short-circuit."""
        app = _build()  # openai is configured via stub's provider_available side_effect
        visible = app._providers_for_tab(_Tab.PROVIDERS)
        app._list_cursor = visible.index("openai")
        with patch.object(app._app.layout, "focus"):
            app._on_edit_credentials()
        assert app._view == _View.PROVIDER_CRED_FORM
        assert app._active_provider == "openai"

    def test_edit_prefills_base_url_from_saved_config(self):
        cfg = _stub_agent_config()
        saved = MagicMock(api_key="sk-prev", base_url="https://saved.example", auth_type="api_key")
        cfg.providers = {"openai": saved}
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        visible = app._providers_for_tab(_Tab.PROVIDERS)
        app._list_cursor = visible.index("openai")
        with patch.object(app._app.layout, "focus"):
            app._on_edit_credentials()
        assert app._cred_base_url.text == "https://saved.example"
        # API key is NEVER prefilled — it's treated as a secret.
        assert app._cred_api_key.text == ""

    def test_edit_on_subscription_opens_token_form_without_auto_detect(self):
        app = _build()
        app._enter_provider_list(_Tab.PLANS)
        visible = app._providers_for_tab(_Tab.PLANS)
        app._list_cursor = visible.index("claude_subscription")
        with (
            patch.object(app._app.layout, "focus"),
            patch("datus.auth.claude_credential.get_claude_subscription_token") as auto_detect,
        ):
            app._on_edit_credentials()
        assert app._view == _View.PROVIDER_TOKEN_FORM
        auto_detect.assert_not_called()

    def test_edit_on_oauth_provider_exits_with_needs_oauth(self):
        app = _build()
        app._enter_provider_list(_Tab.PLANS)
        visible = app._providers_for_tab(_Tab.PLANS)
        app._list_cursor = visible.index("codex")
        with patch.object(app._app, "exit") as mock_exit:
            app._on_edit_credentials()
        result = mock_exit.call_args.kwargs["result"]
        assert result.kind == "needs_oauth"
        assert result.provider == "codex"

    def test_edit_ignored_on_non_provider_views(self):
        """The handler is cursor-driven; no-op when the provider list is empty."""
        app = _build()
        app._enter_provider_list(_Tab.PROVIDERS)
        app._list_cursor = 999  # out of range
        with patch.object(app._app, "exit") as mock_exit:
            app._on_edit_credentials()
        mock_exit.assert_not_called()
        assert app._view == _View.PROVIDER_LIST


class TestModelSelection:
    def test_on_model_enter_exits_with_provider_model(self):
        app = _build()
        app._enter_provider_models("openai")
        app._list_cursor = 0
        with patch.object(app._app, "exit") as mock_exit:
            app._on_model_enter()
        result = mock_exit.call_args.kwargs["result"]
        assert result.kind == "provider_model"
        assert (result.provider, result.model) == ("openai", "gpt-4.1")


# ─────────────────────────────────────────────────────────────────────
# Custom tab
# ─────────────────────────────────────────────────────────────────────


class TestCustomTab:
    def test_enter_custom_list_populates_names_and_appends_add_row(self):
        app = _build()
        app._enter_custom_list()
        items = app._current_items()
        # All stored custom entries + the trailing "+ Add model..." row.
        assert len(items) == len(app._custom_names) + 1
        assert items[-1][0].startswith("+ Add model")

    def test_custom_enter_on_name_exits_with_custom_selection(self):
        app = _build()
        app._enter_custom_list()
        app._list_cursor = app._custom_names.index("my-internal")
        with patch.object(app._app, "exit") as mock_exit:
            app._on_custom_enter()
        result = mock_exit.call_args.kwargs["result"]
        assert result.kind == "custom"
        assert result.name == "my-internal"

    def test_custom_enter_on_add_row_opens_add_model_form(self):
        app = _build()
        app._enter_custom_list()
        app._list_cursor = len(app._custom_names)  # cursor on the "+ Add model..." row
        with patch.object(app._app.layout, "focus"):
            app._on_custom_enter()
        assert app._view == _View.ADD_MODEL_FORM


class TestDeleteCustomShortcut:
    def test_first_press_arms_confirmation_without_exit(self):
        app = _build()
        app._enter_custom_list()
        app._list_cursor = app._custom_names.index("my-internal")
        with patch.object(app._app, "exit") as mock_exit:
            app._on_delete_custom()
        mock_exit.assert_not_called()
        assert app._pending_delete_custom == "my-internal"
        assert app._error_message and "Delete" in app._error_message

    def test_second_press_on_same_row_emits_delete_custom(self):
        app = _build()
        app._enter_custom_list()
        app._list_cursor = app._custom_names.index("my-internal")
        app._on_delete_custom()  # arms
        with patch.object(app._app, "exit") as mock_exit:
            app._on_delete_custom()  # confirms
        result = mock_exit.call_args.kwargs["result"]
        assert result.kind == "delete_custom"
        assert result.name == "my-internal"
        assert app._pending_delete_custom is None

    def test_cursor_move_cancels_pending_confirmation(self):
        """Re-binding guards: moving the cursor clears the arm state."""
        cfg = _stub_agent_config(
            models={"a": MagicMock(type="openai", model="x"), "b": MagicMock(type="openai", model="y")}
        )
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        app._enter_custom_list()
        app._list_cursor = 0
        app._on_delete_custom()  # arms delete of "a"
        assert app._pending_delete_custom == "a"
        # Simulate the down-arrow binding: move cursor and clear pending.
        app._list_cursor = 1
        app._pending_delete_custom = None
        # Now a single press on "b" should only arm, not delete.
        with patch.object(app._app, "exit") as mock_exit:
            app._on_delete_custom()
        mock_exit.assert_not_called()
        assert app._pending_delete_custom == "b"

    def test_delete_ignored_on_add_row(self):
        app = _build()
        app._enter_custom_list()
        app._list_cursor = len(app._custom_names)  # "+ Add model..." row
        with patch.object(app._app, "exit") as mock_exit:
            app._on_delete_custom()
        mock_exit.assert_not_called()
        assert app._pending_delete_custom is None


# ─────────────────────────────────────────────────────────────────────
# Credential form submission
# ─────────────────────────────────────────────────────────────────────


class TestCredentialForm:
    def test_submit_persists_and_advances_to_model_list(self):
        cfg = _stub_agent_config()
        cfg.provider_available = MagicMock(return_value=False)
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        app._active_provider = "openai"
        app._cred_api_key.text = "sk-test"
        app._cred_base_url.text = "https://custom.example"
        # provider_available is re-queried after the save; pretend it now succeeds.
        cfg.provider_available.side_effect = lambda name: name == "openai"
        app._submit_cred_form()
        cfg.set_provider_config.assert_called_once_with(
            provider="openai",
            api_key="sk-test",
            base_url="https://custom.example",
            auth_type="api_key",
        )
        assert app._view == _View.PROVIDER_MODELS


class TestTokenForm:
    def test_submit_with_empty_token_surfaces_error(self):
        app = _build()
        app._active_provider = "claude_subscription"
        app._token_input.text = ""
        app._submit_token_form()
        assert isinstance(app._error_message, str)
        assert "token" in app._error_message.lower()
        assert app._view != _View.PROVIDER_MODELS

    def test_submit_with_valid_token_advances_to_model_list(self):
        app = _build()
        app._active_provider = "claude_subscription"
        app._token_input.text = "sk-ant-oat01-xyz"
        app._submit_token_form()
        assert app._view == _View.PROVIDER_MODELS
        app._cfg.set_provider_config.assert_called_once()


# ─────────────────────────────────────────────────────────────────────
# Add-model form submission
# ─────────────────────────────────────────────────────────────────────


class TestAddModelForm:
    def _prepare(self, **fields):
        app = _build()
        app._enter_custom_list()
        with patch.object(app._app.layout, "focus"):
            app._enter_add_model_form()
        for attr, value in fields.items():
            getattr(app, f"_add_{attr}").text = value
        return app

    def test_submit_exits_with_add_custom_payload(self):
        app = self._prepare(name="my-new", type="openai", model="gpt-4o", base_url="https://x", api_key="sk")
        with patch.object(app._app, "exit") as mock_exit:
            app._submit_add_model_form()
        result = mock_exit.call_args.kwargs["result"]
        assert result.kind == "add_custom"
        assert result.name == "my-new"
        assert result.payload["type"] == "openai"
        assert result.payload["model"] == "gpt-4o"
        assert result.payload["base_url"] == "https://x"
        assert result.payload["api_key"] == "sk"

    def test_submit_rejects_empty_name(self):
        app = self._prepare(name="", type="openai", model="gpt-4o")
        with patch.object(app._app, "exit") as mock_exit:
            app._submit_add_model_form()
        mock_exit.assert_not_called()
        assert app._error_message and "name" in app._error_message.lower()

    def test_submit_rejects_duplicate_name(self):
        app = self._prepare(name="my-internal", type="openai", model="gpt-4o")
        with patch.object(app._app, "exit") as mock_exit:
            app._submit_add_model_form()
        mock_exit.assert_not_called()
        assert app._error_message and "already exists" in app._error_message

    def test_submit_rejects_missing_required_fields(self):
        app = self._prepare(name="x", type="", model="")
        with patch.object(app._app, "exit") as mock_exit:
            app._submit_add_model_form()
        mock_exit.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# Tab switching + seeding
# ─────────────────────────────────────────────────────────────────────


class TestTabCycle:
    def test_forward_cycle_visits_providers_plans_custom(self):
        app = _build()
        assert app._tab == _Tab.PROVIDERS
        app._cycle_tab(+1)
        assert app._tab == _Tab.PLANS
        app._cycle_tab(+1)
        assert app._tab == _Tab.CUSTOM
        app._cycle_tab(+1)
        assert app._tab == _Tab.PROVIDERS

    def test_backward_cycle_goes_providers_custom_plans(self):
        app = _build()
        app._cycle_tab(-1)
        assert app._tab == _Tab.CUSTOM
        app._cycle_tab(-1)
        assert app._tab == _Tab.PLANS
        app._cycle_tab(-1)
        assert app._tab == _Tab.PROVIDERS

    def test_cycle_into_plans_shows_only_plan_providers(self):
        app = _build()
        app._cycle_tab(+1)  # → PLANS
        items = app._current_items()
        labels = [label for label, _ in items]
        # All plan entries must be labelled; no api_key providers here.
        joined = " ".join(labels)
        assert "openai" not in joined
        assert "claude" in joined
        assert "codex" in joined


class TestApplySeed:
    def test_seed_available_provider_enters_model_list(self):
        cfg = _stub_agent_config()
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True), seed_provider="openai")
        early = app._apply_seed()
        assert early is None
        assert app._view == _View.PROVIDER_MODELS
        assert app._active_provider == "openai"

    def test_seed_unavailable_api_key_provider_opens_cred_form(self):
        cfg = _stub_agent_config()
        cfg.provider_available = MagicMock(return_value=False)
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True), seed_provider="openai")
        with patch.object(app._app.layout, "focus"):
            early = app._apply_seed()
        assert early is None
        assert app._view == _View.PROVIDER_CRED_FORM

    def test_seed_oauth_provider_returns_needs_oauth_without_starting_app(self):
        cfg = _stub_agent_config()
        cfg.provider_available = MagicMock(return_value=False)
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True), seed_provider="codex")
        early = app._apply_seed()
        assert isinstance(early, ModelSelection)
        assert early.kind == "needs_oauth"
        assert early.provider == "codex"

    def test_seed_subscription_uses_auto_token_when_available(self):
        cfg = _stub_agent_config()
        cfg.provider_available = MagicMock(return_value=False)
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True), seed_provider="claude_subscription")
        with patch(
            "datus.auth.claude_credential.get_claude_subscription_token",
            return_value=("sk-ant-oat01-abc", "env"),
        ):
            early = app._apply_seed()
        assert early is None
        assert app._view == _View.PROVIDER_MODELS
        cfg.set_provider_config.assert_called_once()

    def test_seed_unknown_provider_is_ignored(self):
        app = _build()
        app._seed_provider = "does-not-exist"
        assert app._apply_seed() is None
        # Default view unchanged.
        assert app._view == _View.PROVIDER_LIST

    def test_seed_tab_custom_enters_custom_list(self):
        cfg = _stub_agent_config()
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True), seed_tab="custom")
        assert app._apply_seed() is None
        assert app._tab == _Tab.CUSTOM
        assert app._view == _View.CUSTOM_LIST


# ─────────────────────────────────────────────────────────────────────
# Cursor / scrolling safety
# ─────────────────────────────────────────────────────────────────────


class TestCursor:
    def test_clamp_cursor_handles_empty_list(self):
        app = _build()
        app._list_cursor = 5
        app._clamp_cursor(0)
        assert app._list_cursor == 0

    def test_clamp_cursor_reigns_in_overshoot(self):
        app = _build()
        app._list_cursor = 99
        app._clamp_cursor(3)
        assert app._list_cursor == 2

    def test_visible_slice_scrolls_to_cursor(self):
        app = _build()
        total = 25
        app._list_cursor = 20
        start, end = app._visible_slice(total)
        assert start <= app._list_cursor < end
        assert end - start <= 15

    def test_visible_slice_returns_full_range_when_total_fits(self):
        app = _build()
        start, end = app._visible_slice(10)
        assert (start, end) == (0, 10)


# ─────────────────────────────────────────────────────────────────────
# Rendering helpers — smoke tests
# ─────────────────────────────────────────────────────────────────────


class TestRendering:
    def test_tab_strip_contains_all_three_labels(self):
        app = _build()
        fragments = app._render_tab_strip()
        flat = "".join(text for _, text in fragments)
        assert "Providers" in flat
        assert "Plans" in flat
        assert "Custom" in flat

    def test_provider_items_mark_current_provider(self):
        cfg = _stub_agent_config(target_provider="openai", target_model="gpt-4.1")
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        labels = [label for label, _ in app._provider_items()]
        openai_label = next(label for label in labels if label.startswith("openai"))
        assert "\u2190 current" in openai_label

    def test_provider_models_mark_current_model(self):
        cfg = _stub_agent_config(target_provider="openai", target_model="gpt-4o")
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        app._enter_provider_models("openai")
        labels = [label for label, _ in app._provider_models_items()]
        assert any("\u2190 current" in label for label in labels)

    def test_configured_marker_uses_checkmark(self):
        cfg = _stub_agent_config()
        # Only openai is available (provider_available stub returns True for it).
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        labels = [label for label, _ in app._provider_items()]
        openai_label = next(label for label in labels if label.startswith("openai"))
        assert "\u2713" in openai_label
        assert "[configured]" not in openai_label

    def test_plans_tab_labels_have_no_parenthesised_tags(self):
        app = _build()
        app._enter_provider_list(_Tab.PLANS)
        labels = [label for label, _ in app._provider_items()]
        assert labels, "Plans tab should not be empty"
        for label in labels:
            assert "(" not in label, f"Plans tab label unexpectedly tagged: {label!r}"
            assert ")" not in label, f"Plans tab label unexpectedly tagged: {label!r}"

    def test_plans_tab_renames_claude_subscription_to_claude_code(self):
        app = _build()
        app._enter_provider_list(_Tab.PLANS)
        labels = [label for label, _ in app._provider_items()]
        joined = " | ".join(labels)
        assert "claude code" in joined
        assert "claude_subscription" not in joined

    def test_plans_tab_replaces_underscores_in_names_with_spaces(self):
        cfg = _stub_agent_config()
        cfg.provider_catalog["providers"]["alibaba_coding"] = {
            "type": "claude",
            "base_url": "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
            "default_model": "qwen3-coder-plus",
            "models": ["qwen3-coder-plus"],
        }
        app = ModelApp(cfg, Console(file=io.StringIO(), no_color=True))
        app._enter_provider_list(_Tab.PLANS)
        labels = [label for label, _ in app._provider_items()]
        joined = " | ".join(labels)
        assert "alibaba coding" in joined
        assert "alibaba_coding" not in joined
