# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.model_commands.ModelCommands``.

The interactive ``/model`` picker lives in :class:`ModelApp` (a dedicated
prompt_toolkit Application); this suite patches ``ModelApp.run`` to
return scripted :class:`ModelSelection` values so the dispatcher logic
can be exercised without a TTY.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.model_app import ModelSelection
from datus.cli.model_commands import ModelCommands
from datus.configuration.agent_config import ProviderConfig

pytestmark = pytest.mark.ci


def _stub_agent_config():
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
        "model_overrides": {},
    }
    cfg.provider_available = MagicMock(return_value=False)
    cfg.providers = {}
    cfg.models = {"my-internal": MagicMock(type="openai", model="internal-gpt")}
    cfg.target = ""
    cfg._target_provider = None
    cfg._target_model = None
    return cfg


class _FakeCli:
    """Minimal stand-in for :class:`DatusCLI` used by :class:`ModelCommands`."""

    def __init__(self, agent_config):
        self.console = Console(file=io.StringIO(), no_color=True)
        self.agent_config = agent_config
        self.tui_app = None


@pytest.fixture
def commands():
    cli = _FakeCli(_stub_agent_config())
    return ModelCommands(cli), cli


# ─────────────────────────────────────────────────────────────────────
# Direct-argument shortcuts (no Application involved)
# ─────────────────────────────────────────────────────────────────────


class TestDirectShortcuts:
    def test_provider_slash_model_switches_directly(self, commands):
        mc, cli = commands
        mc.cmd_model("openai/gpt-4.1")
        cli.agent_config.set_active_provider_model.assert_called_once_with("openai", "gpt-4.1")

    def test_custom_prefix_switches_directly(self, commands):
        mc, cli = commands
        mc.cmd_model("custom:my-internal")
        cli.agent_config.set_active_custom.assert_called_once_with("my-internal")

    def test_switch_does_not_touch_cli_rebuild_hook(self, commands):
        """Regression: runtime switch must not rebuild the agent."""
        mc, cli = commands
        cli._rebuild_llm_after_switch = MagicMock()
        mc.cmd_model("openai/gpt-4.1")
        cli._rebuild_llm_after_switch.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# Interactive path: delegation to ModelApp
# ─────────────────────────────────────────────────────────────────────


class TestInteractiveMenu:
    def test_provider_model_selection_persists_active_target(self, commands):
        mc, cli = commands
        with patch(
            "datus.cli.model_commands.ModelApp",
            return_value=MagicMock(
                run=MagicMock(return_value=ModelSelection(kind="provider_model", provider="openai", model="gpt-4.1"))
            ),
        ):
            mc.cmd_model("")
        cli.agent_config.set_active_provider_model.assert_called_once_with("openai", "gpt-4.1")

    def test_custom_selection_persists_active_custom(self, commands):
        mc, cli = commands
        with patch(
            "datus.cli.model_commands.ModelApp",
            return_value=MagicMock(run=MagicMock(return_value=ModelSelection(kind="custom", name="my-internal"))),
        ):
            mc.cmd_model("")
        cli.agent_config.set_active_custom.assert_called_once_with("my-internal")

    def test_cancelled_selection_is_noop(self, commands):
        mc, cli = commands
        with patch(
            "datus.cli.model_commands.ModelApp",
            return_value=MagicMock(run=MagicMock(return_value=None)),
        ):
            mc.cmd_model("")
        cli.agent_config.set_active_provider_model.assert_not_called()
        cli.agent_config.set_active_custom.assert_not_called()

    def test_provider_only_argument_seeds_picker(self, commands):
        """``/model openai`` passes seed_provider through to the Application."""
        mc, _ = commands
        fake_app = MagicMock(run=MagicMock(return_value=None))
        with patch("datus.cli.model_commands.ModelApp", return_value=fake_app) as ma:
            mc.cmd_model("openai")
        _, kwargs = ma.call_args
        assert kwargs.get("seed_provider") == "openai"


# ─────────────────────────────────────────────────────────────────────
# OAuth handoff loop
# ─────────────────────────────────────────────────────────────────────


class TestOauthHandoff:
    def test_needs_oauth_runs_configure_then_reopens_with_seed(self, commands):
        """OAuth result triggers configure_codex_oauth + a re-entry seeded on the provider."""
        mc, cli = commands
        first_app = MagicMock(run=MagicMock(return_value=ModelSelection(kind="needs_oauth", provider="codex")))
        second_app = MagicMock(
            run=MagicMock(return_value=ModelSelection(kind="provider_model", provider="codex", model="code-1"))
        )
        oauth_result = {"model": "code-1", "api_key": "", "base_url": "https://api.codex", "type": "codex"}
        with (
            patch(
                "datus.cli.model_commands.ModelApp",
                side_effect=[first_app, second_app],
            ) as ma,
            patch(
                "datus.cli.provider_auth_flows.configure_codex_oauth",
                return_value=oauth_result,
            ) as oauth,
        ):
            mc.cmd_model("")

        oauth.assert_called_once()
        # Second entry must be seeded on the freshly configured provider.
        second_call_kwargs = ma.call_args_list[1].kwargs
        assert second_call_kwargs.get("seed_provider") == "codex"
        cli.agent_config.set_provider_config.assert_called_once()
        cli.agent_config.set_active_provider_model.assert_called_once_with("codex", "code-1")

    def test_needs_oauth_aborts_when_helper_fails(self, commands):
        mc, cli = commands
        fake_app = MagicMock(run=MagicMock(return_value=ModelSelection(kind="needs_oauth", provider="codex")))
        with (
            patch("datus.cli.model_commands.ModelApp", return_value=fake_app),
            patch(
                "datus.cli.provider_auth_flows.configure_codex_oauth",
                return_value=None,
            ),
        ):
            mc.cmd_model("")
        cli.agent_config.set_provider_config.assert_not_called()
        cli.agent_config.set_active_provider_model.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# Custom-model persistence
# ─────────────────────────────────────────────────────────────────────


class TestAddCustomModel:
    def test_add_custom_writes_to_config_and_activates(self, commands):
        mc, cli = commands
        fake_app = MagicMock(
            run=MagicMock(
                return_value=ModelSelection(
                    kind="add_custom",
                    name="my-new",
                    payload={"type": "openai", "model": "gpt-4o", "base_url": "", "api_key": "sk"},
                )
            )
        )
        mgr = MagicMock()
        with (
            patch("datus.cli.model_commands.ModelApp", return_value=fake_app),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=mgr,
            ),
        ):
            mc.cmd_model("")
        mgr.update_item.assert_called_once()
        key_arg, value_arg = mgr.update_item.call_args.args[:2]
        assert key_arg == "models"
        assert value_arg == {"my-new": {"type": "openai", "model": "gpt-4o", "base_url": "", "api_key": "sk"}}
        cli.agent_config.set_active_custom.assert_called_once_with("my-new")

    def test_add_custom_failure_prevents_activation(self, commands):
        mc, cli = commands
        fake_app = MagicMock(
            run=MagicMock(
                return_value=ModelSelection(
                    kind="add_custom",
                    name="oops",
                    payload={"type": "openai", "model": "gpt-4o"},
                )
            )
        )
        mgr = MagicMock()
        mgr.update_item.side_effect = RuntimeError("disk full")
        with (
            patch("datus.cli.model_commands.ModelApp", return_value=fake_app),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=mgr,
            ),
        ):
            mc.cmd_model("")
        cli.agent_config.set_active_custom.assert_not_called()


class TestDeleteCustomModel:
    def test_delete_custom_removes_from_memory_and_persists_remaining(self, commands):
        mc, cli = commands
        fake_app = MagicMock(run=MagicMock(return_value=ModelSelection(kind="delete_custom", name="my-internal")))
        mgr = MagicMock()
        mgr.get.return_value = {"my-internal": {"type": "openai"}, "keep": {"type": "openai"}}
        with (
            patch("datus.cli.model_commands.ModelApp", return_value=fake_app),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=mgr,
            ),
        ):
            mc.cmd_model("")
        # In-memory model map must no longer contain the deleted entry.
        assert "my-internal" not in cli.agent_config.models
        # Persistence call must drop the entry via ``delete_old_key`` replace.
        mgr.update_item.assert_called_once()
        args, kwargs = mgr.update_item.call_args
        assert args[0] == "models"
        assert "my-internal" not in args[1]
        assert "keep" in args[1]
        assert kwargs.get("delete_old_key") is True

    def test_delete_custom_clears_active_target_when_it_was_deleted(self, commands):
        mc, cli = commands
        cli.agent_config.target = "my-internal"
        fake_app = MagicMock(run=MagicMock(return_value=ModelSelection(kind="delete_custom", name="my-internal")))
        mgr = MagicMock()
        mgr.get.return_value = {"my-internal": {"type": "openai"}}
        with (
            patch("datus.cli.model_commands.ModelApp", return_value=fake_app),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=mgr,
            ),
        ):
            mc.cmd_model("")
        assert cli.agent_config.target == ""

    def test_delete_unknown_name_is_noop(self, commands):
        mc, cli = commands
        fake_app = MagicMock(run=MagicMock(return_value=ModelSelection(kind="delete_custom", name="ghost")))
        mgr = MagicMock()
        with (
            patch("datus.cli.model_commands.ModelApp", return_value=fake_app),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=mgr,
            ),
        ):
            mc.cmd_model("")
        mgr.update_item.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# Sanity: legacy export still works
# ─────────────────────────────────────────────────────────────────────


class TestExports:
    def test_provider_config_import_still_works(self):
        cfg = ProviderConfig(api_key="x")
        assert cfg.api_key == "x"
