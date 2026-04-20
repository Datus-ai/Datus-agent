# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus.claw.configure.ChannelConfigurator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


@pytest.fixture
def agent_yml(tmp_path: Path) -> Path:
    cfg = {
        "agent": {
            "channels": {
                "existing-feishu": {
                    "adapter": "feishu",
                    "enabled": True,
                    "verbose": "brief",
                    "stream_response": True,
                    "extra": {
                        "app_id": "cli_existing",
                        "app_secret": "SECRETVALUEABCD",
                    },
                }
            }
        }
    }
    path = tmp_path / "agent.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def configurator(agent_yml: Path):
    # Force a fresh ConfigurationManager bound to our tmp file.
    import datus.configuration.agent_config_loader as loader

    loader.CONFIGURATION_MANAGER = None
    from datus.claw.configure import ChannelConfigurator

    return ChannelConfigurator(str(agent_yml))


def _reload(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["agent"]


# ---------------------------------------------------------------------------
# Redaction helper
# ---------------------------------------------------------------------------
def test_redact_masks_secret_keys():
    from datus.claw.configure import _redact

    assert _redact("app_secret", "SECRETVALUE1234") == "***1234"
    assert _redact("bot_token", "xoxb-1234-abcd") == "***abcd"
    # env placeholders pass through
    assert _redact("app_secret", "${FEISHU_APP_SECRET}") == "${FEISHU_APP_SECRET}"
    # non-secret keys pass through
    assert _redact("app_id", "cli_12345") == "cli_12345"
    # short secrets mask entirely
    assert _redact("password", "abc") == "***"


# ---------------------------------------------------------------------------
# Channel name validation
# ---------------------------------------------------------------------------
def test_validate_channel_name_rules():
    from datus.claw.configure import _validate_channel_name

    existing = {"feishu-main": {}}
    assert _validate_channel_name("slack-prod", existing) == (True, "")
    ok, err = _validate_channel_name("", existing)
    assert not ok and "empty" in err
    ok, err = _validate_channel_name("has space", existing)
    assert not ok and "whitespace" in err
    ok, err = _validate_channel_name("has/slash", existing)
    assert not ok
    ok, err = _validate_channel_name("feishu-main", existing)
    assert not ok and "exists" in err


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
def test_list_prints_redacted_secret(configurator, capsys):
    rc = configurator.list()
    out = capsys.readouterr().out
    assert rc == 0
    assert "existing-feishu" in out
    assert "SECRETVALUEABCD" not in out
    assert "***ABCD" in out
    assert "cli_existing" in out  # non-secret is visible


def test_list_empty(tmp_path, monkeypatch, capsys):
    import datus.configuration.agent_config_loader as loader

    loader.CONFIGURATION_MANAGER = None
    path = tmp_path / "agent.yml"
    path.write_text(yaml.safe_dump({"agent": {"models": []}}, sort_keys=False))
    from datus.claw.configure import ChannelConfigurator

    c = ChannelConfigurator(str(path))
    assert c.list() == 0
    assert "No channels configured" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------
def test_add_slack_channel_preserves_existing(configurator, agent_yml):
    prompt_responses = iter(["slack-prod"])
    confirm_responses = iter([True, False, False])  # enabled, override verbose?, install deps?
    getpass_responses = iter(["xapp-TOKEN-1", "xoxb-TOKEN-2"])

    with (
        patch("datus.claw.configure.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        patch("datus.claw.configure.Confirm.ask", side_effect=lambda *a, **kw: next(confirm_responses)),
        patch("datus.claw.configure.getpass", side_effect=lambda prompt="": next(getpass_responses)),
        patch("datus.claw.configure.select_choice", return_value="slack"),
    ):
        rc = configurator.add()

    assert rc == 0
    data = _reload(agent_yml)
    channels = data["channels"]
    # Existing channel untouched
    assert "existing-feishu" in channels
    assert channels["existing-feishu"]["extra"]["app_secret"] == "SECRETVALUEABCD"
    # New channel added
    slack = channels["slack-prod"]
    assert slack["adapter"] == "slack"
    assert slack["enabled"] is True
    assert slack["verbose"] == "brief"
    assert slack["extra"] == {"app_token": "xapp-TOKEN-1", "bot_token": "xoxb-TOKEN-2"}


def test_add_rejects_duplicate_then_accepts(configurator, agent_yml):
    # First Prompt.ask returns a duplicate name, second returns a valid name,
    # third returns the app_id. Confirms: enabled, override verbose?, install deps?
    prompts = iter(["existing-feishu", "feishu-alt", "cli_new"])
    confirms = iter([True, False, False])
    passwords = iter(["sekret"])

    with (
        patch("datus.claw.configure.Prompt.ask", side_effect=lambda *a, **kw: next(prompts)),
        patch("datus.claw.configure.Confirm.ask", side_effect=lambda *a, **kw: next(confirms)),
        patch("datus.claw.configure.getpass", side_effect=lambda prompt="": next(passwords)),
        patch("datus.claw.configure.select_choice", return_value="feishu"),
    ):
        rc = configurator.add()

    assert rc == 0
    data = _reload(agent_yml)
    assert "feishu-alt" in data["channels"]
    assert data["channels"]["feishu-alt"]["extra"]["app_id"] == "cli_new"
    assert data["channels"]["feishu-alt"]["extra"]["app_secret"] == "sekret"


def test_add_triggers_pip_install_when_confirmed(configurator):
    prompts = iter(["feishu-alt", "cli_new"])
    confirms = iter([True, False, True])  # enabled, override?, install deps
    passwords = iter(["sekret"])

    fake_completed = MagicMock(returncode=0)
    with (
        patch("datus.claw.configure.Prompt.ask", side_effect=lambda *a, **kw: next(prompts)),
        patch("datus.claw.configure.Confirm.ask", side_effect=lambda *a, **kw: next(confirms)),
        patch("datus.claw.configure.getpass", side_effect=lambda prompt="": next(passwords)),
        patch("datus.claw.configure.select_choice", return_value="feishu"),
        patch("datus.claw.configure.subprocess.run", return_value=fake_completed) as mock_run,
    ):
        rc = configurator.add()

    assert rc == 0
    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    assert cmd[1:4] == ["-m", "pip", "install"]
    assert "lark-oapi" in cmd


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------
def test_delete_removes_selected_channel(configurator, agent_yml):
    # Seed a second channel so deletion leaves one behind.
    configurator.cm.update_item(
        "channels",
        {"slack-old": {"adapter": "slack", "enabled": False, "extra": {"bot_token": "xoxb-old"}}},
        delete_old_key=False,
    )

    with (
        patch("datus.claw.configure.select_choice", return_value="slack-old"),
        patch("datus.claw.configure.Confirm.ask", return_value=True),
    ):
        rc = configurator.delete()

    assert rc == 0
    data = _reload(agent_yml)
    assert "slack-old" not in data["channels"]
    assert "existing-feishu" in data["channels"]


def test_delete_aborts_when_user_declines(configurator, agent_yml):
    with (
        patch("datus.claw.configure.select_choice", return_value="existing-feishu"),
        patch("datus.claw.configure.Confirm.ask", return_value=False),
    ):
        rc = configurator.delete()
    assert rc == 0
    data = _reload(agent_yml)
    assert "existing-feishu" in data["channels"]


def test_delete_empty_returns_zero(tmp_path, capsys):
    import datus.configuration.agent_config_loader as loader

    loader.CONFIGURATION_MANAGER = None
    path = tmp_path / "agent.yml"
    path.write_text(yaml.safe_dump({"agent": {}}, sort_keys=False))
    from datus.claw.configure import ChannelConfigurator

    c = ChannelConfigurator(str(path))
    assert c.delete() == 0
    assert "nothing to delete" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# install_deps
# ---------------------------------------------------------------------------
def test_install_deps_interactive_invokes_pip(configurator):
    fake_completed = MagicMock(returncode=0)
    with (
        patch("datus.claw.configure.select_choice", return_value="feishu"),
        patch("datus.claw.configure.subprocess.run", return_value=fake_completed) as mock_run,
    ):
        rc = configurator.install_deps_interactive()
    assert rc == 0
    cmd = mock_run.call_args.args[0]
    assert "lark-oapi" in cmd


def test_install_deps_reports_failure(configurator):
    fake_completed = MagicMock(returncode=2)
    with (
        patch("datus.claw.configure.select_choice", return_value="feishu"),
        patch("datus.claw.configure.subprocess.run", return_value=fake_completed),
    ):
        rc = configurator.install_deps_interactive()
    assert rc == 2


def test_install_deps_no_channels(tmp_path, capsys):
    import datus.configuration.agent_config_loader as loader

    loader.CONFIGURATION_MANAGER = None
    path = tmp_path / "agent.yml"
    path.write_text(yaml.safe_dump({"agent": {}}, sort_keys=False))
    from datus.claw.configure import ChannelConfigurator

    c = ChannelConfigurator(str(path))
    rc = c.install_deps_interactive()
    assert rc == 0
    assert "No channels" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run() dispatcher
# ---------------------------------------------------------------------------
def test_run_dispatches_list(configurator):
    with patch.object(configurator, "list", return_value=0) as mock_list:
        assert configurator.run("list") == 0
    mock_list.assert_called_once()


def test_run_prompts_when_action_missing(configurator):
    with (
        patch.object(configurator, "_prompt_action", return_value="list"),
        patch.object(configurator, "list", return_value=0),
    ):
        assert configurator.run(None) == 0


def test_run_unknown_action(configurator, capsys):
    rc = configurator.run("unknown")
    assert rc == 1
    assert "Unknown action" in capsys.readouterr().out


def test_run_returns_one_when_config_missing(tmp_path, capsys):
    import datus.configuration.agent_config_loader as loader

    loader.CONFIGURATION_MANAGER = None
    from datus.claw.configure import ChannelConfigurator

    c = ChannelConfigurator(str(tmp_path / "does_not_exist.yml"))
    assert c.cm is None
    assert c.run("list") == 1


# ---------------------------------------------------------------------------
# CLI dispatch in datus.claw.main
# ---------------------------------------------------------------------------
def test_main_cli_dispatches_configure(monkeypatch, agent_yml):
    from datus.claw import main as claw_main

    fake_instance = MagicMock()
    fake_instance.run.return_value = 0
    monkeypatch.setattr("sys.argv", ["datus-claw", "configure", "list", "--config", str(agent_yml)])

    with (
        patch("datus.claw.configure.ChannelConfigurator", return_value=fake_instance) as mock_cls,
        pytest.raises(SystemExit) as excinfo,
    ):
        claw_main.main()

    assert excinfo.value.code == 0
    mock_cls.assert_called_once_with(str(agent_yml))
    fake_instance.run.assert_called_once_with("list")


def test_main_cli_parser_allows_bare_configure(monkeypatch):
    """Backwards-compat: daemon invocations must still parse without a subcommand."""
    from datus.claw.main import _build_parser

    parser = _build_parser()
    ns = parser.parse_args(["--action", "stop"])
    assert ns.subcommand is None
    assert ns.action == "stop"

    ns = parser.parse_args(["configure"])
    assert ns.subcommand == "configure"
    assert ns.configure_action is None
