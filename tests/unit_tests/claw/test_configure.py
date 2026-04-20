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
    assert _redact("app_secret", "${FEISHU_APP_SECRET}") == "${FEISHU_APP_SECRET}"
    assert _redact("app_id", "cli_12345") == "cli_12345"
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
    assert "existing-feishu" in channels
    assert channels["existing-feishu"]["extra"]["app_secret"] == "SECRETVALUEABCD"
    slack = channels["slack-prod"]
    assert slack["adapter"] == "slack"
    assert slack["enabled"] is True
    assert slack["verbose"] == "brief"
    assert slack["extra"] == {"app_token": "xapp-TOKEN-1", "bot_token": "xoxb-TOKEN-2"}


def test_add_rejects_duplicate_then_accepts(configurator, agent_yml):
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
def test_delete_removes_named_channel(configurator, agent_yml):
    configurator.cm.update_item(
        "channels",
        {"slack-old": {"adapter": "slack", "enabled": False, "extra": {"bot_token": "xoxb-old"}}},
        delete_old_key=False,
    )

    with patch("datus.claw.configure.Confirm.ask", return_value=True):
        rc = configurator.delete("slack-old")

    assert rc == 0
    data = _reload(agent_yml)
    assert "slack-old" not in data["channels"]
    assert "existing-feishu" in data["channels"]


def test_delete_aborts_when_user_declines(configurator, agent_yml):
    with patch("datus.claw.configure.Confirm.ask", return_value=False):
        rc = configurator.delete("existing-feishu")
    assert rc == 0
    data = _reload(agent_yml)
    assert "existing-feishu" in data["channels"]


def test_delete_with_unknown_name_is_noop(configurator, agent_yml, capsys):
    rc = configurator.delete("does-not-exist")
    assert rc == 0
    assert "not found" in capsys.readouterr().out
    assert "existing-feishu" in _reload(agent_yml)["channels"]


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
# toggle_enabled
# ---------------------------------------------------------------------------
def test_toggle_enabled_flips_flag(configurator, agent_yml):
    rc = configurator._toggle_enabled("existing-feishu")
    assert rc == 0
    channel = _reload(agent_yml)["channels"]["existing-feishu"]
    assert channel["enabled"] is False
    # Other fields are preserved.
    assert channel["adapter"] == "feishu"
    assert channel["verbose"] == "brief"
    assert channel["extra"]["app_id"] == "cli_existing"
    assert channel["extra"]["app_secret"] == "SECRETVALUEABCD"

    # Toggling again flips back.
    rc = configurator._toggle_enabled("existing-feishu")
    assert rc == 0
    assert _reload(agent_yml)["channels"]["existing-feishu"]["enabled"] is True


def test_toggle_enabled_unknown_channel(configurator, capsys):
    rc = configurator._toggle_enabled("nope")
    assert rc == 1
    assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# change_verbose
# ---------------------------------------------------------------------------
def test_change_verbose_updates_value(configurator, agent_yml):
    with patch("datus.claw.configure.select_choice", return_value="detail"):
        rc = configurator._change_verbose("existing-feishu")
    assert rc == 0
    assert _reload(agent_yml)["channels"]["existing-feishu"]["verbose"] == "detail"


def test_change_verbose_noop_when_unchanged(configurator, agent_yml, capsys):
    with patch("datus.claw.configure.select_choice", return_value="brief"):
        rc = configurator._change_verbose("existing-feishu")
    assert rc == 0
    assert "unchanged" in capsys.readouterr().out
    assert _reload(agent_yml)["channels"]["existing-feishu"]["verbose"] == "brief"


def test_change_verbose_unknown_channel(configurator, capsys):
    rc = configurator._change_verbose("nope")
    assert rc == 1
    assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# reinstall_deps
# ---------------------------------------------------------------------------
def test_reinstall_deps_invokes_pip(configurator):
    fake_completed = MagicMock(returncode=0)
    with (
        patch("datus.claw.configure.Confirm.ask", return_value=True),
        patch("datus.claw.configure.subprocess.run", return_value=fake_completed) as mock_run,
    ):
        rc = configurator._reinstall_deps("feishu")
    assert rc == 0
    cmd = mock_run.call_args.args[0]
    assert "lark-oapi" in cmd


def test_reinstall_deps_reports_failure(configurator):
    fake_completed = MagicMock(returncode=2)
    with (
        patch("datus.claw.configure.Confirm.ask", return_value=True),
        patch("datus.claw.configure.subprocess.run", return_value=fake_completed),
    ):
        rc = configurator._reinstall_deps("feishu")
    assert rc == 2


def test_reinstall_deps_cancelled(configurator, capsys):
    with patch("datus.claw.configure.Confirm.ask", return_value=False):
        rc = configurator._reinstall_deps("feishu")
    assert rc == 0
    assert "Cancelled" in capsys.readouterr().out


def test_reinstall_deps_unknown_adapter(configurator, capsys):
    rc = configurator._reinstall_deps("unknown-adapter")
    assert rc == 0
    assert "No pip deps registered" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Hub loop (run)
# ---------------------------------------------------------------------------
def test_run_quits_immediately(configurator):
    with patch.object(configurator, "_render_hub", return_value=("quit", None)):
        assert configurator.run() == 0


def test_run_dispatches_add_then_quits(configurator):
    with (
        patch.object(configurator, "_render_hub", side_effect=[("add", None), ("quit", None)]),
        patch.object(configurator, "add", return_value=0) as mock_add,
    ):
        assert configurator.run() == 0
    mock_add.assert_called_once_with()


def test_run_dispatches_channel_submenu_then_quits(configurator):
    with (
        patch.object(
            configurator,
            "_render_hub",
            side_effect=[("channel", "existing-feishu"), ("quit", None)],
        ),
        patch.object(configurator, "_channel_submenu") as mock_submenu,
    ):
        assert configurator.run() == 0
    mock_submenu.assert_called_once_with("existing-feishu")


def test_run_returns_one_when_config_missing(tmp_path):
    import datus.configuration.agent_config_loader as loader

    loader.CONFIGURATION_MANAGER = None
    from datus.claw.configure import ChannelConfigurator

    c = ChannelConfigurator(str(tmp_path / "does_not_exist.yml"))
    assert c.cm is None
    assert c.run() == 1


# ---------------------------------------------------------------------------
# Channel submenu dispatch
# ---------------------------------------------------------------------------
def test_channel_submenu_routes_toggle(configurator):
    with (
        patch("datus.claw.configure._select_menu", return_value="toggle_enabled"),
        patch.object(configurator, "_toggle_enabled") as mock_toggle,
    ):
        configurator._channel_submenu("existing-feishu")
    mock_toggle.assert_called_once_with("existing-feishu")


def test_channel_submenu_routes_change_verbose(configurator):
    with (
        patch("datus.claw.configure._select_menu", return_value="change_verbose"),
        patch.object(configurator, "_change_verbose") as mock_change,
    ):
        configurator._channel_submenu("existing-feishu")
    mock_change.assert_called_once_with("existing-feishu")


def test_channel_submenu_routes_reinstall(configurator):
    with (
        patch("datus.claw.configure._select_menu", return_value="reinstall_deps"),
        patch.object(configurator, "_reinstall_deps") as mock_reinstall,
    ):
        configurator._channel_submenu("existing-feishu")
    mock_reinstall.assert_called_once_with("feishu")


def test_channel_submenu_routes_delete(configurator):
    with (
        patch("datus.claw.configure._select_menu", return_value="delete"),
        patch.object(configurator, "delete") as mock_delete,
    ):
        configurator._channel_submenu("existing-feishu")
    mock_delete.assert_called_once_with("existing-feishu")


def test_channel_submenu_back_is_noop(configurator):
    with (
        patch("datus.claw.configure._select_menu", return_value="back"),
        patch.object(configurator, "delete") as mock_delete,
        patch.object(configurator, "_toggle_enabled") as mock_toggle,
    ):
        configurator._channel_submenu("existing-feishu")
    mock_delete.assert_not_called()
    mock_toggle.assert_not_called()


def test_channel_submenu_cancel_is_noop(configurator):
    with (
        patch("datus.claw.configure._select_menu", return_value=None),
        patch.object(configurator, "delete") as mock_delete,
    ):
        configurator._channel_submenu("existing-feishu")
    mock_delete.assert_not_called()


def test_channel_submenu_unknown_channel(configurator, capsys):
    configurator._channel_submenu("does-not-exist")
    assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _render_hub
# ---------------------------------------------------------------------------
def test_render_hub_returns_add(configurator):
    with patch("datus.claw.configure._select_menu", return_value="__add__"):
        assert configurator._render_hub() == ("add", None)


def test_render_hub_returns_channel(configurator):
    with patch("datus.claw.configure._select_menu", return_value="channel:existing-feishu"):
        assert configurator._render_hub() == ("channel", "existing-feishu")


def test_render_hub_quit_on_cancel(configurator):
    with patch("datus.claw.configure._select_menu", return_value=None):
        assert configurator._render_hub() == ("quit", None)


def test_render_hub_empty_channels(tmp_path):
    import datus.configuration.agent_config_loader as loader

    loader.CONFIGURATION_MANAGER = None
    path = tmp_path / "agent.yml"
    path.write_text(yaml.safe_dump({"agent": {}}, sort_keys=False))
    from datus.claw.configure import ChannelConfigurator

    c = ChannelConfigurator(str(path))
    captured_rows: list = []

    def _capture(rows, **kwargs):
        captured_rows.extend(rows)
        return None

    with patch("datus.claw.configure._select_menu", side_effect=_capture):
        assert c._render_hub() == ("quit", None)

    keys = [r[0] for r in captured_rows]
    assert "__add__" in keys
    assert "__empty__" in keys
    # No channel rows rendered.
    assert not any(k.startswith("channel:") for k in keys)


def test_render_hub_builds_channel_rows(configurator):
    configurator.cm.update_item(
        "channels",
        {"slack-prod": {"adapter": "slack", "enabled": False, "verbose": "brief", "extra": {}}},
        delete_old_key=False,
    )
    captured_rows: list = []

    def _capture(rows, **kwargs):
        captured_rows.extend(rows)
        return None

    with patch("datus.claw.configure._select_menu", side_effect=_capture):
        configurator._render_hub()

    keys = [r[0] for r in captured_rows]
    assert "channel:existing-feishu" in keys
    assert "channel:slack-prod" in keys
    # Enabled/disabled badges rendered in the display column.
    displays = {r[0]: r[1] for r in captured_rows}
    assert "enabled" in displays["channel:existing-feishu"]
    assert "disabled" in displays["channel:slack-prod"]


# ---------------------------------------------------------------------------
# CLI dispatch in datus.claw.main
# ---------------------------------------------------------------------------
def test_main_cli_dispatches_configure(monkeypatch, agent_yml):
    from datus.claw import main as claw_main

    fake_instance = MagicMock()
    fake_instance.run.return_value = 0
    monkeypatch.setattr("sys.argv", ["datus-claw", "configure", "--config", str(agent_yml)])

    with (
        patch("datus.claw.configure.ChannelConfigurator", return_value=fake_instance) as mock_cls,
        pytest.raises(SystemExit) as excinfo,
    ):
        claw_main.main()

    assert excinfo.value.code == 0
    mock_cls.assert_called_once_with(str(agent_yml))
    fake_instance.run.assert_called_once_with()


def test_main_cli_parser_allows_bare_configure(monkeypatch):
    """Daemon invocations still parse without a subcommand."""
    from datus.claw.main import _build_parser

    parser = _build_parser()
    ns = parser.parse_args(["--action", "stop"])
    assert ns.subcommand is None
    assert ns.action == "stop"

    ns = parser.parse_args(["configure"])
    assert ns.subcommand == "configure"
    # Legacy action positional is gone.
    assert not hasattr(ns, "configure_action")


def test_main_cli_parser_rejects_legacy_action():
    """Old `datus-claw configure list` syntax must now fail."""
    from datus.claw.main import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["configure", "list"])
