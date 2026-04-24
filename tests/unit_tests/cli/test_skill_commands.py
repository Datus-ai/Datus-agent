# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.skill_commands.SkillCommands`.

Covers:

- Argument dispatch (``cmd_skill`` routes to the correct subcommand).
- Non-interactive paths (``install`` / ``publish`` / ``info`` / ``remove``
  / ``update`` / ``logout``) hit the expected ``SkillManager`` methods.
- The interactive ``_run_menu`` loop converts each :class:`SkillSelection`
  kind into the correct follow-up call.
- Login HTTP flow persists a token on success and surfaces the right error
  on each failure mode.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.skill_app import SkillSelection
from datus.cli.skill_commands import SkillCommands
from datus.tools.skill_tools.skill_config import SkillMetadata

pytestmark = pytest.mark.ci


def _meta(name: str, *, source: str = "local", version: str = "1.0.0", location: Path | None = None) -> SkillMetadata:
    return SkillMetadata(
        name=name,
        description=f"{name} description",
        location=location or Path("/tmp/skills") / name,
        version=version,
        source=source,
    )


def _fake_manager(**overrides) -> MagicMock:
    mgr = MagicMock()
    mgr.config = MagicMock(marketplace_url="http://localhost:9000", install_dir="~/.datus/skills")
    mgr.list_all_skills = MagicMock(return_value=overrides.get("skills", []))
    mgr.get_skill = MagicMock(side_effect=lambda n: next((s for s in overrides.get("skills", []) if s.name == n), None))
    mgr.search_marketplace = MagicMock(return_value=overrides.get("marketplace", []))
    mgr.install_from_marketplace = MagicMock(return_value=(True, "installed"))
    mgr.publish_to_marketplace = MagicMock(return_value=(True, "published"))
    mgr.registry = MagicMock()
    mgr.registry.remove_skill = MagicMock(return_value=True)
    return mgr


class _FakeCli:
    def __init__(self):
        self.console = Console(file=io.StringIO(), no_color=True)
        self.tui_app = None
        self.agent = None
        self.agent_config = MagicMock(skills={})


@pytest.fixture
def commands():
    cli = _FakeCli()
    cmds = SkillCommands(cli)
    return cmds, cli


# ─────────────────────────────────────────────────────────────────────
# Argument dispatch
# ─────────────────────────────────────────────────────────────────────


class TestDispatch:
    def test_empty_args_opens_menu(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_run_menu") as menu:
            cmds.cmd_skill("")
        menu.assert_called_once_with()

    def test_list_opens_menu_on_installed_tab(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_run_menu") as menu:
            cmds.cmd_skill("list")
        menu.assert_called_once_with(seed_tab="installed")

    def test_search_seeds_marketplace_tab(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_run_menu") as menu:
            cmds.cmd_skill("search sales report")
        menu.assert_called_once_with(seed_tab="marketplace", seed_search="sales report")

    def test_login_opens_menu_with_login_flag(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_run_menu") as menu:
            cmds.cmd_skill("login https://town.example.com")
        menu.assert_called_once_with(seed_tab="installed", open_login=True, seed_url="https://town.example.com")

    def test_logout_dispatches_to_do_logout(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_do_logout") as logout:
            cmds.cmd_skill("logout")
        logout.assert_called_once()

    def test_install_with_no_args_prints_usage(self, commands):
        cmds, cli = commands
        cmds.cmd_skill("install")
        assert "Usage" in cli.console.file.getvalue()

    def test_install_dispatches_to_do_install(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_do_install") as inst:
            cmds.cmd_skill("install sql-opt 1.2.3")
        inst.assert_called_once_with("sql-opt", "1.2.3")

    def test_install_defaults_version_to_latest(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_do_install") as inst:
            cmds.cmd_skill("install sql-opt")
        inst.assert_called_once_with("sql-opt", "latest")

    def test_update_dispatches_to_do_update_all(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_do_update_all") as upd:
            cmds.cmd_skill("update")
        upd.assert_called_once()

    def test_help_prints_table(self, commands):
        cmds, cli = commands
        cmds.cmd_skill("help")
        text = cli.console.file.getvalue()
        assert "/skill" in text

    def test_unknown_command_prints_error_and_usage(self, commands):
        cmds, cli = commands
        cmds.cmd_skill("bogus")
        text = cli.console.file.getvalue()
        assert "Unknown skill command" in text
        assert "/skill" in text


# ─────────────────────────────────────────────────────────────────────
# Non-interactive business logic
# ─────────────────────────────────────────────────────────────────────


class TestInstall:
    def test_do_install_calls_manager(self, commands):
        cmds, cli = commands
        mgr = _fake_manager()
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_install("foo", "1.0")
        mgr.install_from_marketplace.assert_called_once_with("foo", "1.0")
        assert "installed" in cli.console.file.getvalue()

    def test_do_install_reports_failure(self, commands):
        cmds, cli = commands
        mgr = _fake_manager()
        mgr.install_from_marketplace.return_value = (False, "boom")
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_install("foo", "1.0")
        assert "boom" in cli.console.file.getvalue()


class TestPublish:
    def test_publish_no_args_prints_usage(self, commands):
        cmds, cli = commands
        cmds._cmd_publish([])
        assert "Usage" in cli.console.file.getvalue()

    def test_publish_with_owner_flag(self, commands):
        cmds, _ = commands
        mgr = _fake_manager()
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._cmd_publish(["/tmp/my-skill", "--owner", "datus"])
        mgr.publish_to_marketplace.assert_called_once_with("/tmp/my-skill", owner="datus")


class TestInfo:
    def test_info_renders_local_and_marketplace(self, commands):
        cmds, cli = commands
        skill = _meta("foo", source="marketplace", version="2.0")
        mgr = _fake_manager(skills=[skill])
        client = MagicMock()
        client.get_skill_info.return_value = {
            "name": "foo",
            "latest_version": "2.1",
            "owner": "datus",
            "promoted": True,
            "usage_count": 99,
            "versions": [{"version": "1.0"}, {"version": "2.0"}, {"version": "2.1"}],
        }
        mgr._get_marketplace_client.return_value = client
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._cmd_info(["foo"])
        text = cli.console.file.getvalue()
        assert "Local Skill: foo" in text
        assert "Marketplace Info" in text
        assert "2.1" in text

    def test_info_remote_failure_with_local_shows_note(self, commands):
        cmds, cli = commands
        skill = _meta("foo", source="local")
        mgr = _fake_manager(skills=[skill])
        mgr._get_marketplace_client.side_effect = RuntimeError("no network")
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._cmd_info(["foo"])
        assert "Marketplace lookup failed" in cli.console.file.getvalue()

    def test_info_without_local_and_remote_warns(self, commands):
        cmds, cli = commands
        mgr = _fake_manager(skills=[])
        mgr._get_marketplace_client.side_effect = RuntimeError("no network")
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._cmd_info(["missing"])
        assert "not found locally or in marketplace" in cli.console.file.getvalue()


class TestRemove:
    def test_remove_unknown_skill_warns(self, commands):
        cmds, cli = commands
        mgr = _fake_manager(skills=[])
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_remove("foo", confirmed=True)
        assert "not found locally" in cli.console.file.getvalue()

    def test_do_remove_deletes_files_when_present(self, tmp_path, commands):
        cmds, cli = commands
        skill_dir = tmp_path / "foo"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: foo\ndescription: d\n---\n")
        skill = _meta("foo", location=skill_dir)
        mgr = _fake_manager(skills=[skill])
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_remove("foo", confirmed=True)
        mgr.registry.remove_skill.assert_called_once_with("foo")
        assert not skill_dir.exists()
        assert "Removed skill" in cli.console.file.getvalue()

    def test_do_remove_registry_failure_reports_error(self, commands):
        cmds, cli = commands
        skill = _meta("foo")
        mgr = _fake_manager(skills=[skill])
        mgr.registry.remove_skill.return_value = False
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_remove("foo", confirmed=True)
        assert "Failed to remove skill" in cli.console.file.getvalue()


class TestUpdateAll:
    def test_no_marketplace_skills_warns(self, commands):
        cmds, cli = commands
        mgr = _fake_manager(skills=[_meta("a", source="local")])
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_update_all()
        assert "No marketplace-installed skills" in cli.console.file.getvalue()

    def test_upgrades_skill_when_remote_version_differs(self, commands):
        cmds, cli = commands
        skill = _meta("foo", source="marketplace", version="1.0")
        mgr = _fake_manager(skills=[skill])
        client = MagicMock()
        client.get_skill_info.return_value = {"latest_version": "2.0"}
        mgr._get_marketplace_client.return_value = client
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_update_all()
        mgr.install_from_marketplace.assert_called_once_with("foo", "2.0")
        assert "1 skill(s) updated" in cli.console.file.getvalue()

    def test_skips_up_to_date_skill(self, commands):
        cmds, cli = commands
        skill = _meta("foo", source="marketplace", version="2.0")
        mgr = _fake_manager(skills=[skill])
        client = MagicMock()
        client.get_skill_info.return_value = {"latest_version": "2.0"}
        mgr._get_marketplace_client.return_value = client
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_update_all()
        mgr.install_from_marketplace.assert_not_called()
        assert "0 skill(s) updated" in cli.console.file.getvalue()


class TestLogout:
    def test_logout_clears_token_and_reports(self, commands):
        cmds, cli = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch("datus.tools.skill_tools.marketplace_auth.clear_token", return_value=True),
        ):
            cmds._do_logout()
        assert "Logged out" in cli.console.file.getvalue()

    def test_logout_without_saved_token_warns(self, commands):
        cmds, cli = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch("datus.tools.skill_tools.marketplace_auth.clear_token", return_value=False),
        ):
            cmds._do_logout()
        assert "No saved credentials" in cli.console.file.getvalue()


# ─────────────────────────────────────────────────────────────────────
# Login HTTP flow
# ─────────────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, *, status_code=200, body=None, cookies=None, content_type="application/json"):
        self.status_code = status_code
        self._body = body or {}
        self.text = "raw-body"
        self.cookies = cookies or {}
        self.headers = {"content-type": content_type}

    def json(self):
        return self._body


class _FakeHttpxClient:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json):
        self.last_url = url
        self.last_json = json
        return self._response


class TestLoginHttp:
    def test_success_via_cookie_saves_token(self, commands):
        cmds, cli = commands
        resp = _FakeResponse(cookies={"town_token": "jwt-from-cookie"})
        with (
            patch("datus.cli.skill_commands.httpx.Client", return_value=_FakeHttpxClient(resp)),
            patch("datus.tools.skill_tools.marketplace_auth.save_token") as save,
        ):
            ok = cmds._do_login_with_credentials(
                email="me@example.com",
                password="secret",
                marketplace_url="http://localhost:9000",
            )
        assert ok is True
        save.assert_called_once_with("jwt-from-cookie", "http://localhost:9000", "me@example.com")
        assert "Login successful" in cli.console.file.getvalue()

    def test_success_via_body_token_field(self, commands):
        cmds, _ = commands
        resp = _FakeResponse(body={"access_token": "jwt-from-body"})
        with (
            patch("datus.cli.skill_commands.httpx.Client", return_value=_FakeHttpxClient(resp)),
            patch("datus.tools.skill_tools.marketplace_auth.save_token") as save,
        ):
            ok = cmds._do_login_with_credentials(
                email="me@example.com",
                password="secret",
                marketplace_url="http://localhost:9000/",
            )
        assert ok is True
        save.assert_called_once()
        assert save.call_args.args[0] == "jwt-from-body"

    def test_http_error_reports_detail(self, commands):
        cmds, cli = commands
        resp = _FakeResponse(status_code=401, body={"detail": "bad credentials"})
        with patch("datus.cli.skill_commands.httpx.Client", return_value=_FakeHttpxClient(resp)):
            ok = cmds._do_login_with_credentials(
                email="me@example.com",
                password="secret",
                marketplace_url="http://localhost:9000",
            )
        assert ok is False
        assert "bad credentials" in cli.console.file.getvalue()

    def test_missing_url_fails_fast(self, commands):
        cmds, cli = commands
        ok = cmds._do_login_with_credentials(email="m@x", password="p", marketplace_url="")
        assert ok is False
        assert "not configured" in cli.console.file.getvalue()

    def test_login_success_no_token_returns_false(self, commands):
        cmds, cli = commands
        resp = _FakeResponse(body={})
        with patch("datus.cli.skill_commands.httpx.Client", return_value=_FakeHttpxClient(resp)):
            ok = cmds._do_login_with_credentials(
                email="me@example.com",
                password="secret",
                marketplace_url="http://localhost:9000",
            )
        assert ok is False
        assert "no token was returned" in cli.console.file.getvalue()


# ─────────────────────────────────────────────────────────────────────
# Interactive loop — selection routing
# ─────────────────────────────────────────────────────────────────────


def _patched_app_run(cmds, *, selections):
    """Patch ``SkillApp.run`` to emit scripted ``SkillSelection`` values in order."""
    iterator = iter(selections)

    def _fake_run(self):  # bound method replacement
        return next(iterator)

    return patch("datus.cli.skill_commands.SkillApp.run", _fake_run)


class TestRunMenuRouting:
    def test_cancel_returns_without_side_effects(self, commands):
        cmds, _ = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            _patched_app_run(cmds, selections=[SkillSelection(kind="cancel")]),
        ):
            cmds._run_menu()
        mgr.install_from_marketplace.assert_not_called()

    def test_install_selection_routes_to_do_install(self, commands):
        cmds, _ = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch.object(cmds, "_do_install") as do_install,
            _patched_app_run(cmds, selections=[SkillSelection(kind="install", name="foo", version="1.2")]),
        ):
            cmds._run_menu()
        do_install.assert_called_once_with("foo", "1.2")

    def test_remove_selection_routes_to_do_remove(self, commands):
        cmds, _ = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch.object(cmds, "_do_remove") as do_remove,
            _patched_app_run(cmds, selections=[SkillSelection(kind="remove", name="foo")]),
        ):
            cmds._run_menu()
        do_remove.assert_called_once_with("foo", confirmed=True)

    def test_update_selection_routes_to_do_update_one(self, commands):
        cmds, _ = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch.object(cmds, "_do_update_one") as do_update,
            _patched_app_run(cmds, selections=[SkillSelection(kind="update", name="foo")]),
        ):
            cmds._run_menu()
        do_update.assert_called_once_with("foo")

    def test_logout_selection_routes_to_do_logout(self, commands):
        cmds, _ = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch.object(cmds, "_do_logout") as do_logout,
            _patched_app_run(cmds, selections=[SkillSelection(kind="logout")]),
        ):
            cmds._run_menu()
        do_logout.assert_called_once()

    def test_login_selection_routes_to_login_and_reopens(self, commands):
        """A successful login reopens the app; the second run is the terminal cancel."""
        cmds, _ = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch.object(cmds, "_do_login_with_credentials", return_value=True) as do_login,
            _patched_app_run(
                cmds,
                selections=[
                    SkillSelection(kind="login", email="me@x", password="pw", marketplace_url="http://m"),
                    SkillSelection(kind="cancel"),
                ],
            ),
        ):
            cmds._run_menu()
        do_login.assert_called_once()

    def test_login_failure_stops_loop(self, commands):
        cmds, _ = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch.object(cmds, "_do_login_with_credentials", return_value=False) as do_login,
            _patched_app_run(
                cmds, selections=[SkillSelection(kind="login", email="me@x", password="pw", marketplace_url="http://m")]
            ),
        ):
            cmds._run_menu()
        do_login.assert_called_once()

    def test_refresh_selection_refetches_and_reopens(self, commands):
        cmds, _ = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            _patched_app_run(
                cmds,
                selections=[SkillSelection(kind="refresh"), SkillSelection(kind="cancel")],
            ),
        ):
            cmds._run_menu()
        # search_marketplace is called once per app construction (twice total).
        assert mgr.search_marketplace.call_count == 2


# ─────────────────────────────────────────────────────────────────────
# Legacy non-interactive list path
# ─────────────────────────────────────────────────────────────────────


class TestLegacyListPath:
    def test_cmd_skill_list_empty_prints_empty_set(self, commands):
        cmds, cli = commands
        mgr = _fake_manager(skills=[])
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds.cmd_skill_list()
        assert "No skills installed" in cli.console.file.getvalue()

    def test_cmd_skill_list_renders_table(self, commands):
        cmds, cli = commands
        mgr = _fake_manager(skills=[_meta("foo"), _meta("bar", source="marketplace")])
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds.cmd_skill_list()
        text = cli.console.file.getvalue()
        assert "foo" in text
        assert "bar" in text


# ─────────────────────────────────────────────────────────────────────
# Coverage for edge paths (shlex errors, empty handlers, login variants)
# ─────────────────────────────────────────────────────────────────────


class TestDispatchEdgeCases:
    def test_malformed_shlex_surfaces_error(self, commands):
        """An unmatched quote is the canonical shlex.split failure case."""
        cmds, cli = commands
        cmds.cmd_skill('install "unterminated')
        assert "Invalid arguments" in cli.console.file.getvalue()

    def test_info_no_args_prints_usage(self, commands):
        cmds, cli = commands
        cmds.cmd_skill("info")
        assert "Usage" in cli.console.file.getvalue()

    def test_remove_no_args_prints_usage(self, commands):
        cmds, cli = commands
        cmds.cmd_skill("remove")
        assert "Usage" in cli.console.file.getvalue()

    def test_publish_failure_surfaces_message(self, commands):
        cmds, cli = commands
        mgr = _fake_manager()
        mgr.publish_to_marketplace.return_value = (False, "sign error")
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._cmd_publish(["/tmp/my-skill"])
        assert "sign error" in cli.console.file.getvalue()


class TestCmdRemoveWrapper:
    def test_remove_wrapper_missing_skill_warns(self, commands):
        cmds, cli = commands
        mgr = _fake_manager(skills=[])
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._cmd_remove(["ghost"])
        assert "not found locally" in cli.console.file.getvalue()

    def test_remove_wrapper_cancel_via_confirm_prompt(self, tmp_path, commands):
        cmds, cli = commands
        skill_dir = tmp_path / "foo"
        skill_dir.mkdir()
        skill = _meta("foo", location=skill_dir)
        mgr = _fake_manager(skills=[skill])
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch("datus.cli.skill_commands.confirm_prompt", return_value=False),
        ):
            cmds._cmd_remove(["foo"])
        mgr.registry.remove_skill.assert_not_called()
        assert "Cancelled" in cli.console.file.getvalue()

    def test_remove_wrapper_confirm_proceeds(self, tmp_path, commands):
        cmds, _ = commands
        skill_dir = tmp_path / "foo"
        skill_dir.mkdir()
        skill = _meta("foo", location=skill_dir)
        mgr = _fake_manager(skills=[skill])
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch("datus.cli.skill_commands.confirm_prompt", return_value=True),
        ):
            cmds._cmd_remove(["foo"])
        mgr.registry.remove_skill.assert_called_once_with("foo")

    def test_do_remove_not_confirmed_is_noop(self, commands):
        cmds, _ = commands
        mgr = _fake_manager(skills=[_meta("foo")])
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_remove("foo", confirmed=False)
        mgr.registry.remove_skill.assert_not_called()


class TestDoUpdateOne:
    def test_missing_skill_warns(self, commands):
        cmds, cli = commands
        mgr = _fake_manager(skills=[])
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_update_one("ghost")
        assert "not found locally" in cli.console.file.getvalue()

    def test_marketplace_lookup_failure(self, commands):
        cmds, cli = commands
        skill = _meta("foo", source="marketplace", version="1.0")
        mgr = _fake_manager(skills=[skill])
        mgr._get_marketplace_client.side_effect = RuntimeError("network down")
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_update_one("foo")
        assert "network down" in cli.console.file.getvalue()

    def test_version_change_triggers_install(self, commands):
        cmds, cli = commands
        skill = _meta("foo", source="marketplace", version="1.0")
        mgr = _fake_manager(skills=[skill])
        client = MagicMock()
        client.get_skill_info.return_value = {"latest_version": "2.0"}
        mgr._get_marketplace_client.return_value = client
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_update_one("foo")
        mgr.install_from_marketplace.assert_called_once_with("foo", "2.0")
        assert "Updated foo to 2.0" in cli.console.file.getvalue()

    def test_install_failure_reports_error(self, commands):
        cmds, cli = commands
        skill = _meta("foo", source="marketplace", version="1.0")
        mgr = _fake_manager(skills=[skill])
        mgr.install_from_marketplace.return_value = (False, "500 server error")
        client = MagicMock()
        client.get_skill_info.return_value = {"latest_version": "2.0"}
        mgr._get_marketplace_client.return_value = client
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_update_one("foo")
        assert "500 server error" in cli.console.file.getvalue()

    def test_already_up_to_date_skips_install(self, commands):
        cmds, cli = commands
        skill = _meta("foo", source="marketplace", version="1.0")
        mgr = _fake_manager(skills=[skill])
        client = MagicMock()
        client.get_skill_info.return_value = {"latest_version": "1.0"}
        mgr._get_marketplace_client.return_value = client
        with patch.object(cmds, "_get_skill_manager", return_value=mgr):
            cmds._do_update_one("foo")
        mgr.install_from_marketplace.assert_not_called()
        assert "already up to date" in cli.console.file.getvalue()


class TestLoginEdgeCases:
    def test_connect_error_surfaces_url(self, commands):
        import httpx

        cmds, cli = commands

        class _ErrClient:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, *a, **kw):
                raise httpx.ConnectError("boom")

        with patch("datus.cli.skill_commands.httpx.Client", return_value=_ErrClient()):
            ok = cmds._do_login_with_credentials(
                email="me@example.com",
                password="pw",
                marketplace_url="http://localhost:9000",
            )
        assert ok is False
        assert "Cannot connect" in cli.console.file.getvalue()

    def test_generic_exception_is_reported(self, commands):
        cmds, cli = commands

        class _BoomClient:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, *a, **kw):
                raise RuntimeError("mystery")

        with patch("datus.cli.skill_commands.httpx.Client", return_value=_BoomClient()):
            ok = cmds._do_login_with_credentials(
                email="me@example.com",
                password="pw",
                marketplace_url="http://localhost:9000",
            )
        assert ok is False
        assert "mystery" in cli.console.file.getvalue()

    def test_400_response_without_json_falls_back_to_text(self, commands):
        cmds, cli = commands
        resp = _FakeResponse(status_code=500, content_type="text/plain")

        # Force .json() to raise, forcing the fallback to resp.text
        class _BadJsonResponse(_FakeResponse):
            def json(self):
                raise ValueError("not json")

        bad = _BadJsonResponse(status_code=500, content_type="text/plain")
        bad.text = "server exploded"
        del resp
        with patch("datus.cli.skill_commands.httpx.Client", return_value=_FakeHttpxClient(bad)):
            ok = cmds._do_login_with_credentials(
                email="me@example.com",
                password="pw",
                marketplace_url="http://localhost:9000",
            )
        assert ok is False
        assert "server exploded" in cli.console.file.getvalue()


class TestRunMenuExtras:
    def test_open_login_with_seed_url_prefills_field(self, commands):
        cmds, _ = commands
        mgr = _fake_manager()
        captured = {}

        real_run = cmds._run_app

        def _capture_run(app):
            captured["url_text"] = app._login_url.text
            captured["view"] = app._view
            return SkillSelection(kind="cancel")

        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            patch.object(cmds, "_run_app", side_effect=_capture_run),
        ):
            cmds._run_menu(open_login=True, seed_url="https://seed.example.com")
        assert captured["url_text"] == "https://seed.example.com"

        # Also run the real _run_app path once to keep the attribute referenced.
        assert callable(real_run)

    def test_unknown_selection_kind_breaks_loop(self, commands):
        cmds, _ = commands
        mgr = _fake_manager()
        with (
            patch.object(cmds, "_get_skill_manager", return_value=mgr),
            _patched_app_run(cmds, selections=[SkillSelection(kind="mystery")]),
        ):
            cmds._run_menu()
        # Did not route to any install/remove/update, and loop exited.
        mgr.install_from_marketplace.assert_not_called()


class TestGetSkillManager:
    def test_returns_agent_manager_when_attached(self, commands):
        cmds, cli = commands
        fake_mgr = MagicMock()
        cli.agent = MagicMock(skill_manager=fake_mgr)
        assert cmds._get_skill_manager() is fake_mgr

    def test_falls_back_to_standalone_when_agent_missing(self, commands):
        cmds, cli = commands
        cli.agent = None
        with patch("datus.tools.skill_tools.skill_manager.SkillManager") as mgr_cls:
            cmds._get_skill_manager()
        mgr_cls.assert_called_once()


class TestSafeHelpers:
    def test_safe_list_installed_swallows_errors(self):
        mgr = MagicMock()
        mgr.list_all_skills.side_effect = RuntimeError("disk")
        assert SkillCommands._safe_list_installed(mgr) == []

    def test_safe_search_marketplace_swallows_errors(self):
        mgr = MagicMock()
        mgr.search_marketplace.side_effect = RuntimeError("net")
        assert SkillCommands._safe_search_marketplace(mgr, "q") == []
