# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for SkillCommands (datus/cli/skill_commands.py)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from datus.tools.skill_tools.skill_config import SkillMetadata


def _make_cli_mock():
    """Create a mock DatusCLI instance."""
    cli = MagicMock()
    cli.console = MagicMock()
    cli.agent = None
    cli.agent_config = None
    return cli


def _make_skill(**kwargs):
    defaults = dict(name="test-skill", description="A test skill", location=Path("/tmp/test"), tags=["sql"])
    defaults.update(kwargs)
    return SkillMetadata(**defaults)


class TestSkillCommandsDispatch:
    """Tests for cmd_skill dispatcher."""

    def test_empty_shows_usage(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill("")
        cli.console.print.assert_called()

    def test_help_shows_usage(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill("help")
        cli.console.print.assert_called()

    def test_unknown_command(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill("nonexistent")
        # Should print error and usage
        assert cli.console.print.call_count >= 2

    @patch.object(__import__("datus.cli.skill_commands", fromlist=["SkillCommands"]).SkillCommands, "cmd_skill_list")
    def test_dispatch_list(self, mock_list):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill("list")
        mock_list.assert_called_once()

    @patch.object(__import__("datus.cli.skill_commands", fromlist=["SkillCommands"]).SkillCommands, "cmd_skill_search")
    def test_dispatch_search(self, mock_search):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill("search sql")
        mock_search.assert_called_once_with("sql")

    @patch.object(__import__("datus.cli.skill_commands", fromlist=["SkillCommands"]).SkillCommands, "cmd_skill_install")
    def test_dispatch_install(self, mock_install):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill("install my-skill 1.0")
        mock_install.assert_called_once_with("my-skill 1.0")

    @patch.object(__import__("datus.cli.skill_commands", fromlist=["SkillCommands"]).SkillCommands, "cmd_skill_publish")
    def test_dispatch_publish(self, mock_pub):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill("publish /some/path")
        mock_pub.assert_called_once_with("/some/path")

    @patch.object(__import__("datus.cli.skill_commands", fromlist=["SkillCommands"]).SkillCommands, "cmd_skill_info")
    def test_dispatch_info(self, mock_info):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill("info test-skill")
        mock_info.assert_called_once_with("test-skill")

    @patch.object(__import__("datus.cli.skill_commands", fromlist=["SkillCommands"]).SkillCommands, "cmd_skill_update")
    def test_dispatch_update(self, mock_update):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill("update")
        mock_update.assert_called_once()

    @patch.object(__import__("datus.cli.skill_commands", fromlist=["SkillCommands"]).SkillCommands, "cmd_skill_remove")
    def test_dispatch_remove(self, mock_rm):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill("remove old-skill")
        mock_rm.assert_called_once_with("old-skill")


class TestSkillCommandsList:
    def test_list_no_skills(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.list_all_skills.return_value = []
            cmds.cmd_skill_list()
            cli.console.print.assert_called()

    def test_list_with_skills(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        skill = _make_skill(version="1.0", source="local")
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.list_all_skills.return_value = [skill]
            cmds.cmd_skill_list()
            cli.console.print.assert_called()

    def test_list_with_none_description(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        skill = _make_skill(description="")
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.list_all_skills.return_value = [skill]
            cmds.cmd_skill_list()
            cli.console.print.assert_called()


class TestSkillCommandsSearch:
    def test_search_empty_query(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill_search("")
        # Empty query should print usage hint without calling manager
        cli.console.print.assert_called_once()
        printed_text = str(cli.console.print.call_args)
        assert "Usage" in printed_text, "Empty query should print usage hint"

    def test_search_with_results(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.search_marketplace.return_value = [
                {"name": "sql-opt", "latest_version": "1.0", "owner": "test", "tags": ["sql"], "description": "SQL"}
            ]
            cmds.cmd_skill_search("sql")
            mock_mgr.return_value.search_marketplace.assert_called_once_with(query="sql")
            # The rendered Rich Table must contain exactly one row (the single
            # returned skill). `call_count >= 2` would pass for an empty table.
            from rich.table import Table

            printed_tables = [
                c.args[0] for c in cli.console.print.call_args_list if c.args and isinstance(c.args[0], Table)
            ]
            assert len(printed_tables) == 1, f"Expected exactly one Rich Table, got {len(printed_tables)}"
            assert printed_tables[0].row_count == 1, f"Expected 1 result row, got {printed_tables[0].row_count}"

    def test_search_no_results(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.search_marketplace.return_value = []
            cmds.cmd_skill_search("nonexistent")
            mock_mgr.return_value.search_marketplace.assert_called_once_with(query="nonexistent")
            printed_text = str(cli.console.print.call_args_list)
            assert "No skills found" in printed_text, "Should notify user when no results"


class TestSkillCommandsInstall:
    def test_install_no_args(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill_install("")
        cli.console.print.assert_called_once()
        printed_text = str(cli.console.print.call_args)
        assert "Usage" in printed_text, "Empty args should print usage hint"

    def test_install_success(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.install_from_marketplace.return_value = (True, "Installed ok")
            cmds.cmd_skill_install("test-skill")
            mock_mgr.return_value.install_from_marketplace.assert_called_once_with("test-skill", "latest")
            printed_text = str(cli.console.print.call_args_list)
            assert "Installed ok" in printed_text, "Successful install should print success message"

    def test_install_failure(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.install_from_marketplace.return_value = (False, "Error")
            cmds.cmd_skill_install("test-skill")
            mock_mgr.return_value.install_from_marketplace.assert_called_once_with("test-skill", "latest")
            printed_text = str(cli.console.print.call_args_list)
            assert "Error" in printed_text, "Failed install should print error message"

    def test_install_with_version(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.install_from_marketplace.return_value = (True, "ok")
            cmds.cmd_skill_install("test-skill 2.0")
            mock_mgr.return_value.install_from_marketplace.assert_called_once_with("test-skill", "2.0")


class TestSkillCommandsPublish:
    def test_publish_no_args(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill_publish("")
        cli.console.print.assert_called_once()
        printed_text = str(cli.console.print.call_args)
        assert "Usage" in printed_text, "Empty args should print usage hint"

    def test_publish_success(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.publish_to_marketplace.return_value = (True, "Published")
            cmds.cmd_skill_publish("/some/path")
            mock_mgr.return_value.publish_to_marketplace.assert_called_once_with("/some/path", owner="")
            printed_text = str(cli.console.print.call_args_list)
            assert "Published" in printed_text, "Successful publish should print success message"

    def test_publish_with_owner(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.publish_to_marketplace.return_value = (True, "Published")
            cmds.cmd_skill_publish("/path --owner myname")
            mock_mgr.return_value.publish_to_marketplace.assert_called_once_with("/path", owner="myname")


class TestSkillCommandsInfo:
    def test_info_empty_name(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill_info("")
        cli.console.print.assert_called_once()
        printed_text = str(cli.console.print.call_args)
        assert "Usage" in printed_text, "Empty name should print usage hint"

    def test_info_local_skill(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        skill = _make_skill(version="1.0", license="MIT")
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.get_skill.return_value = skill
            mock_client = MagicMock()
            mock_client.get_skill_info.return_value = {
                "name": "test-skill",
                "latest_version": "1.0",
                "owner": "tester",
                "promoted": False,
                "usage_count": 5,
                "versions": [{"version": "1.0"}],
            }
            mock_mgr.return_value._get_marketplace_client.return_value = mock_client
            cmds.cmd_skill_info("test-skill")
            mock_mgr.return_value.get_skill.assert_called_once_with("test-skill")
            printed_text = str(cli.console.print.call_args_list)
            assert "test-skill" in printed_text, "Should print local skill name"

    def test_info_not_found(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.get_skill.return_value = None
            mock_client = MagicMock()
            mock_client.get_skill_info.side_effect = Exception("not found")
            mock_mgr.return_value._get_marketplace_client.return_value = mock_client
            cmds.cmd_skill_info("unknown")
            mock_mgr.return_value.get_skill.assert_called_once_with("unknown")
            printed_text = str(cli.console.print.call_args_list)
            # Pin to the deterministic branch — the string "unknown" is the
            # skill name we passed in and would appear in any echoed output,
            # making that fallback near-tautological.
            assert "not found" in printed_text.lower(), (
                "Should notify user when skill not found locally or in marketplace"
            )

    def test_info_marketplace_error(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        skill = _make_skill()
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.get_skill.return_value = skill
            mock_client = MagicMock()
            mock_client.get_skill_info.side_effect = Exception("timeout")
            mock_mgr.return_value._get_marketplace_client.return_value = mock_client
            cmds.cmd_skill_info("test-skill")
            mock_mgr.return_value.get_skill.assert_called_once_with("test-skill")
            printed_text = str(cli.console.print.call_args_list)
            # Pin to the exception detail — the generic "Marketplace lookup
            # failed" fallback would pass even if the exception message was
            # silently swallowed.
            assert "timeout" in printed_text, "Marketplace error path must surface the exception detail"


class TestSkillCommandsUpdate:
    def test_update_no_marketplace_skills(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.list_all_skills.return_value = [_make_skill(source="local")]
            cmds.cmd_skill_update()
            mock_mgr.return_value.install_from_marketplace.assert_not_called()
            printed_text = str(cli.console.print.call_args_list)
            assert "No marketplace" in printed_text, "Should notify when no marketplace skills to update"

    def test_update_with_version_change(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        skill = _make_skill(source="marketplace", version="1.0")
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.list_all_skills.return_value = [skill]
            mock_client = MagicMock()
            mock_client.get_skill_info.return_value = {"latest_version": "2.0"}
            mock_mgr.return_value._get_marketplace_client.return_value = mock_client
            mock_mgr.return_value.install_from_marketplace.return_value = (True, "Updated")
            cmds.cmd_skill_update()
            mock_mgr.return_value.install_from_marketplace.assert_called_once_with("test-skill", "2.0")
            printed_text = str(cli.console.print.call_args_list)
            # "2.0" is the version number that's likely to show up in any
            # status output, so the OR fallback made the assertion near-
            # tautological. Pin to the install-result status string.
            assert "Updated" in printed_text, "Should report updated version via install-result status"

    def test_update_already_latest(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        skill = _make_skill(source="marketplace", version="1.0")
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.list_all_skills.return_value = [skill]
            mock_client = MagicMock()
            mock_client.get_skill_info.return_value = {"latest_version": "1.0"}
            mock_mgr.return_value._get_marketplace_client.return_value = mock_client
            cmds.cmd_skill_update()
            mock_mgr.return_value.install_from_marketplace.assert_not_called()
            printed_text = str(cli.console.print.call_args_list)
            assert "up to date" in printed_text, "Should report already up to date"

    def test_update_error(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        skill = _make_skill(source="marketplace", version="1.0")
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.list_all_skills.return_value = [skill]
            mock_client = MagicMock()
            mock_client.get_skill_info.side_effect = Exception("offline")
            mock_mgr.return_value._get_marketplace_client.return_value = mock_client
            cmds.cmd_skill_update()
            mock_mgr.return_value.install_from_marketplace.assert_not_called()
            printed_text = str(cli.console.print.call_args_list)
            # Pin to the exception detail — generic "Error" fallback would
            # pass even if the actual exception text was swallowed.
            assert "offline" in printed_text, "Update error path must surface the exception detail"


class TestSkillCommandsRemove:
    def test_remove_empty_name(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill_remove("")
        cli.console.print.assert_called_once()
        printed_text = str(cli.console.print.call_args)
        assert "Usage" in printed_text, "Empty name should print usage hint"

    def test_remove_not_found(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.get_skill.return_value = None
            cmds.cmd_skill_remove("unknown")
            mock_mgr.return_value.get_skill.assert_called_once_with("unknown")
            mock_mgr.return_value.registry.remove_skill.assert_not_called()
            printed_text = str(cli.console.print.call_args_list)
            assert "not found" in printed_text, "Should notify when skill is not found locally"

    def test_remove_local_skill(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        skill = _make_skill(source="local")
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.get_skill.return_value = skill
            mock_mgr.return_value.registry.remove_skill.return_value = True
            cmds.cmd_skill_remove("test-skill")
            mock_mgr.return_value.registry.remove_skill.assert_called_once_with("test-skill")
            printed_text = str(cli.console.print.call_args_list)
            assert "Removed" in printed_text, "Should confirm successful skill removal"

    @patch("datus.cli._cli_utils.confirm_prompt", return_value=True)
    def test_remove_marketplace_skill_deletes_files(self, mock_confirm, tmp_path):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("test")
        skill = _make_skill(source="marketplace", location=skill_dir)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.get_skill.return_value = skill
            mock_mgr.return_value.registry.remove_skill.return_value = True
            cmds.cmd_skill_remove("test-skill")
            assert not skill_dir.exists()


class TestSkillCommandsLogin:
    @patch("datus.tools.skill_tools.marketplace_auth.save_token")
    @patch("datus.cli._cli_utils.prompt_input", side_effect=["test@test.com", "pass"])
    @patch("httpx.Client")
    def test_login_success(self, mock_client_cls, mock_prompt, mock_save):
        from datus.cli.skill_commands import SkillCommands

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.cookies = {"town_token": "jwt-123"}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.config.marketplace_url = "http://localhost:9000"
            cmds.cmd_skill_login()
        mock_save.assert_called_once()

    @patch("datus.cli._cli_utils.prompt_input", side_effect=["test@test.com", "wrong"])
    @patch("httpx.Client")
    def test_login_failure(self, mock_client_cls, mock_prompt):
        from datus.cli.skill_commands import SkillCommands

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"detail": "Bad credentials"}
        mock_resp.text = "Unauthorized"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.config.marketplace_url = "http://localhost:9000"
            cmds.cmd_skill_login()
            printed_text = str(cli.console.print.call_args_list)
            assert "Login failed" in printed_text or "401" in printed_text, (
                "Failed login should print error message with status code or failure reason"
            )

    @patch("datus.cli._cli_utils.prompt_input", side_effect=["test@test.com", "pass"])
    @patch("httpx.Client", side_effect=Exception("conn error"))
    def test_login_connection_error(self, mock_client_cls, mock_prompt):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.config.marketplace_url = "http://localhost:9000"
            cmds.cmd_skill_login()
            printed_text = str(cli.console.print.call_args_list)
            # The exception detail must reach the user verbatim. Dropping the
            # "Login error" fallback: a generic message would pass even if the
            # exception text was silently swallowed.
            assert "conn error" in printed_text, "Connection error must surface the exception detail"


class TestSkillCommandsLogout:
    @patch("datus.tools.skill_tools.marketplace_auth.clear_token", return_value=True)
    def test_logout_success(self, mock_clear):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.config.marketplace_url = "http://localhost:9000"
            cmds.cmd_skill_logout()
        mock_clear.assert_called_once()

    @patch("datus.tools.skill_tools.marketplace_auth.clear_token", return_value=False)
    def test_logout_no_credentials(self, mock_clear):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.config.marketplace_url = "http://localhost:9000"
            cmds.cmd_skill_logout()
            mock_clear.assert_called_once()
            printed_text = str(cli.console.print.call_args_list)
            assert "No saved credentials" in printed_text, "Logout with no stored credentials should warn user"


class TestGetSkillManager:
    def test_from_agent(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cli.agent = MagicMock()
        cli.agent.skill_manager = MagicMock()
        cmds = SkillCommands(cli)
        manager = cmds._get_skill_manager()
        assert manager is cli.agent.skill_manager

    def test_standalone(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cli.agent = None
        cmds = SkillCommands(cli)
        manager = cmds._get_skill_manager()
        assert manager is not None
