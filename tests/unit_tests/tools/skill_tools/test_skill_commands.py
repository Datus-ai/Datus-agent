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


class TestSkillCommandsSearch:
    def test_search_empty_query(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill_search("")

    def test_search_with_results(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.search_marketplace.return_value = [
                {"name": "sql-opt", "latest_version": "1.0", "owner": "test", "tags": ["sql"], "description": "SQL"}
            ]
            cmds.cmd_skill_search("sql")

    def test_search_no_results(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.search_marketplace.return_value = []
            cmds.cmd_skill_search("nonexistent")


class TestSkillCommandsInstall:
    def test_install_no_args(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill_install("")

    def test_install_success(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.install_from_marketplace.return_value = (True, "Installed ok")
            cmds.cmd_skill_install("test-skill")

    def test_install_failure(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.install_from_marketplace.return_value = (False, "Error")
            cmds.cmd_skill_install("test-skill")

    def test_install_with_version(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.install_from_marketplace.return_value = (True, "ok")
            cmds.cmd_skill_install("test-skill 2.0")


class TestSkillCommandsPublish:
    def test_publish_no_args(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill_publish("")

    def test_publish_success(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.publish_to_marketplace.return_value = (True, "Published")
            cmds.cmd_skill_publish("/some/path")

    def test_publish_with_owner(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.publish_to_marketplace.return_value = (True, "Published")
            cmds.cmd_skill_publish("/path --owner myname")


class TestSkillCommandsInfo:
    def test_info_empty_name(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill_info("")

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


class TestSkillCommandsUpdate:
    def test_update_no_marketplace_skills(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.list_all_skills.return_value = [_make_skill(source="local")]
            cmds.cmd_skill_update()

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


class TestSkillCommandsRemove:
    def test_remove_empty_name(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        cmds.cmd_skill_remove("")

    def test_remove_not_found(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.get_skill.return_value = None
            cmds.cmd_skill_remove("unknown")

    def test_remove_local_skill(self):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        skill = _make_skill(source="local")
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.get_skill.return_value = skill
            mock_mgr.return_value.registry.remove_skill.return_value = True
            cmds.cmd_skill_remove("test-skill")

    @patch("builtins.input", return_value="y")
    def test_remove_marketplace_skill_deletes_files(self, mock_input, tmp_path):
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
    @patch("builtins.input", return_value="test@test.com")
    @patch("getpass.getpass", return_value="pass")
    @patch("httpx.Client")
    def test_login_success(self, mock_client_cls, mock_getpass, mock_input, mock_save):
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

    @patch("builtins.input", return_value="test@test.com")
    @patch("getpass.getpass", return_value="wrong")
    @patch("httpx.Client")
    def test_login_failure(self, mock_client_cls, mock_getpass, mock_input):
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

    @patch("builtins.input", return_value="test@test.com")
    @patch("getpass.getpass", return_value="pass")
    @patch("httpx.Client", side_effect=Exception("conn error"))
    def test_login_connection_error(self, mock_client_cls, mock_getpass, mock_input):
        from datus.cli.skill_commands import SkillCommands

        cli = _make_cli_mock()
        cmds = SkillCommands(cli)
        with patch.object(cmds, "_get_skill_manager") as mock_mgr:
            mock_mgr.return_value.config.marketplace_url = "http://localhost:9000"
            cmds.cmd_skill_login()


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
