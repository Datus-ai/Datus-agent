# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for skill CLI handler (datus/cli/skill_cli.py)."""

from unittest.mock import MagicMock, patch


class TestRunSkillCommand:
    """Tests for run_skill_command dispatcher."""

    def _make_args(self, subcommand, skill_args=None, **kwargs):
        args = MagicMock()
        args.subcommand = subcommand
        args.skill_args = skill_args or []
        args.marketplace = None
        args.email = None
        args.password = None
        args.owner = ""
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args

    @patch("datus.cli.skill_cli._get_manager")
    @patch("datus.cli.skill_cli._cmd_list")
    def test_dispatch_list(self, mock_cmd, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("list")
        run_skill_command(args)
        mock_cmd.assert_called_once()

    @patch("datus.cli.skill_cli._get_manager")
    @patch("datus.cli.skill_cli._cmd_search")
    def test_dispatch_search(self, mock_cmd, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("search", skill_args=["sql"])
        run_skill_command(args)
        mock_cmd.assert_called_once()

    @patch("datus.cli.skill_cli._get_manager")
    @patch("datus.cli.skill_cli._cmd_install")
    def test_dispatch_install(self, mock_cmd, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("install", skill_args=["my-skill", "1.0"])
        run_skill_command(args)
        mock_cmd.assert_called_once()

    @patch("datus.cli.skill_cli._get_manager")
    def test_dispatch_install_no_args(self, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("install", skill_args=[])
        ret = run_skill_command(args)
        assert ret == 1

    @patch("datus.cli.skill_cli._get_manager")
    @patch("datus.cli.skill_cli._cmd_publish")
    def test_dispatch_publish(self, mock_cmd, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("publish", skill_args=["/some/path"])
        run_skill_command(args)
        mock_cmd.assert_called_once()

    @patch("datus.cli.skill_cli._get_manager")
    def test_dispatch_publish_no_args(self, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("publish", skill_args=[])
        ret = run_skill_command(args)
        assert ret == 1

    @patch("datus.cli.skill_cli._get_manager")
    @patch("datus.cli.skill_cli._cmd_info")
    def test_dispatch_info(self, mock_cmd, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("info", skill_args=["test-skill"])
        run_skill_command(args)
        mock_cmd.assert_called_once()

    @patch("datus.cli.skill_cli._get_manager")
    def test_dispatch_info_no_args(self, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("info", skill_args=[])
        ret = run_skill_command(args)
        assert ret == 1

    @patch("datus.cli.skill_cli._get_manager")
    @patch("datus.cli.skill_cli._cmd_update")
    def test_dispatch_update(self, mock_cmd, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("update")
        run_skill_command(args)
        mock_cmd.assert_called_once()

    @patch("datus.cli.skill_cli._get_manager")
    @patch("datus.cli.skill_cli._cmd_remove")
    def test_dispatch_remove(self, mock_cmd, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("remove", skill_args=["old-skill"])
        run_skill_command(args)
        mock_cmd.assert_called_once()

    @patch("datus.cli.skill_cli._get_manager")
    def test_dispatch_remove_no_args(self, mock_mgr):
        from datus.cli.skill_cli import run_skill_command

        args = self._make_args("remove", skill_args=[])
        ret = run_skill_command(args)
        assert ret == 1


class TestCmdLogin:
    """Tests for _cmd_login."""

    @patch("datus.cli.skill_cli.httpx.Client")
    @patch("datus.cli.skill_cli.save_token")
    def test_login_success_via_cookie(self, mock_save, mock_client_cls):
        from datus.cli.skill_cli import _cmd_login

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.cookies = {"town_token": "jwt-token-123"}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        args = MagicMock()
        args.email = "test@test.com"
        args.password = "pass"
        _cmd_login("http://localhost:9000", args)

        mock_save.assert_called_once_with("jwt-token-123", "http://localhost:9000", "test@test.com")

    @patch("datus.cli.skill_cli.console")
    @patch("datus.cli.skill_cli.httpx.Client")
    def test_login_failure_http_error(self, mock_client_cls, mock_console):
        from datus.cli.skill_cli import _cmd_login

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"detail": "Bad credentials"}
        mock_resp.text = "Unauthorized"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        args = MagicMock()
        args.email = "test@test.com"
        args.password = "wrong"
        _cmd_login("http://localhost:9000", args)

        # Deterministic path: must print exactly the 401 failure message with
        # the detail from the JSON body.
        mock_console.print.assert_called_once_with("[red]Login failed (401): Bad credentials[/]")

    @patch("datus.cli.skill_cli.console")
    @patch("datus.cli.skill_cli.httpx.Client")
    def test_login_no_token_returned(self, mock_client_cls, mock_console):
        from datus.cli.skill_cli import _cmd_login

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.cookies = {}
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        args = MagicMock()
        args.email = "test@test.com"
        args.password = "pass"
        _cmd_login("http://localhost:9000", args)

        # No token in response — must print exactly the "no token" warning.
        mock_console.print.assert_called_once_with("[red]Login succeeded but no token was returned.[/]")

    @patch("datus.cli.skill_cli.console")
    @patch("datus.cli.skill_cli.httpx.Client", side_effect=Exception("connect error"))
    def test_login_connection_error(self, mock_client_cls, mock_console):
        from datus.cli.skill_cli import _cmd_login

        args = MagicMock()
        args.email = "test@test.com"
        args.password = "pass"
        _cmd_login("http://localhost:9000", args)

        # Generic-exception path must propagate the exception detail verbatim.
        mock_console.print.assert_called_once_with("[red]Login error: connect error[/]")


class TestCmdLogout:
    """Tests for _cmd_logout."""

    @patch("datus.cli.skill_cli.clear_token", return_value=True)
    def test_logout_success(self, mock_clear):
        from datus.cli.skill_cli import _cmd_logout

        _cmd_logout("http://localhost:9000")
        mock_clear.assert_called_once_with("http://localhost:9000")

    @patch("datus.cli.skill_cli.clear_token", return_value=False)
    def test_logout_no_credentials(self, mock_clear):
        from datus.cli.skill_cli import _cmd_logout

        _cmd_logout("http://localhost:9000")
        mock_clear.assert_called_once()


class TestCmdList:
    """Tests for _cmd_list."""

    @patch("datus.cli.skill_cli.console")
    def test_list_no_skills(self, mock_console):
        from datus.cli.skill_cli import _cmd_list

        manager = MagicMock()
        manager.list_all_skills.return_value = []
        _cmd_list(manager)

        manager.list_all_skills.assert_called_once()
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("no skills" in s.lower() for s in printed_args), (
            "Expected console message indicating no installed skills"
        )

    @patch("datus.cli.skill_cli.console")
    def test_list_with_skills(self, mock_console):
        from pathlib import Path

        from datus.cli.skill_cli import _cmd_list
        from datus.tools.skill_tools.skill_config import SkillMetadata

        skill = SkillMetadata(name="test", description="A test", location=Path("/tmp"), version="1.0", tags=["sql"])
        manager = MagicMock()
        manager.list_all_skills.return_value = [skill]
        _cmd_list(manager)

        manager.list_all_skills.assert_called_once()
        # A Rich Table with one row must be printed
        mock_console.print.assert_called_once()
        from rich.table import Table

        printed_obj = mock_console.print.call_args[0][0]
        assert isinstance(printed_obj, Table), "Expected a Rich Table to be printed when skills exist"
        assert printed_obj.row_count == 1, "Expected table to contain exactly one row for the installed skill"


class TestCmdSearch:
    """Tests for _cmd_search."""

    @patch("datus.cli.skill_cli.console")
    def test_search_no_query(self, mock_console):
        from datus.cli.skill_cli import _cmd_search

        manager = MagicMock()
        _cmd_search(manager, "")

        # With empty query, search_marketplace must NOT be called
        manager.search_marketplace.assert_not_called()
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("usage" in s.lower() for s in printed_args), "Expected usage hint when query is empty"

    @patch("datus.cli.skill_cli.console")
    def test_search_with_results(self, mock_console):
        from datus.cli.skill_cli import _cmd_search

        manager = MagicMock()
        manager.search_marketplace.return_value = [{"name": "sql-opt", "latest_version": "1.0", "description": "SQL"}]
        _cmd_search(manager, "sql")

        manager.search_marketplace.assert_called_once_with(query="sql")
        # Result skill name must appear in console output
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("sql-opt" in s for s in printed_args), (
            "Expected console output to contain the result skill name 'sql-opt'"
        )

    @patch("datus.cli.skill_cli.console")
    def test_search_no_results(self, mock_console):
        from datus.cli.skill_cli import _cmd_search

        manager = MagicMock()
        manager.search_marketplace.return_value = []
        _cmd_search(manager, "nonexistent")

        manager.search_marketplace.assert_called_once_with(query="nonexistent")
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("no results" in s.lower() for s in printed_args), (
            "Expected 'no results' message when search returns empty list"
        )


class TestCmdInstall:
    """Tests for _cmd_install."""

    @patch("datus.cli.skill_cli.console")
    def test_install_success(self, mock_console):
        from datus.cli.skill_cli import _cmd_install

        manager = MagicMock()
        manager.install_from_marketplace.return_value = (True, "Installed ok")
        _cmd_install(manager, "test-skill", "latest")

        manager.install_from_marketplace.assert_called_once_with("test-skill", "latest")
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("Installed ok" in s for s in printed_args), (
            "Expected success message 'Installed ok' in console output"
        )

    @patch("datus.cli.skill_cli.console")
    def test_install_failure(self, mock_console):
        from datus.cli.skill_cli import _cmd_install

        manager = MagicMock()
        manager.install_from_marketplace.return_value = (False, "Not found")
        _cmd_install(manager, "nonexistent", "latest")

        manager.install_from_marketplace.assert_called_once_with("nonexistent", "latest")
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("Not found" in s for s in printed_args), "Expected failure message 'Not found' in console output"


class TestCmdPublish:
    """Tests for _cmd_publish."""

    @patch("datus.cli.skill_cli.console")
    def test_publish_success(self, mock_console):
        from datus.cli.skill_cli import _cmd_publish

        manager = MagicMock()
        manager.publish_to_marketplace.return_value = (True, "Published ok")
        _cmd_publish(manager, "/some/path", "")

        manager.publish_to_marketplace.assert_called_once_with("/some/path", owner="")
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("Published ok" in s for s in printed_args), (
            "Expected success message 'Published ok' in console output"
        )

    @patch("datus.cli.skill_cli.console")
    def test_publish_failure(self, mock_console):
        from datus.cli.skill_cli import _cmd_publish

        manager = MagicMock()
        manager.publish_to_marketplace.return_value = (False, "Error")
        _cmd_publish(manager, "/bad/path", "")

        manager.publish_to_marketplace.assert_called_once_with("/bad/path", owner="")
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("Error" in s for s in printed_args), "Expected failure message 'Error' in console output"


class TestCmdInfo:
    """Tests for _cmd_info."""

    @patch("datus.cli.skill_cli.console")
    def test_info_local_only(self, mock_console):
        from pathlib import Path

        from datus.cli.skill_cli import _cmd_info
        from datus.tools.skill_tools.skill_config import SkillMetadata

        skill = SkillMetadata(name="test", description="A test", location=Path("/tmp"))
        manager = MagicMock()
        manager.get_skill.return_value = skill
        client = MagicMock()
        client.get_skill_info.side_effect = Exception("offline")
        manager._get_marketplace_client.return_value = client
        _cmd_info(manager, "test")

        manager.get_skill.assert_called_once_with("test")
        # Local skill info (name) must appear in console output
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("test" in s for s in printed_args), "Expected local skill name 'test' to appear in console output"

    @patch("datus.cli.skill_cli.console")
    def test_info_not_found(self, mock_console):
        from datus.cli.skill_cli import _cmd_info

        manager = MagicMock()
        manager.get_skill.return_value = None
        client = MagicMock()
        client.get_skill_info.side_effect = Exception("not found")
        manager._get_marketplace_client.return_value = client
        _cmd_info(manager, "unknown")

        manager.get_skill.assert_called_once_with("unknown")
        # Neither local nor marketplace found — must print a "not found" message
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("not found" in s.lower() for s in printed_args), (
            "Expected 'not found' message when skill does not exist locally or in marketplace"
        )


class TestCmdUpdate:
    """Tests for _cmd_update."""

    @patch("datus.cli.skill_cli.console")
    def test_update_no_marketplace_skills(self, mock_console):
        from pathlib import Path

        from datus.cli.skill_cli import _cmd_update
        from datus.tools.skill_tools.skill_config import SkillMetadata

        skill = SkillMetadata(name="local", description="test", location=Path("/tmp"), source="local")
        manager = MagicMock()
        manager.list_all_skills.return_value = [skill]
        _cmd_update(manager)

        manager.list_all_skills.assert_called_once()
        # No marketplace skills — install must never be called
        manager.install_from_marketplace.assert_not_called()
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("no marketplace" in s.lower() for s in printed_args), (
            "Expected message indicating no marketplace skills to update"
        )

    @patch("datus.cli.skill_cli.console")
    def test_update_with_marketplace_skills(self, mock_console):
        from pathlib import Path

        from datus.cli.skill_cli import _cmd_update
        from datus.tools.skill_tools.skill_config import SkillMetadata

        skill = SkillMetadata(name="mp-skill", description="test", location=Path("/tmp"), source="marketplace")
        manager = MagicMock()
        manager.list_all_skills.return_value = [skill]
        manager.install_from_marketplace.return_value = (True, "Updated")
        _cmd_update(manager)

        manager.install_from_marketplace.assert_called_once_with("mp-skill")
        # "Updated" success message must appear in console output
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("updated" in s.lower() for s in printed_args), (
            "Expected 'Updated' confirmation in console output after successful update"
        )


class TestCmdRemove:
    """Tests for _cmd_remove."""

    @patch("datus.cli.skill_cli.console")
    def test_remove_success(self, mock_console):
        from datus.cli.skill_cli import _cmd_remove

        manager = MagicMock()
        manager.registry.remove_skill.return_value = True
        _cmd_remove(manager, "test")

        manager.registry.remove_skill.assert_called_once_with("test")
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("test" in s and ("removed" in s.lower() or "Removed" in s) for s in printed_args), (
            "Expected 'Removed test' confirmation in console output"
        )

    @patch("datus.cli.skill_cli.console")
    def test_remove_not_found(self, mock_console):
        from datus.cli.skill_cli import _cmd_remove

        manager = MagicMock()
        manager.registry.remove_skill.return_value = False
        _cmd_remove(manager, "unknown")

        manager.registry.remove_skill.assert_called_once_with("unknown")
        printed_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("unknown" in s and "not found" in s.lower() for s in printed_args), (
            "Expected 'not found' message for unknown skill in console output"
        )


class TestGetManager:
    """Tests for _get_manager helper."""

    def test_get_manager_default(self):
        from datus.cli.skill_cli import _get_manager

        args = MagicMock()
        args.marketplace = None
        manager = _get_manager(args)
        assert manager is not None

    def test_get_manager_custom_marketplace(self):
        from datus.cli.skill_cli import _get_manager

        args = MagicMock()
        args.marketplace = "http://custom:8080"
        manager = _get_manager(args)
        assert manager.config.marketplace_url == "http://custom:8080"
