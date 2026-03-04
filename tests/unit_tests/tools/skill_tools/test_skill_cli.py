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

    @patch("datus.cli.skill_cli.httpx.Client")
    def test_login_failure_http_error(self, mock_client_cls):
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
        # Should not raise
        _cmd_login("http://localhost:9000", args)

    @patch("datus.cli.skill_cli.httpx.Client")
    def test_login_no_token_returned(self, mock_client_cls):
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
        # Should not raise
        _cmd_login("http://localhost:9000", args)

    @patch("datus.cli.skill_cli.httpx.Client", side_effect=Exception("connect error"))
    def test_login_connection_error(self, mock_client_cls):
        from datus.cli.skill_cli import _cmd_login

        args = MagicMock()
        args.email = "test@test.com"
        args.password = "pass"
        # Should not raise
        _cmd_login("http://localhost:9000", args)


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

    def test_list_no_skills(self):
        from datus.cli.skill_cli import _cmd_list

        manager = MagicMock()
        manager.list_all_skills.return_value = []
        _cmd_list(manager)

    def test_list_with_skills(self):
        from pathlib import Path

        from datus.cli.skill_cli import _cmd_list
        from datus.tools.skill_tools.skill_config import SkillMetadata

        skill = SkillMetadata(name="test", description="A test", location=Path("/tmp"), version="1.0", tags=["sql"])
        manager = MagicMock()
        manager.list_all_skills.return_value = [skill]
        _cmd_list(manager)


class TestCmdSearch:
    """Tests for _cmd_search."""

    def test_search_no_query(self):
        from datus.cli.skill_cli import _cmd_search

        manager = MagicMock()
        _cmd_search(manager, "")

    def test_search_with_results(self):
        from datus.cli.skill_cli import _cmd_search

        manager = MagicMock()
        manager.search_marketplace.return_value = [{"name": "sql-opt", "latest_version": "1.0", "description": "SQL"}]
        _cmd_search(manager, "sql")

    def test_search_no_results(self):
        from datus.cli.skill_cli import _cmd_search

        manager = MagicMock()
        manager.search_marketplace.return_value = []
        _cmd_search(manager, "nonexistent")


class TestCmdInstall:
    """Tests for _cmd_install."""

    def test_install_success(self):
        from datus.cli.skill_cli import _cmd_install

        manager = MagicMock()
        manager.install_from_marketplace.return_value = (True, "Installed ok")
        _cmd_install(manager, "test-skill", "latest")

    def test_install_failure(self):
        from datus.cli.skill_cli import _cmd_install

        manager = MagicMock()
        manager.install_from_marketplace.return_value = (False, "Not found")
        _cmd_install(manager, "nonexistent", "latest")


class TestCmdPublish:
    """Tests for _cmd_publish."""

    def test_publish_success(self):
        from datus.cli.skill_cli import _cmd_publish

        manager = MagicMock()
        manager.publish_to_marketplace.return_value = (True, "Published ok")
        _cmd_publish(manager, "/some/path", "")

    def test_publish_failure(self):
        from datus.cli.skill_cli import _cmd_publish

        manager = MagicMock()
        manager.publish_to_marketplace.return_value = (False, "Error")
        _cmd_publish(manager, "/bad/path", "")


class TestCmdInfo:
    """Tests for _cmd_info."""

    def test_info_local_only(self):
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

    def test_info_not_found(self):
        from datus.cli.skill_cli import _cmd_info

        manager = MagicMock()
        manager.get_skill.return_value = None
        client = MagicMock()
        client.get_skill_info.side_effect = Exception("not found")
        manager._get_marketplace_client.return_value = client
        _cmd_info(manager, "unknown")


class TestCmdUpdate:
    """Tests for _cmd_update."""

    def test_update_no_marketplace_skills(self):
        from pathlib import Path

        from datus.cli.skill_cli import _cmd_update
        from datus.tools.skill_tools.skill_config import SkillMetadata

        skill = SkillMetadata(name="local", description="test", location=Path("/tmp"), source="local")
        manager = MagicMock()
        manager.list_all_skills.return_value = [skill]
        _cmd_update(manager)

    def test_update_with_marketplace_skills(self):
        from pathlib import Path

        from datus.cli.skill_cli import _cmd_update
        from datus.tools.skill_tools.skill_config import SkillMetadata

        skill = SkillMetadata(name="mp-skill", description="test", location=Path("/tmp"), source="marketplace")
        manager = MagicMock()
        manager.list_all_skills.return_value = [skill]
        manager.install_from_marketplace.return_value = (True, "Updated")
        _cmd_update(manager)


class TestCmdRemove:
    """Tests for _cmd_remove."""

    def test_remove_success(self):
        from datus.cli.skill_cli import _cmd_remove

        manager = MagicMock()
        manager.registry.remove_skill.return_value = True
        _cmd_remove(manager, "test")

    def test_remove_not_found(self):
        from datus.cli.skill_cli import _cmd_remove

        manager = MagicMock()
        manager.registry.remove_skill.return_value = False
        _cmd_remove(manager, "unknown")


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
