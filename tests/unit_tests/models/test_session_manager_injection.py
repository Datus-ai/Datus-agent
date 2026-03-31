# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from types import SimpleNamespace

from datus.models.session_manager import SessionManager
from datus.utils.path_manager import DatusPathManager


class TestSessionManagerPathManagerInjection:
    def test_uses_explicit_path_manager_when_session_dir_missing(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = SessionManager(path_manager=path_manager)
        try:
            assert manager.session_dir == str(path_manager.sessions_dir)
            assert path_manager.sessions_dir.exists()
        finally:
            manager.close_all_sessions()

    def test_uses_agent_config_path_manager_when_session_dir_missing(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        agent_config = SimpleNamespace(path_manager=path_manager)
        manager = SessionManager(agent_config=agent_config)
        try:
            assert manager.session_dir == str(path_manager.sessions_dir)
            assert path_manager.sessions_dir.exists()
        finally:
            manager.close_all_sessions()

    def test_blank_session_dir_falls_back_to_path_manager(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = SessionManager(session_dir="   ", path_manager=path_manager)
        try:
            assert manager.session_dir == str(path_manager.sessions_dir)
        finally:
            manager.close_all_sessions()
