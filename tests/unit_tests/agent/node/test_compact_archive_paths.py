"""Where the compact archive lands, when the host moved the session tree.

``AgentConfig.session_dir`` is an override, and a host that sets it usually has
a reason the path manager cannot see. The publication runtime is the sharp
case: its ``home`` is a version snapshot mounted READ-ONLY, and the writable
session tree deliberately sits outside it. Deriving the archive directory from
``home`` therefore tried to mkdir inside the snapshot and failed with ``EROFS``
on every compaction — silently, since the archive is optional and the fallback
is to leave long tool output inline.
"""

import os
from pathlib import Path
from typing import AsyncGenerator, Optional
from unittest.mock import patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import CompactConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.path_manager import DatusPathManager


class _Node(AgenticNode):
    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        yield  # pragma: no cover

    def get_node_name(self) -> str:
        return "test_chat"


class _Config:
    """Only what ``_get_archive`` and ``session_manager`` read off the config."""

    def __init__(self, path_manager, session_dir):
        self.path_manager = path_manager
        self.session_dir = session_dir


def _publication_layout(tmp_path, *, read_only=True):
    """The two trees a published surface runs with."""
    snapshot = tmp_path / "publications" / "p1" / "versions" / "v1"
    (snapshot / "files").mkdir(parents=True)
    sessions = tmp_path / "publication-sessions" / "p1" / "user_x"
    sessions.mkdir(parents=True)
    if read_only:
        os.chmod(snapshot, 0o555)
    return snapshot, sessions


def _build_node(snapshot, sessions, *, scope=None, session_id="sid_test"):
    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = _Node.__new__(_Node)
    node.agent_config = _Config(
        DatusPathManager(str(snapshot), project_name="v1", project_root=str(snapshot / "files")),
        str(sessions),
    )
    node.session_id = session_id
    node._archive = None
    node._session_manager = None
    node._compact_cfg = CompactConfig()
    node._compact_cfg.minor.archive_preview_chars = 50
    if scope is not None:
        node.scope = scope
    return node


@pytest.fixture
def layout(tmp_path):
    snapshot, sessions = _publication_layout(tmp_path)
    yield snapshot, sessions
    # Let pytest clean the tree up.
    os.chmod(snapshot, 0o755)


def test_archive_is_built_when_home_is_read_only(layout):
    """The regression: this used to return None and log a warning."""
    snapshot, sessions = layout
    node = _build_node(snapshot, sessions)

    archive = node._get_archive()

    assert archive is not None
    assert archive.dir.is_dir()
    assert os.access(archive.dir, os.W_OK)


def test_the_archive_is_not_written_into_the_snapshot(layout):
    snapshot, sessions = layout
    node = _build_node(snapshot, sessions)

    archive = node._get_archive()

    assert snapshot not in archive.dir.parents
    assert sessions in archive.dir.parents


def test_the_archive_sits_beside_the_session_database(layout):
    """What ``DatusPathManager.session_dir`` documents, and what broke.

    The db goes wherever the session manager says; the archive used to go
    wherever the path manager said. Once those two disagree, deleting a session
    no longer takes its archive with it.
    """
    snapshot, sessions = layout
    node = _build_node(snapshot, sessions)

    archive = node._get_archive()
    db_dir = node.session_manager.session_dir

    assert archive.dir == Path(db_dir) / "sid_test" / "data"


def test_the_archive_follows_the_session_scope(layout):
    """Two consumers of one publication must not share an archive directory."""
    snapshot, sessions = layout
    node = _build_node(snapshot, sessions, scope="user_b")

    archive = node._get_archive()

    assert "user_b" in archive.dir.parts
    assert Path(node.session_manager.session_dir) in archive.dir.parents


def test_a_writable_home_still_works(tmp_path):
    """The ordinary shape is unaffected — the override is what changes things."""
    snapshot, sessions = _publication_layout(tmp_path, read_only=False)
    node = _build_node(snapshot, sessions)

    archive = node._get_archive()

    assert archive is not None
    assert sessions in archive.dir.parents
