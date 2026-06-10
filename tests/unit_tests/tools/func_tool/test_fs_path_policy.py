# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/fs_path_policy.py"""

from pathlib import Path

import pytest

from datus.tools.func_tool.fs_path_policy import (
    PathZone,
    build_walk_patterns,
    classify_path,
    whitelist_anchors,
)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def fake_home(tmp_path):
    home = tmp_path / "fake_home" / ".datus"
    (home / "skills").mkdir(parents=True)
    return home


class TestClassifyInternal:
    def test_relative_inside_root(self, project):
        r = classify_path("src/main.py", root_path=project, current_node="chat")
        assert r.zone == PathZone.INTERNAL
        assert r.display == "src/main.py"

    def test_dot_maps_to_root(self, project):
        r = classify_path(".", root_path=project, current_node="chat")
        assert r.zone == PathZone.INTERNAL
        assert r.display == "."

    def test_absolute_inside_root_is_internal(self, project):
        r = classify_path(str(project / "a.md"), root_path=project, current_node="chat")
        assert r.zone == PathZone.INTERNAL


class TestClassifyHidden:
    def test_datus_subdir_is_hidden(self, project):
        r = classify_path(".datus/sessions/foo.db", root_path=project, current_node="chat")
        assert r.zone == PathZone.HIDDEN

    def test_datus_root_itself_is_hidden(self, project):
        r = classify_path(".datus", root_path=project, current_node="chat")
        assert r.zone == PathZone.HIDDEN


class TestClassifyWhitelist:
    def test_project_skills_whitelisted(self, project):
        r = classify_path(".datus/skills/foo/SKILL.md", root_path=project, current_node="chat")
        assert r.zone == PathZone.WHITELIST
        assert r.display.startswith(".datus/skills/")

    def test_own_memory_dir_is_hidden(self, project):
        # Memory is owned exclusively by the dedicated add_memory/edit_memory
        # tools; the whole subtree is HIDDEN to filesystem tools regardless of
        # current_node.
        r = classify_path(".datus/memory/gen_sql/MEMORY.md", root_path=project, current_node="gen_sql")
        assert r.zone == PathZone.HIDDEN

    def test_other_node_memory_is_hidden(self, project):
        r = classify_path(".datus/memory/chat/MEMORY.md", root_path=project, current_node="gen_sql")
        assert r.zone == PathZone.HIDDEN

    def test_none_node_memory_is_hidden(self, project):
        r = classify_path(".datus/memory/any/MEMORY.md", root_path=project, current_node=None)
        assert r.zone == PathZone.HIDDEN

    def test_home_skills_whitelist(self, project, fake_home):
        r = classify_path(
            str(fake_home / "skills" / "global" / "SKILL.md"),
            root_path=project,
            current_node="chat",
            datus_home=fake_home,
        )
        assert r.zone == PathZone.WHITELIST
        assert r.display.startswith("~/.datus/skills/")


class TestClassifyMemoryAlwaysHidden:
    """Every ``.datus/memory/**`` path is HIDDEN to filesystem tools — the
    dedicated add_memory/edit_memory tools own the subtree, and read-only
    inheritance reaches a child by inlining the parent's memory into the prompt,
    not via a filesystem read path."""

    def test_memory_file_is_hidden(self, project):
        r = classify_path(".datus/memory/chat/MEMORY.md", root_path=project, current_node="gen_sql")
        assert r.zone == PathZone.HIDDEN
        assert r.read_only is False

    def test_own_memory_is_hidden(self, project):
        r = classify_path(".datus/memory/gen_sql/MEMORY.md", root_path=project, current_node="gen_sql")
        assert r.zone == PathZone.HIDDEN


class TestClassifyExternal:
    def test_relative_escape_goes_external(self, project):
        r = classify_path("../other/secret.txt", root_path=project, current_node="chat")
        assert r.zone == PathZone.EXTERNAL
        assert Path(r.display).is_absolute()

    def test_absolute_outside_root_is_external(self, project, tmp_path):
        elsewhere = tmp_path / "other"
        elsewhere.mkdir()
        target = elsewhere / "x.md"
        r = classify_path(str(target), root_path=project, current_node="chat")
        assert r.zone == PathZone.EXTERNAL


class TestRootUnderHome:
    """If the project happens to live under ``~/.datus`` — e.g. someone runs
    ``datus`` inside ``~/.datus/workspace/demo`` — project anchors must still
    beat the global ``~/.datus/skills`` anchor so file visibility matches the
    user's intent ("writing to my project's skills dir, not the global one").
    """

    def test_project_skills_wins_over_global(self, tmp_path):
        home = tmp_path / ".datus"
        home.mkdir()
        (home / "skills").mkdir()
        project = home / "workspace" / "demo"
        project.mkdir(parents=True)
        r = classify_path(
            ".datus/skills/foo/SKILL.md",
            root_path=project,
            current_node="chat",
            datus_home=home,
        )
        assert r.zone == PathZone.WHITELIST
        assert r.display.startswith(".datus/skills/")


class TestWhitelistAnchors:
    def test_anchor_list_contains_project_and_home(self, project, fake_home):
        anchors = whitelist_anchors(root_path=project, current_node="chat", datus_home=fake_home)
        # Exactly skills (project) + plans (project) + skills (home). Memory is
        # never an anchor — it is HIDDEN to filesystem tools.
        assert len(anchors) == 3
        assert (project / ".datus" / "skills").resolve(strict=False) in anchors
        assert (project / ".datus" / "plans").resolve(strict=False) in anchors
        assert (fake_home / "skills").resolve(strict=False) in anchors

    def test_no_memory_anchor_for_any_node(self, project, fake_home):
        for node in (None, "chat", "gen_sql"):
            anchors = whitelist_anchors(root_path=project, current_node=node, datus_home=fake_home)
            assert len(anchors) == 3
            assert (project / ".datus" / "memory" / "chat").resolve(strict=False) not in anchors
            assert (project / ".datus" / "memory" / "gen_sql").resolve(strict=False) not in anchors


class TestBuildWalkPatterns:
    """The walker relies on these patterns to prune ``HIDDEN`` subtrees cheaply
    — ``wcmatch`` is fed ``excludes`` first and then applies ``re_includes`` so
    the allowed subtrees under ``.datus/`` (skills + plans) stay visible. Memory
    is never re-included; the glob strings are the contract pinned here.
    """

    def test_excludes_prune_entire_dot_datus(self, project):
        excludes, _ = build_walk_patterns(root_path=project, current_node="chat")
        # Both the directory itself and its contents must be excluded,
        # otherwise ``.datus`` survives the first-level match.
        assert excludes == [".datus", ".datus/**"]

    def test_re_includes_are_skills_and_plans_only(self, project):
        # Memory is HIDDEN to filesystem tools, so current_node never adds a
        # memory re-include regardless of its value.
        for node in (None, "chat", "gen_sql"):
            _, re_includes = build_walk_patterns(root_path=project, current_node=node)
            assert re_includes == [".datus/skills/**", ".datus/plans/**"]

    def test_patterns_are_posix_for_wcmatch(self, project):
        """All generated patterns are POSIX slashes; wcmatch does not normalize
        separators, so a Windows-style backslash would break globmatch."""
        excludes, re_includes = build_walk_patterns(root_path=project, current_node="chat")
        for pattern in excludes + re_includes:
            assert "\\" not in pattern


class TestSessionDataAnchor:
    """``session_data_dir`` is the compact-archive read-only anchor.

    Without this anchor LLMs would hit a permission prompt every time they
    tried to ``read_file`` an archived tool output — defeating the whole
    "zero information loss" property of the minor compact pass. These tests
    pin the contract: archived path → WHITELIST + read_only; cross-session
    paths → EXTERNAL even though they share the ``sessions/`` root.
    """

    def test_archived_path_is_whitelist_readonly(self, project, fake_home):
        sdd = fake_home / "sessions" / "proj" / "sid42" / "data"
        sdd.mkdir(parents=True)
        archived = sdd / "000001_args_abc.json"
        archived.write_text("{}")

        r = classify_path(
            str(archived),
            root_path=project,
            current_node="chat",
            datus_home=fake_home,
            session_data_dir=sdd,
        )
        assert r.zone == PathZone.WHITELIST
        # MUST be read-only — the compact pass owns archive contents; LLM
        # writes would corrupt hashes and audit trails.
        assert r.read_only is True
        # Display uses the canonical ``~/.datus/...`` form so the LLM can
        # feed it back unambiguously, just like other whitelist entries.
        assert r.display.startswith("~/.datus/sessions/")

    def test_other_session_data_dir_stays_external(self, project, fake_home):
        sessions_root = fake_home / "sessions" / "proj"
        sdd = sessions_root / "sid42" / "data"
        sdd.mkdir(parents=True)
        other_sdd = sessions_root / "sid99" / "data"
        other_sdd.mkdir(parents=True)
        # Cross-session leak guard: even with the *current* session's anchor
        # registered, another session's data dir must NOT be readable.
        r = classify_path(
            str(other_sdd / "foo.txt"),
            root_path=project,
            current_node="chat",
            datus_home=fake_home,
            session_data_dir=sdd,
        )
        assert r.zone == PathZone.EXTERNAL

    def test_archive_path_without_anchor_is_external(self, project, fake_home):
        sdd = fake_home / "sessions" / "proj" / "sid42" / "data"
        sdd.mkdir(parents=True)
        # Caller did not pass session_data_dir → the archive directory has
        # no whitelist anchor, so the path is EXTERNAL (broker will ASK).
        r = classify_path(
            str(sdd / "x.json"),
            root_path=project,
            current_node="chat",
            datus_home=fake_home,
        )
        assert r.zone == PathZone.EXTERNAL

    def test_whitelist_anchors_includes_session_data_when_provided(self, project, fake_home):
        sdd = fake_home / "sessions" / "proj" / "sid42" / "data"
        sdd.mkdir(parents=True)
        anchors = whitelist_anchors(
            root_path=project,
            current_node="chat",
            datus_home=fake_home,
            session_data_dir=sdd,
        )
        assert sdd.resolve() in anchors

    def test_whitelist_anchors_omits_session_data_when_absent(self, project, fake_home):
        anchors = whitelist_anchors(
            root_path=project,
            current_node="chat",
            datus_home=fake_home,
        )
        # No session anchor expected; the list must contain only the
        # project-side anchors + the global ``skills`` dir.
        for anchor in anchors:
            assert "sessions" not in anchor.parts
