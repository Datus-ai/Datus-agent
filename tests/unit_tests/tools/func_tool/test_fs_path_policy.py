# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/fs_path_policy.py"""

from pathlib import Path

import pytest

from datus.tools.func_tool.fs_path_policy import (
    PathAllowlist,
    PathZone,
    build_walk_patterns,
    classify_path,
    strict_mode_rejection_message,
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


class TestPathAllowlistParsing:
    """``agent.filesystem.allow_read``/``allow_write`` → anchors."""

    def test_from_dict_parses_both_lists(self, tmp_path):
        allowlist = PathAllowlist.from_dict(
            {"allow_read": [str(tmp_path / "ro")], "allow_write": [str(tmp_path / "rw")]}
        )
        assert allowlist.read == ((tmp_path / "ro").resolve(),)
        assert allowlist.write == ((tmp_path / "rw").resolve(),)
        assert bool(allowlist) is True
        # write-first ordering mirrors classify_path precedence
        assert allowlist.anchors() == [(tmp_path / "rw").resolve(), (tmp_path / "ro").resolve()]

    def test_missing_section_is_empty(self):
        for raw in (None, {}, {"strict": True}, "not-a-dict", []):
            allowlist = PathAllowlist.from_dict(raw)
            assert allowlist.read == ()
            assert allowlist.write == ()
            assert bool(allowlist) is False

    def test_single_string_accepted(self, tmp_path):
        allowlist = PathAllowlist.from_dict({"allow_write": str(tmp_path / "dags")})
        assert allowlist.write == ((tmp_path / "dags").resolve(),)

    def test_relative_and_blank_entries_dropped(self, tmp_path):
        # A relative entry would resolve against the process CWD — silently
        # granting an unrelated directory. Must be ignored, not resolved.
        allowlist = PathAllowlist.from_dict(
            {"allow_write": ["relative/dags", "  ", None, 42, str(tmp_path / "ok")]},
        )
        assert allowlist.write == ((tmp_path / "ok").resolve(),)

    def test_duplicates_collapsed(self, tmp_path):
        target = str(tmp_path / "dags")
        allowlist = PathAllowlist.from_dict({"allow_write": [target, target + "/", target]})
        assert allowlist.write == ((tmp_path / "dags").resolve(),)

    def test_nonexistent_root_still_anchored(self, tmp_path):
        # The DAGs folder may be created later by the scheduler pod; anchoring
        # must not depend on it existing at config-load time.
        missing = tmp_path / "not-created-yet"
        allowlist = PathAllowlist.from_dict({"allow_write": [str(missing)]})
        assert allowlist.write == (missing.resolve(),)


class TestClassifyWithAllowlist:
    """Configured roots outside the project turn EXTERNAL into WHITELIST.

    This is what lets a strict-mode deployment (no interactive broker) write
    generated DAG files into a folder mounted next to the workspace.
    """

    @pytest.fixture
    def dags(self, tmp_path):
        folder = tmp_path / "services" / "airflow" / "dags" / "ws1" / "proj1"
        folder.mkdir(parents=True)
        return folder

    def test_allow_write_root_is_writable_whitelist(self, project, dags):
        r = classify_path(
            str(dags / "daily_job.py"),
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_write": [str(dags)]}),
        )
        assert r.zone == PathZone.WHITELIST
        assert r.read_only is False
        # Absolute display: nothing project-relative would round-trip here.
        assert r.display == str(dags / "daily_job.py")

    def test_allow_read_root_is_read_only_whitelist(self, project, dags):
        r = classify_path(
            str(dags / "daily_job.py"),
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_read": [str(dags)]}),
        )
        assert r.zone == PathZone.WHITELIST
        assert r.read_only is True

    def test_write_anchor_wins_over_read_anchor(self, project, dags):
        r = classify_path(
            str(dags / "daily_job.py"),
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_read": [str(dags)], "allow_write": [str(dags)]}),
        )
        assert r.read_only is False

    def test_sibling_of_allowed_root_stays_external(self, project, dags):
        # Prefix-only match must not leak: ``.../proj1_backup`` is a sibling,
        # not a descendant, of the allowed ``.../proj1``.
        sibling = dags.parent / "proj1_backup" / "x.py"
        r = classify_path(
            str(sibling),
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_write": [str(dags)]}),
        )
        assert r.zone == PathZone.EXTERNAL

    def test_traversal_out_of_allowed_root_stays_external(self, project, dags):
        r = classify_path(
            str(dags / ".." / ".." / "secrets.env"),
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_write": [str(dags)]}),
        )
        assert r.zone == PathZone.EXTERNAL

    def test_allowlist_never_unhides_project_internals(self, project):
        # Allowlisting an ancestor of the project must not expose ``.datus``:
        # project-side zones are decided before the allowlist is consulted.
        r = classify_path(
            ".datus/sessions/foo.db",
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_write": [str(project.parent)]}),
        )
        assert r.zone == PathZone.HIDDEN

    def test_project_paths_stay_internal_under_allowlisted_ancestor(self, project):
        r = classify_path(
            "src/main.py",
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_write": [str(project.parent)]}),
        )
        assert r.zone == PathZone.INTERNAL
        assert r.display == "src/main.py"

    def test_without_allowlist_dags_path_is_external(self, project, dags):
        r = classify_path(str(dags / "daily_job.py"), root_path=project)
        assert r.zone == PathZone.EXTERNAL

    def test_whitelist_anchors_include_allowlist_roots(self, project, fake_home, dags):
        anchors = whitelist_anchors(
            root_path=project,
            current_node="chat",
            datus_home=fake_home,
            allowlist=PathAllowlist.from_dict({"allow_write": [str(dags)]}),
        )
        assert dags.resolve() in anchors
        # Project-side anchors are still first so longer prefixes keep winning.
        assert anchors[0] == (project / ".datus" / "skills").resolve(strict=False)


class TestStrictModeRejectionMessage:
    """The rejection text is read by the LLM, so its shape is a contract."""

    @pytest.fixture
    def dags(self, tmp_path):
        folder = tmp_path / "services" / "airflow" / "dags" / "ws1" / "proj1"
        folder.mkdir(parents=True)
        return folder

    def test_prefix_and_path_are_preserved(self, project):
        msg = strict_mode_rejection_message("/outside/x.py", root_path=project)
        assert msg.startswith("Path outside workspace is not allowed in strict mode: /outside/x.py")
        assert str(project.resolve()) in msg

    def test_lists_allowlisted_write_roots(self, project, dags):
        msg = strict_mode_rejection_message(
            "/outside/x.py",
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_write": [str(dags)]}),
        )
        assert f"allowed roots: {project.resolve()}, {dags.resolve()}" in msg
        assert "read-only" not in msg

    def test_read_roots_are_reported_separately(self, project, tmp_path):
        shared = tmp_path / "shared"
        shared.mkdir()
        msg = strict_mode_rejection_message(
            "/outside/x.py",
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_read": [str(shared)]}),
        )
        assert f"allowed roots: {project.resolve()}" in msg
        assert f"read-only roots: {shared.resolve()}" in msg

    def test_anchors_under_project_root_are_not_repeated(self, project):
        inside = project / "dags"
        msg = strict_mode_rejection_message(
            "/outside/x.py",
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_write": [str(inside)], "allow_read": [str(inside)]}),
        )
        assert msg.count(str(project.resolve())) == 1
        assert "read-only" not in msg

    def test_long_allowlist_is_truncated(self, project, tmp_path):
        roots = [tmp_path / f"root{i}" for i in range(12)]
        msg = strict_mode_rejection_message(
            "/outside/x.py",
            root_path=project,
            allowlist=PathAllowlist.from_dict({"allow_write": [str(r) for r in roots]}),
        )
        # project root + 7 anchors listed, the rest collapsed into a counter.
        assert str(roots[6].resolve()) in msg
        assert str(roots[7].resolve()) not in msg
        assert "... (+5 more)" in msg
