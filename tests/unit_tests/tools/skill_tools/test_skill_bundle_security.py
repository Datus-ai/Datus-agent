# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for skill_bundle security filters (_safe_members)."""

import io
import tarfile

from datus.tools.skill_tools.skill_bundle import _safe_members


class TestSafeMembers:
    """Tests for _safe_members tar filter (CVE-2007-4559)."""

    def _make_tar_with_members(self, members_spec):
        """Create a tar archive with specified members.

        members_spec: list of (name, type) where type is 'file', 'symlink', 'hardlink', 'dir'
        """
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, mtype in members_spec:
                info = tarfile.TarInfo(name=name)
                if mtype == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = "/etc/passwd"
                elif mtype == "hardlink":
                    info.type = tarfile.LNKTYPE
                    info.linkname = "/etc/shadow"
                elif mtype == "dir":
                    info.type = tarfile.DIRTYPE
                else:
                    info.type = tarfile.REGTYPE
                    info.size = 0
                tar.addfile(info)
        buf.seek(0)
        return tarfile.open(fileobj=buf, mode="r:gz")

    def test_normal_files_pass(self):
        tar = self._make_tar_with_members([("SKILL.md", "file"), ("scripts/run.py", "file")])
        safe = _safe_members(tar)
        assert len(safe) == 2

    def test_symlinks_rejected(self):
        tar = self._make_tar_with_members([("SKILL.md", "file"), ("evil_link", "symlink")])
        safe = _safe_members(tar)
        assert len(safe) == 1
        assert safe[0].name == "SKILL.md"

    def test_hardlinks_rejected(self):
        tar = self._make_tar_with_members([("SKILL.md", "file"), ("evil_hard", "hardlink")])
        safe = _safe_members(tar)
        assert len(safe) == 1

    def test_absolute_path_rejected(self):
        tar = self._make_tar_with_members([("/etc/passwd", "file")])
        safe = _safe_members(tar)
        assert len(safe) == 0

    def test_path_traversal_rejected(self):
        tar = self._make_tar_with_members([("../../../etc/passwd", "file")])
        safe = _safe_members(tar)
        assert len(safe) == 0

    def test_path_traversal_in_middle_rejected(self):
        tar = self._make_tar_with_members([("scripts/../../../etc/passwd", "file")])
        safe = _safe_members(tar)
        assert len(safe) == 0

    def test_directories_pass(self):
        tar = self._make_tar_with_members([("scripts", "dir"), ("scripts/run.py", "file")])
        safe = _safe_members(tar)
        assert len(safe) == 2
