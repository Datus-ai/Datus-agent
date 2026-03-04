# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for skill bundle utilities."""

import tarfile

import pytest

from datus.tools.skill_tools.skill_bundle import (
    calculate_sha256,
    create_bundle,
    extract_bundle,
    extract_bundle_from_bytes,
    list_bundle_contents,
)

SKILL_MD_CONTENT = """---
name: test-skill
description: A test skill
tags: [test]
version: "1.0.0"
---

# Test Skill
"""


@pytest.fixture
def skill_dir(tmp_path):
    """Create a temp skill directory with SKILL.md and scripts/."""
    (tmp_path / "SKILL.md").write_text(SKILL_MD_CONTENT)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.py").write_text("print('hello')")
    (scripts_dir / "setup.sh").write_text("echo setup")
    return tmp_path


@pytest.fixture
def skill_dir_minimal(tmp_path):
    """Create a minimal skill directory with just SKILL.md."""
    (tmp_path / "SKILL.md").write_text(SKILL_MD_CONTENT)
    return tmp_path


class TestCreateBundle:
    def test_creates_tar_gz(self, skill_dir):
        data = create_bundle(skill_dir)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_includes_skill_md(self, skill_dir):
        data = create_bundle(skill_dir)
        import io

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = tar.getnames()
            assert "SKILL.md" in names

    def test_includes_scripts(self, skill_dir):
        data = create_bundle(skill_dir)
        import io

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = tar.getnames()
            assert any("scripts/run.py" in n for n in names)
            assert any("scripts/setup.sh" in n for n in names)

    def test_minimal_skill(self, skill_dir_minimal):
        data = create_bundle(skill_dir_minimal)
        assert isinstance(data, bytes)

    def test_missing_skill_md_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            create_bundle(tmp_path)


class TestExtractBundle:
    def test_extract_to_dir(self, skill_dir, tmp_path):
        data = create_bundle(skill_dir)
        bundle_path = tmp_path / "bundle.tar.gz"
        bundle_path.write_bytes(data)

        dest = tmp_path / "extracted"
        result = extract_bundle(bundle_path, dest)

        assert result == dest
        assert (dest / "SKILL.md").exists()

    def test_extract_from_bytes(self, skill_dir, tmp_path):
        data = create_bundle(skill_dir)
        dest = tmp_path / "extracted"
        result = extract_bundle_from_bytes(data, dest)

        assert result == dest
        assert (dest / "SKILL.md").exists()

    def test_missing_bundle_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_bundle(tmp_path / "nonexistent.tar.gz", tmp_path / "dest")


class TestCalculateSha256:
    def test_consistent_hash(self, skill_dir):
        data = create_bundle(skill_dir)
        h1 = calculate_sha256(data)
        h2 = calculate_sha256(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex length

    def test_different_data_different_hash(self):
        assert calculate_sha256(b"hello") != calculate_sha256(b"world")


class TestListBundleContents:
    def test_list_contents(self, skill_dir, tmp_path):
        data = create_bundle(skill_dir)
        bundle_path = tmp_path / "bundle.tar.gz"
        bundle_path.write_bytes(data)

        names = list_bundle_contents(bundle_path)
        assert "SKILL.md" in names
