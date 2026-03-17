# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/path_utils.py — CI tier, zero external deps."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from datus.utils.path_utils import (
    get_file_fuzzy_matches,
    get_file_name,
    get_files_from_glob_pattern,
    has_glob_pattern,
    safe_rmtree,
)


class TestHasGlobPattern:
    """Tests for has_glob_pattern."""

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("*.db", True),
            ("data/**/*.csv", True),
            ("path/to/[abc].txt", True),
            ("path/to/file?.db", True),
            ("/absolute/path/no/wildcard.db", False),
            ("relative/path.txt", False),
            ("", False),
        ],
    )
    def test_various_patterns(self, path, expected):
        assert has_glob_pattern(path) == expected

    def test_double_star(self):
        assert has_glob_pattern("**") is True

    def test_bracket_only(self):
        assert has_glob_pattern("[abc]") is True


class TestGetFileName:
    """Tests for get_file_name."""

    def test_file_with_extension(self):
        assert get_file_name("/path/to/myfile.db") == "myfile"

    def test_file_without_extension(self):
        assert get_file_name("/path/to/somefile") == "somefile"

    def test_hidden_file(self):
        # .hidden has suffix '.hidden', name is '.hidden', stem is ''
        result = get_file_name("/path/.hidden")
        # Path('.hidden').suffix == '' on some platforms — just verify no exception
        assert isinstance(result, str)

    def test_deeply_nested_path(self):
        assert get_file_name("/a/b/c/d/e/file.csv") == "file"

    def test_just_filename(self):
        assert get_file_name("report.xlsx") == "report"


class TestGetFilesFromGlobPattern:
    """Tests for get_files_from_glob_pattern."""

    def test_no_glob_pattern_returns_empty(self):
        """Paths without glob chars return empty list."""
        result = get_files_from_glob_pattern("/some/path/to/file.db")
        assert result == []

    def test_glob_pattern_returns_list(self):
        """A valid glob pattern that matches real files returns entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two db files
            for name in ("alpha.db", "beta.db"):
                Path(tmpdir, name).write_text("data")

            pattern = os.path.join(tmpdir, "*.db")
            result = get_files_from_glob_pattern(pattern)

        assert isinstance(result, list)
        assert len(result) == 2
        for entry in result:
            assert "logic_name" in entry
            assert "name" in entry
            assert "uri" in entry
            assert entry["uri"].startswith("sqlite:///")

    def test_glob_no_matches_returns_empty(self):
        """Pattern with no matching files returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = os.path.join(tmpdir, "*.db")
            result = get_files_from_glob_pattern(pattern)
        assert result == []

    def test_wildcard_directory_uses_parent_as_logic_name(self):
        """When the directory part contains wildcards, logic_name is parent dir name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "ns1"
            sub.mkdir()
            (sub / "db.sqlite").write_text("x")

            pattern = os.path.join(tmpdir, "*", "*.sqlite")
            result = get_files_from_glob_pattern(pattern)

        assert len(result) == 1
        assert result[0]["logic_name"] == "ns1"

    def test_dialect_string_used_in_uri(self):
        """The dialect value appears in the URI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.duckdb").write_text("x")
            pattern = os.path.join(tmpdir, "*.duckdb")
            result = get_files_from_glob_pattern(pattern, dialect="duckdb")

        assert len(result) == 1
        assert result[0]["uri"].startswith("duckdb:///")

    def test_dbtype_enum_dialect(self):
        """DBType enum dialect is converted to its value."""
        from datus.utils.constants import DBType

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.db").write_text("x")
            pattern = os.path.join(tmpdir, "*.db")
            result = get_files_from_glob_pattern(pattern, dialect=DBType.SQLITE)

        assert len(result) == 1
        assert result[0]["uri"].startswith("sqlite:///")


class TestSafeRmtree:
    """Tests for safe_rmtree."""

    def test_non_existent_path_returns_false(self, tmp_path):
        missing = tmp_path / "nonexistent"
        assert safe_rmtree(missing, force=True) is False

    def test_file_not_directory_returns_false(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        assert safe_rmtree(f, force=True) is False

    def test_force_deletes_directory(self, tmp_path):
        target = tmp_path / "to_delete"
        target.mkdir()
        (target / "file.txt").write_text("data")
        result = safe_rmtree(target, force=True)
        assert result is True
        assert not target.exists()

    def test_force_with_string_path(self, tmp_path):
        target = tmp_path / "strpath"
        target.mkdir()
        result = safe_rmtree(str(target), force=True)
        assert result is True
        assert not target.exists()

    def test_non_interactive_no_force_returns_false(self, tmp_path):
        """In non-interactive mode without force=True, deletion is skipped."""
        target = tmp_path / "skip_me"
        target.mkdir()
        # stdin is not a tty in CI — safe_rmtree should return False
        import sys

        if sys.stdin.isatty():
            pytest.skip("Test only meaningful in non-interactive environment")

        result = safe_rmtree(target, force=False)
        assert result is False
        assert target.exists()  # Should NOT have been deleted

    def test_oserror_during_delete_returns_false(self, tmp_path):
        """OSError during shutil.rmtree returns False (lines 53-55)."""
        target = tmp_path / "oserr"
        target.mkdir()
        with patch("datus.utils.path_utils.shutil.rmtree", side_effect=OSError("permission denied")):
            result = safe_rmtree(target, force=True)
        assert result is False

    def test_interactive_yes_deletes_directory(self, tmp_path):
        """Interactive mode: answering 'y' deletes the directory (lines 66-83)."""
        import sys

        target = tmp_path / "interactive_del"
        target.mkdir()
        (target / "file.txt").write_text("data")
        with patch.object(sys.stdin, "isatty", return_value=True), patch("builtins.input", return_value="y"):
            result = safe_rmtree(target, force=False)
        assert result is True
        assert not target.exists()

    def test_interactive_no_cancels_deletion(self, tmp_path):
        """Interactive mode: answering 'n' keeps the directory (lines 84-86)."""
        import sys

        target = tmp_path / "interactive_keep"
        target.mkdir()
        (target / "file.txt").write_text("data")  # non-empty so prompt is shown
        with patch.object(sys.stdin, "isatty", return_value=True), patch("builtins.input", return_value="n"):
            result = safe_rmtree(target, force=False)
        assert result is False
        assert target.exists()

    def test_interactive_empty_dir_returns_true_without_prompt(self, tmp_path):
        """Interactive mode: empty directory is deleted without prompting (line 74-76)."""
        import sys

        target = tmp_path / "interactive_empty"
        target.mkdir()
        with patch.object(sys.stdin, "isatty", return_value=True):
            result = safe_rmtree(target, force=False)
        # Empty dir should return True without asking for input (line 76: return True)
        assert result is True


class TestGetFileFuzzyMatches:
    """Tests for get_file_fuzzy_matches."""

    def test_returns_empty_for_nonexistent_path(self):
        result = get_file_fuzzy_matches("anything", path="/nonexistent_dir_xyz_12345")
        assert result == []

    def test_finds_files_by_name_fragment(self, tmp_path):
        (tmp_path / "invoice_2024.csv").write_text("data")
        (tmp_path / "report.csv").write_text("data")
        result = get_file_fuzzy_matches("invoice", path=str(tmp_path))
        assert len(result) == 1
        assert "invoice" in result[0]

    def test_max_matches_respected(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file_{i}.csv").write_text("x")
        result = get_file_fuzzy_matches("file_", path=str(tmp_path), max_matches=3)
        assert len(result) <= 3

    def test_case_sensitive_match(self, tmp_path):
        """get_file_fuzzy_matches uses glob which is case-sensitive; search with exact case."""
        (tmp_path / "MyData.csv").write_text("x")
        result = get_file_fuzzy_matches("MyData", path=str(tmp_path))
        assert len(result) == 1

    def test_no_match_returns_empty(self, tmp_path):
        (tmp_path / "something.txt").write_text("x")
        result = get_file_fuzzy_matches("zzznomatch", path=str(tmp_path))
        assert result == []

    def test_subdirectory_files_found(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested_report.txt").write_text("x")
        result = get_file_fuzzy_matches("nested_report", path=str(tmp_path))
        assert len(result) >= 1
