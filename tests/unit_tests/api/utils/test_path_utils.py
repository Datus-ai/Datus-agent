"""Tests for datus.api.utils.path_utils — path traversal prevention."""

from pathlib import Path

import pytest

from datus.api.utils.path_utils import safe_resolve
from datus.utils.exceptions import DatusException


class TestSafeResolveBehavior:
    """Core contract: resolves valid relative paths, blocks escaping."""

    def test_simple_relative_path_resolves_correctly(self, tmp_path):
        """Valid relative path resolves under base."""
        result = safe_resolve(tmp_path, "subdir/file.txt")
        assert result == (tmp_path / "subdir" / "file.txt").resolve()

    def test_plain_filename_resolves_under_base(self, tmp_path):
        """Single filename resolves directly under base."""
        result = safe_resolve(tmp_path, "file.txt")
        assert result == (tmp_path / "file.txt").resolve()

    def test_nested_path_resolves_correctly(self, tmp_path):
        """Multi-level nested path resolves under base."""
        result = safe_resolve(tmp_path, "a/b/c/d.yml")
        assert result == (tmp_path / "a" / "b" / "c" / "d.yml").resolve()

    def test_dotslash_prefix_resolves_under_base(self, tmp_path):
        """Path with ./ prefix resolves under base."""
        result = safe_resolve(tmp_path, "./file.txt")
        assert result == (tmp_path / "file.txt").resolve()


class TestSafeResolveTraversalDetection:
    """Paths that escape base must raise DatusException."""

    def test_double_dot_escapes_raises(self, tmp_path):
        """Path traversal with .. raises DatusException."""
        with pytest.raises(DatusException, match="escapes the project root"):
            safe_resolve(tmp_path, "../../etc/passwd")

    def test_absolute_path_outside_base_raises(self, tmp_path):
        """Absolute path outside base raises DatusException."""
        with pytest.raises(DatusException, match="escapes the project root"):
            safe_resolve(tmp_path, "/etc/passwd")

    def test_dot_dot_at_start_raises(self, tmp_path):
        """Single parent traversal raises DatusException."""
        with pytest.raises(DatusException, match="escapes the project root"):
            safe_resolve(tmp_path, "../sibling/file.txt")


class TestSafeResolveEdgeCases:
    """Boundary conditions for path resolution."""

    def test_empty_string_resolves_to_base(self, tmp_path):
        """Empty path resolves to the base directory itself."""
        result = safe_resolve(tmp_path, "")
        assert result == tmp_path.resolve()

    def test_dot_resolves_to_base(self, tmp_path):
        """Single dot resolves to base directory."""
        result = safe_resolve(tmp_path, ".")
        assert result == tmp_path.resolve()

    def test_return_type_is_path(self, tmp_path):
        """Return value is always a Path instance."""
        result = safe_resolve(tmp_path, "file.txt")
        assert isinstance(result, Path)
