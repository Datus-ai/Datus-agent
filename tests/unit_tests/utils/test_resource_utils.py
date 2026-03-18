# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

from unittest.mock import MagicMock, patch

import pytest

from datus.utils.resource_utils import copy_data_file, do_copy_data_file, package_data_path, read_data_file_text


class TestPackageDataPath:
    def test_returns_path_when_sys_prefix_exists(self, tmp_path):
        resource = "config/test.yml"
        target = tmp_path / "datus" / "config" / "test.yml"
        target.parent.mkdir(parents=True)
        target.write_text("test")
        with patch("sys.prefix", str(tmp_path)):
            result = package_data_path(resource)
        assert result is not None
        assert result == target

    def test_returns_path_when_exec_prefix_exists(self, tmp_path):
        resource = "config/test.yml"
        target = tmp_path / "datus" / "config" / "test.yml"
        target.parent.mkdir(parents=True)
        target.write_text("test")
        non_existent = tmp_path / "nonexistent_prefix"
        with patch("sys.prefix", str(non_existent)), patch("sys.exec_prefix", str(tmp_path)):
            result = package_data_path(resource)
        assert result is not None

    def test_falls_back_to_importlib(self, tmp_path):
        resource = "config/test.yml"
        non_existent = tmp_path / "nonexistent"
        mock_package_path = MagicMock()
        mock_package_path.__truediv__ = lambda self, other: tmp_path / other
        with (
            patch("sys.prefix", str(non_existent)),
            patch("sys.exec_prefix", str(non_existent)),
            patch("importlib.resources.files", return_value=mock_package_path),
        ):
            result = package_data_path(resource)
        assert result is not None


class TestReadDataFileText:
    def test_reads_local_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")
        result = read_data_file_text(str(test_file))
        assert result == "hello world"

    def test_raises_when_file_not_found(self):
        from datus.utils.exceptions import DatusException

        with pytest.raises(DatusException):
            read_data_file_text("nonexistent/path/file.txt", package="datus")

    def test_reads_file_with_custom_encoding(self, tmp_path):
        test_file = tmp_path / "latin.txt"
        test_file.write_bytes("caf\xe9".encode("latin-1"))
        result = read_data_file_text(str(test_file), encoding="latin-1")
        assert "caf" in result


class TestDoCopyDataFile:
    def test_copies_single_file(self, tmp_path):
        src_file = tmp_path / "source.txt"
        src_file.write_text("content")
        target_dir = tmp_path / "dest"

        do_copy_data_file(src_file, target_dir)
        assert (target_dir / "source.txt").exists()
        assert (target_dir / "source.txt").read_text() == "content"

    def test_creates_target_dir_if_not_exists(self, tmp_path):
        src_file = tmp_path / "source.txt"
        src_file.write_text("content")
        target_dir = tmp_path / "new_dir" / "sub_dir"

        do_copy_data_file(src_file, target_dir)
        assert target_dir.exists()
        assert (target_dir / "source.txt").exists()

    def test_does_not_replace_existing_by_default(self, tmp_path):
        src_file = tmp_path / "source.txt"
        src_file.write_text("new content")
        target_dir = tmp_path / "dest"
        target_dir.mkdir()
        existing = target_dir / "source.txt"
        existing.write_text("old content")

        do_copy_data_file(src_file, target_dir, replace=False)
        assert existing.read_text() == "old content"

    def test_replaces_existing_when_replace_true(self, tmp_path):
        src_file = tmp_path / "source.txt"
        src_file.write_text("new content")
        target_dir = tmp_path / "dest"
        target_dir.mkdir()
        existing = target_dir / "source.txt"
        existing.write_text("old content")

        do_copy_data_file(src_file, target_dir, replace=True)
        assert existing.read_text() == "new content"

    def test_copies_directory_recursively(self, tmp_path):
        src_dir = tmp_path / "src_dir"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("file1")
        (src_dir / "file2.txt").write_text("file2")
        target_dir = tmp_path / "dest"

        do_copy_data_file(src_dir, target_dir)
        assert (target_dir / "file1.txt").exists()
        assert (target_dir / "file2.txt").exists()

    def test_copies_nested_directory(self, tmp_path):
        src_dir = tmp_path / "src_dir"
        sub_dir = src_dir / "sub"
        sub_dir.mkdir(parents=True)
        (sub_dir / "nested.txt").write_text("nested")
        target_dir = tmp_path / "dest"

        do_copy_data_file(src_dir, target_dir)
        assert (target_dir / "sub" / "nested.txt").exists()


class TestCopyDataFile:
    def test_copies_existing_file(self, tmp_path):
        src_file = tmp_path / "myfile.txt"
        src_file.write_text("test data")
        target_dir = tmp_path / "output"

        copy_data_file(str(src_file), target_dir)
        assert (target_dir / "myfile.txt").exists()

    def test_no_op_when_source_not_found(self, tmp_path):
        target_dir = tmp_path / "output"
        # Should not raise, just return silently
        copy_data_file("nonexistent_file_xyz.txt", target_dir)
        assert not target_dir.exists() or not any(target_dir.iterdir())

    def test_accepts_path_as_target_dir(self, tmp_path):
        src_file = tmp_path / "myfile.txt"
        src_file.write_text("hello")
        target_dir = tmp_path / "output"

        copy_data_file(str(src_file), target_dir)
        assert (target_dir / "myfile.txt").read_text() == "hello"
