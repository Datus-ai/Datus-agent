"""Directory-descriptor based IO for query artifacts inside one project."""

from __future__ import annotations

import errno
import os
import stat
from contextlib import suppress
from pathlib import Path

MAX_SQL_FILE_BYTES = 16 * 1024 * 1024


def _same_file(left, right):
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _walk_directory(parent_fd, parts, *, create=False):
    current = os.dup(parent_fd)
    try:
        for part in parts:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=current)
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException as exc:
        os.close(current)
        if isinstance(exc, OSError) and exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError("Project directory changed or contains a symbolic link") from None
        raise


class ProjectFiles:
    """Pin the trusted root; use canonical paths only for authorization/ref names.

    Every actual open, mkdir, link and unlink is relative to pinned descriptors,
    so swapping a checked ancestor cannot redirect file IO outside the root.
    """

    def __init__(self, root: Path):
        if os.name != "posix":
            raise NotImplementedError("Secure query file IO requires POSIX directory descriptors")
        self.root = Path(root).resolve(strict=True)
        anchor = os.open(self.root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            self.fd = _walk_directory(anchor, self.root.parts[1:])
        finally:
            os.close(anchor)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.fd is not None:
            descriptor, self.fd = self.fd, None
            os.close(descriptor)

    def verify_current(self):
        """Check the pinned root against its current path from the filesystem anchor."""
        anchor = os.open(self.root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            current = _walk_directory(anchor, self.root.parts[1:])
            try:
                if not _same_file(os.fstat(self.fd), os.fstat(current)):
                    raise ValueError("Project root changed during query execution")
            finally:
                os.close(current)
        finally:
            os.close(anchor)

    def resolve(self, value: str | Path) -> Path:
        candidate = (self.root / value).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("File path must be inside the current project files directory")
        return candidate

    def directory(self, value: str | Path, *, create=False):
        path = self.resolve(value)
        descriptor = _walk_directory(self.fd, path.relative_to(self.root).parts, create=create)
        return ProjectDirectory(self, path, descriptor)

    def read_sql(self, value: str, limit: int):
        path = self.resolve(value)
        with self.directory(path.parent) as directory:
            directory.verify_current()
            with directory.open(path.name, "rb") as source:
                data = source.read(limit + 1)
            directory.verify_current()
        if len(data) > limit:
            raise ValueError("SQL file exceeds the 16 MiB input limit")
        return data.decode("utf-8"), str(path.relative_to(self.root))


class ProjectDirectory:
    def __init__(self, files: ProjectFiles, path: Path, descriptor: int):
        self.files = files
        self.path = path
        self.fd = descriptor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.fd is not None:
            descriptor, self.fd = self.fd, None
            os.close(descriptor)

    @staticmethod
    def _name(name):
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Expected a single project filename")
        return name

    def verify_current(self):
        self.files.verify_current()
        with self.files.directory(self.path) as current:
            if not _same_file(os.fstat(self.fd), os.fstat(current.fd)):
                raise ValueError("Project directory changed during query execution")

    def stat(self, name):
        return os.stat(self._name(name), dir_fd=self.fd, follow_symlinks=False)

    def exists(self, name):
        try:
            self.stat(name)
            return True
        except FileNotFoundError:
            return False

    def open(self, name, mode="rb"):
        name = self._name(name)
        if mode not in ("rb", "xb"):
            raise ValueError("Project files support read or exclusive-create mode only")
        flags = os.O_NOFOLLOW | os.O_CLOEXEC
        flags |= (os.O_RDONLY | os.O_NONBLOCK) if mode == "rb" else (os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        descriptor = None
        try:
            descriptor = os.open(name, flags, mode=0o600, dir_fd=self.fd)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Query files must be regular files")
            file = os.fdopen(descriptor, mode)
            descriptor = None
            return file
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ValueError("Project file changed or contains a symbolic link") from None
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def child(self, name):
        name = self._name(name)
        os.mkdir(name, mode=0o700, dir_fd=self.fd)
        try:
            descriptor = _walk_directory(self.fd, (name,))
        except BaseException:
            with suppress(OSError):
                os.rmdir(name, dir_fd=self.fd)
            raise
        return ProjectDirectory(self.files, self.path / name, descriptor)

    def unlink(self, name, *, expected=None):
        try:
            if expected is not None and not _same_file(self.stat(name), expected):
                return
            os.unlink(self._name(name), dir_fd=self.fd)
        except FileNotFoundError:
            pass

    def remove_child(self, name, expected=None):
        try:
            if expected is None or _same_file(self.stat(name), expected):
                os.rmdir(self._name(name), dir_fd=self.fd)
        except FileNotFoundError:
            pass

    def link_from(self, source, source_name, name):
        os.link(
            source._name(source_name),
            self._name(name),
            src_dir_fd=source.fd,
            dst_dir_fd=self.fd,
            follow_symlinks=False,
        )
