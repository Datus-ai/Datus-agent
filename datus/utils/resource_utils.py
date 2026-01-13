# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

from datus.utils.exceptions import DatusException, ErrorCode


def package_data_path(resource_path: str, package: str = "datus") -> Optional[Path]:
    path = Path(sys.prefix) / package / resource_path
    if path.exists():
        return path
    path = Path(sys.exec_prefix) / package / resource_path
    if path.exists():
        return path
    from importlib import resources

    package_path = resources.files(package)
    if not package_path:
        return None

    return package_path / resource_path


def read_data_file(resource_path: str, package: str = "datus") -> bytes:
    with package_data_path(resource_path, package) as path:
        return path.read_bytes()


def read_data_file_text(resource_path: str, package: str = "datus", encoding="utf-8") -> str:
    fs_path = Path(resource_path).expanduser().resolve()
    if fs_path.exists():
        return fs_path.read_text(encoding=encoding)
    pkg_entry = package_data_path(resource_path, package)
    if pkg_entry is None or not pkg_entry.exists():
        raise DatusException(
            code=ErrorCode.COMMON_FILE_NOT_FOUND,
            message=f"Unable to locate resource '{resource_path}' in package '{package}'",
        )
    return pkg_entry.read_text(encoding=encoding)


def copy_data_file(resource_path: str, target_dir: Union[str, Path], package: str = "datus", replace: bool = False):
    """
    Copy a data file to target directory.
    Args:
        resource_path: Path to the data file or package file.
        target_dir: Path to the directory to copy to.
        package: Name of the package file.
    """
    # Use path directly
    src_path = Path(resource_path).resolve()
    if src_path.exists():
        src_candidate = src_path
    else:
        src_candidate = package_data_path(resource_path, package)
        if src_candidate is None or not src_candidate.exists():
            return
    target_dir_path = (target_dir if isinstance(target_dir, Path) else Path(target_dir)).expanduser()
    do_copy_data_file(src_candidate, target_dir_path, replace=replace)


def do_copy_data_file(src_path: Path, target_dir: Path, replace: bool = False):
    if not target_dir.exists():
        target_dir.mkdir(parents=True)

    if src_path.is_dir():
        for f in src_path.iterdir():
            if f.is_file():
                target_file = target_dir / f.name
                if replace or not target_file.exists():
                    shutil.copy(f, target_file)
            elif f.is_dir():
                do_copy_data_file(f, target_dir=target_dir / f.name, replace=replace)
    else:
        target_file = target_dir / src_path.name
        if replace or not target_file.exists():
            shutil.copy(src_path, target_file)
