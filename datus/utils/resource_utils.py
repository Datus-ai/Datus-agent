import shutil
import sys
from pathlib import Path
from typing import Optional, Union


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
    with package_data_path(resource_path, package) as path:
        return path.read_text(encoding=encoding)


def copy_data_file(resource_path: str, target_dir: Union[str, Path], package: str = "datus", replace: bool = False):
    """
    Copy a data file to target directory.
    Args:
        resource_path: Path to the data file or package file.
        target_dir: Path to the directory to copy to.
        package: Name of the package file.
    """
    src_path = package_data_path(resource_path, package)
    if not src_path.exists():
        return
    target_dir_path = (target_dir if isinstance(target_dir, Path) else Path(target_dir)).expanduser()
    if not target_dir_path.exists():
        target_dir_path.mkdir(parents=True)
    if src_path.is_dir():
        for f in src_path.iterdir():
            do_copy_data_file(f, target_dir_path, replace=replace)
    else:
        do_copy_data_file(src_path, target_dir_path, replace=replace)


def do_copy_data_file(src_dir: Path, target_dir: Path, replace: bool = False):
    if not target_dir.exists():
        target_dir.mkdir(parents=True)

    if src_dir.is_dir():
        for f in src_dir.iterdir():
            do_copy_data_file(f, target_dir=target_dir / f.name, replace=replace)
    else:
        shutil.copy(src_dir, target_dir / src_dir.name)
