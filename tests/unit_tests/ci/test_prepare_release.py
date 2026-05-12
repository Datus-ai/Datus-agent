from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from packaging.version import Version

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "prepare_release.py"


@pytest.fixture()
def prepare_release(monkeypatch):
    module_name = "_test_prepare_release"
    module_spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if module_spec is None or module_spec.loader is None:
        raise AssertionError(f"Unable to load prepare_release from {MODULE_PATH}")
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    module_spec.loader.exec_module(module)
    return module


def _write_release_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path
    (repo_root / "datus").mkdir()
    (repo_root / "datus" / "__init__.py").write_text(
        dedent(
            '''
            """Datus"""

            __version__ = "0.2.6"
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "datus-agent"
            version = "0.2.6"
            dependencies = [
                "datus-db-core>=0.1.3",
                "datus-semantic-core>=0.2.0",
                "datus-bi-core>=0.1.2",
                "datus-scheduler-core>=0.1.1",
            ]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "requirements.txt").write_text(
        dedent(
            """
            datus-db-core>=0.1.3
            datus-semantic-core>=0.2.0
            datus-bi-core>=0.1.2
            datus-scheduler-core>=0.1.1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return repo_root


def test_parse_canonical_version_rejects_normalized_alias(prepare_release):
    with pytest.raises(ValueError, match="Use canonical PEP 440 version"):
        prepare_release.parse_canonical_version("0.3.0-rc.1")


def test_ensure_version_can_advance_rejects_same_or_lower_version(tmp_path, prepare_release):
    repo_root = _write_release_repo(tmp_path)

    with pytest.raises(ValueError, match="must advance current version"):
        prepare_release.ensure_version_can_advance(repo_root, Version("0.2.6"))


def test_prepare_release_updates_version_and_adapter_bounds(tmp_path, monkeypatch, prepare_release):
    repo_root = _write_release_repo(tmp_path)
    monkeypatch.setattr(
        prepare_release,
        "latest_adapter_bounds",
        lambda timeout, allow_prerelease: {
            "datus-db-core": Version("0.1.4"),
            "datus-semantic-core": Version("0.2.1"),
            "datus-bi-core": Version("0.1.3"),
            "datus-scheduler-core": Version("0.1.2"),
        },
    )

    changed = prepare_release.prepare_release(
        repo_root,
        Version("0.2.7"),
        update_adapter_bounds=True,
    )

    assert {path.relative_to(repo_root).as_posix() for path in changed} == {
        "datus/__init__.py",
        "pyproject.toml",
        "requirements.txt",
    }
    assert 'version = "0.2.7"' in (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '__version__ = "0.2.7"' in (repo_root / "datus" / "__init__.py").read_text(encoding="utf-8")
    assert '"datus-db-core>=0.1.4"' in (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "datus-semantic-core>=0.2.1" in (repo_root / "requirements.txt").read_text(encoding="utf-8")


def test_prepare_release_can_leave_adapter_bounds_unchanged(tmp_path, monkeypatch, prepare_release):
    repo_root = _write_release_repo(tmp_path)

    def fail_latest_adapter_bounds(*_args, **_kwargs):
        raise AssertionError("should not fetch PyPI")

    monkeypatch.setattr(
        prepare_release,
        "latest_adapter_bounds",
        fail_latest_adapter_bounds,
    )

    changed = prepare_release.prepare_release(
        repo_root,
        Version("0.2.7"),
        update_adapter_bounds=False,
    )

    assert {path.relative_to(repo_root).as_posix() for path in changed} == {
        "datus/__init__.py",
        "pyproject.toml",
    }
    assert '"datus-db-core>=0.1.3"' in (repo_root / "pyproject.toml").read_text(encoding="utf-8")
