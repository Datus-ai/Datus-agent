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

            from importlib import metadata as importlib_metadata

            try:
                __version__ = importlib_metadata.version("datus-agent")
            except importlib_metadata.PackageNotFoundError:
                __version__ = "0+unknown"
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

            [dependency-groups]
            ci = [
                "datus-metricflow>=0.2.6",
                "datus-semantic-metricflow>=0.2.7",
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


def test_ensure_version_can_advance_can_allow_current_version(tmp_path, prepare_release):
    repo_root = _write_release_repo(tmp_path)

    assert prepare_release.ensure_version_can_advance(repo_root, Version("0.2.6"), allow_current_version=True) is None
    with pytest.raises(ValueError, match="must advance current version"):
        prepare_release.ensure_version_can_advance(repo_root, Version("0.2.5"), allow_current_version=True)


def test_ensure_version_can_allow_same_final_prerelease(tmp_path, prepare_release):
    repo_root = _write_release_repo(tmp_path)

    with pytest.raises(ValueError, match="must advance current version"):
        prepare_release.ensure_version_can_advance(repo_root, Version("0.2.6rc1"))

    assert (
        prepare_release.ensure_version_can_advance(
            repo_root,
            Version("0.2.6rc1"),
            allow_same_final_prerelease=True,
        )
        is None
    )


def test_ensure_version_can_allow_same_final_prerelease_rejects_finalized_tag(tmp_path, prepare_release):
    repo_root = _write_release_repo(tmp_path)
    subprocess.run(["git", "tag", "v0.2.6"], cwd=repo_root, check=True)

    with pytest.raises(ValueError, match="Cannot prepare prerelease 0.2.6rc1 from finalized 0.2.6"):
        prepare_release.ensure_version_can_advance(
            repo_root,
            Version("0.2.6rc1"),
            allow_same_final_prerelease=True,
        )


def test_ensure_version_can_allow_same_final_prerelease_rejects_different_line(tmp_path, prepare_release):
    repo_root = _write_release_repo(tmp_path)

    with pytest.raises(ValueError, match="must advance current version"):
        prepare_release.ensure_version_can_advance(
            repo_root,
            Version("0.2.5rc1"),
            allow_same_final_prerelease=True,
        )


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
    monkeypatch.setattr(
        prepare_release,
        "latest_ci_adapter_bounds",
        lambda timeout, allow_prerelease: {
            "datus-metricflow": Version("0.2.7"),
            "datus-semantic-metricflow": Version("0.2.8"),
        },
    )

    changed = prepare_release.prepare_release(
        repo_root,
        Version("0.2.7"),
        update_adapter_bounds=True,
    )

    assert {path.relative_to(repo_root).as_posix() for path in changed} == {
        "pyproject.toml",
        "requirements.txt",
    }
    assert 'version = "0.2.7"' in (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version("datus-agent")' in (repo_root / "datus" / "__init__.py").read_text(encoding="utf-8")
    assert '"datus-db-core>=0.1.4"' in (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"datus-metricflow>=0.2.7"' in (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "datus-semantic-core>=0.2.1" in (repo_root / "requirements.txt").read_text(encoding="utf-8")


def test_update_dependency_group_lower_bounds_preserves_formatting_and_requirement_metadata(tmp_path, prepare_release):
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        dedent(
            """
            [dependency-groups]
            ci = [
                "Datus_MetricFlow[snowflake]>=0.2.6,<0.3; python_version >= '3.12'",
                "datus-semantic-metricflow>=0.2.7",
            ]
            """
        ).lstrip(),
        encoding="utf-8",
    )

    changed = prepare_release.update_dependency_group_lower_bounds(
        pyproject_path,
        "ci",
        {
            "datus-metricflow": Version("0.2.7"),
            "datus-semantic-metricflow": Version("0.2.8"),
        },
    )

    assert changed is True
    assert (
        pyproject_path.read_text(encoding="utf-8")
        == dedent(
            """
        [dependency-groups]
        ci = [
            "Datus_MetricFlow[snowflake]>=0.2.7,<0.3; python_version >= '3.12'",
            "datus-semantic-metricflow>=0.2.8",
        ]
        """
        ).lstrip()
    )


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
        "pyproject.toml",
    }
    assert '"datus-db-core>=0.1.3"' in (repo_root / "pyproject.toml").read_text(encoding="utf-8")


def test_main_can_treat_already_prepared_release_as_success(tmp_path, prepare_release):
    repo_root = _write_release_repo(tmp_path)

    result = prepare_release.main(
        [
            "--repo-root",
            str(repo_root),
            "--version",
            "0.2.6",
            "--allow-current-version",
            "--allow-no-changes",
            "--no-update-adapter-bounds",
        ]
    )

    assert result == 0
