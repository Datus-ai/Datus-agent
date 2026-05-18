from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import tomllib
from pathlib import Path

INIT_PATH = Path(__file__).resolve().parents[2] / "datus" / "__init__.py"
PYPROJECT_PATH = INIT_PATH.parents[1] / "pyproject.toml"


def _load_datus_init(init_path: Path = INIT_PATH):
    module_spec = importlib.util.spec_from_file_location("_test_datus_init", init_path)
    if module_spec is None or module_spec.loader is None:
        raise AssertionError(f"Unable to load datus package init from {init_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _project_version() -> str:
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    return pyproject["project"]["version"]


def _copied_init_without_pyproject(tmp_path: Path) -> Path:
    package_dir = tmp_path / "site-packages" / "datus"
    package_dir.mkdir(parents=True)
    init_path = package_dir / "__init__.py"
    init_path.write_text(INIT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return init_path


def _patch_pyproject_read(monkeypatch, value: str | Exception) -> None:
    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args, **kwargs) -> str:
        if self.name == "pyproject.toml":
            if isinstance(value, Exception):
                raise value
            return value
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)


def test_source_checkout_version_comes_from_pyproject(monkeypatch):
    requested_names: list[str] = []

    def fake_version(distribution_name: str) -> str:
        requested_names.append(distribution_name)
        return "0.2.6"

    monkeypatch.setattr(metadata, "version", fake_version)

    module = _load_datus_init()

    assert module.__version__ == _project_version()
    assert requested_names == []


def test_source_checkout_falls_back_to_distribution_when_pyproject_is_unavailable(monkeypatch):
    requested_names: list[str] = []

    def fake_version(distribution_name: str) -> str:
        requested_names.append(distribution_name)
        return "1.2.3"

    _patch_pyproject_read(monkeypatch, OSError("missing pyproject"))
    monkeypatch.setattr(metadata, "version", fake_version)

    module = _load_datus_init()

    assert module.__version__ == "1.2.3"
    assert requested_names == ["datus-agent"]


def test_source_checkout_ignores_unrelated_pyproject(monkeypatch):
    requested_names: list[str] = []

    def fake_version(distribution_name: str) -> str:
        requested_names.append(distribution_name)
        return "1.2.3"

    _patch_pyproject_read(monkeypatch, '[project]\nname = "other-package"\nversion = "9.9.9"\n')
    monkeypatch.setattr(metadata, "version", fake_version)

    module = _load_datus_init()

    assert module.__version__ == "1.2.3"
    assert requested_names == ["datus-agent"]


def test_installed_package_version_comes_from_distribution_metadata(monkeypatch, tmp_path):
    init_path = _copied_init_without_pyproject(tmp_path)
    requested_names: list[str] = []

    def fake_version(distribution_name: str) -> str:
        requested_names.append(distribution_name)
        return "1.2.3"

    monkeypatch.setattr(metadata, "version", fake_version)

    module = _load_datus_init(init_path)

    assert module.__version__ == "1.2.3"
    assert requested_names == ["datus-agent"]


def test_package_version_falls_back_when_distribution_metadata_is_missing(monkeypatch):
    def fake_version(_distribution_name: str) -> str:
        raise metadata.PackageNotFoundError("datus-agent")

    _patch_pyproject_read(monkeypatch, OSError("missing pyproject"))
    monkeypatch.setattr(metadata, "version", fake_version)

    module = _load_datus_init()

    assert module.__version__ == "0+unknown"
