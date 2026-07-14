# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.plugins.store` (directory store + sys.path).

CI-level: uses a temp ``~/.datus`` home via a context-local path manager and a
hand-built ``pip install --target`` tree; no subprocess, no network.
"""

import sys

import pytest

from datus.plugins import registry, store
from datus.utils.path_manager import DatusPathManager, reset_path_manager, set_current_path_manager


@pytest.fixture
def home(tmp_path):
    """Point the store at a throwaway ``~/.datus`` and clean up sys.path after."""
    token = set_current_path_manager(DatusPathManager(datus_home=tmp_path))
    before = list(sys.path)
    try:
        yield tmp_path
    finally:
        # Drop any plugin directories this test appended to sys.path.
        sys.path[:] = [p for p in sys.path if p in before or str(tmp_path) not in p]
        registry.invalidate_plugin_cache()
        reset_path_manager(token)


def _write_target(
    directory,
    *,
    name="demo",
    dist="datus-demo-plugin",
    version="0.1.0",
    entry="datus_demo_plugin.plugin:DemoPlugin",
    requires_python=">=3.12",
    group="datus.plugins",
):
    """Write a minimal ``pip install --target`` tree with a plugin dist-info."""
    directory.mkdir(parents=True, exist_ok=True)
    pkg = directory / "datus_demo_plugin"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("__version__ = '0.1.0'\n", encoding="utf-8")
    dist_info = directory / f"{dist.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    if group is not None:
        (dist_info / "entry_points.txt").write_text(f"[{group}]\n{name} = {entry}\n", encoding="utf-8")
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\nRequires-Python: {requires_python}\n",
        encoding="utf-8",
    )
    return directory


# ── name validation ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["demo", "airflow", "my_plugin", "a.b-c", "X9"])
def test_is_valid_name_accepts_safe(name):
    assert store.is_valid_name(name)


@pytest.mark.parametrize("name", ["upgrade", "skill", "plugin"])
def test_reserved_names_rejected(name):
    assert not store.is_valid_name(name)
    with pytest.raises(store.StoreError):
        store.ensure_valid_name(name)


@pytest.mark.parametrize("name", ["", "has space", "-leading", "a/b", "..", None])
def test_unsafe_names_rejected(name):
    assert not store.is_valid_name(name)
    with pytest.raises(store.StoreError):
        store.ensure_valid_name(name)


# ── metadata round-trip ─────────────────────────────────────────────────────


def test_write_and_read_meta(home):
    directory = store.plugin_dir("demo")
    meta = {"name": "demo", "version": "1.0", "install": {"type": "src"}}
    store.write_meta(directory, meta)
    assert store.meta_path(directory).is_file()
    assert store.read_meta(directory) == meta


def test_read_meta_missing_returns_none(home):
    assert store.read_meta(store.plugin_dir("nope")) is None


def test_read_meta_corrupt_returns_none(home):
    directory = store.plugin_dir("demo")
    directory.mkdir(parents=True)
    store.meta_path(directory).write_text("{not json", encoding="utf-8")
    assert store.read_meta(directory) is None


# ── iter_installed ──────────────────────────────────────────────────────────


def test_iter_installed_skips_dirs_without_meta(home):
    store.write_meta(store.plugin_dir("demo"), {"name": "demo", "version": "1.0"})
    (store.plugins_root() / "stray").mkdir(parents=True)  # no metadata → skipped
    installed = store.iter_installed()
    assert [m["name"] for m in installed] == ["demo"]
    assert installed[0]["_dir"] == str(store.plugin_dir("demo"))


def test_iter_installed_empty_when_root_absent(home):
    assert store.iter_installed() == []


# ── introspect_target ───────────────────────────────────────────────────────


def test_introspect_target_reads_identity(home, tmp_path):
    target = _write_target(tmp_path / "target")
    info = store.introspect_target(target)
    assert info == {
        "name": "demo",
        "distribution": "datus-demo-plugin",
        "version": "0.1.0",
        "entry_point": "datus_demo_plugin.plugin:DemoPlugin",
        "requires_python": ">=3.12",
    }


def test_introspect_target_rejects_non_plugin(home, tmp_path):
    target = _write_target(tmp_path / "target", group="console_scripts")
    with pytest.raises(store.StoreError):
        store.introspect_target(target)


# ── activation (sys.path) ───────────────────────────────────────────────────


def test_activate_all_appends_enabled_dirs(home):
    store.write_meta(store.plugin_dir("demo"), {"name": "demo"})
    store.write_meta(store.plugin_dir("other"), {"name": "other"})
    added = store.activate(None)  # None == no filter, activate all
    assert set(added) == {"demo", "other"}
    assert str(store.plugin_dir("demo")) in sys.path
    assert str(store.plugin_dir("other")) in sys.path


def test_activate_filters_by_whitelist(home):
    store.write_meta(store.plugin_dir("demo"), {"name": "demo"})
    store.write_meta(store.plugin_dir("other"), {"name": "other"})
    added = store.activate({"demo"})
    assert added == ["demo"]
    assert str(store.plugin_dir("demo")) in sys.path
    assert str(store.plugin_dir("other")) not in sys.path


def test_activate_is_idempotent(home):
    store.write_meta(store.plugin_dir("demo"), {"name": "demo"})
    assert store.activate(None) == ["demo"]
    assert store.activate(None) == []  # already on sys.path
    assert sys.path.count(str(store.plugin_dir("demo"))) == 1


def test_activate_noop_when_plugins_disabled(home):
    store.write_meta(store.plugin_dir("demo"), {"name": "demo"})
    assert store.activate(None, plugins_enabled=False) == []
    assert str(store.plugin_dir("demo")) not in sys.path


def test_activate_skips_reserved_and_invalid(home):
    store.write_meta(store.plugin_dir("demo"), {"name": "demo"})
    # A directory whose recorded name is reserved must never be injected.
    reserved_dir = store.plugins_root() / "upgrade"
    store.write_meta(reserved_dir, {"name": "upgrade"})
    added = store.activate(None)
    assert added == ["demo"]


def test_activate_name_appends_single_dir(home):
    directory = store.plugin_dir("demo")
    directory.mkdir(parents=True)
    assert store.activate_name("demo") is True
    assert str(directory) in sys.path
    assert store.activate_name("demo") is False  # idempotent


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
