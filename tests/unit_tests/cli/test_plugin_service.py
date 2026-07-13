# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.plugin_service`` (install / uninstall / list).

CI-level: subprocess / shutil / entry-point enumeration are all mocked; no
network, no real package operations.
"""

import hashlib
import json
import sys
import types
import zipfile

import pytest

from datus.cli import plugin_service as svc


class _FakeDist:
    def __init__(self, name, version="1.0.0"):
        self.name = name
        self.version = version


class _FakeEntryPoint:
    def __init__(self, name, package="", version="1.0.0", value="mod:Cls"):
        self.name = name
        self.value = value
        self.dist = _FakeDist(package, version) if package else None


def _fake_proc(returncode=0, stdout="ok", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ── _install_command ──────────────────────────────────────────────────────


def test_install_command_prefers_uv(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/uv")
    cmd, label = svc._install_command("datus-foo")
    assert cmd == ["/usr/bin/uv", "pip", "install", "--python", sys.executable, "datus-foo"]
    assert label == "uv pip install"


def test_install_command_editable_flag(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/uv")
    cmd, _ = svc._install_command("./local", editable=True)
    assert "-e" in cmd and cmd[-1] == "./local"


def test_install_command_pip_fallback(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    cmd, label = svc._install_command("datus-foo")
    assert cmd == [sys.executable, "-m", "pip", "install", "datus-foo"]
    assert label == "pip install"


# ── install ─────────────────────────────────────────────────────────────


def test_install_reports_new_plugins(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/uv")
    state = {"installed": False}

    def fake_run(cmd, **kwargs):
        state["installed"] = True
        return _fake_proc()

    monkeypatch.setattr(svc.subprocess, "run", fake_run)

    def fake_iter():
        return [_FakeEntryPoint("statsig")] if state["installed"] else []

    monkeypatch.setattr("datus.plugins.registry.iter_plugin_entry_points", fake_iter)
    monkeypatch.setattr("datus.plugins.registry.invalidate_plugin_cache", lambda: None)

    result = svc.install("datus-statsig-plugin")
    assert result.ok is True
    assert result.new_plugins == ["statsig"]


def test_install_no_new_plugins(monkeypatch):
    """A package that registers no datus.plugins entry point → empty list."""
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", lambda cmd, **k: _fake_proc())
    monkeypatch.setattr("datus.plugins.registry.iter_plugin_entry_points", lambda: [])
    monkeypatch.setattr("datus.plugins.registry.invalidate_plugin_cache", lambda: None)
    result = svc.install("some-package")
    assert result.ok is True
    assert result.new_plugins == []


def test_install_failure_returncode(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", lambda cmd, **k: _fake_proc(returncode=1, stderr="boom"))
    monkeypatch.setattr("datus.plugins.registry.iter_plugin_entry_points", lambda: [])
    result = svc.install("bad-package")
    assert not result.ok
    assert "code 1" in (result.error or "")
    assert result.stderr == "boom"


def test_install_subprocess_raises(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)

    def boom(cmd, **k):
        raise OSError("uv missing")

    monkeypatch.setattr(svc.subprocess, "run", boom)
    monkeypatch.setattr("datus.plugins.registry.iter_plugin_entry_points", lambda: [])
    result = svc.install("x")
    assert not result.ok
    assert "uv missing" in (result.error or "")


def test_install_empty_source():
    result = svc.install("   ")
    assert not result.ok
    assert "no install source" in (result.error or "")


def test_install_dispatches_bundle_on_extension(monkeypatch):
    """A ``.dplug`` source routes to install_bundle with ``force`` forwarded."""
    captured = {}

    def fake_install_bundle(path, force=False):
        captured["path"] = path
        captured["force"] = force
        return svc.InstallResult(ok=True, source=path, new_plugins=["hello"])

    monkeypatch.setattr(svc, "install_bundle", fake_install_bundle)
    result = svc.install("./hello-1.0-any.dplug", force=True)
    assert result.ok is True
    assert captured == {"path": "./hello-1.0-any.dplug", "force": True}


# ── install_bundle (.dplug offline) ────────────────────────────────────────

_MAIN_WHEEL = "datus_plugin_hello-1.0.0-py3-none-any.whl"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_bundle(path, *, wheels=None, manifest=None):
    """Write a ``.dplug`` zip at ``path``.

    ``wheels`` maps ``filename -> bytes`` (defaults to a main wheel + one dep).
    ``manifest`` overrides the auto-generated (valid) manifest so tests can
    corrupt individual fields. Returns ``str(path)``.
    """
    if wheels is None:
        wheels = {_MAIN_WHEEL: b"MAIN-WHEEL-BYTES", "dep-2.0-py3-none-any.whl": b"DEP-BYTES"}
    if manifest is None:
        manifest = {
            "format": svc.BUNDLE_FORMAT,
            "format_version": svc.BUNDLE_FORMAT_VERSION,
            "plugin": {
                "name": "hello",
                "distribution": "datus-plugin-hello",
                "version": "1.0.0",
                "wheel": _MAIN_WHEEL,
                "entry_point": "datus_plugin_hello.plugin:HelloPlugin",
            },
            "compat": {"requires_python": "", "platform": "any"},
            "wheels": [
                {
                    "file": fn,
                    "sha256": _sha256(data),
                    "role": "plugin" if fn == _MAIN_WHEEL else "dependency",
                }
                for fn, data in wheels.items()
            ],
        }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(svc.BUNDLE_MANIFEST_NAME, json.dumps(manifest))
        for fn, data in wheels.items():
            zf.writestr(f"{svc.BUNDLE_WHEELS_DIR}/{fn}", data)
    return str(path)


def _mock_offline_install(monkeypatch, *, uv=True, proc=None, capture=None):
    """Wire shutil.which, subprocess.run, and the post-install re-scan.

    The entry-point enumeration is stateful: empty before the install
    subprocess runs, ``[hello]`` after — so ``_rescan_plugins`` reports it as
    newly registered (mirroring ``test_install_reports_new_plugins``).
    """
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/uv" if uv else None)
    state = {"installed": False}

    def fake_run(cmd, **kwargs):
        if capture is not None:
            capture["cmd"] = cmd
        state["installed"] = True
        return proc if proc is not None else _fake_proc()

    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "datus.plugins.registry.iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("hello")] if state["installed"] else [],
    )
    monkeypatch.setattr("datus.plugins.registry.invalidate_plugin_cache", lambda: None)


def test_install_bundle_success(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path / "hello.dplug")
    capture = {}
    _mock_offline_install(monkeypatch, uv=True, capture=capture)
    result = svc.install_bundle(bundle)
    assert result.ok is True
    assert result.new_plugins == ["hello"]
    cmd = capture["cmd"]
    assert "--no-index" in cmd and "--find-links" in cmd
    assert cmd[-1].endswith(_MAIN_WHEEL)  # installs the main wheel by path


def test_install_bundle_offline_command_pip_fallback(monkeypatch):
    """Without uv, the offline command uses ``pip install --no-index``."""
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    from pathlib import Path

    cmd, label = svc._bundle_install_command(Path("/tmp/w/main.whl"), Path("/tmp/w"))
    assert cmd[:5] == [sys.executable, "-m", "pip", "install", "--no-index"]
    assert "--find-links" in cmd and cmd[-1].endswith("main.whl")
    assert label == "pip install (offline)"


def test_install_bundle_not_found():
    result = svc.install_bundle("/nonexistent/path/foo.dplug")
    assert not result.ok
    assert "bundle not found" in (result.error or "")


def test_install_bundle_empty_path():
    result = svc.install_bundle("  ")
    assert not result.ok
    assert "no bundle path" in (result.error or "")


def test_install_bundle_missing_manifest(tmp_path):
    path = tmp_path / "nomani.dplug"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("wheels/x.whl", b"x")
    result = svc.install_bundle(str(path))
    assert not result.ok
    assert "no datus-plugin.json" in (result.error or "")


def test_install_bundle_wrong_format(tmp_path):
    bundle = _make_bundle(tmp_path / "bad.dplug", manifest={"format": "something-else", "format_version": 1})
    result = svc.install_bundle(bundle)
    assert not result.ok
    assert "not a datus plugin bundle" in (result.error or "")


def test_install_bundle_unsupported_format_version(tmp_path):
    manifest = {
        "format": svc.BUNDLE_FORMAT,
        "format_version": 999,
        "plugin": {"wheel": _MAIN_WHEEL},
        "wheels": [{"file": _MAIN_WHEEL, "sha256": "x"}],
    }
    bundle = _make_bundle(tmp_path / "future.dplug", manifest=manifest)
    result = svc.install_bundle(bundle)
    assert not result.ok
    assert "unsupported bundle format_version" in (result.error or "")


def test_install_bundle_checksum_mismatch(tmp_path, monkeypatch):
    """A wrong sha256 fails before any install subprocess runs."""
    wheels = {_MAIN_WHEEL: b"REAL-BYTES"}
    manifest = {
        "format": svc.BUNDLE_FORMAT,
        "format_version": svc.BUNDLE_FORMAT_VERSION,
        "plugin": {"name": "hello", "wheel": _MAIN_WHEEL},
        "compat": {"platform": "any"},
        "wheels": [{"file": _MAIN_WHEEL, "sha256": "deadbeef", "role": "plugin"}],
    }
    bundle = _make_bundle(tmp_path / "tampered.dplug", wheels=wheels, manifest=manifest)

    def boom(cmd, **k):
        raise AssertionError("subprocess must not run on a checksum mismatch")

    monkeypatch.setattr(svc.subprocess, "run", boom)
    result = svc.install_bundle(bundle)
    assert not result.ok
    assert "checksum mismatch" in (result.error or "")


def test_install_bundle_unsafe_wheel_name(tmp_path, monkeypatch):
    """A traversal filename in the manifest is rejected (zip-slip guard)."""
    manifest = {
        "format": svc.BUNDLE_FORMAT,
        "format_version": svc.BUNDLE_FORMAT_VERSION,
        "plugin": {"name": "hello", "wheel": _MAIN_WHEEL},
        "compat": {"platform": "any"},
        "wheels": [{"file": "../evil.whl", "sha256": "x", "role": "plugin"}],
    }
    bundle = _make_bundle(tmp_path / "evil.dplug", wheels={_MAIN_WHEEL: b"x"}, manifest=manifest)
    monkeypatch.setattr(svc.subprocess, "run", lambda *a, **k: _fake_proc())
    result = svc.install_bundle(bundle)
    assert not result.ok
    assert "unsafe wheel filename" in (result.error or "")


def test_install_bundle_missing_wheel_in_zip(tmp_path):
    manifest = {
        "format": svc.BUNDLE_FORMAT,
        "format_version": svc.BUNDLE_FORMAT_VERSION,
        "plugin": {"name": "hello", "wheel": _MAIN_WHEEL},
        "compat": {"platform": "any"},
        "wheels": [{"file": "ghost-1.0-py3-none-any.whl", "sha256": "x", "role": "dependency"}],
    }
    bundle = _make_bundle(tmp_path / "ghost.dplug", wheels={_MAIN_WHEEL: b"x"}, manifest=manifest)
    result = svc.install_bundle(bundle)
    assert not result.ok
    assert "missing a wheel listed in the manifest" in (result.error or "")


def test_install_bundle_subprocess_failure(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path / "hello.dplug")
    _mock_offline_install(monkeypatch, uv=False, proc=_fake_proc(returncode=1, stderr="offline boom"))
    result = svc.install_bundle(bundle)
    assert not result.ok
    assert "code 1" in (result.error or "")
    assert result.stderr == "offline boom"


def test_install_bundle_python_gate_blocks_without_force(tmp_path, monkeypatch):
    manifest = {
        "format": svc.BUNDLE_FORMAT,
        "format_version": svc.BUNDLE_FORMAT_VERSION,
        "plugin": {"name": "hello", "wheel": _MAIN_WHEEL},
        "compat": {"requires_python": ">=99", "platform": "any"},
        "wheels": [{"file": _MAIN_WHEEL, "sha256": _sha256(b"x"), "role": "plugin"}],
    }
    bundle = _make_bundle(tmp_path / "pygate.dplug", wheels={_MAIN_WHEEL: b"x"}, manifest=manifest)
    monkeypatch.setattr(svc, "_python_satisfies", lambda spec: False)

    def boom(cmd, **k):
        raise AssertionError("subprocess must not run when the compat gate blocks")

    monkeypatch.setattr(svc.subprocess, "run", boom)
    result = svc.install_bundle(bundle)
    assert not result.ok
    assert "requires Python" in (result.error or "")


def test_install_bundle_force_skips_compat_gate(tmp_path, monkeypatch):
    manifest = {
        "format": svc.BUNDLE_FORMAT,
        "format_version": svc.BUNDLE_FORMAT_VERSION,
        "plugin": {"name": "hello", "wheel": _MAIN_WHEEL},
        "compat": {"requires_python": ">=99", "platform": "any"},
        "wheels": [{"file": _MAIN_WHEEL, "sha256": _sha256(b"x"), "role": "plugin"}],
    }
    bundle = _make_bundle(tmp_path / "forced.dplug", wheels={_MAIN_WHEEL: b"x"}, manifest=manifest)
    monkeypatch.setattr(svc, "_python_satisfies", lambda spec: False)
    _mock_offline_install(monkeypatch, uv=True)
    result = svc.install_bundle(bundle, force=True)
    assert result.ok is True


def test_install_bundle_bad_zip(tmp_path):
    path = tmp_path / "corrupt.dplug"
    path.write_bytes(b"not a zip at all")
    result = svc.install_bundle(str(path))
    assert not result.ok
    assert "invalid bundle" in (result.error or "")


# ── uninstall ─────────────────────────────────────────────────────────────


def test_uninstall_maps_plugin_to_distribution(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        "datus.plugins.registry.entry_points_for_group",
        lambda group, name=None: [_FakeEntryPoint("statsig", package="datus-statsig-plugin")],
    )
    monkeypatch.setattr("datus.plugins.registry.invalidate_plugin_cache", lambda: None)
    captured = {}

    def fake_run(cmd, **k):
        captured["cmd"] = cmd
        return _fake_proc()

    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    result = svc.uninstall("statsig")
    assert result.ok is True
    assert result.package == "datus-statsig-plugin"
    assert "datus-statsig-plugin" in captured["cmd"]


def test_uninstall_unknown_plugin(monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.entry_points_for_group", lambda group, name=None: [])
    result = svc.uninstall("mystery")
    assert not result.ok
    assert "no installed plugin" in (result.error or "")


def test_uninstall_failure_returncode(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        "datus.plugins.registry.entry_points_for_group",
        lambda group, name=None: [_FakeEntryPoint("statsig", package="datus-statsig-plugin")],
    )
    monkeypatch.setattr(svc.subprocess, "run", lambda cmd, **k: _fake_proc(returncode=1, stderr="nope"))
    result = svc.uninstall("statsig")
    assert not result.ok
    assert "code 1" in (result.error or "")


# ── list_plugins ────────────────────────────────────────────────────────


def test_list_plugins_without_config(monkeypatch):
    monkeypatch.setattr(
        "datus.plugins.registry.iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("statsig", package="datus-statsig-plugin", version="0.1.0")],
    )
    plugins = svc.list_plugins(agent_config=None)
    assert len(plugins) == 1
    info = plugins[0]
    assert info.name == "statsig"
    assert info.package == "datus-statsig-plugin"
    assert info.version == "0.1.0"
    assert info.active is None  # unknown without a config
    assert info.profiles == []


def test_list_plugins_with_config(monkeypatch):
    monkeypatch.setattr(
        "datus.plugins.registry.iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("statsig", package="datus-statsig-plugin")],
    )

    class _Cfg:
        plugin_services = {"statsig": {"dev": {}, "prod": {}}}

        def active_plugin_names(self):
            return {"statsig"}

        def active_plugin_profiles(self, name):
            return ["prod"]

    plugins = svc.list_plugins(_Cfg())
    info = plugins[0]
    assert info.profiles == ["dev", "prod"]
    assert info.active is True
    assert info.active_profiles == ["prod"]


def test_list_plugins_inactive_when_not_in_whitelist(monkeypatch):
    monkeypatch.setattr(
        "datus.plugins.registry.iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("statsig", package="p")],
    )

    class _Cfg:
        plugin_services = {}

        def active_plugin_names(self):
            return {"other"}  # statsig not listed

        def active_plugin_profiles(self, name):
            return []

    assert svc.list_plugins(_Cfg())[0].active is False


def test_list_plugins_sorted(monkeypatch):
    monkeypatch.setattr(
        "datus.plugins.registry.iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("zeta", package="z"), _FakeEntryPoint("alpha", package="a")],
    )
    assert [p.name for p in svc.list_plugins(None)] == ["alpha", "zeta"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
