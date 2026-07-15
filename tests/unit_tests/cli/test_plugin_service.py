# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.plugin_service`.

CI-level: a throwaway ``~/.datus`` home, ``subprocess.run`` / ``shutil.which``
and the entry-point registry are mocked. No real pip, no network. The mocked
installer populates the ``--target`` directory with a hand-built dist-info so
introspection is exercised against a realistic tree.
"""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from datus.cli import plugin_service as svc
from datus.plugins import registry, store
from datus.utils.path_manager import DatusPathManager, reset_path_manager, set_current_path_manager


@pytest.fixture
def home(tmp_path):
    token = set_current_path_manager(DatusPathManager(datus_home=tmp_path))
    before = list(sys.path)
    try:
        yield tmp_path
    finally:
        sys.path[:] = [p for p in sys.path if p in before or str(tmp_path) not in p]
        registry.invalidate_plugin_cache()
        reset_path_manager(token)


def _fake_proc(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["x"], returncode=returncode, stdout=stdout, stderr=stderr)


def _write_dist_info(
    target: Path,
    *,
    name,
    dist,
    version,
    entry,
    requires_python=">=3.12",
    group="datus.plugins",
    manifest_text="manifest_version: 1\n",
):
    target.mkdir(parents=True, exist_ok=True)
    pkg = target / "datus_demo_plugin"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / store.MANIFEST_FILENAME).write_text(manifest_text, encoding="utf-8")
    dinfo = target / f"{dist.replace('-', '_')}-{version}.dist-info"
    dinfo.mkdir(parents=True, exist_ok=True)
    (dinfo / "entry_points.txt").write_text(f"[{group}]\n{name} = {entry}\n", encoding="utf-8")
    (dinfo / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\nRequires-Python: {requires_python}\n",
        encoding="utf-8",
    )


def _installer(
    name="demo",
    dist="datus-demo-plugin",
    version="0.1.0",
    entry="datus_demo_plugin",
    manifest_text="manifest_version: 1\n",
):
    """Return a fake ``subprocess.run`` that populates the ``--target`` tree."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "--target" in cmd:
            target = Path(cmd[cmd.index("--target") + 1])
            _write_dist_info(target, name=name, dist=dist, version=version, entry=entry, manifest_text=manifest_text)
        return _fake_proc(0)

    fake_run.calls = calls
    return fake_run


def _zip_installer():
    """Fake ``run`` that installs a bundle by populating the ``--target`` dir."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        target = Path(cmd[cmd.index("--target") + 1])
        _write_dist_info(target, name="demo", dist="datus-demo-plugin", version="0.1.0", entry="datus_demo_plugin")
        return _fake_proc(0)

    fake_run.calls = calls
    return fake_run


def _make_bundle(path: Path, *, name="demo", dist="datus-demo-plugin", version="0.1.0", bundle_deps=False, wheels=None):
    """Write a minimal wheelhouse ``.zip`` bundle with a valid manifest."""
    import hashlib

    wheel_name = f"{dist.replace('-', '_')}-{version}-py3-none-any.whl"
    wheels = wheels if wheels is not None else {wheel_name: b"PLUGIN-WHEEL"}
    manifest = {
        "format": svc.BUNDLE_FORMAT,
        "format_version": svc.BUNDLE_FORMAT_VERSION,
        "bundle_deps": bundle_deps,
        "plugin": {
            "name": name,
            "distribution": dist,
            "version": version,
            "wheel": wheel_name,
            "entry_point": "datus_demo_plugin.plugin:DemoPlugin",
        },
        "compat": {"requires_python": "", "platform": "any"},
        "wheels": [
            {"file": fn, "sha256": hashlib.sha256(data).hexdigest(), "role": "plugin" if fn == wheel_name else "dep"}
            for fn, data in wheels.items()
        ],
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(svc.BUNDLE_MANIFEST_NAME, json.dumps(manifest))
        for fn, data in wheels.items():
            zf.writestr(f"{svc.BUNDLE_WHEELS_DIR}/{fn}", data)
    return path


# ── parse_spec ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("pip:datus-foo", ("pip", "datus-foo")),
        ("src:./local", ("src", "./local")),
        ("whl:dist/foo-1.0-py3-none-any.whl", ("whl", "dist/foo-1.0-py3-none-any.whl")),
        ("git:https://github.com/x/y", ("git", "https://github.com/x/y")),
        ("git:git+ssh://git@h/x.git", ("git", "git+ssh://git@h/x.git")),
        ("zip:/abs/bundle.zip", ("zip", "/abs/bundle.zip")),
        ("zip:C:/win/bundle.zip", ("zip", "C:/win/bundle.zip")),  # only first colon split
        ("  git:https://github.com/x/y  ", ("git", "https://github.com/x/y")),  # trimmed
    ],
)
def test_parse_spec_valid(spec, expected):
    assert svc.parse_spec(spec) == expected


@pytest.mark.parametrize(
    "spec,expected_src",
    [
        ("datus-foo", "datus-foo"),  # bare requirement
        ("no-prefix-here", "no-prefix-here"),
        ("datus-foo==1.2.3", "datus-foo==1.2.3"),  # version specifier
        ("datus-foo[extra]", "datus-foo[extra]"),  # extras
        ("foo @ https://example.com/foo-1.0-py3-none-any.whl", "foo @ https://example.com/foo-1.0-py3-none-any.whl"),
    ],
)
def test_parse_spec_defaults_to_pip_without_prefix(spec, expected_src):
    # Missing (or unrecognised) type prefix is treated as a bare pip requirement.
    assert svc.parse_spec(spec) == ("pip", expected_src)


def test_parse_spec_unknown_prefix_falls_back_to_pip():
    # An unrecognised prefix is not a typo error; the whole spec becomes a pip requirement.
    assert svc.parse_spec("wheel:foo") == ("pip", "wheel:foo")


def test_parse_spec_empty_source():
    with pytest.raises(ValueError, match="empty source"):
        svc.parse_spec("pip:")


def test_parse_spec_empty_spec_errors():
    with pytest.raises(ValueError, match="empty install source"):
        svc.parse_spec("   ")


def test_normalize_git_prepends_prefix():
    assert svc._normalize_git("https://h/x") == "git+https://h/x"
    assert svc._normalize_git("git+ssh://h/x") == "git+ssh://h/x"


# ── command builders ────────────────────────────────────────────────────────


def test_target_install_command_prefers_uv(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/uv")
    cmd, label = svc._target_install_command("datus-foo", Path("/t"))
    assert cmd[:3] == ["/usr/bin/uv", "pip", "install"]
    assert "--target" in cmd and cmd[cmd.index("--target") + 1] == "/t"
    assert label == "uv pip install"


def test_target_install_command_falls_back_to_pip(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    cmd, label = svc._target_install_command("datus-foo", Path("/t"), upgrade=True)
    assert cmd[:3] == [sys.executable, "-m", "pip"]
    assert "--upgrade" in cmd
    assert label == "pip install"


def test_bundle_install_command_offline_with_deps(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    cmd, label = svc._bundle_install_command(Path("/w/p.whl"), Path("/w"), Path("/t"), bundle_deps=True)
    assert "--no-index" in cmd and "--find-links" in cmd and "--target" in cmd
    assert "offline" in label


def test_bundle_install_command_online_without_deps(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    cmd, label = svc._bundle_install_command(Path("/w/p.whl"), Path("/w"), Path("/t"), bundle_deps=False)
    assert "--no-index" not in cmd
    assert "--find-links" in cmd and "--target" in cmd


# ── install: pip / src / whl / git via --target ─────────────────────────────


def test_install_pip_lands_in_plugins_dir(home, monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer())
    result = svc.install("pip:datus-demo-plugin")
    assert result.ok is True, result.error
    assert result.name == "demo"
    assert result.new_plugins == ["demo"]
    dest = store.plugin_dir("demo")
    assert dest.is_dir()
    meta = store.read_meta(dest)
    assert meta["install"] == {"type": "pip", "source": "datus-demo-plugin", "ref": None, "origin_artifact": None}
    assert meta["version"] == "0.1.0"


def test_install_bare_requirement_defaults_to_pip(home, monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer())
    result = svc.install("datus-demo-plugin")  # no type prefix
    assert result.ok is True, result.error
    meta = store.read_meta(store.plugin_dir("demo"))
    assert meta["install"] == {"type": "pip", "source": "datus-demo-plugin", "ref": None, "origin_artifact": None}


def test_install_src_records_absolute_path(home, tmp_path, monkeypatch):
    src = tmp_path / "proj"
    src.mkdir()
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer())
    result = svc.install(f"src:{src}")
    assert result.ok is True, result.error
    meta = store.read_meta(store.plugin_dir("demo"))
    assert meta["install"]["type"] == "src"
    assert meta["install"]["source"] == str(src.resolve())


def test_install_src_missing_dir_errors_before_subprocess(home, monkeypatch):
    called = []
    monkeypatch.setattr(svc.subprocess, "run", lambda *a, **k: called.append(1) or _fake_proc(0))
    result = svc.install("src:/no/such/dir")
    assert not result.ok and "not found" in result.error
    assert called == []


def test_install_whl_missing_file_errors(home):
    result = svc.install("whl:/no/such.whl")
    assert not result.ok and "not found" in result.error


def test_install_git_normalizes_and_records_ref(home, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        target = Path(cmd[cmd.index("--target") + 1])
        _write_dist_info(target, name="demo", dist="datus-demo-plugin", version="0.1.0", entry="datus_demo_plugin")
        return _fake_proc(0)

    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    result = svc.install("git:https://github.com/x/y")
    assert result.ok is True, result.error
    assert "git+https://github.com/x/y" in captured["cmd"]
    meta = store.read_meta(store.plugin_dir("demo"))
    assert meta["install"]["type"] == "git"
    assert meta["install"]["ref"] == "git+https://github.com/x/y"


def test_install_already_present_requires_force(home, monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer())
    assert svc.install("pip:datus-demo-plugin").ok is True
    again = svc.install("pip:datus-demo-plugin")
    assert not again.ok and "already installed" in again.error
    assert svc.install("pip:datus-demo-plugin", force=True).ok is True


def test_install_non_plugin_target_rejected(home, monkeypatch):
    def fake_run(cmd, **kwargs):
        target = Path(cmd[cmd.index("--target") + 1])
        _write_dist_info(target, name="x", dist="x", version="1", entry="m:C", group="console_scripts")
        return _fake_proc(0)

    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    result = svc.install("pip:not-a-plugin")
    assert not result.ok and "datus plugin" in result.error
    assert not store.plugins_root().joinpath("x").exists()


def test_install_reserved_name_rejected(home, monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer(name="plugin"))
    result = svc.install("pip:datus-demo-plugin")
    assert not result.ok and "reserved" in result.error


def test_install_subprocess_failure(home, monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", lambda *a, **k: _fake_proc(1, stderr="boom"))
    result = svc.install("pip:datus-demo-plugin")
    assert not result.ok and "exited with code 1" in result.error
    assert result.stderr == "boom"


def test_install_empty_source():
    assert not svc.install("").ok


def test_force_reinstall_refreshes_manifest_cache(home, monkeypatch):
    """A same-name force replace must drop the registry's manifest cache.

    The plugin directory is already on ``sys.path`` after the first install, so
    the added-only invalidation in the activate helpers never fires — the
    install path must invalidate unconditionally for the replaced manifest to
    become visible in-process.
    """
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer())
    assert svc.install("pip:datus-demo-plugin").ok is True
    first = registry.load_plugin_manifest("demo")
    assert first is not None and first.skills is None  # cache primed with the v1 manifest

    monkeypatch.setattr(
        svc.subprocess,
        "run",
        _installer(version="0.2.0", manifest_text="manifest_version: 1\nskills: my_skills\n"),
    )
    assert svc.install("pip:datus-demo-plugin", force=True).ok is True
    second = registry.load_plugin_manifest("demo")
    assert second is not None and second.skills == "my_skills"


# ── install: dest_dir (reserved custom destination) ─────────────────────────


def test_install_dest_dir_lands_outside_store(home, tmp_path, monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer())
    dest = tmp_path / "mounts" / "demo-plugin"
    result = svc.install("pip:datus-demo-plugin", dest_dir=str(dest))
    assert result.ok is True, result.error
    assert result.name == "demo"
    assert result.plugin_dir == str(dest)
    meta = store.read_meta(dest)
    assert meta["name"] == "demo"
    assert meta["install"]["type"] == "pip"
    # The managed store is untouched; the custom directory is activated for
    # this process the way an ``agent.plugin_paths`` mount would be.
    assert not store.plugin_dir("demo").exists()
    assert str(dest) in sys.path


def test_install_dest_dir_existing_requires_force(home, tmp_path, monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer())
    dest = tmp_path / "mounts" / "demo-plugin"
    assert svc.install("pip:datus-demo-plugin", dest_dir=str(dest)).ok is True
    again = svc.install("pip:datus-demo-plugin", dest_dir=str(dest))
    assert not again.ok and "already exists" in again.error
    forced = svc.install("pip:datus-demo-plugin", force=True, dest_dir=str(dest))
    assert forced.ok is True, forced.error
    assert store.read_meta(dest)["name"] == "demo"


# ── install: zip bundle ──────────────────────────────────────────────────────


def test_install_zip_without_deps_is_online(home, tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path / "b.zip", bundle_deps=False)
    runner = _zip_installer()
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", runner)
    result = svc.install(f"zip:{bundle}")
    assert result.ok is True, result.error
    assert "--no-index" not in runner.calls[0]
    dest = store.plugin_dir("demo")
    assert (dest / store.ORIGIN_ZIP).is_file()  # original bundle retained
    meta = store.read_meta(dest)
    assert meta["install"]["type"] == "zip"
    assert meta["install"]["origin_artifact"] == store.ORIGIN_ZIP


def test_install_zip_with_deps_is_offline(home, tmp_path, monkeypatch):
    bundle = _make_bundle(
        tmp_path / "b.zip",
        bundle_deps=True,
        wheels={
            "datus_demo_plugin-0.1.0-py3-none-any.whl": b"PLUGIN",
            "requests-2.0-py3-none-any.whl": b"DEP",
        },
    )
    runner = _zip_installer()
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", runner)
    result = svc.install(f"zip:{bundle}")
    assert result.ok is True, result.error
    assert "--no-index" in runner.calls[0]


def test_install_zip_missing_manifest(home, tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("random.txt", "x")
    result = svc.install(f"zip:{path}")
    assert not result.ok and "manifest" in result.error


def test_install_zip_checksum_mismatch_never_installs(home, tmp_path, monkeypatch):
    path = tmp_path / "b.zip"
    manifest = {
        "format": svc.BUNDLE_FORMAT,
        "format_version": svc.BUNDLE_FORMAT_VERSION,
        "bundle_deps": True,
        "plugin": {
            "name": "demo",
            "distribution": "d",
            "version": "1",
            "wheel": "d-1-py3-none-any.whl",
            "entry_point": "m:C",
        },
        "compat": {"requires_python": "", "platform": "any"},
        "wheels": [{"file": "d-1-py3-none-any.whl", "sha256": "deadbeef", "role": "plugin"}],
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(svc.BUNDLE_MANIFEST_NAME, json.dumps(manifest))
        zf.writestr(f"{svc.BUNDLE_WHEELS_DIR}/d-1-py3-none-any.whl", b"REAL")
    ran = []
    monkeypatch.setattr(svc.subprocess, "run", lambda *a, **k: ran.append(1) or _fake_proc(0))
    result = svc.install(f"zip:{path}")
    assert not result.ok and "checksum" in result.error
    assert ran == []


def test_install_zip_slip_rejected(home, tmp_path, monkeypatch):
    path = tmp_path / "b.zip"
    manifest = {
        "format": svc.BUNDLE_FORMAT,
        "format_version": svc.BUNDLE_FORMAT_VERSION,
        "plugin": {"name": "demo", "distribution": "d", "version": "1", "wheel": "../evil.whl", "entry_point": "m:C"},
        "compat": {},
        "wheels": [{"file": "../evil.whl", "sha256": "x", "role": "plugin"}],
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(svc.BUNDLE_MANIFEST_NAME, json.dumps(manifest))
    ran = []
    monkeypatch.setattr(svc.subprocess, "run", lambda *a, **k: ran.append(1) or _fake_proc(0))
    result = svc.install(f"zip:{path}")
    assert not result.ok and "unsafe wheel filename" in result.error
    assert ran == []


def test_install_zip_dest_dir_lands_outside_store(home, tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path / "b.zip", bundle_deps=False)
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _zip_installer())
    dest = tmp_path / "mounts" / "demo-plugin"
    result = svc.install(f"zip:{bundle}", dest_dir=str(dest))
    assert result.ok is True, result.error
    assert result.plugin_dir == str(dest)
    assert (dest / store.ORIGIN_ZIP).is_file()  # original bundle retained in the custom dir
    assert not store.plugin_dir("demo").exists()
    # The fast pre-check refuses the occupied custom destination without force.
    again = svc.install(f"zip:{bundle}", dest_dir=str(dest))
    assert not again.ok and "already exists" in again.error


def test_install_zip_not_found(home):
    assert not svc.install("zip:/no/such.zip").ok


def test_install_zip_bad_archive(home, tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"not a zip")
    result = svc.install(f"zip:{path}")
    assert not result.ok and "invalid bundle" in result.error


# ── uninstall ────────────────────────────────────────────────────────────────


def test_uninstall_removes_dir(home, monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer())
    svc.install("pip:datus-demo-plugin")
    dest = store.plugin_dir("demo")
    assert dest.is_dir()
    result = svc.uninstall("demo")
    assert result.ok and result.package == "datus-demo-plugin"
    assert not dest.exists()
    assert str(dest) not in sys.path


def test_uninstall_unknown(home):
    result = svc.uninstall("mystery")
    assert not result.ok and "no managed plugin" in result.error


def test_uninstall_empty_name(home):
    assert not svc.uninstall("").ok


# ── upgrade ──────────────────────────────────────────────────────────────────


def test_upgrade_pip_adds_upgrade_flag(home, monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer(version="0.1.0"))
    svc.install("pip:datus-demo-plugin")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        target = Path(cmd[cmd.index("--target") + 1])
        _write_dist_info(target, name="demo", dist="datus-demo-plugin", version="0.2.0", entry="datus_demo_plugin")
        return _fake_proc(0)

    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    result = svc.upgrade("demo")
    assert result.ok is True, result.error
    assert "--upgrade" in calls[0]
    assert result.old_version == "0.1.0" and result.new_version == "0.2.0"


def test_upgrade_zip_is_pinned(home, tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path / "b.zip", bundle_deps=False)
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _zip_installer())
    svc.install(f"zip:{bundle}")
    result = svc.upgrade("demo")
    assert not result.ok and "cannot be upgraded" in result.error


def test_upgrade_unknown(home):
    assert not svc.upgrade("mystery").ok


def test_upgrade_rejects_identity_change(home, monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer(version="0.1.0"))
    svc.install("pip:datus-demo-plugin")

    # The re-fetched distribution now registers a DIFFERENT entry-point name;
    # the upgrade must refuse rather than install a renamed plugin into a new
    # directory while leaving the old one behind.
    monkeypatch.setattr(svc.subprocess, "run", _installer(name="renamed", version="0.2.0"))
    result = svc.upgrade("demo")
    assert not result.ok
    assert "change plugin identity" in result.error
    # The original plugin survives; no directory was created for the new name.
    assert store.plugin_dir("demo").is_dir()
    assert not store.plugin_dir("renamed").exists()


# ── export ───────────────────────────────────────────────────────────────────


def test_export_zip_origin_returns_saved_bundle(home, tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path / "orig.zip", bundle_deps=True)
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _zip_installer())
    svc.install(f"zip:{bundle}")

    out = tmp_path / "out"
    result = svc.export("demo", out_dir=str(out))
    assert result.ok is True, result.error
    exported = Path(result.out_path)
    assert exported.is_file()
    # Exported bytes equal the retained original bundle.
    assert exported.read_bytes() == (store.plugin_dir("demo") / store.ORIGIN_ZIP).read_bytes()


def test_export_src_origin_repacks(home, tmp_path, monkeypatch):
    src = tmp_path / "proj"
    src.mkdir()
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", _installer())
    svc.install(f"src:{src}")

    packed = {}

    def fake_pack(source, out_dir=".", with_deps=False):
        packed.update({"source": source, "out_dir": out_dir, "with_deps": with_deps})
        from datus.cli.plugin_pack import PackResult

        return PackResult(ok=True, bundle_path=f"{out_dir}/repacked.zip", plugin_name="demo", wheel_count=1)

    monkeypatch.setattr(svc, "pack", fake_pack)
    result = svc.export("demo", out_dir=str(tmp_path / "out"))
    assert result.ok is True
    assert packed["source"] == str(src.resolve())
    assert packed["with_deps"] is True


def test_export_unknown(home):
    assert not svc.export("mystery").ok


# ── list_plugins ─────────────────────────────────────────────────────────────


class _FakeEntryPoint:
    def __init__(self, name, dist_name, version, value):
        self.name = name
        self.value = value
        self.dist = type("D", (), {"name": dist_name, "version": version})()


class _FakeConfig:
    def __init__(self, active=None, profiles=None, pins=None):
        self._active = active
        self._profiles = profiles or {}
        self._pins = pins or {}
        self.plugin_services = self._profiles

    def active_plugin_names(self):
        return self._active

    def active_plugin_profiles(self, name):
        return self._pins.get(name)


def test_list_merges_managed_and_external(home, monkeypatch):
    store.write_meta(
        store.plugin_dir("demo"),
        {
            "name": "demo",
            "distribution": "datus-demo-plugin",
            "version": "0.1.0",
            "entry_point": "m:C",
            "install": {"type": "src"},
        },
    )
    monkeypatch.setattr(
        registry,
        "iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("statsig", "datus-statsig-plugin", "2.0", "s:S")],
    )
    infos = svc.list_plugins(None)
    by_name = {i.name: i for i in infos}
    assert by_name["demo"].source == "managed" and by_name["demo"].install_type == "src"
    assert by_name["statsig"].source == "external" and by_name["statsig"].version == "2.0"
    assert [i.name for i in infos] == ["demo", "statsig"]  # sorted


def test_list_managed_wins_over_external_duplicate(home, monkeypatch):
    store.write_meta(
        store.plugin_dir("demo"),
        {"name": "demo", "distribution": "managed-dist", "version": "9.9", "install": {"type": "pip"}},
    )
    monkeypatch.setattr(
        registry,
        "iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("demo", "external-dist", "0.0", "e:E")],
    )
    infos = svc.list_plugins(None)
    assert len(infos) == 1
    assert infos[0].source == "managed" and infos[0].version == "9.9"


def test_list_includes_path_mounted_plugins(home, tmp_path, monkeypatch):
    ext = tmp_path / "ext"
    _write_dist_info(ext, name="pathdemo", dist="datus-path-plugin", version="1.2.3", entry="datus_demo_plugin")
    monkeypatch.setattr(registry, "iter_plugin_entry_points", lambda: [])
    cfg = _FakeConfig(active=None)
    cfg.plugin_paths = [str(ext)]
    infos = svc.list_plugins(cfg)
    assert [i.name for i in infos] == ["pathdemo"]
    assert infos[0].source == "path"
    assert infos[0].package == "datus-path-plugin"
    assert infos[0].version == "1.2.3"
    assert infos[0].active is True


def test_list_path_plugin_listed_while_inactive(home, tmp_path, monkeypatch):
    ext = tmp_path / "ext"
    _write_dist_info(ext, name="pathdemo", dist="datus-path-plugin", version="1.2.3", entry="datus_demo_plugin")
    monkeypatch.setattr(registry, "iter_plugin_entry_points", lambda: [])
    cfg = _FakeConfig(active=set())  # whitelist present, plugin not in it
    cfg.plugin_paths = [str(ext)]
    infos = svc.list_plugins(cfg)
    assert [i.name for i in infos] == ["pathdemo"]
    assert infos[0].active is False


def test_list_skips_path_entry_with_broken_manifest(home, tmp_path, monkeypatch):
    ext = tmp_path / "ext"
    _write_dist_info(ext, name="pathdemo", dist="datus-path-plugin", version="1.2.3", entry="datus_demo_plugin")
    (ext / "datus_demo_plugin" / store.MANIFEST_FILENAME).unlink()  # introspection now fails
    monkeypatch.setattr(registry, "iter_plugin_entry_points", lambda: [])
    cfg = _FakeConfig(active=None)
    cfg.plugin_paths = [str(ext)]
    assert svc.list_plugins(cfg) == []


def test_list_managed_wins_over_path_duplicate(home, tmp_path, monkeypatch):
    store.write_meta(
        store.plugin_dir("demo"),
        {"name": "demo", "distribution": "managed-dist", "version": "9.9", "install": {"type": "pip"}},
    )
    ext = tmp_path / "ext"
    _write_dist_info(ext, name="demo", dist="path-dist", version="0.1", entry="datus_demo_plugin")
    monkeypatch.setattr(registry, "iter_plugin_entry_points", lambda: [])
    cfg = _FakeConfig(active=None)
    cfg.plugin_paths = [str(ext)]
    infos = svc.list_plugins(cfg)
    assert len(infos) == 1
    assert infos[0].source == "managed" and infos[0].version == "9.9"


def test_list_applies_activation_and_profiles(home, monkeypatch):
    store.write_meta(store.plugin_dir("demo"), {"name": "demo", "distribution": "d", "version": "1", "install": {}})
    monkeypatch.setattr(registry, "iter_plugin_entry_points", lambda: [])
    cfg = _FakeConfig(active={"demo"}, profiles={"demo": {"prod": {}, "dev": {}}}, pins={"demo": ["prod"]})
    infos = svc.list_plugins(cfg)
    assert infos[0].active is True
    assert infos[0].profiles == ["dev", "prod"]
    assert infos[0].active_profiles == ["prod"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
