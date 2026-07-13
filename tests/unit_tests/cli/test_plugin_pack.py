# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.plugin_pack`` (build ``.dplug`` bundles).

CI-level: the wheel builder / ``pip download`` subprocess is mocked so no
network or real packaging tooling runs. Wheels are minimal hand-built zips
carrying only the ``dist-info`` metadata the packer reads.
"""

import json
import types
import zipfile
from pathlib import Path

import pytest

from datus.cli import plugin_pack as pack
from datus.cli import plugin_service as svc

_WHEEL_NAME = "datus_plugin_hello-1.0.0-py3-none-any.whl"


def _make_wheel(
    path,
    *,
    dist="datus_plugin_hello",
    version="1.0.0",
    ep_group="datus.plugins",
    ep="hello = datus_plugin_hello.plugin:HelloPlugin",
    requires_python=">=3.8",
    include_entry_points=True,
    include_metadata=True,
):
    """Write a minimal but structurally valid wheel (zip) at ``path``."""
    info = f"{dist}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{dist}/__init__.py", "")
        if include_metadata:
            meta = f"Metadata-Version: 2.1\nName: {dist.replace('_', '-')}\nVersion: {version}\n"
            if requires_python:
                meta += f"Requires-Python: {requires_python}\n"
            zf.writestr(f"{info}/METADATA", meta)
        if include_entry_points:
            zf.writestr(f"{info}/entry_points.txt", f"[{ep_group}]\n{ep}\n")
        zf.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    return path


# ── _read_plugin_entry ─────────────────────────────────────────────────────


def test_read_plugin_entry(tmp_path):
    wheel = _make_wheel(tmp_path / _WHEEL_NAME)
    name, target = pack._read_plugin_entry(wheel)
    assert name == "hello"
    assert target == "datus_plugin_hello.plugin:HelloPlugin"


def test_read_plugin_entry_preserves_case(tmp_path):
    wheel = _make_wheel(tmp_path / _WHEEL_NAME, ep="MyPlugin = pkg.plugin:P")
    name, _ = pack._read_plugin_entry(wheel)
    assert name == "MyPlugin"  # configparser must not lowercase the name


def test_read_plugin_entry_no_datus_group(tmp_path):
    wheel = _make_wheel(tmp_path / _WHEEL_NAME, ep_group="console_scripts", ep="foo = pkg:main")
    with pytest.raises(pack.PackError, match="not a datus plugin"):
        pack._read_plugin_entry(wheel)


def test_read_plugin_entry_no_entry_points(tmp_path):
    wheel = _make_wheel(tmp_path / _WHEEL_NAME, include_entry_points=False)
    with pytest.raises(pack.PackError, match="declares no entry points"):
        pack._read_plugin_entry(wheel)


# ── _read_requires_python ───────────────────────────────────────────────────


def test_read_requires_python(tmp_path):
    wheel = _make_wheel(tmp_path / _WHEEL_NAME, requires_python=">=3.12")
    assert pack._read_requires_python(wheel) == ">=3.12"


def test_read_requires_python_absent(tmp_path):
    wheel = _make_wheel(tmp_path / _WHEEL_NAME, requires_python="")
    assert pack._read_requires_python(wheel) is None


def test_read_requires_python_no_metadata(tmp_path):
    wheel = _make_wheel(tmp_path / _WHEEL_NAME, include_metadata=False)
    assert pack._read_requires_python(wheel) is None


# ── _parse_wheel_filename ───────────────────────────────────────────────────


def test_parse_wheel_filename():
    dist, version, plat = pack._parse_wheel_filename(_WHEEL_NAME)
    assert dist == "datus-plugin-hello"
    assert version == "1.0.0"
    assert plat == "any"


# ── pack (end to end, subprocess mocked) ────────────────────────────────────


def _mock_pip_download(monkeypatch, *, dep_name="somedep-2.0-py3-none-any.whl", proc_overrides=None):
    """Fake ``pip download`` that drops a dependency wheel into the -d target."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if proc_overrides is not None:
            return proc_overrides
        if "download" in cmd:
            dest = Path(cmd[cmd.index("-d") + 1])
            (dest / dep_name).write_bytes(b"DEP-CONTENT")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pack.subprocess, "run", fake_run)
    return calls


def test_pack_roundtrip_from_wheel(tmp_path, monkeypatch):
    """pack() builds a bundle install_bundle() can consume — full round-trip."""
    src_wheel = _make_wheel(tmp_path / _WHEEL_NAME, requires_python=">=3.10")
    out_dir = tmp_path / "dist"
    _mock_pip_download(monkeypatch)

    result = pack.pack(str(src_wheel), out_dir=str(out_dir))
    assert result.ok is True, result.error
    assert result.plugin_name == "hello"
    assert result.wheel_count == 2  # main wheel + one dep

    bundle_path = Path(result.bundle_path)
    assert bundle_path.exists()
    assert bundle_path.name == "datus-plugin-hello-1.0.0-any.dplug"

    with zipfile.ZipFile(bundle_path) as zf:
        manifest = json.loads(zf.read(svc.BUNDLE_MANIFEST_NAME))
        names = set(zf.namelist())
    assert manifest["format"] == svc.BUNDLE_FORMAT
    assert manifest["format_version"] == svc.BUNDLE_FORMAT_VERSION
    assert manifest["plugin"] == {
        "name": "hello",
        "distribution": "datus-plugin-hello",
        "version": "1.0.0",
        "wheel": _WHEEL_NAME,
        "entry_point": "datus_plugin_hello.plugin:HelloPlugin",
    }
    assert manifest["compat"]["requires_python"] == ">=3.10"
    assert manifest["compat"]["platform"] == "any"
    assert {w["role"] for w in manifest["wheels"]} == {"plugin", "dependency"}
    assert all(w["sha256"] for w in manifest["wheels"])
    assert f"{svc.BUNDLE_WHEELS_DIR}/{_WHEEL_NAME}" in names

    # The produced bundle installs offline end-to-end (install subprocess mocked).
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        svc.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    monkeypatch.setattr("datus.plugins.registry.iter_plugin_entry_points", lambda: [])
    monkeypatch.setattr("datus.plugins.registry.invalidate_plugin_cache", lambda: None)
    install_result = svc.install_bundle(str(bundle_path))
    assert install_result.ok is True


def test_pack_empty_source():
    result = pack.pack("   ")
    assert not result.ok
    assert "no pack source" in (result.error or "")


def test_pack_not_a_plugin_fails_before_download(tmp_path, monkeypatch):
    src_wheel = _make_wheel(tmp_path / _WHEEL_NAME, ep_group="console_scripts", ep="foo = pkg:main")
    calls = _mock_pip_download(monkeypatch)
    result = pack.pack(str(src_wheel), out_dir=str(tmp_path / "dist"))
    assert not result.ok
    assert "not a datus plugin" in (result.error or "")
    assert calls == []  # entry-point check happens before any dependency download


def test_pack_download_failure(tmp_path, monkeypatch):
    src_wheel = _make_wheel(tmp_path / _WHEEL_NAME)
    _mock_pip_download(
        monkeypatch, proc_overrides=types.SimpleNamespace(returncode=1, stdout="", stderr="network down")
    )
    result = pack.pack(str(src_wheel), out_dir=str(tmp_path / "dist"))
    assert not result.ok
    assert "download dependencies failed" in (result.error or "")
    assert result.stderr == "network down"


def test_pack_forwards_cross_target_flags(tmp_path, monkeypatch):
    src_wheel = _make_wheel(tmp_path / _WHEEL_NAME)
    calls = _mock_pip_download(monkeypatch)
    result = pack.pack(
        str(src_wheel), out_dir=str(tmp_path / "dist"), python_version="3.11", platform_tag="manylinux2014_x86_64"
    )
    assert result.ok is True
    download_cmd = next(c for c in calls if "download" in c)
    assert "--python-version" in download_cmd and "3.11" in download_cmd
    assert "--platform" in download_cmd and "manylinux2014_x86_64" in download_cmd


# ── build / download helpers ────────────────────────────────────────────────


def test_run_raises_on_missing_tool(monkeypatch):
    def boom(cmd, **k):
        raise FileNotFoundError("uv not found")

    monkeypatch.setattr(pack.subprocess, "run", boom)
    with pytest.raises(pack.PackError, match="failed to start"):
        pack._run(["uv", "build"], "build wheel")


def test_parse_wheel_filename_fallback():
    """A filename packaging cannot parse falls back to a positional split."""
    dist, version, plat = pack._parse_wheel_filename("weirdname.whl")
    assert dist == "weirdname"
    assert plat == "any"


def test_build_wheel_from_dir(tmp_path, monkeypatch):
    src = tmp_path / "proj"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(pack.shutil, "which", lambda name: None)  # force python -m build

    def fake_run(cmd, **k):
        assert cmd[:4] == [pack.sys.executable, "-m", "build", "--wheel"]
        (out / _WHEEL_NAME).write_bytes(b"WHEEL")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pack.subprocess, "run", fake_run)
    assert pack._build_wheel_from_dir(src, out).name == _WHEEL_NAME


def test_build_wheel_from_dir_no_output(tmp_path, monkeypatch):
    src = tmp_path / "proj"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(pack.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        pack.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    with pytest.raises(pack.PackError, match="no wheel produced"):
        pack._build_wheel_from_dir(src, out)


def test_download_named_wheel(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()

    def fake_run(cmd, **k):
        assert "--no-deps" in cmd  # the plugin's own wheel only, not its deps
        (out / _WHEEL_NAME).write_bytes(b"WHEEL")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pack.subprocess, "run", fake_run)
    assert pack._download_named_wheel("datus-plugin-hello", out).name == _WHEEL_NAME


def test_pack_from_directory(tmp_path, monkeypatch):
    """pack() from a project dir builds the wheel, then downloads deps."""
    src = tmp_path / "datus-plugin-hello"
    src.mkdir()
    out_dir = tmp_path / "dist"
    monkeypatch.setattr(pack.shutil, "which", lambda name: None)

    def fake_run(cmd, **k):
        if "build" in cmd:
            _make_wheel(Path(cmd[cmd.index("--outdir") + 1]) / _WHEEL_NAME)
        elif "download" in cmd:
            (Path(cmd[cmd.index("-d") + 1]) / "dep-1.0-py3-none-any.whl").write_bytes(b"DEP")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pack.subprocess, "run", fake_run)
    result = pack.pack(str(src), out_dir=str(out_dir))
    assert result.ok is True, result.error
    assert result.plugin_name == "hello"
    assert result.wheel_count == 2


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
