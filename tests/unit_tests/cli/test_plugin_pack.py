# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.plugin_pack`` (build wheelhouse ``.zip`` bundles).

CI-level: the ``pip wheel`` / ``pip download`` subprocess is mocked so no network
or real packaging tooling runs. Wheels are minimal hand-built zips carrying only
the ``dist-info`` metadata the packer reads.
"""

import json
import zipfile
from pathlib import Path

import pytest

from datus.cli import plugin_pack as pack
from datus.cli import plugin_service as svc


def _make_wheel(
    path: Path,
    *,
    dist="datus-demo-plugin",
    version="0.1.0",
    entry="datus_demo_plugin",
    requires_python=">=3.10",
    group="datus.plugins",
    manifest="manifest_version: 1\n",
):
    """Write a minimal wheel zip with just the dist-info the packer reads."""
    dist_info = f"{dist.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        meta = f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n"
        if requires_python:
            meta += f"Requires-Python: {requires_python}\n"
        zf.writestr(f"{dist_info}/METADATA", meta)
        if group is not None:
            zf.writestr(f"{dist_info}/entry_points.txt", f"[{group}]\ndemo = {entry}\n")
        if group is not None and manifest is not None:
            zf.writestr("datus_demo_plugin/datus-plugin.yml", manifest)
    return path


def _mock_run(monkeypatch, *, plugin_wheel=True, dep_wheel=False):
    """Mock ``pip wheel`` (builds the plugin wheel) and ``pip download`` (deps)."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "wheel" in cmd and "--no-deps" in cmd and plugin_wheel:
            out = Path(cmd[cmd.index("--wheel-dir") + 1])
            _make_wheel(out / "datus_demo_plugin-0.1.0-py3-none-any.whl")
        elif "download" in cmd and dep_wheel:
            out = Path(cmd[cmd.index("-d") + 1])
            _make_wheel(out / "requests-2.0-py3-none-any.whl", dist="requests", version="2.0", group=None)
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pack.subprocess, "run", fake_run)
    return calls


# ── wheel-filename parsing ───────────────────────────────────────────────────


def test_parse_wheel_filename():
    assert pack._parse_wheel_filename("datus_demo_plugin-0.1.0-py3-none-any.whl") == (
        "datus-demo-plugin",
        "0.1.0",
        "any",
    )


# ── entry-point reading ──────────────────────────────────────────────────────


def test_read_plugin_entry(tmp_path):
    wheel = _make_wheel(tmp_path / "w.whl")
    name, target = pack._read_plugin_entry(wheel)
    assert name == "demo" and target == "datus_demo_plugin"


def test_read_plugin_entry_rejects_non_plugin(tmp_path):
    wheel = _make_wheel(tmp_path / "w.whl", group=None)
    with pytest.raises(pack.PackError, match="not a datus plugin"):
        pack._read_plugin_entry(wheel)


def test_read_plugin_entry_rejects_legacy_class_entry(tmp_path):
    wheel = _make_wheel(tmp_path / "w.whl", entry="datus_demo_plugin.plugin:DemoPlugin")
    with pytest.raises(pack.PackError, match="legacy class-based contract"):
        pack._read_plugin_entry(wheel)


def test_read_plugin_entry_rejects_missing_manifest(tmp_path):
    wheel = _make_wheel(tmp_path / "w.whl", manifest=None)
    with pytest.raises(pack.PackError, match="datus-plugin.yml"):
        pack._read_plugin_entry(wheel)


def test_read_plugin_entry_rejects_invalid_manifest(tmp_path):
    # A present-but-invalid manifest (unsupported version) is rejected at build
    # time, not deferred to install time.
    wheel = _make_wheel(tmp_path / "w.whl", manifest="manifest_version: 99\n")
    with pytest.raises(pack.PackError, match="invalid"):
        pack._read_plugin_entry(wheel)


# ── pack: default (no deps) ──────────────────────────────────────────────────


def test_pack_without_deps(tmp_path, monkeypatch):
    calls = _mock_run(monkeypatch, dep_wheel=True)  # dep_wheel available but must NOT be fetched
    result = pack.pack("./demo", out_dir=str(tmp_path), with_deps=False)
    assert result.ok is True, result.error
    assert result.wheel_count == 1
    assert result.plugin_name == "demo"
    assert result.with_deps is False
    assert Path(result.bundle_path).name == "datus-demo-plugin-0.1.0.zip"
    # No dependency download happened.
    assert not any("download" in c for c in calls)
    # The plugin wheel was resolved with --no-deps.
    assert any("wheel" in c and "--no-deps" in c for c in calls)

    with zipfile.ZipFile(result.bundle_path) as zf:
        manifest = json.loads(zf.read(svc.BUNDLE_MANIFEST_NAME))
        names = zf.namelist()
    assert manifest["format"] == svc.BUNDLE_FORMAT
    assert manifest["bundle_deps"] is False
    assert manifest["plugin"] == {
        "name": "demo",
        "distribution": "datus-demo-plugin",
        "version": "0.1.0",
        "wheel": "datus_demo_plugin-0.1.0-py3-none-any.whl",
        "entry_point": "datus_demo_plugin",
    }
    assert manifest["compat"]["requires_python"] == ">=3.10"
    assert [w["role"] for w in manifest["wheels"]] == ["plugin"]
    assert all(w["sha256"] for w in manifest["wheels"])
    assert "wheels/datus_demo_plugin-0.1.0-py3-none-any.whl" in names


# ── pack: with deps ──────────────────────────────────────────────────────────


def test_pack_with_deps(tmp_path, monkeypatch):
    calls = _mock_run(monkeypatch, dep_wheel=True)
    result = pack.pack("./demo", out_dir=str(tmp_path), with_deps=True)
    assert result.ok is True, result.error
    assert result.wheel_count == 2
    assert result.with_deps is True
    assert any("download" in c for c in calls)

    with zipfile.ZipFile(result.bundle_path) as zf:
        manifest = json.loads(zf.read(svc.BUNDLE_MANIFEST_NAME))
    assert manifest["bundle_deps"] is True
    assert {w["role"] for w in manifest["wheels"]} == {"plugin", "dependency"}


def test_pack_default_source_is_cwd(tmp_path, monkeypatch):
    calls = _mock_run(monkeypatch)
    result = pack.pack(out_dir=str(tmp_path))
    assert result.ok is True
    # ``.`` is passed through to pip wheel as the source.
    assert calls[0][-1] == "."


# ── pack: failure paths ──────────────────────────────────────────────────────


def test_pack_non_plugin_fails_before_download(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "wheel" in cmd and "--no-deps" in cmd:
            out = Path(cmd[cmd.index("--wheel-dir") + 1])
            _make_wheel(out / "not_a_plugin-1.0-py3-none-any.whl", dist="not-a-plugin", version="1.0", group=None)
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pack.subprocess, "run", fake_run)
    result = pack.pack("./x", out_dir=str(tmp_path), with_deps=True)
    assert not result.ok and "not a datus plugin" in result.error
    assert not any("download" in c for c in calls)


def test_pack_build_failure(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        import subprocess

        return subprocess.CompletedProcess(cmd, 1, "", "wheel build blew up")

    monkeypatch.setattr(pack.subprocess, "run", fake_run)
    result = pack.pack("./x", out_dir=str(tmp_path))
    assert not result.ok
    assert "wheel build blew up" in result.stderr


# ── round-trip: pack output is a valid install bundle ────────────────────────


def test_pack_output_is_installable_bundle(tmp_path, monkeypatch):
    """The packer's manifest is accepted by the service's bundle reader."""
    _mock_run(monkeypatch, dep_wheel=True)
    result = pack.pack("./demo", out_dir=str(tmp_path), with_deps=True)
    assert result.ok is True

    with zipfile.ZipFile(result.bundle_path) as zf:
        manifest = svc._read_bundle_manifest(zf)
        assert svc._bundle_deps_flag(manifest) is True
        wheels_dir = svc._extract_and_verify_wheels(zf, manifest, tmp_path / "extract")
    # Both wheels extracted and checksum-verified without error.
    assert sorted(p.name for p in wheels_dir.glob("*.whl")) == [
        "datus_demo_plugin-0.1.0-py3-none-any.whl",
        "requests-2.0-py3-none-any.whl",
    ]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
