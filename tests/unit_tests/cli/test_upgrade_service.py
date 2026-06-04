# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.cli.upgrade_service``.

CI-level: ``subprocess.run``, ``shutil.which``, ``httpx.Client`` and
``importlib.metadata.distributions`` are monkey-patched so no real pip /
network / disk-outside-tmp access happens.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from datus.cli import upgrade_service as svc


class _FakeDist:
    """Minimal stand-in for ``importlib.metadata.Distribution``."""

    def __init__(self, name: str, version: str, direct_url: str | None = None):
        self.metadata = {"Name": name}
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str):
        if filename == "direct_url.json":
            return self._direct_url
        return None


def _editable_direct_url() -> str:
    return json.dumps({"url": "file:///home/user/datus-bi-superset", "dir_info": {"editable": True}})


def _local_dir_direct_url() -> str:
    # Non-editable directory install (``pip install ./path``): dir_info present.
    return json.dumps({"url": "file:///home/user/some-pkg", "dir_info": {"editable": False}})


def _local_wheel_direct_url() -> str:
    # Local wheel/archive install (``pip install ./foo.whl``): archive_info, no dir_info.
    return json.dumps({"url": "file:///home/user/foo-0.1.0.whl", "archive_info": {}})


def _wheel_direct_url() -> str:
    return json.dumps({"url": "https://pypi.org/simple/foo", "archive_info": {}})


# ── enumerate_datus_packages ──────────────────────────────────────────


class TestEnumerateDatusPackages:
    def test_filters_non_datus_distributions(self, monkeypatch):
        dists = [
            _FakeDist("datus-agent", "0.3.1"),
            _FakeDist("requests", "2.0.0"),
            _FakeDist("datus-bi-superset", "0.1.0"),
        ]
        monkeypatch.setattr(svc.importlib_metadata, "distributions", lambda: iter(dists))
        names = [p.name for p in svc.enumerate_datus_packages()]
        assert names == ["datus-agent", "datus-bi-superset"]

    def test_normalizes_underscore_names(self, monkeypatch):
        dists = [_FakeDist("datus_bi_superset", "0.1.0"), _FakeDist("datus-agent", "0.3.1")]
        monkeypatch.setattr(svc.importlib_metadata, "distributions", lambda: iter(dists))
        names = [p.name for p in svc.enumerate_datus_packages()]
        assert "datus-bi-superset" in names

    def test_synthesizes_datus_agent_when_not_installed(self, monkeypatch):
        dists = [_FakeDist("datus-bi-superset", "0.1.0")]
        monkeypatch.setattr(svc.importlib_metadata, "distributions", lambda: iter(dists))
        monkeypatch.setattr("datus.__version__", "9.9.9", raising=False)
        packages = svc.enumerate_datus_packages()
        agent = next(p for p in packages if p.name == "datus-agent")
        assert agent.editable is True  # source/checkout → never pip over it
        assert agent.version == "9.9.9"

    def test_editable_flag_from_dir_info(self, monkeypatch):
        dists = [_FakeDist("datus-bi-superset", "0.1.0", direct_url=_editable_direct_url())]
        monkeypatch.setattr(svc.importlib_metadata, "distributions", lambda: iter(dists))
        monkeypatch.setattr("datus.__version__", "0.3.1", raising=False)
        pkg = next(p for p in svc.enumerate_datus_packages() if p.name == "datus-bi-superset")
        assert pkg.editable is True

    def test_editable_flag_from_local_dir_file_url(self, monkeypatch):
        # A local directory install (dir_info present, file:// url) is source/editable.
        dists = [_FakeDist("datus-bi-superset", "0.1.0", direct_url=_local_dir_direct_url())]
        monkeypatch.setattr(svc.importlib_metadata, "distributions", lambda: iter(dists))
        monkeypatch.setattr("datus.__version__", "0.3.1", raising=False)
        pkg = next(p for p in svc.enumerate_datus_packages() if p.name == "datus-bi-superset")
        assert pkg.editable is True

    def test_local_wheel_install_is_not_editable(self, monkeypatch):
        # A local wheel/archive (archive_info, file:// url) is upgradable, not editable.
        dists = [_FakeDist("datus-bi-superset", "0.1.0", direct_url=_local_wheel_direct_url())]
        monkeypatch.setattr(svc.importlib_metadata, "distributions", lambda: iter(dists))
        monkeypatch.setattr("datus.__version__", "0.3.1", raising=False)
        pkg = next(p for p in svc.enumerate_datus_packages() if p.name == "datus-bi-superset")
        assert pkg.editable is False

    def test_normal_wheel_install_is_not_editable(self, monkeypatch):
        dists = [_FakeDist("datus-bi-superset", "0.1.0", direct_url=_wheel_direct_url())]
        monkeypatch.setattr(svc.importlib_metadata, "distributions", lambda: iter(dists))
        monkeypatch.setattr("datus.__version__", "0.3.1", raising=False)
        pkg = next(p for p in svc.enumerate_datus_packages() if p.name == "datus-bi-superset")
        assert pkg.editable is False

    def test_missing_direct_url_is_not_editable(self, monkeypatch):
        dists = [_FakeDist("datus-bi-superset", "0.1.0", direct_url=None)]
        monkeypatch.setattr(svc.importlib_metadata, "distributions", lambda: iter(dists))
        monkeypatch.setattr("datus.__version__", "0.3.1", raising=False)
        pkg = next(p for p in svc.enumerate_datus_packages() if p.name == "datus-bi-superset")
        assert pkg.editable is False


# ── _upgrade_command ──────────────────────────────────────────────────


class TestUpgradeCommand:
    def test_prefers_uv_when_on_path(self, monkeypatch):
        monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/local/bin/uv" if name == "uv" else None)
        argv, label = svc._upgrade_command(["datus-agent", "datus-bi-superset"])
        assert argv == [
            "/usr/local/bin/uv",
            "pip",
            "install",
            "--upgrade",
            "--python",
            svc.sys.executable,
            "datus-agent",
            "datus-bi-superset",
        ]
        assert label == "uv pip install --upgrade"

    def test_falls_back_to_pip(self, monkeypatch):
        monkeypatch.setattr(svc.shutil, "which", lambda name: None)
        argv, label = svc._upgrade_command(["datus-agent"])
        assert argv == [svc.sys.executable, "-m", "pip", "install", "--upgrade", "datus-agent"]
        assert label == "pip install --upgrade"


# ── upgrade_packages ──────────────────────────────────────────────────


class TestUpgradePackages:
    def test_all_editable_skips_subprocess(self, monkeypatch):
        called = []
        monkeypatch.setattr(svc.subprocess, "run", lambda *a, **k: called.append(a))
        packages = [svc.DatusPackage("datus-agent", "0.3.1", editable=True)]
        result = svc.upgrade_packages(packages)
        assert result.ok is True
        assert result.packages == []
        assert result.skipped_editable == ["datus-agent"]
        assert called == []

    def test_mixed_passes_only_non_editable(self, monkeypatch):
        monkeypatch.setattr(svc.shutil, "which", lambda name: None)
        captured = {}

        def fake_run(cmd, capture_output, text, check):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(svc.subprocess, "run", fake_run)
        # After-upgrade snapshot: datus-agent bumped, nothing else relevant.
        monkeypatch.setattr(
            svc, "enumerate_datus_packages", lambda: [svc.DatusPackage("datus-agent", "0.3.2", editable=False)]
        )
        packages = [
            svc.DatusPackage("datus-agent", "0.3.1", editable=False),
            svc.DatusPackage("datus-bi-superset", "0.1.0", editable=True),
        ]
        result = svc.upgrade_packages(packages)
        assert result.ok is True
        assert result.packages == ["datus-agent"]
        assert result.skipped_editable == ["datus-bi-superset"]
        assert "datus-agent" in captured["cmd"]
        assert "datus-bi-superset" not in captured["cmd"]

    def test_changes_only_report_version_diffs(self, monkeypatch):
        """A package whose version did not move is omitted from ``changes``."""
        monkeypatch.setattr(svc.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            svc.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        # datus-agent moved 0.3.1 -> 0.3.2; datus-db-core stayed at 0.1.4.
        monkeypatch.setattr(
            svc,
            "enumerate_datus_packages",
            lambda: [
                svc.DatusPackage("datus-agent", "0.3.2", editable=False),
                svc.DatusPackage("datus-db-core", "0.1.4", editable=False),
            ],
        )
        packages = [
            svc.DatusPackage("datus-agent", "0.3.1", editable=False),
            svc.DatusPackage("datus-db-core", "0.1.4", editable=False),
        ]
        result = svc.upgrade_packages(packages)
        assert result.ok is True
        assert result.changes == [("datus-agent", "0.3.1", "0.3.2")]

    def test_no_changes_when_all_current(self, monkeypatch):
        monkeypatch.setattr(svc.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            svc.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr(
            svc, "enumerate_datus_packages", lambda: [svc.DatusPackage("datus-agent", "0.3.1", editable=False)]
        )
        result = svc.upgrade_packages([svc.DatusPackage("datus-agent", "0.3.1", editable=False)])
        assert result.ok is True
        assert result.changes == []

    def test_nonzero_returncode_marks_failure(self, monkeypatch):
        monkeypatch.setattr(svc.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            svc.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="ERROR: resolve failed"),
        )
        result = svc.upgrade_packages([svc.DatusPackage("datus-agent", "0.3.1")])
        assert result.ok is False
        assert result.error == "pip install --upgrade exited with code 1"
        assert "ERROR" in result.stderr

    def test_subprocess_raises_marks_failure(self, monkeypatch):
        monkeypatch.setattr(svc.shutil, "which", lambda name: None)

        def boom(*a, **k):
            raise OSError("uv vanished")

        monkeypatch.setattr(svc.subprocess, "run", boom)
        result = svc.upgrade_packages([svc.DatusPackage("datus-agent", "0.3.1")])
        assert result.ok is False
        assert "uv vanished" in (result.error or "")


# ── version check + cache ─────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload=None, raise_exc=None):
        self._payload = payload
        self._raise = raise_exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url):
        if self._raise is not None:
            raise self._raise
        return _FakeResponse(self._payload)


def _patch_cache_path(monkeypatch, tmp_path):
    path = tmp_path / "version_check.json"
    monkeypatch.setattr(svc, "_cache_path", lambda agent_config=None: path)
    return path


class TestGetLatestVersion:
    def test_uses_fresh_cache_without_network(self, monkeypatch, tmp_path):
        path = _patch_cache_path(monkeypatch, tmp_path)
        path.write_text(json.dumps({"checked_at": time.time(), "latest": "1.2.3", "name": "datus-agent"}))

        def explode():
            raise AssertionError("network must not be hit when cache is fresh")

        monkeypatch.setattr(svc, "_fetch_latest_from_pypi", explode)
        assert svc.get_latest_version() == "1.2.3"

    def test_fetches_and_writes_cache_when_stale(self, monkeypatch, tmp_path):
        path = _patch_cache_path(monkeypatch, tmp_path)
        stale = time.time() - (svc._CACHE_TTL_SECONDS + 100)
        path.write_text(json.dumps({"checked_at": stale, "latest": "0.0.1", "name": "datus-agent"}))
        monkeypatch.setattr(svc, "_fetch_latest_from_pypi", lambda: "2.0.0")
        assert svc.get_latest_version() == "2.0.0"
        written = json.loads(path.read_text())
        assert written["latest"] == "2.0.0"

    def test_http_error_returns_none_and_caches_negative(self, monkeypatch, tmp_path):
        path = _patch_cache_path(monkeypatch, tmp_path)
        monkeypatch.setattr(svc.importlib_metadata, "version", lambda name: "0.3.1", raising=False)
        monkeypatch.setattr(
            "httpx.Client", lambda timeout: _FakeClient(raise_exc=RuntimeError("offline")), raising=False
        )
        assert svc.get_latest_version() is None
        written = json.loads(path.read_text())
        assert written["latest"] is None

    def test_fetch_parses_info_version(self, monkeypatch, tmp_path):
        _patch_cache_path(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "httpx.Client", lambda timeout: _FakeClient(payload={"info": {"version": "3.1.4"}}), raising=False
        )
        assert svc._fetch_latest_from_pypi() == "3.1.4"


class TestNewerVersionAvailable:
    def test_returns_latest_when_strictly_newer(self, monkeypatch):
        monkeypatch.setattr(svc, "get_latest_version", lambda **k: "1.2.0")
        assert svc.newer_version_available("1.0.0") == "1.2.0"

    def test_none_when_equal(self, monkeypatch):
        monkeypatch.setattr(svc, "get_latest_version", lambda **k: "1.0.0")
        assert svc.newer_version_available("1.0.0") is None

    def test_none_when_older(self, monkeypatch):
        monkeypatch.setattr(svc, "get_latest_version", lambda **k: "0.9.0")
        assert svc.newer_version_available("1.0.0") is None

    def test_none_when_latest_missing(self, monkeypatch):
        monkeypatch.setattr(svc, "get_latest_version", lambda **k: None)
        assert svc.newer_version_available("1.0.0") is None

    def test_invalid_version_returns_none(self, monkeypatch):
        monkeypatch.setattr(svc, "get_latest_version", lambda **k: "not-a-version")
        assert svc.newer_version_available("1.0.0") is None

    def test_cached_only_does_not_touch_network(self, monkeypatch, tmp_path):
        path = _patch_cache_path(monkeypatch, tmp_path)
        path.write_text(json.dumps({"checked_at": time.time(), "latest": "5.0.0", "name": "datus-agent"}))

        def explode(**k):
            raise AssertionError("cached_only must not call get_latest_version")

        monkeypatch.setattr(svc, "get_latest_version", explode)
        assert svc.newer_version_available("1.0.0", cached_only=True) == "5.0.0"

    def test_cached_only_cold_cache_returns_none(self, monkeypatch, tmp_path):
        _patch_cache_path(monkeypatch, tmp_path)  # file absent
        assert svc.newer_version_available("1.0.0", cached_only=True) is None


class TestCacheTtl:
    def test_entry_older_than_ttl_is_ignored(self, monkeypatch, tmp_path):
        path = tmp_path / "version_check.json"
        old = time.time() - (svc._CACHE_TTL_SECONDS + 1)
        path.write_text(json.dumps({"checked_at": old, "latest": "1.0.0", "name": "datus-agent"}))
        assert svc._read_cache(path) is None

    def test_fresh_entry_is_returned(self, tmp_path):
        path = tmp_path / "version_check.json"
        path.write_text(json.dumps({"checked_at": time.time(), "latest": "1.0.0", "name": "datus-agent"}))
        assert svc._read_cache(path)["latest"] == "1.0.0"

    def test_corrupt_cache_returns_none(self, tmp_path):
        path = tmp_path / "version_check.json"
        path.write_text("{not json")
        assert svc._read_cache(path) is None
