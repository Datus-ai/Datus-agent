# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.cli.service_adapter_installer``.

CI-level: ``subprocess.run`` and ``importlib.util.find_spec`` are
monkey-patched so no real pip / network access happens.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from datus.cli import service_adapter_installer as sai


class TestPackageFor:
    def test_bi_platforms_mapping(self):
        pkg, mod = sai.package_for("bi_platforms", "Superset")
        assert pkg == "datus-bi-superset"
        assert mod == "datus_bi_superset"

    def test_schedulers_mapping(self):
        pkg, mod = sai.package_for("schedulers", "AIRFLOW")
        assert pkg == "datus-scheduler-airflow"
        assert mod == "datus_scheduler_airflow"

    def test_semantic_layer_mapping(self):
        pkg, mod = sai.package_for("semantic_layer", "MetricFlow")
        assert pkg == "datus-semantic-metricflow"
        assert mod == "datus_semantic_metricflow"

    def test_unknown_section_raises(self):
        with pytest.raises(ValueError):
            sai.package_for("unknown_section", "x")

    def test_blank_type_raises(self):
        with pytest.raises(ValueError):
            sai.package_for("bi_platforms", "   ")


class TestIsAdapterInstalled:
    def test_returns_true_when_spec_found(self, monkeypatch):
        monkeypatch.setattr(
            sai.importlib.util,
            "find_spec",
            lambda name: SimpleNamespace(name=name),
        )
        assert sai.is_adapter_installed("bi_platforms", "superset") is True

    def test_returns_false_when_spec_missing(self, monkeypatch):
        monkeypatch.setattr(sai.importlib.util, "find_spec", lambda name: None)
        assert sai.is_adapter_installed("bi_platforms", "superset") is False

    def test_swallows_import_error(self, monkeypatch):
        def _raise(name):
            raise ImportError("ancestor missing")

        monkeypatch.setattr(sai.importlib.util, "find_spec", _raise)
        assert sai.is_adapter_installed("bi_platforms", "superset") is False

    def test_unknown_section_returns_false(self):
        assert sai.is_adapter_installed("unknown_section", "x") is False


class TestEnsureAdapter:
    def test_returns_ok_when_already_installed(self, monkeypatch):
        monkeypatch.setattr(sai, "is_adapter_installed", lambda *_: True)
        called = []
        monkeypatch.setattr(sai.subprocess, "run", lambda *a, **k: called.append(a) or None)
        result = sai.ensure_adapter("bi_platforms", "superset")
        assert result.ok is True
        assert result.package == "datus-bi-superset"
        assert called == []  # no pip invocation when already installed

    def test_runs_pip_when_missing_and_succeeds(self, monkeypatch):
        monkeypatch.setattr(sai, "is_adapter_installed", lambda *_: False)
        captured = {}

        def fake_run(cmd, capture_output, text, check):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="installed\n", stderr="")

        monkeypatch.setattr(sai.subprocess, "run", fake_run)
        monkeypatch.setattr(sai.importlib, "invalidate_caches", lambda: None)

        result = sai.ensure_adapter("schedulers", "airflow")
        assert result.ok is True
        assert result.package == "datus-scheduler-airflow"
        assert captured["cmd"][:3] == [sai.sys.executable, "-m", "pip"]
        assert "datus-scheduler-airflow" in captured["cmd"]

    def test_pip_failure_surfaces_error(self, monkeypatch):
        monkeypatch.setattr(sai, "is_adapter_installed", lambda *_: False)
        monkeypatch.setattr(
            sai.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="ERROR: not found"),
        )
        result = sai.ensure_adapter("bi_platforms", "ghost")
        assert result.ok is False
        assert "exited with code 1" in (result.error or "")
        assert "ERROR" in result.stderr

    def test_unknown_section_returns_error(self):
        result = sai.ensure_adapter("unknown_section", "x")
        assert result.ok is False
        assert "Unsupported" in (result.error or "")

    def test_semantic_layer_runs_pip_for_metricflow(self, monkeypatch):
        """Creating a semantic_layer service must trigger
        ``pip install datus-semantic-metricflow`` when the package isn't
        already importable — this is the entire point of the new tab."""
        monkeypatch.setattr(sai, "is_adapter_installed", lambda *_: False)
        captured = {}

        def fake_run(cmd, capture_output, text, check):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="installed\n", stderr="")

        monkeypatch.setattr(sai.subprocess, "run", fake_run)
        monkeypatch.setattr(sai.importlib, "invalidate_caches", lambda: None)

        result = sai.ensure_adapter("semantic_layer", "metricflow")
        assert result.ok is True
        assert result.package == "datus-semantic-metricflow"
        assert "datus-semantic-metricflow" in captured["cmd"]

    def test_line_callback_receives_pip_output(self, monkeypatch):
        monkeypatch.setattr(sai, "is_adapter_installed", lambda *_: False)
        monkeypatch.setattr(
            sai.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=0, stdout="line1\nline2", stderr="warn"),
        )
        monkeypatch.setattr(sai.importlib, "invalidate_caches", lambda: None)

        seen = []
        sai.ensure_adapter("bi_platforms", "superset", line_callback=seen.append)
        assert "line1" in seen
        assert "line2" in seen
        assert "warn" in seen


class _FakeEntryPoint:
    """Drop-in for :class:`importlib.metadata.EntryPoint` for tests.

    Only the surface ``hot_reload_adapter`` touches is implemented:
    ``name`` (used by selection) and ``load()`` (returns the register
    callable). The real EntryPoint resolves the attribute behind the
    colon at load-time; this stub returns whatever the test gave it.
    """

    def __init__(self, name, register_fn):
        self.name = name
        self._fn = register_fn

    def load(self):
        return self._fn


class _FakeEntryPoints:
    """Behaves like the ``EntryPoints`` returned by ``entry_points()``."""

    def __init__(self, by_group):
        self._by_group = by_group  # dict[group_name, list[_FakeEntryPoint]]

    def select(self, *, group=None, name=None):
        eps = list(self._by_group.get(group or "", []))
        if name is not None:
            eps = [ep for ep in eps if ep.name == name]
        return eps


class TestHotReloadAdapter:
    def test_returns_false_for_unknown_section(self):
        assert sai.hot_reload_adapter("unknown_section", "x") is False

    def test_imports_module_and_runs_entry_point_register(self, monkeypatch):
        """Adapter packages don't auto-register on import — ``hot_reload_adapter``
        must explicitly load the matching entry-point and call its register
        callable. Otherwise an entry-point-only adapter (the actual shape
        of ``datus-bi-superset`` and ``datus-scheduler-airflow``) ends up
        importable but absent from ``adapter_registry``."""
        called = []
        register = lambda: called.append("register")  # noqa: E731

        monkeypatch.setattr(sai.importlib, "import_module", lambda name: object())
        monkeypatch.delitem(sai.sys.modules, "datus_scheduler_airflow", raising=False)
        fake = _FakeEntryPoints({"datus.schedulers": [_FakeEntryPoint("airflow", register)]})
        import importlib.metadata as metadata

        monkeypatch.setattr(metadata, "entry_points", lambda: fake)
        assert sai.hot_reload_adapter("schedulers", "airflow") is True
        assert called == ["register"]

    def test_returns_false_when_entry_point_missing(self, monkeypatch):
        """No matching entry point → nothing was registered → False.
        The caller (``ServiceCommands._do_save``) keys on this to surface
        a "package installed but adapter not registered" error rather
        than letting the probe fail with a deeper exception."""
        monkeypatch.setattr(sai.importlib, "import_module", lambda name: object())
        monkeypatch.delitem(sai.sys.modules, "datus_scheduler_airflow", raising=False)
        empty = _FakeEntryPoints({})
        import importlib.metadata as metadata

        monkeypatch.setattr(metadata, "entry_points", lambda: empty)
        assert sai.hot_reload_adapter("schedulers", "airflow") is False

    def test_reloads_already_imported_module(self, monkeypatch):
        cached = {}
        sentinel = object()

        def fake_reload(mod):
            cached["reloaded"] = mod
            return mod

        register = lambda: cached.setdefault("registered", True)  # noqa: E731
        monkeypatch.setitem(sai.sys.modules, "datus_scheduler_airflow", sentinel)
        monkeypatch.setattr(sai.importlib, "reload", fake_reload)
        fake = _FakeEntryPoints({"datus.schedulers": [_FakeEntryPoint("airflow", register)]})
        import importlib.metadata as metadata

        monkeypatch.setattr(metadata, "entry_points", lambda: fake)
        assert sai.hot_reload_adapter("schedulers", "airflow") is True
        assert cached["reloaded"] is sentinel
        assert cached["registered"] is True

    def test_import_failure_returns_false(self, monkeypatch):
        def boom(name):
            raise ImportError("nope")

        monkeypatch.setattr(sai.importlib, "import_module", boom)
        monkeypatch.delitem(sai.sys.modules, "datus_scheduler_airflow", raising=False)
        assert sai.hot_reload_adapter("schedulers", "airflow") is False

    def test_bi_branch_invokes_register_then_discover(self, monkeypatch):
        """For BI we still call ``discover_adapters`` after the explicit
        register call so the registry's own cached scan stays consistent
        for callers that walk it themselves (e.g. the listing path)."""
        monkeypatch.setattr(sai.importlib, "import_module", lambda name: object())
        monkeypatch.delitem(sai.sys.modules, "datus_bi_superset", raising=False)
        register_calls = []
        discover_calls = []

        fake = _FakeEntryPoints(
            {"datus.bi_adapters": [_FakeEntryPoint("superset", lambda: register_calls.append(True))]}
        )
        import importlib.metadata as metadata

        monkeypatch.setattr(metadata, "entry_points", lambda: fake)

        fake_registry = SimpleNamespace(discover_adapters=lambda: discover_calls.append(True))
        fake_module = SimpleNamespace(adapter_registry=fake_registry)
        with patch.dict(sai.sys.modules, {"datus_bi_core": fake_module}):
            assert sai.hot_reload_adapter("bi_platforms", "superset") is True
        assert register_calls == [True]
        assert discover_calls == [True]

    def test_semantic_branch_uses_semantic_entry_point_group(self, monkeypatch):
        """Semantic adapter packages register under ``datus.semantic_adapters``
        — verify ``hot_reload_adapter`` looks up that group (not the BI /
        scheduler ones) and runs the matching ``register`` callable."""
        monkeypatch.setattr(sai.importlib, "import_module", lambda name: object())
        monkeypatch.delitem(sai.sys.modules, "datus_semantic_metricflow", raising=False)
        register_calls = []
        discover_calls = []

        fake = _FakeEntryPoints(
            {"datus.semantic_adapters": [_FakeEntryPoint("metricflow", lambda: register_calls.append(True))]}
        )
        import importlib.metadata as metadata

        monkeypatch.setattr(metadata, "entry_points", lambda: fake)

        fake_registry = SimpleNamespace(discover_adapters=lambda: discover_calls.append(True))
        fake_module = SimpleNamespace(semantic_adapter_registry=fake_registry)
        with patch.dict(sai.sys.modules, {"datus.tools.semantic_tools.registry": fake_module}):
            assert sai.hot_reload_adapter("semantic_layer", "metricflow") is True
        assert register_calls == [True]
        assert discover_calls == [True]
