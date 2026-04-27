# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.cli.service_bootstrap``.

CI-level: ``sys.stdin.isatty`` / ``sys.stdout.isatty`` /
``ensure_adapter`` / ``hot_reload_adapter`` are monkey-patched, no real
TTY interaction or pip invocations happen.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli import service_bootstrap as sb

# Captured before any test patches it — restored by tests that exercise
# the actual install pass.
_REAL_BACKGROUND_INSTALL = sb.background_install_missing


def _fake_cli(agent_config) -> SimpleNamespace:
    """Minimal stub matching the surface ``service_bootstrap`` reads."""
    return SimpleNamespace(
        agent_config=agent_config,
        console=Console(file=io.StringIO(), no_color=True, force_terminal=False),
    )


def _make_agent(
    *,
    dashboards=None,
    schedulers=None,
    semantic=None,
    active_dash=None,
    active_sched=None,
    active_semantic=None,
    default_dash=None,
    default_sched=None,
    default_semantic=None,
):
    cfg = MagicMock()
    cfg.dashboard_config = dashboards or {}
    cfg.scheduler_services = schedulers or {}
    cfg.semantic_layer_configs = semantic or {}
    cfg.active_dashboard = MagicMock(return_value=active_dash)
    cfg.active_scheduler = MagicMock(return_value=active_sched)
    cfg.active_semantic = MagicMock(return_value=active_semantic)
    cfg.default_dashboard_service = MagicMock(return_value=default_dash)
    cfg.default_scheduler_service = MagicMock(return_value=default_sched)
    cfg.default_semantic_adapter = MagicMock(return_value=default_semantic)
    cfg.set_active_dashboard = MagicMock()
    cfg.set_active_scheduler = MagicMock()
    cfg.set_active_semantic = MagicMock()
    return cfg


# ─────────────────────────────────────────────────────────────────────
# Guards: stdin/stdout TTY + DATUS_DISABLE_SERVICE_BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────


class TestInteractiveGuard:
    def test_skips_when_stdin_not_a_tty(self, monkeypatch):
        monkeypatch.setattr(sb.sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(sb.sys.stdout, "isatty", lambda: True)
        cfg = _make_agent(dashboards={"superset": MagicMock(default=False)}, default_dash="superset")
        cli = _fake_cli(cfg)
        sb.run(cli)
        cfg.set_active_dashboard.assert_not_called()

    def test_skips_when_stdout_not_a_tty(self, monkeypatch):
        monkeypatch.setattr(sb.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sb.sys.stdout, "isatty", lambda: False)
        cfg = _make_agent(dashboards={"superset": MagicMock(default=False)}, default_dash="superset")
        cli = _fake_cli(cfg)
        sb.run(cli)
        cfg.set_active_dashboard.assert_not_called()

    def test_skips_when_env_disables_bootstrap(self, monkeypatch):
        monkeypatch.setattr(sb.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sb.sys.stdout, "isatty", lambda: True)
        monkeypatch.setenv("DATUS_DISABLE_SERVICE_BOOTSTRAP", "1")
        cfg = _make_agent(dashboards={"superset": MagicMock(default=False)}, default_dash="superset")
        cli = _fake_cli(cfg)
        sb.run(cli)
        cfg.set_active_dashboard.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# bootstrap_default — pinning logic per section
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _force_interactive(monkeypatch):
    """Default interactive context for the rest of the suite."""
    monkeypatch.setattr(sb.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sb.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("DATUS_DISABLE_SERVICE_BOOTSTRAP", raising=False)
    # Stop the install-pass thread from doing real work in every test —
    # individual tests that care monkey-patch back to a real function.
    monkeypatch.setattr(sb, "background_install_missing", lambda cli: None)


class TestBootstrapDefault:
    def test_skips_empty_section_silently(self):
        cfg = _make_agent()  # all sections empty
        cli = _fake_cli(cfg)
        sb.run(cli)
        cfg.set_active_dashboard.assert_not_called()
        cfg.set_active_scheduler.assert_not_called()
        cfg.set_active_semantic.assert_not_called()

    def test_skips_when_already_pinned(self):
        cfg = _make_agent(
            dashboards={"superset": MagicMock(default=False)},
            active_dash="superset",  # already pinned — bootstrap should leave alone
            default_dash="superset",
        )
        cli = _fake_cli(cfg)
        sb.run(cli)
        cfg.set_active_dashboard.assert_not_called()

    def test_pins_single_entry_implicitly(self):
        cfg = _make_agent(
            dashboards={"superset": MagicMock(default=False)},
            default_dash="superset",  # single-entry shortcut returns the only key
        )
        cli = _fake_cli(cfg)
        sb.run(cli)
        cfg.set_active_dashboard.assert_called_once_with("superset")

    def test_pins_yaml_default_flag(self):
        """When ``default: true`` is set on one of multiple entries, the
        bootstrap should pin that one without asking."""
        cfg = _make_agent(
            dashboards={"superset": MagicMock(default=False), "grafana": MagicMock(default=True)},
            default_dash="grafana",
        )
        cli = _fake_cli(cfg)
        sb.run(cli)
        cfg.set_active_dashboard.assert_called_once_with("grafana")

    def test_picker_invoked_when_ambiguous(self):
        cfg = _make_agent(
            dashboards={"superset": MagicMock(default=False), "grafana": MagicMock(default=False)},
            default_dash=None,  # multiple entries, no default
        )
        cli = _fake_cli(cfg)
        with patch.object(sb, "quick_pick_service", return_value="grafana") as picker:
            sb.run(cli)
        picker.assert_called_once()
        cfg.set_active_dashboard.assert_called_once_with("grafana")

    def test_picker_cancel_leaves_unpinned(self):
        cfg = _make_agent(
            dashboards={"a": MagicMock(default=False), "b": MagicMock(default=False)},
            default_dash=None,
        )
        cli = _fake_cli(cfg)
        with patch.object(sb, "quick_pick_service", return_value=None):
            sb.run(cli)
        cfg.set_active_dashboard.assert_not_called()

    def test_resolver_exception_does_not_abort_other_sections(self):
        """If the dashboard resolver raises (e.g. multiple ``default: true``),
        the bootstrap surfaces an error but still processes the next
        sections."""
        from datus.utils.exceptions import DatusException, ErrorCode

        cfg = _make_agent(
            dashboards={"a": MagicMock(default=True), "b": MagicMock(default=True)},
            schedulers={"airflow": {"type": "airflow"}},
            default_sched="airflow",
        )
        cfg.default_dashboard_service.side_effect = DatusException(
            ErrorCode.COMMON_CONFIG_ERROR, message="multiple default true"
        )
        cli = _fake_cli(cfg)
        sb.run(cli)
        # Dashboard pin skipped, but scheduler still pinned.
        cfg.set_active_dashboard.assert_not_called()
        cfg.set_active_scheduler.assert_called_once_with("airflow")


# ─────────────────────────────────────────────────────────────────────
# background_install_missing — adapter pip flow
# ─────────────────────────────────────────────────────────────────────


class TestBackgroundInstall:
    def test_only_processes_missing_packages(self, monkeypatch):
        # Two BI services, only one is missing. Use SimpleNamespace so
        # ``getattr(cfg, 'adapter_type', '')`` returns the literal value
        # rather than a child MagicMock.
        bi_one = SimpleNamespace(adapter_type="superset")
        bi_two = SimpleNamespace(adapter_type="grafana")
        cfg = _make_agent(
            dashboards={"superset": bi_one, "grafana": bi_two},
            default_dash="superset",  # pin already happens
        )
        cli = _fake_cli(cfg)

        installed = {("bi_platforms", "superset")}  # superset is already installed
        monkeypatch.setattr(
            sb,
            "is_adapter_installed",
            lambda section, t: (section, t) in installed,
        )
        ensure_calls: list = []

        def fake_ensure(section, adapter_type):
            ensure_calls.append((section, adapter_type))
            return sb.InstallResult(ok=True, package=f"datus-bi-{adapter_type}", import_name="x")

        monkeypatch.setattr(sb, "ensure_adapter", fake_ensure)
        monkeypatch.setattr(sb, "hot_reload_adapter", lambda *_: True)
        monkeypatch.setattr(
            sb.threading,
            "Thread",
            lambda target, args, daemon: SimpleNamespace(start=lambda: target(*args)),
        )
        # Run the actual install pass (override the autouse stub).
        monkeypatch.setattr(sb, "background_install_missing", _REAL_BACKGROUND_INSTALL)

        sb.run(cli)

        # ``superset`` filtered out, only ``grafana`` triggers an install.
        assert ensure_calls == [("bi_platforms", "grafana")]

    def test_install_failure_is_non_blocking(self, monkeypatch):
        bi = SimpleNamespace(adapter_type="ghost")
        cfg = _make_agent(
            dashboards={"ghost": bi},
            default_dash="ghost",
        )
        cli = _fake_cli(cfg)

        monkeypatch.setattr(sb, "is_adapter_installed", lambda *_: False)
        monkeypatch.setattr(
            sb,
            "ensure_adapter",
            lambda *args, **kwargs: sb.InstallResult(
                ok=False, package="datus-bi-ghost", import_name="datus_bi_ghost", error="not found"
            ),
        )
        hot_reload_called = []
        monkeypatch.setattr(sb, "hot_reload_adapter", lambda *a: hot_reload_called.append(a))
        monkeypatch.setattr(
            sb.threading,
            "Thread",
            lambda target, args, daemon: SimpleNamespace(start=lambda: target(*args)),
        )
        monkeypatch.setattr(sb, "background_install_missing", _REAL_BACKGROUND_INSTALL)

        sb.run(cli)

        # ``hot_reload_adapter`` must not be called when the install failed —
        # ``ensure_adapter`` returning ``ok=False`` is a hard stop.
        assert hot_reload_called == []

    def test_dedupes_targets(self, monkeypatch):
        """Two BI aliases of the same adapter type should only kick off
        one ``pip install``."""
        bi_a = SimpleNamespace(adapter_type="superset")
        bi_b = SimpleNamespace(adapter_type="superset")  # same adapter, different alias
        cfg = _make_agent(
            dashboards={"alias_a": bi_a, "alias_b": bi_b},
            default_dash="alias_a",
        )
        cli = _fake_cli(cfg)
        monkeypatch.setattr(sb, "is_adapter_installed", lambda *_: False)
        ensure_calls: list = []
        monkeypatch.setattr(
            sb,
            "ensure_adapter",
            lambda section, t: (
                ensure_calls.append((section, t)) or sb.InstallResult(ok=True, package=f"datus-bi-{t}", import_name="x")
            ),
        )
        monkeypatch.setattr(sb, "hot_reload_adapter", lambda *_: True)
        monkeypatch.setattr(
            sb.threading,
            "Thread",
            lambda target, args, daemon: SimpleNamespace(start=lambda: target(*args)),
        )
        monkeypatch.setattr(sb, "background_install_missing", _REAL_BACKGROUND_INSTALL)

        sb.run(cli)
        assert ensure_calls == [("bi_platforms", "superset")]


# ─────────────────────────────────────────────────────────────────────
# quick_pick_service — synchronous chooser
# ─────────────────────────────────────────────────────────────────────


class TestQuickPickService:
    def test_returns_chosen_index(self, monkeypatch):
        cli = SimpleNamespace(console=Console(file=io.StringIO(), no_color=True, force_terminal=False))
        monkeypatch.setattr("builtins.input", lambda _: "2")
        assert sb.quick_pick_service(cli, "bi_platforms", ["superset", "grafana"]) == "grafana"

    def test_returns_none_on_blank(self, monkeypatch):
        cli = SimpleNamespace(console=Console(file=io.StringIO(), no_color=True, force_terminal=False))
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert sb.quick_pick_service(cli, "schedulers", ["airflow"]) is None

    def test_returns_none_on_q(self, monkeypatch):
        cli = SimpleNamespace(console=Console(file=io.StringIO(), no_color=True, force_terminal=False))
        monkeypatch.setattr("builtins.input", lambda _: "q")
        assert sb.quick_pick_service(cli, "schedulers", ["airflow"]) is None

    def test_reprompts_on_out_of_range(self, monkeypatch):
        cli = SimpleNamespace(console=Console(file=io.StringIO(), no_color=True, force_terminal=False))
        responses = iter(["99", "1"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        assert sb.quick_pick_service(cli, "bi_platforms", ["a", "b"]) == "a"

    def test_reprompts_on_non_numeric(self, monkeypatch):
        cli = SimpleNamespace(console=Console(file=io.StringIO(), no_color=True, force_terminal=False))
        responses = iter(["abc", "2"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        assert sb.quick_pick_service(cli, "bi_platforms", ["a", "b"]) == "b"
