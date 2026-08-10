# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.package_cli`` (the interactive wizard shell).

CI-level: prompt primitives (``select_multi_choice`` / ``select_choice`` /
``confirm_prompt`` / ``prompt_input``) are patched by module path — the
same pattern as ``test_project_init.py`` — so no pty is needed. The build
itself is mocked; option plumbing is asserted on the captured
``PackageOptions``.
"""

from pathlib import Path

import pytest

from datus.cli import package_builder as pb
from datus.cli import package_cli


@pytest.fixture
def raw_config(monkeypatch):
    raw = {"agentic_nodes": {"helper": {"agent_description": "Helper"}}}
    monkeypatch.setattr(pb, "load_raw_agent_config", lambda: raw)
    monkeypatch.setattr(pb, "resolve_effective_project_name", lambda root, raw_cfg: "proj")
    return raw


@pytest.fixture
def captured_build(monkeypatch):
    captured = {}

    def fake_build(options):
        captured["options"] = options
        return pb.PackageResult(ok=True, zip_path="/tmp/proj.zip", file_count=3, total_bytes=1024)

    monkeypatch.setattr(pb, "build_package", fake_build)
    return captured


def _fail_prompt(*_args, **_kwargs):
    pytest.fail("prompt primitives must not be called in this scenario")


def _patch_prompts(monkeypatch, **replacements):
    for name in ("select_multi_choice", "select_choice", "confirm_prompt", "prompt_input"):
        monkeypatch.setattr(package_cli, name, replacements.get(name, _fail_prompt))


class TestNonInteractivePaths:
    def test_yes_packages_defaults_without_prompts(self, raw_config, captured_build, monkeypatch):
        _patch_prompts(monkeypatch)  # every prompt is a failure
        assert package_cli.run_package_command(["--yes"]) == 0
        options = captured_build["options"]
        assert options.assume_yes is True
        assert options.subagents is None and options.skills is None
        assert options.metrics is None and options.subjects is None and options.reports is None
        assert options.output is None and options.report_dist is None

    def test_non_tty_without_yes_exits_2(self, raw_config, captured_build, monkeypatch):
        monkeypatch.setattr(package_cli, "_is_interactive", lambda: False)
        assert package_cli.run_package_command([]) == 2
        assert "options" not in captured_build

    def test_non_tty_with_yes_succeeds(self, raw_config, captured_build, monkeypatch):
        monkeypatch.setattr(package_cli, "_is_interactive", lambda: False)
        assert package_cli.run_package_command(["-y"]) == 0

    def test_missing_config_exits_3(self, monkeypatch, capsys):
        monkeypatch.setattr(pb, "load_raw_agent_config", lambda: None)
        assert package_cli.run_package_command(["-y"]) == 3
        assert "No agent configuration" in capsys.readouterr().out

    def test_unknown_flag_is_usage_error(self, raw_config):
        with pytest.raises(SystemExit) as excinfo:
            package_cli.run_package_command(["--bogus"])
        assert excinfo.value.code == 2

    def test_build_failure_exits_1(self, raw_config, monkeypatch, capsys):
        monkeypatch.setattr(pb, "build_package", lambda _o: pb.PackageResult(ok=False, error="boom"))
        assert package_cli.run_package_command(["-y"]) == 1
        assert "boom" in capsys.readouterr().out

    def test_secret_findings_reported(self, raw_config, monkeypatch, capsys):
        result = pb.PackageResult(
            ok=False,
            error="secret scan failed",
            secret_findings=[pb.SecretFinding(arcname="conf/agent.yml", locator="x.password", kind="plaintext")],
        )
        monkeypatch.setattr(pb, "build_package", lambda _o: result)
        assert package_cli.run_package_command(["-y"]) == 1
        out = capsys.readouterr().out
        assert "secret detected" in out and "conf/agent.yml" in out


class TestWizard:
    def _patch_enumerations(self, monkeypatch, *, dashboards=()):
        monkeypatch.setattr(pb, "list_subagents", lambda raw: {"helper": "Helper"})
        monkeypatch.setattr(pb, "list_skills", lambda root: {"sql-skill": Path("/global/sql-skill")})
        monkeypatch.setattr(pb, "list_metric_datasources", lambda root: ["ds1"])
        monkeypatch.setattr(pb, "list_subject_paths", lambda root, raw, project: {"sales": "sales (1 metrics)"})
        monkeypatch.setattr(pb, "list_packageable_plugins", lambda root: {"alpha": "alpha — activated"})
        monkeypatch.setattr(
            pb,
            "list_artifact_slugs",
            lambda root, kind: ["r1"] if kind == "report" else list(dashboards),
        )
        monkeypatch.setattr(package_cli, "_is_interactive", lambda: True)

    def test_happy_path_collects_all_steps(self, raw_config, captured_build, monkeypatch):
        self._patch_enumerations(monkeypatch)
        multi_calls = []

        def fake_multi(console, choices, default_selected=None, **_kwargs):
            multi_calls.append(list(choices))
            return list(choices)  # keep everything

        _patch_prompts(
            monkeypatch,
            select_multi_choice=fake_multi,
            # "all files?" → yes; "build now?" → yes
            confirm_prompt=_side_effects([True, True]),
            select_choice=_side_effects(["cdn"]),
            prompt_input=_side_effects([""]),  # output path: accept default
        )
        assert package_cli.run_package_command([]) == 0

        options = captured_build["options"]
        assert options.subagents == ("helper",)
        assert options.skills == ("sql-skill",)
        assert options.metrics == ("ds1",)
        assert options.subjects == ("sales",)
        assert options.plugins == ("alpha",)
        assert options.reports == ("r1",)
        assert options.dashboards == ()  # empty category skipped silently
        assert options.include == () and options.exclude == ()
        assert options.output == Path.cwd() / "proj.zip"
        assert options.report_dist is None
        assert options.assume_yes is False
        # Six populated categories → six multi-select screens.
        assert len(multi_calls) == 6

    def test_empty_selection_needs_confirmation_then_retries(self, raw_config, captured_build, monkeypatch):
        self._patch_enumerations(monkeypatch)
        _patch_prompts(
            monkeypatch,
            # 1st subagent screen: empty → decline "package none" → retry → select.
            select_multi_choice=_side_effects([[], ["helper"], ["sql-skill"], ["ds1"], ["sales"], ["alpha"], ["r1"]]),
            # all-files yes; "no subagents?" no; build-now yes
            confirm_prompt=_side_effects([True, False, True]),
            select_choice=_side_effects(["cdn"]),
            prompt_input=_side_effects([""]),
        )
        assert package_cli.run_package_command([]) == 0
        assert captured_build["options"].subagents == ("helper",)

    def test_empty_selection_confirmed_packages_none(self, raw_config, captured_build, monkeypatch):
        self._patch_enumerations(monkeypatch)
        _patch_prompts(
            monkeypatch,
            select_multi_choice=_side_effects([[], ["sql-skill"], ["ds1"], ["sales"], ["alpha"], []]),
            # all-files yes; "no subagents?" yes; "no reports?" yes; build-now yes
            confirm_prompt=_side_effects([True, True, True, True]),
            prompt_input=_side_effects([""]),
        )
        assert package_cli.run_package_command([]) == 0
        options = captured_build["options"]
        assert options.subagents == ()
        assert options.reports == ()
        # No reports selected → the dist step is skipped entirely.
        assert options.report_dist is None

    def test_summary_decline_aborts_without_building(self, raw_config, captured_build, monkeypatch):
        self._patch_enumerations(monkeypatch)
        _patch_prompts(
            monkeypatch,
            select_multi_choice=lambda console, choices, default_selected=None, **_: list(choices),
            confirm_prompt=_side_effects([True, False]),  # all-files yes; build-now NO
            select_choice=_side_effects(["cdn"]),
            prompt_input=_side_effects([""]),
        )
        # Declining the summary is a cancellation, not a failure — 130 is the
        # shell convention for "interrupted", same as Ctrl+C.
        assert package_cli.run_package_command([]) == 130
        assert "options" not in captured_build

    def test_ctrl_c_in_a_wizard_step_aborts_without_building(self, raw_config, captured_build, monkeypatch):
        """Prompts opt into ``cancellable=True`` so Ctrl+C raises instead of
        returning a plausible answer (``[]`` reads as "deselect everything",
        and a default "yes" would have started the build)."""
        self._patch_enumerations(monkeypatch)

        def interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt

        _patch_prompts(
            monkeypatch,
            select_multi_choice=interrupt,
            confirm_prompt=_side_effects([True]),
            prompt_input=_side_effects([""]),
        )
        assert package_cli.run_package_command([]) == 130
        assert "options" not in captured_build

    def test_ctrl_c_during_build_discards_the_partial_zip(self, raw_config, monkeypatch, tmp_path, capsys):
        """A truncated archive that looks complete is worse than none."""
        partial = tmp_path / "half.zip"
        partial.write_bytes(b"PK\x03\x04 truncated")

        def interrupted_build(_options):
            raise KeyboardInterrupt

        monkeypatch.setattr(pb, "build_package", interrupted_build)
        self._patch_enumerations(monkeypatch)
        _patch_prompts(
            monkeypatch,
            select_multi_choice=lambda console, choices, default_selected=None, **_: list(choices),
            # overwrite the existing file? / all files? / build now?
            confirm_prompt=_side_effects([True, True, True]),
            select_choice=_side_effects(["cdn"]),
            prompt_input=_side_effects([str(partial)]),
        )
        assert package_cli.run_package_command([]) == 130
        assert not partial.exists()
        assert "partially written" in capsys.readouterr().out

    def test_output_path_validation_reprompts(self, raw_config, captured_build, monkeypatch):
        self._patch_enumerations(monkeypatch)
        _patch_prompts(
            monkeypatch,
            select_multi_choice=lambda console, choices, default_selected=None, **_: list(choices),
            confirm_prompt=_side_effects([True, True]),
            select_choice=_side_effects(["cdn"]),
            # First answer lacks .zip → warned + re-asked; second accepted.
            prompt_input=_side_effects(["out.tar", "good.zip"]),
        )
        assert package_cli.run_package_command([]) == 0
        assert captured_build["options"].output == Path("good.zip")

    def test_custom_file_scope_patterns_validated(self, raw_config, captured_build, monkeypatch):
        self._patch_enumerations(monkeypatch)
        _patch_prompts(
            monkeypatch,
            select_multi_choice=lambda console, choices, default_selected=None, **_: list(choices),
            confirm_prompt=_side_effects([False, True]),  # NOT all files; build-now yes
            select_choice=_side_effects(["cdn"]),
            # output; include (bad → retry → good); exclude
            prompt_input=_side_effects(["", "[bad", r"^knowledge/, ^docs/", r"\.tmp$"]),
        )
        assert package_cli.run_package_command([]) == 0
        options = captured_build["options"]
        assert options.include == ("^knowledge/", "^docs/")
        assert options.exclude == (r"\.tmp$",)

    def test_report_dist_step_validates_directory(self, raw_config, captured_build, monkeypatch, tmp_path):
        self._patch_enumerations(monkeypatch)
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.css").write_text("css", encoding="utf-8")
        (dist / "index.umd.js").write_text("js", encoding="utf-8")
        _patch_prompts(
            monkeypatch,
            select_multi_choice=lambda console, choices, default_selected=None, **_: list(choices),
            confirm_prompt=_side_effects([True, True]),
            select_choice=_side_effects(["dist"]),
            # output; dist dir (bad → retry → good)
            prompt_input=_side_effects(["", str(tmp_path / "nope"), str(dist)]),
        )
        assert package_cli.run_package_command([]) == 0
        assert captured_build["options"].report_dist == dist.resolve()


def _side_effects(values):
    """Return a callable that pops the next canned answer per invocation."""
    remaining = list(values)

    def _next(*_args, **_kwargs):
        if not remaining:
            pytest.fail("prompt called more times than expected")
        return remaining.pop(0)

    return _next


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
