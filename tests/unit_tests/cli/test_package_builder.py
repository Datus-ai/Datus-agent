# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.package_builder`` (project → self-contained zip).

CI-level: no network, no real ``~/.datus`` — HOME is redirected into
``tmp_path`` and dependency enumeration is mocked. Fixture projects are
built on disk and results are verified by re-opening the produced zip
(cross-component: the consumer sees what the packer wrote, not mock echoes).
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Never

import pytest
import yaml

from datus.cli import package_builder as pb

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

_PLAINTEXT_KEY = "sk-plaintextsecret1234567890abcdef"


def _agent_yaml(fake_home: Path) -> dict:
    """Kitchen-sink raw config covering every secret-bearing section."""
    return {
        "agent": {
            "home": str(fake_home / ".datus"),
            "providers": {
                "openai": {"api_key": _PLAINTEXT_KEY},
                "claude": {"api_key": "${ANTHROPIC_API_KEY}"},
            },
            "models": {
                "internal": {
                    "type": "openai",
                    "api_key": "plain-model-key",
                    "default_headers": {"Authorization": "Bearer abc123token"},
                }
            },
            "services": {
                "datasources": {
                    "sales_db": {
                        "type": "starrocks",
                        "host": "${SR_HOST}",
                        "port": "${SR_PORT:-9030}",
                        "username": "admin",
                        "password": "hunter2",
                        "private_key": "-----BEGIN PRIVATE KEY-----\nxyz\n-----END PRIVATE KEY-----",
                    },
                    "pg_main": {"type": "postgres", "uri": "postgresql://svc:p4ss@db.example.com/warehouse"},
                    "local_lite": {"type": "sqlite", "uri": "sqlite:///data.db"},
                },
                "bi_platforms": {
                    "superset": {
                        "type": "superset",
                        "username": "admin",
                        "password": "admin",
                        "extra": {"provider": "db"},
                    }
                },
                "schedulers": {"airflow": {"type": "airflow", "password": "${AIRFLOW_PASSWORD}"}},
                "mcp_servers": {"jira": {"headers": {"Authorization": "Bearer tok"}, "env": {"JIRA_TOKEN": "t0k"}}},
            },
            "document": {"tavily_api_key": "tvly-plain"},
            "custom_future_section": {"foo": "bar", "nested": {"keep": "me"}},
            "agentic_nodes": {
                "sales_helper": {
                    "agent_description": "Sales Q&A",
                    "system_prompt": "sales_helper",
                    "prompt_version": "1.0",
                },
                "ops_helper": {"agent_description": "Ops Q&A"},
            },
        }
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A packaged-project fixture rooted in an isolated CWD + HOME."""
    fake_home = tmp_path / "fakehome"
    root = tmp_path / "proj"
    monkeypatch.setenv("HOME", str(fake_home))
    # ``Path.home()`` reads USERPROFILE on Windows — isolate both.
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.chdir(root_prepare(root, fake_home))
    monkeypatch.setattr(
        pb,
        "enumerate_datus_packages",
        lambda: [pb.DatusPackage(name="datus-agent", version="0.9.9", editable=False)],
    )
    return root


def root_prepare(root: Path, fake_home: Path) -> Path:
    (root / "conf").mkdir(parents=True)
    (root / "conf" / "agent.yml").write_text(yaml.safe_dump(_agent_yaml(fake_home), sort_keys=False), encoding="utf-8")
    (root / ".datus").mkdir()
    (root / ".datus" / "config.yml").write_text(
        "project_name: fixture_proj\ndefault_datasource: sales_db\n", encoding="utf-8"
    )

    # Generic project files.
    (root / "knowledge").mkdir()
    (root / "knowledge" / "notes.md").write_text("domain knowledge", encoding="utf-8")
    deep = root / "docs" / "guides" / "internal"
    deep.mkdir(parents=True)
    (deep / "deep.md").write_text("deep file", encoding="utf-8")

    # Runtime state that must never ship.
    for dirname in ("sessions", "data", "logs", "run", "cache", "save", "trajectory", "output_v1", ".venv", ".git"):
        (root / dirname).mkdir(exist_ok=True)
        (root / dirname / "junk.txt").write_text("junk", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "m.pyc").write_text("x", encoding="utf-8")
    (root / ".env").write_text("SECRET=leak", encoding="utf-8")
    (root / "db.duckdb.wal").write_text("wal", encoding="utf-8")
    # Editor swap/backup junk — transient files that vanish mid-build.
    (root / "conf" / ".agent.yml.swp").write_bytes(b"vim swap")
    (root / "knowledge" / "notes.md~").write_text("backup", encoding="utf-8")
    (root / ".DS_Store").write_bytes(b"\x00\x01")
    # macOS litter: AppleDouble xattr sidecars (non-APFS volumes) and
    # Archive-Utility leftovers from a previous unzip.
    (root / "knowledge" / "._notes.md").write_bytes(b"\x00\x05\x16\x07AppleDouble")
    (root / "__MACOSX").mkdir()
    (root / "__MACOSX" / "junk.txt").write_text("litter", encoding="utf-8")
    # REPL command history lands at {home}/history under home: . — user
    # activity that must never ship. A nested file named "history" is fine.
    (root / "history").write_text("# 2026-08-07\n+select * from secret_table\n", encoding="utf-8")
    (root / "docs" / "history").write_text("legit project doc", encoding="utf-8")
    (root / ".datus" / "memory").mkdir()
    (root / ".datus" / "memory" / "private.md").write_text("private memory", encoding="utf-8")
    (root / ".datus" / "plans").mkdir()
    (root / ".datus" / "plans" / "draft.md").write_text("draft", encoding="utf-8")

    # Skills: one project, one global (same name to test precedence) + one global-only.
    for base, marker in ((root / ".datus" / "skills", "project"), (fake_home / ".datus" / "skills", "global")):
        skill = base / "shared-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"# {marker}", encoding="utf-8")
    only_global = fake_home / ".datus" / "skills" / "global-only"
    only_global.mkdir(parents=True)
    (only_global / "SKILL.md").write_text("# global only", encoding="utf-8")

    # Subagent template in the source home.
    template_dir = fake_home / ".datus" / "template"
    template_dir.mkdir(parents=True)
    (template_dir / "sales_helper_system_1.0.j2").write_text("prompt {{x}}", encoding="utf-8")

    # Metrics for two datasources. Metric docs carry their subject as a
    # ``subject_tree:`` tag — the same shape gen-metrics writes.
    for ds, subject in (("sales_db", "sales"), ("pg_main", "ops")):
        ds_dir = root / "subject" / "semantic_models" / ds
        (ds_dir / "metrics").mkdir(parents=True)
        (ds_dir / "orders.yml").write_text("data_source:\n  name: orders\n", encoding="utf-8")
        # Two docs per file, in different subject areas, with the tag under
        # ``locked_metadata`` — the shape gen_metrics actually writes.
        (ds_dir / "metrics" / "orders_metrics.yml").write_text(
            f"metric:\n  name: gmv_{ds}\n  locked_metadata:\n    tags:\n"
            f"    - 'subject_tree: {subject}/revenue'\n"
            f"---\nmetric:\n  name: shared_{ds}\n  locked_metadata:\n    tags:\n"
            f"    - 'subject_tree: shared/misc'\n",
            encoding="utf-8",
        )
    # Reference-SQL summaries: what the receiver re-indexes, each tagged with
    # the subject tree it belongs to.
    summaries = root / "subject" / "sql_summaries"
    summaries.mkdir(parents=True)
    for name, subject in (("q_sales", "sales/orders"), ("q_ops", "ops/pipeline")):
        (summaries / f"{name}.yaml").write_text(
            f'id: {name}\nname: "{name}"\nsql: "SELECT 1"\nsummary: "s"\n'
            f'search_text: "t"\nsubject_tree: "{subject}"\ntags: ""\n',
            encoding="utf-8",
        )

    # Raw corpus + DDL: ordinary project content, shipped by the generic walk.
    (root / "reference_sql").mkdir()
    (root / "reference_sql" / "queries.sql").write_text("SELECT 1;", encoding="utf-8")
    (root / "migrations").mkdir()
    (root / "migrations" / "001_init.sql").write_text("CREATE TABLE t(a INT);", encoding="utf-8")

    # One report + one dashboard artifact.
    for kind_dir, slug in (("reports", "daily_gmv"), ("dashboards", "ops_view")):
        art = root / kind_dir / slug
        for sub in ("analysis", "queries", "render"):
            (art / sub).mkdir(parents=True)
        (art / "manifest.json").write_text(json.dumps({"slug": slug, "kind": kind_dir[:-1]}), encoding="utf-8")
        (art / "analysis" / "intent.md").write_text("## intent", encoding="utf-8")
        (art / "render" / "app.jsx").write_text("export default 1", encoding="utf-8")
        (art / "render" / "scratch.tmp").write_text("stray", encoding="utf-8")
    (root / "reports" / "daily_gmv" / "queries" / "q.sql").write_text("SELECT 1", encoding="utf-8")
    return root


def _build(root: Path, **kwargs) -> pb.PackageResult:
    kwargs.setdefault("assume_yes", True)
    return pb.build_package(pb.PackageOptions(root=root, **kwargs))


def _namelist(result: pb.PackageResult) -> set:
    assert result.ok is True, result.error
    with zipfile.ZipFile(result.zip_path) as zf:
        return set(zf.namelist())


def _member(result: pb.PackageResult, name: str) -> bytes:
    with zipfile.ZipFile(result.zip_path) as zf:
        return zf.read(name)


# --------------------------------------------------------------------------- #
# Collection & filtering                                                      #
# --------------------------------------------------------------------------- #


class TestCollection:
    def test_builtin_exclusions(self, project):
        names = _namelist(_build(project))
        for banned in (
            "sessions/junk.txt",
            "data/junk.txt",
            "logs/junk.txt",
            "run/junk.txt",
            "cache/junk.txt",
            "save/junk.txt",
            "trajectory/junk.txt",
            "output_v1/junk.txt",
            ".venv/junk.txt",
            ".git/junk.txt",
            "__pycache__/m.pyc",
            ".env",
            "db.duckdb.wal",
            "conf/.agent.yml.swp",
            "knowledge/notes.md~",
            ".DS_Store",
            "knowledge/._notes.md",
            "__MACOSX/junk.txt",
            "history",
            ".datus/memory/private.md",
            ".datus/plans/draft.md",
        ):
            assert banned not in names, banned
        assert "knowledge/notes.md" in names
        assert "docs/guides/internal/deep.md" in names
        # Exclusion is top-level only — a project doc named "history" ships.
        assert "docs/history" in names
        # Generated files present; the source agent.yml was never copied.
        assert {
            "conf/agent.yml",
            ".datus/config.yml",
            "requirements.txt",
            "README.md",
            "package_manifest.json",
        } <= names

    def test_include_restricts_generic_walk(self, project):
        names = _namelist(_build(project, include=(r"^knowledge/",)))
        assert "knowledge/notes.md" in names
        assert "docs/guides/internal/deep.md" not in names
        # Selector-owned + generated content is not subject to user includes.
        assert "conf/agent.yml" in names
        assert "reports/daily_gmv/manifest.json" in names

    def test_exclude_drops_matches(self, project):
        names = _namelist(_build(project, exclude=(r"deep\.md$",)))
        assert "docs/guides/internal/deep.md" not in names
        assert "knowledge/notes.md" in names

    def test_invalid_regex_fails_cleanly(self, project):
        result = _build(project, include=("[unclosed",))
        assert not result.ok
        assert "invalid" in result.error and "[unclosed" in result.error

    def test_output_zip_not_packaged_into_itself(self, project):
        first = _build(project)
        assert first.ok is True, first.error
        second = _build(project)
        names = _namelist(second)
        assert not any(name.endswith(".zip") for name in names)

    def test_file_vanishing_mid_build_is_skipped_with_warning(self, project, monkeypatch):
        """A file collected by the walk but deleted before finalize (editor
        swap files, concurrent cleanup) must not fail the whole build."""
        original = pb.collect_project_files

        def with_ghost(*args, **kwargs):
            entries, warns = original(*args, **kwargs)
            entries.append(pb.StagedEntry(arcname="ghost.txt", source=project / "nonexistent.txt"))
            return entries, warns

        monkeypatch.setattr(pb, "collect_project_files", with_ghost)
        result = _build(project)
        assert result.ok is True, result.error
        assert "ghost.txt" not in _namelist(result)
        assert any("ghost.txt" in warning and "vanished" in warning for warning in result.warnings)

    @pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs Developer Mode or admin on Windows")
    def test_symlink_escaping_root_dropped(self, project, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        (project / "knowledge" / "link.txt").symlink_to(outside)
        result = _build(project)
        names = _namelist(result)
        assert "knowledge/link.txt" not in names
        assert any("escaping project root" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# agent.yml generation                                                        #
# --------------------------------------------------------------------------- #


class TestAgentYmlGeneration:
    def _generated(self, result: pb.PackageResult) -> dict:
        return yaml.safe_load(_member(result, "conf/agent.yml"))["agent"]

    def test_home_and_project_name_pinned(self, project):
        agent = self._generated(_build(project))
        assert agent["home"] == "."
        assert agent["project_name"] == "fixture_proj"
        assert "project_root" not in agent

    def test_all_secrets_replaced_with_placeholders(self, project):
        result = _build(project)
        text = _member(result, "conf/agent.yml").decode("utf-8")
        for secret in (_PLAINTEXT_KEY, "hunter2", "plain-model-key", "abc123token", "p4ss", "t0k", "tvly-plain"):
            assert secret not in text, secret
        agent = self._generated(result)
        assert agent["providers"]["openai"]["api_key"] == "${OPENAI_API_KEY}"
        assert agent["providers"]["claude"]["api_key"] == "${ANTHROPIC_API_KEY}"
        assert agent["services"]["schedulers"]["airflow"]["password"] == "${AIRFLOW_PASSWORD}"
        vars_seen = {binding.var for binding in result.env_vars}
        assert {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"} <= vars_seen
        preexisting = {binding.var for binding in result.env_vars if binding.preexisting}
        assert "ANTHROPIC_API_KEY" in preexisting and "AIRFLOW_PASSWORD" in preexisting

    def test_non_secret_placeholders_harvested_for_readme(self, project):
        """Placeholders on NON-secret fields (host/port/…) never pass through
        the secret rewriters, but the receiver still must export them — they
        must land in the env bindings and the README table."""
        result = _build(project)
        by_var = {binding.var: binding for binding in result.env_vars}
        assert "SR_HOST" in by_var and by_var["SR_HOST"].preexisting is True
        assert "SR_PORT" in by_var  # ${VAR:-default} form is harvested too
        readme = _member(result, "README.md").decode("utf-8")
        assert "SR_HOST" in readme

    def test_uri_password_component_rewritten(self, project):
        agent = self._generated(_build(project))
        uri = agent["services"]["datasources"]["pg_main"]["uri"]
        assert uri.startswith("postgresql://svc:${DATUS_DS_PG_MAIN_URI_PASSWORD}@db.example.com")
        assert agent["services"]["datasources"]["local_lite"]["uri"] == "sqlite:///data.db"

    def test_passwordless_uri_left_untouched(self, project):
        alloc = pb._PlaceholderAllocator()
        container = {"uri": "postgresql://svc@db.example.com/warehouse"}
        pb._rewrite_uri_password(container, "uri", "DATUS_DS_X_URI_PASSWORD", "x.uri", alloc)
        # No password component -> host/port/database must survive verbatim.
        assert container["uri"] == "postgresql://svc@db.example.com/warehouse"
        assert alloc.bindings == []

    def test_unparseable_uri_with_password_component_fully_replaced(self, project):
        alloc = pb._PlaceholderAllocator()
        container = {"uri": "weird-scheme://user:p4ss@[bad host/db"}
        pb._rewrite_uri_password(container, "uri", "DATUS_DS_X_URI_PASSWORD", "x.uri", alloc)
        # The whole value goes, not just the password: a partial redaction of an
        # unparseable URI could still carry the secret in what it left behind.
        assert container["uri"] == "${DATUS_DS_X_URI_PASSWORD}"
        assert [b.var for b in alloc.bindings] == ["DATUS_DS_X_URI_PASSWORD"]

    def test_non_secret_extras_untouched(self, project):
        agent = self._generated(_build(project))
        assert agent["services"]["bi_platforms"]["superset"]["extra"]["provider"] == "db"

    def test_unknown_sections_pass_through(self, project):
        agent = self._generated(_build(project))
        assert agent["custom_future_section"] == {"foo": "bar", "nested": {"keep": "me"}}

    def test_subagent_filtering(self, project):
        result = _build(project, subagents=("sales_helper",))
        agent = self._generated(result)
        assert set(agent["agentic_nodes"]) == {"sales_helper"}

    def test_same_value_reuses_var_and_collision_suffixes(self, tmp_path, monkeypatch):
        alloc = pb._PlaceholderAllocator()
        first = alloc.allocate("samesecret", "DATUS_X_KEY", "a.key")
        second = alloc.allocate("samesecret", "DATUS_Y_KEY", "b.key")
        assert first == second == "${DATUS_X_KEY}"
        third = alloc.allocate("othersecret", "DATUS_X_KEY", "c.key")
        assert third == "${DATUS_X_KEY_2}"


# --------------------------------------------------------------------------- #
# Component selectors                                                         #
# --------------------------------------------------------------------------- #


class TestSelectors:
    def test_unknown_subagent_fails(self, project):
        result = _build(project, subagents=("ghost",))
        assert not result.ok and "unknown subagent" in result.error

    def test_subagent_template_staged(self, project):
        names = _namelist(_build(project, subagents=("sales_helper",)))
        assert "template/sales_helper_system_1.0.j2" in names

    def test_skills_project_wins_and_global_materialized(self, project):
        result = _build(project)
        names = _namelist(result)
        assert ".datus/skills/shared-skill/SKILL.md" in names
        assert ".datus/skills/global-only/SKILL.md" in names
        assert _member(result, ".datus/skills/shared-skill/SKILL.md") == b"# project"

    def test_unknown_skill_fails(self, project):
        result = _build(project, skills=("nope",))
        assert not result.ok and "unknown skill" in result.error

    def test_metrics_selection_and_rebuild_script(self, project):
        result = _build(project, metrics=("sales_db",))
        names = _namelist(result)
        assert "subject/semantic_models/sales_db/orders.yml" in names
        assert "subject/semantic_models/pg_main/orders.yml" not in names
        script = _member(result, "scripts/rebuild_kb.sh").decode("utf-8")
        semantic_pos = script.index("--components semantic_model")
        metrics_pos = script.index("--components metrics")
        assert semantic_pos < metrics_pos
        assert '--semantic_yaml "subject/semantic_models/sales_db/metrics/orders_metrics.yml"' in script
        assert "-y" in script and "--datasource sales_db" in script
        # The default ``check`` strategy ingests nothing — the script must
        # pin strategies: first semantic call overwrites (fresh store),
        # metrics calls are incremental.
        assert "--components semantic_model" in script and "--kb_update_strategy overwrite" in script
        assert script.count("--kb_update_strategy check") == 0
        metrics_line = next(line for line in script.splitlines() if "--components metrics" in line)
        assert "--kb_update_strategy incremental" in metrics_line

    def test_rebuild_script_multi_datasource_truncates_once(self, project):
        script = _member(_build(project), "scripts/rebuild_kb.sh").decode("utf-8")
        semantic_metric = [
            line
            for line in script.splitlines()
            if "--components semantic_model" in line or "--components metrics" in line
        ]
        # Only ONE overwrite across the semantic/metric calls — a second one
        # would wipe the first datasource's freshly built entries.
        assert sum("--kb_update_strategy overwrite" in line for line in semantic_metric) == 1
        assert sum("--kb_update_strategy incremental" in line for line in semantic_metric) == 3

    def test_reference_sql_rebuild_uses_summaries_not_the_llm(self, project):
        script = _member(_build(project), "scripts/rebuild_kb.sh").decode("utf-8")
        line = next(line for line in script.splitlines() if "--components reference_sql" in line)
        # Re-index the packaged summaries verbatim: no --sql_dir, no LLM spend.
        assert "--from_summaries" in line and "--sql_dir" not in line
        assert "--datasource sales_db" in line  # the project's default pin

    def test_appledouble_sidecars_never_reach_selectors(self, project):
        """AppleDouble ``._*`` files (macOS xattr sidecars on SMB/FAT
        volumes) must be invisible to every selector — the metrics rglob
        would otherwise classify ``._orders.yml`` as a semantic-model file
        and write it into rebuild_kb.sh."""
        ds_dir = project / "subject" / "semantic_models" / "sales_db"
        (ds_dir / "._orders.yml").write_bytes(b"\x00\x05\x16\x07AppleDouble")
        (project / ".datus" / "skills" / "shared-skill" / "._SKILL.md").write_bytes(b"\x00\x05")
        (project / "reports" / "daily_gmv" / "render" / "._app.jsx").write_bytes(b"\x00\x05")

        result = _build(project)
        names = _namelist(result)
        assert not any("._" in name for name in names), [n for n in names if "._" in n]
        script = _member(result, "scripts/rebuild_kb.sh").decode("utf-8")
        assert "._orders" not in script

    def test_no_metrics_no_rebuild_script(self, project):
        names = _namelist(_build(project, metrics=(), subjects=()))
        assert "scripts/rebuild_kb.sh" not in names
        assert not any(name.startswith("subject/semantic_models/") for name in names)

    def test_subject_menu_is_a_two_level_tree(self, project):
        """The menu is the subject tree down to two levels — root-only choices
        select far too much (baisheng: 61 paths under 3 roots). Deeper levels
        are folded into their depth-2 parent so the menu stays readable."""
        paths = pb.list_subject_paths(project, pb.load_raw_agent_config() or {}, "fixture_proj")
        # 'shared' comes from the second metric doc in each metrics file.
        assert set(paths) == {
            "sales",
            "sales/revenue",
            "sales/orders",
            "ops",
            "ops/revenue",
            "ops/pipeline",
            "shared",
            "shared/misc",
        }
        # Roots stay in the menu so the tree is connected; children are indented.
        assert not paths["sales"].startswith(" ")
        assert paths["sales/revenue"].strip().startswith("└ revenue")
        # Counts roll up: a root carries everything beneath it.
        assert "1 metrics" in paths["sales"] and "1 reference SQL" in paths["sales"]
        assert "1 metrics" in paths["sales/revenue"] and "reference SQL" not in paths["sales/revenue"]

    def test_second_level_selection_narrows_further_than_the_root(self, project):
        """The whole point of the two-level menu: 'sales' takes both the
        revenue metric and the orders summary, 'sales/revenue' takes only
        the metric."""
        names = _namelist(_build(project, subjects=("sales/revenue",)))
        assert "subject/semantic_models/sales_db/metrics/orders_metrics.yml" in names
        assert "subject/sql_summaries/q_sales.yaml" not in names  # tagged sales/orders

    def test_subject_selection_gates_metrics_and_summaries(self, project):
        result = _build(project, subjects=("sales",))
        names = _namelist(result)
        # sales side travels; ops side does not.
        assert "subject/sql_summaries/q_sales.yaml" in names
        assert "subject/sql_summaries/q_ops.yaml" not in names
        assert "subject/semantic_models/sales_db/metrics/orders_metrics.yml" in names
        assert "subject/semantic_models/pg_main/metrics/orders_metrics.yml" not in names
        # Semantic-model docs are table definitions, never subject-scoped.
        assert "subject/semantic_models/pg_main/orders.yml" in names

    def test_raw_corpus_and_ddl_ship_as_project_content(self, project):
        """Only the summaries are subject-gated; the .sql sources travel as
        ordinary files so nothing is silently dropped."""
        names = _namelist(_build(project, subjects=()))
        assert "reference_sql/queries.sql" in names
        assert "migrations/001_init.sql" in names
        assert not any(name.startswith("subject/sql_summaries/") for name in names)

    def test_result_reports_what_the_selection_produced(self, project):
        """Counting zip entries by hand is error-prone (``unzip -l`` wraps
        long/CJK names), so the result must state the outcome itself."""
        result = _build(project, subjects=("sales",))
        assert result.selections["subjects"] == ["sales"]
        assert result.selections["reference_sql_entries"] == 1
        assert result.selections["reference_sql_entries"] == sum(
            1 for name in _namelist(result) if name.startswith("subject/sql_summaries/")
        )

    def test_metric_filtering_is_per_document_not_per_file(self, project):
        """One metric file spans several subject areas (baisheng: 22 metrics
        across 4), so file-level filtering would be all-or-nothing — exactly
        what selecting a subject is meant to avoid."""
        result = _build(project, subjects=("sales",))
        body = _member(result, "subject/semantic_models/sales_db/metrics/orders_metrics.yml").decode("utf-8")
        assert "gmv_sales_db" in body  # under the selected subject
        assert "shared_sales_db" not in body  # under an unselected one
        # The pg_main file has no document under 'sales' at all — whole file out.
        assert "subject/semantic_models/pg_main/metrics/orders_metrics.yml" not in _namelist(result)

    def test_metric_tag_read_from_locked_metadata(self, project):
        """gen_metrics writes the tag under metric.locked_metadata.tags; reading
        only metric.tags made every generated metric look untagged, which
        disabled subject filtering entirely."""
        path = project / "subject" / "semantic_models" / "sales_db" / "metrics" / "orders_metrics.yml"
        assert set(pb._metric_doc_subjects(path)) == {"sales/revenue", "shared/misc"}

    def test_untagged_metrics_ship_with_a_warning(self, project):
        """An untagged metric file matches no subject and would drop out of
        every filtered package — taking its rebuild line with it."""
        untagged = project / "subject" / "semantic_models" / "sales_db" / "metrics" / "legacy_metrics.yml"
        untagged.write_text("metric:\n  name: legacy_gmv\n", encoding="utf-8")

        result = _build(project, subjects=("ops",))  # sales subjects deselected
        names = _namelist(result)
        assert "subject/semantic_models/sales_db/metrics/legacy_metrics.yml" in names
        assert any("no subject_tree tag" in warning for warning in result.warnings)
        # And it still gets a rebuild line, so the receiver can index it.
        script = _member(result, "scripts/rebuild_kb.sh").decode("utf-8")
        assert "legacy_metrics.yml" in script

    def test_untagged_summary_ships_with_a_warning(self, project):
        """A summary carrying no subject_tree matches no selection — it must
        still travel (with a warning) instead of vanishing from every package."""
        (project / "subject" / "sql_summaries" / "q_orphan.yaml").write_text(
            'id: q_orphan\nname: "q_orphan"\nsql: "SELECT 2"\nsummary: "s"\nsearch_text: "t"\n',
            encoding="utf-8",
        )
        result = _build(project, subjects=("sales",))
        assert "subject/sql_summaries/q_orphan.yaml" in _namelist(result)
        assert any("no subject_tree" in warning for warning in result.warnings)

    def test_unknown_subject_fails(self, project):
        result = _build(project, subjects=("ghost",))
        assert not result.ok and "unknown subject" in result.error

    def test_artifact_allowlist_filters_stray_files(self, project):
        names = _namelist(_build(project))
        assert "reports/daily_gmv/render/app.jsx" in names
        assert "reports/daily_gmv/render/scratch.tmp" not in names
        assert "reports/daily_gmv/manifest.json" in names
        assert "dashboards/ops_view/render/app.jsx" in names

    def test_unknown_report_slug_fails(self, project):
        result = _build(project, reports=("ghost_slug",))
        assert not result.ok and "unknown report" in result.error

    def test_report_selection_excludes_others(self, project):
        names = _namelist(_build(project, reports=()))
        assert not any(name.startswith("reports/") for name in names)
        assert any(name.startswith("dashboards/") for name in names)

    def test_report_dist_rewrites_index_html(self, project, tmp_path):
        from datus.agent.node.visual_artifact._artifact_html_renderer import CDN_BUNDLE_CSS, CDN_BUNDLE_JS

        index = project / "reports" / "daily_gmv" / "index.html"
        index.write_text(f'<link href="{CDN_BUNDLE_CSS}"><script src="{CDN_BUNDLE_JS}">', encoding="utf-8")
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.css").write_text("css", encoding="utf-8")
        (dist / "index.umd.js").write_text("js", encoding="utf-8")

        result = _build(project, report_dist=dist)
        names = _namelist(result)
        assert "reports/daily_gmv/_assets/index.css" in names
        assert "reports/daily_gmv/_assets/index.umd.js" in names
        html = _member(result, "reports/daily_gmv/index.html").decode("utf-8")
        assert "_assets/index.css" in html and "_assets/index.umd.js" in html
        assert "unpkg.com" not in html

    def test_report_dist_missing_asset_keeps_cdn_html(self, project, tmp_path):
        from datus.agent.node.visual_artifact._artifact_html_renderer import CDN_BUNDLE_CSS, CDN_BUNDLE_JS

        index = project / "reports" / "daily_gmv" / "index.html"
        original = f'<link href="{CDN_BUNDLE_CSS}"><script src="{CDN_BUNDLE_JS}">'
        index.write_text(original, encoding="utf-8")
        dist = tmp_path / "half-dist"
        dist.mkdir()
        (dist / "index.css").write_text("css", encoding="utf-8")  # index.umd.js missing

        result = _build(project, report_dist=dist)
        names = _namelist(result)
        # Never rewrite to assets that cannot ship: html stays CDN, no _assets staged.
        assert "reports/daily_gmv/_assets/index.umd.js" not in names
        # Byte-identical, not merely "still mentions the CDN" — a half-rewrite
        # that swapped only the CSS would leave the page unloadable.
        assert _member(result, "reports/daily_gmv/index.html").decode("utf-8") == original
        assert any("missing index.umd.js" in warning for warning in result.warnings)

    def test_index_html_kept_as_is_without_dist(self, project):
        index = project / "reports" / "daily_gmv" / "index.html"
        index.write_text("<html>cdn</html>", encoding="utf-8")
        result = _build(project)
        assert _member(result, "reports/daily_gmv/index.html") == b"<html>cdn</html>"


# --------------------------------------------------------------------------- #
# Generated deliverables                                                      #
# --------------------------------------------------------------------------- #


class TestGeneratedFiles:
    def test_requirements_pins(self, project):
        result = _build(project)
        assert _member(result, "requirements.txt").decode("utf-8") == "datus-agent==0.9.9\n"

    @pytest.mark.parametrize("assume_yes", [True, False])
    def test_editable_installs_warn_but_never_block(self, project, monkeypatch, assume_yes):
        """Editable installs are pinned as-is with a warning — no
        confirmation gate in either interactive or --yes mode."""
        monkeypatch.setattr(
            pb,
            "enumerate_datus_packages",
            lambda: [pb.DatusPackage(name="datus-agent", version="0.9.9", editable=True)],
        )
        result = pb.build_package(pb.PackageOptions(root=project, assume_yes=assume_yes))
        assert result.ok is True and result.error is None
        assert any("editable/source installs pinned as-is" in warning for warning in result.warnings)

    def test_install_plugins_script(self, project, monkeypatch):
        (project / ".datus" / "config.yml").write_text(
            "project_name: fixture_proj\nplugins:\n  alpha:\n    enabled: true\n  beta:\n    enabled: false\n",
            encoding="utf-8",
        )
        import datus.plugins.store as store

        monkeypatch.setattr(
            store, "iter_installed", lambda: [{"name": "alpha", "distribution": "datus-alpha", "version": "1.2.3"}]
        )
        result = _build(project)
        script = _member(result, "scripts/install_plugins.sh").decode("utf-8")
        assert "datus plugin install datus-alpha==1.2.3" in script
        assert "beta" not in script

    def test_init_script_orders_the_setup_steps(self, project, monkeypatch):
        import datus.plugins.store as store

        monkeypatch.setattr(
            store, "iter_installed", lambda: [{"name": "alpha", "distribution": "datus-alpha", "version": "1.2.3"}]
        )
        result = _build(project)
        init = _member(result, "scripts/init.sh").decode("utf-8")
        # dependencies -> plugins -> knowledge base, in that order.
        assert (
            init.index("-m pip install -r requirements.txt")
            < init.index("scripts/install_plugins.sh")
            < init.index("scripts/rebuild_kb.sh")
        )
        # Bound to the active interpreter: a bare ``pip`` can belong to a
        # different Python than the venv that will run datus.
        assert "uv pip install --python" in init and '"$PYTHON" -m pip install' in init
        assert "set -euo pipefail" in init
        # No backticks on an executable line: inside a double-quoted echo they
        # are command substitution, so a line naming the datus command would
        # launch the REPL instead of printing its name.
        runnable = [line for line in init.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        assert not [line for line in runnable if "`" in line]
        # Missing env vars are a warning, not a hard stop: the dependency
        # step needs none and the config carries ${VAR:-default} fallbacks.
        assert "WARNING: unset environment variables" in init and "OPENAI_API_KEY" in init
        with zipfile.ZipFile(result.zip_path) as zf:
            assert (zf.getinfo("scripts/init.sh").external_attr >> 16) & 0o111

    def test_init_script_skips_absent_steps(self, project):
        init = _member(_build(project, metrics=(), subjects=()), "scripts/init.sh").decode("utf-8")
        assert "-m pip install -r requirements.txt" in init
        assert "rebuild_kb.sh" not in init  # nothing to rebuild
        assert "install_plugins.sh" not in init  # no plugins activated/installed

    def test_plugin_install_lines_are_rerunnable(self, project, monkeypatch):
        import datus.plugins.store as store

        monkeypatch.setattr(
            store, "iter_installed", lambda: [{"name": "alpha", "distribution": "datus-alpha", "version": "1.2.3"}]
        )
        script = _member(_build(project), "scripts/install_plugins.sh").decode("utf-8")
        # Without --force a second run fails on the already-installed plugin.
        assert "datus plugin install datus-alpha==1.2.3 --force" in script

    def test_no_plugins_no_script(self, project):
        names = _namelist(_build(project))
        assert "scripts/install_plugins.sh" not in names

    def test_plugins_selectable_without_activation_list(self, project, monkeypatch):
        """A project with no ``plugins:`` key can still ship install lines for
        what is installed — previously no script was generated at all."""
        import datus.plugins.store as store

        monkeypatch.setattr(
            store,
            "iter_installed",
            lambda: [
                {"name": "alpha", "distribution": "datus-alpha", "version": "1.2.3"},
                {"name": "beta", "distribution": "datus-beta", "version": "0.4.0"},
            ],
        )
        assert set(pb.list_packageable_plugins(project)) == {"alpha", "beta"}

        result = _build(project, plugins=("beta",))
        script = _member(result, "scripts/install_plugins.sh").decode("utf-8")
        assert "datus plugin install datus-beta==0.4.0" in script
        assert "alpha" not in script

    def test_plugin_selection_rejects_unknown_name(self, project):
        result = _build(project, plugins=("ghost-plugin",))
        assert not result.ok and "unknown plugin" in result.error

    def test_empty_plugin_selection_drops_script(self, project, monkeypatch):
        import datus.plugins.store as store

        monkeypatch.setattr(
            store, "iter_installed", lambda: [{"name": "alpha", "distribution": "datus-alpha", "version": "1.2.3"}]
        )
        names = _namelist(_build(project, plugins=()))
        assert "scripts/install_plugins.sh" not in names

    def test_project_config_pins_name(self, project):
        result = _build(project)
        payload = yaml.safe_load(_member(result, ".datus/config.yml"))
        assert payload["project_name"] == "fixture_proj"

    def test_readme_lists_env_vars(self, project):
        readme = _member(_build(project), "README.md").decode("utf-8")
        assert "OPENAI_API_KEY" in readme
        assert "pip install -r requirements.txt" in readme
        assert ".env" in readme

    def test_scripts_are_executable_in_zip(self, project):
        result = _build(project)
        with zipfile.ZipFile(result.zip_path) as zf:
            info = zf.getinfo("scripts/rebuild_kb.sh")
            assert (info.external_attr >> 16) & 0o111, "rebuild_kb.sh must carry the executable bit"


# --------------------------------------------------------------------------- #
# Final secret scan                                                           #
# --------------------------------------------------------------------------- #


class TestSecretScan:
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param("token ghp_abcdefghijklmnopqrstuv123456 in text", id="github_token"),
            pytest.param("aws AKIAIOSFODNN7EXAMPLE key", id="aws_key_id"),
            pytest.param("-----BEGIN RSA PRIVATE KEY-----\nabc\n", id="pem"),
            pytest.param("slack xoxb-123456789012-abcdef token", id="slack_token"),
            pytest.param("fernet gAAAAABkX3q7abcdefghijklmnopqrstuv payload", id="fernet_token"),
        ],
    )
    def test_secret_material_in_project_file_fails_build(self, project, payload):
        (project / "knowledge" / "leak.md").write_text(payload, encoding="utf-8")
        result = _build(project)
        assert not result.ok
        assert any(finding.arcname == "knowledge/leak.md" for finding in result.secret_findings)

    def test_binary_files_skipped(self, project):
        (project / "knowledge" / "blob.bin").write_bytes(b"\x00\x01ghp_abcdefghijklmnopqrstuv123456")
        result = _build(project)
        assert result.ok is True and result.secret_findings == []

    def test_generated_yaml_self_check_catches_missed_section(self, project):
        # Simulate a future config section the sanitizer table doesn't know:
        # the value-driven self-check must fail the build rather than leak.
        raw = yaml.safe_load((project / "conf" / "agent.yml").read_text(encoding="utf-8"))
        raw["agent"]["future_section"] = {"service_password": "plaintext-oops"}
        (project / "conf" / "agent.yml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        result = _build(project)
        assert not result.ok
        assert any(
            finding.kind == "plaintext_secret_key" and "future_section" in finding.locator
            for finding in result.secret_findings
        )

    def test_clean_project_passes(self, project):
        result = _build(project)
        assert result.ok is True and result.secret_findings == [] and result.error is None

    def test_scan_truncation_is_surfaced_as_warning(self, project, monkeypatch):
        monkeypatch.setattr(pb, "_SCAN_READ_CAP_BYTES", 1024)
        # Token sits PAST the cap: undetectable, so the truncation must be loud.
        (project / "knowledge" / "big.md").write_text(
            "x" * 2048 + "\nghp_abcdefghijklmnopqrstuv123456\n", encoding="utf-8"
        )
        result = _build(project)
        assert result.ok is True  # the token past the cap is genuinely unseen
        assert any("secret scan truncated" in warning and "big.md" in warning for warning in result.warnings)

    def test_broken_plugin_schema_degrades_to_all_leaves(self, project, monkeypatch):
        import datus.plugins.registry as registry

        def boom(_name: str) -> Never:
            raise RuntimeError("broken manifest")

        monkeypatch.setattr(registry, "plugin_config_schema", boom)
        raw = {"plugins": {"alpha": {"prod": {"endpoint": "https://x", "note": "plain"}}}}
        alloc = pb._PlaceholderAllocator()
        warnings: list[str] = []
        pb._sanitize_plugin_profiles(raw["plugins"], alloc, warnings)
        # Lookup failure == no schema: every string leaf becomes a placeholder.
        assert raw["plugins"]["alpha"]["prod"]["endpoint"].startswith("${")
        assert raw["plugins"]["alpha"]["prod"]["note"].startswith("${")
        assert any("no config schema" in warning for warning in warnings)


# --------------------------------------------------------------------------- #
# End-to-end layout                                                           #
# --------------------------------------------------------------------------- #


class TestEndToEnd:
    def test_layout_and_manifest_integrity(self, project, tmp_path):
        result = _build(project)
        names = _namelist(result)
        assert not any(".." in name or name.startswith("/") or "\\" in name for name in names)

        manifest = json.loads(_member(result, "package_manifest.json"))
        assert manifest["format"] == "datus-project-package"
        assert manifest["project_name"] == "fixture_proj"
        assert set(manifest["selections"]) >= {"subagents", "skills", "metrics", "reports", "dashboards"}
        listed = {entry["path"] for entry in manifest["files"]}
        assert listed == names - {"package_manifest.json"}

        # sha256 integrity spot-check against the extracted bytes.
        import hashlib

        by_path = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
        body = _member(result, "conf/agent.yml")
        assert hashlib.sha256(body).hexdigest() == by_path["conf/agent.yml"]

        # Per-file provenance distinguishes pack-time products from copies.
        source_by_path = {entry["path"]: entry["source"] for entry in manifest["files"]}
        assert source_by_path["conf/agent.yml"] == "generated"
        assert source_by_path["knowledge/notes.md"] == "project"

        # The unzipped tree parses as a valid agent config shape. Blanket
        # extractall is safe *here only*: this archive is the one build_package
        # just wrote, and the member paths were asserted traversal-free above.
        # Code reading an archive from elsewhere must extract member by member.
        extract = tmp_path / "unpacked"
        with zipfile.ZipFile(result.zip_path) as zf:
            zf.extractall(extract)
        agent = yaml.safe_load((extract / "conf" / "agent.yml").read_text(encoding="utf-8"))["agent"]
        assert agent["home"] == "." and agent["project_name"] == "fixture_proj"

    def test_manifest_env_vars_carry_config_paths(self, project):
        """Each ``${VAR}`` the receiver must export names the config fields that
        reference it — without them they can only guess what a variable is for.
        """
        manifest = json.loads(_member(_build(project), "package_manifest.json"))

        assert manifest["format_version"] == 2
        records = manifest["env_vars"]
        assert records, "fixture project has credential-bearing config"
        assert all(set(r) == {"var", "config_paths", "preexisting"} for r in records)
        # Deterministic ordering, outer and inner.
        assert [r["var"] for r in records] == sorted(r["var"] for r in records)
        assert all(r["config_paths"] == sorted(set(r["config_paths"])) for r in records)

        by_var = {r["var"]: r for r in records}
        assert "providers.openai.api_key" in by_var["OPENAI_API_KEY"]["config_paths"]
        # ANTHROPIC_API_KEY was already a ${VAR} in the source config; OPENAI's
        # was a literal the packer rewrote.
        assert by_var["ANTHROPIC_API_KEY"]["preexisting"] is True
        assert by_var["OPENAI_API_KEY"]["preexisting"] is False

    def test_manifest_env_vars_match_result_bindings(self, project):
        """The manifest view and the in-process view cannot drift apart."""
        result = _build(project)
        manifest = json.loads(_member(result, "package_manifest.json"))

        assert {r["var"] for r in manifest["env_vars"]} == {b.var for b in result.env_vars}

    def test_manifest_config_paths_match_readme_table(self, project):
        """README table and manifest are generated from the same helper."""
        result = _build(project)
        manifest = json.loads(_member(result, "package_manifest.json"))
        readme = _member(result, "README.md").decode("utf-8")

        for record in manifest["env_vars"]:
            for config_path in record["config_paths"]:
                assert config_path in readme

    def test_manifest_carries_no_secret_material(self, project):
        """The manifest is appended AFTER scan_for_secrets runs, so it never goes
        through the scanner — assert directly that config paths, not values,
        are what ships.
        """
        raw = _member(_build(project), "package_manifest.json").decode("utf-8")

        for secret in (_PLAINTEXT_KEY, "hunter2", "p4ss", "t0k", "tvly-plain", "abc123token"):
            assert secret not in raw


class TestGroupEnvVars:
    """``group_env_vars`` — the shared README/manifest view of the bindings."""

    def test_preexisting_uses_all_not_any(self):
        """A var counts as preexisting only if EVERY site already held a ${VAR}.
        Using ``any`` would let one harmless harvested occurrence hide the fact
        that a real credential was stripped somewhere else.
        """
        records = pb.group_env_vars(
            [
                pb.EnvVarBinding(var="V", config_path="a.key", preexisting=True),
                pb.EnvVarBinding(var="V", config_path="b.key", preexisting=False),
            ]
        )

        assert len(records) == 1
        assert records[0].preexisting is False
        assert records[0].config_paths == ["a.key", "b.key"]

    def test_all_preexisting_stays_true(self):
        records = pb.group_env_vars(
            [
                pb.EnvVarBinding(var="V", config_path="a.key", preexisting=True),
                pb.EnvVarBinding(var="V", config_path="b.key", preexisting=True),
            ]
        )

        assert records[0].preexisting is True

    def test_dedupes_and_sorts_config_paths(self):
        records = pb.group_env_vars(
            [
                pb.EnvVarBinding(var="V", config_path="z.key", preexisting=False),
                pb.EnvVarBinding(var="V", config_path="a.key", preexisting=False),
                pb.EnvVarBinding(var="V", config_path="z.key", preexisting=False),
            ]
        )

        assert records[0].config_paths == ["a.key", "z.key"]

    def test_sorts_vars(self):
        records = pb.group_env_vars(
            [
                pb.EnvVarBinding(var="Z_VAR", config_path="a.key", preexisting=False),
                pb.EnvVarBinding(var="A_VAR", config_path="b.key", preexisting=False),
            ]
        )

        assert [r.var for r in records] == ["A_VAR", "Z_VAR"]

    def test_empty_input(self):
        assert pb.group_env_vars([]) == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
