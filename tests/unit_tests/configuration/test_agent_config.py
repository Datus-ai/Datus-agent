# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Additional unit tests for datus/configuration/agent_config.py

Covers: resolve_env, file_stem_from_uri, DbConfig.filter_kwargs,
BenchmarkConfig.validate, DocumentConfig.from_dict/merge_cli_args,
load_model_config, AgentConfig helper methods.

CI-level: zero external deps, zero network.
"""

import argparse

import pytest

from datus.configuration.agent_config import (
    AgentConfig,
    BenchmarkConfig,
    DbConfig,
    DocumentConfig,
    ModelConfig,
    NodeConfig,
    ServicesConfig,
    ValidationConfig,
    file_stem_from_uri,
    load_model_config,
    resolve_env,
)
from datus.utils.exceptions import DatusException

pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# resolve_env
# ---------------------------------------------------------------------------


class TestResolveEnv:
    def test_plain_string_unchanged(self):
        assert resolve_env("hello") == "hello"

    def test_env_var_substituted(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_KEY", "secret123")
        result = resolve_env("${MY_TEST_KEY}")
        assert result == "secret123"

    def test_missing_env_var_returns_placeholder(self, monkeypatch):
        # Make sure this env var is not set
        monkeypatch.delenv("DATUS_NONEXISTENT_VAR_XYZ", raising=False)
        result = resolve_env("${DATUS_NONEXISTENT_VAR_XYZ}")
        assert result == "<MISSING:DATUS_NONEXISTENT_VAR_XYZ>"

    def test_multiple_env_vars_in_string(self, monkeypatch):
        monkeypatch.setenv("HOST_VAR", "localhost")
        monkeypatch.setenv("PORT_VAR", "5432")
        result = resolve_env("${HOST_VAR}:${PORT_VAR}")
        assert result == "localhost:5432"

    def test_none_returns_none(self):
        assert resolve_env(None) is None

    def test_empty_string_returns_empty(self):
        assert resolve_env("") == ""

    def test_non_string_returns_as_is(self):
        assert resolve_env(42) == 42

    def test_no_placeholder_unchanged(self):
        assert resolve_env("plain/path/no/vars") == "plain/path/no/vars"


# ---------------------------------------------------------------------------
# file_stem_from_uri
# ---------------------------------------------------------------------------


class TestFileStemFromUri:
    def test_sqlite_uri(self):
        assert file_stem_from_uri("sqlite:////tmp/foo.db") == "foo"

    def test_duckdb_uri(self):
        assert file_stem_from_uri("duckdb:///path/to/demo.duckdb") == "demo"

    def test_plain_path(self):
        assert file_stem_from_uri("/abs/path/bar.duckdb") == "bar"

    def test_relative_path(self):
        assert file_stem_from_uri("foo.db") == "foo"

    def test_empty_string(self):
        assert file_stem_from_uri("") == ""

    def test_no_extension(self):
        result = file_stem_from_uri("mydb")
        assert result == "mydb"


# ---------------------------------------------------------------------------
# DbConfig.filter_kwargs
# ---------------------------------------------------------------------------


class TestDbConfigFilterKwargs:
    def test_valid_fields_mapped(self):
        kwargs = {"type": "sqlite", "uri": "sqlite:///test.db", "database": "test"}
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        assert cfg.type == "sqlite"
        assert "test.db" in cfg.uri

    def test_unknown_fields_go_to_extra(self):
        kwargs = {"type": "mysql", "host": "localhost", "custom_option": "value123"}
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        assert cfg.extra is not None
        assert "custom_option" in cfg.extra
        assert cfg.extra["custom_option"] == "value123"

    def test_name_sets_logic_name(self):
        kwargs = {"type": "sqlite", "uri": "sqlite:///db.db", "name": "my_logic_name"}
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        assert cfg.logic_name == "my_logic_name"

    def test_sqlite_extracts_database_stem(self):
        kwargs = {"type": "sqlite", "uri": "sqlite:///path/mydata.db"}
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        assert cfg.database == "mydata"

    def test_duckdb_extracts_database_stem(self):
        kwargs = {"type": "duckdb", "uri": "duckdb:///warehouse.duckdb"}
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        assert cfg.database == "warehouse"

    def test_extra_with_unknown_fields(self):
        # Unknown fields without existing 'extra' go into extra dict
        kwargs = {
            "type": "mysql",
            "new_custom": "new_val",
            "another_key": "another_val",
        }
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        assert cfg.extra["new_custom"] == "new_val"
        assert cfg.extra["another_key"] == "another_val"

    def test_none_values_ignored_for_extra(self):
        kwargs = {"type": "sqlite", "uri": "x.db", "some_none_field": None}
        cfg = DbConfig.filter_kwargs(DbConfig, kwargs)
        # None values should not be added to extra
        assert cfg.extra is None or "some_none_field" not in cfg.extra


# ---------------------------------------------------------------------------
# BenchmarkConfig.validate
# ---------------------------------------------------------------------------


class TestBenchmarkConfigValidate:
    def test_valid_config_passes(self):
        cfg = BenchmarkConfig(
            question_key="question",
            question_file="dev.json",
            question_id_key="id",
        )
        assert cfg.validate() is None
        assert cfg.question_key == "question"
        assert cfg.question_file == "dev.json"
        assert cfg.question_id_key == "id"

    def test_missing_question_key_raises(self):
        cfg = BenchmarkConfig(question_file="dev.json", question_id_key="id")
        with pytest.raises(DatusException):
            cfg.validate()

    def test_missing_question_file_raises(self):
        cfg = BenchmarkConfig(question_key="question", question_id_key="id")
        with pytest.raises(DatusException):
            cfg.validate()

    def test_missing_question_id_key_raises(self):
        cfg = BenchmarkConfig(question_key="question", question_file="dev.json")
        with pytest.raises(DatusException):
            cfg.validate()

    def test_filter_kwargs_keeps_valid_fields(self):
        data = {
            "question_key": "q",
            "question_file": "f.json",
            "question_id_key": "id",
            "unknown_field": "ignored",
        }
        cfg = BenchmarkConfig.filter_kwargs(BenchmarkConfig, data)
        assert cfg.question_key == "q"
        assert not hasattr(cfg, "unknown_field")


# ---------------------------------------------------------------------------
# DocumentConfig
# ---------------------------------------------------------------------------


class TestDocumentConfig:
    def test_from_dict_basic(self):
        data = {"type": "github", "source": "owner/repo", "version": "1.0"}
        cfg = DocumentConfig.from_dict(data)
        assert cfg.type == "github"
        assert cfg.source == "owner/repo"
        assert cfg.version == "1.0"

    def test_from_dict_ignores_unknown_fields(self):
        data = {"type": "local", "unknown_field": "ignored"}
        cfg = DocumentConfig.from_dict(data)
        assert cfg.type == "local"

    def test_defaults(self):
        cfg = DocumentConfig.from_dict({})
        assert cfg.type == "local"
        assert cfg.chunk_size == 1024
        assert cfg.max_depth == 2

    def test_merge_cli_args_overrides_type(self):
        cfg = DocumentConfig.from_dict({"type": "local"})
        args = argparse.Namespace(
            source_type="github",
            source=None,
            version=None,
            github_ref=None,
            github_token=None,
            paths=None,
            chunk_size=None,
            max_depth=None,
            include_patterns=None,
            exclude_patterns=None,
        )
        merged = cfg.merge_cli_args(args)
        assert merged.type == "github"

    def test_merge_cli_args_none_values_not_override(self):
        cfg = DocumentConfig.from_dict({"type": "website", "version": "2.0"})
        args = argparse.Namespace(
            source_type=None,
            source=None,
            version=None,
            github_ref=None,
            github_token=None,
            paths=None,
            chunk_size=None,
            max_depth=None,
            include_patterns=None,
            exclude_patterns=None,
        )
        merged = cfg.merge_cli_args(args)
        # None args should not override existing values
        assert merged.type == "website"
        assert merged.version == "2.0"

    def test_merge_cli_args_resolves_env_for_strings(self, monkeypatch):
        monkeypatch.setenv("DOC_SOURCE", "myrepo/docs")
        cfg = DocumentConfig.from_dict({})
        args = argparse.Namespace(
            source_type=None,
            source="${DOC_SOURCE}",
            version=None,
            github_ref=None,
            github_token=None,
            paths=None,
            chunk_size=None,
            max_depth=None,
            include_patterns=None,
            exclude_patterns=None,
        )
        merged = cfg.merge_cli_args(args)
        assert merged.source == "myrepo/docs"


# ---------------------------------------------------------------------------
# load_model_config
# ---------------------------------------------------------------------------


class TestLoadModelConfig:
    def test_basic_config(self):
        data = {
            "type": "openai",
            "api_key": "sk-test",
            "model": "gpt-4",
        }
        cfg = load_model_config(data)
        assert cfg.type == "openai"
        assert cfg.model == "gpt-4"
        assert cfg.max_retry == 3
        assert cfg.retry_interval == 2.0

    def test_custom_retry_settings(self):
        data = {
            "type": "openai",
            "api_key": "sk-test",
            "model": "gpt-4",
            "max_retry": 5,
            "retry_interval": 1.0,
        }
        cfg = load_model_config(data)
        assert cfg.max_retry == 5
        assert cfg.retry_interval == 1.0

    def test_temperature_and_top_p(self):
        data = {
            "type": "openai",
            "api_key": "sk-test",
            "model": "kimi-k2.5",
            "temperature": 1.0,
            "top_p": 0.95,
        }
        cfg = load_model_config(data)
        assert cfg.temperature == 1.0
        assert cfg.top_p == 0.95

    def test_none_temperature_by_default(self):
        data = {"type": "openai", "api_key": "sk", "model": "gpt-4"}
        cfg = load_model_config(data)
        assert cfg.temperature is None
        assert cfg.top_p is None

    def test_enable_thinking(self):
        data = {
            "type": "anthropic",
            "api_key": "sk",
            "model": "claude-3-5",
            "enable_thinking": True,
        }
        cfg = load_model_config(data)
        assert cfg.enable_thinking is True

    def test_default_headers(self):
        data = {
            "type": "openai",
            "api_key": "sk",
            "model": "gpt-4",
            "default_headers": {"X-Custom": "value"},
        }
        cfg = load_model_config(data)
        assert cfg.default_headers == {"X-Custom": "value"}

    def test_base_url_resolved(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com")
        data = {
            "type": "openai",
            "api_key": "sk",
            "model": "gpt-4",
            "base_url": "${LLM_BASE_URL}",
        }
        cfg = load_model_config(data)
        assert cfg.base_url == "https://api.example.com"

    def test_to_dict(self):
        cfg = ModelConfig(type="openai", api_key="sk", model="gpt-4")
        d = cfg.to_dict()
        assert d["type"] == "openai"
        assert d["model"] == "gpt-4"


class TestAgentConfigServiceSelectors:
    def _make(self, tmp_path, *, services=None, agentic_nodes=None):
        return AgentConfig(
            nodes={"test": NodeConfig(model="test-model", input=None)},
            home=str(tmp_path / "h"),
            target="mock",
            models={
                "mock": {
                    "type": "openai",
                    "api_key": "k",
                    "model": "m",
                    "base_url": "http://localhost:0",
                }
            },
            services=services or {"datasources": {}},
            agentic_nodes=agentic_nodes or {},
            skip_init_dirs=True,
        )

    def test_resolve_semantic_adapter_returns_explicit_configured_adapter(self, tmp_path):
        cfg = self._make(
            tmp_path,
            services={
                "datasources": {},
                "semantic_layer": {
                    "metricflow": {"timeout": 300},
                },
            },
        )
        assert cfg.resolve_semantic_adapter("metricflow") == "metricflow"

    def test_resolve_semantic_adapter_auto_selects_single_configured_entry(self, tmp_path):
        cfg = self._make(
            tmp_path,
            services={
                "datasources": {},
                "semantic_layer": {
                    "metricflow": {"timeout": 300},
                },
            },
        )
        assert cfg.resolve_semantic_adapter() == "metricflow"

    def test_resolve_semantic_adapter_defaults_to_metricflow_without_service_config(self, tmp_path):
        cfg = self._make(
            tmp_path,
            services={
                "datasources": {},
            },
        )
        assert cfg.resolve_semantic_adapter() == "metricflow"

    def test_build_semantic_adapter_config_defaults_to_metricflow_without_service_config(self, tmp_path):
        cfg = self._make(
            tmp_path,
            services={
                "datasources": {},
            },
        )

        config = cfg.build_semantic_adapter_config()

        assert config["type"] == "metricflow"
        assert config["agent_home"] == str(tmp_path / "h")
        assert config["semantic_models_path"].endswith("subject/semantic_models")

    def test_resolve_semantic_adapter_requires_explicit_choice_for_multiple_entries(self, tmp_path):
        cfg = self._make(
            tmp_path,
            services={
                "datasources": {},
                "semantic_layer": {
                    "metricflow": {"timeout": 300},
                    "cube": {"timeout": 60},
                },
            },
        )
        with pytest.raises(DatusException, match="Multiple semantic layers are configured"):
            cfg.resolve_semantic_adapter()

    def test_default_scheduler_service_prefers_single_default(self, tmp_path):
        cfg = self._make(
            tmp_path,
            services={
                "datasources": {},
                "schedulers": {
                    "airflow_prod": {"type": "airflow", "default": True},
                    "airflow_dev": {"type": "airflow"},
                },
            },
        )
        assert cfg.default_scheduler_service() == "airflow_prod"

    def test_default_scheduler_service_rejects_multiple_defaults(self, tmp_path):
        with pytest.raises(DatusException, match="Multiple scheduler services are marked"):
            self._make(
                tmp_path,
                services={
                    "datasources": {},
                    "schedulers": {
                        "airflow_prod": {"type": "airflow", "default": True},
                        "airflow_dev": {"type": "airflow", "default": True},
                    },
                },
            )

    def test_get_scheduler_config_requires_explicit_choice_when_multiple_instances_exist(self, tmp_path):
        cfg = self._make(
            tmp_path,
            services={
                "datasources": {},
                "schedulers": {
                    "airflow_prod": {"type": "airflow"},
                    "airflow_dev": {"type": "airflow"},
                },
            },
        )
        with pytest.raises(DatusException, match="set `scheduler_service` on the scheduler node"):
            cfg.get_scheduler_config()

    def test_get_scheduler_config_returns_requested_instance(self, tmp_path):
        cfg = self._make(
            tmp_path,
            services={
                "datasources": {},
                "schedulers": {
                    "airflow_prod": {"type": "airflow", "api_base_url": "http://prod"},
                    "airflow_dev": {"type": "airflow", "api_base_url": "http://dev"},
                },
            },
        )
        assert cfg.get_scheduler_config("airflow_dev")["api_base_url"] == "http://dev"

    def test_init_scheduler_services_requires_declared_type(self, tmp_path):
        with pytest.raises(DatusException, match="must declare a scheduler `type`"):
            self._make(
                tmp_path,
                services={
                    "datasources": {},
                    "schedulers": {
                        "airflow_prod": {"api_base_url": "http://prod"},
                    },
                },
            )


# ---------------------------------------------------------------------------
# AgentConfig.api_config
# ---------------------------------------------------------------------------


class TestAgentConfigApiSection:
    def _make(self, tmp_path, api=None):
        from datus.configuration.agent_config import AgentConfig, NodeConfig

        kwargs = dict(
            nodes={"test": NodeConfig(model="test-model", input=None)},
            home=str(tmp_path / "h"),
            target="mock",
            models={
                "mock": {
                    "type": "openai",
                    "api_key": "k",
                    "model": "m",
                    "base_url": "http://localhost:0",
                }
            },
            skip_init_dirs=True,
        )
        if api is not None:
            kwargs["api"] = api
        return AgentConfig(**kwargs)

    def test_default_api_config_empty(self, tmp_path):
        cfg = self._make(tmp_path)
        assert cfg.api_config == {}

    def test_api_config_parsed(self, tmp_path):
        api = {"auth_provider": {"class": "pkg.mod.Cls", "kwargs": {"a": 1}}}
        cfg = self._make(tmp_path, api=api)
        assert cfg.api_config == api


class TestNormalizeProjectName:
    """Tests for the _normalize_project_name helper."""

    def test_replaces_slashes(self):
        from datus.configuration.agent_config import _normalize_project_name

        assert _normalize_project_name("/Users/me/proj") == "Users-me-proj"

    def test_strips_leading_dash_only(self):
        from datus.configuration.agent_config import _normalize_project_name

        # Leading slash -> leading '-' which is stripped.
        assert _normalize_project_name("/a/b/c") == "a-b-c"

    def test_root_falls_back_to_underscore_root(self):
        from datus.configuration.agent_config import _normalize_project_name

        assert _normalize_project_name("/") == "_root"

    def test_empty_falls_back_to_underscore_root(self):
        from datus.configuration.agent_config import _normalize_project_name

        assert _normalize_project_name("") == "_root"

    def test_long_path_truncated_with_md5(self):
        import re

        from datus.configuration.agent_config import _PROJECT_NAME_MAX_LEN, _normalize_project_name

        long_cwd = "/" + "/".join("seg" + str(i) for i in range(200))
        name = _normalize_project_name(long_cwd)
        assert len(name) <= _PROJECT_NAME_MAX_LEN
        # Expect "<prefix>-<7 hex chars>" at the tail.
        assert re.search(r"-[0-9a-f]{7}$", name), name

    def test_backslashes_treated_like_slashes(self):
        from datus.configuration.agent_config import _normalize_project_name

        # ``:`` is outside the backend-accepted segment class and is sanitized to ``_``.
        assert _normalize_project_name("C:\\Users\\me\\proj") == "C_-Users-me-proj"

    def test_special_chars_sanitized_to_underscore(self):
        """Chars outside [A-Za-z0-9_.-] are replaced so backend _safe_path_segment accepts the result."""
        from datus.configuration.agent_config import _normalize_project_name

        assert _normalize_project_name("/Users/Felix Liu/proj") == "Users-Felix_Liu-proj"
        assert _normalize_project_name("/a(b)/c@d") == "a_b_-c_d"

    def test_derived_name_passes_backend_segment_check(self):
        """Automatically derived names must pass the backend-side segment validator."""
        from datus.configuration.agent_config import _normalize_project_name
        from datus.storage.rdb.sqlite_backend import _safe_path_segment

        for cwd in [
            "/Users/Felix Liu/proj",
            "/a(b)/c@d",
            "C:\\Users\\me\\proj",
            "/",
            "",
            "/tmp/x.y/z",
        ]:
            name = _normalize_project_name(cwd)
            # Must not raise.
            assert _safe_path_segment(name, "project") == name


class TestAgentConfigProjectLayout:
    """Verify AgentConfig derives project-aware storage paths correctly."""

    def _make(self, tmp_path, *, project_name=None, project_root=None, **extra_kwargs):
        from datus.configuration.agent_config import AgentConfig, NodeConfig

        kwargs = dict(
            nodes={"test": NodeConfig(model="test-model", input=None)},
            home=str(tmp_path / "datus_home"),
            target="mock",
            models={
                "mock": {
                    "type": "openai",
                    "api_key": "k",
                    "model": "m",
                    "base_url": "http://localhost:0",
                }
            },
            skip_init_dirs=True,
        )
        if project_name is not None:
            kwargs["project_name"] = project_name
        if project_root is not None:
            kwargs["project_root"] = str(project_root)
        kwargs.update(extra_kwargs)
        return AgentConfig(**kwargs)

    def test_sessions_and_data_sharded_by_project_name(self, tmp_path):
        project_root = tmp_path / "my_project"
        cfg = self._make(tmp_path, project_name="demo_project", project_root=project_root)

        datus_home = (tmp_path / "datus_home").resolve()
        # data_dir is the backend root (no project suffix); project sharding
        # is surfaced via project_data_dir and owned by backends.
        assert cfg.path_manager.data_dir == datus_home / "data"
        assert cfg.path_manager.project_data_dir == datus_home / "data" / "demo_project"
        assert cfg.path_manager.sessions_dir == datus_home / "sessions" / "demo_project"

    def test_subject_tree_anchored_to_project_root(self, tmp_path):
        project_root = tmp_path / "my_project"
        cfg = self._make(tmp_path, project_name="demo_project", project_root=project_root)

        subject = project_root.resolve() / "subject"
        assert cfg.path_manager.subject_dir == subject
        assert cfg.path_manager.semantic_models_dir == subject / "semantic_models"
        assert cfg.path_manager.sql_summaries_dir == subject / "sql_summaries"
        assert cfg.path_manager.ext_knowledge_dir == subject / "ext_knowledge"

    def test_project_name_auto_derived_from_cwd_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = self._make(tmp_path)

        # Should be a sanitized version of tmp_path (all '/' replaced with '-')
        assert cfg.project_name
        assert "/" not in cfg.project_name

    def test_knowledge_base_home_kwarg_silently_ignored(self, tmp_path):
        """Removed setting: passing it via YAML/kwargs is silently dropped (no raise, no effect)."""
        cfg = self._make(
            tmp_path,
            project_name="demo_project",
            project_root=tmp_path / "my_project",
            knowledge_base_home=str(tmp_path / "ignored_kb"),
        )
        # KB still anchors to project_root/subject — kwarg is dropped.
        assert cfg.path_manager.subject_dir == (tmp_path / "my_project").resolve() / "subject"
        assert not hasattr(cfg, "knowledge_base_home")

    def test_project_name_is_read_only(self, tmp_path):
        """project_name is immutable post-construction; no runtime switching."""
        cfg = self._make(tmp_path, project_name="first", project_root=tmp_path)
        with pytest.raises(AttributeError):
            cfg.project_name = "second"  # type: ignore[misc]

    def test_invalid_project_name_rejected(self, tmp_path):
        """YAML project_name must match _PROJECT_NAME_RE — slashes are rejected."""
        with pytest.raises(DatusException):
            self._make(tmp_path, project_name="bad/name", project_root=tmp_path)

    def test_overlong_project_name_rejected(self, tmp_path):
        from datus.configuration.agent_config import _PROJECT_NAME_MAX_LEN

        with pytest.raises(DatusException):
            self._make(tmp_path, project_name="a" * (_PROJECT_NAME_MAX_LEN + 1), project_root=tmp_path)


# ---------------------------------------------------------------------------
# AgentConfig.filesystem_strict
# ---------------------------------------------------------------------------


class TestAgentConfigFilesystemStrict:
    """``filesystem_strict`` is the process-wide fail-closed switch for
    FilesystemFuncTool EXTERNAL access. It has three input channels
    (``agent.filesystem.strict`` in YAML, ``--filesystem-strict`` CLI flag
    via ``override_by_args``, direct setter from API/gateway bootstraps) and
    all three must land on the same underlying property.
    """

    def _make(self, tmp_path, **extra_kwargs):
        from datus.configuration.agent_config import AgentConfig, NodeConfig

        kwargs = dict(
            nodes={"test": NodeConfig(model="test-model", input=None)},
            home=str(tmp_path / "h"),
            target="mock",
            models={
                "mock": {
                    "type": "openai",
                    "api_key": "k",
                    "model": "m",
                    "base_url": "http://localhost:0",
                }
            },
            skip_init_dirs=True,
        )
        kwargs.update(extra_kwargs)
        return AgentConfig(**kwargs)

    def test_default_false(self, tmp_path):
        cfg = self._make(tmp_path)
        assert cfg.filesystem_strict is False

    def test_from_yaml_true(self, tmp_path):
        cfg = self._make(tmp_path, filesystem={"strict": True})
        assert cfg.filesystem_strict is True

    def test_from_yaml_false_explicit(self, tmp_path):
        cfg = self._make(tmp_path, filesystem={"strict": False})
        assert cfg.filesystem_strict is False

    def test_from_yaml_missing_key_defaults_false(self, tmp_path):
        # ``agent.filesystem: {}`` (empty dict) must still default to False,
        # not crash on a missing ``strict`` key.
        cfg = self._make(tmp_path, filesystem={})
        assert cfg.filesystem_strict is False

    def test_setter_coerces_to_bool(self, tmp_path):
        cfg = self._make(tmp_path)
        cfg.filesystem_strict = 1
        assert cfg.filesystem_strict is True
        cfg.filesystem_strict = 0
        assert cfg.filesystem_strict is False
        cfg.filesystem_strict = True
        assert cfg.filesystem_strict is True

    def test_override_by_args_true_flips(self, tmp_path):
        cfg = self._make(tmp_path, filesystem={"strict": False})
        cfg.override_by_args(filesystem_strict=True)
        assert cfg.filesystem_strict is True

    def test_override_by_args_false_flips(self, tmp_path):
        # --no-filesystem-strict must be able to override a YAML True.
        cfg = self._make(tmp_path, filesystem={"strict": True})
        cfg.override_by_args(filesystem_strict=False)
        assert cfg.filesystem_strict is False

    def test_override_by_args_none_preserves_yaml(self, tmp_path):
        # When neither CLI flag is passed, argparse leaves filesystem_strict=None
        # and the YAML-derived value must survive.
        cfg = self._make(tmp_path, filesystem={"strict": True})
        cfg.override_by_args(filesystem_strict=None)
        assert cfg.filesystem_strict is True

    def test_override_by_args_missing_preserves_yaml(self, tmp_path):
        # override_by_args is also called without a filesystem_strict key
        # (e.g. in non-CLI code paths). That must not reset the flag.
        cfg = self._make(tmp_path, filesystem={"strict": True})
        cfg.override_by_args()
        assert cfg.filesystem_strict is True


class TestAgentConfigLanguage:
    """``agent.language`` is the default response language for all agentic
    nodes. Chat API requests may override it per-task on the cloned config.
    """

    def _make(self, tmp_path, **extra_kwargs):
        kwargs = dict(
            nodes={"test": NodeConfig(model="test-model", input=None)},
            home=str(tmp_path / "h"),
            target="mock",
            models={
                "mock": {
                    "type": "openai",
                    "api_key": "k",
                    "model": "m",
                    "base_url": "http://localhost:0",
                }
            },
            skip_init_dirs=True,
        )
        kwargs.update(extra_kwargs)
        return AgentConfig(**kwargs)

    def test_default_language_is_none(self, tmp_path):
        """Unset language lets the model choose its own response language."""
        cfg = self._make(tmp_path)
        assert cfg.language is None

    def test_custom_language_preserved(self, tmp_path):
        cfg = self._make(tmp_path, language="zh")
        assert cfg.language == "zh"

    def test_runtime_override_sets_language(self, tmp_path):
        cfg = self._make(tmp_path, language="en")
        cfg.language = "ja"
        assert cfg.language == "ja"


class TestServicesConfigFromDict:
    def test_bi_platforms_key_is_parsed(self):
        cfg = ServicesConfig.from_dict({"bi_platforms": {"superset": {"type": "superset"}}})
        assert cfg.bi_platforms == {"superset": {"type": "superset"}}

    def test_legacy_bi_tools_key_is_accepted_with_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="services.bi_tools is deprecated"):
            cfg = ServicesConfig.from_dict({"bi_tools": {"superset": {"type": "superset"}}})
        assert cfg.bi_platforms == {"superset": {"type": "superset"}}

    def test_bi_platforms_takes_precedence_over_legacy_key(self):
        cfg = ServicesConfig.from_dict(
            {
                "bi_platforms": {"superset": {"type": "superset"}},
                "bi_tools": {"grafana": {"type": "grafana"}},
            }
        )
        assert cfg.bi_platforms == {"superset": {"type": "superset"}}

    def test_legacy_databases_key_is_rejected(self):
        """Old 'services.databases' layout must raise and point users at the migrator."""
        with pytest.raises(DatusException, match="services.databases has been renamed to services.datasources"):
            ServicesConfig.from_dict({"databases": {"my_db": {"type": "sqlite"}}})

    def test_datasources_key_without_legacy_parses_cleanly(self):
        """With only 'datasources' present, from_dict returns an empty dataclass (entries populated later)."""
        cfg = ServicesConfig.from_dict({"datasources": {"my_db": {"type": "sqlite"}}})
        # from_dict intentionally leaves datasources empty — AgentConfig._init_services_config fills it.
        assert cfg.datasources == {}


# ---------------------------------------------------------------------------
# Provider-level configuration (new schema)
# ---------------------------------------------------------------------------


class TestProviderConfigurationDispatch:
    """Cover ``ProviderConfig`` + the three-way dispatch in ``active_model()``.

    Scenarios exercised:
      - legacy string ``target`` continues to index ``agent.models``.
      - structured ``(provider, model)`` synthesizes a ``ModelConfig``
        from ``agent.providers`` plus the injected catalog.
      - ``set_active_*`` helpers mutate in-memory state and persist to
        ``./.datus/config.yml``.
      - ``provider_available`` returns ``True`` when credentials are
        present in overrides or env.
    """

    def _stub_catalog(self):
        return {
            "providers": {
                "openai": {
                    "type": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                    "default_model": "gpt-4.1",
                    "models": ["gpt-4.1", "gpt-4o"],
                },
                "kimi": {
                    "type": "kimi",
                    "base_url": "https://api.moonshot.cn/v1",
                    "api_key_env": "KIMI_API_KEY",
                    "default_model": "kimi-k2.5",
                },
            },
            "model_overrides": {
                "kimi-k2.5": {"temperature": 1.0, "top_p": 0.95},
            },
        }

    def _make(self, tmp_path, **extra):
        """Build an :class:`AgentConfig` with the stub catalog pre-injected."""
        kwargs = dict(
            nodes={"test": NodeConfig(model="test-model", input=None)},
            home=str(tmp_path / "datus_home"),
            target="legacy",
            models={
                "legacy": {
                    "type": "openai",
                    "api_key": "legacy-key",
                    "model": "legacy-model",
                    "base_url": "https://legacy.example.com",
                }
            },
            project_root=str(tmp_path),
            skip_init_dirs=True,
        )
        kwargs.update(extra)
        cfg = AgentConfig(**kwargs)
        cfg.set_provider_catalog(self._stub_catalog())
        return cfg

    # ── Legacy dispatch unchanged ──────────────────────────────────

    def test_active_model_legacy_path_unchanged(self, tmp_path):
        cfg = self._make(tmp_path)
        active = cfg.active_model()
        assert isinstance(active, ModelConfig)
        assert active.model == "legacy-model"
        assert active.api_key == "legacy-key"

    # ── Provider-level dispatch ────────────────────────────────────

    def test_provider_level_target_synthesizes_model_config(self, tmp_path):
        cfg = self._make(
            tmp_path,
            providers={"openai": {"api_key": "sk-test"}},
            target_provider="openai",
            target_model="gpt-4.1",
        )
        active = cfg.active_model()
        assert active.type == "openai"
        assert active.api_key == "sk-test"
        assert active.model == "gpt-4.1"
        assert active.base_url == "https://api.openai.com/v1"

    def test_model_overrides_applied_when_synthesizing(self, tmp_path):
        cfg = self._make(
            tmp_path,
            providers={"kimi": {"api_key": "km-test"}},
            target_provider="kimi",
            target_model="kimi-k2.5",
        )
        active = cfg.active_model()
        assert active.temperature == 1.0
        assert active.top_p == 0.95

    def test_env_fallback_used_when_user_api_key_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
        cfg = self._make(
            tmp_path,
            providers={"openai": {}},  # no explicit api_key
            target_provider="openai",
            target_model="gpt-4.1",
        )
        active = cfg.active_model()
        assert active.api_key == "env-secret"

    def test_active_model_raises_when_nothing_is_configured(self, tmp_path):
        cfg = self._make(tmp_path, target="", models={})
        with pytest.raises(DatusException) as exc_info:
            cfg.active_model()
        assert "/model" in exc_info.value.message
        assert "datus init" in exc_info.value.message

    # ── Setters ────────────────────────────────────────────────────

    def test_set_active_provider_model_writes_project_config(self, tmp_path):
        cfg = self._make(tmp_path)
        cfg.set_active_provider_model("openai", "gpt-4.1")
        # In-memory target now routes through provider synthesis.
        assert cfg._target_provider == "openai"
        assert cfg._target_model == "gpt-4.1"

        project_cfg = tmp_path / ".datus" / "config.yml"
        import yaml

        payload = yaml.safe_load(project_cfg.read_text(encoding="utf-8"))
        assert payload["target"] == {"provider": "openai", "model": "gpt-4.1"}

    def test_set_active_custom_writes_custom_target(self, tmp_path):
        cfg = self._make(tmp_path)
        cfg.set_active_custom("legacy")
        assert cfg.target == "legacy"
        assert cfg._target_provider is None

        project_cfg = tmp_path / ".datus" / "config.yml"
        import yaml

        payload = yaml.safe_load(project_cfg.read_text(encoding="utf-8"))
        assert payload["target"] == {"custom": "legacy"}

    def test_set_active_custom_rejects_unknown_name(self, tmp_path):
        cfg = self._make(tmp_path)
        with pytest.raises(DatusException):
            cfg.set_active_custom("not-registered")

    def test_set_provider_config_mutates_in_memory(self, tmp_path):
        cfg = self._make(tmp_path)
        cfg.set_provider_config("kimi", api_key="km-new", base_url="https://custom", persist=False)
        assert cfg.providers["kimi"].api_key == "km-new"
        assert cfg.providers["kimi"].base_url == "https://custom"

    # ── provider_available ─────────────────────────────────────────

    def test_provider_available_true_when_user_api_key_set(self, tmp_path):
        cfg = self._make(tmp_path, providers={"openai": {"api_key": "sk-test"}})
        assert cfg.provider_available("openai") is True

    def test_provider_available_true_when_env_fallback_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
        cfg = self._make(tmp_path)
        assert cfg.provider_available("openai") is True

    def test_provider_available_false_when_no_credentials(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = self._make(tmp_path)
        assert cfg.provider_available("openai") is False

    # ── reasoning_effort ───────────────────────────────────────────

    def test_target_reasoning_effort_overrides_legacy_model(self, tmp_path):
        cfg = self._make(tmp_path, target_reasoning_effort="high")
        active = cfg.active_model()
        assert active.reasoning_effort == "high"

    def test_target_reasoning_effort_overrides_synthesized_model(self, tmp_path):
        cfg = self._make(
            tmp_path,
            providers={"openai": {"api_key": "sk-test"}},
            target_provider="openai",
            target_model="gpt-4.1",
            target_reasoning_effort="low",
        )
        active = cfg.active_model()
        assert active.reasoning_effort == "low"

    def test_target_reasoning_effort_off_clears_enable_thinking(self, tmp_path):
        cfg = self._make(
            tmp_path,
            models={
                "legacy": {
                    "type": "openai",
                    "api_key": "k",
                    "model": "legacy-model",
                    "enable_thinking": True,
                }
            },
            target_reasoning_effort="off",
        )
        active = cfg.active_model()
        assert active.reasoning_effort == "off"
        assert active.enable_thinking is False

    def test_global_reasoning_effort_kwarg_falls_back(self, tmp_path):
        """Top-level ``reasoning_effort`` in agent.yml acts as a default when
        no project-level override is set."""
        cfg = self._make(tmp_path, reasoning_effort="medium")
        active = cfg.active_model()
        assert active.reasoning_effort == "medium"

    def test_project_reasoning_effort_wins_over_global(self, tmp_path):
        cfg = self._make(
            tmp_path,
            reasoning_effort="medium",
            target_reasoning_effort="high",
        )
        active = cfg.active_model()
        assert active.reasoning_effort == "high"

    def test_set_active_reasoning_effort_persists_to_project_config(self, tmp_path):
        cfg = self._make(tmp_path)
        cfg.set_active_reasoning_effort("high")
        assert cfg._target_reasoning_effort == "high"

        project_cfg = tmp_path / ".datus" / "config.yml"
        import yaml

        payload = yaml.safe_load(project_cfg.read_text(encoding="utf-8"))
        assert payload["reasoning_effort"] == "high"

    def test_set_active_reasoning_effort_rejects_invalid_value(self, tmp_path):
        cfg = self._make(tmp_path)
        with pytest.raises(DatusException):
            cfg.set_active_reasoning_effort("nuclear")

    def test_set_active_reasoning_effort_none_clears_override(self, tmp_path):
        cfg = self._make(tmp_path, target_reasoning_effort="high")
        cfg.set_active_reasoning_effort(None, persist=False)
        assert cfg._target_reasoning_effort is None

    def test_model_overrides_reasoning_effort_picked_up(self, tmp_path):
        """``providers.yml`` ``model_overrides.<model>.reasoning_effort`` flows
        through ``_synthesize_model`` into the resolved :class:`ModelConfig`."""
        cfg = self._make(
            tmp_path,
            providers={"openai": {"api_key": "sk"}},
            target_provider="openai",
            target_model="gpt-4.1",
        )
        # Inject a reasoning_effort override for gpt-4.1 in the catalog.
        catalog = cfg.provider_catalog
        catalog.setdefault("model_overrides", {})["gpt-4.1"] = {"reasoning_effort": "high"}
        cfg.set_provider_catalog(catalog)
        active = cfg.active_model()
        assert active.reasoning_effort == "high"


# ---------------------------------------------------------------------------
# set_agentic_node_override
# ---------------------------------------------------------------------------


class TestSetAgenticNodeOverride:
    """Exercises the override helper wired to the unified agent TUI.

    Only the in-memory contract is asserted here (``persist=False``). The
    on-disk path is exercised indirectly via ``_persist_agentic_node_override``
    tests below.
    """

    def _make(self, tmp_path, **extra):
        kwargs = dict(
            nodes={"test": NodeConfig(model="test-model", input=None)},
            home=str(tmp_path / "home"),
            target="legacy",
            models={"legacy": {"type": "openai", "api_key": "k", "model": "legacy-model"}},
            project_root=str(tmp_path),
            skip_init_dirs=True,
        )
        kwargs.update(extra)
        return AgentConfig(**kwargs)

    def test_write_both_fields_from_empty(self, tmp_path):
        cfg = self._make(tmp_path)
        cfg.set_agentic_node_override("gen_sql", model="legacy", max_turns=25, persist=False)
        entry = cfg.agentic_nodes["gen_sql"]
        assert entry["model"] == "legacy"
        assert entry["max_turns"] == 25
        # ``system_prompt`` is auto-filled so the YAML stays round-trippable.
        assert entry["system_prompt"] == "gen_sql"

    def test_clear_model_preserves_max_turns(self, tmp_path):
        """Passing ``model=None`` clears only that key; max_turns stays
        put unless it is also set to ``None``."""
        cfg = self._make(
            tmp_path,
            agentic_nodes={"gen_sql": {"system_prompt": "gen_sql", "model": "legacy", "max_turns": 25}},
        )
        cfg.set_agentic_node_override("gen_sql", model=None, max_turns=42, persist=False)
        entry = cfg.agentic_nodes["gen_sql"]
        assert "model" not in entry
        assert entry["max_turns"] == 42

    def test_clear_max_turns_preserves_model(self, tmp_path):
        cfg = self._make(
            tmp_path,
            agentic_nodes={"gen_sql": {"system_prompt": "gen_sql", "model": "legacy", "max_turns": 25}},
        )
        cfg.set_agentic_node_override("gen_sql", model="legacy", max_turns=None, persist=False)
        entry = cfg.agentic_nodes["gen_sql"]
        assert entry["model"] == "legacy"
        assert "max_turns" not in entry

    def test_existing_sibling_keys_are_preserved(self, tmp_path):
        """Overrides must never clobber user-authored fields under the
        same node (``tools``, ``rules``, ``scoped_context``)."""
        cfg = self._make(
            tmp_path,
            agentic_nodes={
                "my_custom": {
                    "system_prompt": "my_custom",
                    "tools": "db_tools",
                    "rules": ["r1"],
                    "scoped_context": {"datasource": "ds1"},
                }
            },
        )
        cfg.set_agentic_node_override("my_custom", model="legacy", max_turns=15, persist=False)
        entry = cfg.agentic_nodes["my_custom"]
        assert entry["tools"] == "db_tools"
        assert entry["rules"] == ["r1"]
        assert entry["scoped_context"] == {"datasource": "ds1"}
        assert entry["model"] == "legacy"
        assert entry["max_turns"] == 15

    def test_max_turns_coerced_to_int(self, tmp_path):
        cfg = self._make(tmp_path)
        cfg.set_agentic_node_override("gen_sql", model=None, max_turns="30", persist=False)  # type: ignore[arg-type]
        assert cfg.agentic_nodes["gen_sql"]["max_turns"] == 30


class TestValidationConfigFromDict:
    """``ValidationConfig.from_dict`` must survive malformed YAML input.

    YAML can produce non-mapping values for ``validation:`` (``false``,
    ``[]``, a stray scalar). Those must fall back to defaults — otherwise
    AgentConfig construction crashes with a raw AttributeError when the
    user pastes a broken config (reviewer feedback, PR #657).
    """

    def test_none_returns_defaults(self):
        cfg = ValidationConfig.from_dict(None)
        assert cfg.skill_validators_enabled is True
        assert cfg.max_retries == 3

    def test_empty_dict_returns_defaults(self):
        cfg = ValidationConfig.from_dict({})
        assert cfg.skill_validators_enabled is True
        assert cfg.max_retries == 3

    def test_false_scalar_does_not_crash(self):
        # YAML: ``validation: false``  → caller hands us ``False``.
        cfg = ValidationConfig.from_dict(False)
        assert cfg.skill_validators_enabled is True
        assert cfg.max_retries == 3

    def test_list_does_not_crash(self):
        # YAML: ``validation: []``
        cfg = ValidationConfig.from_dict([])
        assert cfg.skill_validators_enabled is True
        assert cfg.max_retries == 3

    def test_string_does_not_crash(self):
        cfg = ValidationConfig.from_dict("yes please")
        assert cfg.skill_validators_enabled is True
        assert cfg.max_retries == 3

    def test_valid_dict_read_correctly(self):
        cfg = ValidationConfig.from_dict({"skill_validators_enabled": False, "max_retries": 5})
        assert cfg.skill_validators_enabled is False
        assert cfg.max_retries == 5

    def test_negative_retries_clamped(self):
        cfg = ValidationConfig.from_dict({"max_retries": -4})
        assert cfg.max_retries == 0

    def test_non_numeric_retries_falls_back_to_default(self):
        cfg = ValidationConfig.from_dict({"max_retries": "oops"})
        assert cfg.max_retries == 3
