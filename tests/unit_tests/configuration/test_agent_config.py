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
    BenchmarkConfig,
    DbConfig,
    DocumentConfig,
    ModelConfig,
    file_stem_from_uri,
    load_model_config,
    rag_storage_path,
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
# rag_storage_path
# ---------------------------------------------------------------------------


class TestRagStoragePath:
    def test_unified_datus_db_path(self):
        path = rag_storage_path("/data")
        assert path.endswith("datus_db")

    def test_includes_base_path(self):
        path = rag_storage_path("/custom/base")
        assert "/custom/base" in path


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
        if cfg.extra:
            assert "some_none_field" not in cfg.extra


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
        cfg.validate()  # should not raise

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
