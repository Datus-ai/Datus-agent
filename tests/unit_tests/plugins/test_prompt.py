# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.plugins.prompt`` (Jinja2 rendering + secret stripping)."""

from datus.plugins.base import PluginManifest
from datus.plugins.prompt import render_plugin_prompt, strip_secret_fields

SCHEMA = {
    "type": "object",
    "properties": {
        "base_url": {"type": "string"},
        "api_key": {"type": "string", "x-secret": True},
        "region": {"type": "string"},
    },
}


# ---------------------------------------------------------------------------
# strip_secret_fields
# ---------------------------------------------------------------------------


def test_strip_keeps_declared_non_secret_fields():
    profiles = {"prod": {"base_url": "https://x", "api_key": "sk-REAL", "region": "us"}}
    assert strip_secret_fields(profiles, SCHEMA) == {"prod": {"base_url": "https://x", "region": "us"}}


def test_strip_drops_undeclared_fields():
    """Whitelist semantics: fields absent from the schema never pass through."""
    profiles = {"prod": {"base_url": "https://x", "password": "hunter2", "extra": "e"}}
    assert strip_secret_fields(profiles, SCHEMA) == {"prod": {"base_url": "https://x"}}


def test_strip_without_schema_yields_names_only():
    profiles = {"prod": {"api_key": "sk-REAL"}, "staging": {"api_key": "sk-2"}}
    assert strip_secret_fields(profiles, None) == {"prod": {}, "staging": {}}


def test_strip_tolerates_malformed_input():
    assert strip_secret_fields("not-a-dict", SCHEMA) == {}
    assert strip_secret_fields({"prod": "not-a-dict"}, SCHEMA) == {"prod": {}}
    assert strip_secret_fields({42: {"base_url": "x"}}, SCHEMA) == {}


def test_strip_x_secret_must_be_literal_true():
    schema = {"properties": {"a": {"x-secret": "yes"}, "b": {"x-secret": False}}}
    profiles = {"p": {"a": "1", "b": "2"}}
    # Only a literal ``True`` marks a secret; other values keep the field.
    assert strip_secret_fields(profiles, schema) == {"p": {"a": "1", "b": "2"}}


NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "base_url": {"type": "string"},
        "s3": {
            "type": "object",
            "properties": {
                "region": {"type": "string"},
                "secret_access_key": {"type": "string", "x-secret": True},
            },
        },
        "credentials": {"type": "object", "x-secret": True, "properties": {"user": {"type": "string"}}},
    },
}


def test_strip_recurses_into_declared_nested_objects():
    """Nested ``x-secret`` leaves and undeclared nested keys are stripped."""
    profiles = {
        "prod": {
            "base_url": "https://x",
            "s3": {"region": "us", "secret_access_key": "REAL-SECRET", "endpoint": "http://minio"},
        }
    }
    assert strip_secret_fields(profiles, NESTED_SCHEMA) == {"prod": {"base_url": "https://x", "s3": {"region": "us"}}}


def test_strip_drops_block_level_secret_objects():
    profiles = {"prod": {"credentials": {"user": "root"}, "base_url": "https://x"}}
    assert strip_secret_fields(profiles, NESTED_SCHEMA) == {"prod": {"base_url": "https://x"}}


def test_strip_nested_non_dict_value_yields_empty_object():
    """A declared object whose stored value is not a dict exposes nothing."""
    profiles = {"prod": {"s3": "oops"}}
    assert strip_secret_fields(profiles, NESTED_SCHEMA) == {"prod": {"s3": {}}}


def test_strip_recurses_three_levels():
    schema = {
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {
                    "b": {
                        "type": "object",
                        "properties": {
                            "keep": {"type": "string"},
                            "drop": {"type": "string", "x-secret": True},
                        },
                    }
                },
            }
        },
    }
    profiles = {"p": {"a": {"b": {"keep": "1", "drop": "2", "undeclared": "3"}}}}
    assert strip_secret_fields(profiles, schema) == {"p": {"a": {"b": {"keep": "1"}}}}


# ---------------------------------------------------------------------------
# render_plugin_prompt
# ---------------------------------------------------------------------------


def _manifest(tmp_path, template_body, *, rel="prompt.md.j2", schema=SCHEMA, declared_rel=None):
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_text(template_body, encoding="utf-8")
    return PluginManifest(
        name="hello",
        package_dir=tmp_path,
        system_prompt=declared_rel or rel,
        config_schema=schema,
    )


def test_render_with_profiles(tmp_path):
    template = (
        "## {{ plugin_name }}\n"
        "{% if profiles %}"
        "{% for name, p in profiles.items() %}- {{ name }}: {{ p.base_url }}\n{% endfor %}"
        "{% else %}not configured{% endif %}"
    )
    manifest = _manifest(tmp_path, template)
    profiles = {
        "prod": {"base_url": "https://p", "api_key": "sk-REAL"},
        "staging": {"base_url": "https://s", "api_key": "sk-2"},
    }
    rendered = render_plugin_prompt(manifest, profiles)
    assert "## hello" in rendered
    assert "- prod: https://p" in rendered
    assert "- staging: https://s" in rendered
    assert "sk-REAL" not in rendered


def test_render_unconfigured_branch(tmp_path):
    template = "{% if profiles %}configured{% else %}run the hello-setup skill{% endif %}"
    manifest = _manifest(tmp_path, template)
    assert render_plugin_prompt(manifest, {}) == "run the hello-setup skill"


def test_render_secrets_unreachable_even_if_referenced(tmp_path):
    """A template asking for a secret field fails (StrictUndefined) → skipped."""
    manifest = _manifest(tmp_path, "key={{ profiles['prod'].api_key }}")
    rendered = render_plugin_prompt(manifest, {"prod": {"api_key": "sk-REAL"}})
    assert rendered is None


def test_render_undefined_variable_skipped(tmp_path, caplog):
    manifest = _manifest(tmp_path, "{{ nonexistent_variable }}")
    with caplog.at_level("WARNING"):
        assert render_plugin_prompt(manifest, {}) is None
    assert "failed to render" in caplog.text


def test_render_syntax_error_skipped(tmp_path, caplog):
    manifest = _manifest(tmp_path, "{% if %}")
    with caplog.at_level("WARNING"):
        assert render_plugin_prompt(manifest, {}) is None
    assert "failed to render" in caplog.text


def test_render_missing_template_skipped(tmp_path, caplog):
    manifest = PluginManifest(name="hello", package_dir=tmp_path, system_prompt="missing.j2")
    with caplog.at_level("WARNING"):
        assert render_plugin_prompt(manifest, {}) is None
    assert "does not exist" in caplog.text


def test_render_path_escape_rejected(tmp_path, caplog):
    outside = tmp_path / "outside.j2"
    outside.write_text("ESCAPED", encoding="utf-8")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    manifest = PluginManifest(name="hello", package_dir=pkg, system_prompt="../outside.j2")
    with caplog.at_level("WARNING"):
        assert render_plugin_prompt(manifest, {}) is None
    assert "escapes the package directory" in caplog.text


def test_render_no_template_declared(tmp_path):
    manifest = PluginManifest(name="hello", package_dir=tmp_path)
    assert render_plugin_prompt(manifest, {"p": {}}) is None


def test_render_empty_output_becomes_none(tmp_path):
    manifest = _manifest(tmp_path, "{% if false %}never{% endif %}")
    assert render_plugin_prompt(manifest, {}) is None


def test_render_config_path_in_context(tmp_path):
    manifest = _manifest(tmp_path, "cfg={{ config_path }}")
    assert render_plugin_prompt(manifest, {}, config_path="/srv/agent.yml") == "cfg=/srv/agent.yml"


def test_render_config_mutable_in_context(tmp_path):
    manifest = _manifest(tmp_path, "{% if config_mutable %}RW{% else %}RO{% endif %}")
    assert render_plugin_prompt(manifest, {}, config_mutable=True) == "RW"
    assert render_plugin_prompt(manifest, {}, config_mutable=False) == "RO"


def test_render_config_mutable_defaults_true(tmp_path):
    manifest = _manifest(tmp_path, "{% if config_mutable %}RW{% else %}RO{% endif %}")
    assert render_plugin_prompt(manifest, {}) == "RW"


def test_render_template_without_config_mutable_still_renders(tmp_path):
    """Back-compat: templates that never reference the new variable render
    unchanged under both values (StrictUndefined only fails on reference)."""
    manifest = _manifest(tmp_path, "{% if profiles %}configured{% else %}unconfigured{% endif %}")
    assert render_plugin_prompt(manifest, {}, config_mutable=True) == "unconfigured"
    assert render_plugin_prompt(manifest, {}, config_mutable=False) == "unconfigured"


def test_render_nested_template_dir(tmp_path):
    manifest = _manifest(tmp_path, "nested ok", rel="prompts/system.md.j2")
    assert render_plugin_prompt(manifest, {}) == "nested ok"
