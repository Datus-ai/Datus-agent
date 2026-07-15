# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.plugins.base`` (manifest parsing)."""

from pathlib import Path

from datus.plugins.base import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    PluginManifest,
    parse_code_ref,
    parse_manifest,
    read_manifest_file,
)

PKG = Path("/tmp/pkg")


# ---------------------------------------------------------------------------
# parse_code_ref
# ---------------------------------------------------------------------------


def test_parse_code_ref_valid_forms():
    assert parse_code_ref("pkg.mod:func") == ("pkg.mod", "func")
    assert parse_code_ref("pkg:func") == ("pkg", "func")
    assert parse_code_ref("pkg.mod:Class.method") == ("pkg.mod", "Class.method")
    assert parse_code_ref("  pkg.mod:func  ") == ("pkg.mod", "func")


def test_parse_code_ref_invalid_forms():
    assert parse_code_ref(None) is None
    assert parse_code_ref(42) is None
    assert parse_code_ref("") is None
    assert parse_code_ref("no_colon") is None
    assert parse_code_ref(":func") is None
    assert parse_code_ref("pkg:") is None
    assert parse_code_ref("pkg mod:func") is None
    assert parse_code_ref("pkg.mod:func()") is None
    assert parse_code_ref("pkg..mod:func") is None
    assert parse_code_ref("1pkg:func") is None


# ---------------------------------------------------------------------------
# parse_manifest — version gate and root shape
# ---------------------------------------------------------------------------


def test_parse_manifest_minimal():
    manifest = parse_manifest({"manifest_version": MANIFEST_VERSION}, "hello", PKG)
    assert isinstance(manifest, PluginManifest)
    assert manifest.name == "hello"
    assert manifest.package_dir == PKG
    assert manifest.cli is None
    assert manifest.tool_transformers == {}
    assert manifest.permissions == {}
    assert manifest.system_prompt is None
    assert manifest.skills is None
    assert manifest.config_schema is None


def test_parse_manifest_non_dict_root_rejected(caplog):
    with caplog.at_level("WARNING"):
        assert parse_manifest(["not", "a", "dict"], "hello", PKG) is None
        assert parse_manifest("scalar", "hello", PKG) is None
        assert parse_manifest(None, "hello", PKG) is None
    assert "root must be a mapping" in caplog.text


def test_parse_manifest_missing_version_rejected(caplog):
    with caplog.at_level("WARNING"):
        assert parse_manifest({"cli": "pkg.mod:main"}, "hello", PKG) is None
    assert "manifest_version" in caplog.text


def test_parse_manifest_newer_version_rejected(caplog):
    with caplog.at_level("WARNING"):
        assert parse_manifest({"manifest_version": MANIFEST_VERSION + 1}, "hello", PKG) is None
    assert "newer datus" in caplog.text


def test_parse_manifest_non_int_version_rejected():
    assert parse_manifest({"manifest_version": "1"}, "hello", PKG) is None


def test_parse_manifest_unknown_keys_warned_but_kept(caplog):
    with caplog.at_level("WARNING"):
        manifest = parse_manifest({"manifest_version": 1, "skil": "skills", "cli": "pkg.mod:main"}, "hello", PKG)
    assert manifest.cli == "pkg.mod:main"
    assert "skil" in caplog.text


# ---------------------------------------------------------------------------
# parse_manifest — per-section salvage
# ---------------------------------------------------------------------------


def test_parse_manifest_full():
    data = {
        "manifest_version": 1,
        "description": "Manage things.",
        "cli": "pkg.cli:main",
        "tool_transformers": {
            "db_tools.execute_sql": "pkg.tf:enforce",
            "execute_sql": ["pkg.tf:audit", "pkg.tf:enforce"],
        },
        "permissions": {"normal": {"allow": ["greet:*"]}},
        "system_prompt": "prompts/system.md.j2",
        "skills": "skills",
        "config_schema": {
            "type": "object",
            "required": ["api_key"],
            "properties": {"api_key": {"type": "string", "x-secret": True}},
        },
    }
    manifest = parse_manifest(data, "hello", PKG)
    assert manifest.description == "Manage things."
    assert manifest.cli == "pkg.cli:main"
    assert manifest.tool_transformers == {
        "db_tools.execute_sql": ["pkg.tf:enforce"],
        "execute_sql": ["pkg.tf:audit", "pkg.tf:enforce"],
    }
    assert manifest.permissions == {"normal": {"allow": ["greet:*"]}}
    assert manifest.system_prompt == "prompts/system.md.j2"
    assert manifest.skills == "skills"
    assert manifest.config_schema["required"] == ["api_key"]


def test_parse_manifest_bad_section_does_not_kill_others(caplog):
    """A malformed permissions block is dropped while cli stays usable."""
    data = {
        "manifest_version": 1,
        "cli": "pkg.cli:main",
        "permissions": "not-a-dict",
        "tool_transformers": ["not-a-dict"],
        "skills": 42,
    }
    with caplog.at_level("WARNING"):
        manifest = parse_manifest(data, "hello", PKG)
    assert manifest.cli == "pkg.cli:main"
    assert manifest.permissions == {}
    assert manifest.tool_transformers == {}
    assert manifest.skills is None


def test_parse_manifest_invalid_cli_ref_dropped(caplog):
    with caplog.at_level("WARNING"):
        manifest = parse_manifest({"manifest_version": 1, "cli": "not a ref"}, "hello", PKG)
    assert manifest.cli is None
    assert "dotted code ref" in caplog.text


def test_parse_manifest_absolute_paths_dropped(caplog):
    data = {"manifest_version": 1, "skills": "/etc/skills", "system_prompt": "/etc/prompt.j2"}
    with caplog.at_level("WARNING"):
        manifest = parse_manifest(data, "hello", PKG)
    assert manifest.skills is None
    assert manifest.system_prompt is None
    assert "relative to the package dir" in caplog.text


def test_parse_manifest_transformer_entries_salvaged(caplog):
    data = {
        "manifest_version": 1,
        "tool_transformers": {
            "ok": "pkg.tf:good",
            "mixed": ["pkg.tf:good", "not a ref", 42],
            "": "pkg.tf:good",
            "all_bad": ["nope"],
        },
    }
    with caplog.at_level("WARNING"):
        manifest = parse_manifest(data, "hello", PKG)
    assert manifest.tool_transformers == {"ok": ["pkg.tf:good"], "mixed": ["pkg.tf:good"]}


def test_parse_manifest_invalid_config_schema_dropped(caplog):
    data = {"manifest_version": 1, "config_schema": {"type": "object", "required": "not-a-list"}}
    with caplog.at_level("WARNING"):
        manifest = parse_manifest(data, "hello", PKG)
    assert manifest.config_schema is None
    assert "not a valid JSON Schema" in caplog.text


def test_parse_manifest_non_dict_config_schema_dropped():
    manifest = parse_manifest({"manifest_version": 1, "config_schema": ["a"]}, "hello", PKG)
    assert manifest.config_schema is None


# ---------------------------------------------------------------------------
# read_manifest_file
# ---------------------------------------------------------------------------


def test_read_manifest_file_roundtrip(tmp_path):
    (tmp_path / MANIFEST_FILENAME).write_text(
        "manifest_version: 1\ncli: pkg.cli:main\nskills: skills\n", encoding="utf-8"
    )
    manifest = read_manifest_file(tmp_path, "hello")
    assert manifest.cli == "pkg.cli:main"
    assert manifest.skills == "skills"
    assert manifest.package_dir == tmp_path


def test_read_manifest_file_missing_returns_none(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        assert read_manifest_file(tmp_path, "hello") is None
    assert MANIFEST_FILENAME in caplog.text


def test_read_manifest_file_invalid_yaml_returns_none(tmp_path, caplog):
    (tmp_path / MANIFEST_FILENAME).write_text("cli: [unclosed", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert read_manifest_file(tmp_path, "hello") is None
    assert "not valid YAML" in caplog.text


def test_read_manifest_file_preserves_property_order(tmp_path):
    """YAML mapping order must survive into config_schema (drives TUI field order)."""
    (tmp_path / MANIFEST_FILENAME).write_text(
        "manifest_version: 1\n"
        "config_schema:\n"
        "  type: object\n"
        "  properties:\n"
        "    zeta: {type: string}\n"
        "    alpha: {type: string}\n"
        "    mid: {type: string}\n",
        encoding="utf-8",
    )
    manifest = read_manifest_file(tmp_path, "hello")
    assert list(manifest.config_schema["properties"]) == ["zeta", "alpha", "mid"]
