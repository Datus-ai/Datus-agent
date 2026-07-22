# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.plugins.base`` (manifest parsing)."""

from pathlib import Path

from datus.plugins.base import (
    _MAX_COMMAND_DEPTH,
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    PluginCommand,
    PluginCommandArg,
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
# parse_manifest — commands (descriptive CLI catalogue)
# ---------------------------------------------------------------------------


def test_parse_manifest_commands_full():
    data = {
        "manifest_version": 1,
        "commands": [
            {
                "name": "sync",
                "description": "Sync tables",
                "args": [
                    {"name": "table", "required": True, "description": "table name"},
                    {"name": "--limit", "description": "max rows"},
                ],
            },
            {"name": "status"},
        ],
    }
    manifest = parse_manifest(data, "hello", PKG)
    assert manifest.commands == [
        PluginCommand(
            name="sync",
            description="Sync tables",
            args=[
                PluginCommandArg(name="table", required=True, description="table name"),
                PluginCommandArg(name="--limit", required=False, description="max rows"),
            ],
        ),
        PluginCommand(name="status", description=None, args=[]),
    ]


def test_parse_manifest_commands_default_empty():
    manifest = parse_manifest({"manifest_version": 1}, "hello", PKG)
    assert manifest.commands == []


def test_parse_manifest_commands_non_list_dropped(caplog):
    with caplog.at_level("WARNING"):
        manifest = parse_manifest({"manifest_version": 1, "commands": "sync"}, "hello", PKG)
    assert manifest.commands == []


def test_parse_manifest_commands_bad_entries_salvaged(caplog):
    """A command without a valid name, and a malformed arg, are dropped
    individually while the rest survives."""
    data = {
        "manifest_version": 1,
        "commands": [
            "not-a-dict",
            {"description": "no name here"},
            {
                "name": "sync",
                "args": [
                    "not-a-dict",
                    {"required": True},  # arg without name
                    {"name": "table", "required": True},
                ],
            },
        ],
    }
    with caplog.at_level("WARNING"):
        manifest = parse_manifest(data, "hello", PKG)
    assert manifest.commands == [
        PluginCommand(name="sync", description=None, args=[PluginCommandArg(name="table", required=True)])
    ]


def test_parse_manifest_commands_non_list_args_dropped(caplog):
    with caplog.at_level("WARNING"):
        manifest = parse_manifest(
            {"manifest_version": 1, "commands": [{"name": "sync", "args": "table"}]}, "hello", PKG
        )
    assert manifest.commands == [PluginCommand(name="sync")]


def test_parse_manifest_commands_nested_subcommands():
    """A command group nests subcommands (each with its own args) recursively."""
    data = {
        "manifest_version": 1,
        "commands": [
            {
                "name": "dags",
                "description": "DAG operations",
                "subcommands": [
                    {
                        "name": "trigger",
                        "args": [
                            {"name": "dag_id", "required": True},
                            {"name": "--conf", "description": "run config"},
                        ],
                    },
                    {"name": "list"},
                ],
            }
        ],
    }
    manifest = parse_manifest(data, "airflow", PKG)
    assert manifest.commands == [
        PluginCommand(
            name="dags",
            description="DAG operations",
            args=[],
            subcommands=[
                PluginCommand(
                    name="trigger",
                    args=[
                        PluginCommandArg(name="dag_id", required=True),
                        PluginCommandArg(name="--conf", description="run config"),
                    ],
                ),
                PluginCommand(name="list"),
            ],
        )
    ]


def test_parse_manifest_subcommands_bad_entries_salvaged(caplog):
    """Malformed subcommand entries are dropped individually, siblings survive."""
    data = {
        "manifest_version": 1,
        "commands": [
            {
                "name": "dags",
                "subcommands": [
                    "not-a-dict",
                    {"description": "no name"},
                    {"name": "list"},
                ],
            }
        ],
    }
    with caplog.at_level("WARNING"):
        manifest = parse_manifest(data, "airflow", PKG)
    assert manifest.commands == [PluginCommand(name="dags", subcommands=[PluginCommand(name="list")])]


def test_parse_manifest_subcommands_non_list_dropped(caplog):
    with caplog.at_level("WARNING"):
        manifest = parse_manifest(
            {"manifest_version": 1, "commands": [{"name": "dags", "subcommands": "list"}]}, "airflow", PKG
        )
    assert manifest.commands == [PluginCommand(name="dags", subcommands=[])]


def test_parse_manifest_subcommands_depth_capped(caplog):
    """Nesting deeper than the recursion cap is dropped rather than parsed forever."""
    # Build commands nested one level beyond the cap.
    entry: dict = {"name": "leaf"}
    for level in range(_MAX_COMMAND_DEPTH + 1):
        entry = {"name": f"g{level}", "subcommands": [entry]}
    with caplog.at_level("WARNING"):
        manifest = parse_manifest({"manifest_version": 1, "commands": [entry]}, "deep", PKG)

    # Walk down; the chain is truncated at the cap (deepest kept level has no subcommands).
    node = manifest.commands[0]
    depth = 1
    while node.subcommands:
        node = node.subcommands[0]
        depth += 1
    assert depth == _MAX_COMMAND_DEPTH
    assert any("deeper than" in r.message for r in caplog.records)


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
