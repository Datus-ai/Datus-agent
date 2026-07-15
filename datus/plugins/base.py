# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""The datus-plugin contract: a declarative ``datus-plugin.yml`` manifest.

A *plugin* is an installable package that extends datus with a ``datus
<plugin> ...`` CLI subcommand plus optional bundled skills, a system-prompt
section, bash-permission rules, tool transformers and a profile config schema
— all declared in a single ``datus-plugin.yml`` shipped **inside** the
importable package. A plugin package never imports ``datus.*`` and ships no
plugin class: the only Python it provides are plain functions referenced from
the manifest.

Discovery
=========

The package registers ONE entry point under the ``datus.plugins`` group whose
value is the package name — a pure name → package mapping, not a code
reference::

    [project.entry-points."datus.plugins"]
    hello = "datus_plugin_hello"

datus locates the package directory with ``importlib.util.find_spec`` (which
does **not** execute ``__init__.py``) and reads ``datus-plugin.yml`` from it.
Collecting skills, permissions, prompt sections and config schemas therefore
never runs plugin code; only the ``cli`` entry (on ``datus <name> ...``
dispatch) and declared ``tool_transformers`` are imported, lazily. The
entry-point name is the plugin name: it becomes the ``datus <name>``
subcommand and the ``agent.plugins.<name>`` config key. It must match
``^[A-Za-z0-9][A-Za-z0-9._-]*$`` and must not be a reserved subcommand
(``upgrade``, ``skill``, ``plugin``). A legacy ``pkg.module:Class`` entry
point (non-empty attribute part) is rejected with a migration warning.

Manifest reference
==================

Only ``manifest_version`` is required — every other key is optional::

    manifest_version: 1
    description: "One-line summary shown by `datus plugin info`."

    cli: datus_plugin_hello.cli:main
    # Dotted code ref "module.path:function", called as
    # ``main(argv: list[str], profile: dict) -> int | None``. ``argv`` is
    # everything after ``datus <name>`` with datus' own ``--profile`` /
    # ``--config`` already stripped; ``profile`` is the resolved
    # ``agent.plugins.<name>.<profile>`` dict (env-expanded, ``{}`` when
    # unconfigured). Return an exit code (``None`` is treated as ``0``).
    # Without a ``cli`` key ``datus <name>`` exits 2.

    tool_transformers:
      "db_tools.execute_sql": datus_plugin_hello.transformers:enforce_scope
      "execute_sql":
        - datus_plugin_hello.transformers:audit_args
    # Tool pattern -> dotted ref or list of refs. Pattern syntax matches the
    # proxy layer: bare tool name (``execute_sql``) or ``category.method``
    # with fnmatch globs (``db_tools.*``). Each ref resolves to a callable
    # ``transformer(tool_name, args, context) -> dict`` (sync or async):
    # return the possibly-modified argument dict to continue, or raise to
    # deny the call — the tool then never runs and the model receives the
    # exception message as a normal tool failure (fail closed). ``context``
    # is a plain dict with ``node_name``, ``principal``, ``project_root``
    # and ``agent_config``. Transformers only cover LLM-driven tool calls;
    # direct Python invocations bypass them.

    permissions:
      normal:
        allow: ["greet:*"]
        ask: ["config set:*"]
      auto:
        allow: ["greet:*", "config set:*"]
    # Bash-permission rules for the plugin's own CLI namespace, keyed by
    # permission profile. Only ``normal`` and ``auto`` are accepted
    # (``dangerous`` ignores bash rules by design); actions are ``allow`` /
    # ``ask`` / ``deny``. Patterns are RELATIVE to the namespace — datus
    # prefixes each with ``datus <name> ``, so ``greet:*`` becomes
    # ``datus <name> greet:*`` and a plugin can never affect commands outside
    # its own namespace. Pattern syntax matches ``permissions.bash_commands``
    # in agent.yml (``cmd`` exact, ``cmd:*`` prefix, ``cmd:glob`` prefix +
    # first-arg glob, ``:*`` the whole namespace). These rules gate only the
    # **agent's** bash tool; a human invoking the CLI directly is never
    # gated, and a plugin rule can never loosen a user ``deny``.

    system_prompt: prompts/system.md.j2
    # Path (relative to the package dir) of a Jinja2 template rendered into
    # the agent's system prompt. Render context: ``plugin_name`` (str),
    # ``profiles`` (dict[str, dict] — the project-activated profiles carrying
    # ONLY fields declared non-secret in ``config_schema``; without a schema,
    # profile names map to empty dicts) and ``config_path`` (str | None).
    # Profile values are env-expanded at load, so secret stripping is
    # structural: undeclared or ``x-secret`` fields never reach the template.
    # An installed-but-unconfigured plugin renders with ``profiles == {}`` —
    # use ``{% if profiles %}`` to emit setup guidance (pointing at the
    # bundled setup skill) instead of disappearing from the prompt. datus
    # prepends its own ``## Plugins`` preamble naming the loaded config file,
    # so the template must not hard-code config paths.

    skills: skills
    # Path (relative to the package dir) of a bundled skill directory.
    # Plugins that need configuration should bundle a ``<name>-setup`` skill
    # describing the profile YAML shape, with secrets as ``${ENV_VAR}``
    # placeholders, never literal values.

    config_schema:
      type: object
      required: [api_key]
      properties:
        api_key:
          type: string
          description: "Console API key"
          x-secret: true
        base_url:
          type: string
          description: "API base URL"
          default: "https://api.example.com"
    # Inline JSON Schema (an object schema) describing ONE profile of
    # ``agent.plugins.<name>``. Drives the ``/plugins`` TUI form (property
    # order == field order; ``required`` and ``default`` are honoured) and
    # validates a candidate profile before it is saved. Mark secret fields
    # with ``x-secret: true`` — the TUI masks them and the system-prompt
    # renderer strips them. TUI-entered values are strings, so prefer
    # ``type: string`` with ``pattern``/``enum``; values containing
    # ``${ENV_VAR}`` placeholders are exempt from value-level validation.

The whole plugin system is gated by ``agent.plugins_enabled`` (default
``true``) and by the project's ``plugins:`` whitelist in
``./.datus/config.yml``. Manifest parsing is defensive: a malformed section is
warned about and dropped while the rest of the manifest stays usable; only a
missing or unsupported ``manifest_version`` (or an unreadable file) rejects
the manifest as a whole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

MANIFEST_FILENAME = "datus-plugin.yml"
MANIFEST_VERSION = 1

# "module.path:attr.path" — both sides dotted identifier chains.
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_CODE_REF_RE = re.compile(rf"^{_IDENT}(\.{_IDENT})*:{_IDENT}(\.{_IDENT})*$")

_MANIFEST_KEYS = frozenset(
    {
        "manifest_version",
        "description",
        "cli",
        "tool_transformers",
        "permissions",
        "system_prompt",
        "skills",
        "config_schema",
    }
)


@dataclass(frozen=True)
class PluginManifest:
    """Parsed, per-section-validated view of a plugin's ``datus-plugin.yml``.

    Internal to datus — plugin authors only write the YAML. ``permissions``
    keeps the raw (dict-shaped) declaration; the authoritative per-profile /
    per-action validation lives in
    :func:`datus.plugins.registry.collect_plugin_cli_permissions`.
    """

    name: str
    package_dir: Path
    description: Optional[str] = None
    cli: Optional[str] = None
    tool_transformers: Dict[str, List[str]] = field(default_factory=dict)
    permissions: Dict[str, Any] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    skills: Optional[str] = None
    config_schema: Optional[Dict[str, Any]] = None


def parse_code_ref(ref: Any) -> Optional[Tuple[str, str]]:
    """Split a dotted code reference ``"module.path:attr"`` into its parts.

    Returns ``(module, attr)`` or ``None`` when ``ref`` is not a string of the
    expected shape. Never raises.
    """
    if not isinstance(ref, str):
        return None
    candidate = ref.strip()
    if not _CODE_REF_RE.match(candidate):
        return None
    module, attr = candidate.split(":", 1)
    return module, attr


def _parse_str(data: Dict[str, Any], key: str, name: str) -> Optional[str]:
    """Return ``data[key]`` as a stripped non-empty string, else ``None``."""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        logger.warning("Plugin %r manifest %r must be a non-empty string, got %r; ignoring.", name, key, value)
        return None
    return value.strip()


def _parse_cli(data: Dict[str, Any], name: str) -> Optional[str]:
    """Validate the ``cli`` dotted code ref."""
    value = _parse_str(data, "cli", name)
    if value is None:
        return None
    if parse_code_ref(value) is None:
        logger.warning(
            "Plugin %r manifest `cli` must be a dotted code ref like 'pkg.module:function', got %r; ignoring.",
            name,
            value,
        )
        return None
    return value


def _parse_rel_path(data: Dict[str, Any], key: str, name: str) -> Optional[str]:
    """Validate a package-relative path field (``skills`` / ``system_prompt``).

    Only shape-checks (non-empty string, not absolute). Escape containment
    (``..``) is enforced where the path is resolved against the package dir.
    """
    value = _parse_str(data, key, name)
    if value is None:
        return None
    if Path(value).is_absolute():
        logger.warning("Plugin %r manifest %r must be relative to the package dir, got %r; ignoring.", name, key, value)
        return None
    return value


def _parse_tool_transformers(data: Dict[str, Any], name: str) -> Dict[str, List[str]]:
    """Normalize ``tool_transformers`` to ``{pattern: [code refs]}``.

    Invalid patterns and refs are warned about and dropped individually so one
    bad entry never discards the plugin's other transformers.
    """
    declared = data.get("tool_transformers")
    if declared is None:
        return {}
    if not isinstance(declared, dict):
        logger.warning("Plugin %r manifest `tool_transformers` must be a mapping, got %r; ignoring.", name, declared)
        return {}
    normalized: Dict[str, List[str]] = {}
    for pattern, value in declared.items():
        if not isinstance(pattern, str) or not pattern.strip():
            logger.warning("Plugin %r tool_transformers has invalid pattern %r; ignoring.", name, pattern)
            continue
        refs = value if isinstance(value, list) else [value]
        valid: List[str] = []
        for ref in refs:
            if parse_code_ref(ref) is None:
                logger.warning(
                    "Plugin %r tool_transformers[%r] entry %r is not a dotted code ref; skipping it.",
                    name,
                    pattern,
                    ref,
                )
                continue
            valid.append(ref.strip())
        if valid:
            normalized[pattern.strip()] = valid
    return normalized


def _parse_permissions(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Shape-check ``permissions`` (must be a mapping); detail validation is
    deferred to ``collect_plugin_cli_permissions``."""
    declared = data.get("permissions")
    if declared is None:
        return {}
    if not isinstance(declared, dict):
        logger.warning("Plugin %r manifest `permissions` must be a mapping, got %r; ignoring.", name, declared)
        return {}
    return declared


def _parse_config_schema(data: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """Validate ``config_schema`` as a well-formed JSON Schema mapping."""
    declared = data.get("config_schema")
    if declared is None:
        return None
    if not isinstance(declared, dict):
        logger.warning("Plugin %r manifest `config_schema` must be a mapping, got %r; ignoring.", name, declared)
        return None
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(declared)
    except Exception as exc:  # noqa: BLE001 - a bad schema must not reject the manifest
        logger.warning("Plugin %r manifest `config_schema` is not a valid JSON Schema: %s; ignoring.", name, exc)
        return None
    return declared


def parse_manifest(data: Any, name: str, package_dir: Path) -> Optional[PluginManifest]:
    """Parse raw YAML data into a :class:`PluginManifest`.

    Salvages per section: a malformed field is warned about and dropped while
    the rest stays usable. Returns ``None`` only when the document root is not
    a mapping or ``manifest_version`` is missing/unsupported. Never raises.
    """
    if not isinstance(data, dict):
        logger.warning(
            "Plugin %r %s root must be a mapping, got %s; skipping.", name, MANIFEST_FILENAME, type(data).__name__
        )
        return None
    version = data.get("manifest_version")
    if version != MANIFEST_VERSION:
        if isinstance(version, int) and version > MANIFEST_VERSION:
            logger.warning(
                "Plugin %r declares manifest_version %s, which requires a newer datus; skipping.", name, version
            )
        else:
            logger.warning(
                "Plugin %r %s must declare `manifest_version: %s`, got %r; skipping.",
                name,
                MANIFEST_FILENAME,
                MANIFEST_VERSION,
                version,
            )
        return None
    unknown = set(data) - _MANIFEST_KEYS
    if unknown:
        logger.warning("Plugin %r manifest has unknown keys %s; ignoring them.", name, sorted(unknown))
    return PluginManifest(
        name=name,
        package_dir=Path(package_dir),
        description=_parse_str(data, "description", name),
        cli=_parse_cli(data, name),
        tool_transformers=_parse_tool_transformers(data, name),
        permissions=_parse_permissions(data, name),
        system_prompt=_parse_rel_path(data, "system_prompt", name),
        skills=_parse_rel_path(data, "skills", name),
        config_schema=_parse_config_schema(data, name),
    )


def read_manifest_file(package_dir: Path, name: str) -> Optional[PluginManifest]:
    """Read and parse ``<package_dir>/datus-plugin.yml``. Never raises.

    Returns ``None`` (with a warning) when the file is missing, unreadable
    (e.g. the package lives inside a zip archive), not valid YAML, or rejected
    by :func:`parse_manifest`.
    """
    path = Path(package_dir) / MANIFEST_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Plugin %r has no %s in %s; skipping.", name, MANIFEST_FILENAME, package_dir)
        return None
    except OSError as exc:
        logger.warning("Plugin %r manifest %s is unreadable: %s; skipping.", name, path, exc)
        return None
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning("Plugin %r manifest %s is not valid YAML: %s; skipping.", name, path, exc)
        return None
    return parse_manifest(loaded, name, Path(package_dir))
