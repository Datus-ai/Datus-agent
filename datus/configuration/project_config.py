# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Project-level ``./.datus/config.yml`` override.

A small, strict overlay on the base ``agent.yml`` that lets every project
pin a handful of values without copying the full config:

- ``target``: which LLM to use. Accepts three forms:
  - Legacy string, e.g. ``target: openai`` — selects ``agent.models.openai``.
  - Structured provider+model, e.g.
    ``target: {provider: openai, model: gpt-4.1}`` — selects provider-level
    ``agent.providers.openai`` and runs ``gpt-4.1``.
  - Structured custom, e.g. ``target: {custom: my-internal}`` — explicit
    alias for the legacy string form (selects ``agent.models.my-internal``).
- ``default_datasource``: which datasource to connect to on startup (must
  match a key under ``agent.services.datasources``)
- ``dashboard``: project-level default BI service (must match a key under
  ``agent.services.bi_platforms``). Resolved by ``BIFuncTool`` /
  ``AgentConfig.dashboard_config`` when no explicit ``bi_service`` is
  passed at the call site.
- ``scheduler``: project-level default scheduler service (must match a key
  under ``agent.services.schedulers``). Resolved by ``SchedulerTools`` /
  ``AgentConfig.get_scheduler_config`` when no explicit ``scheduler_service``
  is passed at the call site. Takes precedence over the global
  ``default: true`` flag in ``agent.yml``.
- ``semantic``: project-level default semantic adapter (must match a key
  under ``agent.services.semantic_layer``). Resolved by
  ``AgentConfig.resolve_semantic_adapter`` between the explicit
  ``adapter_type`` argument and the global ``default: true`` flag.
- ``plugins``: per-plugin activation for this project. A mapping of plugin
  name to ``{enabled: bool, active_profile: [<profile>, ...]}``. Omitting
  the whole ``plugins`` key means "activate every installed plugin and all
  of its profiles"; once the key is present it is the authoritative
  whitelist — an installed plugin not listed (or listed with
  ``enabled: false``) is NOT loaded (its CLI subcommand, bundled skills,
  system-prompt section, tool transformers and bash rules are all skipped).
  ``active_profile`` narrows which configured profiles are active; when it
  pins exactly one profile that becomes the ``datus <plugin>`` default.
  Written by the ``/plugins`` TUI and ``datus plugin enable/disable``.
- ``project_name``: shard name for ``~/.datus/sessions/{project_name}/``
  and ``~/.datus/data/{project_name}/`` (optional)
- ``reasoning_effort``: one of ``off|minimal|low|medium|high`` — controls the
  reasoning/thinking effort passed to the active model, mapped to each
  vendor's native dialect by LiteLLM.
- ``bash_allow``: list of bash command patterns (see
  ``datus/tools/permission/bash_rules.py`` for the syntax) appended to
  ``agent.permissions.bash_commands.allow`` at load time. Written by the
  "allow (project)" choice in the bash permission prompt via
  :func:`append_project_bash_allow`.
- ``sql_allow``: list of SQL statement kinds (``insert``, ``drop``, ... —
  see ``parse_sql_statement_kind`` in ``datus/utils/sql_utils.py``) that the
  ``execute_sql`` permission gate auto-allows for this project. Unlike
  ``bash_allow`` these are NOT merged into ``permissions`` rules; they feed
  ``PermissionManager``'s exact-match grant set only. Written by the
  "allow (project)" choice in the SQL permission prompt via
  :func:`append_project_sql_allow`.

Any other keys in the file are ignored with a warning so users do not
mistakenly expect the overlay to accept arbitrary YAML.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

PROJECT_CONFIG_REL = ".datus/config.yml"
ALLOWED_KEYS = frozenset(
    {
        "target",
        "default_datasource",
        "dashboard",
        "scheduler",
        "semantic",
        "plugins",
        "project_name",
        "language",
        "reasoning_effort",
        "bash_allow",
        "sql_allow",
        "sandbox",
    }
)
REASONING_EFFORT_CHOICES = frozenset({"off", "minimal", "low", "medium", "high"})


@dataclass
class ProjectTarget:
    """Structured ``target:`` value from ``./.datus/config.yml``.

    Exactly one of the (provider+model) pair or ``custom`` is populated.
    ``provider`` alone is not a valid state; callers must ensure both
    ``provider`` and ``model`` are set when selecting a provider-level
    entry.
    """

    provider: Optional[str] = None
    model: Optional[str] = None
    custom: Optional[str] = None


@dataclass
class PluginActivation:
    """Per-plugin activation state from ``./.datus/config.yml`` ``plugins:``.

    ``enabled`` gates whether the plugin is loaded at all this project (its
    CLI subcommand, bundled skills, system-prompt section, tool transformers
    and bash rules). ``active_profile`` narrows which configured profiles are
    active; ``None`` means "all profiles active". When it pins exactly one
    profile that profile becomes the ``datus <plugin>`` default (equivalent to
    the old string pin).
    """

    enabled: bool = True
    active_profile: Optional[List[str]] = None


@dataclass
class ProjectOverride:
    """In-memory representation of ``./.datus/config.yml``.

    ``None`` means "not specified — fall back to base agent.yml".
    ``target`` may be a legacy string (``agent.models`` key) or a
    :class:`ProjectTarget` describing a provider-level entry.
    ``reasoning_effort`` accepts ``off|minimal|low|medium|high``; any other
    string is dropped by :func:`load_project_override` with a warning.
    ``plugins`` is ``None`` when the key is absent (activate everything) and a
    (possibly empty) mapping when present — an empty mapping deactivates every
    installed plugin.
    """

    target: Optional[Union[str, ProjectTarget]] = None
    default_datasource: Optional[str] = None
    dashboard: Optional[str] = None
    scheduler: Optional[str] = None
    semantic: Optional[str] = None
    plugins: Optional[Dict[str, PluginActivation]] = None
    project_name: Optional[str] = None
    language: Optional[str] = None
    reasoning_effort: Optional[str] = None
    bash_allow: Optional[list] = None
    sql_allow: Optional[list] = None
    # ``sandbox`` overrides the bash sandbox for this project. Accepted values:
    # ``True``/``False`` toggle ``agent.bash.sandbox.enabled`` (mode follows
    # the global config); the strings ``"strict"``/``"normal"`` enable the
    # sandbox AND pin the mode. ``False`` is a meaningful value (force-off),
    # distinct from ``None``.
    sandbox: Optional[Union[bool, str]] = None

    def is_empty(self) -> bool:
        return (
            self.target is None
            and self.default_datasource is None
            and self.dashboard is None
            and self.scheduler is None
            and self.semantic is None
            and self.plugins is None
            and self.project_name is None
            and self.language is None
            and self.reasoning_effort is None
            and self.bash_allow is None
            and self.sql_allow is None
            and self.sandbox is None
        )


def project_config_path(cwd: Optional[str] = None) -> Path:
    """Return the absolute path to the project-level config file for ``cwd``."""
    return Path(cwd or os.getcwd()) / PROJECT_CONFIG_REL


def _parse_target(raw: Any) -> Optional[Union[str, ProjectTarget]]:
    """Normalize the ``target:`` field from raw YAML into its typed form.

    Accepts a string (legacy) or a mapping with ``provider``+``model`` or
    ``custom``. Mixing the two structured forms is invalid; the stricter
    form wins (``custom`` > provider/model) with a warning so the user
    notices the conflict.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        target = raw.strip()
        return target or None
    if isinstance(raw, dict):
        provider = str(raw.get("provider") or "").strip()
        model = str(raw.get("model") or "").strip()
        custom = str(raw.get("custom") or "").strip()
        if custom:
            if provider or model:
                logger.warning(
                    "project target mixes 'custom' with 'provider'/'model'; keeping 'custom' and ignoring the rest."
                )
            return ProjectTarget(custom=custom)
        if provider and model:
            return ProjectTarget(provider=provider, model=model)
        if provider or model:
            logger.warning("project target must provide both 'provider' and 'model'; ignoring partial value.")
        return None
    logger.warning(f"project target must be a string or mapping, got {type(raw).__name__}. Ignoring.")
    return None


def load_project_override(cwd: Optional[str] = None) -> Optional[ProjectOverride]:
    """Read ``./.datus/config.yml`` relative to ``cwd``.

    Returns ``None`` when the file is missing, empty, or fails to parse —
    the loader treats these as "no override" so the base ``agent.yml`` is
    used unchanged.  Unknown keys are dropped with a warning so users see
    the whitelist is enforced rather than silently ignoring typos.
    """
    path = project_config_path(cwd)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse {path}: {e}. Treating as no override.")
        return None
    except OSError as e:
        logger.warning(f"Failed to read {path}: {e}. Treating as no override.")
        return None
    if not isinstance(raw, dict):
        logger.warning(f"Ignoring {path}: top-level must be a mapping, got {type(raw).__name__}.")
        return None
    unknown = set(raw.keys()) - ALLOWED_KEYS
    if unknown:
        logger.warning(f"Ignoring unknown keys in {path}: {sorted(unknown)}. Only {sorted(ALLOWED_KEYS)} are accepted.")
    return ProjectOverride(
        target=_parse_target(raw.get("target")),
        default_datasource=raw.get("default_datasource"),
        dashboard=_parse_optional_string(raw.get("dashboard"), key="dashboard"),
        scheduler=_parse_optional_string(raw.get("scheduler"), key="scheduler"),
        semantic=_parse_optional_string(raw.get("semantic"), key="semantic"),
        plugins=_parse_plugins(raw.get("plugins")),
        project_name=raw.get("project_name"),
        language=raw.get("language"),
        reasoning_effort=_parse_reasoning_effort(raw.get("reasoning_effort")),
        bash_allow=_parse_bash_allow(raw.get("bash_allow")),
        sql_allow=_parse_sql_allow(raw.get("sql_allow")),
        sandbox=_parse_sandbox(raw.get("sandbox")),
    )


def _parse_bash_allow(raw: Any) -> Optional[list]:
    """Normalize the ``bash_allow:`` field into a list of pattern strings.

    Non-list values and non-string entries are dropped with a warning so a
    typo cannot silently widen (or corrupt) the bash allow-list.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        logger.warning(f"bash_allow must be a list of strings, got {type(raw).__name__}. Ignoring.")
        return None
    patterns = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            patterns.append(entry.strip())
        else:
            logger.warning(f"Ignoring non-string bash_allow entry: {entry!r}")
    return patterns or None


def _parse_sql_allow(raw: Any) -> Optional[list]:
    """Normalize the ``sql_allow:`` field into a list of statement kinds.

    Kinds are lower-cased for exact matching against
    ``parse_sql_statement_kind`` output. Non-list values and non-string
    entries are dropped with a warning; unrecognized kind strings are kept
    (they are inert — nothing ever produces them — so they can never widen
    the grant set, and dropping them would silently discard a future kind
    after a downgrade).
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        logger.warning(f"sql_allow must be a list of strings, got {type(raw).__name__}. Ignoring.")
        return None
    kinds = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            kinds.append(entry.strip().lower())
        else:
            logger.warning(f"Ignoring non-string sql_allow entry: {entry!r}")
    return kinds or None


def _parse_sandbox(raw: Any) -> Optional[Union[bool, str]]:
    """Normalize the ``sandbox:`` field: bool toggle or mode string.

    Booleans (and their common string spellings) toggle ``enabled``;
    ``"strict"``/``"normal"`` enable the sandbox and pin the mode. Anything
    else is dropped with a warning so a typo like ``sandbox: strick`` cannot
    silently change a security posture.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        if lowered in ("strict", "normal"):
            return lowered
    logger.warning(f"sandbox must be true/false/strict/normal, got {raw!r}. Ignoring.")
    return None


def _parse_optional_string(raw: Any, *, key: str) -> Optional[str]:
    """Coerce a YAML scalar into ``Optional[str]`` for ProjectOverride fields.

    Empty strings collapse to ``None`` so the override behaves the same as
    "not specified" rather than overriding the agent.yml value with an
    empty string. Non-string values are dropped with a warning so a
    ``dashboard: 123`` typo fails loudly instead of silently selecting
    the integer.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        logger.warning(f"{key} must be a string, got {type(raw).__name__}. Ignoring.")
        return None
    value = raw.strip()
    return value or None


def _parse_active_profile(plugin: str, raw: Any) -> Optional[List[str]]:
    """Normalize ``plugins.<plugin>.active_profile`` into a list of names.

    Accepts a single string (coerced to a one-element list) or a list of
    strings. Non-string entries are dropped with a warning; a fully invalid or
    empty value resolves to ``None`` ("all profiles active").
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else None
    if isinstance(raw, list):
        profiles: List[str] = []
        for entry in raw:
            if isinstance(entry, str) and entry.strip():
                profiles.append(entry.strip())
            else:
                logger.warning(f"Ignoring non-string active_profile entry for plugin '{plugin}': {entry!r}")
        return profiles or None
    logger.warning(
        f"plugins['{plugin}'].active_profile must be a string or list of strings, got {type(raw).__name__}. Ignoring."
    )
    return None


def _parse_plugin_activation(plugin: str, spec: Any) -> Optional[PluginActivation]:
    """Normalize one ``plugins.<plugin>`` entry into a :class:`PluginActivation`.

    The canonical shape is a mapping with an optional boolean ``enabled``
    (default ``True``) and an optional ``active_profile`` (see
    :func:`_parse_active_profile`). As a shorthand, a bare string or list is
    interpreted as ``active_profile`` with ``enabled: true`` (so a single
    pinned profile becomes the ``datus <plugin>`` default). A boolean is
    interpreted as ``enabled``. Any other type is dropped with a warning.
    """
    if isinstance(spec, bool):
        return PluginActivation(enabled=spec)
    if isinstance(spec, (str, list)):
        return PluginActivation(enabled=True, active_profile=_parse_active_profile(plugin, spec))
    if not isinstance(spec, dict):
        logger.warning(
            f"plugins['{plugin}'] must be a mapping with 'enabled'/'active_profile' "
            f"(or a profile name / list), got {type(spec).__name__}. Ignoring."
        )
        return None
    enabled_raw = spec.get("enabled", True)
    if isinstance(enabled_raw, bool):
        enabled = enabled_raw
    else:
        logger.warning(f"plugins['{plugin}'].enabled must be a boolean, got {enabled_raw!r}. Defaulting to true.")
        enabled = True
    active_profile = _parse_active_profile(plugin, spec.get("active_profile"))
    return PluginActivation(enabled=enabled, active_profile=active_profile)


def _parse_plugins(raw: Any) -> Optional[Dict[str, PluginActivation]]:
    """Normalize the ``plugins:`` field into a ``{plugin: PluginActivation}`` map.

    Declares per-plugin activation for this project. Returns ``None`` ONLY when
    the key is absent (or explicitly null) — meaning "activate every installed
    plugin and all profiles". A present mapping (even empty) is the
    authoritative whitelist: a returned empty dict deactivates all plugins. A
    present-but-malformed value (e.g. ``plugins: 123``) fails closed to an
    empty whitelist rather than ``None``, so a typo can never silently
    re-enable every plugin. Entries whose plugin name is not a non-empty string
    are dropped with a warning.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning(
            f"plugins must be a mapping, got {type(raw).__name__}. "
            "Treating as an empty whitelist (all plugins deactivated for this project)."
        )
        return {}
    parsed: Dict[str, PluginActivation] = {}
    for plugin, spec in raw.items():
        if not isinstance(plugin, str) or not plugin.strip():
            logger.warning(f"plugins key must be a non-empty string, got {plugin!r}. Ignoring.")
            continue
        activation = _parse_plugin_activation(plugin.strip(), spec)
        if activation is not None:
            parsed[plugin.strip()] = activation
    # A present-but-empty mapping is meaningful ("deactivate all"), so return
    # the dict as-is rather than collapsing it to ``None`` (which would read as
    # "key absent — activate everything").
    return parsed


def _parse_reasoning_effort(raw: Any) -> Optional[str]:
    """Normalize the ``reasoning_effort:`` field from raw YAML.

    Accepts any case-insensitive string in :data:`REASONING_EFFORT_CHOICES`;
    anything else is dropped with a warning so typos do not silently change
    behaviour. ``None`` means "not specified — fall back to base agent.yml".
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        logger.warning(f"reasoning_effort must be a string, got {type(raw).__name__}. Ignoring.")
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value not in REASONING_EFFORT_CHOICES:
        logger.warning(
            f"Ignoring invalid reasoning_effort '{raw}'. Expected one of {sorted(REASONING_EFFORT_CHOICES)}."
        )
        return None
    return value


def _target_to_yaml(target: Optional[Union[str, ProjectTarget]]) -> Any:
    if target is None:
        return None
    if isinstance(target, str):
        return target
    if target.custom:
        return {"custom": target.custom}
    if target.provider and target.model:
        return {"provider": target.provider, "model": target.model}
    return None


def _plugins_to_yaml(plugins: Optional[Dict[str, PluginActivation]]) -> Any:
    """Serialize the plugin activation map back to plain YAML structures.

    ``None`` is returned unchanged (the key is then omitted by
    :func:`save_project_override`). A present mapping — including an empty one —
    is written out so "deactivate all" round-trips. ``active_profile`` is
    omitted when ``None`` ("all profiles active").
    """
    if plugins is None:
        return None
    out: Dict[str, Dict[str, Any]] = {}
    for name, activation in plugins.items():
        entry: Dict[str, Any] = {"enabled": bool(activation.enabled)}
        if activation.active_profile is not None:
            entry["active_profile"] = list(activation.active_profile)
        out[name] = entry
    return out


def save_project_override(override: ProjectOverride, cwd: Optional[str] = None) -> Path:
    """Write ``override`` to ``./.datus/config.yml``.

    Creates the ``.datus/`` parent directory if missing.  ``None`` fields
    are omitted so the resulting file only contains the keys the user
    actually set.
    """
    path = project_config_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        k: v
        for k, v in {
            "target": _target_to_yaml(override.target),
            "default_datasource": override.default_datasource,
            "dashboard": override.dashboard,
            "scheduler": override.scheduler,
            "semantic": override.semantic,
            "plugins": _plugins_to_yaml(override.plugins),
            "project_name": override.project_name,
            "language": override.language,
            "reasoning_effort": override.reasoning_effort,
            "bash_allow": override.bash_allow,
            "sql_allow": override.sql_allow,
            # ``sandbox: false`` must round-trip (force-off is meaningful),
            # which the ``is not None`` filter below preserves.
            "sandbox": override.sandbox,
        }.items()
        if v is not None
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return path


def _append_project_list_entry(key: str, value: str, key_comment: str, cwd: Optional[str] = None) -> Path:
    """Append ``value`` to the ``<key>:`` list in ``./.datus/config.yml``.

    Edits at the TEXT level (not load->dump) so user comments and formatting
    in the rest of the file are preserved:

    - file missing     -> create it with a commented ``<key>`` block
    - no ``<key>:``    -> append the block at the end of the file
    - key present      -> insert ``  - "<value>"`` right after the key line
    - value already in the parsed list -> no-op

    Raises ``OSError`` on write failures; callers degrade to a session-level
    grant.
    """
    path = project_config_path(cwd)
    # json.dumps yields a valid double-quoted YAML scalar with proper
    # escaping, so a value containing ``"`` or a trailing backslash cannot
    # corrupt the file (a parse failure would drop ALL project overrides).
    entry_line = f"  - {json.dumps(value)}"

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "# Project-level Datus overrides. See conf/agent.yml.example for the full schema.\n"
            f"# {key_comment}\n"
            f"{key}:\n{entry_line}\n"
        )
        path.write_text(content, encoding="utf-8")
        return path

    text = path.read_text(encoding="utf-8")

    # No-op when the value is already present (compare parsed values, not
    # raw text, so quoting style differences don't cause duplicates).
    try:
        existing = yaml.safe_load(text) or {}
        if isinstance(existing, dict) and value in (existing.get(key) or []):
            return path
    except yaml.YAMLError:
        logger.warning(f"{path} is not valid YAML; appending {key} anyway.")

    lines = text.splitlines()
    key_idx = next(
        (i for i, line in enumerate(lines) if line.startswith(f"{key}:") and not line.lstrip().startswith("#")),
        None,
    )
    if key_idx is None:
        suffix = "" if (not text or text.endswith("\n")) else "\n"
        path.write_text(f"{text}{suffix}{key}:\n{entry_line}\n", encoding="utf-8")
    else:
        lines.insert(key_idx + 1, entry_line)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def append_project_bash_allow(pattern: str, cwd: Optional[str] = None) -> Path:
    """Append a bash allow pattern to ``./.datus/config.yml``'s ``bash_allow`` list.

    Used by the "allow (project)" choice in the bash permission prompt; see
    :func:`_append_project_list_entry` for the text-level edit semantics.
    Raises ``OSError`` on write failures; callers (``PermissionManager.
    add_project_bash_allow``) degrade to a session-level grant.
    """
    pattern = pattern.strip()
    if not pattern:
        raise DatusException(
            code=ErrorCode.COMMON_FIELD_INVALID,
            message_args={
                "field_name": "bash allow pattern",
                "except_values": "non-empty string",
                "your_value": pattern,
            },
        )
    return _append_project_list_entry(
        "bash_allow",
        pattern,
        "bash_allow patterns are appended to agent.permissions.bash_commands.allow.",
        cwd,
    )


def append_project_sql_allow(kind: str, cwd: Optional[str] = None) -> Path:
    """Append a SQL statement kind to ``./.datus/config.yml``'s ``sql_allow`` list.

    Used by the "allow (project)" choice in the SQL permission prompt; see
    :func:`_append_project_list_entry` for the text-level edit semantics.
    Raises ``OSError`` on write failures; callers (``PermissionManager.
    add_project_sql_allow``) degrade to a session-level grant.
    """
    kind = kind.strip().lower()
    if not kind:
        raise DatusException(
            code=ErrorCode.COMMON_FIELD_INVALID,
            message_args={
                "field_name": "sql allow kind",
                "except_values": "non-empty string",
                "your_value": kind,
            },
        )
    return _append_project_list_entry(
        "sql_allow",
        kind,
        "sql_allow statement kinds are auto-allowed by the execute_sql permission gate.",
        cwd,
    )
