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

Any other keys in the file are ignored with a warning so users do not
mistakenly expect the overlay to accept arbitrary YAML.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

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
        "project_name",
        "language",
        "reasoning_effort",
        "bash_allow",
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
class ProjectOverride:
    """In-memory representation of ``./.datus/config.yml``.

    ``None`` means "not specified — fall back to base agent.yml".
    ``target`` may be a legacy string (``agent.models`` key) or a
    :class:`ProjectTarget` describing a provider-level entry.
    ``reasoning_effort`` accepts ``off|minimal|low|medium|high``; any other
    string is dropped by :func:`load_project_override` with a warning.
    """

    target: Optional[Union[str, ProjectTarget]] = None
    default_datasource: Optional[str] = None
    dashboard: Optional[str] = None
    scheduler: Optional[str] = None
    semantic: Optional[str] = None
    project_name: Optional[str] = None
    language: Optional[str] = None
    reasoning_effort: Optional[str] = None
    bash_allow: Optional[list] = None

    def is_empty(self) -> bool:
        return (
            self.target is None
            and self.default_datasource is None
            and self.dashboard is None
            and self.scheduler is None
            and self.semantic is None
            and self.project_name is None
            and self.language is None
            and self.reasoning_effort is None
            and self.bash_allow is None
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
        project_name=raw.get("project_name"),
        language=raw.get("language"),
        reasoning_effort=_parse_reasoning_effort(raw.get("reasoning_effort")),
        bash_allow=_parse_bash_allow(raw.get("bash_allow")),
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
            "project_name": override.project_name,
            "language": override.language,
            "reasoning_effort": override.reasoning_effort,
            "bash_allow": override.bash_allow,
        }.items()
        if v is not None
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return path


def append_project_bash_allow(pattern: str, cwd: Optional[str] = None) -> Path:
    """Append a bash allow pattern to ``./.datus/config.yml``'s ``bash_allow`` list.

    Used by the "allow (project)" choice in the bash permission prompt.
    Edits at the TEXT level (not load->dump) so user comments and formatting
    in the rest of the file are preserved:

    - file missing        -> create it with a commented ``bash_allow`` block
    - no ``bash_allow:``  -> append the block at the end of the file
    - key present         -> insert ``  - "<pattern>"`` right after the key line
    - pattern already in the parsed list -> no-op

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
    path = project_config_path(cwd)
    # json.dumps yields a valid double-quoted YAML scalar with proper
    # escaping, so a pattern containing ``"`` or a trailing backslash cannot
    # corrupt the file (a parse failure would drop ALL project overrides).
    entry_line = f"  - {json.dumps(pattern)}"

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "# Project-level Datus overrides. See conf/agent.yml.example for the full schema.\n"
            "# bash_allow patterns are appended to agent.permissions.bash_commands.allow.\n"
            f"bash_allow:\n{entry_line}\n"
        )
        path.write_text(content, encoding="utf-8")
        return path

    text = path.read_text(encoding="utf-8")

    # No-op when the pattern is already present (compare parsed values, not
    # raw text, so quoting style differences don't cause duplicates).
    try:
        existing = yaml.safe_load(text) or {}
        if isinstance(existing, dict) and pattern in (existing.get("bash_allow") or []):
            return path
    except yaml.YAMLError:
        logger.warning(f"{path} is not valid YAML; appending bash_allow anyway.")

    lines = text.splitlines()
    key_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("bash_allow:") and not line.lstrip().startswith("#")),
        None,
    )
    if key_idx is None:
        suffix = "" if (not text or text.endswith("\n")) else "\n"
        path.write_text(f"{text}{suffix}bash_allow:\n{entry_line}\n", encoding="utf-8")
    else:
        lines.insert(key_idx + 1, entry_line)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
