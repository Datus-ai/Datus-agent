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
- ``project_name``: shard name for ``~/.datus/sessions/{project_name}/``
  and ``~/.datus/data/{project_name}/`` (optional)

Any other keys in the file are ignored with a warning so users do not
mistakenly expect the overlay to accept arbitrary YAML.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

PROJECT_CONFIG_REL = ".datus/config.yml"
ALLOWED_KEYS = frozenset({"target", "default_datasource", "project_name", "language"})


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
    """

    target: Optional[Union[str, ProjectTarget]] = None
    default_datasource: Optional[str] = None
    project_name: Optional[str] = None
    language: Optional[str] = None

    def is_empty(self) -> bool:
        return (
            self.target is None
            and self.default_datasource is None
            and self.project_name is None
            and self.language is None
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
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
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
        project_name=raw.get("project_name"),
        language=raw.get("language"),
    )


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
            "project_name": override.project_name,
            "language": override.language,
        }.items()
        if v is not None
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return path
