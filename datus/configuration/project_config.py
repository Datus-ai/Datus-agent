# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Project-level ``./.datus/config.yml`` override.

A small, strict overlay on the base ``agent.yml`` that lets every project
pin three values without copying the full config:

- ``target``: which LLM to use (must match a key under ``agent.models``)
- ``default_database``: which database to connect to on startup (must
  match a key under ``agent.services.databases``)
- ``project_name``: shard name for ``~/.datus/sessions/{project_name}/``
  and ``~/.datus/data/{project_name}/`` (optional)

Any other keys in the file are ignored with a warning so users do not
mistakenly expect the overlay to accept arbitrary YAML.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

PROJECT_CONFIG_REL = ".datus/config.yml"
ALLOWED_KEYS = frozenset({"target", "default_database", "project_name"})


@dataclass
class ProjectOverride:
    """In-memory representation of ``./.datus/config.yml``.

    ``None`` means "not specified — fall back to base agent.yml".
    """

    target: Optional[str] = None
    default_database: Optional[str] = None
    project_name: Optional[str] = None

    def is_empty(self) -> bool:
        return self.target is None and self.default_database is None and self.project_name is None


def project_config_path(cwd: Optional[str] = None) -> Path:
    """Return the absolute path to the project-level config file for ``cwd``."""
    return Path(cwd or os.getcwd()) / PROJECT_CONFIG_REL


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
        target=raw.get("target"),
        default_database=raw.get("default_database"),
        project_name=raw.get("project_name"),
    )


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
            "target": override.target,
            "default_database": override.default_database,
            "project_name": override.project_name,
        }.items()
        if v is not None
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return path
