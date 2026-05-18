# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Datus - Data engineering agent builds evolvable context for your data system"""

import os
import tomllib
from importlib import metadata as importlib_metadata
from pathlib import Path

# LiteLLM otherwise GETs https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
# at import time. We don't rely on the freshest cost map, so default to the
# bundled backup. User can opt back in by setting the env var to "false".
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

_DISTRIBUTION_NAME = "datus-agent"


def _read_source_tree_version() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None

    project = pyproject.get("project")
    if not isinstance(project, dict) or project.get("name") != _DISTRIBUTION_NAME:
        return None

    version = project.get("version")
    return version if isinstance(version, str) and version else None


def _package_version() -> str:
    # Source checkouts may run without installing the current package, while a
    # stale global datus-agent distribution can still exist on the runner.
    source_tree_version = _read_source_tree_version()
    if source_tree_version is not None:
        return source_tree_version

    try:
        return importlib_metadata.version(_DISTRIBUTION_NAME)
    except importlib_metadata.PackageNotFoundError:
        return "0+unknown"


__version__ = _package_version()
