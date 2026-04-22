"""Semantic model / metric YAML validation with optional deep validation.

When ``metricflow`` is installed (via ``datus-metricflow``), the full 5-layer
validation chain is used:

1. Write YAML to a temp file
2. ``ConfigLinter.lint_file()`` -- structure / field checks
3. ``collect_yaml_config_file_paths()`` -- gather all datasource YAML files
4. ``parse_yaml_file_paths_to_model()`` -- cross-file reference resolution
5. ``ModelValidator().validate_model()`` -- semantic consistency

When ``metricflow`` is **not** installed, falls back to a basic
``yaml.safe_load`` syntax check and logs a one-time warning.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_METRICFLOW_AVAILABLE: bool | None = None


def _check_metricflow() -> bool:
    global _METRICFLOW_AVAILABLE
    if _METRICFLOW_AVAILABLE is None:
        try:
            from metricflow.model.model_validator import ModelValidator  # noqa: F401
            from metricflow.model.parsing.config_linter import ConfigLinter  # noqa: F401
            from metricflow.model.parsing.dir_to_model import (  # noqa: F401
                collect_yaml_config_file_paths,
                parse_yaml_file_paths_to_model,
            )

            _METRICFLOW_AVAILABLE = True
        except ImportError:
            _METRICFLOW_AVAILABLE = False
            logger.warning(
                "metricflow is not installed – semantic model validation will "
                "fall back to YAML syntax check only.  Install via: "
                "pip install datus-metricflow"
            )
    return _METRICFLOW_AVAILABLE


def validate_semantic_yaml(
    yaml_content: str,
    file_path: str,
    datus_home: str,
    datasource: str,
) -> Tuple[bool, List[str]]:
    """Validate semantic model / metric YAML content.

    Uses the full metricflow validation chain when available, otherwise
    falls back to a basic YAML syntax check.

    Args:
        yaml_content: The YAML content to validate.
        file_path: The target file path (used to determine filename and
            for replacement in the collected file list).
        datus_home: Datus home directory path.
        datasource: Current datasource for semantic model directory.

    Returns:
        ``(is_valid, error_messages)``
    """
    if _check_metricflow():
        return _validate_deep(yaml_content, file_path, datus_home, datasource)
    return _validate_yaml_format(yaml_content)


# -- deep validation (metricflow present) ------------------------------------


def _validate_deep(
    yaml_content: str,
    file_path: str,
    datus_home: str,
    datasource: str,
) -> Tuple[bool, List[str]]:
    from metricflow.model.model_validator import ModelValidator
    from metricflow.model.parsing.config_linter import ConfigLinter
    from metricflow.model.parsing.dir_to_model import (
        collect_yaml_config_file_paths,
        parse_yaml_file_paths_to_model,
    )

    temp_dir = None
    temp_file_path = None

    try:
        # Step 1: Create a temporary file with the YAML content
        temp_dir = tempfile.mkdtemp()
        filename = os.path.basename(file_path)
        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        # Step 2: ConfigLinter -- structure / field checks
        linter = ConfigLinter()
        lint_issues = linter.lint_file(temp_file_path)
        if lint_issues:
            return False, [issue.as_readable_str() for issue in lint_issues]

        # Step 3: Collect all existing semantic model files from the same
        # project tree as ``file_path``. Resolving via the CWD-based
        # ``DatusPathManager.semantic_model_path()`` would make validation
        # nondeterministic when the calling process's CWD does not match
        # the project that owns ``file_path``.
        del datasource  # unused after refactor; kept in signature for compatibility
        del datus_home  # resolution below is file_path-relative
        target_path = Path(file_path).resolve()
        semantic_yaml_dir = next(
            (p for p in [target_path.parent, *target_path.parents] if p.name == "semantic_models"),
            target_path.parent,
        )
        file_paths = collect_yaml_config_file_paths(directory=str(semantic_yaml_dir))

        # Replace the original file with the temp file for validation
        if file_path in file_paths:
            file_paths.remove(file_path)
        file_paths.append(temp_file_path)

        # Step 4: Cross-file parsing and reference resolution
        parsing_result = parse_yaml_file_paths_to_model(file_paths, raise_issues_as_exceptions=False)

        if parsing_result.issues and parsing_result.issues.has_blocking_issues:
            return False, _issues_to_strings(parsing_result.issues.all_issues)

        # Step 5: ModelValidator -- semantic consistency
        try:
            semantic_result = ModelValidator().validate_model(parsing_result.model)
            if semantic_result.issues and semantic_result.issues.has_blocking_issues:
                return False, _issues_to_strings(semantic_result.issues.all_issues)
        except Exception as e:
            return False, [f"Semantic validation failed: {e}"]

        return True, []

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if temp_dir and os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def _issues_to_strings(issues) -> List[str]:
    messages = []
    for issue in issues:
        if hasattr(issue, "as_readable_str"):
            messages.append(issue.as_readable_str())
        else:
            messages.append(str(issue))
    return messages


# -- fallback validation (no metricflow) -------------------------------------


def _validate_yaml_format(yaml_content: str) -> Tuple[bool, List[str]]:
    """Fallback: YAML syntax check only."""
    import yaml as _yaml

    try:
        _yaml.safe_load(yaml_content)
        return True, []
    except _yaml.YAMLError as e:
        return False, [str(e)]
