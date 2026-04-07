# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Skill validation tool for the skill-creator subagent.

Validates SKILL.md files: YAML frontmatter, required fields,
allowed_commands patterns, and directory structure.
"""

import re
from pathlib import Path
from typing import List

import yaml

from datus.tools.func_tool.base import FuncToolResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
REQUIRED_FIELDS = ["name", "description"]
VALID_COMMAND_PATTERN = re.compile(r"^[a-zA-Z0-9_]+:.+$")
VALID_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


class SkillValidateTool:
    """Tool that validates a SKILL.md file for correctness.

    Checks:
    - YAML frontmatter parses correctly
    - Required fields (name, description) are present
    - allowed_commands patterns are well-formed
    - Directory structure is correct
    """

    def validate_skill(self, skill_path: str) -> FuncToolResult:
        """Validate a SKILL.md file and report any issues.

        Args:
            skill_path: Absolute path to the SKILL.md file to validate.

        Returns:
            FuncToolResult with validation results: list of errors, warnings, and overall status.
        """
        issues: List[str] = []
        path = Path(skill_path)

        # Check file exists
        if not path.exists():
            return FuncToolResult(success=0, error=f"File not found: {skill_path}")

        if path.name != "SKILL.md":
            issues.append(f"WARNING: Expected filename 'SKILL.md', got '{path.name}'")

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return FuncToolResult(success=0, error=f"Cannot read file: {e}")

        # Check frontmatter exists
        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            return FuncToolResult(
                success=0,
                error="No valid YAML frontmatter found. SKILL.md must start with '---' delimiters containing YAML.",
            )

        # Parse YAML
        raw_yaml = match.group(1)
        try:
            metadata = yaml.safe_load(raw_yaml)
        except yaml.YAMLError as e:
            return FuncToolResult(success=0, error=f"YAML parse error in frontmatter: {e}")

        if not isinstance(metadata, dict):
            return FuncToolResult(success=0, error="Frontmatter must be a YAML mapping (key-value pairs)")

        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in metadata or not metadata[field]:
                issues.append(f"ERROR: Required field '{field}' is missing or empty")

        # Validate name format
        name = metadata.get("name", "")
        if name and not VALID_NAME_PATTERN.match(name):
            issues.append(f"WARNING: Skill name '{name}' should be lowercase with hyphens (e.g., 'sql-optimization')")

        # Validate description length
        description = metadata.get("description", "")
        if description and len(description) < 20:
            issues.append("WARNING: Description is very short. Add more detail about when to trigger this skill.")
        if description and len(description) > 500:
            issues.append("WARNING: Description is very long (>500 chars). Keep it concise.")

        # Validate allowed_commands
        allowed_commands = metadata.get("allowed_commands", [])
        if allowed_commands:
            if not isinstance(allowed_commands, list):
                issues.append("ERROR: 'allowed_commands' must be a list")
            else:
                for cmd in allowed_commands:
                    if not isinstance(cmd, str):
                        issues.append(f"ERROR: allowed_commands entry must be a string, got: {type(cmd).__name__}")
                    elif not VALID_COMMAND_PATTERN.match(cmd):
                        issues.append(
                            f"WARNING: Pattern '{cmd}' should match format 'prefix:glob' (e.g., 'python:scripts/*.py')"
                        )

        # Validate context/agent fields
        context = metadata.get("context")
        if context and context != "fork":
            issues.append(f"WARNING: context should be 'fork' or omitted, got: '{context}'")
        agent = metadata.get("agent")
        if agent and not context:
            issues.append("WARNING: 'agent' field requires context: fork")

        # Check body content
        body = content[match.end() :]
        if not body.strip():
            issues.append("ERROR: SKILL.md body is empty. Add instructions for the agent.")
        else:
            line_count = len(body.strip().splitlines())
            if line_count > 500:
                issues.append(f"WARNING: Body is {line_count} lines. Consider moving content to references/.")

        # Check directory structure
        skill_dir = path.parent
        scripts_dir = skill_dir / "scripts"
        if allowed_commands and not scripts_dir.exists():
            issues.append("WARNING: 'allowed_commands' set but no 'scripts/' directory found.")

        # Build result
        errors = [i for i in issues if i.startswith("ERROR")]
        warnings = [i for i in issues if i.startswith("WARNING")]

        if errors:
            return FuncToolResult(
                success=0,
                error=f"Validation FAILED: {len(errors)} error(s), {len(warnings)} warning(s)\n" + "\n".join(issues),
            )

        if warnings:
            return FuncToolResult(
                result={
                    "status": "PASS with warnings",
                    "warnings": len(warnings),
                    "details": issues,
                    "skill_name": metadata.get("name", ""),
                    "skill_path": str(skill_dir),
                }
            )

        return FuncToolResult(
            result={
                "status": "PASS",
                "warnings": 0,
                "details": [],
                "skill_name": metadata.get("name", ""),
                "skill_path": str(skill_dir),
            }
        )

    def available_tools(self):
        """Return the validate_skill FunctionTool."""
        from datus.tools.func_tool.base import trans_to_function_tool

        return [trans_to_function_tool(self.validate_skill)]
