#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Quick validation script for Datus SKILL.md files.

Validates:
- YAML frontmatter parses correctly
- Required fields (name, description) are present
- allowed_commands patterns are well-formed
- Directory structure is correct

Usage:
    python scripts/quick_validate.py /path/to/skill/SKILL.md
"""

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
REQUIRED_FIELDS = ["name", "description"]
VALID_COMMAND_PATTERN = re.compile(r"^[a-zA-Z0-9_]+:.+$")
VALID_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


def validate_skill(skill_path: str) -> list:
    """Validate a SKILL.md file and return a list of issues."""
    issues = []
    path = Path(skill_path)

    # Check file exists
    if not path.exists():
        issues.append(f"ERROR: File not found: {skill_path}")
        return issues

    if not path.name == "SKILL.md":
        issues.append(f"WARNING: Expected filename 'SKILL.md', got '{path.name}'")

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        issues.append(f"ERROR: Cannot read file: {e}")
        return issues

    # Check frontmatter exists
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        issues.append("ERROR: No valid YAML frontmatter found. Expected '---' delimiters at the start of the file.")
        return issues

    # Parse YAML
    raw_yaml = match.group(1)
    try:
        metadata = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as e:
        issues.append(f"ERROR: YAML parse error: {e}")
        return issues

    if not isinstance(metadata, dict):
        issues.append("ERROR: Frontmatter must be a YAML mapping (key-value pairs)")
        return issues

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in metadata or not metadata[field]:
            issues.append(f"ERROR: Required field '{field}' is missing or empty")

    # Validate name format
    name = metadata.get("name", "")
    if name and not VALID_NAME_PATTERN.match(name):
        issues.append(f"WARNING: Skill name '{name}' should be lowercase with hyphens only (e.g., 'sql-optimization')")

    # Validate description
    description = metadata.get("description", "")
    if description and len(description) < 20:
        issues.append(
            "WARNING: Description is very short. Consider adding more detail about when to trigger this skill."
        )
    if description and len(description) > 500:
        issues.append("WARNING: Description is very long (>500 chars). Keep it concise but comprehensive.")

    # Validate allowed_commands patterns
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
                        f"WARNING: allowed_commands pattern '{cmd}' doesn't match expected format 'prefix:glob_pattern' "
                        f"(e.g., 'python:scripts/*.py')"
                    )

    # Validate context field
    context = metadata.get("context")
    if context and context != "fork":
        issues.append(f"WARNING: context field should be 'fork' or omitted, got: '{context}'")

    # Validate agent field
    agent = metadata.get("agent")
    if agent and not context:
        issues.append(
            "WARNING: 'agent' field is set but 'context' is not 'fork'. Agent type is only used with context: fork"
        )

    # Check body content
    body = content[match.end() :]
    if not body.strip():
        issues.append("ERROR: SKILL.md body is empty. Add instructions for the agent.")
    else:
        line_count = len(body.strip().splitlines())
        if line_count > 500:
            issues.append(
                f"WARNING: SKILL.md body is {line_count} lines. Consider moving detailed content to references/."
            )

    # Check directory structure
    skill_dir = path.parent
    scripts_dir = skill_dir / "scripts"
    if allowed_commands and not scripts_dir.exists():
        issues.append(
            "WARNING: 'allowed_commands' is configured but no 'scripts/' directory found. "
            "Create it to store executable scripts."
        )

    return issues


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_validate.py <path-to-SKILL.md>")
        sys.exit(1)

    skill_path = sys.argv[1]
    issues = validate_skill(skill_path)

    if not issues:
        print(f"PASS: {skill_path} is valid")
        sys.exit(0)

    errors = [i for i in issues if i.startswith("ERROR")]
    warnings = [i for i in issues if i.startswith("WARNING")]

    for issue in issues:
        print(issue)

    print(f"\nSummary: {len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
