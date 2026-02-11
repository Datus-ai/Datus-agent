# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for skills with Python script execution.

Tests the complete flow of loading skills with scripts and executing them.
"""


import pytest

from datus.tools.skill_tools import (
    SkillBashTool,
    SkillConfig,
    SkillFuncTool,
    SkillManager,
    SkillRegistry,
)


@pytest.fixture
def python_script_skill_dir(tmp_path):
    """Create a skill with Python scripts."""
    skill_dir = tmp_path / "python-script-skill"
    skill_dir.mkdir()

    # Create SKILL.md
    (skill_dir / "SKILL.md").write_text(
        """---
name: data-analyzer
description: Analyze data using Python scripts
tags:
  - python
  - analysis
version: 1.0.0
allowed_commands:
  - "python:scripts/*.py"
---

# Data Analyzer Skill

This skill provides data analysis capabilities using Python scripts.

## Available Scripts

### analyze_data.py
Analyzes query results and provides statistical insights.

Usage:
```bash
python scripts/analyze_data.py --input <data.json> --output <report.json>
```

## Workflow
1. Execute SQL query using db_tools.execute_sql
2. Save results to temporary file
3. Run analyze_data.py on the results
"""
    )

    # Create scripts directory
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()

    # Create analyze_data.py
    (scripts_dir / "analyze_data.py").write_text(
        """
import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data.json')
    parser.add_argument('--output', default='report.json')
    args = parser.parse_args()

    result = {
        "status": "success",
        "rows_analyzed": 100,
        "input_file": args.input,
        "output_file": args.output
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
"""
    )

    return skill_dir


class TestSkillPythonScriptExecution:
    """Integration tests for skills with Python script execution."""

    def test_skill_allowed_commands_parsed(self, python_script_skill_dir):
        """Test that allowed_commands are correctly parsed from SKILL.md."""
        config = SkillConfig(directories=[str(python_script_skill_dir.parent)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skill = registry.get_skill("data-analyzer")
        assert skill is not None
        assert "python:scripts/*.py" in skill.allowed_commands

    def test_skill_bash_tool_allows_matching_command(self, python_script_skill_dir):
        """Test that SkillBashTool allows commands matching patterns."""
        config = SkillConfig(directories=[str(python_script_skill_dir.parent)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        skill = registry.get_skill("data-analyzer")

        bash_tool = SkillBashTool(skill_metadata=skill, workspace_root=str(skill.location))

        # Should allow
        result = bash_tool.execute_command("python scripts/analyze_data.py --input test.json")
        assert result.success == 1
        assert "success" in result.result

    def test_skill_bash_tool_denies_non_matching_command(self, python_script_skill_dir):
        """Test that SkillBashTool denies commands not matching patterns."""
        config = SkillConfig(directories=[str(python_script_skill_dir.parent)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        skill = registry.get_skill("data-analyzer")

        bash_tool = SkillBashTool(skill_metadata=skill, workspace_root=str(skill.location))

        # Should deny - not in allowed patterns
        result = bash_tool.execute_command("rm -rf /")
        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_skill_bash_tool_workspace_isolation(self, python_script_skill_dir):
        """Test that script execution is sandboxed to skill directory."""
        config = SkillConfig(directories=[str(python_script_skill_dir.parent)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        skill = registry.get_skill("data-analyzer")

        bash_tool = SkillBashTool(skill_metadata=skill, workspace_root=str(skill.location))

        # Script should run in skill directory
        result = bash_tool.execute_command("python scripts/analyze_data.py")
        assert result.success == 1
        # Verify output contains expected JSON
        assert "success" in result.result

    def test_end_to_end_skill_with_script(self, python_script_skill_dir):
        """End-to-end test: load skill, execute script, get results."""
        config = SkillConfig(directories=[str(python_script_skill_dir.parent)])
        manager = SkillManager(config=config)

        # Create skill func tool
        skill_tool = SkillFuncTool(manager=manager, node_name="chatbot")

        # Load the skill
        skill_result = skill_tool.load_skill("data-analyzer")
        assert skill_result.success == 1
        assert "python scripts/analyze_data.py" in skill_result.result

        # Get the bash tool created for this skill
        bash_tool = skill_tool.get_skill_bash_tool("data-analyzer")
        assert bash_tool is not None

        # Execute script via skill bash tool
        bash_result = bash_tool.execute_command("python scripts/analyze_data.py --input test.json")
        assert bash_result.success == 1
        assert "rows_analyzed" in bash_result.result

    def test_skill_script_with_json_output(self, python_script_skill_dir):
        """Test that script output can be parsed as JSON."""
        import json

        config = SkillConfig(directories=[str(python_script_skill_dir.parent)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        skill = registry.get_skill("data-analyzer")

        bash_tool = SkillBashTool(skill_metadata=skill, workspace_root=str(skill.location))

        result = bash_tool.execute_command("python scripts/analyze_data.py")
        assert result.success == 1

        # Parse JSON output
        output = json.loads(result.result.strip())
        assert output["status"] == "success"
        assert output["rows_analyzed"] == 100

    def test_skill_script_error_handling(self, python_script_skill_dir):
        """Test handling of script execution errors."""
        config = SkillConfig(directories=[str(python_script_skill_dir.parent)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        skill = registry.get_skill("data-analyzer")

        # Create a script that raises an error
        scripts_dir = python_script_skill_dir / "scripts"
        (scripts_dir / "error_script.py").write_text(
            """
raise ValueError("Intentional error")
"""
        )

        bash_tool = SkillBashTool(skill_metadata=skill, workspace_root=str(skill.location))

        result = bash_tool.execute_command("python scripts/error_script.py")
        # Should fail but not crash
        assert result.success == 0
        assert "error" in result.error.lower() or "ValueError" in str(result.result)


class TestSkillScriptPatternVariations:
    """Tests for various script pattern configurations."""

    def test_multiple_script_patterns(self, tmp_path):
        """Test skill with multiple script patterns."""
        skill_dir = tmp_path / "multi-pattern-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text(
            """---
name: multi-pattern
description: Skill with multiple patterns
allowed_commands:
  - "python:scripts/*.py"
  - "python:utils/*.py"
  - "sh:*.sh"
---
# Multi-Pattern Skill
"""
        )

        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "main.py").write_text("print('main')")

        utils_dir = skill_dir / "utils"
        utils_dir.mkdir()
        (utils_dir / "helper.py").write_text("print('helper')")

        (skill_dir / "run.sh").write_text("#!/bin/bash\necho 'shell'")

        config = SkillConfig(directories=[str(tmp_path)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        skill = registry.get_skill("multi-pattern")

        bash_tool = SkillBashTool(skill_metadata=skill, workspace_root=str(skill.location))

        # All patterns should work
        assert bash_tool.execute_command("python scripts/main.py").success == 1
        assert bash_tool.execute_command("python utils/helper.py").success == 1
        assert bash_tool.execute_command("sh run.sh").success == 1

    def test_python_inline_code_pattern(self, tmp_path):
        """Test skill allowing Python inline code execution."""
        skill_dir = tmp_path / "inline-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text(
            """---
name: inline-skill
description: Skill allowing inline Python
allowed_commands:
  - "python:-c:*"
---
# Inline Skill
"""
        )

        config = SkillConfig(directories=[str(tmp_path)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        skill = registry.get_skill("inline-skill")

        bash_tool = SkillBashTool(skill_metadata=skill, workspace_root=str(skill.location))

        result = bash_tool.execute_command('python -c "print(1+2)"')
        assert result.success == 1
        assert "3" in result.result
