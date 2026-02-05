# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Skill bash tool for restricted script execution.

Provides pattern-based command filtering following Claude Code conventions
for executing scripts within skill directories.
"""

import fnmatch
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any, List

from agents import Tool

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.skill_tools.skill_config import SkillMetadata

logger = logging.getLogger(__name__)

# Default timeout for script execution (seconds)
DEFAULT_TIMEOUT = 60

# Maximum output size to return (characters)
MAX_OUTPUT_SIZE = 50000


class SkillBashTool:
    """Execute scripts within skill directories with pattern-based restrictions.

    Provides restricted bash execution for skills that define `allowed_commands`
    in their frontmatter. Commands are only executed if they match the allowed
    patterns.

    Pattern syntax (Claude Code compatible):
    - "python:*" → allows any python command
    - "python:scripts/*.py" → allows only scripts in scripts/ dir
    - "sh:*.sh" → allows shell scripts
    - "gh:*" → allows GitHub CLI commands
    - "node:*" → allows Node.js commands

    Security features:
    - Pattern-based command filtering
    - Working directory locked to skill location
    - Timeout enforcement
    - Output size limiting

    Example usage:
        bash_tool = SkillBashTool(
            skill_metadata=skill,
            workspace_root=str(skill.location)
        )

        # Execute a command
        result = bash_tool.execute_command("python scripts/analyze.py --input data.json")
    """

    def __init__(
        self,
        skill_metadata: SkillMetadata,
        workspace_root: str,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Initialize the skill bash tool.

        Args:
            skill_metadata: SkillMetadata with allowed_commands
            workspace_root: Root directory for script execution
            timeout: Maximum execution time in seconds
        """
        self.skill = skill_metadata
        self.workspace_root = Path(workspace_root).resolve()
        self.allowed_patterns = skill_metadata.allowed_commands
        self.timeout = timeout
        self._tool_context: Any = None

        logger.debug(f"SkillBashTool created for '{skill_metadata.name}' " f"with patterns: {self.allowed_patterns}")

    def set_tool_context(self, ctx: Any) -> None:
        """Set tool context (called by framework before tool invocation).

        Args:
            ctx: Tool context from the agent framework
        """
        self._tool_context = ctx

    def execute_command(self, command: str) -> FuncToolResult:
        """Execute a command if it matches allowed patterns.

        The command is executed in the skill's directory as the working directory.
        Only commands matching the skill's allowed_commands patterns are permitted.

        Pattern examples:
        - "python:*" allows: python script.py, python -c "print('hello')"
        - "python:scripts/*.py" allows: python scripts/analyze.py
        - "sh:*.sh" allows: sh run.sh, sh build.sh

        Args:
            command: The command to execute (e.g., "python scripts/analyze.py --input data.json")

        Returns:
            FuncToolResult with stdout on success, error message on failure

        Example:
            execute_command(command="python scripts/analyze.py --input data.json")
        """
        if not command or not command.strip():
            return FuncToolResult(success=0, error="Empty command provided")

        command = command.strip()

        # Check if command is allowed
        if not self._is_command_allowed(command):
            logger.warning(f"Command not allowed for skill '{self.skill.name}': {command}")
            return FuncToolResult(
                success=0,
                error=f"Command not allowed by skill permissions. "
                f"Allowed patterns: {', '.join(self.allowed_patterns)}",
            )

        # Parse command into argv list to prevent shell injection
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return FuncToolResult(success=0, error=f"Invalid command syntax: {e}")

        try:
            # Execute command in skill directory
            logger.info(f"Executing command for skill '{self.skill.name}': {command}")

            result = subprocess.run(
                argv,
                shell=False,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=self._get_safe_env(),
            )

            # Combine stdout and stderr
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            # Truncate if too long
            if len(output) > MAX_OUTPUT_SIZE:
                output = output[:MAX_OUTPUT_SIZE] + f"\n... [truncated, total {len(output)} chars]"

            if result.returncode != 0:
                return FuncToolResult(
                    success=0,
                    error=f"Command exited with code {result.returncode}",
                    result=output,
                )

            return FuncToolResult(
                success=1,
                result=output,
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out for skill '{self.skill.name}': {command}")
            return FuncToolResult(
                success=0,
                error=f"Command timed out after {self.timeout} seconds",
            )
        except Exception as e:
            logger.error(f"Command execution failed for skill '{self.skill.name}': {e}")
            return FuncToolResult(
                success=0,
                error=f"Command execution failed: {str(e)}",
            )

    def _is_command_allowed(self, command: str) -> bool:
        """Check if a command matches any allowed pattern.

        Pattern format: "prefix:glob_pattern"
        - prefix: The command prefix (python, sh, node, gh, etc.)
        - glob_pattern: Glob pattern for arguments (* matches anything)

        Args:
            command: The command to check

        Returns:
            True if command matches an allowed pattern
        """
        if not self.allowed_patterns:
            return False

        # Parse command to get the base command
        try:
            parts = shlex.split(command)
            if not parts:
                return False
            base_cmd = parts[0]
        except ValueError:
            # If shlex fails, try simple split
            parts = command.split()
            if not parts:
                return False
            base_cmd = parts[0]

        for pattern in self.allowed_patterns:
            if self._matches_pattern(command, base_cmd, pattern):
                return True

        return False

    def _matches_pattern(self, full_command: str, base_cmd: str, pattern: str) -> bool:
        """Check if a command matches a specific pattern.

        Args:
            full_command: Full command string
            base_cmd: Base command (first word)
            pattern: Pattern to match (e.g., "python:scripts/*.py")

        Returns:
            True if matches
        """
        # Handle patterns with and without ":"
        if ":" in pattern:
            prefix, glob_pattern = pattern.split(":", 1)
        else:
            # Pattern without ":" matches exact command prefix
            prefix = pattern
            glob_pattern = "*"

        # Check if base command matches prefix
        if not fnmatch.fnmatch(base_cmd, prefix):
            return False

        # If glob pattern is *, allow any arguments
        if glob_pattern == "*":
            return True

        # For more specific patterns, check the full command
        # Replace additional colons in glob_pattern with spaces (e.g., "-c:*" -> "-c *")
        glob_pattern_normalized = glob_pattern.replace(":", " ")

        # Create a pattern that matches "prefix glob_pattern"
        full_pattern = f"{prefix} {glob_pattern_normalized}"

        # Also try matching just the arguments
        if fnmatch.fnmatch(full_command, full_pattern):
            return True

        # Try matching command arguments against the pattern
        # E.g., "python:scripts/*.py" should match "python scripts/analyze.py"
        try:
            parts = shlex.split(full_command)
            if len(parts) > 1:
                # Check if any argument matches the glob pattern
                for arg in parts[1:]:
                    if fnmatch.fnmatch(arg, glob_pattern):
                        return True
        except ValueError:
            pass

        return False

    def _get_safe_env(self) -> dict:
        """Get a safe environment for command execution.

        Returns:
            Dictionary of environment variables
        """
        import os

        # Start with current environment
        env = os.environ.copy()

        # Add skill-specific environment variables
        env["SKILL_NAME"] = self.skill.name
        env["SKILL_DIR"] = str(self.workspace_root)

        return env

    def available_tools(self) -> List[Tool]:
        """Return the list of tools provided by this class.

        Returns empty list if no allowed_commands are defined.

        Returns:
            List containing the execute_command tool, or empty list
        """
        if not self.allowed_patterns:
            return []

        return [trans_to_function_tool(self.execute_command)]

    @property
    def skill_name(self) -> str:
        """Get the skill name.

        Returns:
            Name of the skill
        """
        return self.skill.name
