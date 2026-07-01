# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
General-purpose bash execution tool.

Provides pattern-based command filtering (Claude Code conventions) and
subprocess-based execution. Decoupled from the Skill system: any caller
(Skill, agentic node, MCP server, ...) can instantiate ``BashTool`` by
supplying ``workspace_root`` and an optional ``allowed_patterns`` list.
Skill-specific context (``SKILL_NAME`` / ``SKILL_DIR`` env vars, etc.) is
injected by the caller through ``extra_env``.
"""

import fnmatch
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Tool

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
MAX_OUTPUT_SIZE = 50000


class BashTool:
    """Execute shell commands with optional pattern-based restrictions.

    Pattern syntax (Claude Code compatible):
    - ``"python:*"`` allows any python command
    - ``"python:scripts/*.py"`` allows only scripts in ``scripts/``
    - ``"sh:*.sh"`` allows shell scripts
    - ``"*"`` matches any command (use for "no whitelist" mode where
      permission control happens at the ``PermissionManager`` layer)

    ``allowed_patterns`` semantics:
    - ``None`` or ``[]``: no commands are allowed; ``available_tools()``
      returns an empty list, so the tool is effectively hidden. This is
      the safe default and matches the legacy SkillBashTool behavior for
      skills without ``allowed_commands``.
    - non-empty list: commands are filtered through ``_is_command_allowed``.

    Security features:
    - Pattern-based filtering
    - Working directory locked to ``workspace_root``
    - Timeout enforcement
    - Output size limiting
    - ``shell=False`` subprocess invocation (argv via ``shlex.split``)
    """

    permission_category: str = "bash_tools"

    def __init__(
        self,
        workspace_root: str,
        allowed_patterns: Optional[List[str]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        extra_env: Optional[Dict[str, str]] = None,
        identity: Optional[str] = None,
    ):
        """Initialize the bash tool.

        Args:
            workspace_root: Working directory for command execution.
            allowed_patterns: Optional list of allowed command patterns.
                ``None`` or empty list disables the tool entirely.
            timeout: Maximum execution time in seconds.
            extra_env: Additional environment variables to inject into the
                child process (merged on top of ``os.environ``). Use this
                to carry caller-specific context (e.g. ``SKILL_NAME``).
            identity: Optional label used in log messages so multiple
                BashTool instances can be distinguished.
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.allowed_patterns = list(allowed_patterns) if allowed_patterns else []
        self.timeout = timeout
        self.extra_env = dict(extra_env) if extra_env else {}
        self.identity = identity
        self._tool_context: Any = None

        logger.debug(
            "BashTool created (identity=%s) workspace=%s patterns=%s",
            self.identity,
            self.workspace_root,
            self.allowed_patterns,
        )

    def set_tool_context(self, ctx: Any) -> None:
        """Set tool context (called by framework before tool invocation)."""
        self._tool_context = ctx

    def execute_command(self, command: str) -> FuncToolResult:
        """Execute a shell command if it matches the allowed patterns.

        The command runs in ``workspace_root`` with ``shell=False``. Only
        commands matching one of ``allowed_patterns`` are permitted.

        Args:
            command: The command to execute (e.g., "python scripts/analyze.py").

        Returns:
            FuncToolResult with stdout on success, error message on failure.
        """
        if not command or not command.strip():
            return FuncToolResult(success=0, error="Empty command provided")

        command = command.strip()

        if not self._is_command_allowed(command):
            logger.warning("Command not allowed (identity=%s): %s", self.identity, command)
            return FuncToolResult(
                success=0,
                error=f"Command not allowed. Allowed patterns: {', '.join(self.allowed_patterns) or '(none)'}",
            )

        try:
            argv = shlex.split(command)
        except ValueError as e:
            return FuncToolResult(success=0, error=f"Invalid command syntax: {e}")

        # Resolve "python" to a real executable — handles environments
        # where only "python3" exists (macOS, some Linux distros).
        if argv and argv[0] == "python":
            argv[0] = sys.executable or shutil.which("python3") or "python3"

        try:
            logger.info("Executing command (identity=%s): %s", self.identity, command)

            result = subprocess.run(
                argv,
                shell=False,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=self._get_safe_env(),
                # Detach the child from the agent's stdin. Without this the child
                # inherits datus's terminal stdin and any command that reads it
                # (``cat``, ``read``, ``python`` awaiting input, an interactive
                # prompt) blocks forever, fighting the TUI's prompt_toolkit for
                # the same TTY and freezing the whole process. DEVNULL delivers
                # an immediate EOF so such commands fail fast instead of hanging.
                stdin=subprocess.DEVNULL,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            if len(output) > MAX_OUTPUT_SIZE:
                output = output[:MAX_OUTPUT_SIZE] + f"\n... [truncated, total {len(output)} chars]"

            if result.returncode != 0:
                return FuncToolResult(
                    success=0,
                    error=f"Command exited with code {result.returncode}",
                    result=output,
                )

            return FuncToolResult(success=1, result=output)

        except subprocess.TimeoutExpired:
            logger.error("Command timed out (identity=%s): %s", self.identity, command)
            return FuncToolResult(success=0, error=f"Command timed out after {self.timeout} seconds")
        except Exception as e:
            logger.error("Command execution failed (identity=%s): %s", self.identity, e)
            return FuncToolResult(success=0, error=f"Command execution failed: {str(e)}")

    def _is_command_allowed(self, command: str) -> bool:
        """Check if a command matches any allowed pattern."""
        if not self.allowed_patterns:
            return False

        try:
            parts = shlex.split(command)
        except ValueError:
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

        Pattern format: ``"prefix:glob_pattern"``. When ``":"`` is absent the
        whole pattern is treated as the prefix and any arguments match.
        """
        if ":" in pattern:
            prefix, glob_pattern = pattern.split(":", 1)
        else:
            prefix = pattern
            glob_pattern = "*"

        if not fnmatch.fnmatch(base_cmd, prefix):
            return False

        if glob_pattern == "*":
            return True

        # Normalize additional colons in glob_pattern (e.g. "-c:*" -> "-c *")
        glob_pattern_normalized = glob_pattern.replace(":", " ")
        full_pattern = f"{prefix} {glob_pattern_normalized}"
        if fnmatch.fnmatch(full_command, full_pattern):
            return True

        try:
            parts = shlex.split(full_command)
            # Only validate the first positional argument after the executable.
            # Matching against arbitrary later args would let a caller smuggle
            # a disallowed flag/script in (e.g. ``python -c "..." scripts/ok.py``
            # against a ``python:scripts/*.py`` rule).
            if len(parts) > 1 and fnmatch.fnmatch(parts[1], glob_pattern):
                return True
        except ValueError:
            pass

        return False

    def _get_safe_env(self) -> dict:
        """Build the environment for command execution.

        Starts from ``os.environ`` and overlays ``self.extra_env`` so callers
        can inject context-specific variables (Skill name/dir, request ID, ...).
        """
        env = os.environ.copy()
        if self.extra_env:
            env.update(self.extra_env)
        return env

    def available_tools(self) -> List[Tool]:
        """Return the tools provided by this class.

        Returns an empty list when no patterns are configured, so callers
        that opt out of bash execution simply omit the tool. Callers that
        want an "unrestricted" tool (gated only by ``PermissionManager``)
        should pass ``allowed_patterns=["*"]``.
        """
        if not self.allowed_patterns:
            return []
        return [trans_to_function_tool(self.execute_command)]
