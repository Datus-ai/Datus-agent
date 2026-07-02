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
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agents import Tool

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger
from datus.utils.tool_archive import build_archived_marker, make_single_line_preview

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 60
MAX_OUTPUT_SIZE = 50000
# Output larger than this is offloaded to a file under the session data dir and
# replaced with a ``[DATUS_ARCHIVED]`` preview marker, so a huge command output
# neither buffers in memory nor pollutes the model context. Only applies when an
# ``output_dir_provider`` is wired (agentic-node sessions); otherwise the tool
# falls back to the in-memory ``MAX_OUTPUT_SIZE`` truncation.
BASH_ARCHIVE_THRESHOLD = 8000
# Single-line preview length carried inline in the archive marker.
BASH_ARCHIVE_PREVIEW_CHARS = 1000


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
        output_dir_provider: Optional[Callable[[], Optional[Path]]] = None,
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
            output_dir_provider: Optional zero-arg callable returning the
                directory where oversized command output is offloaded (the
                session data dir). Resolved lazily per call so the session id
                need not exist at construction time. When ``None`` (or it
                returns ``None``), output is captured in memory and truncated
                at :data:`MAX_OUTPUT_SIZE` — the general-purpose fallback used
                by MCP / standalone callers and tests.
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.allowed_patterns = list(allowed_patterns) if allowed_patterns else []
        self.timeout = timeout
        self.extra_env = dict(extra_env) if extra_env else {}
        self.identity = identity
        self._output_dir_provider = output_dir_provider
        # Monotonic per-instance counter zero-padded into archive filenames so a
        # directory listing sorts in command-invocation order. Paired with a
        # per-instance random token so a recreated/resumed BashTool (which
        # restarts the counter at 0 against the same reused offload dir) never
        # overwrites an earlier instance's archive — stale ``[DATUS_ARCHIVED]``
        # markers would otherwise point at the wrong payload.
        self._bash_output_seq = 0
        self._bash_output_token = uuid.uuid4().hex[:8]
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

    def bash(self, command: str) -> FuncToolResult:
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

        logger.info("Executing command (identity=%s): %s", self.identity, command)
        output_dir = self._resolve_output_dir()
        try:
            if output_dir is not None:
                # Preferred path: stream the child's output straight to disk so
                # a huge output never buffers in memory; decide by file size
                # afterwards. Timeout is handled inside so the partial file is
                # still surfaced.
                return self._execute_with_redirect(argv, command, output_dir)
            return self._execute_in_memory(argv)
        except subprocess.TimeoutExpired:
            logger.error("Command timed out (identity=%s): %s", self.identity, command)
            return FuncToolResult(success=0, error=f"Command timed out after {self.timeout} seconds")
        except Exception as e:
            logger.error("Command execution failed (identity=%s): %s", self.identity, e)
            return FuncToolResult(success=0, error=f"Command execution failed: {str(e)}")

    def _resolve_output_dir(self) -> Optional[Path]:
        """Resolve the offload directory lazily, tolerating any provider error.

        Returns ``None`` (→ in-memory fallback) when no provider is wired, the
        provider yields nothing, or the directory can't be created.
        """
        if self._output_dir_provider is None:
            return None
        try:
            raw = self._output_dir_provider()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("bash output_dir_provider raised: %s", exc)
            return None
        if not raw:
            return None
        path = Path(raw)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - defensive
            logger.debug("bash output dir mkdir failed (%s): %s", path, exc)
            return None
        return path

    def _execute_with_redirect(self, argv: List[str], command: str, output_dir: Path) -> FuncToolResult:
        """Run the command with stdout/stderr redirected to a file on disk.

        stderr is merged into the stdout file (``stderr=STDOUT``) so interleaved
        output keeps its real order. After the process exits the file size
        decides the outcome:

        - empty      → delete the file, return an empty result;
        - <= threshold → read the (small) content back, delete the file;
        - > threshold  → read only a head preview, keep the file and return a
          ``[DATUS_ARCHIVED]`` marker so the model can ``read_file(<path>)``.
        """
        seq = self._bash_output_seq
        self._bash_output_seq += 1
        cmd_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()[:8]
        path = output_dir / f"{seq:06d}_{self._bash_output_token}_bash_{cmd_hash}.txt"

        timed_out = False
        returncode = 0
        try:
            with open(path, "wb") as fh:
                proc = subprocess.run(
                    argv,
                    shell=False,
                    cwd=str(self.workspace_root),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    timeout=self.timeout,
                    env=self._get_safe_env(),
                )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            # The child is killed but the file holds whatever it wrote so far.
            timed_out = True
            logger.error("Command timed out (identity=%s): %s", self.identity, command)

        result_str = self._result_from_output_file(path)

        if timed_out:
            return FuncToolResult(
                success=0,
                error=f"Command timed out after {self.timeout} seconds",
                result=result_str or None,
            )
        if returncode != 0:
            return FuncToolResult(
                success=0,
                error=f"Command exited with code {returncode}",
                result=result_str or None,
            )
        return FuncToolResult(success=1, result=result_str)

    def _result_from_output_file(self, path: Path) -> str:
        """Turn the on-disk output file into a model-facing result string.

        Never loads more than ``BASH_ARCHIVE_PREVIEW_CHARS`` into memory for an
        oversized file.
        """
        try:
            size = path.stat().st_size
        except OSError:
            return ""

        if size == 0:
            self._safe_unlink(path)
            return ""
        if size <= BASH_ARCHIVE_THRESHOLD:
            content = path.read_text(encoding="utf-8", errors="replace")
            self._safe_unlink(path)
            return content
        # Oversized: read only the head for the inline preview; keep the file.
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(BASH_ARCHIVE_PREVIEW_CHARS)
        preview = make_single_line_preview(head, BASH_ARCHIVE_PREVIEW_CHARS)
        return build_archived_marker(path, preview)

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except OSError:  # pragma: no cover - best effort cleanup
            pass

    def _execute_in_memory(self, argv: List[str]) -> FuncToolResult:
        """Fallback path (no offload dir): capture output in memory + truncate."""
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
        return [trans_to_function_tool(self.bash)]
