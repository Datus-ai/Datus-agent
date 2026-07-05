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
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agents import Tool

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.permission.bash_rules import split_pipeline
from datus.utils.loggings import get_logger
from datus.utils.tool_archive import build_archived_marker, make_single_line_preview

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 60
# Upper bound for a per-call ``timeout`` the model may request, so a bad or
# runaway value can't hang the agent indefinitely.
MAX_BASH_TIMEOUT = 600
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

    Execution model:
    - Commands run through a real shell (``bash -c``) so pipelines and other
      shell syntax work. Permission control (deny/ask/allow, per-segment
      pipeline judging, safety ceiling for ``&&``/``$()``/redirection) happens
      upstream in ``PermissionHooks`` + ``bash_rules`` — the execution layer
      trusts an approved command. A hardening prefix (``shopt -u extglob``) is
      prepended.
    - When no ``bash`` is found (e.g. Windows without Git Bash), it falls back
      to the legacy ``shell=False`` single-argv path (``shlex.split``); pipes
      and shell operators are then NOT interpreted.

    Security features:
    - Pattern-based filtering (``allowed_patterns``; per-segment for pipelines)
    - Working directory locked to ``workspace_root``
    - Timeout enforcement with process-group kill (no orphaned pipeline stages)
    - Output size limiting
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

    def bash(self, command: str, timeout: Optional[int] = None) -> FuncToolResult:
        """Run a shell command and return its output.

        The command runs through a real shell (bash), so pipelines and shell
        syntax (``|``, ``&&``, redirection) work. It runs in a FIXED working
        directory and each call is STATELESS: ``cd`` does NOT persist to the
        next call, and shell state (variables, functions) is not carried over —
        use absolute paths instead of relying on ``cd``. Commands cannot read
        stdin (it is closed), so interactive prompts / ``read`` return EOF
        immediately rather than hanging.

        Prefer the dedicated tools when they fit — they are safer (scoped to
        the project via the filesystem policy) and produce cleaner, reviewable
        results:
          * read a file  -> ``read_file`` (not ``cat``/``head``/``tail``)
          * search text  -> ``grep`` (not raw ``grep``/``rg`` in bash)
          * find files   -> ``glob`` (not ``find``/``ls``)
        Use bash for things those tools cannot do (pipelines, running programs,
        git, package managers, etc.).

        Args:
            command: The shell command to execute (e.g. "cat log | grep err").
            timeout: Optional per-command timeout in SECONDS (capped at
                ``MAX_BASH_TIMEOUT``). Defaults to the tool's configured
                timeout. Raise it for a known slow command.

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

        effective_timeout = self._resolve_timeout(timeout)

        try:
            argv = self._build_spawn_argv(command)
        except ValueError as e:
            return FuncToolResult(success=0, error=f"Invalid command syntax: {e}")

        logger.info("Executing command (identity=%s, timeout=%ss): %s", self.identity, effective_timeout, command)
        output_dir = self._resolve_output_dir()
        try:
            if output_dir is not None:
                # Preferred path: stream the child's output straight to disk so
                # a huge output never buffers in memory; decide by file size
                # afterwards. Timeout is handled inside so the partial file is
                # still surfaced.
                return self._execute_with_redirect(argv, command, output_dir, effective_timeout)
            return self._execute_in_memory(argv, effective_timeout)
        except subprocess.TimeoutExpired:
            logger.error("Command timed out (identity=%s): %s", self.identity, command)
            return FuncToolResult(success=0, error=f"Command timed out after {effective_timeout} seconds")
        except Exception as e:
            logger.error("Command execution failed (identity=%s): %s", self.identity, e)
            return FuncToolResult(success=0, error=f"Command execution failed: {str(e)}")

    def _resolve_timeout(self, timeout: Optional[int]) -> int:
        """Clamp an optional per-call timeout into ``(0, MAX_BASH_TIMEOUT]``.

        Invalid / non-positive values fall back to the instance default so a
        bad model-supplied timeout can't disable the limit or hang the call.
        """
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            return self.timeout
        return min(timeout, MAX_BASH_TIMEOUT)

    # Cached bash path (None when unavailable → legacy single-argv fallback).
    _bash_path_cache: Optional[str] = None
    _bash_path_resolved: bool = False

    @classmethod
    def _resolve_bash(cls) -> Optional[str]:
        if not cls._bash_path_resolved:
            cls._bash_path_cache = shutil.which("bash")
            cls._bash_path_resolved = True
        return cls._bash_path_cache

    def _shell_prefix(self) -> str:
        """Hardening prefix prepended to every ``bash -c`` command.

        - ``shopt -u extglob``: disable extended globs so a malicious filename
          can't expand into something unexpected after permission validation
          (mirrors Claude Code's bashProvider hardening).
        - ``python`` shim: shadow ``python`` with the interpreter datus runs
          under, preserving the legacy ``argv[0]=='python' -> sys.executable``
          rewrite for environments where only ``python3`` exists.

        NOTE: intentionally NO ``set -o pipefail`` — bash's default takes the
        LAST stage's exit code, matching Claude Code. pipefail would flag the
        ubiquitous ``... | head`` as a failure because the upstream stage dies
        with SIGPIPE (exit 141) when ``head`` closes the pipe early.
        """
        parts = [
            "shopt -u extglob 2>/dev/null || true",
        ]
        py = sys.executable or shutil.which("python3")
        if py:
            parts.append(f'python() {{ {shlex.quote(py)} "$@"; }}')
        return "; ".join(parts) + "; "

    def _build_spawn_argv(self, command: str) -> List[str]:
        """Build the argv to spawn for ``command``.

        Real-shell path: ``[bash, -c, <prefix><command>]``. Falls back to the
        legacy ``shell=False`` single-argv (with the ``python`` rewrite) when
        no bash is available — pipes/operators then don't work, which is the
        pre-existing behavior on such platforms.
        """
        bash_path = self._resolve_bash()
        if bash_path:
            return [bash_path, "-c", self._shell_prefix() + command]

        # Legacy fallback: no shell interpretation.
        argv = shlex.split(command)
        if argv and argv[0] == "python":
            argv[0] = sys.executable or shutil.which("python3") or "python3"
        return argv

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

    def _execute_with_redirect(self, argv: List[str], command: str, output_dir: Path, timeout: int) -> FuncToolResult:
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
        with open(path, "wb") as fh:
            proc = subprocess.Popen(
                argv,
                shell=False,
                cwd=str(self.workspace_root),
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=self._get_safe_env(),
                # New session/process group so a timeout can kill the whole
                # pipeline (all stages), not just the bash launcher.
                start_new_session=(os.name == "posix"),
            )
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # The tree is killed but the file holds whatever it wrote so far.
                timed_out = True
                logger.error("Command timed out (identity=%s): %s", self.identity, command)
                self._kill_process_tree(proc)
                try:
                    # Reap the launcher so it doesn't linger as a zombie.
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:  # pragma: no cover - kill already sent
                    pass

        result_str = self._result_from_output_file(path)

        if timed_out:
            return FuncToolResult(
                success=0,
                error=f"Command timed out after {timeout} seconds",
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

    def _execute_in_memory(self, argv: List[str], timeout: int) -> FuncToolResult:
        """Fallback path (no offload dir): capture output in memory + truncate."""
        proc = subprocess.Popen(
            argv,
            shell=False,
            cwd=str(self.workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._get_safe_env(),
            # Detach the child from the agent's stdin. Without this the child
            # inherits datus's terminal stdin and any command that reads it
            # (``cat``, ``read``, ``python`` awaiting input, an interactive
            # prompt) blocks forever, fighting the TUI's prompt_toolkit for
            # the same TTY and freezing the whole process. DEVNULL delivers
            # an immediate EOF so such commands fail fast instead of hanging.
            stdin=subprocess.DEVNULL,
            # Process group so a timeout kills the whole pipeline, not just bash.
            start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            partial = stdout or ""
            if stderr:
                partial += f"\n[stderr]\n{stderr}"
            return FuncToolResult(
                success=0,
                error=f"Command timed out after {timeout} seconds",
                result=partial or None,
            )

        output = stdout or ""
        if stderr:
            output += f"\n[stderr]\n{stderr}"

        if len(output) > MAX_OUTPUT_SIZE:
            output = output[:MAX_OUTPUT_SIZE] + f"\n... [truncated, total {len(output)} chars]"

        if proc.returncode != 0:
            return FuncToolResult(
                success=0,
                error=f"Command exited with code {proc.returncode}",
                result=output,
            )

        return FuncToolResult(success=1, result=output)

    def _kill_process_tree(self, proc: "subprocess.Popen") -> None:
        """Kill a timed-out process and its whole group (all pipeline stages).

        On POSIX the child was started in a new session, so ``killpg`` reaches
        every stage. Falls back to killing just the launcher elsewhere.
        """
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:  # pragma: no cover - already dead
                pass

    def _is_command_allowed(self, command: str) -> bool:
        """Check if a command matches any allowed pattern.

        The ``["*"]`` wildcard (used by AgenticNode, where the real gate is
        ``PermissionManager``) short-circuits to allowed. For a restrictive
        whitelist (skills) a pure pipeline is allowed only if EVERY segment
        matches; any non-pipeline shell construct (``&&``, ``$()``, ...) is
        rejected outright since per-segment matching can't reason about it.
        """
        if not self.allowed_patterns:
            return False
        if "*" in self.allowed_patterns:
            return True

        segments = split_pipeline(command)
        if segments is None:
            # ``||`` / ``|&`` / empty segment / unbalanced quotes — cannot be
            # matched segment-by-segment under a restrictive whitelist.
            return False
        return all(self._segment_allowed(seg) for seg in segments)

    def _segment_allowed(self, segment: str) -> bool:
        """True if a single command segment matches any allowed pattern.

        A segment containing non-pipe shell metacharacters (chaining, command
        substitution, redirection) is rejected — the whitelist speaks about
        one command, not a compound expression.
        """
        try:
            parts = shlex.split(segment)
        except ValueError:
            return False
        if not parts:
            return False
        if any(tok in ("&&", "||", ";", "|", "&", "`", "$(", ">", "<") for tok in parts):
            return False
        base_cmd = parts[0]
        return any(self._matches_pattern(segment, base_cmd, pattern) for pattern in self.allowed_patterns)

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
