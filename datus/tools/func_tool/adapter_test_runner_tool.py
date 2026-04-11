# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Pytest runner for generated adapter projects.

This tool is intentionally narrow: it runs ``pytest`` inside a generated
adapter project directory so the gen_adapter agent can close the loop
(scaffold -> implement -> validate -> run tests -> fix -> re-run).

Security / scope constraints (enforced by ``_validate_inputs``):

- ``project_dir`` must be an absolute path and must contain ``pyproject.toml``
  (whitelist: only scaffolded adapter projects are runnable, not arbitrary
  directories).
- ``test_subpath`` must be a relative path under ``tests/``, must not contain
  ``..`` segments, and must resolve inside ``project_dir``.
- No free-form pytest args are accepted — the argument list is fixed to
  ``-q --tb=short --no-header`` to prevent plugin injection.
- The subprocess is hard-timed to :data:`PYTEST_TIMEOUT_SECONDS`.
- Stdout/stderr are truncated to :data:`_MAX_OUTPUT_CHARS` tail bytes so a
  failing test cannot blow up the LLM context window.

This is **not** a general-purpose shell tool.
"""

import os
import subprocess
import sys
from typing import List

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

#: Hard timeout for a single pytest invocation (seconds).
PYTEST_TIMEOUT_SECONDS = 120

#: Maximum characters of stdout/stderr returned to the caller. Longer output
#: is truncated from the head — the tail carries the failure summary.
_MAX_OUTPUT_CHARS = 8000


def _truncate_tail(text: str, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate *text* to *max_chars* keeping the tail (which contains the
    pytest summary / failure details)."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return f"[... {dropped} chars truncated from head ...]\n{text[-max_chars:]}"


def _validate_inputs(project_dir: str, test_subpath: str) -> str:
    """Validate inputs; return an error message or empty string on success."""
    if not project_dir or not isinstance(project_dir, str):
        return "project_dir must be a non-empty string"
    if not os.path.isabs(project_dir):
        return f"project_dir must be an absolute path, got {project_dir!r}"
    if not os.path.isdir(project_dir):
        return f"project_dir does not exist or is not a directory: {project_dir}"

    pyproject = os.path.join(project_dir, "pyproject.toml")
    if not os.path.isfile(pyproject):
        return f"project_dir must contain pyproject.toml (only scaffolded adapter projects can be run): {project_dir}"

    if not test_subpath or not isinstance(test_subpath, str):
        return "test_subpath must be a non-empty string"
    if os.path.isabs(test_subpath):
        return f"test_subpath must be a relative path, got {test_subpath!r}"
    # Normalize and check for traversal. Use posix-style split so Windows paths
    # don't accidentally pass. Adapter projects use ``tests/...`` by convention.
    normalized_parts = test_subpath.replace("\\", "/").split("/")
    if any(part in ("..", "") for part in normalized_parts[:-1]) or ".." in normalized_parts:
        return f"test_subpath must not contain '..' segments, got {test_subpath!r}"
    if normalized_parts[0] != "tests":
        return f"test_subpath must start with 'tests/', got {test_subpath!r}"

    resolved = os.path.realpath(os.path.join(project_dir, test_subpath))
    project_real = os.path.realpath(project_dir)
    if not (resolved == project_real or resolved.startswith(project_real + os.sep)):
        return f"test_subpath resolves outside project_dir: {test_subpath!r}"
    if not os.path.exists(resolved):
        return f"test path does not exist: {resolved}"

    return ""


class AdapterTestRunnerTool:
    """FuncTool for running pytest inside a scaffolded adapter project.

    The tool is mounted by :class:`GenAdapterAgenticNode` so the LLM can close
    the gen-adapter loop: scaffold -> implement -> run contract tests -> fix
    any failures -> re-run.
    """

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config

    def available_tools(self) -> List[Tool]:
        return [trans_to_function_tool(self.run_adapter_pytest)]

    def run_adapter_pytest(
        self,
        project_dir: str,
        test_subpath: str = "tests/unit",
    ) -> FuncToolResult:
        """Run pytest inside a generated adapter project and return the results.

        This is a **narrow** pytest runner — it only accepts a project
        directory and a test subpath. It does not accept pytest flags,
        environment variables, or filter patterns. Use it to run the contract
        tests (or any ``tests/unit`` tests) against a scaffolded adapter
        project and iterate on the implementation.

        Args:
            project_dir: Absolute path to the scaffolded adapter project
                directory. Must contain a ``pyproject.toml``. The project
                directory is added to ``PYTHONPATH`` so the generated package
                is importable without a prior ``pip install -e``.
            test_subpath: Relative path under the project directory pointing
                at the tests to run. Must start with ``tests/`` and must not
                contain ``..``. Defaults to ``tests/unit``.

        Returns:
            FuncToolResult whose ``result`` dict contains:
                - ``exit_code`` (int): Pytest exit code (0 = all passed).
                - ``passed`` (bool): True iff ``exit_code == 0``.
                - ``stdout`` (str): Pytest stdout, truncated to ~8000 chars tail.
                - ``stderr`` (str): Pytest stderr, truncated to ~8000 chars tail.
                - ``timed_out`` (bool): True if the run exceeded the hard timeout.
                - ``command`` (str): The command that was executed.
        """
        validation_error = _validate_inputs(project_dir, test_subpath)
        if validation_error:
            logger.warning("run_adapter_pytest rejected input: %s", validation_error)
            return FuncToolResult(success=0, error=validation_error)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_subpath,
            "-q",
            "--tb=short",
            "--no-header",
        ]

        # Make the generated package importable without a prior editable install.
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = project_dir + (os.pathsep + existing_pp if existing_pp else "")

        command_str = " ".join(cmd)
        logger.info("Running adapter pytest: %s (cwd=%s)", command_str, project_dir)

        try:
            completed = subprocess.run(
                cmd,
                cwd=project_dir,
                env=env,
                timeout=PYTEST_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning("adapter pytest timed out after %ds", PYTEST_TIMEOUT_SECONDS)
            stdout = _truncate_tail(exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
            stderr = _truncate_tail(exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
            return FuncToolResult(
                result={
                    "exit_code": -1,
                    "passed": False,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": True,
                    "command": command_str,
                }
            )
        except FileNotFoundError as exc:
            msg = f"pytest executable not found: {exc}"
            logger.error(msg)
            return FuncToolResult(success=0, error=msg)

        result = {
            "exit_code": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout": _truncate_tail(completed.stdout),
            "stderr": _truncate_tail(completed.stderr),
            "timed_out": False,
            "command": command_str,
        }
        logger.info(
            "adapter pytest finished: exit=%d passed=%s",
            completed.returncode,
            result["passed"],
        )
        return FuncToolResult(result=result)
