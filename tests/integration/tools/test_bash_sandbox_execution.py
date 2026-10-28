# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Real OS-sandbox execution tests for the bash sandbox.

These spawn the actual mechanism (macOS ``sandbox-exec`` / Linux ``bwrap``)
and verify the kernel-enforced guarantees: writes and reads outside the
allowlist fail, allowlisted and system paths keep working, and ``cd`` cannot
escape. Skipped wherever no mechanism is available (e.g. CI runners without
bubblewrap) — the pure argv/profile generation logic is covered by
``tests/unit_tests/tools/func_tool/test_bash_sandbox.py`` on every platform.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from datus.tools.func_tool import bash_sandbox
from datus.tools.func_tool.bash_sandbox import SandboxPolicy, SandboxSettings, wrap_argv
from datus.tools.func_tool.bash_tool import BashTool

requires_sandbox = pytest.mark.skipif(
    bash_sandbox.detect_mechanism() is None,
    reason="no OS sandbox mechanism available (needs macOS sandbox-exec or Linux bwrap)",
)

BASH = shutil.which("bash")


def run_sandboxed(command: str, policy: SandboxPolicy, cwd: str):
    argv = wrap_argv([BASH, "-c", command], policy)
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=30)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws.resolve()


@pytest.fixture
def outside_home_dir():
    """A directory outside every default-writable root (workspace, tmp).

    pytest's ``tmp_path`` lives under the process tmp dir, which the default
    policy makes writable — escape targets must live elsewhere, so this one
    goes under the real home directory.
    """
    path = Path(tempfile.mkdtemp(prefix=".datus_sbx_test_", dir=str(Path.home())))
    yield path.resolve()
    # Safe: removes only the uniquely-named dir mkdtemp created two lines up;
    # it must live under $HOME because the sandbox escape target has to sit
    # outside the default-writable tmp tree.
    shutil.rmtree(path, ignore_errors=True)  # audit-noqa: rmtree_outside_tmp


def minimal_policy(workspace, readable=()):
    """Workspace-only write policy; system read baselines come from the builders."""
    return SandboxPolicy(
        cwd=str(workspace),
        writable_roots=(str(workspace),),
        readable_roots=(str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve()), *readable),
    )


@requires_sandbox
class TestSandboxedExecution:
    def test_write_inside_workspace_succeeds(self, workspace):
        result = run_sandboxed("echo data > inside.txt", minimal_policy(workspace), str(workspace))
        assert result.returncode == 0, result.stderr
        assert (workspace / "inside.txt").read_text().strip() == "data"

    def test_write_outside_workspace_fails(self, workspace, outside_home_dir):
        target = outside_home_dir / "escape.txt"
        result = run_sandboxed(f"echo pwned > {target}", minimal_policy(workspace), str(workspace))
        assert result.returncode != 0
        assert not target.exists()

    def test_cd_escape_write_fails(self, workspace, outside_home_dir):
        target = outside_home_dir / "cd_escape.txt"
        result = run_sandboxed(
            f"cd {outside_home_dir} 2>/dev/null; touch {target}",
            minimal_policy(workspace),
            str(workspace),
        )
        assert result.returncode != 0
        assert not target.exists()

    def test_read_outside_allowlist_fails(self, workspace, outside_home_dir):
        secret = outside_home_dir / "secret.txt"
        secret.write_text("top-secret")
        result = run_sandboxed(f"cat {secret}", minimal_policy(workspace), str(workspace))
        # macOS Seatbelt: EPERM on open; Linux bwrap: path not bound -> ENOENT.
        assert result.returncode != 0
        assert "top-secret" not in result.stdout

    def test_read_inside_allowlist_succeeds(self, workspace, outside_home_dir):
        shared = outside_home_dir / "shared.txt"
        shared.write_text("readable")
        policy = minimal_policy(workspace, readable=(str(outside_home_dir),))
        result = run_sandboxed(f"cat {shared}", policy, str(workspace))
        assert result.returncode == 0, result.stderr
        assert "readable" in result.stdout

    def test_system_read_succeeds(self, workspace):
        result = run_sandboxed(
            "cat /etc/hosts > /dev/null && ls /usr/bin | head -1", minimal_policy(workspace), str(workspace)
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()

    def test_write_to_system_dirs_fails(self, workspace):
        result = run_sandboxed(
            "touch /usr/local/datus_sbx_probe 2>/dev/null", minimal_policy(workspace), str(workspace)
        )
        assert result.returncode != 0
        assert not Path("/usr/local/datus_sbx_probe").exists()

    def test_pipeline_works_inside_sandbox(self, workspace):
        (workspace / "log.txt").write_text("a\nerror: boom\nb\n")
        result = run_sandboxed("cat log.txt | grep error | wc -l", minimal_policy(workspace), str(workspace))
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1"


@requires_sandbox
class TestBashToolFullChain:
    """End-to-end through BashTool.bash() with the default build_policy."""

    def _tool(self, workspace):
        return BashTool(
            workspace_root=str(workspace),
            allowed_patterns=["*"],
            sandbox_settings=SandboxSettings(enabled=True),
        )

    def test_python_shim_runs_in_sandbox(self, workspace):
        result = self._tool(workspace).bash('python -c "print(11 + 31)"')
        assert result.success == 1, result.error
        assert "42" in result.result

    def test_workspace_write_succeeds(self, workspace):
        result = self._tool(workspace).bash("echo ok > out.txt && cat out.txt")
        assert result.success == 1, result.error
        assert "ok" in result.result

    def test_home_escape_fails(self, workspace, outside_home_dir):
        target = outside_home_dir / "full_chain_escape.txt"
        result = self._tool(workspace).bash(f"echo pwned > {target}")
        assert not target.exists()
        # bash surfaces the denial as a non-zero exit -> FuncToolResult error.
        assert result.success == 0

    def test_tmp_stays_writable(self, workspace):
        # The default policy keeps tmp writable (scratch space contract).
        result = self._tool(workspace).bash('f=$(mktemp) && echo scratch > "$f" && cat "$f" && rm -f "$f"')
        assert result.success == 1, result.error
        assert "scratch" in result.result


@pytest.fixture
def loopback_server():
    """Local TCP listener so network tests never leave the machine."""
    import socketserver
    import threading

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.sendall(b"pong")

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


@requires_sandbox
class TestStrictTier:
    """Kernel-level checks for the multi-tenant strict mode."""

    def _tool(self, workspace, mode, read_dirs=None):
        return BashTool(
            workspace_root=str(workspace),
            allowed_patterns=["*"],
            sandbox_settings=SandboxSettings(enabled=True, mode=mode),
            sandbox_read_dirs=read_dirs,
        )

    def test_injected_read_dirs_honored_in_normal_mode(self, workspace, outside_home_dir):
        secret = outside_home_dir / "shared.txt"
        secret.write_text("normal-readable")
        tool = self._tool(workspace, "normal", read_dirs=[str(outside_home_dir)])
        result = tool.bash(f"cat {secret}")
        assert result.success == 1, result.error
        assert "normal-readable" in result.result

    def test_injected_read_dirs_blocked_in_strict_mode(self, workspace, outside_home_dir):
        # Same injection (the datus-home slot) must be ignored by strict —
        # this is the multi-tenant "~/.datus is off limits" guarantee.
        secret = outside_home_dir / "shared.txt"
        secret.write_text("strict-blocked")
        tool = self._tool(workspace, "strict", read_dirs=[str(outside_home_dir)])
        result = tool.bash(f"cat {secret}")
        assert result.success == 0
        assert "strict-blocked" not in (result.result or "")

    def test_strict_hides_env_secret_under_real_sandbox(self, workspace, monkeypatch):
        monkeypatch.setenv("DATUS_SBX_SECRET", "sk-real-leak")
        result = self._tool(workspace, "strict").bash('echo "[${DATUS_SBX_SECRET:-absent}]"')
        assert result.success == 1, result.error
        assert "[absent]" in result.result


@requires_sandbox
class TestDenyNetwork:
    def _policy(self, workspace, deny):
        return SandboxPolicy(
            cwd=str(workspace),
            writable_roots=(str(workspace),),
            readable_roots=(str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve())),
            deny_network=deny,
        )

    def test_loopback_allowed_by_default(self, workspace, loopback_server):
        # bash's /dev/tcp builtin needs no external binaries.
        cmd = f"exec 3<>/dev/tcp/127.0.0.1/{loopback_server} && echo connected"
        result = run_sandboxed(cmd, self._policy(workspace, deny=False), str(workspace))
        assert result.returncode == 0, result.stderr
        assert "connected" in result.stdout

    def test_loopback_blocked_with_deny_network(self, workspace, loopback_server):
        cmd = f"exec 3<>/dev/tcp/127.0.0.1/{loopback_server} && echo connected"
        result = run_sandboxed(cmd, self._policy(workspace, deny=True), str(workspace))
        assert result.returncode != 0
        assert "connected" not in result.stdout
