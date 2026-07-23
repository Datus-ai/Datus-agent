# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
OS-level sandbox for the general-purpose ``BashTool``.

When enabled, every ``bash`` tool command is wrapped in a kernel-enforced
sandbox that restricts filesystem access to an allowlist of directories:

- **macOS**: ``sandbox-exec`` with a generated Seatbelt profile. Writes are
  denied outside the writable roots; reads (``file-read-data``) are denied
  outside the readable + writable roots. Metadata reads (stat/readlink) stay
  allowed so PATH lookup, ``cd`` and dyld keep working.
- **Linux**: ``bubblewrap`` (``bwrap``) mount namespace. Only allowlisted
  directories are bind-mounted (system dirs read-only, workspace/tmp
  writable); everything else simply does not exist inside the sandbox.

This module depends only on the stdlib plus leaf ``datus.utils`` helpers
(``loggings``, ``exceptions``), so ``datus.configuration`` can import
:class:`SandboxSettings` without any dependency cycle. The permission layer
(``PermissionHooks`` / ``bash_rules``) is unaffected — the sandbox is a second,
kernel-level line of defense at the execution layer.

Two modes: ``normal`` (CLI default — datus home readable, session data dir
writable) and ``strict`` (multi-tenant tier — project workspace + tmp +
explicit allowlists only, minimal child environment). Network stays shared
unless ``deny_network`` is set.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

MECHANISM_SEATBELT = "seatbelt"
MECHANISM_BWRAP = "bwrap"

# Sandbox modes. ``normal`` keeps the CLI-friendly defaults (datus home
# readable for skills/templates, session data dir writable for command-side
# output). ``strict`` is the multi-tenant hardening tier: file access shrinks
# to the workspace + tmp + explicit allowlists ONLY — the caller-injected
# datus-home read root and session-dir write root are ignored — and the child
# environment is reduced to a small allowlist so process-wide secrets
# (LLM API keys, DB passwords) never reach sandboxed commands.
MODE_NORMAL = "normal"
MODE_STRICT = "strict"
SANDBOX_MODES = (MODE_NORMAL, MODE_STRICT)

# Environment variables forwarded to the child in strict mode (plus every
# ``LC_*`` locale variable and the caller's explicit ``extra_env``). The
# ``python`` shim uses an absolute interpreter path, so nothing here is
# needed for it to work.
STRICT_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "TZ",
        "TERM",
        "TMPDIR",
        "USER",
        "LOGNAME",
        "SHELL",
        "PWD",
        "COLUMNS",
        "LINES",
    }
)

# Seconds allowed for the one-shot availability probe.
_PROBE_TIMEOUT = 10
_PROBE_PROFILE = "(version 1)(allow default)"

# macOS system roots that stay readable inside the sandbox: shells, coreutils,
# interpreters, dyld caches, frameworks, timezone/locale data. All values are
# canonical (``/etc`` and ``/tmp`` are symlinks into ``/private``; Seatbelt
# matches on the resolved path).
_MACOS_SYSTEM_READ_ROOTS = (
    "/usr",
    "/bin",
    "/sbin",
    "/opt",
    "/System",
    "/Library",
    "/private/etc",
    "/private/var/db",
    "/private/var/select",
    "/dev",
)

# Paths whose DATA must stay readable as exact literals. Reading a
# directory's entries counts as ``file-read-data`` on the directory itself,
# and dyld reads ``/`` during cryptex/firmlink resolution at every process
# start — denying it aborts (SIGABRT) before main() runs.
_MACOS_READ_LITERALS = ("/",)

# Device nodes that remain writable on macOS (bwrap covers these via
# ``--dev /dev``). ``/dev/stdout`` / ``/dev/stderr`` are symlinks into
# ``/dev/fd`` which is exempted as a subpath.
_MACOS_WRITE_DEVICE_LITERALS = (
    "/dev/null",
    "/dev/zero",
    "/dev/tty",
    "/dev/random",
    "/dev/urandom",
)
_MACOS_WRITE_DEVICE_SUBPATHS = ("/dev/fd",)

# Linux system roots bind-mounted read-only. On merged-usr distros ``/bin``,
# ``/sbin`` and ``/lib*`` are symlinks into ``/usr`` and are recreated as
# symlinks inside the sandbox instead of bind mounts. ``/run`` is needed for
# systemd-resolved's resolv.conf target.
_LINUX_SYSTEM_READ_DIRS = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/etc",
    "/opt",
    "/run",
)

# Provided as fresh tmpfs mounts inside the bwrap sandbox rather than binding
# the host directories: commands get a private scratch space and can't read
# other processes' host tmp files. Writable roots equal to these paths are
# skipped (the tmpfs already provides them); writable roots UNDER them (e.g. a
# pytest tmp_path workspace) are bind-mounted after the tmpfs and shadow it.
_BWRAP_TMPFS_DIRS = ("/tmp", "/var/tmp")


class SandboxUnavailableError(DatusException):
    """Raised when wrapping is requested but no OS sandbox mechanism works."""

    def __init__(self, message: str):
        super().__init__(ErrorCode.BASH_SANDBOX_UNAVAILABLE, message=message)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return default


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def _coerce_mode(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in SANDBOX_MODES:
        return value.strip().lower()
    if value is not None:
        logger.warning("Invalid sandbox mode %r; expected one of %s. Using %r.", value, SANDBOX_MODES, MODE_NORMAL)
    return MODE_NORMAL


def strict_env(base_env: Dict[str, str]) -> Dict[str, str]:
    """Reduce an environment mapping to the strict-mode allowlist.

    Keeps :data:`STRICT_ENV_ALLOWLIST` plus ``LC_*`` locale variables so
    shells and coreutils behave normally while process-wide secrets stay out
    of the sandbox. Callers layer their explicit ``extra_env`` on top.
    """
    return {k: v for k, v in base_env.items() if k in STRICT_ENV_ALLOWLIST or k.startswith("LC_")}


@dataclass
class SandboxSettings:
    """Parsed ``agent.bash.sandbox`` configuration section.

    Mutable at runtime by design: ``/sandbox on|off`` flips ``enabled`` in
    place and every ``BashTool`` holding this object picks the change up on
    its next call (the tool and ``AgentConfig`` share the same instance).
    """

    enabled: bool = False
    mode: str = MODE_NORMAL
    allow_read: List[str] = field(default_factory=list)
    allow_write: List[str] = field(default_factory=list)
    # ``deny_network`` cuts network access inside the sandbox (bwrap
    # ``--unshare-net`` / Seatbelt ``deny network*``). Orthogonal to ``mode``;
    # off by default because it also blocks legitimate outbound calls.
    deny_network: bool = False

    @property
    def is_strict(self) -> bool:
        return self.mode == MODE_STRICT

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "SandboxSettings":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            enabled=_coerce_bool(raw.get("enabled"), False),
            mode=_coerce_mode(raw.get("mode")),
            allow_read=_coerce_str_list(raw.get("allow_read")),
            allow_write=_coerce_str_list(raw.get("allow_write")),
            deny_network=_coerce_bool(raw.get("deny_network"), False),
        )


@dataclass(frozen=True)
class SandboxPolicy:
    """Resolved allowlist for one command execution.

    All paths are canonical (``os.path.realpath``), deduplicated and verified
    to exist. Platform system baselines (:data:`_MACOS_SYSTEM_READ_ROOTS` /
    :data:`_LINUX_SYSTEM_READ_DIRS`) are added by the mechanism-specific
    builders, not stored here, so the policy itself is platform-neutral.
    """

    cwd: str
    writable_roots: Tuple[str, ...]
    # Extra read-only roots beyond the platform baseline (datus home, the
    # python interpreter prefixes, user allow_read). Never overlaps
    # ``writable_roots`` — write implies read in both mechanisms.
    readable_roots: Tuple[str, ...]
    # Cut network access: bwrap ``--unshare-net`` / Seatbelt ``deny network*``
    # (the latter also covers unix-domain sockets).
    deny_network: bool = False


def _normalize_roots(candidates: Sequence[Any]) -> List[str]:
    """Expand ``~``, resolve symlinks, drop missing paths and duplicates."""
    out: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        resolved = os.path.realpath(os.path.expanduser(str(candidate)))
        if os.path.exists(resolved) and resolved not in out:
            out.append(resolved)
    return out


def build_policy(
    settings: SandboxSettings,
    workspace_root: Path,
    dynamic_write_dirs: Sequence[Any] = (),
    extra_read_dirs: Sequence[str] = (),
    runtime_read_dirs: Sequence[str] = (),
) -> SandboxPolicy:
    """Merge defaults, configuration and per-call dirs into a resolved policy.

    Writable: workspace root, per-call dynamic dirs (session output dir), the
    process tmp dir plus ``/tmp`` / ``/var/tmp`` (realpath'd — ``/private/*``
    on macOS), and ``settings.allow_write``.

    Readable: the python interpreter prefixes (the ``python`` shim in
    ``BashTool._shell_prefix`` points at ``sys.executable``, whose venv often
    lives outside the workspace), caller-injected ``extra_read_dirs`` (datus
    home, so skills/templates stay usable), validated per-invocation
    ``runtime_read_dirs`` and ``settings.allow_read``.

    Strict mode ignores the caller-injected ``dynamic_write_dirs`` and
    ``extra_read_dirs`` entirely — in a multi-tenant deployment those point
    into the shared ``~/.datus`` tree, which strict tenants must not touch.
    ``runtime_read_dirs`` survive strict mode because they are narrow,
    invocation-scoped capabilities (for example the exact plugin directory
    whose metadata name matched the command). Callers MUST validate them
    before passing them here; raw user paths never belong in this parameter.
    Filtering here (rather than at each call site) makes the guarantee hold
    no matter who constructs the tool. Oversized-output archiving keeps
    working: the redirect file is opened by the parent process and inherited
    as an fd, and both mechanisms check permissions at open time only.
    Explicit ``allow_read``/``allow_write`` still apply — they are the
    operator's deliberate exceptions.
    """
    if settings.is_strict:
        dynamic_write_dirs = ()
        extra_read_dirs = ()
    writable = _normalize_roots(
        [
            workspace_root,
            *dynamic_write_dirs,
            tempfile.gettempdir(),
            "/tmp",
            "/var/tmp",
            *settings.allow_write,
        ]
    )
    readable = [
        root
        for root in _normalize_roots(
            [sys.prefix, sys.base_prefix, *extra_read_dirs, *runtime_read_dirs, *settings.allow_read]
        )
        if root not in writable
    ]
    return SandboxPolicy(
        cwd=os.path.realpath(str(workspace_root)),
        writable_roots=tuple(writable),
        readable_roots=tuple(readable),
        deny_network=settings.deny_network,
    )


# ---------------------------------------------------------------------------
# Mechanism detection (cached per process)
# ---------------------------------------------------------------------------

_UNSET = object()
_mechanism_cache: Any = _UNSET


def _find_sandbox_exec() -> Optional[str]:
    if os.path.exists("/usr/bin/sandbox-exec"):
        return "/usr/bin/sandbox-exec"
    return shutil.which("sandbox-exec")


def _find_bwrap() -> Optional[str]:
    return shutil.which("bwrap")


def _probe(argv: List[str]) -> bool:
    """Run a trivial command under the candidate wrapper.

    ``which`` alone is not enough: bwrap can be installed yet unusable (no
    user namespaces inside containers, non-setuid builds), which would turn
    every command into a confusing runtime failure instead of one clear
    fail-closed error.
    """
    try:
        result = subprocess.run(argv, capture_output=True, timeout=_PROBE_TIMEOUT)
        return result.returncode == 0
    except Exception as e:
        logger.debug("Sandbox probe failed for %s: %s", argv[0], e)
        return False


def _detect_mechanism_uncached() -> Optional[str]:
    if sys.platform == "darwin":
        exe = _find_sandbox_exec()
        if exe and _probe([exe, "-p", _PROBE_PROFILE, "/usr/bin/true"]):
            return MECHANISM_SEATBELT
        return None
    if sys.platform.startswith("linux"):
        exe = _find_bwrap()
        true_bin = shutil.which("true") or "/bin/true"
        if exe and _probe([exe, "--ro-bind", "/", "/", true_bin]):
            return MECHANISM_BWRAP
        return None
    return None


def detect_mechanism() -> Optional[str]:
    """Return ``"seatbelt"`` / ``"bwrap"`` / ``None``, probing once per process."""
    global _mechanism_cache
    if _mechanism_cache is _UNSET:
        _mechanism_cache = _detect_mechanism_uncached()
        logger.info("Bash sandbox mechanism: %s", _mechanism_cache or "unavailable")
    return _mechanism_cache


def _reset_detection_cache() -> None:
    global _mechanism_cache
    _mechanism_cache = _UNSET


def is_available() -> bool:
    return detect_mechanism() is not None


def unavailable_reason() -> str:
    if sys.platform == "darwin":
        return "macOS sandbox-exec is missing or failed its probe"
    if sys.platform.startswith("linux"):
        return "bubblewrap (bwrap) is not installed or cannot run on this system (e.g. no user namespaces)"
    return f"no OS sandbox mechanism exists for platform {sys.platform!r} (Windows is unsupported)"


# ---------------------------------------------------------------------------
# macOS Seatbelt
# ---------------------------------------------------------------------------


def _sb_quote(path: str) -> str:
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _sb_filters(subpaths: Sequence[str], literals: Sequence[str] = ()) -> str:
    lines = [f"    (subpath {_sb_quote(p)})" for p in subpaths]
    lines.extend(f"    (literal {_sb_quote(p)})" for p in literals)
    return "\n".join(lines)


def build_seatbelt_profile(policy: SandboxPolicy) -> str:
    """Generate the Seatbelt profile text for ``sandbox-exec -p``.

    Structure: ``(allow default)`` plus two targeted denies. A ``(deny
    default)`` profile needs dozens of mach/sysctl/iokit allowances that drift
    across OS releases; scoping the denies to filesystem access keeps the
    sandbox lightweight and predictable.

    Only ``file-read-data`` is denied for reads — denying ``file-read*`` would
    also block metadata (stat/readlink), breaking PATH lookup, ``cd`` and
    dyld everywhere. Outside the allowlist, file names are visible but their
    contents are not.
    """
    write_subpaths = policy.writable_roots
    read_subpaths = tuple(
        dict.fromkeys(
            (
                *_MACOS_SYSTEM_READ_ROOTS,
                *policy.readable_roots,
                *policy.writable_roots,
                *_MACOS_WRITE_DEVICE_SUBPATHS,
            )
        )
    )
    profile = (
        "(version 1)\n"
        "(allow default)\n"
        "(deny file-write*\n"
        "  (require-not (require-any\n"
        f"{_sb_filters((*write_subpaths, *_MACOS_WRITE_DEVICE_SUBPATHS), _MACOS_WRITE_DEVICE_LITERALS)}\n"
        "  )))\n"
        "(deny file-read-data\n"
        "  (require-not (require-any\n"
        f"{_sb_filters(read_subpaths, _MACOS_READ_LITERALS)}\n"
        "  )))\n"
    )
    if policy.deny_network:
        profile += "(deny network*)\n"
    return profile


# ---------------------------------------------------------------------------
# Linux bubblewrap
# ---------------------------------------------------------------------------


def _mount_args_for_system_dir(path: str) -> List[str]:
    """``--symlink`` for merged-usr symlinks, ``--ro-bind`` for real dirs.

    Missing paths yield no args: bwrap errors out on a nonexistent bind
    source, and filtering in Python keeps behavior deterministic without
    depending on ``--ro-bind-try`` (bwrap >= 0.3).
    """
    if os.path.islink(path):
        return ["--symlink", os.readlink(path), path]
    if os.path.isdir(path):
        return ["--ro-bind", path, path]
    return []


def build_bwrap_prefix(policy: SandboxPolicy) -> List[str]:
    """Build the ``bwrap ...`` argv prefix implementing the policy.

    Mount order matters — later mounts shadow earlier ones — so read-only
    binds come first and writable binds last (e.g. the session output dir
    stays writable underneath the read-only datus home). Network is left
    shared and the host pid namespace is kept so ``killpg`` from
    ``BashTool._kill_process_tree`` still reaches every child.
    """
    bwrap = _find_bwrap()
    if not bwrap:
        raise SandboxUnavailableError(unavailable_reason())
    args = [bwrap, "--die-with-parent"]
    if policy.deny_network:
        args.append("--unshare-net")
    for system_dir in _LINUX_SYSTEM_READ_DIRS:
        args.extend(_mount_args_for_system_dir(system_dir))
    args.extend(["--proc", "/proc", "--dev", "/dev"])
    for tmp_dir in _BWRAP_TMPFS_DIRS:
        args.extend(["--tmpfs", tmp_dir])
    for readable in policy.readable_roots:
        args.extend(["--ro-bind", readable, readable])
    for writable in policy.writable_roots:
        if writable in _BWRAP_TMPFS_DIRS:
            continue
        args.extend(["--bind", writable, writable])
    args.extend(["--chdir", policy.cwd])
    return args


def wrap_argv(argv: List[str], policy: SandboxPolicy) -> List[str]:
    """Wrap a spawn argv in the platform sandbox.

    Raises :class:`SandboxUnavailableError` when no mechanism is usable —
    callers are expected to have checked :func:`is_available` already; this
    is the fail-closed backstop.
    """
    mechanism = detect_mechanism()
    if mechanism == MECHANISM_SEATBELT:
        exe = _find_sandbox_exec()
        if not exe:
            raise SandboxUnavailableError(unavailable_reason())
        return [exe, "-p", build_seatbelt_profile(policy), *argv]
    if mechanism == MECHANISM_BWRAP:
        return [*build_bwrap_prefix(policy), *argv]
    raise SandboxUnavailableError(unavailable_reason())
