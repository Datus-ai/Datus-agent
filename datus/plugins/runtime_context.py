# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Request-scoped runtime context for managed plugin CLI subprocesses.

In a multi-tenant API deployment the authoritative :class:`AgentConfig` may
exist only in the parent process (it is supplied by an AuthProvider), while a
``datus <plugin>`` command runs in a fresh subprocess.  This module carries the
minimum invocation-specific configuration across that process boundary without
requiring an on-disk ``agent.yml``.

The context is intentionally passed through one command-scoped environment
variable.  It contains only the invoked plugin's resolved profile and, when
needed, the exact plugin directory selected by the normal managed-store /
``agent.plugin_paths`` precedence.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datus.tools.permission.bash_rules import split_pipeline

RUNTIME_CONTEXT_ENV = "DATUS_PLUGIN_RUNTIME_CONTEXT"
RUNTIME_CONTEXT_VERSION = 1
RUNTIME_CONTEXT_PREFIX = "v1."
MAX_RUNTIME_CONTEXT_SIZE = 64 * 1024
_DATUS_COMMAND_WORD_RE = re.compile(r"(?<![A-Za-z0-9_.-])datus(?![A-Za-z0-9_.-])")


class PluginRuntimeContextError(ValueError):
    """Safe, user-facing failure while preparing or decoding runtime context."""


@dataclass(frozen=True)
class PluginRuntimeContext:
    """Configuration consumed by one ``datus <plugin>`` subprocess."""

    plugin_name: str
    profile: Dict[str, Any]
    plugin_path: Optional[str] = None
    version: int = RUNTIME_CONTEXT_VERSION

    def encode(self) -> str:
        """Return the versioned, ASCII-only environment value."""
        try:
            raw = json.dumps(
                {
                    "version": self.version,
                    "plugin_name": self.plugin_name,
                    "profile": self.profile,
                    "plugin_path": self.plugin_path,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise PluginRuntimeContextError(
                f"Plugin profile for `{self.plugin_name}` is not JSON-serializable"
            ) from exc
        encoded = RUNTIME_CONTEXT_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii")
        if len(encoded.encode("ascii")) > MAX_RUNTIME_CONTEXT_SIZE:
            raise PluginRuntimeContextError(
                f"Plugin runtime context for `{self.plugin_name}` exceeds {MAX_RUNTIME_CONTEXT_SIZE // 1024} KiB"
            )
        return encoded

    @classmethod
    def decode(cls, value: str, *, expected_plugin: Optional[str] = None) -> "PluginRuntimeContext":
        """Validate and decode an environment value without logging its contents."""
        if not isinstance(value, str) or not value.startswith(RUNTIME_CONTEXT_PREFIX):
            raise PluginRuntimeContextError("Unsupported plugin runtime context version")
        if len(value.encode("utf-8", errors="ignore")) > MAX_RUNTIME_CONTEXT_SIZE:
            raise PluginRuntimeContextError(f"Plugin runtime context exceeds {MAX_RUNTIME_CONTEXT_SIZE // 1024} KiB")
        encoded = value[len(RUNTIME_CONTEXT_PREFIX) :]
        try:
            raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PluginRuntimeContextError("Malformed plugin runtime context") from exc
        if not isinstance(data, dict):
            raise PluginRuntimeContextError("Malformed plugin runtime context")
        if data.get("version") != RUNTIME_CONTEXT_VERSION:
            raise PluginRuntimeContextError("Unsupported plugin runtime context version")
        plugin_name = data.get("plugin_name")
        profile = data.get("profile")
        plugin_path = data.get("plugin_path")
        if not isinstance(plugin_name, str) or not plugin_name:
            raise PluginRuntimeContextError("Plugin runtime context has no valid plugin name")
        if expected_plugin is not None and plugin_name != expected_plugin:
            raise PluginRuntimeContextError(f"Plugin runtime context is for `{plugin_name}`, not `{expected_plugin}`")
        if not isinstance(profile, dict):
            raise PluginRuntimeContextError("Plugin runtime context profile must be an object")
        if plugin_path is not None and (not isinstance(plugin_path, str) or not plugin_path.strip()):
            raise PluginRuntimeContextError("Plugin runtime context path must be a non-empty string")
        return cls(
            plugin_name=plugin_name,
            profile=profile,
            plugin_path=plugin_path,
            version=RUNTIME_CONTEXT_VERSION,
        )


@dataclass(frozen=True)
class PreparedPluginInvocation:
    """Bash execution overrides for one managed plugin invocation."""

    command: str
    env: Dict[str, str]
    sandbox_read_dirs: List[str]


def split_plugin_globals(args: List[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Consume leading ``--profile`` / ``--config`` options for a plugin."""
    profile: Optional[str] = None
    config: Optional[str] = None
    i = 0
    while i < len(args):
        token = args[i]
        if token in ("--profile", "--config"):
            if i + 1 >= len(args):
                break
            if token == "--profile":
                profile = args[i + 1]
            else:
                config = args[i + 1]
            i += 2
            continue
        if token.startswith("--profile="):
            profile = token.split("=", 1)[1]
            i += 1
            continue
        if token.startswith("--config="):
            config = token.split("=", 1)[1]
            i += 1
            continue
        break
    return profile, config, args[i:]


def has_plugin_config_global(args: List[str]) -> bool:
    """Return whether leading plugin globals contain any ``--config`` form.

    Unlike :func:`split_plugin_globals`, this also recognizes a trailing
    ``--config`` with no value. Local CLI dispatch preserves that malformed
    token for backwards compatibility, but managed dispatch must reject every
    attempt to select a file-backed config.
    """
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--config" or token.startswith("--config="):
            return True
        if token == "--profile":
            if i + 1 >= len(args):
                return False
            i += 2
            continue
        if token.startswith("--profile="):
            i += 1
            continue
        return False
    return False


def load_runtime_context_from_env(*, expected_plugin: Optional[str] = None) -> Optional[PluginRuntimeContext]:
    """Return the runtime context from this process, or ``None`` when absent."""
    value = os.environ.get(RUNTIME_CONTEXT_ENV)
    if value is None:
        return None
    return PluginRuntimeContext.decode(value, expected_plugin=expected_plugin)


def prepare_plugin_invocation(command: str, agent_config: Any) -> Optional[PreparedPluginInvocation]:
    """Prepare a managed ``datus <plugin>`` command or pure pipeline.

    Returns ``None`` for commands with no plugin CLI segment.  Commands that
    contain a plugin invocation but cannot be bridged safely fail closed rather
    than falling back to a local config file.
    """
    segments = split_pipeline(command)
    if segments is None or _has_unsupported_shell_controls(command):
        if _contains_datus_command(command):
            raise PluginRuntimeContextError(
                "Managed plugin commands support a single command or a pure `|` pipeline only"
            )
        return None

    plugin_segments: List[Tuple[int, List[str]]] = []
    for index, segment in enumerate(segments):
        try:
            argv = shlex.split(segment)
        except ValueError as exc:
            if "datus" in segment:
                raise PluginRuntimeContextError(f"Invalid managed plugin command syntax: {exc}") from exc
            return None
        if argv and Path(argv[0]).name == "datus":
            plugin_segments.append((index, argv))
        elif any(Path(token).name == "datus" for token in argv):
            raise PluginRuntimeContextError(
                "Managed plugin commands must invoke `datus` directly at the start of a pipeline segment"
            )

    if not plugin_segments:
        return None
    if len(plugin_segments) != 1:
        raise PluginRuntimeContextError("A managed Bash command may invoke only one plugin CLI")

    segment_index, argv = plugin_segments[0]
    if len(argv) < 2 or argv[1].startswith("-"):
        return None
    plugin_name = argv[1]

    from datus.plugins import store

    if plugin_name in store.RESERVED_PLUGIN_NAMES:
        return None
    if not store.is_valid_name(plugin_name):
        raise PluginRuntimeContextError(f"Invalid plugin name `{plugin_name}`")

    plugin_args = argv[2:]
    profile_name, _config_path, _rest = split_plugin_globals(plugin_args)
    if has_plugin_config_global(plugin_args):
        raise PluginRuntimeContextError(
            "`--config` is unavailable for managed plugin commands; the AuthProvider AgentConfig is authoritative"
        )
    if not getattr(agent_config, "plugins_enabled", True):
        raise PluginRuntimeContextError("Plugins are disabled (`agent.plugins_enabled: false`)")
    if hasattr(agent_config, "plugin_active") and not agent_config.plugin_active(plugin_name):
        raise PluginRuntimeContextError(f"Plugin `{plugin_name}` is not active for this project")

    plugin_path = _resolve_plugin_path(agent_config, plugin_name)
    profile = agent_config.get_plugin_profile(plugin_name, profile_name)
    runtime = PluginRuntimeContext(
        plugin_name=plugin_name,
        profile=profile,
        plugin_path=str(plugin_path) if plugin_path is not None else None,
    )
    encoded = runtime.encode()

    # The payload initially enters Bash through its environment.  The prologue
    # copies it into a randomly-named, non-exported shell variable and unsets
    # the exported name before the pipeline is spawned.  The inline assignment
    # then exposes it only to the datus segment; sibling pipeline commands do
    # not inherit it.
    internal_var = f"__datus_plugin_ctx_{uuid.uuid4().hex}"
    while internal_var in command:
        internal_var = f"__datus_plugin_ctx_{uuid.uuid4().hex}"
    wrapped_segments = list(segments)
    wrapped_segments[segment_index] = f'{RUNTIME_CONTEXT_ENV}="${{{internal_var}}}" {wrapped_segments[segment_index]}'
    wrapped_command = f'{internal_var}="${{{RUNTIME_CONTEXT_ENV}}}"; unset {RUNTIME_CONTEXT_ENV}; ' + " | ".join(
        wrapped_segments
    )
    read_dirs = [str(plugin_path)] if plugin_path is not None else []
    return PreparedPluginInvocation(
        command=wrapped_command,
        env={RUNTIME_CONTEXT_ENV: encoded},
        sandbox_read_dirs=read_dirs,
    )


def _resolve_plugin_path(agent_config: Any, plugin_name: str) -> Optional[Path]:
    """Resolve the selected plugin directory using normal store precedence."""
    from datus.plugins import store
    from datus.plugins.registry import plugin_entry_point_exists

    managed = store.plugin_dir(plugin_name)
    if managed.is_dir() and store.plugin_name_for_dir(managed) == plugin_name:
        return managed.resolve()
    for name, directory in store.iter_extra_plugin_dirs(getattr(agent_config, "plugin_paths", None)):
        if name == plugin_name:
            return directory.resolve()
    if plugin_entry_point_exists(plugin_name):
        # Installed in the current interpreter's site-packages; sys.prefix is
        # already readable in the sandbox and the child uses the same Python.
        return None
    raise PluginRuntimeContextError(
        f"No installed plugin named `{plugin_name}` was found in the managed store, "
        "`agent.plugin_paths`, or the current Python environment"
    )


def _contains_datus_command(command: str) -> bool:
    """Conservative detection used only to prevent unsafe local-config fallback."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>()`")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return "datus" in command
    command_start = True
    for token in tokens:
        if token in {"|", "||", "|&", "&&", ";", "&", "(", ")", "`"}:
            command_start = True
            continue
        if command_start:
            if "=" in token and not token.startswith("="):
                continue
            if Path(token).name == "datus":
                return True
            command_start = False
    # shlex intentionally keeps the contents of double quotes together. Bash
    # still evaluates command substitutions inside them, so conservatively
    # recognize a datus command word in those compound tokens as well.
    for token in tokens:
        if ("$(" in token or "`" in token) and _DATUS_COMMAND_WORD_RE.search(token):
            return True
    return False


def _has_unsupported_shell_controls(command: str) -> bool:
    """Detect top-level shell controls while allowing literals inside quotes.

    A plugin argument such as ``python -c 'a=1; print(a)'`` is safe to retain
    in a pure pipeline, while a top-level ``;``/``&&``/redirection or command
    substitution would change which processes receive the runtime context.
    """
    in_single = in_double = False
    i = 0
    while i < len(command):
        char = command[i]
        if char == "\\" and not in_single:
            i += 2
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if in_single:
            i += 1
            continue
        if char == "`":
            return True
        if char == "$" and i + 1 < len(command) and command[i + 1] in "({":
            return True
        if not in_double and (char in ";&<>()\n"):
            return True
        i += 1
    return False


__all__ = [
    "MAX_RUNTIME_CONTEXT_SIZE",
    "PreparedPluginInvocation",
    "PluginRuntimeContext",
    "PluginRuntimeContextError",
    "RUNTIME_CONTEXT_ENV",
    "has_plugin_config_global",
    "load_runtime_context_from_env",
    "prepare_plugin_invocation",
    "split_plugin_globals",
]
