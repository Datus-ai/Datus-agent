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
variable.  It contains only the invoked plugin's resolved profile, any
manifest-declared one-hop delegate profiles, and the exact plugin directories
selected by the normal managed-store / ``agent.plugin_paths`` precedence.

The Bash command itself is preserved verbatim: :func:`prepare_plugin_invocation`
locates the single ``datus`` command word and inserts an inline environment
assignment in front of it, so redirections, ``&&``/``;`` lists, groupings and
expansions in the model-written command keep working.

The bridge only recognizes ``datus`` when it appears as a real command word.  An
invocation hidden inside a string a wrapper interprets later (``sh -c "datus
..."``) is not rewritten, so that subprocess simply receives no runtime context
and resolves configuration the ordinary way; such wrapper commands never
auto-allow in the permission layer either (see
:mod:`datus.tools.permission.bash_rules`).
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

RUNTIME_CONTEXT_ENV = "DATUS_PLUGIN_RUNTIME_CONTEXT"
RUNTIME_CONTEXT_VERSION = 1
RUNTIME_CONTEXT_PREFIX = "v1."
MAX_RUNTIME_CONTEXT_SIZE = 64 * 1024
_DATUS_COMMAND_WORD_RE = re.compile(r"(?<![A-Za-z0-9_.-])datus(?![A-Za-z0-9_.-])")

# Tokens that may legally precede a command word inside one simple command.
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# `2>&1`, `<&3`, `3>&-` name their target inside the operator, while a bare
# `>`/`2>>`/`<>` takes the FOLLOWING word as its target. Tested in that order
# because _REDIRECT_TARGET_RE would also match `2>&1`.
_REDIRECT_DUP_RE = re.compile(r"^\d*[<>]&(?:\d+-?|-)$")
_REDIRECT_OP_RE = re.compile(r"^\d*(?:>>|>&|<&|<>|>|<)$")
_REDIRECT_TARGET_RE = re.compile(r"^\d*(?:>>|>|<)\S+$")
_COMMAND_PREFIX_WORDS = frozenset({"!", "time", "if", "elif", "while", "until", "then", "do", "else"})

# Characters that end a simple command at the top level of the command line.
_SEPARATOR_CHARS = ";&|\n"
_GROUPING_CHARS = "(){}"

# Bounds recursion while matching nested substitutions so a pathological
# command can never hang the scanner; deeper nesting fails closed.
_MAX_NESTING = 16


class PluginRuntimeContextError(ValueError):
    """Safe, user-facing failure while preparing or decoding runtime context."""


@dataclass(frozen=True)
class PluginRuntimeTarget:
    """One pre-authorized plugin profile available to a composed invocation."""

    profile: Dict[str, Any]
    plugin_path: Optional[str] = None


@dataclass(frozen=True)
class PluginRuntimeContext:
    """Configuration consumed by one ``datus <plugin>`` subprocess."""

    plugin_name: str
    profile: Dict[str, Any]
    plugin_path: Optional[str] = None
    delegates: Dict[str, PluginRuntimeTarget] = field(default_factory=dict)
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
                    "delegates": {
                        name: {
                            "profile": target.profile,
                            "plugin_path": target.plugin_path,
                        }
                        for name, target in self.delegates.items()
                    },
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
        if not isinstance(profile, dict):
            raise PluginRuntimeContextError("Plugin runtime context profile must be an object")
        if plugin_path is not None and (not isinstance(plugin_path, str) or not plugin_path.strip()):
            raise PluginRuntimeContextError("Plugin runtime context path must be a non-empty string")
        raw_delegates = data.get("delegates", {})
        if not isinstance(raw_delegates, dict):
            raise PluginRuntimeContextError("Plugin runtime context delegates must be an object")
        delegates: Dict[str, PluginRuntimeTarget] = {}
        for delegate_name, raw_target in raw_delegates.items():
            if not isinstance(delegate_name, str) or not delegate_name or delegate_name == plugin_name:
                raise PluginRuntimeContextError("Plugin runtime context has an invalid delegate name")
            if not isinstance(raw_target, dict):
                raise PluginRuntimeContextError(f"Plugin runtime context delegate `{delegate_name}` must be an object")
            delegate_profile = raw_target.get("profile")
            delegate_path = raw_target.get("plugin_path")
            if not isinstance(delegate_profile, dict):
                raise PluginRuntimeContextError(
                    f"Plugin runtime context delegate `{delegate_name}` profile must be an object"
                )
            if delegate_path is not None and (not isinstance(delegate_path, str) or not delegate_path.strip()):
                raise PluginRuntimeContextError(
                    f"Plugin runtime context delegate `{delegate_name}` path must be a non-empty string"
                )
            delegates[delegate_name] = PluginRuntimeTarget(delegate_profile, delegate_path)

        context = cls(
            plugin_name=plugin_name,
            profile=profile,
            plugin_path=plugin_path,
            delegates=delegates,
            version=RUNTIME_CONTEXT_VERSION,
        )
        if expected_plugin is None or expected_plugin == plugin_name:
            return context
        target = delegates.get(expected_plugin)
        if target is None:
            raise PluginRuntimeContextError(f"Plugin runtime context is for `{plugin_name}`, not `{expected_plugin}`")
        # A delegate receives only its own resolved profile/path. It cannot
        # reuse sibling delegations or extend the chain transitively.
        return cls(
            plugin_name=expected_plugin,
            profile=target.profile,
            plugin_path=target.plugin_path,
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
    """Return the runtime context from this process, or ``None`` when absent.

    Selecting a delegate narrows the inherited environment value to that
    delegate before plugin code runs.  Any subprocess it starts therefore
    cannot reuse the primary plugin's sibling delegates or extend the chain.
    """
    value = os.environ.get(RUNTIME_CONTEXT_ENV)
    if value is None:
        return None
    context = PluginRuntimeContext.decode(value, expected_plugin=expected_plugin)
    if expected_plugin is not None:
        os.environ[RUNTIME_CONTEXT_ENV] = context.encode()
    return context


def _load_manifest_for_invocation(plugin_name: str, plugin_path: Optional[Path]) -> Any:
    """Read the manifest of the plugin copy this invocation will actually run.

    ``plugin_path`` is the directory the normal store / ``agent.plugin_paths``
    precedence already selected, so the manifest is read from there. Falling
    back to the ``sys.path`` entry-point registry would be wrong in the two
    deployments that matter most: a plugin mounted through
    ``agent.plugin_paths`` (multi-tenant sandboxes) is not on the host
    process's ``sys.path`` at all, and a host that never activated the managed
    store sees a stale, empty registry. Both cases would silently yield "no
    manifest" and drop every declared delegation.

    ``plugin_path is None`` means the plugin lives in this interpreter's
    site-packages, where the entry-point registry is the correct source.
    """
    from datus.plugins import store
    from datus.plugins.registry import load_plugin_manifest

    if plugin_path is None:
        return load_plugin_manifest(plugin_name)
    return store.manifest_for_dir(plugin_path, plugin_name)


def _resolve_profile_delegates(
    agent_config: Any,
    plugin_name: str,
    profile: Dict[str, Any],
    plugin_path: Optional[Path] = None,
) -> Dict[str, PluginRuntimeTarget]:
    """Resolve manifest-declared, one-hop plugin profile references.

    ``x-plugin-profile-ref`` lives on a top-level ``config_schema`` property.
    The property's value in the current profile names the delegated plugin;
    ``profile_field`` optionally names the sibling property containing its
    profile name. ``default_profile: same-name`` falls back to the primary
    profile's injected ``name``. Only these exact references are copied from
    the authoritative AuthProvider AgentConfig.
    """
    from datus.plugins import store

    manifest = _load_manifest_for_invocation(plugin_name, plugin_path)
    if manifest is None:
        # Never silent: without a manifest no delegation can be declared, and
        # the failure would otherwise only surface as the delegate subprocess
        # rejecting its runtime context.
        logger.warning(
            "Plugin `%s` manifest could not be read from %s; any declared plugin profile "
            "references are ignored for this invocation.",
            plugin_name,
            plugin_path or "the current Python environment",
        )
        return {}
    schema = manifest.config_schema
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return {}

    delegates: Dict[str, PluginRuntimeTarget] = {}
    for plugin_field, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            continue
        reference = property_schema.get("x-plugin-profile-ref")
        if reference is None:
            continue
        if not isinstance(reference, dict):
            raise PluginRuntimeContextError(
                f"Plugin `{plugin_name}` config_schema property `{plugin_field}` has an invalid "
                "x-plugin-profile-ref declaration"
            )

        raw_delegate_name = profile.get(plugin_field)
        if raw_delegate_name is None or str(raw_delegate_name).strip() == "":
            continue
        delegate_name = str(raw_delegate_name).strip()
        if not store.is_valid_name(delegate_name) or delegate_name in store.RESERVED_PLUGIN_NAMES:
            raise PluginRuntimeContextError(
                f"Plugin `{plugin_name}` profile field `{plugin_field}` names invalid plugin `{delegate_name}`"
            )
        if delegate_name == plugin_name:
            raise PluginRuntimeContextError(f"Plugin `{plugin_name}` cannot delegate its runtime profile to itself")
        if hasattr(agent_config, "plugin_active") and not agent_config.plugin_active(delegate_name):
            raise PluginRuntimeContextError(f"Delegated plugin `{delegate_name}` is not active for this project")

        profile_field = reference.get("profile_field")
        if profile_field is not None and (not isinstance(profile_field, str) or not profile_field.strip()):
            raise PluginRuntimeContextError(
                f"Plugin `{plugin_name}` profile reference for `{plugin_field}` has an invalid profile_field"
            )
        profile_field = profile_field.strip() if isinstance(profile_field, str) else None
        default_profile = reference.get("default_profile", "plugin-default")
        if default_profile not in {"plugin-default", "same-name"}:
            raise PluginRuntimeContextError(
                f"Plugin `{plugin_name}` profile reference for `{plugin_field}` has unsupported "
                f"default_profile {default_profile!r}"
            )

        delegate_profile_name: Optional[str] = None
        if profile_field:
            raw_profile_name = profile.get(profile_field)
            if raw_profile_name is not None and str(raw_profile_name).strip():
                delegate_profile_name = str(raw_profile_name).strip()
        if delegate_profile_name is None and default_profile == "same-name":
            raw_primary_name = profile.get("name")
            if raw_primary_name is None or not str(raw_primary_name).strip():
                raise PluginRuntimeContextError(
                    f"Plugin `{plugin_name}` profile needs a name to resolve same-name delegate `{delegate_name}`"
                )
            delegate_profile_name = str(raw_primary_name).strip()

        delegate_profile = agent_config.get_plugin_profile(delegate_name, delegate_profile_name)
        delegate_path = _resolve_plugin_path(agent_config, delegate_name)
        target = PluginRuntimeTarget(
            profile=dict(delegate_profile),
            plugin_path=str(delegate_path) if delegate_path is not None else None,
        )
        previous = delegates.get(delegate_name)
        if previous is not None and previous != target:
            raise PluginRuntimeContextError(
                f"Plugin `{plugin_name}` resolves conflicting profiles for delegate `{delegate_name}`"
            )
        delegates[delegate_name] = target
    return delegates


def prepare_plugin_invocation(command: str, agent_config: Any) -> Optional[PreparedPluginInvocation]:
    """Prepare a managed ``datus <plugin>`` command.

    Any shell command shape is accepted as long as the single plugin CLI
    invocation sits at a top-level command position: pipelines, redirections,
    ``;``/``&&``/``||``/newline lists, ``(...)``/``{...}`` groupings, heredocs
    and expansions all keep their original text.

    Returns ``None`` for commands with no plugin CLI segment.  Commands that
    contain a plugin invocation but cannot be bridged safely fail closed rather
    than falling back to a local config file.
    """
    scan = _scan_command(command)
    if scan.error is not None:
        # The command word cannot be located, so be maximally conservative: any
        # `datus` word at all fails the command rather than letting it run
        # unbridged (which would fall back to local config resolution). Token
        # position analysis is unreliable here precisely because the parse failed.
        if _DATUS_COMMAND_WORD_RE.search(command):
            raise PluginRuntimeContextError(f"Invalid managed plugin command syntax: {scan.error}")
        return None
    for substitution in scan.substitutions:
        if _contains_datus_command(substitution):
            raise PluginRuntimeContextError(
                "Managed plugin commands cannot invoke `datus` inside a command substitution "
                "(`$(...)`, backticks); run the plugin command directly and pipe its output instead"
            )

    plugin_invocations: List[Tuple[int, List[str]]] = []
    for span in scan.spans:
        index = _command_word_index(span.words)
        if index is None:
            continue
        word_offset = span.words[index][0]
        try:
            argv = shlex.split(command[word_offset : span.end])
        except ValueError as exc:
            if "datus" in command[span.start : span.end]:
                raise PluginRuntimeContextError(f"Invalid managed plugin command syntax: {exc}") from exc
            continue
        if argv and Path(argv[0]).name == "datus":
            plugin_invocations.append((word_offset, argv))
        elif any(Path(token).name == "datus" for token in argv):
            raise PluginRuntimeContextError(
                "Managed plugin commands must invoke `datus` directly as the command word, "
                "not through a wrapper such as `timeout`, `env`, `xargs` or `sh -c`"
            )

    if not plugin_invocations:
        return None
    if len(plugin_invocations) != 1:
        raise PluginRuntimeContextError("A managed Bash command may invoke only one plugin CLI")

    insert_at, argv = plugin_invocations[0]
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
    delegates = _resolve_profile_delegates(agent_config, plugin_name, profile, plugin_path)
    runtime = PluginRuntimeContext(
        plugin_name=plugin_name,
        profile=profile,
        plugin_path=str(plugin_path) if plugin_path is not None else None,
        delegates=delegates,
    )
    encoded = runtime.encode()

    # The payload initially enters Bash through its environment.  The prologue
    # copies it into a randomly-named, non-exported shell variable and unsets
    # the exported name before any command in the line is spawned.  The inline
    # assignment then exports it only for the datus command; sibling commands
    # in the same line do not inherit it.
    internal_var = f"__datus_plugin_ctx_{uuid.uuid4().hex}"
    while internal_var in command:
        internal_var = f"__datus_plugin_ctx_{uuid.uuid4().hex}"
    wrapped_command = (
        f'{internal_var}="${{{RUNTIME_CONTEXT_ENV}}}"; unset {RUNTIME_CONTEXT_ENV}; '
        + command[:insert_at]
        + f'{RUNTIME_CONTEXT_ENV}="${{{internal_var}}}" '
        + command[insert_at:]
    )
    read_dirs = [str(plugin_path)] if plugin_path is not None else []
    read_dirs.extend(
        target.plugin_path
        for target in delegates.values()
        if target.plugin_path is not None and target.plugin_path not in read_dirs
    )
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


def _command_word_index(words: List[Tuple[int, str]]) -> Optional[int]:
    """Index of the command word among one simple command's raw words.

    Variable assignments (``FOO=1 datus ...``), redirections placed before the
    command word (``> out.txt datus ...``, ``2>&1 datus ...``) and shell
    keywords that introduce a command (``do``, ``then``, ``!``, ``time``) are
    skipped. Returns ``None`` when the command consists of prefix words only.
    """
    i = 0
    while i < len(words):
        text = words[i][1]
        if text in _COMMAND_PREFIX_WORDS or _ASSIGNMENT_RE.match(text):
            i += 1
            continue
        if _REDIRECT_DUP_RE.match(text):
            # Self-contained fd duplication: `2>&1`, `<&3`, `3>&-`.
            i += 1
            continue
        if _REDIRECT_OP_RE.match(text):
            # A bare operator takes the next word as its target. Checked before
            # _REDIRECT_TARGET_RE, whose `\S+` would swallow `>>` itself.
            i += 2
            continue
        if _REDIRECT_TARGET_RE.match(text):
            # Target attached to the operator: `>out.txt`.
            i += 1
            continue
        return i
    return None


@dataclass(frozen=True)
class _CommandSpan:
    """One top-level simple command: its offsets and its raw words."""

    start: int
    end: int
    words: List[Tuple[int, str]]


@dataclass(frozen=True)
class _ShellScan:
    """Structure of one Bash command line, as far as this bridge needs it.

    ``spans`` describes the top-level simple commands; ``substitutions`` holds
    the raw text of every command substitution and ``${...}`` expansion found,
    so a ``datus`` word hiding in one can be rejected; ``error`` is set when the
    line cannot be parsed at all.
    """

    spans: List[_CommandSpan] = field(default_factory=list)
    substitutions: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _scan_command(command: str) -> _ShellScan:
    """Split a command line into its top-level simple-command spans.

    Only enough Bash syntax is modelled to know *where* a command word may
    start: quoting, expansions, command substitutions, heredoc bodies and
    comments are skipped, and separators (``;``, ``&``, ``|``, newline) plus
    groupings (``(``, ``)``, ``{``, ``}``) end the current span. Redirections
    stay inside the span they belong to.
    """
    spans: List[_CommandSpan] = []
    substitutions: List[str] = []
    heredocs: List[Tuple[str, bool]] = []
    span_start: Optional[int] = None
    words: List[Tuple[int, str]] = []
    i = 0
    n = len(command)

    def close_span(end: int) -> None:
        nonlocal span_start, words
        if span_start is not None:
            if words:
                spans.append(_CommandSpan(span_start, end, words))
            span_start = None
            words = []

    while i < n:
        char = command[i]
        if char in " \t\r":
            i += 1
            continue
        if char == "\n":
            close_span(i)
            i += 1
            if heredocs:
                i = _skip_heredoc_bodies(command, i, heredocs)
                heredocs = []
            continue
        if char in _SEPARATOR_CHARS or char in _GROUPING_CHARS:
            close_span(i)
            i += 1
            continue
        if char == "#":
            # An unquoted `#` starting a word begins a comment.
            close_span(i)
            newline = command.find("\n", i)
            i = n if newline == -1 else newline
            continue
        if span_start is None:
            span_start = i
        word_start = i
        i, error = _scan_word(command, i, substitutions, heredocs)
        if error is not None:
            return _ShellScan(spans, substitutions, error)
        words.append((word_start, command[word_start:i]))
    close_span(n)
    return _ShellScan(spans, substitutions, None)


def _scan_word(
    command: str,
    i: int,
    substitutions: List[str],
    heredocs: List[Tuple[str, bool]],
) -> Tuple[int, Optional[str]]:
    """Advance past one word, returning the offset after it.

    Braces are literal here: ``--opt={a,b}`` is one word to Bash, and only a
    brace at a word boundary (handled by :func:`_scan_command`) groups commands.
    """
    n = len(command)
    while i < n:
        char = command[i]
        if char == "\\":
            i += 2
            continue
        if char in " \t\r" or char in _SEPARATOR_CHARS or char in "()":
            return i, None
        if char == "'":
            end = command.find("'", i + 1)
            if end == -1:
                return n, "unbalanced single quote"
            i = end + 1
            continue
        if char == '"':
            i, error = _skip_double_quoted(command, i, substitutions=substitutions)
            if error is not None:
                return n, error
            continue
        if char == "`":
            end = _find_backtick_end(command, i + 1)
            if end is None:
                return n, "unbalanced backtick"
            substitutions.append(command[i + 1 : end])
            i = end + 1
            continue
        if char == "$" and i + 1 < n and command[i + 1] == "(":
            i, inner, error = _scan_substitution(command, i + 2)
            if error is not None:
                return n, error
            substitutions.append(inner)
            continue
        if char == "$" and i + 1 < n and command[i + 1] == "{":
            i, inner, error = _skip_brace_expansion(command, i + 2)
            if error is not None:
                return n, error
            substitutions.append(inner)
            continue
        if char == "<" and command.startswith("<<<", i):
            # A here-string takes its payload inline — consuming all three
            # characters keeps the trailing `<<` from re-matching as a heredoc,
            # whose body skip would swallow the rest of the command.
            i += 3
            continue
        if char == "<" and command.startswith("<<", i):
            i, error = _consume_heredoc_delimiter(command, i + 2, heredocs)
            if error is not None:
                return n, error
            continue
        if char in "<>" and command.startswith(f"{char}&", i):
            # `2>&1` / `<&3` keep the `&` out of separator handling.
            i += 2
            continue
        i += 1
    return n, None


def _skip_double_quoted(
    command: str,
    i: int,
    nesting: int = 0,
    substitutions: Optional[List[str]] = None,
) -> Tuple[int, Optional[str]]:
    """Advance past a double-quoted string starting at ``i``."""
    n = len(command)
    j = i + 1
    while j < n:
        char = command[j]
        if char == "\\":
            j += 2
            continue
        if char == '"':
            return j + 1, None
        if char == "`":
            end = _find_backtick_end(command, j + 1)
            if end is None:
                return n, "unbalanced backtick"
            if substitutions is not None:
                substitutions.append(command[j + 1 : end])
            j = end + 1
            continue
        if char == "$" and j + 1 < n and command[j + 1] == "(":
            j, inner, error = _scan_substitution(command, j + 2, nesting + 1)
            if error is not None:
                return n, error
            if substitutions is not None:
                substitutions.append(inner)
            continue
        if char == "$" and j + 1 < n and command[j + 1] == "{":
            j, inner, error = _skip_brace_expansion(command, j + 2)
            if error is not None:
                return n, error
            if substitutions is not None:
                substitutions.append(inner)
            continue
        j += 1
    return n, "unbalanced double quote"


def _scan_substitution(command: str, i: int, nesting: int = 0) -> Tuple[int, str, Optional[str]]:
    """Match a ``$(...)`` substitution whose body starts at ``i``.

    Returns the offset after the closing paren and the raw body text.
    """
    if nesting > _MAX_NESTING:
        return len(command), command[i:], "command substitution nested too deeply"
    n = len(command)
    start = i
    parens = 0
    j = i
    while j < n:
        char = command[j]
        if char == "\\":
            j += 2
            continue
        if char == "'":
            end = command.find("'", j + 1)
            if end == -1:
                return n, command[start:], "unbalanced single quote"
            j = end + 1
            continue
        if char == '"':
            j, error = _skip_double_quoted(command, j, nesting + 1)
            if error is not None:
                return n, command[start:], error
            continue
        if char == "`":
            end = _find_backtick_end(command, j + 1)
            if end is None:
                return n, command[start:], "unbalanced backtick"
            j = end + 1
            continue
        if char == "$" and j + 1 < n and command[j + 1] == "(":
            j, _inner, error = _scan_substitution(command, j + 2, nesting + 1)
            if error is not None:
                return n, command[start:], error
            continue
        if char == "(":
            parens += 1
            j += 1
            continue
        if char == ")":
            if parens:
                parens -= 1
                j += 1
                continue
            return j + 1, command[start:j], None
        j += 1
    return n, command[start:], "unterminated command substitution"


def _skip_brace_expansion(command: str, i: int) -> Tuple[int, str, Optional[str]]:
    """Match a ``${...}`` expansion whose body starts at ``i``.

    The body is returned so a command substitution smuggled into a default
    value (``${x:-$(datus ...)}``) is still inspected.
    """
    n = len(command)
    start = i
    depth = 1
    j = i
    while j < n:
        char = command[j]
        if char == "\\":
            j += 2
            continue
        if char == "$" and j + 1 < n and command[j + 1] == "{":
            # Only a nested expansion opens a level — a bare `{` in a default
            # value (``${A:-{x\}}``) is literal text to Bash.
            depth += 1
            j += 2
            continue
        if char == "}":
            depth -= 1
            j += 1
            if depth == 0:
                return j, command[start : j - 1], None
            continue
        j += 1
    return n, command[start:], "unbalanced ${...} expansion"


def _find_backtick_end(command: str, i: int) -> Optional[int]:
    """Offset of the backtick closing a substitution opened before ``i``."""
    n = len(command)
    while i < n:
        if command[i] == "\\":
            i += 2
            continue
        if command[i] == "`":
            return i
        i += 1
    return None


def _consume_heredoc_delimiter(
    command: str,
    i: int,
    heredocs: List[Tuple[str, bool]],
) -> Tuple[int, Optional[str]]:
    """Read the delimiter word of a ``<<``/``<<-`` heredoc starting at ``i``."""
    n = len(command)
    strip_tabs = False
    if i < n and command[i] == "-":
        strip_tabs = True
        i += 1
    while i < n and command[i] in " \t":
        i += 1
    parts: List[str] = []
    while i < n:
        char = command[i]
        if char == "'":
            end = command.find("'", i + 1)
            if end == -1:
                return n, "unbalanced single quote"
            parts.append(command[i + 1 : end])
            i = end + 1
            continue
        if char == '"':
            end = command.find('"', i + 1)
            if end == -1:
                return n, "unbalanced double quote"
            parts.append(command[i + 1 : end])
            i = end + 1
            continue
        if char in " \t\r\n<>" or char in _SEPARATOR_CHARS or char in _GROUPING_CHARS:
            break
        parts.append(char)
        i += 1
    delimiter = "".join(parts)
    if delimiter:
        heredocs.append((delimiter, strip_tabs))
    return i, None


def _skip_heredoc_bodies(command: str, i: int, heredocs: List[Tuple[str, bool]]) -> int:
    """Advance past the bodies of the heredocs opened on the previous line.

    Body text is never a command position, so a ``datus`` word inside a payload
    must not be mistaken for a second plugin invocation. An unterminated body
    consumes the remainder of the line, which is what Bash does too.
    """
    n = len(command)
    for delimiter, strip_tabs in heredocs:
        while i < n:
            line_end = command.find("\n", i)
            line = command[i:] if line_end == -1 else command[i:line_end]
            i = n if line_end == -1 else line_end + 1
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate.rstrip("\r") == delimiter:
                break
    return i


__all__ = [
    "MAX_RUNTIME_CONTEXT_SIZE",
    "PreparedPluginInvocation",
    "PluginRuntimeContext",
    "PluginRuntimeContextError",
    "PluginRuntimeTarget",
    "RUNTIME_CONTEXT_ENV",
    "has_plugin_config_global",
    "load_runtime_context_from_env",
    "prepare_plugin_invocation",
    "split_plugin_globals",
]
