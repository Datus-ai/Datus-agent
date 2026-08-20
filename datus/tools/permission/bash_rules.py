# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Fine-grained bash command permission rules.

Evaluates a bash command string against allow/deny/ask pattern lists and
returns a decision for ``PermissionHooks._handle_bash_permission``. This is
the command-level refinement of the coarse ``bash_tools.bash -> ASK`` rule:
instead of prompting for every command, users (and profiles) can express
"``git log`` is always fine, ``rm`` is never fine, ``docker`` should ask".

Pattern syntax (Claude-Code-style, extended to multi-word prefixes — the
legacy ``BashTool._matches_pattern`` only fnmatches ``argv[0]`` and cannot
express ``git status``):

- ``"git status"`` (no colon)  -> exact match: argv equals the pattern tokens.
- ``"git log:*"``              -> prefix match: argv starts with the prefix
  tokens; anything (including nothing) may follow.
- ``"python:scripts/*.py"``    -> prefix match AND the first token after the
  prefix (or the joined remainder) fnmatches the glob. Only the first
  positional argument is matched to prevent smuggling a disallowed flag in
  front (mirrors the rationale in ``BashTool._matches_pattern``).

Sub-commands: a command chained with top-level, unquoted ``|``, ``&&``,
``||`` or ``;`` is split into sub-commands and each is judged independently,
then the results are aggregated (any DENY -> DENY; any safety-forced -> ASK;
all ALLOW -> ALLOW; otherwise ASK). This lets a chain of allow-listed
read-only commands (``cat log | grep err | wc -l``, ``git fetch && git
status``) auto-run, and — just as importantly — lets the confirmation prompt
name EVERY sub-command that needs approval instead of one representative.
Shell constructs that segmentation cannot model still hit the safety ceiling
as a whole: ``$()``, backticks, ``${}``, redirection, background ``&``,
``|&``, newlines, subshells/braces and compound keywords (``for``/``if``/…).
Two redirections are exempt because they name no path and so cannot do
anything the argv could not: fd duplication (``2>&1``, ``>&2``, ``<&0``) and a
``/dev/null`` target (see ``_BENIGN_REDIRECTION_RE``).

The safety ceiling is a *static* verdict, not a final one: under a profile with
``permissions.auto_review`` enabled the shared reviewer in ``auto_reviewer.py``
sees the full command text and may still resolve such an ASK. Only unparseable
commands are kept behind direct user confirmation.

Per-segment decision order (deny-bypass-resistant, mirroring Claude Code's
asymmetry of aggressive deny / conservative allow):

1. unparseable / empty        -> ASK (safety_forced)
2. deny rules                 -> DENY (anchored AND unanchored, so a deny on
                                 ``rm:*`` also catches ``xargs rm -rf x`` and a
                                 pipeline segment ``... | xargs rm``)
3. safety ceiling             -> ASK (safety_forced) for shell wrappers
                                 (``bash -c`` re-introduces a shell that the
                                 outer argv match is blind to), interpreter
                                 inline-code flags (``python -c`` / ``perl -e``
                                 execute a string the rules cannot see),
                                 compound-construct fragments (``for``, ``do``,
                                 ``(``, ``{`` — segmentation does not model
                                 them), and residual shell metacharacters;
                                 allow rules cannot override
4. ask rules                  -> ASK (anchored)
5. allow rules                -> ALLOW (anchored only, never unanchored)
6. rules.default              -> usually ASK

The legacy ``classifier`` config block is retained as a deprecated bash-only
override for the shared automatic reviewer settings.
"""

import fnmatch
import re
import shlex
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from datus.tools.permission.permission_config import PermissionLevel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Shell metacharacters (other than a plain top-level ``|``) that change
# execution semantics. BashTool executes commands through a real shell, so
# these ARE interpreted — but per-segment rule matching cannot reason about
# command substitution, chaining, or redirection, so commands containing them
# never auto-allow (they hit the safety ceiling → confirmation).
#
# ``|`` is deliberately excluded here: chained commands are split into
# sub-commands by ``split_command_chain`` before this check runs, so each
# segment is judged on its own and chains of allow-listed read-only commands
# can auto-run. The regex still lists ``;``/``&``/``&&``/``||`` because a
# segment reaching this check kept them (an un-splittable form such as
# background ``&``), and because ``BashTool._segment_allowed`` shares this
# function with only ``split_pipeline`` in front of it.
_SHELL_METACHARS_RE = re.compile(r"[;&<>`\n]|\$\(|\$\{|\|\||&&")

# Redirections that cannot express anything the argv already can't, and so are
# exempt from the redirection safety ceiling **in the permission layer only**:
#
#   * file-descriptor duplication / closing — ``2>&1``, ``>&2``, ``<&0``,
#     ``2>&-``. These rewire the command's own streams. No path is named, so
#     nothing can be written anywhere, and no new command can be introduced.
#   * a ``/dev/null`` target — ``>/dev/null``, ``2>>/dev/null``,
#     ``&>/dev/null``. The write goes to the null device by definition.
#
# ``2>&1`` and ``2>/dev/null`` are among the most common shell idioms there
# are; forcing a confirmation prompt for them trains users to click through
# prompts, which costs more safety than the ceiling buys. Every other
# redirection stays on the ceiling: ``> file`` names an arbitrary path, so an
# allow rule matched on argv (``ls:*`` says nothing about where output lands)
# cannot vouch for it.
#
# Scope is deliberate. This is applied by ``_evaluate_single_command`` and
# nowhere else: ``contains_shell_metachars`` itself is unchanged because
# ``BashTool._segment_allowed`` shares it as the execution-layer whitelist for
# skills — a separate trust boundary that must keep rejecting every
# redirection outright.
_BENIGN_REDIRECTION_RE = re.compile(
    # 2>&1, >&2, <&0, 2>&-
    r"\d*[<>]&(?:\d+|-)"
    # >/dev/null, 2>/dev/null, >>/dev/null, &>/dev/null, &>>/dev/null.
    # The target must end the token: without the lookahead, ``> /dev/nullx``
    # would match its ``/dev/null`` prefix and leave a stray ``x`` behind,
    # exempting a write to a real file.
    r"|(?:\d*>{1,2}|&>{1,2})\s*/dev/null(?=\s|$)"
)


def _without_benign_redirections(text: str) -> str:
    """Blank out the redirections listed in ``_BENIGN_REDIRECTION_RE``.

    Substitutes a space rather than an empty string so stripping cannot fuse
    two tokens into one (``a 2>&1b`` must not become ``a b``).
    """
    return _BENIGN_REDIRECTION_RE.sub(" ", text)


def _last_nonspace(chars: List[str]) -> str:
    """Last non-whitespace character accumulated so far, or ``""``."""
    for ch in reversed(chars):
        if not ch.isspace():
            return ch
    return ""


# Shell keywords and grouping constructs. Segmentation splits on operators but
# does not model compound commands, so ``for f in *; do rm $f; done`` arrives
# here as three fragments (``for f in *``, ``do rm $f``, ``done``) that must
# never be approved as three ordinary commands. Any fragment that opens or
# belongs to such a construct hits the safety ceiling, which forces the WHOLE
# chain to a confirmation prompt with no project-persistence option.
_SHELL_KEYWORDS = frozenset(
    {
        "for",
        "while",
        "until",
        "if",
        "then",
        "elif",
        "else",
        "fi",
        "do",
        "done",
        "case",
        "esac",
        "in",
        "select",
        "function",
        "coproc",
    }
)


def contains_shell_metachars(text: str) -> bool:
    """True when ``text`` contains any non-pipe shell metacharacter.

    Checks the RAW string (quoting is ignored on purpose): neither the
    permission safety ceiling nor a restrictive execution-layer whitelist can
    reason about chaining / substitution / redirection, so a quoted
    metacharacter is treated just as conservatively as an unquoted one.
    Shared by ``evaluate_bash_command`` (→ forced ASK) and
    ``BashTool._segment_allowed`` (→ reject).

    Deliberately knows nothing about ``_BENIGN_REDIRECTION_RE``: the
    permission layer strips those forms from its own input before calling this,
    while the execution-layer whitelist keeps rejecting every redirection. Add
    the exemption at the call site, never here.
    """
    return bool(_SHELL_METACHARS_RE.search(text))


# Commands that execute their arguments (or arbitrary strings) as new
# commands. An allow rule matched against the OUTER argv says nothing about
# the wrapped command, so these never auto-allow either.
_WRAPPER_COMMANDS = frozenset(
    {
        "bash",
        "sh",
        "zsh",
        "dash",
        "ksh",
        "fish",
        "csh",
        "tcsh",
        "xargs",
        "env",
        "sudo",
        "doas",
        "nohup",
        "eval",
        "exec",
        "command",
        "time",
        "timeout",
        "watch",
        "script",
    }
)

# Script interpreters are NOT blanket wrappers: an allow rule can
# meaningfully match the script path (the documented ``python:scripts/*.py``
# form), so listing them in ``_WRAPPER_COMMANDS`` would make every such rule
# dead (the safety ceiling runs before allow matching). But their inline-code
# flags (``python -c``, ``perl -e``, ``node --eval`` …) execute an arbitrary
# string the rules cannot see — exactly the wrapper problem — so those forms
# hit the safety ceiling instead of ever auto-allowing. Values are
# (short option letters, long flags); short letters are matched inside
# combined clusters too (``-uc`` contains ``c``). Options precede the
# script/program path for all of these interpreters, so scanning stops at the
# first non-option token to avoid false ASKs on script-owned arguments.
_INTERPRETER_INLINE_CODE_FLAGS: Dict[str, tuple] = {
    "python": (frozenset("c"), frozenset()),
    "perl": (frozenset("eE"), frozenset()),
    "ruby": (frozenset("e"), frozenset()),
    "node": (frozenset("ep"), frozenset({"--eval", "--print"})),
    "nodejs": (frozenset("ep"), frozenset({"--eval", "--print"})),
    "php": (frozenset("r"), frozenset()),
}

# ``python``, ``python3``, ``python3.12`` … all resolve to the python spec.
_PYTHON_VERSIONED_RE = re.compile(r"^python\d*(\.\d+)*$")


def _has_inline_code_flag(argv: List[str]) -> bool:
    """True when a known interpreter is invoked with an inline-code flag."""
    spec = _INTERPRETER_INLINE_CODE_FLAGS.get(argv[0])
    if spec is None and _PYTHON_VERSIONED_RE.match(argv[0]):
        spec = _INTERPRETER_INLINE_CODE_FLAGS["python"]
    if spec is None:
        return False
    short_letters, long_flags = spec
    for token in argv[1:]:
        if not token.startswith("-") or token == "-":
            break
        if token.startswith("--"):
            if token.split("=", 1)[0] in long_flags:
                return True
            continue
        if any(letter in token[1:] for letter in short_letters):
            return True
    return False


def _is_compound_construct(command: str, argv: List[str]) -> bool:
    """True when a segment belongs to a compound shell construct.

    ``split_command_chain`` splits on control operators but does not model
    loops, conditionals, subshells or brace groups, so ``for f in *; do rm $f;
    done`` arrives as three fragments. Treating those as three ordinary
    commands would let ``do rm $f`` be approved (and persisted) as ``rm:*``.
    Detected by a leading shell keyword or an unbalanced grouping character,
    both of which force the whole chain to the safety ceiling.
    """
    if argv and argv[0] in _SHELL_KEYWORDS:
        return True
    stripped = command.strip()
    return bool(stripped) and (stripped[0] in "({" or stripped[-1] in ")}")


# Multi-command CLIs whose first subcommand carries the semantics; session
# buckets use the first two tokens so approving ``git log`` never covers
# ``git push``.
_GROUP_COMMANDS = frozenset(
    {
        # datus itself hosts plugin CLIs (``datus <plugin> ...``); approving
        # one plugin's namespace must not green-light the others.
        "datus",
        "git",
        "docker",
        "npm",
        "pnpm",
        "yarn",
        "uv",
        "pip",
        "pip3",
        "cargo",
        "go",
        "kubectl",
        "make",
        "poetry",
        "conda",
    }
)


def _normalize_datus_plugin_argv(argv: List[str]) -> List[str]:
    """Drop leading datus-global ``--profile`` flags from ``datus <plugin> ...``.

    The plugin dispatcher (``datus/cli/main.py`` ``_split_plugin_globals``)
    consumes ``--profile <p>`` / ``--profile=<p>`` appearing between the
    plugin name and the first command token, so ``datus hello --profile prod
    config set x`` runs the same subcommand as ``datus hello config set x``.
    Rule matching must see through that, or every profile-qualified
    invocation falls out of the plugin's declared allow/ask/deny patterns.

    Deliberately NOT stripped:

    * ``--config <path>`` — pointing datus at a different agent.yml rebinds
      the plugin's credentials/endpoints; a grant persisted for one config
      must not silently cover another. Commands carrying it fall through to
      the default decision (ask).
    * ``--profile`` in sub-command position (``datus hello greet --profile
      x``) — from the first command token onward every flag belongs to the
      plugin, mirroring the dispatcher.

    Tradeoff: because ask/allow matching sees through ``--profile``, an allow
    rule or project grant approved under one profile also covers the others —
    profiles of one plugin share a trust domain by design (unlike ``--config``,
    which swaps the whole config file). Users who need to fence off a specific
    profile can write a deny rule mentioning it: deny rules are additionally
    matched against the RAW argv (see ``_evaluate_single_command``).
    """
    if len(argv) < 3 or argv[0] != "datus" or argv[1].startswith("-"):
        return argv
    i = 2
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok == "--profile":
            if i + 1 >= n:
                break
            i += 2
            continue
        if tok.startswith("--profile="):
            i += 1
            continue
        break
    if i == 2:
        return argv
    return argv[:2] + argv[i:]


class BashClassifierConfig(BaseModel):
    """Deprecated bash-only configuration alias for the shared AI reviewer.

    ``PermissionHooks`` maps explicitly configured fields onto
    ``permissions.auto_review`` for bash actions. The separate classifier
    interface remains only as a test/embedder compatibility seam.
    """

    enabled: bool = Field(default=False, description="Enable AI review for bash ASK actions")
    model: Optional[str] = Field(default=None, description="Reviewer model reference")
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for a verdict to act",
    )


class BashCommandRules(BaseModel):
    """Command-level allow/deny/ask rules for the bash tool.

    Configured under ``agent.permissions.bash_commands`` in agent.yml (see
    ``conf/agent.yml.example``) and attached to profiles in ``profiles.py``.
    """

    allow: List[str] = Field(default_factory=list, description="Patterns that auto-allow")
    deny: List[str] = Field(default_factory=list, description="Patterns that block (highest priority)")
    ask: List[str] = Field(default_factory=list, description="Patterns that force a confirmation")
    default: PermissionLevel = Field(
        default=PermissionLevel.ASK, description="Decision when no rule matches (ask unless overridden)"
    )
    classifier: BashClassifierConfig = Field(default_factory=BashClassifierConfig)

    model_config = ConfigDict(use_enum_values=True)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["BashCommandRules"]:
        """Create from the agent.yml ``bash_commands`` dict; None when absent.

        Validation errors propagate so ``AgentConfig._init_permissions_config``
        can apply its existing fail-closed fallback (revert to normal profile).
        """
        if not data:
            return None
        # Only pass ``default`` / ``classifier`` when the YAML actually set
        # them, so ``model_fields_set`` faithfully records explicitness — the
        # hook's profile-posture fallback and ``merge_with`` both rely on it.
        kwargs: Dict[str, Any] = {
            "allow": data.get("allow", []),
            "deny": data.get("deny", []),
            "ask": data.get("ask", []),
        }
        if "default" in data:
            kwargs["default"] = PermissionLevel(data["default"])
        if data.get("classifier") is not None:
            kwargs["classifier"] = BashClassifierConfig(**data["classifier"])
        return cls(**kwargs)

    def merge_with(self, override: Optional["BashCommandRules"]) -> "BashCommandRules":
        """Layer another ruleset on top: lists concatenate, scalars override.

        ``default`` and ``classifier`` are replaced only when the override set
        them explicitly (``model_fields_set``), so a node-level override that
        just adds an allow pattern does not silently reset the default. When
        NEITHER side set them, the merged result keeps them unset too — the
        permission hook uses that to fall back to the profile's
        ``default_permission`` (e.g. ALLOW under dangerous).
        """
        if override is None:
            return self
        kwargs: Dict[str, Any] = {
            "allow": self.allow + override.allow,
            "deny": self.deny + override.deny,
            "ask": self.ask + override.ask,
        }
        if "default" in override.model_fields_set:
            kwargs["default"] = PermissionLevel(override.default)
        elif "default" in self.model_fields_set:
            kwargs["default"] = PermissionLevel(self.default)
        if "classifier" in override.model_fields_set:
            kwargs["classifier"] = override.classifier
        elif "classifier" in self.model_fields_set:
            kwargs["classifier"] = self.classifier
        return BashCommandRules(**kwargs)

    def is_empty(self) -> bool:
        """True when no patterns are configured (fall back to the coarse rule)."""
        return not (self.allow or self.deny or self.ask)


class BashDecisionSource(str, Enum):
    """Why ``evaluate_bash_command`` decided the way it did."""

    DENY_RULE = "deny_rule"
    ASK_RULE = "ask_rule"
    ALLOW_RULE = "allow_rule"
    SAFETY = "safety"
    DEFAULT = "default"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class BashSegmentDecision:
    """One sub-command's contribution to an ASK decision.

    Carries everything the confirmation prompt and the grant writers need:
    the segment text (so the prompt can name each sub-command), its own
    session bucket, and its own rule provenance.
    """

    command: str
    level: PermissionLevel
    source: BashDecisionSource
    matched_pattern: Optional[str]
    reason: str
    bucket: str
    safety_forced: bool = False


@dataclass(frozen=True)
class BashRuleDecision:
    """Result of evaluating one bash command against a ruleset.

    Attributes:
        level: The permission decision (real enum, not the pydantic string).
        source: Which stage of the decision order produced it.
        matched_pattern: The rule pattern that fired, when source is a rule.
        reason: Human-readable explanation for prompts and logs.
        bucket: Session-approval bucket key of the REPRESENTATIVE sub-command
            (e.g. ``git push`` or an ask-rule pattern). Kept for logging and
            for the hook's profile-default fallback; grants are scoped per
            ``ask_segments`` entry, not to this single bucket.
        safety_forced: True when the ASK came from the safety ceiling or an
            unparseable command. Such decisions are never auto-resolved by the
            deprecated bash-only classifier seam, and are never offered as
            persistent project-level allows. The shared structured reviewer in
            ``auto_reviewer.py`` MAY still resolve them, because it inspects the
            complete command text that the argv-level rules cannot see (a
            wrapper, metacharacter, or ``$()`` is judged on its actual effect).
            Genuinely unparseable commands stay behind direct user
            confirmation — the reviewer is never consulted for them.
        ask_segments: For an ASK decision, EVERY non-allow sub-command in
            source order (a one-element tuple for a plain single command).
            The prompt lists them all and the session/project grants cover
            them all — approving one representative must never green-light
            unreviewed sub-commands. Empty for ALLOW/DENY decisions.
    """

    level: PermissionLevel
    source: BashDecisionSource
    matched_pattern: Optional[str]
    reason: str
    bucket: str
    safety_forced: bool = False
    ask_segments: Tuple[BashSegmentDecision, ...] = ()


def _split_pattern(pattern: str) -> tuple[List[str], Optional[str]]:
    """Split a rule pattern into prefix tokens and an optional glob.

    ``"git log:*"`` -> (["git", "log"], "*"); ``"git status"`` -> (["git",
    "status"], None) meaning exact match.
    """
    if ":" in pattern:
        prefix, glob = pattern.split(":", 1)
    else:
        prefix, glob = pattern, None
    return prefix.split(), glob


def _match_at(argv: List[str], offset: int, prefix_tokens: List[str], glob: Optional[str]) -> bool:
    """Match prefix tokens + glob against ``argv`` starting at ``offset``."""
    end = offset + len(prefix_tokens)
    if end > len(argv):
        return False
    for tok, pat in zip(argv[offset:end], prefix_tokens):
        if not fnmatch.fnmatch(tok, pat):
            return False
    remainder = argv[end:]
    if glob is None:
        # Exact pattern: nothing may follow the prefix tokens.
        return not remainder
    if glob == "*":
        return True
    if not remainder:
        return False
    # Only the FIRST positional token (or the joined remainder as a whole)
    # may satisfy the glob — matching arbitrary later args would let a caller
    # smuggle a disallowed flag in front (see BashTool._matches_pattern).
    return fnmatch.fnmatch(remainder[0], glob) or fnmatch.fnmatch(" ".join(remainder), glob)


def command_matches_pattern(argv: List[str], pattern: str, anchor: bool = True) -> bool:
    """Check whether tokenized command ``argv`` matches ``pattern``.

    Args:
        argv: ``shlex.split`` result of the command string.
        pattern: Rule pattern (see module docstring for syntax).
        anchor: When True the prefix must start at ``argv[0]``. Deny rules
            additionally match unanchored (any offset) so ``rm:*`` catches
            ``xargs rm`` / ``find . -exec rm``; allow rules must never do this.
    """
    prefix_tokens, glob = _split_pattern(pattern)
    if not prefix_tokens or not argv:
        return False
    if anchor:
        return _match_at(argv, 0, prefix_tokens, glob)
    return any(_match_at(argv, i, prefix_tokens, glob) for i in range(len(argv) - len(prefix_tokens) + 1))


def session_bucket_for(argv: List[str], matched_pattern: Optional[str]) -> str:
    """Session-approval bucket key for a command.

    A matched ask-rule pattern is its own bucket. Otherwise grouped CLIs
    (git/docker/npm/...) bucket on the first two tokens so approving one
    subcommand never green-lights the others; everything else buckets on
    ``argv[0]``.
    """
    if matched_pattern:
        return matched_pattern
    if not argv:
        return "<empty>"
    if argv[0] in _GROUP_COMMANDS and len(argv) >= 2 and not argv[1].startswith("-"):
        return f"{argv[0]} {argv[1]}"
    return argv[0]


def _split_top_level(command: str, *, chain: bool) -> Optional[List[str]]:
    """Split on top-level, unquoted control operators.

    ``chain=False`` splits on ``|`` only; ``chain=True`` also splits on
    ``&&``, ``||`` and ``;``. Only single/double quotes and backslash escaping
    are tracked — enough to keep ``grep "a|b"``, ``git commit -m "fix; bug"``
    and ``find ... -exec ls {} \\;`` from being split. Returns ``None`` when
    the string is not cleanly segmentable (see the public wrappers).
    """
    segments: List[str] = []
    buf: List[str] = []
    in_single = in_double = False
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if c == "\\" and not in_single:
            # Backslash escapes the next char outside single quotes; keep both
            # so shlex re-parses them and the escaped char can't be a split.
            buf.append(c)
            if i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            i += 1
            continue
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c in "|&;" and not in_single and not in_double:
            nxt = command[i + 1] if i + 1 < n else ""
            if c == "|":
                if nxt == "&":
                    # `|&` (stderr pipe) is not a plain pipe.
                    return None
                if nxt == "|" and not chain:
                    # `||` (logical OR) is not a pipeline.
                    return None
                step = 2 if nxt == "|" else 1
            elif not chain:
                # `&` / `;` are ordinary characters for the pipeline-only
                # splitter; the caller's metacharacter check rejects them.
                buf.append(c)
                i += 1
                continue
            elif c == "&":
                if nxt != "&":
                    if nxt == ">" or _last_nonspace(buf) in "<>":
                        # The `&` belongs to a redirection, not to job control:
                        # `2>&1`, `>&2`, `<&0` (preceded by the operator) or
                        # `&>file` (followed by it). Keep it in the segment and
                        # let the redirection ceiling judge the whole form —
                        # bailing here would send `ls 2>&1 | head` to the
                        # ceiling as one opaque string and hide `head` from the
                        # per-sub-command prompt. A genuine trailing `&`
                        # (`sleep 5 &`) still returns None below.
                        buf.append(c)
                        i += 1
                        continue
                    # A lone `&` backgrounds the command — not a sequencer.
                    return None
                step = 2
            else:  # ";"
                step = 1
            segments.append("".join(buf))
            buf = []
            i += step
            continue
        buf.append(c)
        i += 1
    segments.append("".join(buf))
    if in_single or in_double:
        return None
    stripped = [s.strip() for s in segments]
    if any(s == "" for s in stripped):
        return None
    return stripped


def split_pipeline(command: str) -> Optional[List[str]]:
    """Split a command on top-level, unquoted ``|`` into pipeline segments.

    Used by the execution layer (``BashTool._is_command_allowed``, the legacy
    ``allowed_patterns`` whitelist for skills). Deliberately narrower than
    :func:`split_command_chain`: that whitelist is a separate trust boundary
    and must keep rejecting every non-pipe construct outright.

    Returns:
        * ``[command]`` — no top-level pipe (a plain single command).
        * ``["seg1", "seg2", ...]`` — a clean pipeline, each part stripped.
        * ``None`` — NOT a simple pipeline: ``||`` (logical OR), ``|&``
          (stderr pipe), an empty segment (``ls |``, ``| ls``), or unbalanced
          quotes. The caller routes these to the safety ceiling.
    """
    return _split_top_level(command, chain=False)


def split_command_chain(command: str) -> Optional[List[str]]:
    """Split a command into sub-commands on ``|``, ``&&``, ``||`` and ``;``.

    Used by the permission layer so every sub-command is judged, displayed and
    granted on its own. Returns ``None`` — routing the whole command to the
    safety ceiling — for forms that cannot be segmented safely: ``|&``, a lone
    background ``&``, an empty segment (``ls &&``, ``;;``, ``| ls``) or
    unbalanced quotes.

    An ``&`` that belongs to a redirection (``2>&1``, ``>&2``, ``<&0``,
    ``&>file``) is not a background operator and does not stop segmentation;
    ``sleep 5 &`` still does.

    Note that ``None`` is not the only path to the safety ceiling: a segment
    that still carries ``$()``, backticks, a newline or a redirection other
    than the benign forms in ``_BENIGN_REDIRECTION_RE`` is caught per-segment
    by :func:`contains_shell_metachars`.
    """
    return _split_top_level(command, chain=True)


def _evaluate_single_command(command: str, rules: BashCommandRules) -> BashRuleDecision:
    """Evaluate one non-pipeline command segment. See module docstring order."""
    try:
        argv = shlex.split(command)
    except ValueError as e:
        first_word = command.strip().split()[0] if command.strip() else "<empty>"
        return BashRuleDecision(
            level=PermissionLevel.ASK,
            source=BashDecisionSource.UNPARSEABLE,
            matched_pattern=None,
            reason=f"Command could not be parsed ({e})",
            bucket=first_word,
            safety_forced=True,
        )
    if not argv:
        return BashRuleDecision(
            level=PermissionLevel.ASK,
            source=BashDecisionSource.UNPARSEABLE,
            matched_pattern=None,
            reason="Empty command",
            bucket="<empty>",
            safety_forced=True,
        )

    # ``datus <plugin> --profile <p> <subcommand>`` runs the same subcommand
    # as its unqualified form — normalize so plugin-declared rules and
    # session buckets keep matching (see _normalize_datus_plugin_argv).
    raw_argv = argv
    argv = _normalize_datus_plugin_argv(argv)

    # 1. Deny — most aggressive: anchored and unanchored, before everything
    # else so a deny can never be downgraded by a later stage. Unanchored
    # matching means a deny on ``rm:*`` also catches a segment ``xargs rm``.
    # Deny also matches the RAW (pre-normalization) argv, so a user rule like
    # ``datus hello --profile prod:*`` can block a specific plugin profile
    # even though ask/allow matching sees through the flag.
    for pattern in rules.deny:
        if (
            command_matches_pattern(argv, pattern, anchor=True)
            or command_matches_pattern(argv, pattern, anchor=False)
            or (
                raw_argv is not argv
                and (
                    command_matches_pattern(raw_argv, pattern, anchor=True)
                    or command_matches_pattern(raw_argv, pattern, anchor=False)
                )
            )
        ):
            return BashRuleDecision(
                level=PermissionLevel.DENY,
                source=BashDecisionSource.DENY_RULE,
                matched_pattern=pattern,
                reason=f"Blocked by deny rule '{pattern}'",
                bucket=session_bucket_for(argv, None),
                safety_forced=False,
            )

    # 2. Safety ceiling — wrappers, compound-construct fragments and shell
    # metacharacters can never auto-allow, regardless of allow rules.
    if _is_compound_construct(command, argv):
        return BashRuleDecision(
            level=PermissionLevel.ASK,
            source=BashDecisionSource.SAFETY,
            matched_pattern=None,
            reason=f"'{command.strip()}' is part of a compound shell construct that the rules cannot decompose",
            bucket=session_bucket_for(argv, None),
            safety_forced=True,
        )
    if argv[0] in _WRAPPER_COMMANDS:
        return BashRuleDecision(
            level=PermissionLevel.ASK,
            source=BashDecisionSource.SAFETY,
            matched_pattern=None,
            reason=f"'{argv[0]}' executes its arguments as a new command; rules cannot see the wrapped command",
            bucket=session_bucket_for(argv, None),
            safety_forced=True,
        )
    if _has_inline_code_flag(argv):
        return BashRuleDecision(
            level=PermissionLevel.ASK,
            source=BashDecisionSource.SAFETY,
            matched_pattern=None,
            reason=f"'{argv[0]}' with an inline-code flag executes an arbitrary string; rules cannot see the code",
            bucket=session_bucket_for(argv, None),
            safety_forced=True,
        )
    # ``2>&1`` / ``2>/dev/null`` and friends are dropped before the check —
    # see ``_BENIGN_REDIRECTION_RE`` for which forms and why. Ordering matters:
    # the wrapper and inline-code ceilings above already fired, so a command
    # that reaches here cannot smuggle a second command past this exemption
    # (``sh -c "... 2>&1"`` is stopped by the wrapper rule).
    if contains_shell_metachars(_without_benign_redirections(command)):
        return BashRuleDecision(
            level=PermissionLevel.ASK,
            source=BashDecisionSource.SAFETY,
            matched_pattern=None,
            reason="Command contains shell metacharacters",
            bucket=session_bucket_for(argv, None),
            safety_forced=True,
        )

    # 3. Ask rules beat allow rules.
    for pattern in rules.ask:
        if command_matches_pattern(argv, pattern, anchor=True):
            return BashRuleDecision(
                level=PermissionLevel.ASK,
                source=BashDecisionSource.ASK_RULE,
                matched_pattern=pattern,
                reason=f"Confirmation required by ask rule '{pattern}'",
                bucket=session_bucket_for(argv, pattern),
                safety_forced=False,
            )

    # 4. Allow rules — anchored only.
    for pattern in rules.allow:
        if command_matches_pattern(argv, pattern, anchor=True):
            return BashRuleDecision(
                level=PermissionLevel.ALLOW,
                source=BashDecisionSource.ALLOW_RULE,
                matched_pattern=pattern,
                reason=f"Allowed by rule '{pattern}'",
                bucket=session_bucket_for(argv, pattern),
                safety_forced=False,
            )

    # 5. Default.
    return BashRuleDecision(
        level=PermissionLevel(rules.default),
        source=BashDecisionSource.DEFAULT,
        matched_pattern=None,
        reason="No bash command rule matched",
        bucket=session_bucket_for(argv, None),
        safety_forced=False,
    )


# Severity for picking the representative sub-command of a chain that is
# neither fully allowed nor denied/safety-forced. An ask-rule segment outranks
# a default segment so the chain's overall source is ASK_RULE — this matters
# because the hook's dangerous-profile fallback only re-evaluates DEFAULT
# decisions, so a DEFAULT representative could let a permissive profile swallow
# an ask-rule segment.
_ASK_SEVERITY = {
    BashDecisionSource.ASK_RULE: 2,
    BashDecisionSource.DEFAULT: 1,
}


def _as_segment(command: str, decision: BashRuleDecision) -> BashSegmentDecision:
    """Project a per-segment ``BashRuleDecision`` onto a segment record."""
    return BashSegmentDecision(
        command=command.strip(),
        level=decision.level,
        source=decision.source,
        matched_pattern=decision.matched_pattern,
        reason=decision.reason,
        bucket=decision.bucket,
        safety_forced=decision.safety_forced,
    )


def _with_ask_segments(command: str, decision: BashRuleDecision) -> BashRuleDecision:
    """Attach the single-command ask segment so callers have one code path."""
    if decision.level != PermissionLevel.ASK:
        return decision
    return replace(decision, ask_segments=(_as_segment(command, decision),))


def _evaluate_chain(segments: List[str], rules: BashCommandRules) -> BashRuleDecision:
    """Aggregate per-sub-command decisions for a chain (``a | b && c; d``).

    * any sub-command DENY             -> DENY
    * all sub-commands ALLOW           -> ALLOW
    * otherwise                        -> ASK carrying EVERY non-allow
      sub-command in ``ask_segments``; ``safety_forced`` when any of them is,
      and represented by the most severe one (ask-rule over default) for
      bucket/source/pattern.
    """
    decisions = [_evaluate_single_command(seg, rules) for seg in segments]

    for seg, d in zip(segments, decisions):
        if d.level == PermissionLevel.DENY:
            return BashRuleDecision(
                level=PermissionLevel.DENY,
                source=BashDecisionSource.DENY_RULE,
                matched_pattern=d.matched_pattern,
                reason=f"Sub-command '{seg}' matched deny rule '{d.matched_pattern}'",
                bucket=d.bucket,
                safety_forced=False,
            )

    if all(d.level == PermissionLevel.ALLOW for d in decisions):
        patterns = ", ".join(d.matched_pattern for d in decisions if d.matched_pattern)
        return BashRuleDecision(
            level=PermissionLevel.ALLOW,
            source=BashDecisionSource.ALLOW_RULE,
            matched_pattern=None,
            reason=f"All sub-commands matched allow rules ({patterns})",
            bucket=session_bucket_for(shlex.split(segments[0]), None),
            safety_forced=False,
        )

    ask_segments = tuple(_as_segment(seg, d) for seg, d in zip(segments, decisions) if d.level != PermissionLevel.ALLOW)
    # A safety-forced sub-command pins the whole chain to the safety ceiling —
    # but unlike the old short-circuit, the other sub-commands stay visible so
    # the prompt can still list everything that needs approval.
    forced = [s for s in ask_segments if s.safety_forced]
    rep = forced[0] if forced else max(ask_segments, key=lambda s: _ASK_SEVERITY.get(s.source, 0))
    return BashRuleDecision(
        level=PermissionLevel.ASK,
        source=BashDecisionSource.SAFETY if forced else rep.source,
        matched_pattern=None if forced else rep.matched_pattern,
        reason=f"Requires confirmation: {rep.reason}",
        bucket=rep.bucket,
        safety_forced=bool(forced),
        ask_segments=ask_segments,
    )


def evaluate_bash_command(command: str, rules: BashCommandRules) -> BashRuleDecision:
    """Evaluate a bash command against a ruleset.

    Routes chained commands (``a | b``, ``a && b``, ``a; b``, ``a || b``)
    through per-sub-command judging so a chain of allow-listed read-only
    commands can auto-run and an ASK names every sub-command involved, while
    un-segmentable shell constructs (``$()``, backticks, redirection,
    background ``&``, newlines) fall to the safety ceiling. See the module
    docstring for the full decision order.
    """
    # Empty / whitespace-only input has no chain structure to speak of;
    # let the single-command path return the UNPARSEABLE decision.
    if not command.strip():
        return _with_ask_segments(command, _evaluate_single_command(command, rules))

    segments = split_command_chain(command)
    if segments is None:
        # Not segmentable (``|&``, background ``&``, empty segment, unbalanced
        # quotes). Unbalanced quotes make shlex raise → route to the
        # single-command path so the decision is UNPARSEABLE; everything else
        # is an un-analyzable shell construct → safety ceiling.
        try:
            shlex.split(command)
        except ValueError:
            return _with_ask_segments(command, _evaluate_single_command(command, rules))
        return _with_ask_segments(
            command,
            BashRuleDecision(
                level=PermissionLevel.ASK,
                source=BashDecisionSource.SAFETY,
                matched_pattern=None,
                # Not every failure here is a metacharacter — an empty segment
                # (``ls &&``, ``a;; b``) is a malformed chain. This text reaches
                # the user through the prompt body and the AI reviewer through
                # ``static_assessment.reason``, so it names the real cause.
                reason="Command uses a shell construct that cannot be split into sub-commands",
                bucket=command.strip().split()[0],
                safety_forced=True,
            ),
        )
    if len(segments) == 1:
        return _with_ask_segments(segments[0], _evaluate_single_command(segments[0], rules))
    return _evaluate_chain(segments, rules)


# Complete PermissionConfig's forward reference when this module loaded first
# (see the mirror block at the bottom of permission_config.py).
from datus.tools.permission import permission_config as _permission_config  # noqa: E402

_permission_config.PermissionConfig.model_rebuild(
    _types_namespace={"BashCommandRules": BashCommandRules},
)
