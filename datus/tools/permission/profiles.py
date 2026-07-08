# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Predefined permission profiles (normal / auto / dangerous).

A Permission Profile is a named base ``PermissionConfig`` that users can
select via ``agent.yml`` (``permissions.profile: <name>``) or switch to
at runtime with ``/profile``. User-defined ``permissions.rules`` are
layered on top via ``PermissionConfig.merge_with`` (last-match-wins).

The three profiles embody three security postures:

* ``normal``:    read-only tools, semantic tools, and skill loading allowed,
  all other writes ASK, named destructive tools DENY. Default for new installs.
* ``auto``:      Normal + workspace writes auto-execute, BI/scheduler
  non-trigger writes auto, DB writes still ASK.
* ``dangerous``: everything ALLOW, including EXTERNAL filesystem paths in
  interactive mode. Workflow (non-interactive) flows still fail closed on
  EXTERNAL paths regardless of profile.

Filesystem decision matrix (rules here interact with the zone gate in
``PermissionHooks._handle_filesystem_zone``):

============  ===================  ==============  ==============  ==============
operation     zone                 normal          auto            dangerous
============  ===================  ==============  ==============  ==============
read          INTERNAL/WHITELIST   bypass          bypass          bypass
read          HIDDEN               tool not-found  tool not-found  tool not-found
read          EXTERNAL (interactive)  ASK(path)    ASK(path)       bypass
read          EXTERNAL (strict)    tool fail       tool fail       tool fail
read          EXTERNAL (non-interactive)  raise    raise           raise
write         INTERNAL             rule lookup ASK bypass          bypass
write         WHITELIST            tool reject     tool reject     tool reject
write         HIDDEN               tool not-found  tool not-found  tool not-found
write         EXTERNAL (interactive)  ASK(path)    ASK(path)       bypass
write         EXTERNAL (strict)    tool fail       tool fail       tool fail
write         EXTERNAL (non-interactive)  raise    raise           raise
============  ===================  ==============  ==============  ==============

The zone gate consults ``active_profile`` and the tool name; the rules
below cover the cases where the zone gate returns ``False`` (e.g.
``normal × INTERNAL × write_file`` lands here as ``default=ASK``).
"""

from typing import Optional

from datus.tools.permission.bash_rules import BashCommandRules
from datus.tools.permission.permission_config import (
    PermissionConfig,
    PermissionLevel,
    PermissionRule,
)

PROFILE_NAMES: tuple[str, ...] = ("normal", "auto", "dangerous")


def _rule(tool: str, pattern: str, permission: PermissionLevel) -> PermissionRule:
    return PermissionRule(tool=tool, pattern=pattern, permission=permission)


# --- Normal ------------------------------------------------------------------
# default=ASK + semantic/read/skill-load ALLOW + named destructives DENY + MCP/script ASK.
_NORMAL_RULES = [
    # context search / date utilities
    _rule("context_search_tools", "*", PermissionLevel.ALLOW),
    _rule("date_parsing_tools", "*", PermissionLevel.ALLOW),
    # db read. ``execute_sql`` is the unified SQL entry point; its read-vs-write
    # gating is handled dynamically per statement type in
    # ``PermissionHooks._handle_sql_permission`` (read-only bypass, writes/DDL
    # ASK), so it intentionally has no static rule here.
    _rule("db_tools", "verify_sql", PermissionLevel.ALLOW),
    _rule("db_tools", "list_*", PermissionLevel.ALLOW),
    _rule("db_tools", "search_*", PermissionLevel.ALLOW),
    _rule("db_tools", "describe_*", PermissionLevel.ALLOW),
    _rule("db_tools", "get_*", PermissionLevel.ALLOW),
    _rule("db_tools", "search_*", PermissionLevel.ALLOW),
    # bi read + destructive deny
    _rule("bi_tools", "list_*", PermissionLevel.ALLOW),
    _rule("bi_tools", "get_*", PermissionLevel.ALLOW),
    _rule("bi_tools", "delete_*", PermissionLevel.DENY),
    # semantic read
    _rule("semantic_tools", "list_*", PermissionLevel.ALLOW),
    _rule("semantic_tools", "search_*", PermissionLevel.ALLOW),
    _rule("semantic_tools", "get_*", PermissionLevel.ALLOW),
    _rule("semantic_tools", "query_metrics", PermissionLevel.ALLOW),
    # semantic generation helpers
    _rule("semantic_tools", "check_semantic_object_exists", PermissionLevel.ALLOW),
    _rule("semantic_tools", "end_*_generation", PermissionLevel.ALLOW),
    _rule("semantic_tools", "generate_*_id", PermissionLevel.ALLOW),
    _rule("semantic_tools", "*", PermissionLevel.ALLOW),
    # scheduler read + destructive deny
    _rule("scheduler_tools", "list_*", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "get_*", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "delete_job", PermissionLevel.DENY),
    # filesystem read. Writes are handled by the zone × profile gate in
    # ``PermissionHooks._handle_filesystem_zone`` — see this file's docstring
    # for the full decision matrix. Patterns here must match a real
    # ``FilesystemFuncTool.available_tools()`` entry; dead rules silently
    # become noise once tools are renamed.
    _rule("filesystem_tools", "read_*", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "glob", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "grep", PermissionLevel.ALLOW),
    # persistent memory: ALLOW. add_memory/edit_memory only touch a single
    # hidden, 2000-byte-capped MEMORY.md with no external reach — gating
    # benign self-notes behind a prompt would fire on routine "remember this"
    # turns for zero safety benefit.
    _rule("memory_tools", "*", PermissionLevel.ALLOW),
    # plan read
    _rule("tools", "todo_list", PermissionLevel.ALLOW),
    _rule("tools", "todo_read", PermissionLevel.ALLOW),
    # ``ask_user`` IS the user-interaction channel — gating "may I ask
    # the user?" behind a permission prompt is absurd. Always ALLOW.
    _rule("tools", "ask_user", PermissionLevel.ALLOW),
    # ``tools`` bucket is chat's catch-all for benign helpers (platform
    # doc lookups, doc search, etc.). Read-only patterns follow the
    # project-wide convention: ``list_*`` / ``search_*`` / ``get_*``.
    _rule("tools", "list_*", PermissionLevel.ALLOW),
    _rule("tools", "search_*", PermissionLevel.ALLOW),
    _rule("tools", "get_*", PermissionLevel.ALLOW),
    _rule("tools", "validate_skill", PermissionLevel.ALLOW),
    # sub-agent delegation: ALLOW. The ``task()`` tool just spawns a
    # subagent; the tools the subagent actually invokes are gated by the
    # subagent's own PermissionHooks instance. Double-prompting here would
    # fire on nearly every chat interaction for zero safety benefit.
    _rule("sub_agent_tools", "*", PermissionLevel.ALLOW),
    # reference templates: read-only end to end — search/get/render are pure
    # lookups and ``execute_reference_template`` renders Jinja then runs the
    # internal read-only query path (never a write). Historically gen_sql lumped
    # these into ``semantic_tools`` (ALLOW) while chat left them in the
    # catch-all (ASK); the dedicated category unifies on ALLOW.
    _rule("reference_template_tools", "*", PermissionLevel.ALLOW),
    # artifact authoring helpers (``start_new_*`` / ``bind_existing_*`` /
    # ``save_query*`` / ``validate_render``) are subagent-internal state
    # mutations confined to the artifact tree; users review the artifact as
    # a whole via the rendered preview, not per-call prompts. Mirrors the
    # historical lumping into ``semantic_tools``.
    _rule("artifact_tools", "*", PermissionLevel.ALLOW),
    # platform doc lookups are read-only local reads.
    _rule("platform_doc_tools", "list_*", PermissionLevel.ALLOW),
    _rule("platform_doc_tools", "get_*", PermissionLevel.ALLOW),
    _rule("platform_doc_tools", "search_*", PermissionLevel.ALLOW),
    # web_tool reaches the public network (Tavily search / httpx fetch), but it is
    # read-only retrieval and ``web_fetch`` is hardened against SSRF (non-public
    # targets are refused). ALLOW keeps it from prompting on every lookup. NOTE:
    # vendor-native web tools (Codex hosted web_search, Anthropic web_search_20250305
    # / web_fetch_20250910) run server-side and do NOT pass through local
    # PermissionHooks at all — gating the local backends at ASK could not be
    # enforced on the native ones anyway, so we keep both consistently at ALLOW.
    _rule("web_tool", "web_search", PermissionLevel.ALLOW),
    _rule("web_tool", "web_fetch", PermissionLevel.ALLOW),
    # mcp: ASK; skill loading ALLOW.
    _rule("mcp.*", "*", PermissionLevel.ASK),
    _rule("skills", "*", PermissionLevel.ALLOW),
    # General-purpose bash execution: always ASK in normal/auto so a stray
    # command can't run without user consent. ``dangerous`` profile (default
    # ALLOW, no rules) lets it through. This coarse rule is the FALLBACK for
    # when the command-level ``bash_commands`` ruleset below is absent or
    # errors out (defense in depth) — the fine-grained gate in
    # ``PermissionHooks._handle_bash_permission`` normally decides first.
    _rule("bash_tools", "bash", PermissionLevel.ASK),
]

# Command-level bash whitelist for ``normal``: commands that cannot mutate
# state or execute arbitrary code REGARDLESS of their flags, so prompting on
# them is pure friction (they're the read-only backbone of data/dev work in a
# pipeline like ``cat log | grep err | wc -l``).
#
# SECURITY — what is deliberately NOT here:
#   * awk / sed — programmable: ``awk 'BEGIN{system("...")}'`` execs, ``sed -i``
#     writes in place. Prefix+first-arg matching can't see the quoted program,
#     so blanket-allowing them would auto-run arbitrary code. Left at ASK; a
#     user who accepts the risk can add ``awk:*`` / ``sed:*`` to agent.yml.
#   * find — ``-exec`` / ``-delete`` run commands / delete files.
#   * tee / sort -o / dd — write files.
#   * curl / wget / nc / ssh — network egress.
#   * rm / mv / cp / mkdir / touch / chmod — filesystem writes (mkdir/touch
#     move to ``auto`` below).
# Wrappers (bash -c, sudo, xargs, env, ...) and any non-pipe metacharacter
# command are force-ASKed by the safety ceiling in ``evaluate_bash_command``
# regardless of this list. Tunable via ``permissions.bash_commands``.
_NORMAL_BASH_ALLOW = [
    # environment / identity info
    "pwd",
    "whoami",
    "id:*",
    # ``hostname NAME`` / ``date -s`` are mutating forms (root-only, but they
    # still break the read-only contract), so no blanket ``:*`` here: exact
    # match only, plus ``date +FORMAT`` for the common read-only formatting.
    "hostname",
    "uname:*",
    "date",
    "date:+*",
    "which:*",
    "type:*",
    "printenv:*",
    "locale:*",
    "echo:*",
    "printf:*",
    "seq:*",
    "true",
    "false",
    # directory / file listing & inspection
    "ls:*",
    "tree:*",
    "stat:*",
    "file:*",
    "du:*",
    "df:*",
    "wc:*",
    "basename:*",
    "dirname:*",
    "realpath:*",
    "readlink:*",
    # file content readers (stdout only — cannot write or exec)
    "cat:*",
    "head:*",
    "tail:*",
    "tac:*",
    "nl:*",
    "rev:*",
    # text search / filter / transform (read input, write only to stdout)
    "grep:*",
    "egrep:*",
    "fgrep:*",
    "rg:*",
    "cut:*",
    "tr:*",
    "uniq:*",
    "comm:*",
    "column:*",
    "fold:*",
    "paste:*",
    "expand:*",
    "unexpand:*",
    # compare
    "diff:*",
    "cmp:*",
    # checksums / encoding (read-only computation)
    "md5sum:*",
    "sha1sum:*",
    "sha256sum:*",
    "cksum:*",
    "base64:*",
    # git read-only
    "git status",
    "git branch",
    "git log:*",
    "git diff:*",
    "git show:*",
    "git remote:*",
    "git tag",
    "git describe:*",
    "git rev-parse:*",
    "git blame:*",
    "git ls-files:*",
    "git shortlog:*",
]

# ``auto`` adds lightweight workspace writes, consistent with auto's
# filesystem posture (INTERNAL writes auto-allow). Read-only commands already
# come from ``_NORMAL_BASH_ALLOW``. ``sort`` is here (not normal) because
# ``sort -o FILE`` can write a file.
_AUTO_BASH_ALLOW = _NORMAL_BASH_ALLOW + [
    "sort:*",
    "mkdir:*",
    "touch:*",
]

NORMAL = PermissionConfig(
    default_permission=PermissionLevel.ASK,
    rules=_NORMAL_RULES,
    bash_commands=BashCommandRules(allow=_NORMAL_BASH_ALLOW),
)

# --- Auto --------------------------------------------------------------------
# Normal's rules + workspace writes + BI create/update + scheduler non-trigger.
# DB writes remain ASK (no env detection in MVP). Named destructives are
# *downgraded* from DENY to ASK — the user is already in a productive
# posture, so forcing them to switch to ``dangerous`` just to remove one
# chart is hostile. ASK still gates each call via the broker.
_AUTO_EXTRA_RULES = [
    # workspace writes — ALLOW promotes the INTERNAL × write decision in
    # ``_handle_filesystem_zone`` so the auto profile no longer falls through
    # to ``default=ASK`` like normal does. EXTERNAL paths are still gated by
    # the zone branch (ASK in auto, ALLOW only in dangerous).
    _rule("filesystem_tools", "write_file", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "edit_file", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "delete_file", PermissionLevel.ALLOW),
    # plan writes
    _rule("tools", "todo_write", PermissionLevel.ALLOW),
    _rule("tools", "todo_update", PermissionLevel.ALLOW),
    _rule("semantic_tools", "validate_semantic", PermissionLevel.ALLOW),
    # bi write (excluding delete_*, which stays DENY from NORMAL via earlier rule)
    _rule("bi_tools", "create_*", PermissionLevel.ALLOW),
    _rule("bi_tools", "update_*", PermissionLevel.ALLOW),
    _rule("bi_tools", "add_*", PermissionLevel.ALLOW),
    # scheduler non-trigger writes
    _rule("scheduler_tools", "submit_*", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "update_job", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "pause_job", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "resume_job", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "trigger_*", PermissionLevel.ASK),
    # db writes: always ASK. ``execute_sql`` writes/DDL are gated dynamically by
    # ``PermissionHooks._handle_sql_permission`` (read-only bypass), so only the
    # standalone cross-DB transfer tool needs an explicit ASK rule here.
    _rule("db_tools", "transfer_query_result", PermissionLevel.ASK),
    # Named destructives — downgrade NORMAL's DENY to ASK in Auto. The user
    # can confirm at the prompt; DENY would force a profile switch to
    # ``dangerous`` (which is far more permissive than just ``delete``).
    _rule("bi_tools", "delete_*", PermissionLevel.ASK),
    _rule("scheduler_tools", "delete_job", PermissionLevel.ASK),
]

AUTO = PermissionConfig(
    default_permission=PermissionLevel.ASK,
    rules=_NORMAL_RULES + _AUTO_EXTRA_RULES,
    bash_commands=BashCommandRules(allow=_AUTO_BASH_ALLOW),
)

# --- Dangerous ---------------------------------------------------------------
# default=ALLOW, no rules. PathZone at hook layer still gates EXTERNAL fs.
# bash_commands stays None: the fine-grained bash gate steps aside and the
# profile default of ALLOW lets every command through, preserving the
# historical allow-everything behavior of this profile.
DANGEROUS = PermissionConfig(
    default_permission=PermissionLevel.ALLOW,
    rules=[],
)


_PROFILES: dict[str, PermissionConfig] = {
    "normal": NORMAL,
    "auto": AUTO,
    "dangerous": DANGEROUS,
}


def get_profile(name: str) -> PermissionConfig:
    """Return the profile config for ``name``.

    Raises ``ValueError`` with an actionable message if ``name`` is unknown.
    Callers that want to fall back (e.g. ``AgentConfig`` on invalid YAML)
    must catch the exception themselves — this function never silently
    substitutes a default, so bugs that would otherwise mask bad config are
    caught at the call site.
    """
    try:
        return _PROFILES[name]
    except KeyError as e:
        raise ValueError(f"Unknown profile {name!r}. Valid options: {', '.join(PROFILE_NAMES)}") from e


def build_user_overrides(
    profile_name: str,
    user_raw: Optional[dict] = None,
) -> Optional[PermissionConfig]:
    """Construct the ``user_overrides`` PermissionConfig for ``switch_profile``.

    Mirrors the default-permission injection in :func:`build_effective_config`
    so a runtime profile switch preserves the new profile's safety posture
    instead of inheriting ``PermissionConfig.from_dict``'s built-in
    ``"allow"`` default. Returns ``None`` when there are no user rules to
    layer — callers can pass that directly to ``switch_profile``.

    Args:
        profile_name: Target profile name; raises ``ValueError`` if unknown.
        user_raw: Raw user permissions dict (without the ``profile`` key).

    Returns:
        The user-overrides ``PermissionConfig``, or ``None`` if ``user_raw``
        is empty.
    """
    if not user_raw:
        return None
    if "default" not in user_raw and "default_permission" not in user_raw:
        base = get_profile(profile_name)
        dp = base.default_permission
        user_raw = {
            **user_raw,
            "default_permission": dp.value if hasattr(dp, "value") else dp,
        }
    return PermissionConfig.from_dict(user_raw)


def build_effective_config(
    profile_name: str,
    user_raw: Optional[dict] = None,
    plugin_bash_rules: Optional[BashCommandRules] = None,
) -> PermissionConfig:
    """Build the effective permission config for a profile + user overrides.

    Used by both startup config loading (``AgentConfig._init_permissions_config``)
    and runtime profile switching (CLI ``/profile`` handler) so the
    default-preservation invariant lives in one place.

    If ``user_raw`` has no explicit ``default`` / ``default_permission``
    key, the profile base's default is injected before ``from_dict`` parses
    it — this prevents ``merge_with`` from silently clobbering the profile's
    safety posture with ``PermissionConfig.from_dict``'s built-in
    ``"allow"`` default (see spec decision #3).

    Args:
        profile_name: One of ``PROFILE_NAMES``. Raises ``ValueError`` on
            unknown names.
        user_raw: The raw user permissions dict (without the ``profile``
            key). ``None`` or ``{}`` yields the bare profile base.
        plugin_bash_rules: Bash rules declared by installed plugins for this
            profile (``collect_plugin_cli_permissions()[profile_name]``).
            Layered between the profile base and the user rules: since
            ``evaluate_bash_command`` applies deny > safety > ask > allow
            regardless of list order, a plugin ``allow`` can never override a
            user ``deny``. Ignored when the profile carries no
            ``bash_commands`` ruleset (dangerous stays fully open).

    Returns:
        The merged ``PermissionConfig`` ready to install on
        ``AgentConfig.permissions_config`` and ``PermissionManager.global_config``.
    """
    base = get_profile(profile_name)
    if plugin_bash_rules is not None and not plugin_bash_rules.is_empty() and base.bash_commands is not None:
        # Rebuild instead of mutating: ``get_profile`` returns shared
        # module-level singletons.
        base = PermissionConfig(
            default_permission=base.default_permission,
            rules=list(base.rules),
            bash_commands=base.bash_commands.merge_with(plugin_bash_rules),
        )
    if not user_raw:
        return base

    if "default" not in user_raw and "default_permission" not in user_raw:
        dp = base.default_permission
        user_raw = {
            **user_raw,
            "default_permission": dp.value if hasattr(dp, "value") else dp,
        }

    user_cfg = PermissionConfig.from_dict(user_raw)
    return base.merge_with(user_cfg)
