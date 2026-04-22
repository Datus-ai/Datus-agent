# Permission Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship three predefined `Permission Profile`s (`normal` / `auto` / `dangerous`) as a single-PR MVP layered on the existing `PermissionConfig` rule engine, with `agent.yml` default + runtime `/profile` CLI selection.

**Architecture:** Add `profiles.py` with the three `PermissionConfig` constants (base rules). `AgentConfig` merges profile base with user rules using the existing `merge_with` semantics. `PermissionManager` gains `active_profile` state and a `switch_profile()` method that clears `_session_approvals`. A new `/profile` slash command opens an arrow-key selection dialog via `InteractionBroker`, with a mandatory second confirmation every time the session transitions into `dangerous`. Status bar gains a `profile` segment.

**Tech Stack:** Python 3.12+, `pydantic` (existing), `pytest` + `pytest-asyncio`, `prompt_toolkit` (existing styles), `uv` for tooling.

**Spec reference:** `docs/superpowers/specs/2026-04-22-permission-profile-design.md`

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `datus/tools/permission/profiles.py` | **new** | Three `PermissionConfig` constants; `get_profile(name)` lookup with explicit error |
| `datus/tools/permission/permission_manager.py` | modify | `active_profile` attribute; `switch_profile()` that rebuilds `global_config` and clears session approvals |
| `datus/configuration/agent_config.py` | modify | Parse `permissions.profile`; merge profile base with user rules; expose `active_profile_name` |
| `datus/cli/status_bar.py` | modify | `StatusBarState.profile` field, render in `format_plain` and `to_formatted_tokens`; `StatusBarProvider` reads from CLI |
| `datus/cli/repl.py` | modify | Hold `active_profile` at CLI level; register styles for `profile.auto` / `profile.dangerous`; add `_cmd_profile` handler; wire handler map |
| `datus/cli/slash_registry.py` | modify | Add `SlashSpec("profile", ...)` in the `system` group |
| `conf/agent.yml` | modify | Add example `permissions.profile: normal` stanza |
| `tests/unit_tests/tools/permission/test_profiles.py` | **new** | Snapshot rules for each profile; `get_profile()` error handling |
| `tests/unit_tests/tools/permission/test_permission_manager.py` | extend | `switch_profile()` clears approvals, rebuilds config |
| `tests/unit_tests/configuration/test_agent_config.py` | extend (or new) | `profile` parsing, default, fallback, merge with user rules |
| `tests/unit_tests/cli/test_status_bar.py` | extend | `profile` field renders with correct style class |
| `tests/unit_tests/cli/test_slash_registry.py` | extend | `/profile` spec registered; handler map has `profile` |
| `tests/test_cli_commands.py` | extend | `/profile` selection dialog, Dangerous confirmation flow |

Existing files that **do not change**:
- `datus/tools/permission/permission_config.py` — `PermissionRule`, `PermissionConfig`, `merge_with` are reused as-is
- `datus/tools/permission/permission_hooks.py` — existing `on_tool_start` flow transparent to profile rules
- `datus/tools/func_tool/fs_path_policy.py` — `PathZone` EXTERNAL→ASK continues to gate Dangerous filesystem boundary

---

## Task 1: Define the three profile rule sets (`profiles.py`)

**Files:**
- Create: `datus/tools/permission/profiles.py`
- Test: `tests/unit_tests/tools/permission/test_profiles.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit_tests/tools/permission/test_profiles.py
"""Tests for predefined permission profiles."""
import pytest

from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel
from datus.tools.permission.profiles import (
    AUTO,
    DANGEROUS,
    NORMAL,
    PROFILE_NAMES,
    get_profile,
)


class TestProfileRegistry:
    def test_three_profiles_exist(self):
        assert PROFILE_NAMES == ("normal", "auto", "dangerous")

    def test_get_profile_returns_expected_instance(self):
        assert get_profile("normal") is NORMAL
        assert get_profile("auto") is AUTO
        assert get_profile("dangerous") is DANGEROUS

    def test_get_profile_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("yolo")

    def test_normal_default_is_ask(self):
        assert NORMAL.default_permission == PermissionLevel.ASK

    def test_auto_default_is_ask(self):
        assert AUTO.default_permission == PermissionLevel.ASK

    def test_dangerous_default_is_allow(self):
        assert DANGEROUS.default_permission == PermissionLevel.ALLOW


class TestNormalProfile:
    def test_read_tools_allowed(self):
        """Normal allows context search, date parsing, and DB/BI/FS reads."""
        config = NORMAL
        assert _resolve(config, "context_search_tools", "search_metrics") == PermissionLevel.ALLOW
        assert _resolve(config, "db_tools", "read_query") == PermissionLevel.ALLOW
        assert _resolve(config, "db_tools", "list_tables") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "list_dashboards") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "read_file") == PermissionLevel.ALLOW

    def test_writes_ask(self):
        """Normal ASKs on all write-ish tools via default_permission."""
        config = NORMAL
        assert _resolve(config, "db_tools", "execute_ddl") == PermissionLevel.ASK
        assert _resolve(config, "filesystem_tools", "write_file") == PermissionLevel.ASK
        assert _resolve(config, "tools", "todo_write") == PermissionLevel.ASK

    def test_named_destructive_denied(self):
        """Normal DENYs named destructive BI and scheduler tools."""
        config = NORMAL
        assert _resolve(config, "bi_tools", "delete_dashboard") == PermissionLevel.DENY
        assert _resolve(config, "bi_tools", "delete_chart") == PermissionLevel.DENY
        assert _resolve(config, "scheduler_tools", "delete_job") == PermissionLevel.DENY

    def test_mcp_and_skills_ask(self):
        config = NORMAL
        assert _resolve(config, "mcp.filesystem", "read_file") == PermissionLevel.ASK
        assert _resolve(config, "skills", "any-skill") == PermissionLevel.ASK


class TestAutoProfile:
    def test_inherits_normal_reads(self):
        config = AUTO
        assert _resolve(config, "db_tools", "read_query") == PermissionLevel.ALLOW
        assert _resolve(config, "context_search_tools", "search_metrics") == PermissionLevel.ALLOW

    def test_workspace_writes_allowed(self):
        config = AUTO
        assert _resolve(config, "filesystem_tools", "write_file") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "edit_file") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "create_directory") == PermissionLevel.ALLOW

    def test_bi_write_allowed_but_delete_denied(self):
        config = AUTO
        assert _resolve(config, "bi_tools", "create_dashboard") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "update_chart") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "delete_dashboard") == PermissionLevel.DENY

    def test_scheduler_trigger_still_asks(self):
        config = AUTO
        assert _resolve(config, "scheduler_tools", "submit_sql_job") == PermissionLevel.ALLOW
        assert _resolve(config, "scheduler_tools", "trigger_scheduler_job") == PermissionLevel.ASK

    def test_db_writes_still_ask(self):
        """No env detection in MVP — all DB writes always ASK."""
        config = AUTO
        assert _resolve(config, "db_tools", "execute_ddl") == PermissionLevel.ASK
        assert _resolve(config, "db_tools", "execute_write") == PermissionLevel.ASK
        assert _resolve(config, "db_tools", "transfer_query_result") == PermissionLevel.ASK

    def test_mcp_and_skills_still_ask(self):
        config = AUTO
        assert _resolve(config, "mcp.filesystem", "read_file") == PermissionLevel.ASK
        assert _resolve(config, "skills", "any-skill") == PermissionLevel.ASK


class TestDangerousProfile:
    def test_everything_allowed_by_default(self):
        config = DANGEROUS
        assert _resolve(config, "db_tools", "execute_ddl") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "delete_dashboard") == PermissionLevel.ALLOW
        assert _resolve(config, "scheduler_tools", "delete_job") == PermissionLevel.ALLOW
        assert _resolve(config, "mcp.anything", "whatever") == PermissionLevel.ALLOW
        assert _resolve(config, "skills", "any-skill") == PermissionLevel.ALLOW


def _resolve(config: PermissionConfig, category: str, pattern: str) -> PermissionLevel:
    """Walk the rules last-match-wins, returning the final PermissionLevel."""
    result = config.default_permission
    for rule in config.rules:
        if _matches(rule, category, pattern):
            result = PermissionLevel(rule.permission) if isinstance(rule.permission, str) else rule.permission
    return result


def _matches(rule, category, pattern):
    import fnmatch
    if rule.tool != "*" and not fnmatch.fnmatch(category, rule.tool):
        return False
    if rule.pattern != "*" and not fnmatch.fnmatch(pattern, rule.pattern):
        return False
    return True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/tools/permission/test_profiles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'datus.tools.permission.profiles'` (or equivalent import failure).

- [ ] **Step 3: Implement `profiles.py`**

```python
# datus/tools/permission/profiles.py
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Predefined permission profiles (normal / auto / dangerous).

A Permission Profile is a named base ``PermissionConfig`` that users can
select via ``agent.yml`` (``permissions.profile: <name>``) or switch to
at runtime with ``/profile``. User-defined ``permissions.rules`` are
layered on top via ``PermissionConfig.merge_with`` (last-match-wins).

The three profiles embody three security postures:

* ``normal``:    read-only tools allowed, all writes ASK, named
  destructive tools DENY. Default for new installs.
* ``auto``:      Normal + workspace writes auto-execute, BI/scheduler
  non-trigger writes auto, DB writes still ASK.
* ``dangerous``: everything ALLOW. Filesystem EXTERNAL paths still
  prompt via ``PathZone`` at the hook layer — that gate is orthogonal
  to the rule engine.
"""

from datus.tools.permission.permission_config import (
    PermissionConfig,
    PermissionLevel,
    PermissionRule,
)

PROFILE_NAMES: tuple[str, ...] = ("normal", "auto", "dangerous")


def _rule(tool: str, pattern: str, permission: PermissionLevel) -> PermissionRule:
    return PermissionRule(tool=tool, pattern=pattern, permission=permission)


# --- Normal ------------------------------------------------------------------
# default=ASK + explicit reads ALLOW + named destructives DENY + MCP/skills ASK.
_NORMAL_RULES = [
    # context search / date utilities
    _rule("context_search_tools", "*", PermissionLevel.ALLOW),
    _rule("date_parsing_tools", "*", PermissionLevel.ALLOW),
    # db read
    _rule("db_tools", "read_query", PermissionLevel.ALLOW),
    _rule("db_tools", "list_*", PermissionLevel.ALLOW),
    _rule("db_tools", "describe_*", PermissionLevel.ALLOW),
    _rule("db_tools", "get_*", PermissionLevel.ALLOW),
    # bi read + destructive deny
    _rule("bi_tools", "list_*", PermissionLevel.ALLOW),
    _rule("bi_tools", "get_*", PermissionLevel.ALLOW),
    _rule("bi_tools", "delete_*", PermissionLevel.DENY),
    # semantic read
    _rule("semantic_tools", "list_*", PermissionLevel.ALLOW),
    _rule("semantic_tools", "search_*", PermissionLevel.ALLOW),
    _rule("semantic_tools", "get_*", PermissionLevel.ALLOW),
    _rule("semantic_tools", "query_metrics", PermissionLevel.ALLOW),
    # scheduler read + destructive deny
    _rule("scheduler_tools", "list_*", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "get_*", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "delete_job", PermissionLevel.DENY),
    # filesystem read
    _rule("filesystem_tools", "read_*", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "list_*", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "directory_tree", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "search_files", PermissionLevel.ALLOW),
    # plan read
    _rule("tools", "todo_read", PermissionLevel.ALLOW),
    # mcp + skills: ASK (explicit so future defaults can shift default_permission)
    _rule("mcp.*", "*", PermissionLevel.ASK),
    _rule("skills", "*", PermissionLevel.ASK),
]

NORMAL = PermissionConfig(
    default_permission=PermissionLevel.ASK,
    rules=_NORMAL_RULES,
)

# --- Auto --------------------------------------------------------------------
# Normal's rules + workspace writes + BI create/update + scheduler non-trigger.
# DB writes remain ASK (no env detection in MVP). Named destructives remain DENY.
_AUTO_EXTRA_RULES = [
    # workspace writes (PathZone handles EXTERNAL ASK at hook layer)
    _rule("filesystem_tools", "write_file", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "edit_file", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "create_directory", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "move_file", PermissionLevel.ALLOW),
    # plan writes
    _rule("tools", "todo_write", PermissionLevel.ALLOW),
    _rule("tools", "todo_update", PermissionLevel.ALLOW),
    # generation finalize helpers
    _rule("semantic_tools", "end_*_generation", PermissionLevel.ALLOW),
    _rule("semantic_tools", "validate_semantic", PermissionLevel.ALLOW),
    _rule("semantic_tools", "generate_*_id", PermissionLevel.ALLOW),
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
    # db writes: always ASK
    _rule("db_tools", "execute_ddl", PermissionLevel.ASK),
    _rule("db_tools", "execute_write", PermissionLevel.ASK),
    _rule("db_tools", "transfer_query_result", PermissionLevel.ASK),
    _rule("db_tools", "write_query", PermissionLevel.ASK),
    # Re-assert deny on bi_tools.delete_* after the create_*/update_*/add_* allows,
    # so the delete rule wins via last-match-wins regardless of declaration order.
    _rule("bi_tools", "delete_*", PermissionLevel.DENY),
]

AUTO = PermissionConfig(
    default_permission=PermissionLevel.ASK,
    rules=_NORMAL_RULES + _AUTO_EXTRA_RULES,
)

# --- Dangerous ---------------------------------------------------------------
# default=ALLOW, no rules. PathZone at hook layer still gates EXTERNAL fs.
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
        raise ValueError(
            f"Unknown profile {name!r}. Valid options: {', '.join(PROFILE_NAMES)}"
        ) from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/tools/permission/test_profiles.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Ruff format and check**

Run: `uv run ruff format . && uv run ruff check --fix .`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add datus/tools/permission/profiles.py tests/unit_tests/tools/permission/test_profiles.py
git commit -m "[Feature] Add permission profile rule sets (normal/auto/dangerous)

Define three predefined PermissionConfig profiles as the MVP base for
/profile runtime selection."
```

---

## Task 2: Wire `profile` into `AgentConfig._init_permissions_config`

**Files:**
- Modify: `datus/configuration/agent_config.py:749-767` (replace `_init_permissions_config`)
- Add: `active_profile_name` attribute
- Test: `tests/unit_tests/configuration/test_agent_config.py` (create if missing; extend otherwise)

- [ ] **Step 1: Confirm existing test file or add new**

Run: `ls tests/unit_tests/configuration/ 2>/dev/null`
If `test_agent_config.py` exists, extend it. Otherwise create minimally scoped to the new behavior only.

- [ ] **Step 2: Write the failing test**

```python
# tests/unit_tests/configuration/test_agent_config_permissions.py
"""Tests for permission profile loading in AgentConfig."""
from datus.configuration.agent_config import AgentConfig
from datus.tools.permission.permission_config import PermissionLevel
from datus.tools.permission.profiles import AUTO


def _make_config(permissions_raw: dict | None):
    """Build an AgentConfig instance with only the fields this test exercises.

    AgentConfig has many required fields; test fixtures should live in a
    conftest. For this test we call the private helper directly on a
    zero-arg instance if the public API is not available — prefer the
    public constructor with a realistic minimum YAML if one is already
    documented in the repo."""
    cfg = AgentConfig.__new__(AgentConfig)
    cfg.permissions_config = cfg._init_permissions_config(permissions_raw or {})
    cfg.active_profile_name = getattr(cfg, "active_profile_name", "normal")
    return cfg


def test_missing_permissions_yields_normal_profile():
    cfg = _make_config(None)
    assert cfg.active_profile_name == "normal"
    # Normal profile has explicit read allows
    assert any(
        r.tool == "db_tools" and r.pattern == "read_query"
        for r in cfg.permissions_config.rules
    )


def test_profile_field_selects_auto():
    cfg = _make_config({"profile": "auto"})
    assert cfg.active_profile_name == "auto"
    # Auto has the workspace write allows
    assert any(
        r.tool == "filesystem_tools" and r.pattern == "write_file"
        and PermissionLevel(r.permission) == PermissionLevel.ALLOW
        for r in cfg.permissions_config.rules
    )


def test_user_rules_layered_on_profile_base():
    """User's permissions.rules should be appended after profile rules,
    so last-match-wins lets users override."""
    cfg = _make_config({
        "profile": "auto",
        "rules": [
            {"tool": "db_tools", "pattern": "execute_ddl", "permission": "deny"},
        ],
    })
    rules = cfg.permissions_config.rules
    # User rule must be present and appear after the auto base rule
    auto_idx = next(
        i for i, r in enumerate(rules)
        if r.tool == "db_tools" and r.pattern == "execute_ddl" and PermissionLevel(r.permission) == PermissionLevel.ASK
    )
    user_idx = next(
        i for i, r in enumerate(rules)
        if r.tool == "db_tools" and r.pattern == "execute_ddl" and PermissionLevel(r.permission) == PermissionLevel.DENY
    )
    assert user_idx > auto_idx, "user rule must be appended after profile base"


def test_invalid_profile_falls_back_to_normal(caplog):
    cfg = _make_config({"profile": "yolo"})
    assert cfg.active_profile_name == "normal"
    assert any("Invalid profile" in rec.message for rec in caplog.records)


def test_dangerous_profile_loads():
    cfg = _make_config({"profile": "dangerous"})
    assert cfg.active_profile_name == "dangerous"
    assert cfg.permissions_config.default_permission == PermissionLevel.ALLOW
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/configuration/test_agent_config_permissions.py -v`
Expected: FAIL on multiple tests (profile field ignored, `active_profile_name` missing, etc.).

- [ ] **Step 4: Modify `_init_permissions_config` and add `active_profile_name`**

In `datus/configuration/agent_config.py`:

Replace the existing method (at line ~749):

```python
def _init_permissions_config(self, permissions_raw: Dict[str, Any]):
    """Initialize unified permission configuration.

    Loads the base profile (default: ``normal``) and layers user-supplied
    ``rules`` on top via ``PermissionConfig.merge_with`` (last-match-wins).
    Sets ``self.active_profile_name`` so the CLI status bar and
    ``/profile`` command can read the source of truth from one place.

    Args:
        permissions_raw: Raw permissions config from agent.yml. May be
            empty ({}) — treated as "no profile override, no user rules",
            equivalent to ``{"profile": "normal", "rules": []}``.

    Returns:
        PermissionConfig instance (never ``None``; a profile is always
        applied).
    """
    from datus.tools.permission.permission_config import PermissionConfig
    from datus.tools.permission.profiles import get_profile

    permissions_raw = permissions_raw or {}
    requested_profile = permissions_raw.get("profile", "normal")

    try:
        base = get_profile(requested_profile)
        self.active_profile_name = requested_profile
    except ValueError as e:
        logger.warning(
            f"Invalid profile {requested_profile!r} in agent.yml: {e}. "
            f"Falling back to 'normal'."
        )
        base = get_profile("normal")
        self.active_profile_name = "normal"

    # Remove the ``profile`` key so PermissionConfig.from_dict only
    # consumes ``default`` and ``rules`` as before.
    user_raw = {k: v for k, v in permissions_raw.items() if k != "profile"}
    user_cfg = PermissionConfig.from_dict(user_raw) if user_raw else None

    return base.merge_with(user_cfg) if user_cfg else base
```

Also ensure `active_profile_name` has an initialization default — add near other attribute initializations around line 582. Find:

```python
# Initialize unified permission system
self.permissions_config = self._init_permissions_config(kwargs.get("permissions", {}))
```

and ensure `self.active_profile_name` is initialized before that call (in case `_init_permissions_config` raises before setting it). Insert:

```python
# Active profile name — set by ``_init_permissions_config``. Pre-seed to
# ``normal`` so downstream readers always see a valid string even if
# permission init fails.
self.active_profile_name: str = "normal"
# Initialize unified permission system
self.permissions_config = self._init_permissions_config(kwargs.get("permissions", {}))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/configuration/test_agent_config_permissions.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Run full permission test suite**

Run: `uv run pytest tests/unit_tests/tools/permission/ tests/unit_tests/configuration/ -v`
Expected: All tests PASS (no regressions in existing permission tests).

- [ ] **Step 7: Format, lint, and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add datus/configuration/agent_config.py tests/unit_tests/configuration/test_agent_config_permissions.py
git commit -m "[Feature] Load permission profile in AgentConfig

_init_permissions_config now selects a base profile (default: normal)
and layers user-supplied rules on top via merge_with. Invalid profile
names fall back to normal with a warning."
```

---

## Task 3: Add `switch_profile()` to `PermissionManager`

**Files:**
- Modify: `datus/tools/permission/permission_manager.py` (add `active_profile`, `switch_profile`)
- Test: `tests/unit_tests/tools/permission/test_permission_manager.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_tests/tools/permission/test_permission_manager.py`:

```python
class TestPermissionManagerProfileSwitching:
    """PermissionManager.switch_profile() updates the global config and
    clears all session approvals so a switch to a stricter profile never
    leaves behind prior ``allow`` grants."""

    def test_active_profile_defaults_to_normal(self):
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        assert mgr.active_profile == "normal"

    def test_switch_profile_updates_active_name(self):
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        mgr.switch_profile("auto")
        assert mgr.active_profile == "auto"

    def test_switch_profile_replaces_global_config(self):
        from datus.tools.permission.permission_config import PermissionLevel
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        mgr.switch_profile("dangerous")
        # Dangerous has default ALLOW with no rules
        assert mgr.global_config.default_permission == PermissionLevel.ALLOW
        assert len(mgr.global_config.rules) == 0

    def test_switch_profile_clears_session_approvals(self):
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        mgr.approve_for_session("db_tools", "execute_ddl")
        assert mgr._session_approvals
        mgr.switch_profile("auto")
        assert mgr._session_approvals == {}

    def test_switch_profile_with_user_overrides(self):
        from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        user_overrides = PermissionConfig(
            default_permission=PermissionLevel.ASK,
            rules=[
                PermissionRule(tool="db_tools", pattern="execute_ddl", permission=PermissionLevel.DENY)
            ],
        )
        mgr.switch_profile("auto", user_overrides=user_overrides)
        # The user rule must be present in the merged config
        matching = [r for r in mgr.global_config.rules
                    if r.tool == "db_tools" and r.pattern == "execute_ddl"]
        # Final matching rule's permission should be DENY
        final = matching[-1].permission
        final_level = PermissionLevel(final) if isinstance(final, str) else final
        assert final_level == PermissionLevel.DENY

    def test_switch_profile_unknown_raises(self):
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        import pytest
        with pytest.raises(ValueError, match="Unknown profile"):
            mgr.switch_profile("yolo")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/tools/permission/test_permission_manager.py::TestPermissionManagerProfileSwitching -v`
Expected: FAIL on `active_profile` attribute missing, `switch_profile` method missing.

- [ ] **Step 3: Extend `PermissionManager`**

In `datus/tools/permission/permission_manager.py`:

Update the imports near the top to include profile helpers:

```python
from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.profiles import get_profile
```

Modify `__init__` (around line 49-67). Find:

```python
def __init__(
    self,
    global_config: Optional[PermissionConfig] = None,
    node_overrides: Optional[Dict[str, PermissionConfig]] = None,
):
    """Initialize the permission manager. ..."""
    self.global_config = global_config or PermissionConfig()
    self.node_overrides = node_overrides or {}
    self._permission_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[bool]]] = None

    # Cache for session-approved permissions (tool_category.tool_name -> approved)
    self._session_approvals: Dict[str, bool] = {}

    logger.debug(f"PermissionManager initialized with {len(self.global_config.rules)} global rules")
```

Replace with:

```python
def __init__(
    self,
    global_config: Optional[PermissionConfig] = None,
    node_overrides: Optional[Dict[str, PermissionConfig]] = None,
    active_profile: str = "normal",
):
    """Initialize the permission manager.

    Args:
        global_config: Global permission configuration (typically the
            result of ``profile_base.merge_with(user_rules)``).
        node_overrides: Per-node permission overrides.
        active_profile: Name of the currently active profile. Informational
            — the rules baked into ``global_config`` are authoritative;
            this string is what the status bar shows and what
            ``switch_profile`` updates.
    """
    self.global_config = global_config or PermissionConfig()
    self.node_overrides = node_overrides or {}
    self.active_profile = active_profile
    self._permission_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[bool]]] = None

    # Cache for session-approved permissions (tool_category.tool_name -> approved)
    self._session_approvals: Dict[str, bool] = {}

    logger.debug(
        f"PermissionManager initialized: profile={self.active_profile}, "
        f"{len(self.global_config.rules)} global rules"
    )
```

Add the new `switch_profile` method after `clear_session_approvals` (around line 270):

```python
def switch_profile(
    self,
    profile_name: str,
    user_overrides: Optional[PermissionConfig] = None,
) -> None:
    """Switch to a different permission profile.

    Replaces ``global_config`` with ``get_profile(profile_name)`` merged
    with ``user_overrides`` (if any), updates ``active_profile``, and
    clears ``_session_approvals`` so prior ``always-allow`` grants never
    leak into the new profile's security posture.

    Args:
        profile_name: One of ``"normal"``, ``"auto"``, ``"dangerous"``.
            Raises ``ValueError`` on unknown names.
        user_overrides: Optional user rules to layer on top (typically
            the ``permissions.rules`` portion of ``agent.yml``).

    Raises:
        ValueError: if ``profile_name`` is not a known profile.
    """
    base = get_profile(profile_name)  # raises ValueError on unknown name
    self.global_config = base.merge_with(user_overrides) if user_overrides else base
    self.active_profile = profile_name
    self._session_approvals.clear()
    logger.info(
        f"Profile switched to '{profile_name}': "
        f"{len(self.global_config.rules)} effective rules, session approvals cleared"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/tools/permission/test_permission_manager.py -v`
Expected: All tests PASS (including the existing ones — `active_profile` default is `"normal"` to match the new default).

- [ ] **Step 5: Full permission test suite**

Run: `uv run pytest tests/unit_tests/tools/permission/ -v`
Expected: All PASS.

- [ ] **Step 6: Format, lint, and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add datus/tools/permission/permission_manager.py tests/unit_tests/tools/permission/test_permission_manager.py
git commit -m "[Feature] Add switch_profile() to PermissionManager

switch_profile() loads the profile base via get_profile(), merges
user overrides, and clears session approvals so prior always-allow
grants never survive across profiles."
```

---

## Task 4: Add `profile` field to the status bar

**Files:**
- Modify: `datus/cli/status_bar.py` (`StatusBarState.profile`, render, provider)
- Modify: `datus/cli/repl.py` (register styles for `profile.auto` and `profile.dangerous`)
- Test: `tests/unit_tests/cli/test_status_bar.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_tests/cli/test_status_bar.py`:

```python
class TestStatusBarProfile:
    def test_profile_renders_in_plain_output(self):
        from datus.cli.status_bar import StatusBarState

        state = StatusBarState(profile="normal")
        rendered = state.format_plain()
        assert "normal" in rendered

    def test_profile_default_is_normal(self):
        from datus.cli.status_bar import StatusBarState

        state = StatusBarState()
        assert state.profile == "normal"

    def test_profile_renders_with_style_class(self):
        from datus.cli.status_bar import StatusBarState

        state = StatusBarState(profile="dangerous")
        tokens = state.to_formatted_tokens()
        styles = [style for style, _text in tokens]
        assert "class:status-bar.profile.dangerous" in styles

    def test_profile_auto_style_class(self):
        from datus.cli.status_bar import StatusBarState

        state = StatusBarState(profile="auto")
        tokens = state.to_formatted_tokens()
        styles = [style for style, _text in tokens]
        assert "class:status-bar.profile.auto" in styles

    def test_profile_normal_uses_generic_class(self):
        """normal is the default; it should render with the neutral
        ``status-bar.profile`` class (not a variant)."""
        from datus.cli.status_bar import StatusBarState

        state = StatusBarState(profile="normal")
        tokens = state.to_formatted_tokens()
        styles = [style for style, _text in tokens]
        assert "class:status-bar.profile" in styles
        assert "class:status-bar.profile.dangerous" not in styles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/cli/test_status_bar.py -v`
Expected: FAIL — `StatusBarState` has no `profile` field.

- [ ] **Step 3: Extend `StatusBarState`**

In `datus/cli/status_bar.py`:

Add `profile` to the dataclass (insert after `plan_mode`, around line 60):

```python
@dataclass
class StatusBarState:
    """Snapshot of status bar data rendered before each prompt."""

    agent: str = "chat"
    model: str = "-"
    connector: str = ""
    cumulative_tokens: int = 0
    cached_tokens: int = 0
    context_used: int = 0
    context_total: int = 0
    plan_mode: bool = False
    agent_running: bool = False
    profile: str = "normal"
```

Update `format_plain` to include the profile segment between `connector` and `model` (replace the existing method, ~line 75-90):

```python
def format_plain(self) -> str:
    """Render the status bar as a plain string (used for tests and logs)."""
    segments = ["Datus"]
    if self.plan_mode:
        segments.append("PLAN")
    segments.append(self.agent)
    if self.connector:
        segments.append(self.connector)
    segments.append(self.profile)
    segments.extend(
        [
            self.model,
            self.tokens_display(),
            self.context_display(),
        ]
    )
    return " " + " │ ".join(segments) + " "
```

Update `to_formatted_tokens` — find the block that appends connector/model (~line 92-120) and insert a profile segment with a class name that varies by profile risk:

```python
def to_formatted_tokens(self) -> List[Tuple[str, str]]:
    """Return prompt_toolkit formatted text tokens with styled segments."""
    sep: Tuple[str, str] = ("class:status-bar.sep", " │ ")
    pad: Tuple[str, str] = ("class:status-bar", " ")

    tokens: List[Tuple[str, str]] = [pad, ("class:status-bar.brand", "Datus")]
    if self.plan_mode:
        tokens.extend([sep, ("class:status-bar.plan", "PLAN")])
    tokens.extend([sep, ("class:status-bar.agent", self.agent)])
    if self.connector:
        tokens.extend([sep, ("class:status-bar.connector", self.connector)])
    # Profile segment. Variant class (``status-bar.profile.auto`` /
    # ``.dangerous``) lets repl.py render risky profiles with bold/red
    # while leaving normal neutral.
    profile_class = f"class:status-bar.profile.{self.profile}" if self.profile in ("auto", "dangerous") else "class:status-bar.profile"
    tokens.extend([sep, (profile_class, self.profile)])
    tokens.extend(
        [
            sep,
            ("class:status-bar.model", self.model),
            sep,
            ("class:status-bar.tokens", self.tokens_display()),
            sep,
            ("class:status-bar.ctx", self.context_display()),
        ]
    )
    if self.agent_running:
        tokens.extend([sep, ("class:status-bar.running", "● running")])
    tokens.append(pad)
    return tokens
```

Update `StatusBarProvider.current_state` (~line 129-148) to populate the `profile` field by reading from the CLI. Insert a new `_resolve_profile` helper and wire it:

```python
def current_state(self) -> StatusBarState:
    cumulative, cached = self._resolve_session_totals()
    tui_app = getattr(self._cli, "tui_app", None)
    agent_running = False
    if tui_app is not None:
        try:
            agent_running = tui_app.agent_running.is_set()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"status_bar: failed to read agent_running: {e}")
    return StatusBarState(
        agent=self._resolve_agent(),
        model=self._resolve_model(),
        connector=self._resolve_connector(),
        cumulative_tokens=cumulative,
        cached_tokens=cached,
        context_used=self._resolve_context_used(),
        context_total=self._resolve_context_total(),
        plan_mode=bool(getattr(self._cli, "plan_mode_active", False)),
        agent_running=agent_running,
        profile=self._resolve_profile(),
    )

def _resolve_profile(self) -> str:
    """Return the active profile name from the CLI, defaulting to ``normal``.

    The CLI owns the mutable ``active_profile`` string (initialized from
    ``agent_config.active_profile_name`` and toggled by ``/profile``). If
    the attribute is not yet wired during tests or early init, fall back
    to ``normal`` rather than raising.
    """
    return getattr(self._cli, "active_profile", None) or "normal"
```

- [ ] **Step 4: Add the profile styles to `repl.py`**

In `datus/cli/repl.py` around line 429-443, inside the `Style.from_dict({...})` block, add three new entries alongside `status-bar.plan`:

```python
"status-bar.plan": "#9a9aaa",
"status-bar.profile": "#9a9aaa",
"status-bar.profile.auto": "ansicyan bold",
"status-bar.profile.dangerous": "ansired bold",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/cli/test_status_bar.py -v`
Expected: All PASS.

- [ ] **Step 6: Format, lint, and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add datus/cli/status_bar.py datus/cli/repl.py tests/unit_tests/cli/test_status_bar.py
git commit -m "[Feature] Render active profile in CLI status bar

Add StatusBarState.profile field with variant style classes
(status-bar.profile.auto / .dangerous) so risky profiles show
in cyan / bold red and normal stays neutral."
```

---

## Task 5: Register `/profile` in the slash command registry

**Files:**
- Modify: `datus/cli/slash_registry.py` (add `SlashSpec`)
- Test: `tests/unit_tests/cli/test_slash_registry.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit_tests/cli/test_slash_registry.py`:

```python
def test_profile_command_registered():
    from datus.cli.slash_registry import SLASH_COMMANDS

    names = {spec.name for spec in SLASH_COMMANDS}
    assert "profile" in names


def test_profile_command_is_in_system_group():
    from datus.cli.slash_registry import SLASH_COMMANDS

    spec = next(s for s in SLASH_COMMANDS if s.name == "profile")
    assert spec.group == "system"
    assert spec.summary  # non-empty summary for autocomplete
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/cli/test_slash_registry.py -v`
Expected: FAIL — `profile` not registered.

- [ ] **Step 3: Add the `SlashSpec`**

In `datus/cli/slash_registry.py`, inside the `SLASH_COMMANDS` tuple, append in the `system` section (after line 82, before the closing `)`):

```python
    SlashSpec(
        "profile",
        "Switch the permission profile (normal / auto / dangerous)",
        "system",
    ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/cli/test_slash_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add datus/cli/slash_registry.py tests/unit_tests/cli/test_slash_registry.py
git commit -m "[Feature] Register /profile slash command spec

/profile lives in the 'system' group and appears in the autocomplete
menu alongside /mcp, /skill, and /services."
```

---

## Task 6: Implement the `/profile` command handler

**Files:**
- Modify: `datus/cli/repl.py` — add `active_profile` attribute, `_cmd_profile` handler, wire into handler map
- Test: `tests/test_cli_commands.py` (extend) — selection dialog, Dangerous confirmation flow

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_commands.py` (or a new `tests/unit_tests/cli/test_profile_command.py` if that fits repo conventions better — check existing structure):

```python
# tests/unit_tests/cli/test_profile_command.py
"""Tests for the /profile slash command handler.

Uses a minimal CLI stub + InteractionBroker mock so we exercise handler
logic without spinning up prompt_toolkit's event loop.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel
from datus.tools.permission.permission_manager import PermissionManager


class _FakeBroker:
    """Async stand-in for ``InteractionBroker.request``.

    ``scripted_responses`` is a list of ``(choice_key, callback_message)``
    tuples — the broker returns them in order as each request is made."""

    def __init__(self, scripted_responses):
        self.scripted_responses = list(scripted_responses)
        self.requests = []

    async def request(self, contents, choices, default_choices=None):
        self.requests.append({"contents": contents, "choices": choices, "default": default_choices})

        async def _callback(_msg):
            return None

        choice = self.scripted_responses.pop(0)
        return choice, _callback


class _FakeCLI:
    """Minimal CLI surface exercised by ``_cmd_profile``."""

    def __init__(self, broker, manager, agent_config):
        self.broker = broker
        self.console = MagicMock()
        self.agent_config = agent_config
        self.active_profile = agent_config.active_profile_name
        self.chat_commands = MagicMock()
        self.chat_commands.current_node = MagicMock()
        self.chat_commands.current_node.permission_manager = manager


def _make_agent_config(profile: str = "normal"):
    """Build a barely-populated AgentConfig for these tests."""
    from datus.configuration.agent_config import AgentConfig
    cfg = AgentConfig.__new__(AgentConfig)
    cfg.active_profile_name = profile
    cfg.permissions_config = cfg._init_permissions_config({"profile": profile})
    cfg.active_profile_name = profile  # re-seed in case _init reset it
    return cfg


@pytest.mark.asyncio
async def test_profile_switch_to_auto(monkeypatch):
    """Selecting 'auto' in the dialog switches the CLI and manager."""
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    manager.approve_for_session("db_tools", "execute_ddl")
    agent_config = _make_agent_config("normal")
    broker = _FakeBroker(scripted_responses=["auto"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"
    assert manager._session_approvals == {}


@pytest.mark.asyncio
async def test_profile_switch_dangerous_requires_confirmation(monkeypatch):
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    agent_config = _make_agent_config("normal")
    # First dialog: user picks 'dangerous'. Second dialog: confirms 'enable'.
    broker = _FakeBroker(scripted_responses=["dangerous", "enable"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")

    assert cli.active_profile == "dangerous"
    assert manager.active_profile == "dangerous"
    # Two separate broker requests were made (selection + confirmation)
    assert len(broker.requests) == 2


@pytest.mark.asyncio
async def test_profile_switch_dangerous_cancelled(monkeypatch):
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    agent_config = _make_agent_config("auto")
    # First dialog: user picks 'dangerous'. Second dialog: cancels.
    broker = _FakeBroker(scripted_responses=["dangerous", "cancel"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")

    # Profile unchanged
    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"


@pytest.mark.asyncio
async def test_profile_dialog_cancel_keeps_current():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    agent_config = _make_agent_config("auto")
    broker = _FakeBroker(scripted_responses=["cancel"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"


@pytest.mark.asyncio
async def test_profile_select_same_profile_is_noop():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    manager.approve_for_session("db_tools", "execute_ddl")
    agent_config = _make_agent_config("auto")
    broker = _FakeBroker(scripted_responses=["auto"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")

    # No switch happened — approvals preserved
    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"
    assert manager._session_approvals  # not cleared
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/cli/test_profile_command.py -v`
Expected: FAIL — `DatusCLI._cmd_profile_async` does not exist.

- [ ] **Step 3: Add the CLI-level `active_profile` attribute**

In `datus/cli/repl.py`, locate the `DatusCLI.__init__` area near line 143 where `plan_mode_active = False` is set. Add:

```python
self.plan_mode_active = False
# Active permission profile name. Initialized from agent_config if loaded;
# mutated by ``/profile``. Read by StatusBarProvider for display.
self.active_profile: str = getattr(self.agent_config, "active_profile_name", "normal")
```

(Adjust exactly to match whatever `self.agent_config` is named and when it's attached — if it's initialized later, set `self.active_profile = "normal"` here and re-assign after agent_config is wired. Inspect line 143 context carefully before inserting.)

- [ ] **Step 4: Add `_cmd_profile` handler and its async core**

In `datus/cli/repl.py`, after `_cmd_help` (around line 1425), add:

```python
def _cmd_profile(self, args: str) -> None:
    """Entry point for ``/profile``.

    Delegates to an async implementation because profile selection goes
    through ``InteractionBroker.request`` (an async API). Uses
    ``asyncio.run()`` when no loop is running (standard REPL) and
    ``asyncio.ensure_future`` inside an existing loop (TUI path).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        asyncio.run(self._cmd_profile_async(args))
    else:
        # Schedule on the running loop; fire-and-forget is acceptable because
        # the dialog is modal and user input won't race with other turns.
        asyncio.ensure_future(self._cmd_profile_async(args))


async def _cmd_profile_async(self, args: str) -> None:
    """Open the profile selection dialog, then apply the choice.

    Sequence:
        1. Build a choices dict reflecting the three profiles + cancel.
        2. Request a selection from the broker.
        3. If the choice is ``"cancel"`` or equals the active profile,
           return without mutating state.
        4. If the choice is ``"dangerous"``, require a second confirmation
           with default-highlighted Cancel. Every session transition into
           dangerous triggers this.
        5. Rebuild ``agent_config.permissions_config`` so future nodes see
           the new base; call ``switch_profile`` on the current node's
           manager for immediate effect; update ``self.active_profile``.
    """
    from datus.cli.execution_state import InteractionCancelled
    from datus.tools.permission.permission_config import PermissionConfig
    from datus.tools.permission.profiles import PROFILE_NAMES, get_profile

    current = self.active_profile
    choices_dict = {
        "normal": f"normal      Read-only + confirm every write{'  ← current' if current == 'normal' else ''}",
        "auto": f"auto        Workspace writes auto; DB/MCP still ask{'  ← current' if current == 'auto' else ''}",
        "dangerous": f"dangerous   Nearly all writes auto (⚠ see warning){'  ← current' if current == 'dangerous' else ''}",
        "cancel": "cancel      Keep current profile",
    }

    broker = getattr(self, "broker", None)
    if broker is None:
        self.console.print("[yellow]/profile requires an interactive session.[/]")
        return

    try:
        choice, callback = await broker.request(
            contents=[f"### Select Permission Profile\n\nCurrent: **{current}**"],
            choices=[choices_dict],
            default_choices=["cancel"],
        )
    except InteractionCancelled:
        return

    if choice == "cancel":
        await callback("**Kept current profile**")
        return

    if choice not in PROFILE_NAMES:
        self.console.print(f"[bold red]Unknown profile:[/] {choice}")
        return

    if choice == current:
        await callback(f"**Already on {choice}**")
        return

    # Dangerous second confirmation — every session transition.
    if choice == "dangerous":
        confirm_content = (
            "### ⚠ DANGEROUS PROFILE — Explicit Confirmation Required\n\n"
            "Switching to Dangerous will auto-execute:\n"
            "  • All DB writes (including DDL, DELETE)\n"
            "  • All BI/Scheduler writes (including deletes)\n"
            "  • All MCP tools\n"
            "  • All skills\n\n"
            "Still protected: writes outside workspace require ASK; "
            "~/.datus internals remain hidden."
        )
        try:
            confirm_choice, confirm_cb = await broker.request(
                contents=[confirm_content],
                choices=[{
                    "cancel": "Cancel (stay on current profile)",
                    "enable": "Enable Dangerous for this session",
                }],
                default_choices=["cancel"],
            )
        except InteractionCancelled:
            return
        if confirm_choice != "enable":
            await confirm_cb("**Dangerous mode cancelled**")
            return
        await confirm_cb("**Dangerous mode enabled**")

    # Apply the switch.
    new_base = get_profile(choice)

    # Rebuild agent_config's merged permissions_config so future nodes see it.
    # User rules (permissions.rules) were layered at startup; re-use them.
    user_rules_cfg: PermissionConfig | None = None
    # Extract only the user-rule slice of the original raw permissions by
    # filtering out the ``profile`` key and rebuilding.
    raw_permissions = getattr(self.agent_config, "_raw_permissions", {}) or {}
    raw_user = {k: v for k, v in raw_permissions.items() if k != "profile"}
    if raw_user:
        user_rules_cfg = PermissionConfig.from_dict(raw_user)

    self.agent_config.permissions_config = (
        new_base.merge_with(user_rules_cfg) if user_rules_cfg else new_base
    )
    self.agent_config.active_profile_name = choice

    # Immediate effect on the current node (if any).
    current_node = getattr(self.chat_commands, "current_node", None)
    if current_node is not None and hasattr(current_node, "permission_manager"):
        current_node.permission_manager.switch_profile(choice, user_overrides=user_rules_cfg)

    prior_approvals = 0
    if current_node is not None and hasattr(current_node, "permission_manager"):
        prior_approvals = len(getattr(current_node.permission_manager, "_session_approvals", {}))

    self.active_profile = choice
    self.console.print(
        f"[green]Profile switched:[/] {current} → {choice}\n"
        f"[dim]Session approvals cleared (was: {prior_approvals})[/]"
    )
```

Add the import at the top of `repl.py` if not already present:

```python
import asyncio
```

- [ ] **Step 5: Wire `_cmd_profile` into the handler map**

In `_build_slash_handler_map` (around line 265-301), append inside the `system` section:

```python
            "services": self.service_commands.cmd_services,
            "profile": self._cmd_profile,
        }
```

- [ ] **Step 6: Persist raw permissions on `AgentConfig`**

The handler reads `self.agent_config._raw_permissions` to reconstruct user rules on switch. In `datus/configuration/agent_config.py`, inside `_init_permissions_config` (modified in Task 2), store the raw dict before processing:

```python
def _init_permissions_config(self, permissions_raw: Dict[str, Any]):
    """..."""
    from datus.tools.permission.permission_config import PermissionConfig
    from datus.tools.permission.profiles import get_profile

    permissions_raw = permissions_raw or {}
    # Stash a copy so /profile can rebuild effective config on switch
    # without re-reading YAML.
    self._raw_permissions = dict(permissions_raw)
    requested_profile = permissions_raw.get("profile", "normal")
    # ... rest unchanged
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/cli/test_profile_command.py -v`
Expected: All PASS.

- [ ] **Step 8: Full unit test suite (catch regressions)**

Run: `uv run pytest tests/unit_tests/ -q`
Expected: All PASS.

- [ ] **Step 9: Format, lint, and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add datus/cli/repl.py datus/configuration/agent_config.py tests/unit_tests/cli/test_profile_command.py
git commit -m "[Feature] Implement /profile command with selection UI

/profile opens an arrow-key selection dialog via InteractionBroker.
Selecting dangerous triggers a second confirmation (default-highlighted
Cancel) that runs every session transition into dangerous. Switches
rebuild agent_config.permissions_config and call switch_profile() on
the current node's PermissionManager for immediate effect."
```

---

## Task 7: Add profile to `conf/agent.yml` example

**Files:**
- Modify: `conf/agent.yml`

- [ ] **Step 1: Inspect existing agent.yml structure**

Run: `grep -n "^agent:\|^permissions:\|^skills:" conf/agent.yml`

- [ ] **Step 2: Add or update the permissions stanza**

Insert a permissions block with comments documenting the field. The exact location should be near other top-level sections (look for `skills:` or equivalent). Add:

```yaml
# ---------------------------------------------------------------------------
# Permission profile — pick a base security posture, layer user-defined
# rules on top via ``rules``. Override at runtime with ``/profile``.
#
# Profiles:
#   normal    — Read-only allowed; all writes prompt; named destructive deny
#   auto      — Normal + workspace writes auto; DB writes still prompt
#   dangerous — Nearly everything auto (writes outside workspace still prompt)
#
# User rules in ``rules:`` are evaluated after the profile base, so they
# override the profile via last-match-wins. See docs/superpowers/specs/
# 2026-04-22-permission-profile-design.md for the full rule listing.
# ---------------------------------------------------------------------------
permissions:
  profile: normal
  # rules:
  #   - tool: db_tools
  #     pattern: execute_ddl
  #     permission: deny
```

- [ ] **Step 3: Sanity-check that startup still succeeds**

Run: `uv run python -c "from datus.configuration.agent_config import AgentConfig; print('ok')"`
Expected: `ok` (no import errors).

If a full init smoke test exists (`uv run datus --help` or equivalent), run that too.

- [ ] **Step 4: Commit**

```bash
git add conf/agent.yml
git commit -m "[Feature] Add default permission profile to agent.yml

Document the three profiles in-line so operators see the options
without hunting for docs."
```

---

## Task 8: Full regression + PR prep

- [ ] **Step 1: Run the full unit-test suite with coverage**

Run: `uv run pytest tests/unit_tests/ --cov=datus --cov-report=xml:coverage.xml --cov-fail-under=80`
Expected: PASS with ≥80% coverage.

- [ ] **Step 2: Check diff coverage against upstream/main**

Run: `uv run diff-cover coverage.xml --compare-branch=upstream/main --fail-under=80`
Expected: PASS with ≥80% diff coverage.
If it fails, run `uv run diff-cover coverage.xml --compare-branch=upstream/main --show-uncovered` to see which new lines need tests, then add focused tests.

- [ ] **Step 3: Full format and lint**

Run: `uv run ruff format . && uv run ruff check --fix .`
Expected: exit 0, zero modifications.

- [ ] **Step 4: Smoke test the CLI interactively**

```bash
uv run datus
```

In the REPL:
1. Type `/profile` — dialog should appear showing `normal` / `auto` / `dangerous` / `cancel` with `← current` next to `normal`.
2. Select `auto` — status bar should update to show `auto` in cyan.
3. Type `/profile` → `dangerous` → confirm at the second prompt. Status bar should turn `dangerous` bold red.
4. Try a write operation on a file outside the project (e.g., `~/tmp/x.txt`). Expect ASK despite Dangerous being active (PathZone EXTERNAL).
5. Switch back with `/profile` → `normal`. Verify any prior "always allow" session approvals are gone (trigger a write — should prompt again).

- [ ] **Step 5: Commit any follow-up fixes, then open PR**

```bash
git push origin HEAD
```

Open a PR with title `[Feature] Permission Profile (normal / auto / dangerous)` and body following `.github/PULL_REQUEST_TEMPLATE.md` (Why / Solution / Test Cases).

---

## Self-Review Checklist

Before handing off to execution, verify:

1. **Spec coverage** — every §2 decision maps to at least one task:
   - Decision 1 (MVP bundles) → Task 1
   - Decision 2 (config + CLI selection) → Tasks 2, 6, 7
   - Decision 3 (profile base + user rules merge) → Tasks 2, 3, 6
   - Decision 4 (three profiles) → Task 1
   - Decision 5 (Dangerous every-session confirm) → Task 6, Step 4
   - Decision 6 (MCP ASK) → Task 1 (NORMAL + AUTO rules include `mcp.*` ask)
   - Decision 7 (switch clears approvals) → Task 3
   - Decision 8 (default normal) → Task 2
   - Decision 9 (Dangerous named-delete allow) → Task 1 DANGEROUS has no overrides
   - Decision 10 (Auto DB writes ask) → Task 1 AUTO has explicit `db_tools.*` ask
   - Decision 11 (single PR, no backward compat) → Task 8

2. **No placeholders** — every step shows actual code, actual commands, actual expected output.

3. **Type consistency** — `active_profile` (string) is the same field across PermissionManager, AgentConfig, and DatusCLI. `get_profile()` returns `PermissionConfig`. `switch_profile()` signature matches its test.

4. **Trap doors** — PathZone filesystem gating is referenced but not modified (correct per spec §4.3). SQL parsing is explicitly out of scope (Task 1 `DANGEROUS` allows everything, plan §9 documents the limit).

Issues found: **none**. The plan is internally consistent and each task produces a testable, committable increment.
