# Permission Profile Design

**Date:** 2026-04-22
**Status:** Draft, pending implementation plan
**Scope:** MVP (`v1`)

## 1. Background and Motivation

Datus-Agent today exposes a unified permission system (`datus/tools/permission/`) that evaluates `PermissionRule(tool, pattern, permission)` entries with `last-match-wins` semantics and supports per-node overrides via `merge_with`. The infrastructure was introduced in PR #405 to gate Skills, and has since been extended to cover native tools and MCP.

The rule list is powerful but flat: users (and the product itself) must hand-craft dozens of rules to get a coherent security posture. There is no notion of a "security preset" — you either allow everything, deny everything, or write bespoke rules.

Separately, Claude Code conflates plan-mode, auto-accept, and `--dangerously-skip-permissions` into a single dimension. This is the wrong shape for Datus, which handles data infrastructure (DBs, BI, schedulers) where the right security preset varies by *what kind of work* the user is doing, not by whether planning or execution is active.

**Goal:** introduce a `Permission Profile` dimension — orthogonal to plan-mode — that lets users pick one of three predefined security presets (`normal` / `auto` / `dangerous`), with user-defined rules layered on top as fine-grained overrides.

## 2. Design Decisions (Locked)

| # | Decision |
|---|---------|
| 1 | Scope: MVP predefined rule bundles; no rule-engine changes |
| 2 | Selection UX: `agent.yml: permissions.profile` sets default; `/profile` CLI command overrides for the current session |
| 3 | Profile = base rules; user `permissions.rules` layered on top via `merge_with` (`last-match-wins`) |
| 4 | Ship all three profiles: `normal`, `auto`, `dangerous` |
| 5 | Switching to `dangerous` requires an explicit second-confirmation every session (not just first-time) |
| 6 | MCP tools: always `ASK` under `normal`/`auto`, `allow` under `dangerous`; users can whitelist via `permissions.rules` |
| 7 | Profile switching clears `PermissionManager._session_approvals` |
| 8 | Default profile: `normal` |
| 9 | Dangerous-profile named-delete tools (`delete_dashboard`, `delete_chart`, `delete_job`, `delete_dataset`): `allow` (fully yolo under Dangerous) |
| 10 | Auto-profile DB writes (`execute_ddl`, `execute_write`, `transfer_query_result`, `write_query`): `ask` every call (no env detection in MVP) |
| 11 | Implementation: single PR, no backward-compatibility shim (no existing users depend on the current permission schema) |

Decisions are final; subsequent implementation-plan work should not revisit them without explicit user re-approval.

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  agent.yml                                                       │
│  permissions:                                                    │
│    profile: auto           ← default profile (optional)          │
│    rules: [...]            ← user overrides, layered on profile  │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ProfileRegistry (new: datus/tools/permission/profiles.py)      │
│    NORMAL:    PermissionConfig(...)                             │
│    AUTO:      PermissionConfig(...)                             │
│    DANGEROUS: PermissionConfig(...)                             │
│    get_profile(name) -> PermissionConfig                        │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  AgentConfig.load_permissions()  (modified)                     │
│    profile_name = cfg.get("profile", "normal")                  │
│    base = ProfileRegistry.get_profile(profile_name)             │
│    user_cfg = PermissionConfig.from_dict(cfg)                   │
│    effective = base.merge_with(user_cfg)                        │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PermissionManager  (modified)                                  │
│    + active_profile: str       ← current profile name           │
│    + switch_profile(name)      ← clears session_approvals,      │
│                                   rebuilds effective config     │
│    (check_permission / filter_* unchanged)                      │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PermissionHooks  (unchanged)                                   │
│    - check_permission + InteractionBroker (y/a/n)               │
│    - PathZone filesystem gating                                 │
└─────────────────────────────────────────────────────────────────┘
           ▲
           │ (runtime switching)
┌─────────────────────────────────────────────────────────────────┐
│  CLI: /profile  (new command)                                   │
│    - Selection UI via InteractionBroker.request(choices=[...])  │
│    - dangerous selection triggers second confirmation           │
└─────────────────────────────────────────────────────────────────┘
```

**Reuse summary:**
- `PermissionRule` / `PermissionConfig` / `PermissionHooks` / `fs_path_policy.PathZone`: no changes
- `PermissionManager`: additive only (new fields/methods, nothing removed)
- `StatusBarState`: one new field (`profile`) and corresponding render tokens

## 4. Profile Rule Content

All profiles rely on `default_permission` + explicit rules. Rules below are expressed in abbreviated form `(tool_category, pattern, permission)`; the canonical source will be Python constants in `profiles.py`.

### 4.1 Normal (`default_permission: ask`)

Read-only allow, all writes ASK, named-destructive DENY.

| Category | Pattern | Permission | Purpose |
|----------|---------|-----------|---------|
| `context_search_tools` | `*` | allow | All knowledge search |
| `date_parsing_tools` | `*` | allow | Date utilities |
| `db_tools` | `read_query` / `list_*` / `describe_*` / `get_*` | allow | DB read |
| `bi_tools` | `list_*` / `get_*` | allow | BI read |
| `bi_tools` | `delete_*` | **deny** | Named destructive always blocked |
| `semantic_tools` | `list_*` / `search_*` / `get_*` / `query_metrics` | allow | Semantic read |
| `scheduler_tools` | `list_*` / `get_*` | allow | Scheduler read |
| `scheduler_tools` | `delete_job` | **deny** | Named destructive |
| `filesystem_tools` | `read_*` / `list_*` / `directory_tree` / `search_files` | allow | FS read |
| `tools` | `todo_read` | allow | Plan read |
| `mcp.*` | `*` | ask | All MCP prompts |
| `skills` | `*` | ask | All skill invocations prompt |

### 4.2 Auto (`default_permission: ask`)

Inherits all Normal rules, adds (via `last-match-wins`):

| Category | Pattern | Permission | Purpose |
|----------|---------|-----------|---------|
| `filesystem_tools` | `write_file` / `edit_file` / `create_directory` / `move_file` | allow | Workspace writes; `PathZone` forces ASK for EXTERNAL paths |
| `tools` | `todo_write` / `todo_update` | allow | Plan writes |
| `semantic_tools` | `end_*_generation` / `validate_semantic` / `generate_*_id` | allow | Generation finalize helpers |
| `bi_tools` | `create_*` / `update_*` / `add_*` | allow | BI write (excluding delete_*) |
| `scheduler_tools` | `submit_*` / `update_job` / `pause_job` / `resume_job` | allow | Scheduler write |
| `scheduler_tools` | `trigger_*` | ask | Job trigger remains prompted |
| `db_tools` | `execute_ddl` / `execute_write` / `transfer_query_result` / `write_query` | ask | DB writes always prompt (no env detection in MVP) |

Auto preserves Normal's `deny` on named destructives (`bi_tools.delete_*`, `scheduler_tools.delete_job`). MCP and skills stay `ask`.

### 4.3 Dangerous (`default_permission: allow`)

No rules needed — everything falls through to `allow`, including named-destructive tools (`delete_*`, `execute_ddl`, etc.). The only remaining gate is at the hook layer:

- `PermissionHooks._handle_filesystem_zone` already forces ASK for `PathZone.EXTERNAL` paths, regardless of rule verdict. This keeps `~/.ssh/*`, `~/.aws/credentials`, `/etc/*` prompting even under Dangerous.
- `PathZone.HIDDEN` paths (internal `.datus/*` except whitelisted) remain invisible to the LLM (tool returns `not found`).

SQL-level destructive detection (e.g., `DROP TABLE` inside `execute_ddl`) is explicitly out of scope for MVP. Users selecting Dangerous accept this risk.

## 5. CLI UX

### 5.1 Status Bar

`StatusBarState` gains a `profile` field (`str`). Rendered as a dedicated segment between `connector` and `model`:

```
 Datus │ chat │ starrocks: starrocks │ normal │ claude-sonnet-4-6 │ 0K │ 0K/0K 0%
                                       ^^^^^^
```

Style classes by risk:

| Profile | Style |
|---------|-------|
| `normal` | Default / muted |
| `auto` | `class:status-bar.profile.auto` (cyan) |
| `dangerous` | `class:status-bar.profile.dangerous` (bold red) |

### 5.2 `/profile` Command

```
/profile         → opens selection dialog
/profile list    → lists profiles with descriptions; does not switch
```

Selection dialog (via `broker.request(choices=[...])`):

```
┌─────────────────────────────────────────────────────────┐
│ Select Permission Profile                               │
│                                                         │
│ Current: auto                                           │
│                                                         │
│ ▸ normal      Read-only + confirm every write          │
│   auto        Workspace writes auto; DB/MCP still ask  │
│   dangerous   Nearly all writes auto (see warning)     │
│   cancel      Keep current profile                     │
└─────────────────────────────────────────────────────────┘
```

Arrow keys + Enter. Selecting `cancel` aborts; selecting the same as current is a no-op with an informational message.

### 5.3 Dangerous Confirmation (every session)

If the user selects `dangerous`, a second `broker.request` confirmation is shown:

```
┌─────────────────────────────────────────────────────────────┐
│ DANGEROUS PROFILE — Explicit Confirmation Required          │
│                                                             │
│ Switching to Dangerous will auto-execute:                   │
│   • All DB writes (including DDL, DELETE)                   │
│   • All BI/Scheduler writes (including deletes)             │
│   • All MCP tools                                           │
│   • All skills                                              │
│                                                             │
│ Still protected: writes outside workspace require ASK;      │
│ ~/.datus internals remain hidden.                           │
│                                                             │
│ ▸ Cancel (stay on current profile)                          │
│   Enable Dangerous for this session                         │
└─────────────────────────────────────────────────────────────┘
```

Default highlight is `Cancel`. The user must actively arrow-down and Enter to confirm. This prompt shows every time the session transitions into Dangerous — switching out and back requires re-confirmation.

### 5.4 Switch Feedback

After a successful switch:

```
Profile switched: auto → dangerous
Session approvals cleared (was: 3)
Effective rules: 0 base + 2 from agent.yml = 2 active
```

### 5.5 Error Handling

| Scenario | Behavior |
|----------|---------|
| `/profile` invoked in non-interactive mode (API / gateway) | Returns `Requires interactive session` error |
| Broker raises `InteractionCancelled` (Ctrl+C mid-dialog) | Original profile retained |
| `agent.yml` profile value invalid | Warning logged at startup; fallback to `normal` |
| `/profile <name>` where `name` already active | No-op with `Already on <name>` |
| `/profile list` | Prints table; no state change |

## 6. Affected Files

| File | Change | Notes |
|------|--------|-------|
| `datus/tools/permission/profiles.py` | **new** | `NORMAL / AUTO / DANGEROUS: PermissionConfig`, `get_profile(name)` |
| `datus/tools/permission/permission_manager.py` | modify | `active_profile`, `switch_profile()` |
| `datus/configuration/agent_config.py` | modify | Parse `permissions.profile`, merge with profile base |
| `datus/cli/repl.py` + `datus/cli/chat_commands.py` | modify | `/profile` registration and handler |
| `datus/cli/status_bar.py` | modify | `StatusBarState.profile` + render tokens |
| `datus/cli/styles.py` (if present) | modify | New style classes for `profile.auto` / `profile.dangerous` |
| `conf/agent.yml` | modify | Example `permissions.profile: normal` entry |
| `datus/tools/permission/permission_hooks.py` | **no change** | Existing ASK flow handles profile rules transparently |
| `datus/tools/func_tool/fs_path_policy.py` | **no change** | `PathZone` EXTERNAL-ASK already enforces Dangerous boundary |

## 7. Testing Strategy

CI-tier tests only (zero external deps), following the Source → Test Mapping Rule.

### 7.1 New Tests

**`tests/unit_tests/tools/permission/test_profiles.py`**
- Snapshot each profile's rule count and categories
- `get_profile("normal" | "auto" | "dangerous")` returns expected structure
- `get_profile("unknown")` raises or returns `None` deterministically

### 7.2 Extended Tests

**`tests/unit_tests/tools/permission/test_permission_manager.py`**
- `switch_profile("auto")` clears `_session_approvals`
- `switch_profile()` rebuilds effective config (profile base + user rules)
- `active_profile` reflects current selection
- User rules correctly override profile rules via `merge_with`

**`tests/unit_tests/configuration/test_agent_config.py`** (or equivalent)
- `permissions.profile` parsed from YAML
- Missing `profile` defaults to `normal`
- Invalid profile logs warning and falls back to `normal`
- `permissions.rules` merged on top of profile base

**`tests/test_cli_commands.py`**
- `/profile` with no args opens selection dialog
- `/profile list` prints table without switching
- Selecting `dangerous` triggers second confirmation
- Cancelling second confirmation retains original profile
- Dangerous confirmation runs every time the session transitions into Dangerous
- Non-interactive mode rejects `/profile` with informative error

**`tests/unit_tests/cli/test_status_bar.py`**
- `profile` field renders in `format_plain` and `to_formatted_tokens`
- Style classes applied per profile risk level
- Missing profile defaults to `normal` in display

### 7.3 Representative Assertions

- `test_profile_normal_blocks_writes` — `normal` + `filesystem_tools.write_file` → `ASK`
- `test_profile_auto_allows_workspace_writes` — `auto` + `filesystem_tools.write_file` → `ALLOW` (and `PathZone` handles EXTERNAL)
- `test_profile_dangerous_still_asks_external` — Dangerous profile + path outside workspace → `ASK` at hook layer
- `test_profile_user_rules_override` — user rule with `permission: deny` on `execute_ddl` under Auto wins over base Auto ASK
- `test_profile_switch_clears_session_approvals` — cache emptied after `switch_profile`
- `test_profile_unknown_falls_back` — invalid name yields `normal`

### 7.4 Out of Scope

- Nightly/regression tests: none required. No LLM calls, no real DB writes, no MCP servers involved in the profile layer itself.
- SQL parsing / DDL classification: explicitly excluded from MVP.

## 8. Delivery Plan

Single PR containing:

1. `profiles.py` with the three predefined `PermissionConfig` bundles
2. `PermissionManager` additions (`active_profile`, `switch_profile`)
3. `AgentConfig` loader modifications
4. `/profile` CLI command and broker-driven selection / confirmation UI
5. Status bar integration
6. Default `conf/agent.yml` update
7. All tests listed in §7
8. PR title: `[Feature] Permission Profile (normal / auto / dangerous)`

Backward compatibility is explicitly not a concern — the permission system has no existing production users beyond the authors.

## 9. Non-Goals (MVP)

The following are intentionally deferred; they are valid P1/P2 candidates but are out of scope for this spec:

- `tool + target + env + action` composite rule matching (would require extending `PermissionRule` schema and a SQL AST classifier)
- `datasource.env` field (`dev` / `staging` / `prod`) driving Auto-profile DB-write behavior
- User-defined custom profiles loaded from `~/.datus/profiles/*.yml`
- Integration with plan-mode behavior (orthogonal dimension; current plan-mode behavior is unchanged)
- SQL-level destructive detection (`DROP TABLE` inside `execute_ddl`) for Dangerous hard-block list
- `hard_block` explicit denylist for credential exfiltration or sandbox escape
- Scope-specific session approval bucketing (one approval cache per profile)

## 10. Open Questions

None. All dimensions surfaced during brainstorming have been resolved (see §2).
