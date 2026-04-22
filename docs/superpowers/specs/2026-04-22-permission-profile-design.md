# Permission Profile 设计方案

**日期:** 2026-04-22
**状态:** 草稿，待进入实施计划
**范围:** MVP（`v1`）

## 1. 背景与动机

Datus-Agent 当前的统一权限系统（`datus/tools/permission/`）通过 `PermissionRule(tool, pattern, permission)` 规则集评估权限，采用 `last-match-wins` 语义，并通过 `merge_with` 支持按节点覆盖。该基础设施在 PR #405 中引入，最初用于 Skills 鉴权，后来扩展到覆盖 native tools 和 MCP。

规则列表虽然能力强，但结构扁平：用户（以及产品本身）必须手写大量规则才能构造出一致的安全姿态。当前不存在"安全预设"的概念——要么全放开，要么全收紧，要么一条条自己写。

另外，Claude Code 把 plan-mode、auto-accept、`--dangerously-skip-permissions` 放在一个维度上，这对 Datus 是错误的形状。Datus 处理的是数据基础设施（DB、BI、调度器），安全预设应该按**用户当前做的工作类型**来选，而不是按"正在规划还是执行"来区分。

**目标：** 引入一个与 plan-mode 正交的 `Permission Profile` 维度，让用户能选择三个预设安全方案（`normal` / `auto` / `dangerous`）之一，同时允许用户自定义规则作为细粒度覆盖叠加在其上。

## 2. 已锁定的设计决策

| # | 决策 |
|---|------|
| 1 | 范围：MVP 预定义规则组合；不改规则引擎 |
| 2 | 选择 UX：`agent.yml: permissions.profile` 设默认；`/profile` CLI 命令在本 session 覆盖 |
| 3 | Profile = base 规则；用户的 `permissions.rules` 通过 `merge_with` 叠加（`last-match-wins`） |
| 4 | 三个 profile 全部交付：`normal`、`auto`、`dangerous` |
| 5 | 切换到 `dangerous` 每次 session 都需要额外二次确认（不是仅首次） |
| 6 | MCP 工具：`normal`/`auto` 下始终 `ASK`，`dangerous` 下 `allow`；用户可通过 `permissions.rules` 加白名单 |
| 7 | 切换 profile 时清空 `PermissionManager._session_approvals` |
| 8 | 默认 profile：`normal` |
| 9 | Dangerous profile 下命名删除工具（`delete_dashboard`、`delete_chart`、`delete_job`、`delete_dataset`）：`allow`（放手模式，全开） |
| 10 | Auto profile 下 DB 写操作（`execute_ddl`、`execute_write`、`transfer_query_result`、`write_query`）：每次 `ask`（MVP 无环境检测） |
| 11 | 实施：单 PR，不做向后兼容处理（现有权限 schema 无老用户依赖） |

决策已终版。后续进入 writing-plans 阶段不再回炉；如需变更必须经过用户显式重新确认。

## 3. 架构

```
┌─────────────────────────────────────────────────────────────────┐
│  agent.yml                                                       │
│  permissions:                                                    │
│    profile: auto           ← 默认 profile（可缺省）               │
│    rules: [...]            ← 用户覆盖，叠加在 profile 之上        │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ProfileRegistry（新增：datus/tools/permission/profiles.py）     │
│    NORMAL:    PermissionConfig(...)                             │
│    AUTO:      PermissionConfig(...)                             │
│    DANGEROUS: PermissionConfig(...)                             │
│    get_profile(name) -> PermissionConfig                        │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  AgentConfig.load_permissions()（改动）                          │
│    profile_name = cfg.get("profile", "normal")                  │
│    base = ProfileRegistry.get_profile(profile_name)             │
│    user_cfg = PermissionConfig.from_dict(cfg)                   │
│    effective = base.merge_with(user_cfg)                        │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PermissionManager（改动）                                       │
│    + active_profile: str        ← 当前 profile 名                │
│    + switch_profile(name)       ← 清空 session_approvals、       │
│                                     重建 effective config       │
│    （check_permission / filter_* 不变）                          │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PermissionHooks（不变）                                         │
│    - check_permission + InteractionBroker (y/a/n)               │
│    - PathZone filesystem gating                                 │
└─────────────────────────────────────────────────────────────────┘
           ▲
           │ （运行时切换）
┌─────────────────────────────────────────────────────────────────┐
│  CLI: /profile（新增命令）                                       │
│    - 通过 InteractionBroker.request(choices=[...]) 做选项 UI    │
│    - 选到 dangerous 触发二次确认                                 │
└─────────────────────────────────────────────────────────────────┘
```

**复用总览：**
- `PermissionRule` / `PermissionConfig` / `PermissionHooks` / `fs_path_policy.PathZone`：零改动
- `PermissionManager`：仅增量（新字段/新方法，无移除）
- `StatusBarState`：新增一个字段（`profile`）和对应渲染 token

## 4. Profile 规则内容

所有 profile 都采用 `default_permission` + 显式 rules 模式。下表用简写形式 `(tool_category, pattern, permission)` 表达规则；代码层面真正的来源是 `profiles.py` 里的 Python 常量。

### 4.1 Normal（`default_permission: ask`）

只读全放开，所有写操作都 ASK，命名 destructive DENY。

| 类别 | Pattern | 权限 | 用途 |
|------|---------|------|------|
| `context_search_tools` | `*` | allow | 全部知识搜索 |
| `date_parsing_tools` | `*` | allow | 日期工具 |
| `db_tools` | `read_query` / `list_*` / `describe_*` / `get_*` | allow | DB 只读 |
| `bi_tools` | `list_*` / `get_*` | allow | BI 只读 |
| `bi_tools` | `delete_*` | **deny** | 命名 destructive 永远屏蔽 |
| `semantic_tools` | `list_*` / `search_*` / `get_*` / `query_metrics` | allow | Semantic 只读 |
| `scheduler_tools` | `list_*` / `get_*` | allow | Scheduler 只读 |
| `scheduler_tools` | `delete_job` | **deny** | 命名 destructive |
| `filesystem_tools` | `read_*` / `list_*` / `directory_tree` / `search_files` | allow | FS 只读 |
| `tools` | `todo_read` | allow | Plan 只读 |
| `mcp.*` | `*` | ask | 所有 MCP 弹框 |
| `skills` | `*` | ask | 所有 skill 调用弹框 |

### 4.2 Auto（`default_permission: ask`）

继承 Normal 的全部规则，并叠加（`last-match-wins`）：

| 类别 | Pattern | 权限 | 用途 |
|------|---------|------|------|
| `filesystem_tools` | `write_file` / `edit_file` / `create_directory` / `move_file` | allow | workspace 写；`PathZone` 对 EXTERNAL 强制 ASK |
| `tools` | `todo_write` / `todo_update` | allow | Plan 写 |
| `semantic_tools` | `end_*_generation` / `validate_semantic` / `generate_*_id` | allow | 生成收尾辅助 |
| `bi_tools` | `create_*` / `update_*` / `add_*` | allow | BI 写（不含 delete_*） |
| `scheduler_tools` | `submit_*` / `update_job` / `pause_job` / `resume_job` | allow | Scheduler 写 |
| `scheduler_tools` | `trigger_*` | ask | job 触发仍弹框 |
| `db_tools` | `execute_ddl` / `execute_write` / `transfer_query_result` / `write_query` | ask | DB 写始终弹框（MVP 无环境检测） |

Auto 保留 Normal 对命名 destructive 的 `deny`（`bi_tools.delete_*`、`scheduler_tools.delete_job`）。MCP 和 skills 维持 `ask`。

### 4.3 Dangerous（`default_permission: allow`）

无需规则——所有工具都通过 `allow` 兜底，包括命名 destructive（`delete_*`、`execute_ddl` 等）。唯一剩下的闸门在 hook 层：

- `PermissionHooks._handle_filesystem_zone` 已经对 `PathZone.EXTERNAL` 路径强制 ASK，不管规则怎么判。这保证 `~/.ssh/*`、`~/.aws/credentials`、`/etc/*` 在 Dangerous 下仍然弹框。
- `PathZone.HIDDEN` 路径（`.datus/*` 内部非白名单部分）对 LLM 保持不可见（工具返回 `not found`）。

SQL 层面的 destructive 检测（比如 `execute_ddl` 里的 `DROP TABLE`）明确不在 MVP 范围内。选择 Dangerous 的用户接受这个风险。

## 5. CLI UX

### 5.1 状态栏

`StatusBarState` 新增 `profile` 字段（`str`），渲染为一个独立段，位于 `connector` 和 `model` 之间：

```
 Datus │ chat │ starrocks: starrocks │ normal │ claude-sonnet-4-6 │ 0K │ 0K/0K 0%
                                       ^^^^^^
```

按风险等级使用不同样式：

| Profile | 样式 |
|---------|------|
| `normal` | 默认/灰色 |
| `auto` | `class:status-bar.profile.auto`（青色） |
| `dangerous` | `class:status-bar.profile.dangerous`（粗体红色） |

### 5.2 `/profile` 命令

```
/profile         → 弹出选择框
/profile list    → 列出 profile 及说明；不切换
```

选择框通过 `broker.request(choices=[...])` 呈现：

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

方向键 + 回车确认。选 `cancel` 中止；选中已是当前的 profile 则 no-op 并提示。

### 5.3 Dangerous 确认（每 session）

如果用户选中 `dangerous`，弹第二层 `broker.request` 确认：

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

默认高亮 `Cancel`。用户必须主动按方向键到第二项再回车。**每次** session 从其他 profile 切进 Dangerous 都会重新触发这个弹框——切出再切回也要重新确认。

### 5.4 切换反馈

切换成功后：

```
Profile switched: auto → dangerous
Session approvals cleared (was: 3)
Effective rules: 0 base + 2 from agent.yml = 2 active
```

### 5.5 错误处理

| 场景 | 行为 |
|------|------|
| 非交互模式（API / gateway）下执行 `/profile` | 返回 `Requires interactive session` 错误 |
| broker 抛 `InteractionCancelled`（Ctrl+C） | 保留原 profile |
| `agent.yml` 的 profile 值非法 | 启动时打 warning，fallback 到 `normal` |
| `/profile <name>` 的 `name` 已是当前 profile | no-op 并提示 `Already on <name>` |
| `/profile list` | 打印表格；不改状态 |

## 6. 受影响文件

| 文件 | 改动类型 | 备注 |
|------|---------|------|
| `datus/tools/permission/profiles.py` | **新增** | `NORMAL / AUTO / DANGEROUS: PermissionConfig`、`get_profile(name)` |
| `datus/tools/permission/permission_manager.py` | 修改 | `active_profile`、`switch_profile()` |
| `datus/configuration/agent_config.py` | 修改 | 解析 `permissions.profile`，与 profile base 合并 |
| `datus/cli/repl.py` + `datus/cli/chat_commands.py` | 修改 | `/profile` 命令注册与 handler |
| `datus/cli/status_bar.py` | 修改 | `StatusBarState.profile` + 渲染 token |
| `datus/cli/styles.py`（如果存在） | 修改 | 新增 `profile.auto` / `profile.dangerous` 样式类 |
| `conf/agent.yml` | 修改 | 加入示例 `permissions.profile: normal` |
| `datus/tools/permission/permission_hooks.py` | **不变** | 现有 ASK 流程对 profile 规则透明 |
| `datus/tools/func_tool/fs_path_policy.py` | **不变** | `PathZone` EXTERNAL→ASK 已经兜底 Dangerous 边界 |

## 7. 测试策略

仅 CI 层测试（零外部依赖），遵循 Source → Test Mapping Rule。

### 7.1 新增测试

**`tests/unit_tests/tools/permission/test_profiles.py`**
- 对每个 profile 的规则数量和类别做快照
- `get_profile("normal" | "auto" | "dangerous")` 返回预期结构
- `get_profile("unknown")` 确定性地抛异常或返回 `None`

### 7.2 扩展测试

**`tests/unit_tests/tools/permission/test_permission_manager.py`**
- `switch_profile("auto")` 清空 `_session_approvals`
- `switch_profile()` 重建 effective config（profile base + 用户 rules）
- `active_profile` 反映当前选择
- 用户 rules 通过 `merge_with` 正确覆盖 profile 规则

**`tests/unit_tests/configuration/test_agent_config.py`**（或对应文件）
- 从 YAML 解析 `permissions.profile`
- 缺省 `profile` 时默认 `normal`
- 非法 profile 打 warning 并 fallback 到 `normal`
- `permissions.rules` 合并到 profile base 之上

**`tests/test_cli_commands.py`**
- `/profile` 无参数时弹选择框
- `/profile list` 打印表格但不切换
- 选中 `dangerous` 触发二次确认
- 取消二次确认保留原 profile
- 每次 session 切入 Dangerous 都触发二次确认
- 非交互模式下 `/profile` 返回可读的错误

**`tests/unit_tests/cli/test_status_bar.py`**
- `profile` 字段在 `format_plain` 和 `to_formatted_tokens` 中正确渲染
- 按 profile 风险等级应用正确的样式类
- profile 缺省时显示 `normal`

### 7.3 代表性断言

- `test_profile_normal_blocks_writes`：`normal` + `filesystem_tools.write_file` → `ASK`
- `test_profile_auto_allows_workspace_writes`：`auto` + `filesystem_tools.write_file` → `ALLOW`（EXTERNAL 由 `PathZone` 处理）
- `test_profile_dangerous_still_asks_external`：Dangerous profile + workspace 外路径 → hook 层 `ASK`
- `test_profile_user_rules_override`：Auto 下对 `execute_ddl` 设 `permission: deny` 的用户规则覆盖基线 ASK
- `test_profile_switch_clears_session_approvals`：`switch_profile` 后缓存清空
- `test_profile_unknown_falls_back`：非法名称 fallback 到 `normal`

### 7.4 不在范围

- Nightly/regression 测试：不需要。Profile 层本身不涉及 LLM 调用、真实 DB 写、真实 MCP server。
- SQL 解析 / DDL 分类：明确排除于 MVP。

## 8. 交付计划

单 PR 包含：

1. `profiles.py`，提供三个预定义 `PermissionConfig` 组合
2. `PermissionManager` 的增量改动（`active_profile`、`switch_profile`）
3. `AgentConfig` 加载器改动
4. `/profile` CLI 命令和基于 broker 的选项/确认 UI
5. 状态栏集成
6. 默认 `conf/agent.yml` 更新
7. §7 列出的全部测试
8. PR 标题：`[Feature] Permission Profile (normal / auto / dangerous)`

明确不做向后兼容——权限系统目前没有生产用户依赖，除了作者自己。

## 9. 非目标（MVP 以外）

以下内容有意推迟，是合理的 P1/P2 候选，但不在本 spec 范围：

- `tool + target + env + action` 组合式规则匹配（需要扩展 `PermissionRule` schema 并引入 SQL AST 分类器）
- `datasource.env` 字段（`dev` / `staging` / `prod`）驱动 Auto profile 下 DB 写的行为
- 从 `~/.datus/profiles/*.yml` 加载用户自定义 profile
- 与 plan-mode 的集成（正交维度；plan-mode 当前行为不变）
- SQL 层的 destructive 检测（`execute_ddl` 内 `DROP TABLE`）作为 Dangerous hard-block 列表
- 用于凭证泄露或 sandbox 逃逸的 `hard_block` 显式 denylist
- 按 profile 分 bucket 的 session approval 缓存（每个 profile 独立的授权缓存）

## 10. 未决问题

无。所有 brainstorming 中浮现的维度都已解决（参见 §2）。
