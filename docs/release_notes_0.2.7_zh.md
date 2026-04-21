# Datus-Agent 0.2.7 Release Note（中文版）

> 面向用户的发版说明。覆盖范围：自 `v0.2.6` (2026-03-21) 以来由 Leo / Felix.Liu 负责落地的改动。
> 发版分支：`fstool-fix`（合入 `main` 后打 `v0.2.7`）。

---

## 一、本次版本关键词

1. **新增独立 REST API 服务（`datus-api`）** —— 从 Streamlit 后端抽出，可独立部署。
2. **CLI Print Mode + Proxy Tools** —— 支持 stdin/stdout 非交互式调用，便于外部 orchestrator 接管工具执行。
3. **Gateway 网关 Daemon 模式 + 流式输出** —— IM 场景（飞书）端到端打通 streaming。
4. **Auto Memory（MEMORY.md）** —— Agentic Node 获得跨会话的持久化记忆能力。
5. **文件系统工具重构** —— 统一到 `project_root` + `PathZone` 访问策略；`filesystem_strict` 升级为一等 AgentConfig 字段。
6. **Project-level 配置 + Feedback Agent + Session 作用域隔离** —— 多项目、多 Agent 协作的基础能力补齐。

---

## 二、New Features

### 2.1 REST API 服务与 Chat API

| 能力 | PR | 说明 |
|---|---|---|
| 新增 `datus-api` CLI 入口与 FastAPI 模块 | [#520](https://github.com/Datus-ai/Datus-agent/pull/520) | 独立的 REST API，包含 agents / chat / database / explorer / KB / MCP 路由；消除对 `datus_backend` 的 ORM 依赖；可独立部署 |
| Datus Chat 模块 | [#554](https://github.com/Datus-ai/Datus-agent/pull/554) | 新的 chat 子模块，作为 Web/IM 调用 Agentic 能力的统一入口 |
| Chat API Streaming 输出 | [#555](https://github.com/Datus-ai/Datus-agent/pull/555) | 新增 `--stream` CLI 标志与按请求级的 `stream_response`，逐 token 输出 thinking delta；客户端需处理 `create/append/update` 三类 message 事件 |
| `ask_user_tool` 多选支持 | [#553](https://github.com/Datus-ai/Datus-agent/pull/553) | SSE `UserInteractionInput.input` 由 `List[str]` 升级为 `List[List[str]]` 以承载 multiSelect |
| API Server 参数透传 | [#538](https://github.com/Datus-ai/Datus-agent/pull/538) | 新增 `--source` / `--interactive` / `--no-interactive`，且支持 `ChatInput.interactive` / `ChatInput.source` 按请求覆盖 |
| Web API 文档 | [#530](https://github.com/Datus-ai/Datus-agent/pull/530) | 完整 REST API 使用手册 |

### 2.2 CLI Print Mode（stdin/stdout）

| 能力 | PR | 说明 |
|---|---|---|
| `--print` 非交互模式 | [#489](https://github.com/Datus-ai/Datus-agent/pull/489) | JSON Lines 格式通过 stdin/stdout 流式通信；新增 `--resume` 恢复既有 session（要求同时指定 `--print`）；抽离 `node_factory` / `action_content_builder` |
| Proxy Tools for Print Mode | [#501](https://github.com/Datus-ai/Datus-agent/pull/501) | `--proxy_tools` 把指定工具替换为 channel-based 代理：Agent 发出 `call-tool` 事件到 stdout，外部进程通过 stdin 回传结果；适合 VSCode 插件、Web 前端接管工具执行 |

### 2.3 Datus Gateway IM 网关

| 能力 | PR | 说明 |
|---|---|---|
| Daemon 模式 | [#559](https://github.com/Datus-ai/Datus-agent/pull/559) | 新增 `--daemon`、`--action start/stop/restart/status`、`--pid-file`、`--daemon-log-file`；POSIX-only 调用加平台守卫；`--log-level` 在 daemon 模式下生效 |
| Gateway Streaming 输出 | [#562](https://github.com/Datus-ai/Datus-agent/pull/562) | 网关向 IM 下行支持逐 token 增量输出 |
| 飞书 Streaming 支持 | [#565](https://github.com/Datus-ai/Datus-agent/pull/565) | 飞书 adapter 端接 Gateway streaming |
| Gateway/API Feedback Action | [#593](https://github.com/Datus-ai/Datus-agent/pull/593) | 修复 ContextVar 跨线程传递导致的 `project_name` 空串问题，IM 场景也能正确解析 project |

### 2.4 Auto Memory

- [#498](https://github.com/Datus-ai/Datus-agent/pull/498) **MEMORY.md 持久化**：Agentic Node 的系统提示词会自动注入 `MEMORY.md` 内容（≤200 行为 L1 主文件；超过的部分由模型按需读取 L2 子文件），不新增任何工具，复用现有 `FilesystemFuncTool`。
- 适用节点：`chat` + 自定义 subagent（`GenSQL` / `GenReport` 强制保留 filesystem tools）。
- 增强：空 `MEMORY.md` 场景下显式下发 `"empty -- no memories saved yet"` 占位符，避免模型误判注入失败（[#595](https://github.com/Datus-ai/Datus-agent/pull/595)）。

### 2.5 其他用户可见 Feature

- [#579](https://github.com/Datus-ai/Datus-agent/pull/579) **Feedback Agent**：新增反馈 Agent 节点。
- [#583](https://github.com/Datus-ai/Datus-agent/pull/583) **CLI 底部栏 UX 重设计**。
- [#575](https://github.com/Datus-ai/Datus-agent/pull/575) **SubAgentTaskTool 通用化**：`SubAgentConfig` / `AgenticNode` 新增 `subagents` 字段；`chat` 默认 `'*'`，其他节点（GenSQL、GenReport）默认 `'explore'`；强制 2 级深度。
- [#592](https://github.com/Datus-ai/Datus-agent/pull/592) **Project-level 配置**。
- [#523](https://github.com/Datus-ai/Datus-agent/pull/523) **`--session-scope` 参数**：session 目录隔离，支持按 scope 划分会话存储。

---

## 三、Refactor & Breaking Changes（升级重点）

### 3.1 文件系统结构与工具重构（重点）

> 涉及 [#578](https://github.com/Datus-ai/Datus-agent/pull/578)、[#588](https://github.com/Datus-ai/Datus-agent/pull/588)、[#561](https://github.com/Datus-ai/Datus-agent/pull/561)、[#587](https://github.com/Datus-ai/Datus-agent/pull/587)；以及 `fstool-fix` 分支上 `d6c610bf`、`ddf42ba8`。

**变更摘要**

1. **KB 内容锚定规则变更**
   - 旧：`knowledge_base_home` 可任意指定；路径内包含 `{current_database}` 层级。
   - 新：KB 内容**强制**锚定到 `{project_root}/subject/{semantic_models|sql_summaries|ext_knowledge|metrics}/`；不再有 `{db}/` 子目录；`agent.yml` 里 `agent.knowledge_base_home` 字段被**静默忽略**。
2. **Global 存储按 project 分片**
   - `~/.datus/sessions/{project_name}/{session_id}.db`
   - `~/.datus/data/{project_name}/datus_db/`
   - `project_name` 规则：`os.getcwd().replace("/", "-").lstrip("-")`，空路径回落 `_root`，超长路径尾部加 7 位 md5。
3. **FilesystemFuncTool 工具集精简**（[#561](https://github.com/Datus-ai/Datus-agent/pull/561)）
   - `read_file` 新增 `offset` / `limit`。
   - `edit_file` 简化为单对 `old_string` / `new_string`（带唯一性校验）。
   - 新增 `glob` 与 `grep`；移除 `read_multiple_files` / `create_directory` / `list_directory` / `directory_tree` / `move_file` / `search_files`。
4. **访问区分级**（[#588](https://github.com/Datus-ai/Datus-agent/pull/588)）：`PathZone` = `INTERNAL` / `WHITELIST`（`~/.datus/skills`、`.datus/skills`、`.datus/memory/{current_node}`）/ `HIDDEN`（`.datus` 其他内容）/ `EXTERNAL`（项目根以外）。
5. **`filesystem_strict` 一等字段**（`d6c610bf`）
   - `agent.yml` 新增 `agent.filesystem.strict: true/false`。
   - CLI 互斥三态：`--filesystem-strict` / `--no-filesystem-strict` / 两者均不传（保留 YAML 值）。
   - API（`chat_task_manager`）与 Gateway（`main._run_gateway`）在 strict 模式下会硬拒绝 `EXTERNAL` 路径，不再向权限 broker 发交互 prompt，避免无人值守场景挂起。

### 3.2 Tool Dispatch API 重写

- [#569](https://github.com/Datus-ai/Datus-agent/pull/569) 用 `POST /api/v1/tools/{tool_name}` 直连 `ContextSearchTools` 注册表，**移除旧 route** `execute_tool` 与一组陈旧 CLI 模型（`SchemaLinkingToolInput` / `SearchMetricsToolInput` 等）。

### 3.3 Session & 观测

- [#547](https://github.com/Datus-ai/Datus-agent/pull/547) SSE 事件新增 `depth` / `parent_action_id` / `subagent-complete`，前端可构建层级活动树。
- [#544](https://github.com/Datus-ai/Datus-agent/pull/544) `session_total_tokens` 语义调整为 **最后一次模型调用的 input_tokens**（真实上下文占用），而非累加；CLI 展示与 SSE `end` 事件新增 `requests`（LLM 调用次数）。
- [#594](https://github.com/Datus-ai/Datus-agent/pull/594) `.sessions`/`.resume` 与 `GET /api/v1/chat/sessions` 按当前 agent 过滤；API 新增 `subagent_id` 可选参数。

---

## 四、Bug Fixes

- [#570](https://github.com/Datus-ai/Datus-agent/pull/570) streaming delta chunk 不再 `strip()`，保留 token 之间的空白（飞书下 `"Hi" + " again!"` 不再变成 `"Hiagain!"`）。
- [#568](https://github.com/Datus-ai/Datus-agent/pull/568) 群聊中 bot 不再回复未参与过的 thread。
- [#577](https://github.com/Datus-ai/Datus-agent/pull/577) session compact 结果持久化修复。
- [#548](https://github.com/Datus-ai/Datus-agent/pull/548) compact API 不再恒返回 `success=false`（补齐 `_get_or_create_session`）。
- [#533](https://github.com/Datus-ai/Datus-agent/pull/533) 工具结果以 JSON mode 序列化，保留 session history 中的结构（修复 `datetime/Decimal/UUID` 被 `str(dict)` 破坏）。
- [#531](https://github.com/Datus-ai/Datus-agent/pull/531) 任务阻塞时主动下发 ping，避免 SSE 连接被中间层掐断。
- [#529](https://github.com/Datus-ai/Datus-agent/pull/529) 自定义 subagent 的 `node_name` 从 `agentic_nodes` key 正确解析，`scoped_context` 不再为空。
- [#516](https://github.com/Datus-ai/Datus-agent/pull/516) `filesystem_tools` 不再被错误代理到 `gen_semantic_model`、`gen_metrics`、`gen_sql_summary`、`gen_ext_knowledge`。
- [#505](https://github.com/Datus-ai/Datus-agent/pull/505) 加载历史 session 使用真实 id。
- [#587](https://github.com/Datus-ai/Datus-agent/pull/587) 移除 KB 生成 prompt 里过时的 `{current_database}` 目录层级。

---

## 五、从 0.2.6 升级指南

### 5.1 升级命令

```bash
uv sync
# 或 pip 安装方式
pip install -U datus-agent
```

### 5.2 必检项（Breaking / 行为变化）

| 项 | 0.2.6 行为 | 0.2.7 行为 | 动作 |
|---|---|---|---|
| `agent.knowledge_base_home` | 生效 | **静默忽略** | 删除该配置；把本地已生成的 KB 内容迁移至 `./subject/` |
| KB 目录结构 | `subject/{kind}/{db}/...` | `subject/{kind}/...`（无 `{db}`） | 执行下方迁移脚本或重新 bootstrap |
| `~/.datus/sessions/` | 扁平 | 按 `{project_name}` 分片 | 旧 session 文件移动到 `~/.datus/sessions/{project_name}/` 下即可复用 |
| `~/.datus/data/datus_db/` | 全局 | `~/.datus/data/{project_name}/datus_db/` | 同上 |
| FilesystemFuncTool API | 包含 `list_directory`/`directory_tree`/`search_files` 等 | 已移除，改用 `glob` / `grep` | 如你自己的插件/脚本调用了旧工具名，迁移至新工具 |
| `execute_tool` REST API | `GET/POST /api/v1/tools/{tool_name}` 旧实现 | 由 `ToolService` 直连 | 重新对齐请求 body 为工具的原生参数 |
| `session_total_tokens` 语义 | 累加 input_tokens | 最后一次调用的 input_tokens | 监控面板/报表需要重新校准 |
| `UserInteractionInput.input` | `List[str]` | `List[List[str]]` | API 客户端改造 payload |

### 5.3 KB 目录迁移脚本（参考）

```bash
# 0.2.6 → 0.2.7：去掉多余的 {current_database} 层
cd ./subject
for kind in semantic_models sql_summaries ext_knowledge metrics; do
  [ -d "$kind" ] || continue
  find "$kind" -mindepth 2 -maxdepth 2 -type d | while read db_dir; do
    mv "$db_dir"/* "$kind/" 2>/dev/null && rmdir "$db_dir"
  done
done
```

### 5.4 新能力快速体验

```bash
# 启动独立 API
datus-api server --host 0.0.0.0 --port 8080 --stream --interactive

# CLI print mode + proxy tools
datus chat --print --proxy_tools query_sql --resume <session_id>

# Datus Gateway 网关以 daemon 启动
datus-gateway --daemon --action start --pid-file ~/.datus/run/gateway.pid

# 开启 filesystem strict（无人值守安全）
datus chat --filesystem-strict
```

---

## 六、贡献者（本次版本 Leo / Felix.Liu 部分）

共 39 个提交，含 15 项 Feature / 4 项 Refactor / 8 项 Enhancement / 10 项 Bug Fix / 1 项 Doc。
