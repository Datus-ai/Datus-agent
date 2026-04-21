# Feedback Agent 工作流程与 Gateway 触发方式

## 概述

Feedback Agent 是 Datus-Agent 中的一个**元代理（meta-agent）**，不直接处理用户的业务请求，而是对已经发生的会话做事后分析：识别其中值得沉淀的知识，并委托给对应的 `gen_*` 子代理归档，以及更新调用方（caller node）的 `MEMORY.md`。

- 节点实现：`datus/agent/node/feedback_agentic_node.py` (`FeedbackAgenticNode`)
- 节点类型：`NodeType.TYPE_FEEDBACK`
- 系统提示词：`datus/prompts/prompt_templates/feedback_system_1.0.j2`
- 默认子代理：`gen_ext_knowledge, gen_sql_summary, gen_metrics, gen_skill`（可由 `agent.yml` 的 `agentic_nodes.feedback.subagents` 覆盖）
- 对用户隐藏：在 `datus/utils/constants.py` 的 `HIDDEN_SYS_SUB_AGENTS = {"feedback"}` 中声明，不出现在 autocomplete / `.agent` 选择器中，但仍可通过显式路径调用。

## 工作流程

### 1. 接管源会话（session takeover）

`FeedbackNodeInput.source_session_id` 指向一个已经发生过对话的会话。`execute_stream` 启动时：

1. 通过 `SessionManager.copy_session(source_session_id, target_node_name="feedback")` 复制源会话消息到一个新的 feedback 会话。
2. 替换 system prompt 为 feedback 专用模板，使 LLM 以"会话复盘分析师"的身份重新阅读整段历史。
3. Feedback 本身**不维护自己的 memory**——系统提示词中通过 `memory_node_name_override=self._resolve_caller_node_name()` 将调用方节点（通常是 `chat` 或 `gen_sql`）的 `MEMORY.md` 注入到上下文。

### 2. 工具集合装配（`setup_tools`）

- **`sub_agent_task_tool`**：`task(type=..., prompt=..., description=...)`，用于委托 `gen_ext_knowledge` / `gen_sql_summary` / `gen_metrics` / `gen_skill` 归档。
- **`filesystem_func_tool`**：以 caller node 为根设置白名单，保证可写 `.datus/memory/{caller}/**`（而不是 feedback 自己的路径）。caller 变更时会通过 setter 重建白名单，避免调用方 memory 路径被分类成 HIDDEN。
- **`ask_user_tool`**：仅在 `execution_mode="interactive"` 时挂载。

### 3. LLM 分析与归档

系统提示词（`feedback_system_1.0.j2`）要求 LLM：

1. **Review**：通读整个会话历史。
2. **Identify**：按五类标记候选——Business Knowledge / SQL Patterns / Metrics / Skills / Memory。
3. **Filter**：只保留非平凡、可复用、具体、已验证的信息。
4. **Archive**：对每个入选条目调用对应的 `task()` 子代理，或直接 `write_file` / `edit_file` 更新 caller 的 `MEMORY.md`。
5. **Summarize**：输出归档了什么、跳过了什么及原因。

### 4. 结果收敛

`_extract_storage_info` 遍历本次 run 的 actions，统计成功的 `task` 工具调用，按 `gen_*` 前缀聚合得到 `items_saved` 和 `storage_summary`，最终封装为 `FeedbackNodeResult`（含 `response` / `items_saved` / `storage_summary` / `tokens_used`）。

## 在 Gateway（IM 网关）中的触发方式

Gateway 对 feedback agent 的触发**完全由消息的 emoji reaction 驱动**——用户不需要（也不能）在聊天里打 `/feedback`。入口在 `datus/gateway/bridge.py` 的 `ChannelBridge`。

### 触发链路

```
IM 平台 reaction 事件
   └── ChannelAdapter.dispatch_reaction(event)
         └── ChannelBridge.handle_reaction(event, adapter)
               ├── 从 _bot_message_map 取出 bot 消息对应的 session_id / 原始文本
               ├── 用 _reacted_bot_messages 做幂等去重（每条 bot 消息仅触发一次）
               ├── build_reaction_feedback_prompt(reaction_emoji, reference_msg)
               │     → '[The user reacted to this message "..." with [thumbsup]] ...'
               └── ChatTaskManager.start_chat(
                       request=StreamChatInput(
                           message=prompt,
                           session_id=None,             # 新建 feedback 会话
                           source_session_id=<原会话>,   # 供 feedback 节点复制接管
                           stream_response=False,
                       ),
                       sub_agent_id="feedback",
                   )
```

### 关键机制

1. **bot 消息追踪**：`handle_message` 每次成功回复用户时，会调用 `_track_bot_message(bot_msg_id, session_id, msg, text)`，把 bot 消息 ID 与原会话上下文（`session_id`、`channel_id`、`conversation_id`、`sender_id`、累积 `text`）写入 `_bot_message_map`（LRU，容量 10000）。流式回复的多片段会按同一 `bot_msg_id` 拼接成完整 reference。
2. **一次性反馈**：`_reacted_bot_messages` 记录已触发过 feedback 的 bot 消息 id，即使用户后续再加/换 emoji 也不会重复触发。
3. **Prompt 构造**：`datus/utils/feedback_prompt.py::build_reaction_feedback_prompt` 生成标准格式 `'[The user reacted to this message "{ref}" with [{emoji}]] {optional reaction_msg}'`，其中 `reference_msg` 默认截断到 500 字符。
4. **静默归档**：feedback 的消费循环 `async for _ in self._task_manager.consume_events(task): pass` 仅消耗事件而不回发任何消息到聊天线程——用户只看到自己表情回应的那个消息，背后 feedback agent 已完成归档。
5. **会话隔离**：`source_session_id` 指向原业务会话；feedback 节点内部通过 `SessionManager.copy_session(...)` 开出新的 feedback 会话用于分析，不污染原始会话。

### 用户视角的操作

用户只需在 Slack / 飞书里**对 bot 回复的某条消息加一个 emoji 反应**（例如 👍 / 👎），即触发 feedback agent 对该消息所在会话的复盘与沉淀。无需任何斜杠命令。

## 在 CLI / API 中的触发方式（补充）

Gateway 之外，feedback 节点还能通过以下方式调用：

- **CLI 子代理切换**：Agent 节点切换时，`repl.py` 会将 `prev_node_name` 赋给新节点的 `caller_node_name`，使 feedback 能把 memory 正确写回调用方路径。对用户隐藏但可显式路由。
- **Chat API**：任何经由 `ChatTaskManager.start_chat(..., sub_agent_id="feedback")` 的调用都等价于 Gateway 的触发语义（需要在 `StreamChatInput` 中提供 `source_session_id`）。

## 关键文件索引

| 角色 | 路径 |
|------|------|
| 节点实现 | `datus/agent/node/feedback_agentic_node.py` |
| 节点类型常量 | `datus/configuration/node_type.py` (`TYPE_FEEDBACK`) |
| 节点工厂注册 | `datus/agent/node/node.py` (`Node.new_instance`) |
| 系统提示词模板 | `datus/prompts/prompt_templates/feedback_system_1.0.j2` |
| Reaction → Prompt 构造 | `datus/utils/feedback_prompt.py` |
| Gateway 触发入口 | `datus/gateway/bridge.py` (`handle_reaction`, `_track_bot_message`) |
| Reaction 事件模型 | `datus/gateway/models.py` (`ReactionEvent`) |
| 对外隐藏标记 | `datus/utils/constants.py` (`HIDDEN_SYS_SUB_AGENTS`) |
| 输入/输出 schema | `datus/schemas/feedback_agentic_node_models.py` |
| 单测 | `tests/unit_tests/agent/node/test_feedback_agentic_node.py`, `tests/unit_tests/gateway/test_bridge.py` |
