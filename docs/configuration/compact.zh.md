# 会话 Compact

Datus 内置两种 compact 策略，协同工作以保持 LLM 会话在模型上下文窗口内：

| 模式  | 驱动方     | 触发场景                                | 成本                          | 结果                                                            |
| ----- | ---------- | --------------------------------------- | ----------------------------- | --------------------------------------------------------------- |
| Major | LLM        | 历史接近上下文上限（默认 ≥ 90%）        | 一次 LLM 调用（`max_turns=1`）| 整段会话被结构化的 10 段摘要替换                                |
| Minor | 纯规则     | 滚动窗口积累足够 + cache 过期 / token 比例 | 数次磁盘写入                  | 长 tool I/O 落盘归档；原始短消息保持不动                        |

两者统一由 `AgenticNode.compact()` 编排——这是暴露给 CLI、SDK
`RunHooks`、以及模型 overflow 兜底的**唯一**公开入口。

## 配置

在 `agent.yml` 增加 `agent.compact` 段。所有字段可选，缺省按下面的默认值生效。

```yaml
agent:
  compact:
    major:
      enabled: true
      token_threshold: 0.9           # 占用 >= 上下文长度的 90% → 触发 major
    minor:
      enabled: true
      # 滚动窗口（按 tool-call 轮次计）
      window_size: 10                # 保留最近 N 轮；压缩 [_compacted_until, T-N)
      # 触发条件（OR）
      cache_ttl_seconds: 270         # Anthropic 临时 cache TTL ~5 分钟，留出 safety margin
      token_threshold: 0.6           # 历史 token / 上下文长度超过此值即触发
      # 归档行为
      archive_threshold: 1000        # arguments/output >= 此字符数即落盘
      archive_preview_chars: 200     # 占位符里保留的预览字符数
      archive_error_preview_chars: 1000  # 错误 output 预览更长
      archive_dir: null              # null → path_manager.session_data_dir(session_id)
```

### 参数说明

* `major.token_threshold` 同时是 auto 调度器的硬上限：一旦历史越过此值，下一次入口（CLI、user turn 或 tool-end hook）会**强制** major，忽略任何 minor 信号。
* `minor.window_size` 同时控制两件事：最近多少轮 tool-call 不被压缩；下一次 minor 之前**至少**需要积累多少新历史。用同一个参数控制两端可以让滚动窗口保持自一致。
* `minor.cache_ttl_seconds` 建议略低于模型提供商的 prompt cache TTL（Anthropic 临时 cache ≈ 5 分钟）。若距上次 LLM 调用超过此值，prefix cache 已经失效——此时改写历史的边际成本为零，正是免费的 compact 窗口。
* `minor.token_threshold` 是回压触发：即使 cache 还热，只要历史占用超过此比例，仍然压缩，以避免靠近 major 阈值。
* `minor.archive_threshold` 是唯一的**落盘阈值**；不存在 per-tool 白名单。长 `write_file` 内容、大 `read_query` 输出、子代理 `task` 返回，统统走同一个阈值判断。

## 何时触发？

四个入口，全部通过 `AgenticNode.compact()`：

1. **CLI `/compact`** — 同步运行 `compact(mode="major")`。minor 不开放给手动触发，因为它由条件驱动，手动调用要么是 noop，要么和 hook 自动排程重复。
2. **每轮 user turn 之前** — `execute_stream()` 顶部调用 `compact(mode="auto", reason="pre_user_turn")`。占用超过 `major.token_threshold` 走 major；滚动窗口条件满足走 minor；否则 noop。
3. **多轮 tool-call 内部** — `CompactHook`（Agents SDK 的 `RunHooks`）在每次成功 tool 调用后触发 `on_tool_end`。它会增加 `_tool_call_turn_count`，调用同一个 auto 调度器，然后：
   * 选中 major → **阻塞** Runner 循环直到摘要落盘（不阻塞下一轮还会 overflow）；
   * 选中 minor → 用 `asyncio.create_task` 异步排程，避免 SDK 循环卡在磁盘 I/O 上。
4. **模型 overflow 兜底** — `openai_compatible` 捕获 `MaxTurnsExceeded` 时，如果 caller 提供了 `compact_callback`（即 node 的 `compact` 方法），就调用 `compact_callback(mode="major", reason="overflow")` 并把这次 run **重试一次**。单次重试限制防止 compact 本身没能释放足够空间导致的死循环。

调度器（`_decide_compact_mode`）按以下优先级判断：

1. `_history_token_ratio_sync() >= major.token_threshold` → **major**
2. `_should_minor_compact()` 返回非 None → **minor**
3. 其他 → **noop**

`_should_minor_compact()` 要求满足滚动窗口前置条件，再叠加一个 OR 触发：

* `cache_expired` — `_seconds_since_last_llm_call() > cache_ttl_seconds`
* `token_ratio` — `_history_token_ratio_sync() >= token_threshold`

前置条件 `T - _compacted_until_turn >= 2 * window_size` 保证至少有一段成熟的新历史可以压缩，这同时也是滚动窗口天然的速率限制器。

## 执行流程

### Major

```
_major_compact(reason)
├─ ① _dump_session_history_jsonl()     ── 把整段 session 落盘到
│                                          {sessions_dir}/{sid}/data/history_{ts}.jsonl
├─ ② render_major_compact_prompt(...)  ── j2 模板 `compact_major_1.0`
│                                          填入 node_role、history 路径、archive_dir
├─ ③ model.generate_with_tools(prompt, instruction=system_prompt, max_turns=1)
│                                       ── LLM 输出结构化的 10 段 markdown 摘要
├─ ④ session.clear_session()
└─ ⑤ session.add_items([continuation_user_message])
                                        ── 单条 user 消息：摘要 + 恢复指针
```

最终 session 里只剩一条 user 消息；下一轮 LLM 看到这条摘要，可以用 `read_file(<history_jsonl>)` 取任何原始 item，用 `read_file(<archive_dir>/...)` 取归档的 tool I/O。

`_compacted_until` 重置为 0；滚动窗口从头开始计数。

### Minor

```
_minor_compact(reason)
├─ items = await session.get_items()
├─ (lo, hi) = _resolve_window_bounds(items)
│                                       ── lo = _compacted_until，
│                                          hi 取最近 window_size 个 function_call
│                                          的起始位置，使 items[hi:] 全是最新窗口
├─ for idx, item in enumerate(items[lo:hi]):
│       rewritten[idx] = maybe_truncate_item(item, archive, threshold, idx)
│                                       ── arguments / output 超过 archive_threshold
│                                          的整段落盘
├─ 没有变更 → 推进 _compacted_until 后返回
├─ session.clear_session()
├─ session.add_items(rewritten)
└─ _compacted_until = hi
```

#### 哪些内容会被归档

窗口内每个 item：

* `function_call.arguments` 序列化长度 ≥ `archive_threshold` → 整段写入 `{idx:06d}_args_{hash8}.json`，原字段替换为占位指针。
* `function_call_output.output` 长度 ≥ `archive_threshold` → 整段写入 `{idx:06d}_output_{hash8}.txt`，同样的指针替换。
* `reasoning` / `message` 类型 item 原样保留。

#### 指针结构

```json
{
  "_archived": true,
  "path": "/Users/.../sessions/<project>/<sid>/data/000042_args_5c0e0ea4.json",
  "original_len": 12345,
  "content_hash": "sha256:5c0e0ea4...",
  "preview": "...原始内容前 200 个字符...",
  "recovery_hint": "read_file('/.../000042_args_5c0e0ea4.json') to load the original args"
}
```

通过 `FuncToolResult.success == 0`（或对非合法 JSON 的兜底字符串匹配 `\"success\": 0` / `\"error\":` / `Traceback`）识别出的错误输出会用 `archive_error_preview_chars` 加长预览，LLM 在不读文件的情况下也能看到错误关键信息。

## 存储布局

所有 compact 相关产物都和 session db 同根：

```
~/.datus/sessions/<project_name>/<session_id>.db        # 会话 SQLite
~/.datus/sessions/<project_name>/<session_id>/data/     # compact 数据目录
    ├── history_20260521T180153Z.jsonl                  # major 历史 dump
    ├── 000042_args_5c0e0ea4.json                       # minor 归档（args）
    └── 000048_output_a1b2c3d4.txt                      # minor 归档（output）
```

路径全部通过 `path_manager.session_data_dir(session_id)` 解析，compact 代码里不硬编码任何字面路径。`session_manager.delete_session` 在删除 db 时会 `rmtree` 整个 `~/.datus/sessions/<project_name>/<session_id>/` 目录。

## 文件系统权限

恢复链路要工作，LLM 必须能在**不弹权限确认**的前提下 `read_file` 归档目录。`fs_path_policy.classify_path` 接受 `session_data_dir` 参数；落在当前 session 的 data 目录下的路径会被分类为 `WHITELIST`（只读）。其他 session 的 data 目录仍属 `EXTERNAL`，即使路径能被猜出也无法读到其他 session 的归档。

接线发生在两个位置：

* `AgenticNode._make_filesystem_func_tool()` 把 `session_data_dir = path_manager.session_data_dir(self.session_id)` 注入 `FilesystemFuncTool`。
* `AgenticNode._make_filesystem_policy()` 把同一个值注入 `PermissionHooks` 使用的 `FilesystemPolicy`。

## 可观测性

major 和 minor 都会在完成时打印一行 INFO 级日志：

```
INFO  Starting major compact for session <sid> (reason=<reason>)
INFO  Major compact complete: <N> chars summary, <K> output tokens, history=<path>
INFO  Minor compact done: window=[<lo>,<hi>) archived=<n> reason=<reason>
```

手动审计磁盘归档：

```bash
ls -l ~/.datus/sessions/<project>/<session_id>/data/
cat  ~/.datus/sessions/<project>/<session_id>/data/history_*.jsonl | head
```

CLI 状态栏和 `/compact` 命令输出里都会显示 `session_id`。

## 关闭 / 调优

* 关闭 minor：`agent.compact.minor.enabled: false`。major 仍会在 90% 阈值（以及 CLI `/compact`）触发。
* 关闭 major：`agent.compact.major.enabled: false`。但 overflow 兜底分支仍然会显式请求 `mode="major"`；要彻底关掉 LLM 摘要，把阈值设到 `1.0` 并依赖 minor + overflow。
* 在 cache 命中率和 token 压力之间权衡，可调 `minor.token_threshold`：值越小越早触发 minor（更多磁盘归档，rewritten 位置导致的 cache 失效更频繁）；值越大允许历史更长（cache 复用更好，但接近 major 阈值的风险更大）。
* 归档粒度 vs 归档数量的权衡可调 `minor.archive_threshold`：阈值高 → 更多内容留在 session 里（磁盘 I/O 少但 session db 大）；阈值低 → 更多内容落盘（session db 小，LLM 需要恢复细节时 `read_file` 调用多）。
