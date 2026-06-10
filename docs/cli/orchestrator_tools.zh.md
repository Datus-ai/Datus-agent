# Orchestrator Tools

`--orchestrator-tools` 会在 CLI print mode 中暴露 issue 生命周期工具。它面向外部 orchestrator：orchestrator 把 Datus 当作 worker 运行，同时由 orchestrator 持有 tracker 凭证、issue 状态流转、人工输入和最终任务上报逻辑。

这个参数不会让 Datus 直接调用 GitHub、Jira、Linear 或其他 tracker。它只是在 agent 上注册一个 `orchestrator_tools` 工具类别。生产环境中应同时使用 `--proxy_tools orchestrator_tools.*`，把这些工具调用代理回外部 orchestrator。

## 适用场景

适合使用 orchestrator tools 的情况：

- 外部 runtime 用 `--print` 为单个 issue 或任务启动 Datus。
- 模型需要请求追加 issue comment、移动 issue 状态、询问人工输入或上报最终任务结果。
- tracker 凭证必须保留在 Datus-agent 之外。
- orchestrator 已经消费 print-mode JSON lines，并能通过 stdin 返回代理工具结果。

不要在普通交互式 CLI 会话里使用这个参数。`--orchestrator-tools` 必须配合 `--print`；没有 print mode 时会直接报 CLI 参数错误。

## 基本用法

```bash
datus --datasource analytics \
  --print "Investigate issue DAT-123 and report the fix status." \
  --orchestrator-tools \
  --proxy_tools orchestrator_tools.*
```

代理参数的实际拼写是带下划线的 `--proxy_tools`。`--orchestrator-tools` 会先注册工具，再执行 proxy 包装，因此 `orchestrator_tools.*` 能匹配到这些新注册的工具。

本地调试时也可以不加 `--proxy_tools`：

```bash
datus --datasource analytics \
  --print "Summarize what you would report to the issue." \
  --orchestrator-tools
```

这种模式下工具会暴露给模型，但每次调用都会返回 `success: 0`，错误信息会说明该调用必须由 orchestrator 代理。这样可以只验证 prompt 和工具选择行为，不会真的操作 tracker。

## 可用工具

| 工具 | 用途 | 主要参数 |
|------|------|----------|
| `create_issue_comment` | 请求 orchestrator 给当前 issue 追加 Markdown 评论 | `body`，可选 `issue_id` |
| `update_issue_status` | 请求 orchestrator 把 issue 移动到某个 tracker 状态 | `status`，可选 `issue_id` |
| `request_human_input` | 请求操作员提供决策或缺失输入 | `question`，可选 `issue_id`，可选 `next_status` |
| `mark_blocked` | 上报阻塞当前任务的具体原因 | `reason`，可选 `issue_id`，可选 `next_status`，可选 `artifacts`，可选 `validation` |
| `finish_mission` | 向 orchestrator 上报最终任务结果 | `outcome`，`summary`，可选 `issue_id`，可选 `next_status`，可选 `artifacts`，可选 `validation` |

`finish_mission.outcome` 应使用以下值之一：

- `completed`
- `needs_review`
- `blocked`
- `failed`

`artifacts` 和 `validation` 是结构化对象列表，具体字段形状属于 orchestrator 自己的契约；Datus 只负责透传。

## 代理协议

Print mode 会向 stdout 输出一行一个 `MessagePayload` JSON 对象。当模型调用被代理的 orchestrator 工具时，orchestrator 会看到一个 `call-tool` content item：

```json
{
  "message_id": "tool_123",
  "role": "assistant",
  "content": [
    {
      "type": "call-tool",
      "payload": {
        "callToolId": "tool_123",
        "toolName": "finish_mission",
        "toolParams": {
          "outcome": "completed",
          "summary": "Docs updated and validation passed.",
          "next_status": "In Review"
        }
      }
    }
  ]
}
```

orchestrator 执行 tracker 侧操作后，需要向 stdin 写入一行 `call-tool-result` JSON，并带上相同的 `callToolId`：

```json
{
  "message_id": "tool_123_result",
  "role": "user",
  "content": [
    {
      "type": "call-tool-result",
      "payload": {
        "callToolId": "tool_123",
        "result": {
          "success": 1,
          "result": {
            "tracker_id": "DAT-123",
            "status": "In Review"
          }
        }
      }
    }
  ]
}
```

Datus 会用这个代理结果继续模型执行。如果 stdin 在 orchestrator 返回前关闭，未完成的代理工具调用会被取消。

## 运行注意事项

- tracker 凭证应保留在 orchestrator 中，不要写入 Datus 配置。
- 除非 orchestrator 也负责其他工具类别，否则只代理 `orchestrator_tools.*`。
- 把 `request_human_input` 视为暂停点：orchestrator 应把问题展示给操作员，并在拿到答案后通过 print-mode `user-interaction` payload 返回。
- 正常完成时使用 `finish_mission`，遇到必须依赖外部输入或外部状态变化的阻塞时使用 `mark_blocked`。
