# Chat 接口

Chat 相关接口驱动 Agent 的对话循环。流式接口以 Server-Sent Events 返回,其余接口使用标准
[`Result[T]` 封装](introduction.zh.md#response-envelope)。

所有接口均支持 `X-Datus-User-Id` 请求头以实现按用户的会话隔离。

## 接口列表

### `POST /api/v1/chat/stream`

发送对话消息,以 SSE 形式流式返回响应。

**Body**:

| 字段             | 类型     | 说明 |
|------------------|----------|------|
| `message`        | string   | 必填,用户消息 |
| `session_id`     | string?  | 复用以延续已有会话 |
| `subagent_id`    | string?  | 内置 subagent 名(`gen_metrics`、`gen_semantic_model` 等)或自定义 id |
| `plan_mode`      | bool     | 是否启用 plan 模式 |
| `catalog`/`database`/`db_schema` | string? | 数据库上下文 |
| `table_paths`/`metric_paths`/`sql_paths`/`knowledge_paths` | string[]? | `@` 引用路径 |
| `max_turns`      | int      | 默认 `30` |
| `prompt_language`| string   | `en`(默认)或 `zh`——选择 prompt 模板变体 |
| `language`       | string?  | 响应语言覆盖（如 `en`、`zh`、`ja`）。留空时回退到 yaml 中的 `agent.language`；若 yaml 也未设置，则不向 system prompt 注入语言指令，由模型自行选择回答语言。控制所有 agentic 节点的回答文本、写入文件的注释、调用子 agent 的 prompt 及 `ask_user` 问题所使用的自然语言；代码、SQL、标识符保持原样。 |
| `stream_response`| bool?   | 是否逐 token 流式下发 thinking 内容;`null` 时使用服务端 `--stream` 启动参数(默认 `false`) |

**响应**:`text/event-stream`,格式见下文 [流式格式](#streaming-format)。

### `POST /api/v1/chat/resume`

重连仍在运行的任务,从游标处继续消费事件。

**Body**:

| 字段            | 类型 | 说明 |
|-----------------|------|------|
| `session_id`    | str  | 必填 |
| `from_event_id` | int? | 事件游标,省略则自动恢复 |

**响应**:`text/event-stream`。任务不存在或已过期时,返回 JSON 形式的 `Result[dict]`,
`errorCode = "TASK_NOT_FOUND"`;此时请使用 `GET /chat/history` 获取持久化的对话内容。

### `POST /api/v1/chat/stop`

中断运行中的会话。

**Body**:`{ "session_id": "..." }`

**响应**:`Result[dict]`,`data = { session_id, stopped: true }`;会话非运行状态时返回
`errorCode = "SESSION_NOT_RUNNING"`。

### `POST /api/v1/chat/sessions/{session_id}/compact`

对某会话的对话历史进行总结压缩。

**响应**:`Result[CompactSessionData]`,含 `success`、`new_token_count`、`tokens_saved`、`compression_ratio`。

### `GET /api/v1/chat/sessions`

列出当前用户的全部会话。

**响应**:`Result[ChatSessionData]`,数组元素为 `{ session_id, user_query, created_at, last_updated,
total_turns, token_count, last_sql_queries, is_active }`。

### `DELETE /api/v1/chat/sessions/{session_id}`

按 id 删除会话。

### `GET /api/v1/chat/history?session_id=...`

返回某会话的完整对话消息。

**响应**:`Result[ChatHistoryData]`,`messages: SSEMessagePayload[]`。

### `POST /api/v1/chat/user_interaction`

提交用户对对话中交互式提问的回答。

**Body**:

| 字段              | 类型     | 说明 |
|-------------------|----------|------|
| `session_id`      | string   | 活跃会话 |
| `interaction_key` | string   | 交互请求对应的 key |
| `input`           | string[] | 每个预期答案一个元素 |

---

## 流式格式 {#streaming-format}

流式响应使用 Server-Sent Events。每个事件由三行加一个空行组成:

```
id: <自增整数>
event: <事件类型>
data: <JSON 负载>

```

- `id` 在会话中从 `0` 开始单调递增。
- `event` 为事件类型(见下表)。
- `data` 为单行 JSON。
- 任务空闲但仍在运行时,服务端每 10 秒发送一条 `id: -1`、`event: ping` 的心跳。

响应头:

```
Content-Type: text/event-stream; charset=utf-8
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

### 事件类型

流上共有 5 种顶层事件。绝大多数对话内容通过 `message` 事件下发,其余为基础设施事件。

| 事件      | 何时发送 | `data` 类型 |
|-----------|---------|-------------|
| `session` | 会话创建后立即发送一次 | `SessionData` |
| `message` | 每个 Agent action 都会下发一条 | `MessageData` |
| `error`   | 致命错误时发送一次,流终止 | `ErrorData` |
| `ping`    | 任务空闲但仍在运行时,每 ~10 秒一次 | `{}` |
| `end`     | 成功结束时作为最后一个事件 | `EndData` |

#### `SessionData`

```json
{
  "session_id":     "chat_session_a1b2c3d4",
  "llm_session_id": "sess_7f1c..."
}
```

#### `EndData`

```json
{
  "session_id":     "chat_session_a1b2c3d4",
  "llm_session_id": "sess_7f1c...",
  "total_events":   42,
  "action_count":   7,
  "duration":       8.31
}
```

#### `ErrorData`

```json
{
  "error":          "LLM call timed out",
  "error_type":     "TimeoutError",
  "session_id":     "chat_session_a1b2c3d4",
  "llm_session_id": "sess_7f1c..."
}
```

#### `MessageData`

`MessageData` 是每个 `message` 事件的统一外层结构:

```json
{
  "type":    "createMessage",
  "payload": {
    "message_id":       "act_0001",
    "role":             "assistant",
    "content":          [ /* 一个或多个 content 项,见下 */ ],
    "depth":            0,
    "parent_action_id": null
  }
}
```

- 大多数流式事件的 `type` 为 `createMessage`。开启 [Thinking-delta 流式下发](#thinking-delta-streaming) 后,
  还会出现 `appendMessage` 和 `updateMessage`。客户端遇到未知 `type` 应优雅忽略。
- 流式期间 `role` 始终为 `assistant`;`GET /chat/history` 拉取时,用户消息会以 `role: "user"` 出现。
- `message_id` 即 action id;当 content 为用户交互时,**它同时也是 `interactionKey`**(详见下文)。
- `depth` 表示动作的嵌套层级:`0` 为主 agent,`1` 为通过 `task()` 工具调起的 sub-agent。客户端可据此
  对 sub-agent 事件进行分组或缩进展示。
- `parent_action_id` 是触发此 sub-agent 动作的父 `task()` 工具调用的 action id。主 agent 动作(`depth=0`)
  该字段为 `null`。结合 `depth`,前端可构建 agent 活动的层级树。

### content 元素类型

`content[]` 的每一项形如 `{ "type": <类型>, "payload": <类型相关> }`。Agent 可能下发以下类型:

#### `markdown`

助手生成或工具回显的 Markdown/纯文本片段。

```json
{
  "type": "markdown",
  "payload": { "content": "销售额前 5 的客户如下..." }
}
```

#### `thinking`

LLM 中间推理内容,UI 通常折叠展示。

```json
{
  "type": "thinking",
  "payload": { "content": "需要按 customer_id 把 orders 与 customers 关联..." }
}
```

#### `code`

代码块,通常是生成的 SQL,`codeType` 标识语言。

```json
{
  "type": "code",
  "payload": {
    "codeType": "sql",
    "content":  "SELECT customer_id, SUM(amount) FROM orders GROUP BY 1"
  }
}
```

#### `call-tool`

Agent 开始调用某个工具时下发。`callToolId` 用于和后续的 `call-tool-result` 关联。

```json
{
  "type": "call-tool",
  "payload": {
    "callToolId": "tool_call_8f2e",
    "toolName":   "execute_sql",
    "toolParams": { "sql": "SELECT 1" }
  }
}
```

#### `call-tool-result`

工具执行完成后下发,`callToolId` 与对应的 `call-tool` 一致。`result` 是工具原始输出,`shortDesc` 是简短摘要(如有)。

```json
{
  "type": "call-tool-result",
  "payload": {
    "callToolId": "tool_call_8f2e",
    "toolName":   "execute_sql",
    "duration":   0.42,
    "shortDesc":  "返回 5 行",
    "result":     { "columns": ["customer_id", "total"], "rows": [["c1", 1234], ...] }
  }
}
```

#### `error`

某个 action 失败时下发(整体任务可能仍在继续)。注意与顶层 `error` 事件区分,后者会终止整个流。

```json
{
  "type": "error",
  "payload": { "content": "execute_sql 失败:relation \"orderz\" does not exist" }
}
```

#### `user-interaction`

Agent 需要用户做决策才能继续时下发。SSE 流随后暂停,直到通过
[`POST /chat/user_interaction`](#post-apiv1chatuser_interaction) 回传答案。
**外层 `MessageData` 的 `message_id` 与 `payload.interactionKey` 数值相同**,任一作为 `interaction_key` 回传均可。

```json
{
  "type": "user-interaction",
  "payload": {
    "interactionKey": "act_0007",
    "actionType":     "choose_table",
    "requests": [
      {
        "content":       "`customers` 命中多张表,请选择:",
        "contentType":   "markdown",
        "options": [
          { "key": "1", "title": "sales.customers" },
          { "key": "2", "title": "crm.customers"   }
        ],
        "defaultChoice": "1",
        "allowFreeText": false
      }
    ]
  }
}
```

`requests` 字段说明:

- 它是**数组**:一次交互可能同时提多个问题,用户需按顺序全部回答。
- 自由文本类问题的 `options` 为 `null`;否则是 `{ key, title }` 列表,用户的回答应填所选项的 `key`(如 `"1"`)。
- `allowFreeText: true` 表示即使有 `options`,也允许用户输入自定义答案。
- `contentType` 通常为 `markdown`。

#### `subagent-complete`

当 sub-agent 完成委派任务时下发。此事件的 `depth >= 1` 且 `parent_action_id` 非空。它作为汇总标记,
客户端可据此将 sub-agent 活动折叠为一行状态摘要。

```json
{
  "type": "subagent-complete",
  "payload": {
    "subagentType": "gen_sql",
    "toolCount":    5,
    "duration":     3.21
  }
}
```

| 字段           | 类型   | 描述 |
|----------------|--------|------|
| `subagentType` | string | 运行的 sub-agent 名称(如 `explore`、`gen_sql`) |
| `toolCount`    | int    | sub-agent 执行的工具调用次数 |
| `duration`     | float  | sub-agent 运行的挂钟时间(秒) |

> 外层 `MessageData.payload` 将携带 `depth: 1` 和指向触发该 sub-agent 的 `task()` 工具调用的
> `parent_action_id`。完整 payload 结构参见 [MessageData](#messagedata)。

### Thinking-delta 流式下发 {#thinking-delta-streaming}

`stream_response` 字段控制 LLM 的 thinking 内容是逐 token 增量下发,还是作为单条完整消息一次性下发。

**优先级**:请求级 `stream_response` > 服务端 `--stream` 启动参数 > 默认 `false`。

#### 开启时(`stream_response: true`)

Thinking token 在生成时即时推送,使用三种 `MessageData.type`:

1. **首个 delta** — `createMessage`,携带一个 `thinking` content 项。客户端据此创建消息容器。
2. **后续 delta** — `appendMessage`,携带一个 `thinking` content 项,仅包含增量文本。客户端将其追加到已有消息。
3. **最终响应** — LLM 推理结束后,`response` action 到达。由于此 `message_id` 之前已流式下发过 delta,
   事件使用 `updateMessage` **替换**先前所有 delta,写入完整的 thinking + response 内容。

SSE 帧示例:

```
id: 3
event: message
data: {"type":"createMessage","payload":{"message_id":"act_0003","role":"assistant","content":[{"type":"thinking","payload":{"content":"让我分析一下"}}],"depth":0,"parent_action_id":null}}

id: 4
event: message
data: {"type":"appendMessage","payload":{"message_id":"act_0003","role":"assistant","content":[{"type":"thinking","payload":{"content":"销售数据"}}],"depth":0,"parent_action_id":null}}

id: 5
event: message
data: {"type":"appendMessage","payload":{"message_id":"act_0003","role":"assistant","content":[{"type":"thinking","payload":{"content":"，需要关联 orders 表..."}}],"depth":0,"parent_action_id":null}}

id: 6
event: message
data: {"type":"updateMessage","payload":{"message_id":"act_0003","role":"assistant","content":[{"type":"thinking","payload":{"content":"让我分析一下销售数据，需要关联 orders 表..."}},{"type":"markdown","payload":{"content":"销售额前 5 的客户如下:\n"}}],"depth":0,"parent_action_id":null}}
```

#### 关闭时(`stream_response: false` 或不传)

Thinking delta **不会**通过 SSE 下发。完整的 thinking 内容仅在最终的 `createMessage` 事件中与 response 内容一同出现。

### 完整事件帧示例

主 agent 事件(`depth=0`):

```
id: 5
event: message
data: {"type":"createMessage","payload":{"message_id":"act_0005","role":"assistant","content":[{"type":"markdown","payload":{"content":"销售额前 5 的客户:\n"}}],"depth":0,"parent_action_id":null}}
```

Sub-agent 事件(`depth=1`):

```
id: 12
event: message
data: {"type":"createMessage","payload":{"message_id":"act_0012","role":"assistant","content":[{"type":"call-tool","payload":{"callToolId":"act_0012","toolName":"execute_sql","toolParams":{"sql":"SELECT 1"}}}],"depth":1,"parent_action_id":"act_0006"}}
```

Sub-agent 完成:

```
id: 15
event: message
data: {"type":"createMessage","payload":{"message_id":"act_0015","role":"assistant","content":[{"type":"subagent-complete","payload":{"subagentType":"gen_sql","toolCount":5,"duration":3.21}}],"depth":1,"parent_action_id":"act_0006"}}
```

### 按游标续传

客户端中途断开后,对话仍在服务端继续运行,缓冲事件在任务结束后保留 5 分钟。调用 `/chat/resume` 续传:

- 提供 `from_event_id` 时严格从该 id 重放。
- 省略 `from_event_id` 时,服务端向前回退一个事件恢复,便于客户端安全地重新处理上一个可能未处理完的事件。

### 停止语义

`POST /chat/stop` 会先优雅地中断当前工具调用,再取消后台任务。客户端随后会收到剩余缓冲事件,紧接着流正常结束。

---

## 完整示例

下面演示四个最常见的使用流程:发起新对话、断线续传、复用 session 继续追问、以及响应交互请求。

### 1. 发起新对话

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Datus-User-Id: alice' \
  -d '{ "message": "上月销售额前 5 的客户" }'
```

收到的第一个事件是 `session`,其中的 `session_id` 是本轮对话的唯一标识,请保存:

```
id: 0
event: session
data: {"session_id":"chat_session_a1b2c3d4","llm_session_id":"sess_7f1c..."}
```

之后是一连串 `message` / `action` 事件,最终以 `end` 事件结束。

### 2. 断线续传

客户端中途断开后,服务端会在短时间内继续运行任务。记录下你最后成功处理的事件 `id`(例如 `17`),然后重连:

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/resume \
  -H 'Content-Type: application/json' \
  -H 'X-Datus-User-Id: alice' \
  -d '{ "session_id": "chat_session_a1b2c3d4", "from_event_id": 18 }'
```

省略 `from_event_id` 则由服务端自动从最后一个已下发事件之前恢复。如果任务已过期,
响应会是 `errorCode = "TASK_NOT_FOUND"` 的 JSON `Result`,此时请改用 `GET /chat/history` 拿到持久化的历史。

### 3. 复用 session 进行追问

想在既有对话上继续追问,只需再次调用 `/chat/stream` 并带上同一个 `session_id`:

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Datus-User-Id: alice' \
  -d '{
        "session_id": "chat_session_a1b2c3d4",
        "message":    "再按地区拆分看看"
      }'
```

助手会自动沿用完整的对话上下文。可以通过 `GET /chat/sessions` 列出所有会话,通过
`GET /chat/history?session_id=...` 查看任一会话的全部消息。

### 4. 响应交互请求

有时 Agent 需要用户临时做决策(例如消歧表名)。此时会下发一个 `message` 事件,其 `content[]` 中包含一个
`user-interaction` 元素,SSE 流随后暂停,直到收到回答。

示例 — 假设你收到:

```
id: 23
event: message
data: {"type":"createMessage","payload":{"message_id":"act_0007","role":"assistant","content":[{"type":"user-interaction","payload":{"interactionKey":"act_0007","actionType":"choose_table","requests":[{"content":"`customers` 命中多张表,请选择:","contentType":"markdown","options":[{"key":"1","title":"sales.customers"},{"key":"2","title":"crm.customers"}],"defaultChoice":"1","allowFreeText":false}]}}]}}
```

读取 `payload.interactionKey`(此处为 `"act_0007"`),将 `requests[0].content` 与 `options` 展示给用户。
用户选择 `sales.customers` 后,提交其 `key`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/user_interaction \
  -H 'Content-Type: application/json' \
  -H 'X-Datus-User-Id: alice' \
  -d '{
        "session_id":      "chat_session_a1b2c3d4",
        "interaction_key": "act_0007",
        "input":           ["1"]
      }'
```

回答被接受后,SSE 流恢复,最终以 `end` 事件收尾。`input` 是数组,多问题提问(每个 `requests[]` 项一个回答)
可以在一次调用中一并提交。自由文本回答时,直接传用户输入的字符串即可,无需 option key。

---

## JavaScript 客户端

```js
const resp = await fetch("/api/v1/chat/stream", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Datus-User-Id": "alice",
  },
  body: JSON.stringify({ message: "上月销售额前 5 的客户" }),
});

const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buf = "";
let lastId = -1;

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += decoder.decode(value, { stream: true });

  let sep;
  while ((sep = buf.indexOf("\n\n")) !== -1) {
    const frame = buf.slice(0, sep);
    buf = buf.slice(sep + 2);

    const lines = frame.split("\n");
    const id    = parseInt(lines.find(l => l.startsWith("id: "))?.slice(4)    ?? "-1", 10);
    const event =          lines.find(l => l.startsWith("event: "))?.slice(7) ?? "";
    const data  = JSON.parse(lines.find(l => l.startsWith("data: "))?.slice(6) ?? "{}");

    if (id >= 0) lastId = id;
    handleEvent(event, data);
  }
}
```

## Python 客户端

```python
import json, httpx

async def stream_chat(message: str, user_id: str = "alice"):
    headers = {"X-Datus-User-Id": user_id}
    payload = {"message": message}
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            "http://127.0.0.1:8000/api/v1/chat/stream",
            json=payload,
            headers=headers,
        ) as resp:
            event = {}
            async for line in resp.aiter_lines():
                if line == "":
                    if event:
                        yield event
                        event = {}
                    continue
                key, _, value = line.partition(": ")
                if key == "data":
                    event["data"] = json.loads(value)
                else:
                    event[key] = value
```
