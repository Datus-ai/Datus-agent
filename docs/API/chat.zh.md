# Chat 接口

Chat 相关接口驱动 Agent 的对话循环。流式接口以 Server-Sent Events 返回,其余接口使用标准
[`Result[T]` 封装](introduction.zh.md#响应封装)。

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
| `source`         | string?  | `web` / `vscode` — 切换文件系统工具的代理模式 |
| `catalog`/`database`/`db_schema` | string? | 数据库上下文 |
| `table_paths`/`metric_paths`/`sql_paths`/`knowledge_paths` | string[]? | `@` 引用路径 |
| `max_turns`      | int      | 默认 `30` |
| `prompt_language`| string   | `en`(默认)或 `zh` |

**响应**:`text/event-stream`,格式见下文 [流式格式](#流式格式)。

### `POST /api/v1/chat/resume`

重连仍在运行的任务,从游标处继续消费事件。

**Body**:

| 字段            | 类型 | 说明 |
|-----------------|------|------|
| `session_id`    | str  | 必填 |
| `from_event_id` | int? | 事件游标,省略则自动恢复 |
| `source`        | str? | `web`/`vscode` |

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

## 流式格式

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

| 事件      | `data` 类型       | 含义 |
|-----------|-------------------|------|
| `session` | `SessionData`     | 会话开始后立即发送,携带 `session_id` 与 `llm_session_id` |
| `message` | `MessageData`     | 创建 / 追加 / 整体更新一段助手消息 |
| `action`  | `MessageData`     | 工具调用 / 子动作进度,包括交互式提问 |
| `error`   | `ErrorData`       | 致命错误,任务终止 |
| `ping`    | `{}`              | 心跳,可忽略 |
| `end`     | `EndData`         | 终止事件,含本次运行的统计信息 |

### `data` 类型定义

**SessionData** — 流开始时发送一次。

```json
{
  "session_id": "chat_session_a1b2c3d4",
  "llm_session_id": "sess_7f1c..."
}
```

**MessageData** — 被 `message` 与 `action` 两种事件复用。通过 `type` 字段区分三种子操作:

| `type`           | `payload` 结构                                                     |
|------------------|--------------------------------------------------------------------|
| `createMessage`  | `{ message_id, role, content[] }` — 新建一条消息                   |
| `appendMessage`  | `{ message_id, type, content }` — 向消息追加内容                   |
| `updateMessage`  | `{ message_id, payload }` — 整体替换消息内容                       |

每个 `content` 元素形如:

```json
{ "type": "markdown", "payload": { "content": "你好,我可以帮你..." } }
```

`type` 取值为 `markdown`、`code`(payload 为 `{ code_type, content }`)或 `csv`(payload 为 `{ content }`)。

一个完整的 `message` 事件帧:

```
id: 5
event: message
data: {"type":"appendMessage","payload":{"message_id":"m-1","type":"markdown","content":{"content":"销售额前 5 的客户:\n"}}}
```

**带交互请求的 action 事件** — 当 Agent 需要用户做决策(例如消歧表名、在多条 SQL 候选中二选一)时,
会发送一个 `action` 事件,其 `payload` 即交互提问本体。该事件的 `message_id` 就是后续调用
[`POST /chat/user_interaction`](#post-apiv1chatuser_interaction) 所需的 **`interaction_key`**。
此时 SSE 流会暂停,直到收到用户回答。

**ErrorData**:

```json
{
  "error": "LLM call timed out",
  "error_type": "TimeoutError",
  "session_id": "chat_session_a1b2c3d4",
  "llm_session_id": "sess_7f1c..."
}
```

**EndData** — 成功执行结束时作为最后一个事件:

```json
{
  "session_id": "chat_session_a1b2c3d4",
  "llm_session_id": "sess_7f1c...",
  "total_events": 42,
  "action_count": 7,
  "duration": 8.31
}
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

有时 Agent 需要用户临时做决策(例如消歧表名、在多条 SQL 候选中二选一)。此时会下发一个 `action` 事件,
其 `payload` 描述提问及选项,**该事件的 `message_id` 即作为 `interaction_key` 回传**。

示例 — 假设你收到:

```
id: 23
event: action
data: {"type":"createMessage","payload":{"message_id":"act-need-table-choice","role":"assistant","content":[{"type":"markdown","payload":{"content":"`customers` 命中多张表,请选择:\n1. sales.customers\n2. crm.customers"}}]}}
```

此时 SSE 流暂停,客户端提交用户的回答:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/user_interaction \
  -H 'Content-Type: application/json' \
  -H 'X-Datus-User-Id: alice' \
  -d '{
        "session_id":      "chat_session_a1b2c3d4",
        "interaction_key": "act-need-table-choice",
        "input":           ["1"]
      }'
```

回答被接受后,SSE 流恢复,继续下发 `message` / `action`,最终以 `end` 事件收尾。`input` 是数组,
多答案提问(例如一次填写多个参数)可在一次调用中一并提交。

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
