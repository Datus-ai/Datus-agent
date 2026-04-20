# Chat API

The chat endpoints drive the agentic conversation loop. Streaming endpoints return Server-Sent Events; everything
else returns the standard [`Result[T]` envelope](introduction.md#response-envelope).

All endpoints accept the `X-Datus-User-Id` header for per-user session isolation.

## Endpoints

### `POST /api/v1/chat/stream`

Send a chat message and stream the response as Server-Sent Events.

**Body**:

| Field            | Type     | Notes |
|------------------|----------|-------|
| `message`        | string   | Required. User message |
| `session_id`     | string?  | Reuse to continue an existing session |
| `subagent_id`    | string?  | Built-in name (`gen_metrics`, `gen_semantic_model`, …) or custom subagent id |
| `plan_mode`      | bool     | Enable plan mode |
| `catalog`/`database`/`db_schema` | string? | Database context |
| `table_paths`/`metric_paths`/`sql_paths`/`knowledge_paths` | string[]? | `@`-reference paths |
| `max_turns`      | int      | Default `30` |
| `prompt_language`| string   | `en` (default) or `zh` |
| `stream_response`| bool?   | Stream thinking deltas token-by-token; `null` defers to server `--stream` flag (default `false`) |

**Response**: `text/event-stream`. See [Streaming format](#streaming-format) below.

### `POST /api/v1/chat/resume`

Reconnect to a still-running task and continue consuming events from a cursor.

**Body**:

| Field           | Type | Notes |
|-----------------|------|-------|
| `session_id`    | str  | Required |
| `from_event_id` | int? | Event cursor; omit to auto-resume |

**Response**: `text/event-stream`. If the task is unknown or expired, the response is a JSON `Result[dict]` with
`errorCode = "TASK_NOT_FOUND"`; use `GET /chat/history` to fetch the persisted conversation.

### `POST /api/v1/chat/stop`

Interrupt a running session.

**Body**: `{ "session_id": "..." }`

**Response**: `Result[dict]` with `data = { session_id, stopped: true }`. Returns
`errorCode = "SESSION_NOT_RUNNING"` when the session is not active.

### `POST /api/v1/chat/sessions/{session_id}/compact`

Summarize and compress a session's conversation history.

**Response**: `Result[CompactSessionData]` containing `success`, `new_token_count`, `tokens_saved`,
`compression_ratio`.

### `GET /api/v1/chat/sessions`

List all chat sessions for the current user.

**Response**: `Result[ChatSessionData]` with an array of `{ session_id, user_query, created_at, last_updated,
total_turns, token_count, last_sql_queries, is_active }`.

### `DELETE /api/v1/chat/sessions/{session_id}`

Delete a session by id.

### `GET /api/v1/chat/history?session_id=...`

Return the full conversation messages for a session.

**Response**: `Result[ChatHistoryData]` with `messages: SSEMessagePayload[]`.

### `POST /api/v1/chat/user_interaction`

Submit the user's answer to an interactive prompt raised during the chat.

**Body**:

| Field             | Type     | Notes |
|-------------------|----------|-------|
| `session_id`      | string   | Active session |
| `interaction_key` | string   | Key of the interaction request |
| `input`           | string[] | One element per expected answer |

---

## Streaming format

Streaming responses use Server-Sent Events. Each event is encoded as three lines followed by a blank line:

```
id: <sequential int>
event: <event type>
data: <JSON payload>

```

- `id` is monotonically increasing per session, starting at `0`.
- `event` is the event type (see below).
- `data` is a single-line JSON document.
- A heartbeat with `id: -1` and `event: ping` is sent every 10 seconds while the task is idle but still running.

Responses set the headers:

```
Content-Type: text/event-stream; charset=utf-8
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

### Event types

The stream emits five top-level event types. Most of the conversation flows through `message` events; the others
are infrastructure.

| Event     | When | `data` shape |
|-----------|------|--------------|
| `session` | Once, immediately after the session is created | `SessionData` |
| `message` | Repeatedly, for every action produced by the agent | `MessageData` |
| `error`   | Once, on fatal failure (terminates the task) | `ErrorData` |
| `ping`    | Every ~10 s while the task is idle but still running | `{}` |
| `end`     | Once, as the final event of a successful run | `EndData` |

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

`MessageData` is the wrapper used by every `message` event:

```json
{
  "type":    "createMessage",
  "payload": {
    "message_id":       "act_0001",
    "role":             "assistant",
    "content":          [ /* one or more content items, see below */ ],
    "depth":            0,
    "parent_action_id": null
  }
}
```

- `type` is `createMessage` for most streamed actions. When [thinking-delta streaming](#thinking-delta-streaming)
  is enabled, `appendMessage` and `updateMessage` are also used. Clients should treat unknown `type` values
  gracefully.
- `role` is `assistant` while streaming. When fetching `GET /chat/history`, user-authored turns appear with
  `role: "user"`.
- `message_id` is the action id; it is **also the `interactionKey`** when the content describes a user interaction
  (see below).
- `depth` indicates the nesting level of the action: `0` for the main agent, `1` for a sub-agent invoked via the
  `task()` tool. Clients can use this to visually group or indent sub-agent events.
- `parent_action_id` is the action id of the parent `task()` tool call that spawned this sub-agent action. It is
  `null` for main-agent actions (`depth=0`). Together with `depth`, it allows the frontend to build a hierarchical
  tree of agent activities.

### Content item types

Each entry of `content[]` is `{ "type": <kind>, "payload": <kind-specific> }`. The agent can emit any of the
following kinds:

#### `markdown`

Plain text/markdown chunk produced by the assistant (or surfaced from a tool).

```json
{
  "type": "markdown",
  "payload": { "content": "Here are the top 5 customers..." }
}
```

#### `thinking`

Intermediate reasoning emitted by the LLM. Many UIs render this in a collapsed "thinking" block.

```json
{
  "type": "thinking",
  "payload": { "content": "Need to join orders with customers on customer_id..." }
}
```

#### `code`

A code block, typically generated SQL. `codeType` indicates the language.

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

Emitted when the agent starts calling a tool. Use `callToolId` to correlate with the matching `call-tool-result`.

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

Emitted when a tool finishes. `callToolId` matches the prior `call-tool`. `result` is the raw tool output, and
`shortDesc` is a brief human-readable summary when available.

```json
{
  "type": "call-tool-result",
  "payload": {
    "callToolId": "tool_call_8f2e",
    "toolName":   "execute_sql",
    "duration":   0.42,
    "shortDesc":  "5 rows returned",
    "result":     { "columns": ["customer_id", "total"], "rows": [["c1", 1234], ...] }
  }
}
```

#### `error`

Emitted when an action fails (the overall task may still continue). Distinct from the top-level `error` event,
which terminates the stream.

```json
{
  "type": "error",
  "payload": { "content": "execute_sql failed: relation \"orderz\" does not exist" }
}
```

#### `user-interaction`

Emitted when the agent needs the user to make a decision before continuing. The stream pauses until the answer is
posted back via [`POST /chat/user_interaction`](#post-apiv1chatuser_interaction). The `message_id` of the enclosing
`MessageData` is the same value as `payload.interactionKey`; either can be used as `interaction_key` in the reply.

```json
{
  "type": "user-interaction",
  "payload": {
    "interactionKey": "act_0007",
    "actionType":     "choose_table",
    "requests": [
      {
        "content":       "Multiple tables match `customers`. Pick one:",
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

Notes on `requests`:

- It is an **array**: a single interaction may ask several questions at once. The user must answer all of them
  in order.
- `options` is `null` for free-text questions; otherwise it is a list of `{ key, title }`. The user's reply is
  expected to be the `key` of the chosen option (e.g. `"1"`).
- `allowFreeText: true` means the user may type a custom answer even when `options` is non-empty.
- `contentType` is usually `markdown`.

#### `subagent-complete`

Emitted when a sub-agent finishes its delegated task. This event always has `depth >= 1` and a non-null
`parent_action_id`. It acts as a summary marker — clients can use it to collapse sub-agent activity into a
single status line.

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

| Field          | Type   | Description |
|----------------|--------|-------------|
| `subagentType` | string | Name of the sub-agent that ran (e.g. `explore`, `gen_sql`) |
| `toolCount`    | int    | Number of tool calls the sub-agent made |
| `duration`     | float  | Wall-clock seconds the sub-agent ran |

> The enclosing `MessageData.payload` will carry `depth: 1` and a `parent_action_id` pointing to the `task()` tool
> call that spawned the sub-agent. See [MessageData](#messagedata) for the full payload schema.

### Thinking-delta streaming

The `stream_response` field controls whether LLM thinking content is delivered incrementally (token-by-token) or
as a single complete message.

**Priority**: per-request `stream_response` > server-level `--stream` startup flag > default `false`.

#### When enabled (`stream_response: true`)

Thinking tokens are pushed as they arrive, using three `MessageData.type` values:

1. **First delta** — `createMessage` with a `thinking` content item. This creates the message container on the
   client.
2. **Subsequent deltas** — `appendMessage` with a `thinking` content item containing only the incremental text.
   The client appends the delta to the existing message.
3. **Final response** — When the LLM finishes reasoning, a `response` action arrives. Because earlier deltas
   were streamed for this `message_id`, the event uses `updateMessage` to **replace** all previously streamed
   deltas with the complete thinking + response content.

Example SSE frames:

```
id: 3
event: message
data: {"type":"createMessage","payload":{"message_id":"act_0003","role":"assistant","content":[{"type":"thinking","payload":{"content":"Let me analyze"}}],"depth":0,"parent_action_id":null}}

id: 4
event: message
data: {"type":"appendMessage","payload":{"message_id":"act_0003","role":"assistant","content":[{"type":"thinking","payload":{"content":" the sales data"}}],"depth":0,"parent_action_id":null}}

id: 5
event: message
data: {"type":"appendMessage","payload":{"message_id":"act_0003","role":"assistant","content":[{"type":"thinking","payload":{"content":" by joining orders..."}}],"depth":0,"parent_action_id":null}}

id: 6
event: message
data: {"type":"updateMessage","payload":{"message_id":"act_0003","role":"assistant","content":[{"type":"thinking","payload":{"content":"Let me analyze the sales data by joining orders..."}},{"type":"markdown","payload":{"content":"Here are the top 5 customers:\n"}}],"depth":0,"parent_action_id":null}}
```

#### When disabled (`stream_response: false` or omitted)

Thinking deltas are **not** sent over the wire. The complete thinking content appears only once, inside the
final `createMessage` event alongside the response content.

### A complete frame example

Main agent event (`depth=0`):

```
id: 5
event: message
data: {"type":"createMessage","payload":{"message_id":"act_0005","role":"assistant","content":[{"type":"markdown","payload":{"content":"Here are the top 5 customers:\n"}}],"depth":0,"parent_action_id":null}}
```

Sub-agent event (`depth=1`):

```
id: 12
event: message
data: {"type":"createMessage","payload":{"message_id":"act_0012","role":"assistant","content":[{"type":"call-tool","payload":{"callToolId":"act_0012","toolName":"execute_sql","toolParams":{"sql":"SELECT 1"}}}],"depth":1,"parent_action_id":"act_0006"}}
```

Sub-agent completion:

```
id: 15
event: message
data: {"type":"createMessage","payload":{"message_id":"act_0015","role":"assistant","content":[{"type":"subagent-complete","payload":{"subagentType":"gen_sql","toolCount":5,"duration":3.21}}],"depth":1,"parent_action_id":"act_0006"}}
```

### Resume by cursor

If the client disconnects mid-stream, the conversation continues running on the server and buffered events are
kept for 5 minutes after completion. To resume, call `/chat/resume`:

- With `from_event_id` — replay strictly from that id.
- Without `from_event_id` — the server resumes from just before the last delivered event, so the client can
  safely re-process the last event it may not have fully handled.

### Stop semantics

`POST /chat/stop` interrupts the current tool call cleanly, then cancels the background task. The next event the
client receives is the remaining buffered output followed by the end of the stream.

---

## End-to-end demo

The following walkthrough covers the four most common flows: starting a new conversation, reconnecting after a
network drop, reusing a session for a follow-up turn, and responding to an interaction request.

### 1. Start a new conversation

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Datus-User-Id: alice' \
  -d '{ "message": "Show top 5 customers last month" }'
```

The first event you receive is `session`, carrying the `session_id` assigned to this conversation — remember it:

```
id: 0
event: session
data: {"session_id":"chat_session_a1b2c3d4","llm_session_id":"sess_7f1c..."}
```

Subsequent `message` / `action` events stream the assistant's response. The stream ends with an `end` event.

### 2. Resume after disconnect

If the client drops in the middle of a response, the server keeps running the task for a short grace period.
Record the last `id` you successfully processed (e.g. `17`) and reconnect:

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/resume \
  -H 'Content-Type: application/json' \
  -H 'X-Datus-User-Id: alice' \
  -d '{ "session_id": "chat_session_a1b2c3d4", "from_event_id": 18 }'
```

Omit `from_event_id` to let the server auto-resume from just before the last event it delivered. If the task has
already expired, you'll get a JSON `Result` with `errorCode = "TASK_NOT_FOUND"` instead of an SSE stream; in that
case fetch the persisted history via `GET /chat/history`.

### 3. Reuse a session for a follow-up turn

To continue an existing conversation with a new user message, call `/chat/stream` again and pass the same
`session_id`:

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Datus-User-Id: alice' \
  -d '{
        "session_id": "chat_session_a1b2c3d4",
        "message":    "Break that down by region"
      }'
```

The assistant reuses the full conversation context. You can list all active sessions with
`GET /chat/sessions` and fetch messages for any of them with `GET /chat/history?session_id=...`.

### 4. Respond to an interaction request

Occasionally the agent needs a user decision mid-flight (e.g. disambiguating a table). It emits a `message` event
whose `content[]` contains a `user-interaction` item. The stream then pauses until the answer arrives.

Example — assume you received:

```
id: 23
event: message
data: {"type":"createMessage","payload":{"message_id":"act_0007","role":"assistant","content":[{"type":"user-interaction","payload":{"interactionKey":"act_0007","actionType":"choose_table","requests":[{"content":"Multiple tables match `customers`. Pick one:","contentType":"markdown","options":[{"key":"1","title":"sales.customers"},{"key":"2","title":"crm.customers"}],"defaultChoice":"1","allowFreeText":false}]}}]}}
```

Read `payload.interactionKey` (= `"act_0007"`) and present the `requests[0].content` plus its `options` to the
user. When the user picks `sales.customers`, post their `key`:

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

As soon as the answer is accepted, the stream resumes and eventually emits an `end` event. `input` is a list so
multi-question prompts (one entry per `requests[]` item) can be answered in a single call. For free-text answers,
send the typed string instead of an option key.

---

## JavaScript client

```js
const resp = await fetch("/api/v1/chat/stream", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Datus-User-Id": "alice",
  },
  body: JSON.stringify({ message: "Top 5 customers last month" }),
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

## Python client

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
