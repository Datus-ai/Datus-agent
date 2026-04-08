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
| `source`         | string?  | `web` / `vscode` — switches filesystem tools to proxy mode |
| `catalog`/`database`/`db_schema` | string? | Database context |
| `table_paths`/`metric_paths`/`sql_paths`/`knowledge_paths` | string[]? | `@`-reference paths |
| `max_turns`      | int      | Default `30` |
| `prompt_language`| string   | `en` (default) or `zh` |

**Response**: `text/event-stream`. See [Streaming format](#streaming-format) below.

### `POST /api/v1/chat/resume`

Reconnect to a still-running task and continue consuming events from a cursor.

**Body**:

| Field           | Type | Notes |
|-----------------|------|-------|
| `session_id`    | str  | Required |
| `from_event_id` | int? | Event cursor; omit to auto-resume |
| `source`        | str? | `web`/`vscode` |

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

| Event     | `data` shape              | Meaning |
|-----------|---------------------------|---------|
| `session` | `SessionData`             | Sent right after the session starts; carries `session_id` and `llm_session_id` |
| `message` | `MessageData`             | Create / append / update an assistant message segment |
| `action`  | `MessageData`             | Tool call / sub-action progress, including interactive prompts |
| `error`   | `ErrorData`               | A fatal error occurred; the task is terminated |
| `ping`    | `{}`                      | Heartbeat; safe to ignore |
| `end`     | `EndData`                 | Final event with summary counters |

### `data` type schemas

**SessionData** — emitted once at the start of the stream.

```json
{
  "session_id": "chat_session_a1b2c3d4",
  "llm_session_id": "sess_7f1c..."
}
```

**MessageData** — used by both `message` and `action` events. The `type` field selects one of three sub-operations:

| `type`           | `payload` shape                                                   |
|------------------|-------------------------------------------------------------------|
| `createMessage`  | `{ message_id, role, content[] }` — start a new message           |
| `appendMessage`  | `{ message_id, type, content }` — stream more content into it     |
| `updateMessage`  | `{ message_id, payload }` — replace message content wholesale     |

Each `content` item takes the form:

```json
{ "type": "markdown", "payload": { "content": "Hello, I can help..." } }
```

`type` is one of `markdown`, `code` (with `{ code_type, content }`), or `csv` (with `{ content }`).

A complete `message` event looks like:

```
id: 5
event: message
data: {"type":"appendMessage","payload":{"message_id":"m-1","type":"markdown","content":{"content":"Here are the top 5 customers:\n"}}}
```

**Action events with interaction requests** — when the agent needs the user to pick among options or answer a
question, an `action` event arrives whose `payload` carries an interaction prompt. The `message_id` of that
action acts as the `interaction_key` that must be posted back via
[`POST /chat/user_interaction`](#post-apiv1chatuser_interaction). The stream pauses until the answer arrives.

**ErrorData**:

```json
{
  "error": "LLM call timed out",
  "error_type": "TimeoutError",
  "session_id": "chat_session_a1b2c3d4",
  "llm_session_id": "sess_7f1c..."
}
```

**EndData** — always the last event of a successful run:

```json
{
  "session_id": "chat_session_a1b2c3d4",
  "llm_session_id": "sess_7f1c...",
  "total_events": 42,
  "action_count": 7,
  "duration": 8.31
}
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

Occasionally the agent needs a user decision mid-flight (e.g. disambiguating a table, choosing between SQL
candidates). It emits an `action` event whose message payload describes the question and lists the options. The
`message_id` of that event is the **`interaction_key`** you must submit back.

Example — assume you received:

```
id: 23
event: action
data: {"type":"createMessage","payload":{"message_id":"act-need-table-choice","role":"assistant","content":[{"type":"markdown","payload":{"content":"Multiple tables match `customers`. Pick one:\n1. sales.customers\n2. crm.customers"}}]}}
```

The SSE stream now pauses. Post the user's answer:

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

As soon as the answer is accepted, the stream resumes emitting `message` and `action` events and eventually an
`end` event. `input` is a list so multi-answer prompts (e.g. filling in several parameters at once) can be
submitted in a single call.

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
