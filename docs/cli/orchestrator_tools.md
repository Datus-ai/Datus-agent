# Orchestrator Tools

`--orchestrator-tools` exposes issue lifecycle tools in CLI print mode. It is intended for an external orchestrator that runs Datus as a worker while the orchestrator owns tracker credentials, issue status changes, operator questions, and final task reporting.

The flag does not make Datus call GitHub, Jira, Linear, or any other tracker directly. It registers a small `orchestrator_tools` tool category on the agent. In production, those tools should be proxied back to the orchestrator with `--proxy_tools orchestrator_tools.*`.

## When To Use

Use orchestrator tools when:

- An external runtime invokes Datus with `--print` for one issue or task.
- The model should be able to request issue comments, status moves, human input, or final mission reporting.
- Tracker credentials should stay outside Datus-agent.
- The orchestrator already consumes print-mode JSON lines and can answer proxied tool calls through stdin.

Do not use this flag for normal interactive CLI sessions. `--orchestrator-tools` requires `--print`; using it without print mode is a CLI error.

## Basic Usage

```bash
datus --datasource analytics \
  --print "Investigate issue DAT-123 and report the fix status." \
  --orchestrator-tools \
  --proxy_tools orchestrator_tools.*
```

The proxy flag is spelled `--proxy_tools` with an underscore. `--orchestrator-tools` is attached before proxying, so `orchestrator_tools.*` can match the newly registered tools.

For local debugging, you can omit `--proxy_tools`:

```bash
datus --datasource analytics \
  --print "Summarize what you would report to the issue." \
  --orchestrator-tools
```

In that mode the tools are visible to the model, but each call returns `success: 0` with an error saying the call must be proxied. This verifies prompt and tool selection behavior without touching a tracker.

## Available Tools

| Tool | Purpose | Key arguments |
|------|---------|---------------|
| `create_issue_comment` | Ask the orchestrator to append a Markdown comment to the current issue | `body`, optional `issue_id` |
| `update_issue_status` | Ask the orchestrator to move an issue to a tracker status | `status`, optional `issue_id` |
| `request_human_input` | Ask the operator for a decision or missing input | `question`, optional `issue_id`, optional `next_status` |
| `mark_blocked` | Report a concrete blocker that prevents completion | `reason`, optional `issue_id`, optional `next_status`, optional `artifacts`, optional `validation` |
| `finish_mission` | Report the final task outcome to the orchestrator | `outcome`, `summary`, optional `issue_id`, optional `next_status`, optional `artifacts`, optional `validation` |

`finish_mission.outcome` should be one of:

- `completed`
- `needs_review`
- `blocked`
- `failed`

`artifacts` and `validation` are lists of structured objects. Their exact shape belongs to the orchestrator contract; Datus passes them through.

## Proxy Protocol

Print mode writes one `MessagePayload` JSON object per stdout line. When a proxied orchestrator tool is called, the orchestrator sees a `call-tool` content item:

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

The orchestrator executes the tracker-side operation, then writes a `call-tool-result` JSON line to stdin with the same `callToolId`:

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

Datus resumes the model run with that proxied result. If stdin closes before the orchestrator answers, pending proxied tool calls are cancelled.

## Operational Notes

- Keep tracker credentials in the orchestrator, not in Datus configuration.
- Proxy only `orchestrator_tools.*` unless the orchestrator also owns other tool categories.
- Treat `request_human_input` as a pause point: the orchestrator should surface the question to an operator and return the answer as a print-mode `user-interaction` payload when available.
- Use `finish_mission` for normal completion and `mark_blocked` when work cannot continue without external input or state changes.
