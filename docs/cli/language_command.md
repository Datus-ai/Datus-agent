# Language Command `/language`

## Overview

The `/language` slash command pins the **response language** used by every
agentic node — replies, file comments, sub-agent prompts, and `ask_user`
questions. Code, SQL, and identifiers are unaffected and stay in their
original form.

The setting can be persisted globally (in `agent.yml` as `agent.language`)
or scoped to the current project (in `.datus/config.yml` as `language`);
the project-level value always wins at runtime.

This is the runtime equivalent of the `agent.language` configuration field
documented in [Agent configuration](../configuration/agent.md).

---

## Supported Languages

| Code | Language |
|------|----------|
| `auto` | Let the model decide (clear any override) |
| `en` | English |
| `zh` | Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `pt` | Portuguese |
| `ru` | Russian |
| `it` | Italian |

Unknown codes are accepted with a warning — useful for languages not
listed above; the value is passed through to the system prompt as-is.

---

## Basic Usage

### Interactive picker

Type `/language` with no arguments to open the TUI. The picker is a
two-step flow:

1. Pick a language code
2. Pick the persistence scope (project vs global)

```text
/language
```

### Direct shortcuts

```text
# Pick code, then choose scope interactively
/language zh
/language en

# Persist to project (.datus/config.yml)
/language zh --project

# Persist to global (agent.yml)
/language zh --global

# Clear the override (model decides)
/language auto
/language --clear            # equivalent: clear project override only
/language auto --global      # clear global override
```

---

## Resolution Order

The active language is resolved in this order:

1. `.datus/config.yml` → `language:` (project override)
2. `agent.yml` → `agent.language:` (global default)
3. Otherwise: no language directive is injected; the model picks its own
   response language.

When the override is cleared, the project falls back to the global value
(if any), and clearing the global value falls back to "model decides".

---

## Examples

```bash
# Pin Chinese for the current project
/language zh --project

# Make English the default everywhere
/language en --global

# Drop the project override and inherit the global default again
/language --clear

# Hand control back to the model entirely
/language auto --global
```

See also:

- Configuration field: [`agent.language`](../configuration/agent.md)
- Per-request override: `language` field in
  [`POST /api/v1/chat/stream`](../API/chat.md#post-apiv1chatstream)
