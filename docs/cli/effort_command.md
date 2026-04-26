# Effort Command `/effort`

## Overview

The `/effort` slash command sets the **reasoning effort** level used by the
active LLM. A single knob covers every supported provider: LiteLLM maps
the level to each provider's native dialect — OpenAI `reasoning_effort`,
Anthropic `thinking.budget_tokens`, Gemini `thinking_config.thinking_budget`,
DeepSeek / Kimi reasoning, etc.

The setting can be persisted globally (in `agent.yml`) or scoped to the
current project (in `.datus/config.yml`); the project-level value always
wins at runtime.

---

## Effort Levels

| Level | Meaning |
|-------|---------|
| `off` | Disable reasoning (no thinking) |
| `minimal` | Minimal effort (fast; gpt-5 family) |
| `low` | Low effort |
| `medium` | Medium effort (balanced) |
| `high` | High effort (deep reasoning) |

If the active model does not support reasoning, the level is silently
ignored. `/effort status` shows whether the active model can actually
consume the hint (queried via `litellm.supports_reasoning`).

---

## Basic Usage

### Interactive picker

Type `/effort` with no arguments to open the TUI. The picker is a
two-step flow:

1. Pick an effort level
2. Pick the persistence scope (project vs global)

```text
/effort
```

### Direct shortcuts

```text
# Pick level, then choose scope interactively
/effort high
/effort minimal

# Persist to project (.datus/config.yml)
/effort high --project

# Persist to global (agent.yml)
/effort high --global

# Disable reasoning
/effort off

# Remove the project-level override (fall back to global / model defaults)
/effort --clear

# Show the effective level and where it came from
/effort status
```

---

## Resolution Order

When the agent starts a turn, the effective effort is resolved in this
order (first match wins):

1. `.datus/config.yml` → `reasoning_effort:` (project override)
2. `agent.yml` → top-level `reasoning_effort:` (global default)
3. `agent.models.<active>.reasoning_effort` (model-level config)
4. `agent.models.<active>.enable_thinking: true` → treated as `medium`
5. Otherwise: not set (model-level defaults apply)

`/effort status` prints the resolved level alongside its source label
(`project` / `global` / `model` / `not set`).

---

## Examples

```bash
# One-off: switch to high effort for the current project
/effort high --project

# Make "low" the default for every project on this machine
/effort low --global

# Quickly turn reasoning off for the current project
/effort off --project

# Reset and let the model defaults apply again
/effort --clear

# Inspect what's currently effective
/effort status
```

See also: [`/model`](model_command.md) for switching the active LLM.
