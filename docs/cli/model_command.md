# Model Command `/model`

## Overview

The `/model` slash command lets you switch the active LLM provider and model at runtime without editing configuration files. It provides an interactive TUI picker and direct shortcut syntax for quick switching.

All selections are persisted to the project-level `.datus/config.yml` so they survive session restarts and are scoped to the current working directory.

---

## Basic Usage

### Interactive Picker

Type `/model` (or `/models`) with no arguments to open the full TUI picker:

```text
/model
```

The picker has three tabs, navigable with **Tab** / **Shift+Tab**:

| Tab | Description |
|-----|-------------|
| **Providers** | General-purpose LLM providers (OpenAI, DeepSeek, Claude, etc.) |
| **Plans** | Coding-oriented plan providers (Alibaba Coding, GLM Coding, etc.) |
| **Custom** | Self-hosted or private-deployment models from `agent.models` |

Use **Up/Down** arrows to navigate, **Enter** to select, and **Esc** / **q** to cancel.

### Direct Shortcuts

Switch without opening the picker:

```text
# Switch to a specific provider + model
/model openai/gpt-4.1
/model deepseek/deepseek-chat
/model claude/claude-sonnet-4-5

# Switch to a custom model entry
/model custom:my-internal-model

# Open the picker pre-drilled into a provider
/model openai
```

---

## Provider Authentication

### API Key Providers

For providers that require an API key (OpenAI, DeepSeek, Claude API, Kimi, Qwen, Gemini, etc.):

1. Select the provider in the picker
2. If no credentials are configured, a credential form appears
3. Enter your API key (masked input)
4. The key is persisted to `agent.yml` under `agent.providers.<name>`

API keys can also be pre-configured via environment variables. Each provider has a default env var name defined in `conf/providers.yml` (e.g., `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`).

### Claude Subscription

For Claude Pro/Max subscription users:

1. Select **claude_subscription** in the picker
2. The system auto-detects locally stored subscription tokens
3. If auto-detection fails, you are prompted to enter the token manually (`sk-ant-oat01-...`)

### Codex (ChatGPT Plus/Pro) OAuth

For ChatGPT Plus/Pro users via Codex:

1. Select **codex** in the picker
2. A browser window opens for OAuth authentication
3. After completing the browser flow, connectivity is verified automatically
4. On success, you return to the picker with Codex models available

---

## Custom Models

### Adding a Custom Model

1. Navigate to the **Custom** tab in the picker
2. Select **+ Add custom model**
3. Fill in the form fields:
   - **Name** — Unique identifier (used as `agent.models.<name>` key)
   - **Type** — Interface type (`openai`, `claude`, `deepseek`, `kimi`, `gemini`, etc.)
   - **Base URL** — API endpoint URL
   - **API Key** — Authentication credential (masked)
   - **Model** — Model name / SKU
4. Press **Enter** to save

Custom models are stored in `agent.yml` under `agent.models` and appear in the **Custom** tab.

### Deleting a Custom Model

1. Navigate to the **Custom** tab
2. Select the model to delete
3. Press **d** twice (two-press confirmation) to delete

---

## Configuration Files

### Project Override (`.datus/config.yml`)

When you select a model via `/model`, the choice is saved to `.datus/config.yml` in the current working directory:

```yaml
# Provider-level selection
target:
  provider: openai
  model: gpt-4.1

# Or, custom model selection
target:
  custom: my-internal-model
```

This file supports three keys only:

| Key | Description |
|-----|-------------|
| `target` | Active LLM selection (provider+model or custom name) |
| `default_database` | Default database for this project |
| `project_name` | Override for session/data shard name |

### Provider Credentials (`agent.yml`)

Provider credentials live in `agent.yml` under `agent.providers`:

```yaml
agent:
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
    deepseek:
      api_key: ${DEEPSEEK_API_KEY}
    claude_subscription:
      auth_type: subscription
    codex:
      auth_type: oauth
```

### Provider Catalog (`conf/providers.yml`)

The provider catalog defines available providers, their models, base URLs, and model specs. It is read-only and shipped with Datus. Adding a new provider to this file makes it available in the `/model` picker without code changes.

---

## How It Works

Model resolution follows this precedence:

1. **Project override** — `.datus/config.yml` in the current working directory
2. **Base config** — `agent.target` in `agent.yml`

The `/model` command updates the project override. It does **not** rebuild the agent — the switch takes effect on the next LLM call within the same session.
