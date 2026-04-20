# Models

`agent.models` registers the LLM providers your Datus Agent can call, and `agent.target` picks the default. Every node and subagent can override the choice per-entry via its own `model:` field.

```yaml
agent:
  target: openai
  models:
    openai:
      type: openai
      base_url: https://api.openai.com/v1
      api_key: ${OPENAI_API_KEY}
      model: gpt-5.2
```

## Default model (`agent.target`)

`agent.target` names the entry under `agent.models` that nodes use when they don't specify their own `model:`. It must match a key in the `models` map.

```yaml
agent:
  target: openai   # any key from agent.models
```

Per-node overrides (e.g. `agentic_nodes.chat.model: claude`) always win over `target`. See **[Agent](agent.md)** for per-node configuration.

## Required fields

Each entry under `agent.models.<key>` requires:

- **`<key>`** — any name you choose. `agent.target` and node `model:` fields reference this name.
- **`type`** — provider type. Use the value shown in the [examples below](#configuration-examples).
- **`base_url`** — provider API endpoint.
- **`api_key`** — credentials. Prefer `${ENV_VAR}` over inline secrets.
- **`model`** — model name on the provider side.

## Optional fields

| Field | Default | Purpose |
|---|---|---|
| `temperature` | provider default | Sampling temperature override. |
| `top_p` | provider default | Nucleus sampling override. |
| `auth_type` | `api_key` | `api_key`, `subscription`, or `oauth`. See [Authentication modes](#authentication-modes). |
| `enable_thinking` | `false` | Turn on reasoning/thinking mode for providers that support it. |
| `max_retry` | `3` | Retries on stream-connection errors. |
| `retry_interval` | `2.0` | Seconds between retries. |

## Supported providers

The names below are also the keys the `datus-agent configure` wizard writes into `agent.models`.

### General-purpose

| Provider | Typical models | Auth |
|---|---|---|
| `openai` | `gpt-5.2`, `gpt-4.1`, `o3` | API key |
| `deepseek` | `deepseek-chat`, `deepseek-reasoner` | API key |
| `claude` | `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5` | API key |
| `kimi` | `kimi-k2.5`, `kimi-k2-thinking` | API key |
| `qwen` | `qwen3-max`, `qwen3-coder-plus` | API key |
| `gemini` | `gemini-3-pro-preview`, `gemini-3-flash-preview` | API key |
| `minimax` | `MiniMax-M2.7`, `MiniMax-M2.5` | API key |
| `glm` | `glm-5`, `glm-4.7` | API key |

### Multi-provider gateways

| Provider | Typical models | Auth |
|---|---|---|
| `openrouter` | `openrouter/anthropic/claude-sonnet-4-6`, `openrouter/openai/gpt-5.2`, `openrouter/google/gemini-3-pro-preview` (300+ models) | API key |

### Special-auth

| Provider | Auth | Notes |
|---|---|---|
| `claude_subscription` | Claude Pro/Max subscription token | Reuses your local `claude setup-token` credential when present; otherwise paste a `sk-ant-oat01-…` token. |
| `codex` | OAuth (PKCE + Device Code) | Drives ChatGPT Plus/Pro via Codex; tokens are persisted under `~/.datus/oauth/`. |

### Coding Plan providers

These point at vendor coding/planning endpoints. They are configured exactly like a normal provider and can serve as `agent.target` or be referenced from any node.

| Provider | Default model | Notes |
|---|---|---|
| `alibaba_coding` | `qwen3-coder-plus` | DashScope coding endpoint |
| `glm_coding` | `glm-5` | GLM coding endpoint |
| `minimax_coding` | `MiniMax-M2.7` | MiniMax coding endpoint |
| `kimi_coding` | `kimi-for-coding` | Kimi coding endpoint |

!!! tip "When to choose a coding plan provider"
    Stick with a regular provider for general chat, SQL generation, or cost efficiency.

    Pick a `*_coding` provider when you want a default model tuned for planning, code generation, and structured task decomposition — especially if you use [Plan Mode](../cli/plan_mode.md) frequently.

## Configuration examples

=== "OpenAI"

    ```yaml
    openai:
      type: openai
      base_url: https://api.openai.com/v1
      api_key: ${OPENAI_API_KEY}
      model: gpt-5.2
    ```

=== "Anthropic Claude"

    ```yaml
    claude:
      type: claude
      base_url: https://api.anthropic.com
      api_key: ${ANTHROPIC_API_KEY}
      model: claude-sonnet-4-6
    ```

=== "DeepSeek"

    ```yaml
    deepseek:
      type: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-chat
    ```

=== "Google Gemini"

    ```yaml
    gemini:
      type: gemini
      base_url: https://generativelanguage.googleapis.com/v1beta
      api_key: ${GEMINI_API_KEY}
      model: gemini-3-pro-preview
    ```

=== "Kimi (Moonshot)"

    ```yaml
    kimi:
      type: kimi
      base_url: https://api.moonshot.cn/v1
      api_key: ${KIMI_API_KEY}
      model: kimi-k2.5
    ```

=== "Qwen (Alibaba)"

    ```yaml
    qwen:
      type: openai
      base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key: ${DASHSCOPE_API_KEY}
      model: qwen3-max
    ```

=== "MiniMax"

    ```yaml
    minimax:
      type: minimax
      base_url: https://api.minimaxi.com/v1
      api_key: ${MINIMAX_API_KEY}
      model: MiniMax-M2.7
    ```

=== "GLM (Zhipu)"

    ```yaml
    glm:
      type: glm
      base_url: https://open.bigmodel.cn/api/paas/v4
      api_key: ${GLM_API_KEY}
      model: glm-5
    ```

=== "OpenRouter"

    ```yaml
    openrouter:
      type: openrouter
      base_url: https://openrouter.ai/api/v1
      api_key: ${OPENROUTER_API_KEY}
      model: openrouter/anthropic/claude-sonnet-4-6
    ```

    Model names use the `openrouter/<provider>/<model>` convention; browse the full catalog at [openrouter.ai/models](https://openrouter.ai/models).

=== "Alibaba Coding Plan"

    ```yaml
    alibaba_coding:
      type: claude
      base_url: https://coding-intl.dashscope.aliyuncs.com/apps/anthropic
      api_key: ${DASHSCOPE_API_KEY}
      model: qwen3-coder-plus
      temperature: 1.0
      top_p: 0.95
    ```

=== "GLM Coding Plan"

    ```yaml
    glm_coding:
      type: claude
      base_url: https://open.bigmodel.cn/api/anthropic
      api_key: ${GLM_API_KEY}
      model: glm-5
    ```

=== "MiniMax Coding Plan"

    ```yaml
    minimax_coding:
      type: claude
      base_url: https://api.minimaxi.com/anthropic
      api_key: ${MINIMAX_API_KEY}
      model: MiniMax-M2.7
    ```

=== "Kimi Coding Plan"

    ```yaml
    kimi_coding:
      type: claude
      base_url: https://api.kimi.com/coding/
      api_key: ${KIMI_API_KEY}
      model: kimi-for-coding
    ```

=== "Claude Subscription"

    ```yaml
    claude_subscription:
      type: claude
      base_url: https://api.anthropic.com
      api_key: ${CLAUDE_CODE_OAUTH_TOKEN}
      model: claude-sonnet-4-6
      auth_type: subscription
    ```

=== "Codex"

    ```yaml
    codex:
      type: codex
      base_url: https://chatgpt.com/backend-api/codex
      api_key: ""
      model: codex-mini-latest
      auth_type: oauth
    ```

## Authentication modes

### `api_key` (default)

Paste an API key, or use `${ENV_VAR}`. Works for every provider unless noted below.

### `subscription` — Claude Pro/Max

Used by `claude_subscription`. Datus reuses the token written by `claude setup-token` automatically; you can also set `CLAUDE_CODE_OAUTH_TOKEN` or paste the token directly into `api_key`.

### `oauth` — Codex

Used by `codex` to authenticate against your ChatGPT Plus/Pro account.

On first run Datus will:

1. Print a verification URL and a one-time code.
2. Open the URL in your browser — sign in with your ChatGPT account and paste the code.
3. Save the access token locally; subsequent runs reuse and refresh it automatically.

Leave `api_key: ""` in YAML — Datus fills it in at runtime.

## Adding a custom provider

`datus-agent configure` reads its provider list from [`conf/providers.yml`](https://github.com/Datus-ai/Datus-agent/blob/main/conf/providers.yml). To add a provider not in the picker, append an entry to that file and re-run the wizard.

## Auto-applied overrides

The configure wizard writes these required parameters for you:

- `kimi-k2.5` → `temperature: 1.0`, `top_p: 0.95`
- `qwen3-coder-plus` → `temperature: 1.0`, `top_p: 0.95`
