# Agent

The agent configuration defines the core settings for your Datus Agent, including the target model selection and all available LLM providers that can be used throughout the system.

## Configuration Structure

### LLM Configuration (Two-Tier Provider Model)

LLM selection uses a two-tier system. Most users only need the provider-level configuration; custom entries are for self-hosted or private endpoints.

#### Provider-Level Configuration (Preferred)

Configure credentials under `agent.providers.<name>`. Available models and metadata are loaded from `conf/providers.yml` automatically. Use the [`/model`](../cli/model_command.md) slash command in the CLI to switch between providers and models interactively.

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

Only credentials need to be specified — `type`, `base_url`, and model lists are inherited from `conf/providers.yml`.

#### Custom / Legacy Models

For self-hosted or private-deployment models not covered by `providers.yml`, use `agent.models`:

```yaml
agent:
  models:
    my-internal:
      type: openai
      base_url: https://internal.example.com/v1
      api_key: ${MY_KEY}
      model: internal-gpt-4
```

Custom entries appear under the **Custom** tab in `/model` and can be activated with `/model custom:my-internal`.

### Target Model Selection

The active model is determined by project-level override (`.datus/config.yml`) or the base `agent.target` field. The `/model` command writes to the project override automatically.

#### Project-Level Override (`.datus/config.yml`)

The `/model` command persists selections to `.datus/config.yml` in the current working directory:

```yaml
# Provider-level selection
target:
  provider: openai
  model: gpt-4.1

# Or, custom model selection
target:
  custom: my-internal
```

#### Legacy Target (backwards compatible)

The `agent.target` field in `agent.yml` is still honored when no project-level override exists:

```yaml
agent:
  target: openai  # Legacy: key from agent.models section
```

Resolution order: `.datus/config.yml` → `agent.target`.

### Response Language

`language` is an **optional** field that pins the natural language used by every agentic node for **user-facing** outputs — replies, summaries, clarifying questions, sub-agent prompts issued via `task`, and prose written into files. Code, SQL, identifiers, file paths, URLs, and JSON keys stay in their original form regardless of the setting.

When omitted, no directive is injected into the system prompt and the model picks its own response language per turn.

```yaml
agent:
  # Leave it out entirely to let the model decide.
  # Set a code to pin every agentic node to that language.
  language: zh   # Common codes: en, zh, ja, ko, es, fr, de, pt, ru, it
```

Built-in code → name mapping (injected into the system prompt): `en` → English, `zh` / `zh-cn` → Chinese, `zh-tw` → Traditional Chinese, `ja` → Japanese, `ko` → Korean, `es` → Spanish, `fr` → French, `de` → German, `pt` → Portuguese, `ru` → Russian, `it` → Italian. Unknown codes are used verbatim.

Chat API requests can override this per task by sending a `language` field in the request body (see [Chat API](../API/chat.md)). CLI usage inherits the yaml default.

### Models Configuration (Custom Entries)

The `agent.models` section is used for self-hosted or private-deployment LLM endpoints. For standard providers (OpenAI, DeepSeek, etc.), use `agent.providers` instead.

**Required Parameters per custom entry:**

- **Entry key (`models.<key>`)** — Logical identifier, referenced by `target: {custom: <key>}` or node `model` fields
- **`type`** — Interface type (`openai`, `claude`, `deepseek`, `kimi`, `gemini`, `minimax`, `glm`, `codex`)
- **`base_url`** — API endpoint URL
- **`api_key`** — API key (supports `${ENV_VAR}` substitution)
- **`model`** — Model name / SKU

```yaml
agent:
  models:
    my-internal:
      type: openai
      base_url: https://internal.example.com/v1
      api_key: ${MY_KEY}
      model: internal-gpt-4
```

!!! tip "Environment Variables"
    Use environment variables to securely store API keys and other sensitive information:

    ```yaml
    # Recommended: Using environment variables
    api_key: ${YOUR_API_KEY}

    # Not recommended for production
    api_key: "sk-your-actual-key-here"
    ```

## Supported LLM Providers

Providers are defined in `conf/providers.yml` and activated by adding credentials under `agent.providers`. Use the `/model` command to configure and switch providers interactively.

- Provider credentials live under `agent.providers.<name>` in `agent.yml`
- The active provider/model is stored in `.datus/config.yml` (written by `/model`)
- Node-level `model` overrides can still reference entries in `agent.models` for custom endpoints

### General-purpose providers

| Provider | Typical models | Interface Type | Auth |
|----------|----------------|----------------|------|
| `openai` | `gpt-5.2`, `gpt-4.1`, `o3` | `openai` | API key |
| `deepseek` | `deepseek-chat`, `deepseek-reasoner` | `deepseek` | API key |
| `claude` | `claude-sonnet-4-5`, `claude-opus-4-5` | `claude` | API key |
| `kimi` | `kimi-k2.5`, `kimi-k2-thinking` | `kimi` | API key |
| `qwen` | `qwen3-max`, `qwen3-coder-plus` | `openai` | API key |
| `gemini` | `gemini-2.5-flash`, `gemini-2.5-pro` | `gemini` | API key |
| `minimax` | `MiniMax-M2.7`, `MiniMax-M2.5` | `minimax` | API key |
| `glm` | `glm-5`, `glm-4.7` | `glm` | API key |

### Special-auth providers

| Provider | Interface Type | Auth | Notes |
|----------|----------------|------|-------|
| `claude_subscription` | `claude` | Claude subscription token | The wizard first tries to auto-detect a local subscription credential and otherwise prompts for `sk-ant-oat01-...` |
| `codex` | `codex` | OAuth | Uses locally available Codex OAuth credentials and verifies connectivity |

### Coding Plan providers

These providers target coding/planning-oriented endpoints. Even though their names include `coding`, they are configured exactly like any other model entry and can be used as `agent.target` or referenced from node-level `model` fields.

| Provider | Default model | Interface Type | Notes |
|----------|---------------|----------------|-------|
| `alibaba_coding` | `qwen3-coder-plus` | `claude` | DashScope Anthropic-compatible coding endpoint |
| `glm_coding` | `glm-5` | `claude` | GLM Anthropic-compatible coding endpoint |
| `minimax_coding` | `MiniMax-M2.7` | `claude` | MiniMax Anthropic-compatible coding endpoint |
| `kimi_coding` | `kimi-for-coding` | `claude` | Kimi coding endpoint |

!!! tip "When to choose a coding plan provider"
    If your priority is general chat, SQL generation, or cost efficiency, start with a regular provider.

    If you want a default model that is better aligned with planning, code generation, and structured task decomposition, or if you use [Plan Mode](../cli/plan_mode.md) frequently, add one of the `*_coding` providers and route specific nodes to it.

### Environment variables and model overrides

All providers support environment-variable references in `api_key`, for example:

```yaml
api_key: ${OPENAI_API_KEY}
```

For OpenAI, DeepSeek, Claude, Kimi, Qwen, and Gemini, the configuration wizard can prompt with provider-specific environment variable hints. For `minimax`, `glm`, and the `*_coding` providers, you can still enter values such as `${MINIMAX_API_KEY}`, `${GLM_API_KEY}`, `${KIMI_API_KEY}`, or `${DASHSCOPE_API_KEY}` directly.

The current implementation also auto-applies fixed parameter overrides for a few models:

- `kimi-k2.5`: `temperature: 1.0`, `top_p: 0.95`
- `qwen3-coder-plus`: `temperature: 1.0`, `top_p: 0.95`

### Provider Configuration Examples

With the new provider-level configuration, you only need to set credentials. All other fields (`type`, `base_url`, models list) are inherited from `conf/providers.yml`:

=== "API Key Providers (Minimal)"

    ```yaml
    agent:
      providers:
        openai:
          api_key: ${OPENAI_API_KEY}
        deepseek:
          api_key: ${DEEPSEEK_API_KEY}
        claude:
          api_key: ${ANTHROPIC_API_KEY}
        gemini:
          api_key: ${GEMINI_API_KEY}
        kimi:
          api_key: ${KIMI_API_KEY}
        qwen:
          api_key: ${DASHSCOPE_API_KEY}
    ```

=== "Claude Subscription"

    ```yaml
    agent:
      providers:
        claude_subscription:
          auth_type: subscription
          # Token is auto-detected or entered via /model
    ```

=== "Codex (ChatGPT Plus/Pro)"

    ```yaml
    agent:
      providers:
        codex:
          auth_type: oauth
          # OAuth flow is handled via /model
    ```

=== "Custom Model (agent.models)"

    ```yaml
    agent:
      models:
        my-internal:
          type: openai
          base_url: https://internal.example.com/v1
          api_key: ${MY_KEY}
          model: internal-gpt-4
    ```

!!! note "Legacy Format"
    The previous format with full `type`, `base_url`, `api_key`, and `model` under `agent.models` is still supported for backward compatibility. Existing configurations continue to work without changes.

## Agentic Nodes

`agent.agentic_nodes` is where Datus configures chat and subagent behavior.

This section is used for:

- built-in agentic nodes such as `chat`, `explore`, `gen_sql`, `gen_report`, `gen_dashboard`, and `scheduler`
- custom subagents created with the unified agent TUI (`/agent` or `/subagent`)
- advanced manual aliases that point a custom name at a built-in node class

### Common Fields

The runtime currently reads these commonly used fields from `agentic_nodes` entries:

- `model`: provider key from `agent.models` (custom entries only; omit to inherit the active provider/model)
- `system_prompt`: subagent name / prompt template base name
- `node_class`: node implementation to use, such as `gen_sql`, `gen_report`, `explore`, `gen_table`, `gen_skill`, `gen_dashboard`, or `scheduler`
- `prompt_version`, `prompt_language`
- `agent_description`
- `tools`, `mcp`, `skills`
- `rules`
- `max_turns`
- `workspace_root`
- `scoped_context`
- `subagents`
- `semantic_adapter` for semantic-model and metrics agents
- `bi_platform` for dashboard agents
- `scheduler_service` for scheduler agents

`scoped_kb_path` is deprecated. New configs use shared global storage with query-time filters instead of per-subagent scoped KB directories.

### `subagents` Delegation Control

`subagents` controls whether the node exposes the `task()` tool and which subagent types it may delegate to.

- `subagents: "*"`: allow all discoverable subagents except self
- `subagents: explore, gen_sql`: allow only a named subset
- blank or omitted:
  - `chat` defaults to `*`
  - most other agentic nodes default to `explore`
  - explicitly setting an empty value disables delegation

### `scoped_context`

`scoped_context` limits what the subagent should see from shared metadata and knowledge:

```yaml
scoped_context:
  datasource: finance
  tables: mart.finance_daily, mart.finance_budget
  metrics: finance.revenue.daily_revenue
  sqls: finance.revenue.region_rollup
```

When writing YAML manually, set `datasource` explicitly. The unified agent TUI's Custom-tab wizard fills it from the current database automatically.

### Example

```yaml
agent:
  agentic_nodes:
    chat:
      model: claude
      max_turns: 50
      subagents: "*"

    finance_report:
      node_class: gen_report
      model: claude
      system_prompt: finance_report
      prompt_version: "1.0"
      prompt_language: en
      agent_description: "Finance reporting assistant"
      tools: semantic_tools.*, db_tools.*, context_search_tools.list_subject_tree
      subagents: explore, gen_sql
      max_turns: 30
      scoped_context:
        datasource: finance
        tables: mart.finance_daily
        metrics: finance.revenue.daily_revenue
        sqls: finance.revenue.region_rollup

    sales_dashboard:
      node_class: gen_dashboard
      model: claude
      bi_platform: superset
      max_turns: 30

    semantic_metrics:
      node_class: gen_metrics
      model: claude
      semantic_adapter: metricflow
      max_turns: 30

    etl_scheduler:
      node_class: scheduler
      model: claude
      scheduler_service: airflow_prod
      max_turns: 30
```

## Complete Configuration Example

Here's a comprehensive agent configuration example with the provider-level format:

```yaml title="agent.yml"
agent:
  # Provider credentials (models and metadata from conf/providers.yml)
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
    deepseek:
      api_key: ${DEEPSEEK_API_KEY}
    claude:
      api_key: ${ANTHROPIC_API_KEY}
    gemini:
      api_key: ${GEMINI_API_KEY}
    claude_subscription:
      auth_type: subscription
    codex:
      auth_type: oauth

  # Custom models for self-hosted endpoints (optional)
  models:
    my-internal:
      type: openai
      base_url: https://internal.example.com/v1
      api_key: ${MY_KEY}
      model: internal-gpt-4
```

And the corresponding project-level override:

```yaml title=".datus/config.yml"
target:
  provider: openai
  model: gpt-4.1
default_database: my_duckdb
```
