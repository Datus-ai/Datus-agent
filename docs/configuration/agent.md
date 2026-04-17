# Agent

The agent configuration defines the core settings for your Datus Agent, including the target model selection and all available LLM providers that can be used throughout the system.

## Configuration Structure

### Target Model

The `target` field specifies the default LLM configuration that will be used across all nodes unless explicitly overridden.

```yaml
agent:
  target: openai  # Default model configuration key from models section
```

### Models Configuration

Configure LLM providers that your agent can use. Each model configuration includes the provider type, API endpoints, credentials, and specific model names.

**Required Parameters per provider entry:**

- **Provider key (`models.<key>`)** - Logical provider identifier, referenced by `agent.target` and node `model` fields (you can name it as needed)
- **`type`** - Interface type corresponding to LLM manufacturers
- **`base_url`** - Base address of the model provider's API endpoint
- **`api_key`** - API key for accessing the LLM service (supports environment variables)
- **`model`** - Specific model name to use from the provider

```yaml
agent:
  target: provider_name
  models:
    provider_name:
      type: provider_type
      base_url: https://api.example.com/v1
      api_key: ${API_KEY_ENV_VAR}
      model: model-name
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

Providers selected in `datus-agent configure` are written into `agent.models`, and `agent.target` points to the default one. In practice:

- The provider name you choose in `datus-agent configure` becomes the key under `agent.models.<provider>`
- You can later edit `agent.yml` manually and bind different nodes to different model entries

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
      model: claude-sonnet-4-5
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
      model: gemini-2.5-flash
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

## Agentic Nodes

`agent.agentic_nodes` is where Datus configures chat and subagent behavior.

This section is used for:

- built-in agentic nodes such as `chat`, `explore`, `gen_sql`, `gen_report`, `gen_dashboard`, and `scheduler`
- custom subagents created with `.subagent`
- advanced manual aliases that point a custom name at a built-in node class

### Common Fields

The runtime currently reads these commonly used fields from `agentic_nodes` entries:

- `model`: provider key from `agent.models`
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
- `bi_platform` for dashboard agents

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
  namespace: finance
  tables: mart.finance_daily, mart.finance_budget
  metrics: finance.revenue.daily_revenue
  sqls: finance.revenue.region_rollup
```

When writing YAML manually, set `namespace` explicitly. The `.subagent` wizard fills it from the current database automatically.

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
        namespace: finance
        tables: mart.finance_daily
        metrics: finance.revenue.daily_revenue
        sqls: finance.revenue.region_rollup

    sales_dashboard:
      node_class: gen_dashboard
      model: claude
      bi_platform: superset
      max_turns: 30
```

## Complete Configuration Example

Here's a comprehensive agent configuration example with multiple providers:

```yaml title="datus-config.yaml"
# Complete Datus Agent Configuration
agent:
  target: alibaba_coding
  models:
    openai:
      type: openai
      base_url: https://api.openai.com/v1
      api_key: ${OPENAI_API_KEY}
      model: gpt-5.2

    gemini:
      type: gemini
      base_url: https://generativelanguage.googleapis.com/v1beta
      api_key: ${GEMINI_API_KEY}
      model: gemini-2.5-flash

    claude:
      type: claude
      base_url: https://api.anthropic.com
      api_key: ${ANTHROPIC_API_KEY}
      model: claude-sonnet-4-5

    deepseek:
      type: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-chat

    alibaba_coding:
      type: claude
      base_url: https://coding-intl.dashscope.aliyuncs.com/apps/anthropic
      api_key: ${DASHSCOPE_API_KEY}
      model: qwen3-coder-plus
      temperature: 1.0
      top_p: 0.95
```
