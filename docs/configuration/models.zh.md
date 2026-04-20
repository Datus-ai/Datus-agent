# 模型

`agent.models` 用于注册 Datus Agent 可调用的 LLM 提供方，`agent.target` 指定默认使用哪一个。每个节点和 subagent 都可以通过自身的 `model:` 字段单独覆盖。

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

## 默认模型（`agent.target`）

`agent.target` 指向 `agent.models` 中的某一个 key，节点在未单独指定 `model:` 时会使用它，必须能在 `models` 映射里找到对应条目。

```yaml
agent:
  target: openai   # 必须是 agent.models 中的某个键名
```

节点级覆盖（例如 `agentic_nodes.chat.model: claude`）的优先级始终高于 `target`。节点级配置详见 **[Agent](agent.md)**。

## 必填字段

`agent.models.<key>` 下的每个条目都需要：

- **`<key>`** —— 任意名称，由 `agent.target` 和节点的 `model:` 字段引用。
- **`type`** —— provider 类型，使用 [下方示例](#configuration-examples) 中显示的值。
- **`base_url`** —— 服务端 API 地址。
- **`api_key`** —— 凭据，建议使用 `${ENV_VAR}` 而非明文。
- **`model`** —— 服务端的具体模型名。

## 可选字段

| 字段 | 默认值 | 说明 |
|---|---|---|
| `temperature` | provider 默认值 | 采样温度。 |
| `top_p` | provider 默认值 | nucleus sampling 阈值。 |
| `auth_type` | `api_key` | 取值 `api_key`、`subscription`、`oauth`，详见 [认证模式](#authentication-modes)。 |
| `enable_thinking` | `false` | 启用支持的 provider 的推理/思考模式。 |
| `max_retry` | `3` | 流式连接错误重试次数。 |
| `retry_interval` | `2.0` | 重试间隔（秒）。 |

## 支持的提供方

下面这些键名也是 `datus-agent configure` 向导写入 `agent.models` 的默认 key。

### 通用 provider

| 提供方 | 典型模型 | 认证方式 |
|---|---|---|
| `openai` | `gpt-5.2`、`gpt-4.1`、`o3` | API Key |
| `deepseek` | `deepseek-chat`、`deepseek-reasoner` | API Key |
| `claude` | `claude-sonnet-4-6`、`claude-opus-4-6`、`claude-haiku-4-5` | API Key |
| `kimi` | `kimi-k2.5`、`kimi-k2-thinking` | API Key |
| `qwen` | `qwen3-max`、`qwen3-coder-plus` | API Key |
| `gemini` | `gemini-3-pro-preview`、`gemini-3-flash-preview` | API Key |
| `minimax` | `MiniMax-M2.7`、`MiniMax-M2.5` | API Key |
| `glm` | `glm-5`、`glm-4.7` | API Key |

### 多模型聚合 gateway

| 提供方 | 典型模型 | 认证方式 |
|---|---|---|
| `openrouter` | `openrouter/anthropic/claude-sonnet-4-6`、`openrouter/openai/gpt-5.2`、`openrouter/google/gemini-3-pro-preview`（300+ 款） | API Key |

### 特殊认证 provider

| 提供方 | 认证方式 | 说明 |
|---|---|---|
| `claude_subscription` | Claude Pro/Max 订阅 token | 优先复用本地 `claude setup-token` 的凭据，否则可手动粘贴 `sk-ant-oat01-…`。 |
| `codex` | OAuth（PKCE + Device Code） | 通过 ChatGPT Plus/Pro 的 Codex；token 持久化在 `~/.datus/oauth/`。 |

### Coding Plan provider

这些条目指向各家厂商的 coding/planning endpoint。配置方式与普通 provider 完全一致，可作为 `agent.target` 也可作为节点级 `model`。

| 提供方 | 默认模型 | 说明 |
|---|---|---|
| `alibaba_coding` | `qwen3-coder-plus` | DashScope coding endpoint |
| `glm_coding` | `glm-5` | GLM coding endpoint |
| `minimax_coding` | `MiniMax-M2.7` | MiniMax coding endpoint |
| `kimi_coding` | `kimi-for-coding` | Kimi coding endpoint |

!!! tip "如何选择 coding plan provider"
    如果你更看重通用问答、SQL 生成和成本控制，优先选择常规 provider。

    如果默认模型需要更偏向规划、代码生成和结构化拆解，或者你会频繁使用 [计划模式](../cli/plan_mode.zh.md)，可以追加一个 `*_coding` provider 并按节点切换。

## 配置示例

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

=== "GLM (智谱)"

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

    模型名遵循 `openrouter/<provider>/<model>` 约定，完整列表见 [openrouter.ai/models](https://openrouter.ai/models)。

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

## 认证模式 {#authentication-modes}

### `api_key`（默认）

直接填 API Key 或使用 `${ENV_VAR}`。绝大多数 provider 都用这种方式。

### `subscription` —— Claude Pro/Max

`claude_subscription` 使用。Datus 会自动复用 `claude setup-token` 写入的本地 token；也可以设置 `CLAUDE_CODE_OAUTH_TOKEN` 环境变量，或者直接把 token 粘贴到 `api_key`。

### `oauth` —— Codex

`codex` 使用，用于以 ChatGPT Plus/Pro 账号认证。

首次运行时 Datus 会：

1. 输出验证 URL 和一次性 code。
2. 在浏览器中打开 URL，使用 ChatGPT 账号登录并粘贴 code。
3. 把 access token 保存到本地；后续运行会自动复用并续期。

YAML 中 `api_key: ""` 留空即可，Datus 会在运行时自动填入。

## 添加自定义 provider

`datus-agent configure` 的 provider 列表来自 [`conf/providers.yml`](https://github.com/Datus-ai/Datus-agent/blob/main/conf/providers.yml)。如果想新增不在选单里的 provider，往该文件追加一段并重新运行向导即可。

## 自动参数覆盖 {#auto-applied-parameter-overrides}

配置向导会替你写入以下强制参数：

- `kimi-k2.5` → `temperature: 1.0`、`top_p: 0.95`
- `qwen3-coder-plus` → `temperature: 1.0`、`top_p: 0.95`
