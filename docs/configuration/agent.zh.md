# Agent 配置

Agent 配置定义 Datus Agent 的核心设置，包括默认目标模型与整个系统可用的 LLM 提供方。

## 结构

### 目标模型（target）
在未单独覆盖时，所有节点默认使用 `target` 指向的 LLM 设置：
```yaml
agent:
  target: openai
```

### 模型提供方（models） {#models-configuration}
为智能体配置可用的 LLM 提供方：

**每个提供方条目的必填参数：**

- **提供方键名 (`models.<key>`)** —— 逻辑标识符，由 `agent.target` 和节点 `model` 字段引用（可自定义命名）
- `type`：接口类型（与厂商适配）
- `base_url`：接口基础地址
- `api_key`：访问密钥（支持环境变量）
- `model`：具体模型名

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

!!! tip "环境变量"
    建议通过环境变量管理敏感信息：
    ```yaml
    api_key: ${YOUR_API_KEY}
    # 不建议在生产中明文写入
    api_key: "sk-your-actual-key-here"
    ```

## 支持的提供方

初始化/配置向导中的 provider，最终都会写入 `agent.models`，并通过 `agent.target` 指定默认模型。也就是说：

- 你在 `datus-agent configure` 里选择的 provider 名称，就是 `agent.models.<provider>` 的键名
- 你也可以在后续手动编辑 `agent.yml`，把某个节点单独切换到另一套模型配置

### 通用 provider

| 提供方 | 典型模型 | 接口类型 | 认证方式 |
|---|---|---|---|
| `openai` | `gpt-5.2`、`gpt-4.1`、`o3` | `openai` | API Key |
| `deepseek` | `deepseek-chat`、`deepseek-reasoner` | `deepseek` | API Key |
| `claude` | `claude-sonnet-4-5`、`claude-opus-4-5` | `claude` | API Key |
| `kimi` | `kimi-k2.5`、`kimi-k2-thinking` | `kimi` | API Key |
| `qwen` | `qwen3-max`、`qwen3-coder-plus` | `openai` | API Key |
| `gemini` | `gemini-2.5-flash`、`gemini-2.5-pro` | `gemini` | API Key |
| `minimax` | `MiniMax-M2.7`、`MiniMax-M2.5` | `minimax` | API Key |
| `glm` | `glm-5`、`glm-4.7` | `glm` | API Key |

### 特殊认证 provider

| 提供方 | 接口类型 | 认证方式 | 说明 |
|---|---|---|---|
| `claude_subscription` | `claude` | Claude 订阅 token | 优先自动探测本地订阅凭据，失败时可手动粘贴 `sk-ant-oat01-...` |
| `codex` | `codex` | OAuth | 读取本地 Codex OAuth 凭据并校验连通性 |

### Coding Plan provider

这些 provider 面向编码/规划类 endpoint。虽然它们的名字里带 `coding`，但在配置层面和普通 provider 没有区别，仍然可以作为 `agent.target` 或节点级 `model` 使用。

| 提供方 | 默认模型 | 接口类型 | 说明 |
|---|---|---|---|
| `alibaba_coding` | `qwen3-coder-plus` | `claude` | 阿里云 DashScope 的 Anthropic-compatible coding endpoint |
| `glm_coding` | `glm-5` | `claude` | GLM 的 Anthropic-compatible coding endpoint |
| `minimax_coding` | `MiniMax-M2.7` | `claude` | MiniMax 的 Anthropic-compatible coding endpoint |
| `kimi_coding` | `kimi-for-coding` | `claude` | Kimi 的 coding endpoint |

!!! tip "如何选择 coding plan provider"
    如果你更看重通用问答、SQL 生成和成本控制，通常优先选择常规 provider。

    如果你希望默认模型更偏向规划、代码生成、结构化拆解，或者你会频繁使用 [计划模式](../cli/plan_mode.zh.md)，可以额外配置一个 `*_coding` provider，并按节点切换使用。

### 与环境变量的关系

所有 provider 的 `api_key` 都支持环境变量，例如：

```yaml
api_key: ${OPENAI_API_KEY}
```

对于 OpenAI、DeepSeek、Claude、Kimi、Qwen、Gemini，配置向导会自动提示对应环境变量。对于 `minimax`、`glm` 和各类 `*_coding` provider，你也可以在输入 API Key 时直接填入 `${MINIMAX_API_KEY}`、`${GLM_API_KEY}`、`${KIMI_API_KEY}`、`${DASHSCOPE_API_KEY}` 这类环境变量引用。

另外，当前实现会对少数模型自动补充固定参数覆盖：

- `kimi-k2.5`：自动设置 `temperature: 1.0`、`top_p: 0.95`
- `qwen3-coder-plus`：自动设置 `temperature: 1.0`、`top_p: 0.95`

### 配置示例

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
  api_key: ${CODEX_OAUTH_TOKEN}
  model: codex-mini-latest
  auth_type: oauth
```

## 完整示例
```yaml title="datus-config.yaml"
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
