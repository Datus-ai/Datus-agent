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

| 提供方 | 模型 | 接口类型 |
|---|---|---|
| OpenAI | GPT-4、GPT-5 等 | `openai` |
| Anthropic | Claude 4（Sonnet、Opus）、Claude 3 | `claude` |
| Google | Gemini（Pro、Flash、Ultra） | `gemini` |
| DeepSeek | Chat / Reasoning | `deepseek` |
| Kimi | Moonshot Kimi | `openai` |
| Qwen | 阿里通义千问 | `openai` |

### 配置示例

=== "OpenAI"
```yaml
openai:
  type: openai
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: gpt-4-turbo
```

=== "Anthropic Claude"
```yaml
anthropic:
  type: claude
  base_url: https://api.anthropic.com
  api_key: ${ANTHROPIC_API_KEY}
  model: claude-sonnet-4-20250514
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
google:
  type: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
  api_key: ${GEMINI_API_KEY}
  model: gemini-2.5-flash
```

=== "Kimi (Moonshot)"
```yaml
kimi:
  type: openai
  base_url: https://api.moonshot.cn/v1
  api_key: ${KIMI_API_KEY}
  model: kimi-k2-turbo-preview
```

=== "Qwen (Alibaba)"
```yaml
qwen:
  type: openai
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: ${QWEN_API_KEY}
  model: qwen-turbo
```

=== "Azure OpenAI"
```yaml
azure_openai:
  type: openai
  base_url: https://${AZURE_OPENAI_ENDPOINT}/openai/deployments/${AZURE_DEPLOYMENT_NAME}
  api_key: ${AZURE_OPENAI_API_KEY}
  model: gpt-4
```

## 完整示例
```yaml title="datus-config.yaml"
agent:
  target: openai

models:
  openai:
    type: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: gpt-4-turbo

  google:
    type: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta
    api_key: ${GEMINI_API_KEY}
    model: gemini-2.5-flash

  anthropic:
    type: claude
    base_url: https://api.anthropic.com
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-sonnet-4-20250514

  deepseek_v3:
    type: deepseek
    base_url: https://api.deepseek.com
    api_key: ${DEEPSEEK_API_KEY}
    model: deepseek-chat

  azure_openai:
    type: openai
    base_url: https://${AZURE_OPENAI_ENDPOINT}/openai/deployments/${AZURE_DEPLOYMENT_NAME}
    api_key: ${AZURE_OPENAI_API_KEY}
    model: gpt-4
```
