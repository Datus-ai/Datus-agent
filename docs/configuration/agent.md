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

**Required Parameters:**

- **`provider`** - The LLM provider name (you can name it as needed, not required to match the actual provider name)
- **`type`** - Interface type corresponding to LLM manufacturers
- **`base_url`** - Base address of the model provider's API endpoint
- **`api_key`** - API key for accessing the LLM service (supports environment variables)
- **`model`** - Specific model name to use from the provider

```yaml
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

Datus Agent supports a wide range of LLM providers through standardized interfaces:

| Provider | Models | Interface Type |
|----------|--------|----------------|
| **OpenAI** | GPT-4, GPT-5, and other OpenAI models | `openai` |
| **Anthropic** | Claude 4 (Sonnet, Opus) and Claude 3 models | `claude` |
| **Google** | Gemini models (Pro, Flash, Ultra) | `gemini` |
| **DeepSeek** | DeepSeek Chat and DeepSeek Reasoning models | `deepseek` |
| **Kimi** | Moonshot AI's Kimi models | `openai` |
| **Qwen** | Alibaba's Qwen series models | `openai` |

### Provider Configuration Examples

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
      type: openai  # Uses OpenAI-compatible interface
      base_url: https://api.moonshot.cn/v1
      api_key: ${KIMI_API_KEY}
      model: kimi-k2-turbo-preview
    ```

=== "Qwen (Alibaba)"

    ```yaml
    qwen:
      type: openai  # Uses OpenAI-compatible interface
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

## Complete Configuration Example

Here's a comprehensive agent configuration example with multiple providers:

```yaml title="datus-config.yaml"
# Complete Datus Agent Configuration
agent:
  target: openai  # Default model for all operations

models:
  # Production OpenAI configuration
  openai:
    type: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: gpt-4-turbo
    
  # Alternative Google Gemini
  google:
    type: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta
    api_key: ${GEMINI_API_KEY}
    model: gemini-2.5-flash
    
  # Anthropic Claude for reasoning
  anthropic:
    type: claude
    base_url: https://api.anthropic.com
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-sonnet-4-20250514
    
  # Cost-effective DeepSeek
  deepseek_v3:
    type: deepseek
    base_url: https://api.deepseek.com
    api_key: ${DEEPSEEK_API_KEY}
    model: deepseek-chat
    
  # Azure OpenAI (Enterprise)
  azure_openai:
    type: openai
    base_url: https://${AZURE_OPENAI_ENDPOINT}/openai/deployments/${AZURE_DEPLOYMENT_NAME}
    api_key: ${AZURE_OPENAI_API_KEY}
    model: gpt-4
```
