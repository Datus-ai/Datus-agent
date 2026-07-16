# 模型 API

## 获取可用模型列表

返回 `agent.yml` 中已配置凭据的所有 provider 的汇总模型目录。

### 请求

```http
GET /api/v1/models
```

无需请求体或查询参数。

### 响应

```json
{
  "success": true,
  "data": {
    "models": [
      {
        "provider": "openai",
        "id": "gpt-4.1",
        "name": "GPT-4.1",
        "context_length": 1047576,
        "max_tokens": 32768,
        "pricing": {
          "prompt": "0.000002",
          "completion": "0.000008"
        }
      },
      {
        "provider": "deepseek",
        "id": "deepseek-chat",
        "name": "DeepSeek Chat",
        "context_length": 65536
      }
    ],
    "providers": ["openai", "deepseek"],
    "fetched_at": "2026-04-22T10:30:00Z",
    "source": "cache"
  },
  "errorCode": null,
  "errorMessage": null
}
```

### 响应字段

#### `ModelsData`

| 字段 | 类型 | 说明 |
|------|------|------|
| `models` | `ModelInfo[]` | 所有已配置 provider 的可用模型平铺列表 |
| `providers` | `string[]` | 响应中包含的 provider 标识 |
| `fetched_at` | `string?` | OpenRouter 缓存的 ISO-8601 时间戳（使用本地目录时为 null） |
| `source` | `string` | 数据来源：`"cache"`（来自 OpenRouter）或 `"catalog"`（本地 `providers.yml`） |

#### `ModelInfo`

| 字段 | 类型 | 说明 |
|------|------|------|
| `provider` | `string` | `providers.yml` 中的 provider 标识（例如 `"openai"`、`"deepseek"`） |
| `id` | `string` | SDK 使用的模型标识（例如 `"gpt-4.1"`、`"deepseek-chat"`） |
| `name` | `string?` | 便于阅读的模型名称 |
| `context_length` | `int?` | 最大上下文窗口（token 数） |
| `max_tokens` | `int?` | 最大输出 token 数 |
| `pricing` | `ModelPricing?` | 单 token 价格（如可用） |

#### `ModelPricing`

| 字段 | 类型 | 说明 |
|------|------|------|
| `prompt` | `string?` | 输入 token 单价（美元，以字符串保留以避免舍入） |
| `completion` | `string?` | 输出 token 单价（美元，以字符串保留以避免舍入） |

### 数据来源

模型元数据按以下两级回退顺序解析：

1. **OpenRouter 缓存**（`~/.datus/cache/openrouter_models.json`）— 包含价格和上下文长度等最完整的数据；通过 OpenRouter API 自动刷新，超时时间为 8 秒。
2. **Provider 目录**（`conf/providers.yml`）— 静态模型列表，通过 `model_specs` 提供 `context_length` 和 `max_tokens`；缓存不可用时使用。

### 过滤规则

只返回已配置凭据的 provider。满足以下任一条件即视为可用：

- `api_key` 已设置（非空，且不是 `${...}` 占位符）
- `auth_type` 为 `oauth` 或 `subscription`，且 token 有效

未配置凭据的 provider 不会出现在响应中。
