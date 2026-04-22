# Models API

## List Available Models

Returns the consolidated model catalog for all providers that have credentials configured in `agent.yml`.

### Request

```
GET /api/v1/models
```

No request body or query parameters required.

### Response

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

### Response Fields

#### `ModelsData`

| Field | Type | Description |
|-------|------|-------------|
| `models` | `ModelInfo[]` | Flat list of available models across all configured providers |
| `providers` | `string[]` | Provider keys represented in this response |
| `fetched_at` | `string?` | ISO-8601 timestamp of the OpenRouter cache (null when source is catalog) |
| `source` | `string` | Data origin: `"cache"` (OpenRouter-derived) or `"catalog"` (local `providers.yml`) |

#### `ModelInfo`

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `string` | Provider key from `providers.yml` (e.g., `"openai"`, `"deepseek"`) |
| `id` | `string` | Model slug as consumed by the SDK (e.g., `"gpt-4.1"`, `"deepseek-chat"`) |
| `name` | `string?` | Human-readable model name |
| `context_length` | `int?` | Maximum context window in tokens |
| `max_tokens` | `int?` | Maximum completion tokens |
| `pricing` | `ModelPricing?` | Per-token pricing when available |

#### `ModelPricing`

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `string?` | Price per input token (USD, preserved as string to avoid rounding) |
| `completion` | `string?` | Price per output token (USD, preserved as string to avoid rounding) |

### Data Sources

Model metadata is resolved with a two-tier fallback:

1. **OpenRouter cache** (`~/.datus/cache/openrouter_models.json`) — richest data including pricing and context lengths, auto-refreshed from the OpenRouter API with an 8-second timeout.
2. **Provider catalog** (`conf/providers.yml`) — static model list with `model_specs` for context_length and max_tokens. Used when the cache is unavailable.

### Filtering

Only providers with configured credentials are included. A provider is considered available when:

- Its `api_key` is set (non-empty, not a `${...}` placeholder)
- Or its `auth_type` is `oauth` or `subscription` with valid tokens

Providers without credentials are excluded from the response.
