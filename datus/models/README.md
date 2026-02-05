# datus/models/

## Overview

The `datus/models/` module provides a unified multi-LLM integration layer that abstracts different language model providers behind a common interface. It solves the challenge of working with multiple LLM APIs (OpenAI, Claude, DeepSeek, Qwen, Gemini) while providing advanced features like Model Context Protocol (MCP) support, session management, streaming responses, and robust error handling.

The module uses a factory pattern for dynamic model instantiation, supports both synchronous and asynchronous operations, and integrates seamlessly with the datus-agent workflow system. Notable technologies include OpenAI Agents Python for MCP integration, SQLite for session persistence, and LangSmith for tracing.

## File Structure & Capabilities

### Core Architecture Files

**`base.py`** - Abstract base class and factory
- `LLMBaseModel`: Abstract base class defining the unified interface
- Factory method `create_model()` for dynamic model instantiation based on configuration
- Session management interface with SQLite-backed persistence
- Abstract methods for generation, JSON output, tool integration, and streaming

**`openai_compatible.py`** - Shared base for OpenAI-compatible APIs
- `OpenAICompatibleModel`: Base class for models using OpenAI-compatible endpoints
- Unified error handling and retry logic with exponential backoff
- MCP server integration with connection management
- Streaming support with action history tracking
- LiteLLM integration for unified token counting (supports 100+ models)
- Model info retrieval and context length management

### Provider Implementations

Currently, only Claude uses a separate implementation; all other models inherit from OpenAICompatibleModel.

**`openai_model.py`** - OpenAI GPT models
- Inherits from `OpenAICompatibleModel`
- Uses LiteLLM for unified token counting across all providers
- Supports structured output and reasoning content

**`claude_model.py`** - Anthropic Claude models
- Custom implementation for Claude's native API
- Anthropic tool format conversion for MCP integration
- Prompt caching support for improved performance
- Using prompt cache manually

**`deepseek_model.py`** - DeepSeek models
- OpenAI-compatible implementation via LiteLLM
- LLM trace saving functionality for debugging
- Support for reasoning models (DeepSeek R1)

**`qwen_model.py`** - Qwen models
- LiteLLM integration for unified token counting
- Dashscope API integration

**`gemini_model.py`** - Google Gemini models
- Google Generative AI client integration
- LiteLLM integration for unified token counting

### Utility Modules

**`session_manager.py`** - Multi-turn conversation management
- `SessionManager`: Manages SQLite-backed conversation sessions
- Persistent storage in `{agent.home}/sessions/` (fixed path, configure via agent.home in agent.yml)
- Session lifecycle management (create, clear, delete, list)

**`mcp_utils.py`** - Model Context Protocol utilities
- `multiple_mcp_servers()`: Async context manager for MCP server lifecycle (This should be changed later)
- Connection retry logic and error handling
- Safe server startup and cleanup

**`mcp_result_extractors.py`** - MCP result processing
- `extract_sql_contexts()`: Extracts SQL execution results from MCP agent runs
- Database-specific function mapping (Snowflake, SQLite, StarRocks, DuckDB)
- Reflection and reasoning content extraction

## Key Interface Methods
### `generate(prompt, enable_thinking=False, **kwargs) -> str`
### `generate_with_json_output(prompt, **kwargs) -> Dict`
### async `generate_with_tools(prompt, mcp_servers=None, tools=None, **kwargs) -> Dict`

### async `generate_with_tools_stream(prompt, tools, mcp_servers, instruction, output_type,
        max_turns, action_history_manager, **kwargs) -> AsyncGenerator[ActionHistory, None]:
)

## How to use this module

### Basic Model Usage

```python
from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel

# Load configuration
config = AgentConfig.from_file("conf/agent.yml")

# Create model using factory method
model = LLMBaseModel.create_model(config, model_name="gpt-4")

# Basic text generation
response = model.generate("Explain quantum computing")
print(response)

# Structured JSON output
json_response = model.generate_with_json_output(
    "Generate a summary with title and content",
    response_format={"type": "json_object"}
)
print(json_response)
```

### MCP Tool Integration (Async)

```python
import asyncio
from datus.models.base import LLMBaseModel
from agents.mcp import MCPServerStdio

async def use_mcp_tools():
    # Initialize model
    model = LLMBaseModel.create_model(config, "claude-3-5-sonnet")

    # Setup MCP servers (database tools, etc.)
    mcp_servers = {
        "snowflake": MCPServerStdio(
            command="mcp-server-snowflake",
            env={"SNOWFLAKE_USER": "user", "SNOWFLAKE_PASSWORD": "pass"}
        )
    }

    # Generate with tool access
    result = await model.generate_with_tools(
        prompt="Show me the top 10 customers by revenue",
        mcp_servers=mcp_servers,
        instruction="You are a data analyst. Use available tools to query data.",
        output_type=str,
        max_turns=10
    )

    print("Response:", result["content"])
    print("SQL Contexts:", result["sql_contexts"])

# Run async function
asyncio.run(use_mcp_tools())
```

### Structured Output with Pydantic Models

The SDK 0.7.0 supports structured output using Pydantic models. This provides automatic validation and eliminates the need for JSON format instructions in prompts.

```python
import asyncio
from datus.models.base import LLMBaseModel
from datus.schemas import GenSqlOutput

async def generate_sql_with_structure():
    model = LLMBaseModel.create_model(config, "gpt-4")

    # Use Pydantic model for structured output
    result = await model.generate_with_tools(
        prompt="Generate a query for top 10 customers by revenue",
        output_type=GenSqlOutput,  # Pydantic model
        strict_json_schema=True,  # Enable strict JSON mode (default)
        instruction="You are a SQL expert. Generate SQL queries.",
        max_turns=10
    )

    # result["content"] is now a GenSqlOutput object
    sql_result = result["content"]
    print(f"SQL: {sql_result.sql}")
    print(f"Explanation: {sql_result.explanation}")
    print(f"Tables: {sql_result.tables}")

asyncio.run(generate_sql_with_structure())
```

**Benefits of Structured Output**:
- No need for JSON format instructions in prompts
- Automatic response validation against Pydantic schema
- Type-safe output handling
- Better error messages when output doesn't match schema

**Available Output Types** (from `datus.schemas`):
- `GenSqlOutput` - SQL generation with explanation and tables
- `SemanticModelGenerationOutput` - Semantic model generation
- `MetricGenerationOutput` - Metric generation
- `SqlSummaryGenerationOutput` - SQL summary generation
- `ReportGenerationOutput` - Report generation
- `ExtKnowledgeGenerationOutput` - External knowledge generation
- `CompareOutput` - SQL comparison analysis

### Required Environment Variables

```bash
# API Keys (choose based on models used)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="sk-..."
export QWEN_API_KEY="sk-..."
export GEMINI_API_KEY="..."

# Optional tracing
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
export LANGSMITH_PROJECT="datus-agent"
```

### Session Management

```python
# Create session for multi-turn conversation
session = model.create_session("user_123")

# Use session in tool calls
result = await model.generate_with_tools(
    prompt="What tables are available?",
    mcp_servers=mcp_servers,
    session=session  # Maintains conversation context
)

# List and manage sessions
sessions = model.list_sessions()
model.clear_session("user_123")
model.delete_session("user_123")
```

## How to contribute to this module

### Adding a New LLM Model

To add support for a new LLM provider (e.g., "NewModel"), follow these steps:

1. **Create the model implementation**:

```python
# datus/models/newmodel_model.py
import os
from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel

class NewModelModel(OpenAICompatibleModel):
    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)

    def _get_api_key(self) -> str:
        api_key = self.model_config.api_key or os.environ.get("NEWMODEL_API_KEY")
        if not api_key:
            raise ValueError("NewModel API key required")
        return api_key

    def _get_base_url(self) -> str:
        return self.model_config.base_url or "https://api.newmodel.com/v1"

    # No need to override token_count - it's inherited from OpenAICompatibleModel
    # which uses LiteLLM's unified token counter for accurate counting
    # across all model providers
```

2. **Update the base model registry**:

```python
# datus/models/base.py - Add to MODEL_TYPE_MAP
MODEL_TYPE_MAP: ClassVar[Dict[str, str]] = {
    LLMProvider.DEEPSEEK: "DeepSeekModel",
    LLMProvider.QWEN: "QwenModel",
    LLMProvider.OPENAI: "OpenAIModel",
    LLMProvider.CLAUDE: "ClaudeModel",
    LLMProvider.GEMINI: "GeminiModel",
    LLMProvider.NEWMODEL: "NewModelModel",  # Add this line
}
```

3. **Add provider constant**:

```python
# datus/utils/constants.py
class LLMProvider:
    OPENAI = "openai"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    GEMINI = "gemini"
    NEWMODEL = "newmodel"  # Add this line
```

4. **Update configuration schema**:

```yaml
# conf/agent.yml - Add model configuration
models:
  newmodel-base:
    type: newmodel
    model: newmodel-1.0
    api_key: ${NEWMODEL_API_KEY}
    base_url: https://api.newmodel.com/v1
    temperature: 0.7
    max_tokens: 4096
```

### Related Modules to Update

When adding a new model, consider updating these related modules:

- **`datus/configuration/agent_config.py`**: Add model validation if needed
- **`datus/cli/`**: Update CLI commands to support the new provider
- **`tests/`**: Add comprehensive tests for the new model
- **Documentation**: Update main README and configuration examples

### Testing Your New Model

```python
# tests/test_newmodel_model.py
import pytest
from datus.models.newmodel_model import NewModelModel
from datus.configuration.agent_config import ModelConfig

def test_newmodel_basic_generation():
    config = ModelConfig(
        type="newmodel",
        model="newmodel-1.0",
        api_key="test-key"
    )
    model = NewModelModel(config)

    # Mock the API call for testing
    response = model.generate("Hello, world!")
    assert isinstance(response, str)
    assert len(response) > 0
```

The modular architecture makes adding new models straightforward while maintaining consistency across the unified interface.

## Regression Testing

The models module includes comprehensive test coverage to ensure reliability across all LLM providers and core functionality.

### Running Basic Acceptance Tests

For quick validation of core functionality across all model implementations:

```bash
# Run basic acceptance tests for all models (quick validation)
pytest tests/test_*_model.py -m acceptance -q
```

This runs a focused subset of tests marked with the `@pytest.mark.acceptance` decorator, providing fast feedback on essential functionality like:
- Model instantiation and configuration
- Basic text generation
- JSON output parsing
- Error handling for common failure cases

### Running Full Test Suite

For comprehensive testing including edge cases, streaming, MCP integration, and session management:

```bash
# Run all model tests with verbose output
pytest tests/test_*_model.py -v
```

### Test Environment Setup

Before running tests, ensure you have the required API keys configured:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export QWEN_API_KEY="sk-..."
export GEMINI_API_KEY="..."
```