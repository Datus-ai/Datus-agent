# Subagent 命令设计: `/gen_semantic_model`, `/gen_metrics`, `/gen_sql_summary`

## 1. 概述

本文档描述了为 Datus CLI 添加三个新 subagent 命令的设计。这些命令将提供交互式的对话界面，用于生成语义模型、指标和 SQL 摘要。

### 现状分析

**现有 Subagent 实现：**
- CLI 目前支持 `/subagent <message>` 语法（如 `/chatbot What is...`）
- Subagent 在 `agent.yml` 的 `agentic_nodes` 中定义
- 每个 subagent 使用 `GenSQLAgenticNode` 作为基础实现
- Subagent 从配置中动态检测并自动补全

**设计目标：**
- 提供交互式的对话界面用于生成语义模型、指标和 SQL 摘要
- **在一次对话中完成：生成 YAML → 展示 → 编辑 → 确认 → 同步到数据库**
- 支持 Console 内编辑 YAML 内容
- 使用 Hooks 机制在 Agent 执行流程中插入用户交互

## 2. 提议的 Subagent 命令

### 2.1 `/gen_semantic_model` - 语义模型生成

**用途：** 从表定义交互式生成语义模型

**配置 (agent.yml)：**
```yaml
agentic_nodes:
  gen_semantic_model:
    model: deepseek
    system_prompt: gen_semantic_model
    prompt_version: "1.0"
    tools: db_tools.*, generation_tools.*
    hooks: generation_hooks.GenerationHooks  # 关键：使用 Hooks 实现交互式编辑
    mcp: filesystem_mcp
    workspace_root: ${MF_MODEL_PATH}
    agent_description: "语义模型生成助手"
    rules:
      - 使用 get_table_ddl 工具获取表结构，LLM 生成 MetricFlow YAML
      - 使用 filesystem_mcp 的 write_file 工具保存文件
      - 使用 metricflow_mcp 的 mf validate-configs 工具验证配置
      - 一次只处理一个表的语义模型
      - Hooks 会自动处理展示、编辑、确认和同步流程
```

**使用示例：**
```bash
# 单表生成
Datus> /gen_semantic_model 为 orders 表生成语义模型

# 多表生成
Datus> /gen_semantic_model 为 customers 和 orders 表创建模型

# 使用上下文注入
Datus> /gen_semantic_model @catalog/database/schema/orders 生成语义模型
```

**实现细节（基于 Hooks 的交互式流程）：**
1. **LLM 调用工具**：通过 `get_table_ddl` (from db_tools) 获取表结构，LLM 生成 YAML 内容
2. **MCP 保存文件**：LLM 使用 filesystem_mcp 的 `write_file` 工具保存 YAML 文件
3. **MCP 验证配置**：（可选）使用 metricflow_mcp 的 `mf validate-configs` 工具验证
4. **Hooks 拦截**：`GenerationHooks.on_tool_end()` 拦截 MCP `write_file` 工具的返回结果
5. **展示内容**：Hooks 在 Console 展示生成的 YAML 内容（语法高亮）
6. **用户交互**：Hooks 提示用户选择：
   - 选项 1: Accept and sync - 接受并同步到 LanceDB
   - 选项 2: Edit content - 进入编辑模式
   - 选项 3: Cancel - 取消操作
7. **编辑模式**（选项 2）：
   - Hooks 使用系统默认编辑器（vim/nano/etc）打开 YAML 文件
   - 用户编辑完成后验证 YAML 语法
   - 提示用户确认：Save to RAG / Edit again / Cancel
8. **自动同步**（选项 1 或编辑后选择 Save）：
   - Hooks 根据文件类型调用相应的同步方法：
     - `_sync_semantic_models_to_db()` - 同步语义模型
     - `_sync_metrics_to_db()` - 同步指标定义
     - `_sync_sql_history_to_db()` - 同步 SQL 历史
   - 同步方法直接调用 LanceDB storage 的 `store()` / `store_batch()` 方法
9. **返回结果**：显示同步成功信息，一次对话完成所有步骤

**完整工作流示例（一次对话）：**
```bash
Datus> /gen_semantic_model 为 orders 表生成语义模型

AI> ⏳ 正在生成语义模型...

============================================================
✅ Generated YAML for: /path/to/orders.yml
============================================================
semantic_model:
  name: orders
  description: Order transaction data
  identifiers:
    - name: order_id
      type: primary
  dimensions:
    - name: order_date
      type: time
    - name: customer_id
      type: foreign
  measures:
    - name: order_amount
      agg: sum
============================================================

Options:
  1. Accept and sync
  2. Edit content
  3. Cancel

Your choice (1-3): 1

# 如果选择 edit:
📝 YAML Editor - Edit the content below
============================================================
Instructions:
  • Edit the YAML content
  • Press ESC + Enter to finish editing
  • Type 'cancel' on a new line to abort
============================================================

[多行编辑界面，带 YAML 语法高亮]

Confirm changes and sync to database? (yes/no): yes

✅ Saved to: /path/to/orders.yml
⏳ Syncing to LanceDB...
✅ Successfully synced to LanceDB!
   Synced 1 semantic model

AI> 完成！已为 orders 表生成语义模型并同步到数据库。
```

### 2.2 `/gen_metrics` - 指标定义生成

**用途：** 从 SQL 查询或语义模型交互式生成指标

**配置 (agent.yml)：**
```yaml
agentic_nodes:
  gen_metrics:
    model: deepseek
    system_prompt: gen_metrics
    prompt_version: "1.0"
    tools: ""
    hooks: generation_hooks.GenerationHooks  # 使用 Hooks 实现交互式编辑
    mcp: filesystem_mcp
    workspace_root: ${MF_MODEL_PATH}
    agent_description: "指标定义生成助手"
    rules:
      - 分析用户提供的 SQL 查询内容
      - 生成对应的 MetricFlow 指标定义
      - 使用 filesystem_mcp 的 write_file 工具保存文件
      - Hooks 会自动处理展示、编辑、确认和同步流程
```

**使用示例：**
```bash
# 从 SQL 查询生成
Datus> /gen_metrics 从这个 SQL 创建指标: SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id

# 从自然语言生成
Datus> /gen_metrics 为 orders 表生成收入指标

# 使用语义模型引用
Datus> /gen_metrics 使用 orders 和 customers 语义模型定义转化指标
```

**实现细节（基于 Hooks 的交互式流程）：**
与 `/gen_semantic_model` 相同，通过 `GenerationHooks` 实现一次对话完成整个流程。

**完整工作流示例（一次对话）：**
```bash
Datus> /gen_metrics 从这个 SQL 创建指标: SELECT SUM(amount) FROM orders

AI> ⏳ 正在分析 SQL 查询...
AI> ⏳ 正在生成指标...

============================================================
✅ Generated YAML for: /path/to/order_metrics.yml
============================================================
metric:
  name: total_order_amount
  description: Sum of all order amounts
  constraint: ""
  sql_query: SELECT SUM(amount) FROM orders
============================================================

Options:
  1. Accept and sync
  2. Edit content
  3. Cancel

Your choice (1-3): 1

✅ Saved to: /path/to/order_metrics.yml
⏳ Syncing to LanceDB...
✅ Successfully synced to LanceDB!
   Synced 1 metric

AI> 完成！已生成指标并同步到数据库。数据库当前共有 42 个指标。
```

### 2.3 `/gen_sql_summary` - SQL 历史摘要

**用途：** 交互式 SQL 查询分析和知识提取

**配置 (agent.yml)：**
```yaml
agentic_nodes:
  gen_sql_summary:
    model: deepseek
    system_prompt: gen_sql_summary
    prompt_version: "1.0"
    tools: ""
    hooks: generation_hooks.GenerationHooks  # 使用 Hooks 实现交互式编辑
    mcp: filesystem_mcp
    workspace_root: ./storage/sql_history
    agent_description: "SQL 历史分析助手"
    rules:
      - 分析用户提供的 SQL 查询内容
      - 生成标准化的 YAML 摘要格式
      - 使用 filesystem_mcp 的 write_file 工具保存文件
      - YAML 格式与 LanceDB sql_history 表 schema 一致
      - Hooks 会自动处理展示、编辑、确认和同步流程
```

**使用示例：**
```bash
# 分析当前 SQL
Datus> /gen_sql_summary 总结我刚运行的查询

# 从文件批量分析
Datus> /gen_sql_summary 分析 queries.sql 文件中的所有查询

# 自然语言查询
Datus> /gen_sql_summary 我们的 SQL 历史中最常见的销售分析模式是什么？
```

**YAML 文件格式示例：**
```yaml
# storage/sql_history/monthly_sales_001.yml
id: monthly_sales_001
name: Monthly Sales Report
sql: |
  SELECT
    DATE_TRUNC('month', order_date) as month,
    SUM(amount) as total_sales
  FROM orders
  WHERE order_date >= '2024-01-01'
  GROUP BY 1
  ORDER BY 1
comment: Calculate monthly sales totals for 2024
summary: Aggregates order amounts by month, filtering for current year data
filepath: reports/monthly_sales.sql
domain: sales
layer1: reporting
layer2: monthly_aggregation
tags: sales, monthly, aggregation, time_series
```

**实现细节（基于 Hooks 的交互式流程）：**
与 `/gen_semantic_model` 相同，通过 `GenerationHooks` 实现一次对话完成整个流程。

**完整工作流示例（一次对话）：**
```bash
Datus> /gen_sql_summary 总结这个查询: SELECT DATE_TRUNC('month', order_date) as month, SUM(amount) FROM orders GROUP BY 1

AI> ⏳ 正在分析 SQL 查询...

============================================================
✅ Generated YAML for: /path/to/monthly_sales_001.yml
============================================================
sql_history:
  id: monthly_sales_001
  name: Monthly Sales Report
  sql: |
    SELECT DATE_TRUNC('month', order_date) as month,
           SUM(amount) FROM orders GROUP BY 1
  comment: Calculate monthly sales totals
  summary: Aggregates order amounts by month
  filepath: ""
  domain: sales
  layer1: reporting
  layer2: monthly_aggregation
  tags: monthly, aggregation, time_series
============================================================

Options:
  1. Accept and sync
  2. Edit content
  3. Cancel

Your choice (1-3): 1

✅ Saved to: /path/to/monthly_sales_001.yml
⏳ Syncing to LanceDB...
✅ Successfully synced to LanceDB!
   Synced 1 SQL history entry

AI> 完成！已生成 SQL 摘要并同步到数据库。数据库当前共有 128 条 SQL 历史记录。
```

## 3. 架构与实现

### 3.1 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI 层 (repl.py)                          │
│  - 命令解析: /gen_semantic_model, /gen_metrics 等           │
│  - 从 agent.yml 检测 subagent                                │
│  - 流式输出显示                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            ChatCommands (chat_commands.py)                   │
│  - execute_chat_command(message, subagent_name)              │
│  - Node 生命周期管理                                         │
│  - Session 和 context 处理                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         GenSQLAgenticNode (gen_sql_agentic_node.py)         │
│  - Tool 设置 (根据 subagent 类型)                              │
│  - Hooks 加载和集成 (GenerationHooks)                        │
│  - MCP server 集成 (filesystem, metricflow)                  │
│  - Template context 准备                                     │
│  - 流式执行                                                  │
└─────────────────────────────────────────────────────────────┘
                  │                           │
                  │                           │ (LLM 调用工具/MCP)
                  ▼                           ▼
   ┌──────────────────────┐    ┌─────────────────────────────┐
   │  GenerationHooks     │    │   Tools + MCP Servers       │
   │ (Hooks 拦截层)       │    │                             │
   │                      │    │  [gen_semantic_model]       │
   │  on_tool_end():      │◄───│  • generation_tools.*           │
   │  1. 拦截 MCP 返回结果│    │  • db_tools.*              │
   │  2. 展示 YAML        │    │                             │
   │  3. 用户交互选择     │    │  [gen_metrics / gen_sql_summary]│
   │     - Accept/Edit    │    │  • 无 native tools           │
   │     - Cancel         │    │                             │
   │  4. 编辑模式         │    │  [Common MCP Tools]         │
   │     (多行编辑器)     │    │  • filesystem_mcp: write_file│
   │  5. 验证 YAML        │    │  • metricflow_mcp: validate  │
   │  6. 重新保存文件     │    │                             │
   │  7. 自动同步到 LanceDB│    │  [LLM 职责]                │
   │                      │    │  • 生成 YAML 内容            │
   └──────────────────────┘    └─────────────────────────────┘
             │b
             ▼ (调用 _sync_* 内部方法)
   ┌──────────────────────┐
   │   LanceDB            │
   │   Vector Store       │
   └──────────────────────┘
```

### 3.2 Native Tools API 设计

**GenerationTools 类 (参考 ContextSearchTools)：**
```python
# datus/tools/generation_tools.py
from typing import List
from agents import Tool
from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import rag_by_configuration
from datus.tools.tools import FuncToolResult, trans_to_function_tool

class GenerationTools:
    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self.metrics_rag = rag_by_configuration(agent_config)

    def available_tools(self) -> List[Tool]:
        """
        提供语义模型检查工具供 LLM 调用。

        注意：
        - get_table_ddl 已在 DBFuncTool (datus/tools/tools.py) 中实现
        - gen_semantic_model 通过 db_tools.* 访问 get_table_ddl
        - GenerationTools 只负责语义模型存在性检查
        """
        return [
            trans_to_function_tool(func)
            for func in (
                self.check_semantic_model_exists,
            )
        ]

    def check_semantic_model_exists(
        self,
        table_name: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> FuncToolResult:
        """
        Check if semantic model already exists in LanceDB.

        Use this tool when you need to:
        - Avoid generating duplicate semantic models
        - Check if a table already has semantic model definition

        Args:
            table_name: Name of the database table
            catalog_name: Catalog name (optional)
            database_name: Database name (optional)
            schema_name: Schema name (optional)

        Returns:
            dict: Check results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'exists' (bool): Whether semantic model exists
                    - 'file_path' (str): Path to existing semantic model file if exists
        """
```

**注意：get_table_ddl 工具说明**

`get_table_ddl` 工具已在 `datus/tools/tools.py` 的 `DBFuncTool` 类中实现：

```python
# datus/tools/tools.py
class DBFuncTool:
    def get_table_ddl(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Get complete DDL definition for a database table.

        Use this tool when you need to:
        - Generate semantic models (LLM needs complete DDL for accurate generation)
        - Understand table structure including constraints, indexes, and relationships
        - Analyze foreign key relationships for semantic model generation

        Args:
            table_name: Name of the database table
            catalog: Optional catalog name to filter tables
            database: Optional database name to filter tables
            schema_name: Optional schema name to filter tables

        Returns:
            dict: DDL results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'ddl' (str): Complete CREATE TABLE DDL statement
                    - 'table_info' (dict): Table metadata including catalog, database, schema
        """
```

**gen_semantic_model 配置使用方式：**
```yaml
gen_semantic_model:
  tools: db_tools.*, generation_tools.*
  # db_tools.* 包含 get_table_ddl
  # generation_tools.* 包含 check_semantic_model_exists
```

### 3.3 与现有代码集成
        """
        Generate metrics YAML file from SQL query.

        Use this tool when you need to:
        - Create MetricFlow metrics from SQL
        - Auto-generate semantic models if missing
        - Define simple or derived metrics

        Args:
            sql_query: SQL query to analyze
            table_names: Optional list of table names (auto-extracted if not provided)
            domain: Business domain
            layer1: Primary category
            layer2: Secondary category

        Returns:
            dict: Generation results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'yaml_file_path': Path to generated YAML file
                    - 'yaml_content': YAML file content as string for display
                    - 'metrics_generated': Number of metrics generated
        """
        # 实现逻辑参考 generate_metrics_node.py

    def generate_sql_summary(
        self,
        sql_query: str,
        filepath: str = "",
        domain: str = "",
        layer1: str = "",
        layer2: str = "",
    ) -> FuncToolResult:
        """
        Generate SQL summary YAML file for SQL history.

        Use this tool when you need to:
        - Analyze SQL queries and extract metadata
        - Generate structured summaries with classification
        - Store SQL knowledge for future reuse

        Args:
            sql_query: SQL query to analyze
            filepath: Source file path (optional)
            domain: Business domain
            layer1: Primary category
            layer2: Secondary category

        Returns:
            dict: Generation results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'yaml_file_path': Path to generated YAML file
                    - 'yaml_content': YAML file content as string for display
                    - 'summary_data': Dict with id, name, tags, etc.
        """
        # 实现逻辑参考 sql_history_init.py 中的 analyze_sql_history

    # ============================================================
    # 内部方法：供 GenerationHooks 调用，不暴露为 LLM tools
    # ============================================================

    def _sync_semantic_models_to_db(
        self,
        yaml_directory: str,
        file_pattern: str = "*.yml",
    ) -> dict:
        """
        内部方法：同步语义模型 YAML 文件到 LanceDB。

        此方法由 GenerationHooks 自动调用，不需要暴露为 tool。

        Args:
            yaml_directory: YAML 文件目录
            file_pattern: 文件匹配模式 (默认 "*.yml")

        Returns:
            dict: 同步结果，包含 success, error, result 字段
        """
        # 实现逻辑参考 metrics_init.py 的 init_semantic_yaml_metrics
        pass

    def _sync_metrics_to_db(
        self,
        yaml_directory: str,
        file_pattern: str = "*.yml",
    ) -> dict:
        """
        内部方法：同步指标 YAML 文件到 LanceDB。

        此方法由 GenerationHooks 自动调用，不需要暴露为 tool。

        Args:
            yaml_directory: YAML 文件目录
            file_pattern: 文件匹配模式 (默认 "*.yml")

        Returns:
            dict: 同步结果，包含 success, error, result 字段
        """
        # 实现逻辑参考 metrics_init.py 的 init_semantic_yaml_metrics
        pass

    def _sync_sql_history_to_db(
        self,
        yaml_file: str,
    ) -> dict:
        """
        内部方法：同步 SQL 历史 YAML 文件到 LanceDB。

        此方法由 GenerationHooks 自动调用，不需要暴露为 tool。

        Args:
            yaml_file: YAML 文件路径

        Returns:
            dict: 同步结果，包含 success, error, result 字段
        """
        # 实现逻辑参考 sql_history_init.py 的 init_sql_history
        pass
```

### 3.3 GenerationHooks 实现

**核心类：** `datus/cli/generation_hooks.py`

```python
from agents import AgentHooks
from datus.cli.yaml_editor import edit_yaml_multiline
import yaml
import os

class GenerationHooks(AgentHooks):
    """
    Hooks for generation tools to enable interactive YAML editing.

    This class intercepts tool execution and provides:
    1. Display generated YAML content
    2. Allow user to accept/edit/cancel
    3. Multi-line YAML editor with syntax highlighting
    4. YAML validation
    5. Auto-sync to LanceDB
    """

    def __init__(self, agent_config):
        super().__init__()
        self.agent_config = agent_config

    async def on_tool_end(self, context, agent, tool, result):
        """
        拦截 MCP 文件工具的返回结果，提供交互式编辑流程。

        工作流程：
        1. 检查是否是 filesystem_mcp.write_file 工具
        2. 展示生成的 YAML 内容
        3. 提示用户选择（接受/编辑/取消）
        4. 如果编辑：打开多行编辑器
        5. 验证 YAML 语法
        6. 重新保存文件
        7. 自动同步到 LanceDB
        """
        # 只处理文件保存工具
        if tool.name not in [
            "write_file",
            "edit_file"
        ]:
            return result

        # 检查是否是 YAML 文件
        file_path = result.result.get("path", "")
        if not file_path.endswith(('.yml', '.yaml')):
            return result

        # 获取刚写入的文件内容
        yaml_content = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
        except Exception:
            return result

        if not yaml_content:
            return result

        # 展示生成的 YAML
        print("\n" + "="*60)
        print(f"✅ Generated YAML for: {file_path}")
        print("="*60)
        print(yaml_content)
        print("="*60 + "\n")

        # 用户交互
        print("Options:")
        print("  1. Accept and sync")
        print("  2. Edit content")
        print("  3. Cancel")

        choice = input("\nYour choice (1-3): ").strip()

        if choice == '3':
            raise CancelledError("User cancelled the operation")

        final_content = yaml_content

        # 选项 1: 直接接受
        if choice == '1':
            pass  # 使用原始内容，不做修改

        # 选项 2: 编辑内容
        elif choice == '2':
            # 打开多行编辑器
            edited_content, confirmed = edit_yaml_multiline(yaml_content)

            if not confirmed:
                raise CancelledError("User cancelled after editing")

            # 验证 YAML 语法
            try:
                yaml.safe_load(edited_content)
            except yaml.YAMLError as e:
                print(f"❌ Invalid YAML syntax: {e}")
                raise ValueError(f"Invalid YAML: {e}")

            final_content = edited_content

        # 无效输入：默认为接受
        else:
            print(f"⚠️  Invalid choice '{choice}', defaulting to accept")
            pass  # 使用原始内容

        # 保存最终内容到文件
        with open(file_path, 'w') as f:
            f.write(final_content)

        print(f"\n✅ Saved to: {file_path}")

        # 自动同步到 LanceDB
        print("⏳ Syncing to LanceDB...")

        # 根据文件类型选择同步方法
        try:
            if "semantic_model" in file_path or "models" in file_path:
                sync_result = self._sync_semantic_models_to_db(
                    yaml_directory=os.path.dirname(file_path)
                )
            elif "metric" in file_path or "metrics" in file_path:
                sync_result = self._sync_metrics_to_db(
                    yaml_directory=os.path.dirname(file_path)
                )
            elif "sql_history" in file_path or "sql_history" in os.path.dirname(file_path):
                sync_result = self._sync_sql_history_to_db(
                    yaml_file=file_path
                )
            else:
                sync_result = {"success": 0, "error": "Unknown file type"}

            if sync_result.get("success"):
                print(f"✅ Successfully synced to LanceDB!")
                message = sync_result.get("result", {}).get("message", "")
                if message:
                    print(f"   {message}")
            else:
                print(f"❌ Sync failed: {sync_result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Sync failed: {str(e)}")
            sync_result = {"success": 0, "error": str(e)}

        # 返回修改后的结果
        result.result["yaml_content"] = final_content
        result.result["synced"] = sync_result.get("success", 0) == 1

        return result
```

### 3.4 Console YAML 编辑器实现

**核心模块：** `datus/cli/yaml_editor.py`

```python
from prompt_toolkit import prompt
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.data import YamlLexer

def edit_yaml_multiline(initial_content: str) -> tuple[str, bool]:
    """
    Allow user to edit YAML content in console with multi-line input.

    使用 prompt-toolkit 提供：
    - 多行输入
    - YAML 语法高亮（Pygments）
    - ESC + Enter 完成编辑

    Args:
        initial_content: Initial YAML content to edit

    Returns:
        (edited_content, confirmed): Tuple of edited content and confirmation
    """
    print("\n" + "="*60)
    print("📝 YAML Editor - Edit the content below")
    print("="*60)
    print("Instructions:")
    print("  • Edit the YAML content")
    print("  • Press ESC + Enter to finish editing")
    print("  • Type 'cancel' on a new line to abort")
    print("="*60 + "\n")

    print("Current content:")
    print("-" * 60)
    print(initial_content)
    print("-" * 60 + "\n")

    try:
        edited = prompt(
            "Enter new YAML content:\n",
            multiline=True,
            lexer=PygmentsLexer(YamlLexer),
            default=initial_content,
        )

        if edited.strip().lower() == 'cancel':
            return initial_content, False

        # 确认
        confirm = prompt("\nConfirm changes and sync to database? (yes/no): ")
        confirmed = confirm.strip().lower() in ['yes', 'y']

        return edited, confirmed

    except KeyboardInterrupt:
        return initial_content, False
```

**依赖安装：**

```toml
# pyproject.toml
[tool.poetry.dependencies]
prompt-toolkit = "^3.0.0"  # 多行输入和交互
pygments = "^2.0.0"        # 语法高亮
```

### 3.5 系统提示词模板

**位置：** `datus/prompts/`

1. `generate_semantic_model_system.j2` (现有，如需要可更新)
2. `generate_metrics_system.j2` (现有，如需要可更新)
3. `generate_sql_summary_system.j2` (新)

**模板结构示例：**
```jinja2
你是 {{ agent_description }}。

## 可用工具
- 原生工具: {{ native_tools }}
- MCP 服务器: {{ mcp_tools }}

## 工作空间
- 根路径: {{ workspace_root }}
- 命名空间: {{ namespace }}

## 规则
{% for rule in rules %}
- {{ rule }}
{% endfor %}

## 重要提示
- Hooks 会自动处理 YAML 展示、编辑和同步流程
- 你需要使用 get_table_ddl 获取表结构，然后生成 YAML 内容
- 使用 filesystem_mcp 的 write_file 工具保存文件
- 使用 metricflow_mcp 的 validate-configs 工具验证配置
- 用户交互由 Hooks 层完成，不在对话中进行

## 任务
分析用户的请求，使用可用的工具生成所需的输出。
```

### 3.4 与现有代码集成

**需要的改动：**

1. **新增核心文件：**
   - `datus/tools/generation_tools.py` - 新的 GenerationTools 类
   - `datus/cli/generation_hooks.py` - GenerationHooks 类
   - `datus/cli/yaml_editor.py` - 多行 YAML 编辑器
   - `datus/prompts/gen_semantic_model_system.j2` - 简化的系统提示词
   - `datus/prompts/gen_metrics_system.j2` - 简化的系统提示词
   - `datus/prompts/gen_sql_summary_system.j2` - 新的系统提示词

2. **agent.yml 配置：**
   - 在 `agentic_nodes` 下添加三个新条目
   - gen_semantic_model: `tools: db_tools.*, generation_tools.*`
   - gen_metrics: `tools: ""`
   - gen_sql_summary: `tools: ""`

3. **GenSQLAgenticNode 扩展：**
   - 在 `_setup_tool_pattern` 中添加 `generation_tools` 的处理
   - 添加 hooks 加载和集成逻辑

4. **现有 Node 保留：**
   - `generate_semantic_model_node.py` - 保留现有实现
   - `generate_metrics_node.py` - 保留现有实现
   - 未来可以将 `!gen_semantic_model` 等命令改为调用 native tool

5. **无需改动：**
   - CLI 命令解析（已支持动态 subagent）
   - ChatCommands（已处理 subagent 执行）
   - Storage init 流程（metrics_init.py, sql_history_init.py 保持不变）


## 5. 实施步骤

### 阶段 1: 实现核心组件
- 创建 `datus/tools/generation_tools.py`
  - 实现 `check_semantic_model_exists` 工具
  - 实现 `get_table_ddl` 工具
- 创建 `datus/cli/generation_hooks.py`
  - 实现 `GenerationHooks` 类
  - 拦截 MCP 文件工具结果
  - 处理用户交互和自动同步
- 创建 `datus/cli/yaml_editor.py`
  - 实现多行 YAML 编辑器
  - 添加依赖：prompt-toolkit, pygments

### 阶段 2: 配置 Subagent
- 在 `agent.yml` 中添加三个 agentic_nodes 配置
- 每个配置添加 `hooks: generation_hooks.GenerationHooks`
- 创建/更新系统提示词模板
- 更新 `GenSQLAgenticNode` 以支持 hooks 加载

### 阶段 3: 测试与文档
- 单元测试：测试生成和同步逻辑
- 集成测试：端到端测试完整流程
- 用户文档：编写使用指南和示例
- 性能测试：验证响应时间和资源使用

## 6. 示例工作流（一次对话完成）

### 工作流 1: 快速生成并同步语义模型
```bash
Datus> /gen_semantic_model 为 orders 表生成语义模型

AI> ⏳ 正在生成语义模型...

============================================================
✅ Generated YAML for: /workspace/orders.yml
============================================================
semantic_model:
  name: orders
  description: Order transaction data
  identifiers:
    - name: order_id
      type: primary
  dimensions:
    - name: order_date
      type: time
    - name: customer_id
      type: foreign
  measures:
    - name: order_amount
      agg: sum
============================================================

Options:
  1. Accept and sync
  2. Edit content
  3. Cancel

Your choice (1-3): 1

✅ Saved to: /workspace/orders.yml
⏳ Syncing to LanceDB...
✅ Successfully synced to LanceDB!
   Synced 1 semantic model

AI> 完成！已为 orders 表生成语义模型并同步到数据库。
```

### 工作流 2: 编辑后同步指标
```bash
Datus> /gen_metrics 从这个 SQL 创建指标: SELECT SUM(amount) FROM orders

AI> ⏳ 正在生成指标...

============================================================
✅ Generated YAML for: /workspace/order_metrics.yml
============================================================
metric:
  name: total_order_amount
  description: Sum of all order amounts
  constraint: ""
  sql_query: SELECT SUM(amount) FROM orders
============================================================

Options:
  1. Accept and sync
  2. Edit content
  3. Cancel

Your choice (1-3): 2

📝 YAML Editor - Edit the content below
============================================================
Instructions:
  • Edit the YAML content
  • Press ESC + Enter to finish editing
  • Type 'cancel' on a new line to abort
============================================================

[用户在多行编辑器中修改 description]

Confirm changes and sync to database? (yes/no): yes

✅ Saved to: /workspace/order_metrics.yml
⏳ Syncing to LanceDB...
✅ Successfully synced to LanceDB!
   Synced 1 metric

AI> 完成！已生成并同步 1 个指标到数据库。
```

### 工作流 3: SQL 历史分析（一次完成）
```bash
Datus> /gen_sql_summary 总结这个查询: SELECT DATE_TRUNC('month', order_date) as month, SUM(amount) FROM orders GROUP BY 1

AI> ⏳ 正在分析 SQL 查询...

============================================================
✅ Generated YAML for: /workspace/monthly_sales_001.yml
============================================================
sql_history:
  id: monthly_sales_001
  name: Monthly Sales Report
  sql: |
    SELECT DATE_TRUNC('month', order_date) as month,
           SUM(amount) FROM orders GROUP BY 1
  comment: Calculate monthly sales totals
  summary: Aggregates order amounts by month
  domain: sales
  layer1: reporting
  layer2: monthly_aggregation
  tags: monthly, aggregation, time_series
============================================================

Options:
  1. Accept and sync
  2. Edit content
  3. Cancel

Your choice (1-3): 1

✅ Saved to: /workspace/monthly_sales_001.yml
⏳ Syncing to LanceDB...
✅ Successfully synced to LanceDB!
   Synced 1 SQL history entry

AI> 完成！已生成 SQL 摘要并同步到数据库。数据库当前共有 129 条 SQL 历史记录。

# 后续可以直接搜索使用
Datus> / 我们的 SQL 历史中常见的月度报表模式是什么？
AI> [使用 search_historical_sql 工具搜索]
AI> 找到 5 个相关查询，常见模式：DATE_TRUNC('month', ...) 和 GROUP BY 月份
```


---

## 附录：实现要点总结

### 核心架构变更

**从 Node 实现 → Subagent + MCP + Hooks**

- **旧架构：** 每个功能是独立的 Node (generate_semantic_model_node.py, generate_metrics_node.py)
- **新架构：**
  - gen_semantic_model: GenerationTools + MCP Servers + Hooks
  - gen_metrics/gen_sql_summary: 纯 LLM 生成 + MCP Servers + Hooks
- **优势：**
  - 交互式对话体验
  - 用户可以编辑和确认生成的内容
  - 一次对话完成整个流程
  - 保留现有 Node 实现

### 工作流程

**一体化对话流程（推荐）**
- 用户通过 `/gen_semantic_model`, `/gen_metrics`, `/gen_sql_summary` 对话
- LLM 调用生成工具生成 YAML 文件
- 用户继续对话："同步到数据库"
- LLM 调用同步工具导入 LanceDB
- 全程在对话中完成，无需退出运行命令

**分离式工作流（可选）**
- 阶段 1: 使用 subagent 生成 YAML 文件
- 用户手动编辑和检查 YAML
- 阶段 2: 使用 subagent 同步到数据库，或运行 `bootstrap_kb` 批量同步

**优势：**
- 灵活：可以一气呵成，也可以分步进行
- 可检查：YAML 文件持久化，用户可以检查和修改
- 可批量：支持批量同步多个文件


### 实现参考

- **GenerationTools 结构：** 参考 `datus/tools/context_search.py`
- **Semantic Model 逻辑：** 参考 `datus/storage/metric/metrics_init.py`
- **SQL Summary 逻辑：** 参考 `datus/storage/sql_history/sql_history_init.py`
- **Tool 注册：** 参考 `GenSQLAgenticNode._setup_tool_pattern()`