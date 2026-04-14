# Adapter 生成指南

## 概述

Adapter 生成功能帮助你为外部平台创建 Datus adapter 项目脚手架。通过 AI 助手，你可以生成完整的 adapter 项目骨架、获得平台特定逻辑的实现指导，并通过合约测试验证结果——所有步骤都在一个交互式工作流中完成。

## 什么是 Adapter？

**Adapter** 是一个 Python 包，用于将 Datus 连接到外部平台。Datus 支持四种 adapter 类型：

| 类型 | 基类 | 示例平台 | 包命名 |
|------|------|---------|--------|
| **semantic** | `BaseSemanticAdapter` | Cube, Looker, dbt Semantic Layer | `datus_semantic_{platform}` |
| **bi** | `BIAdapterBase` | Superset, Metabase, Grafana | `datus_bi_{platform}` |
| **db** | `BaseSqlConnector` | ClickHouse, Snowflake, BigQuery | `datus_{platform}` |
| **scheduler** | `BaseSchedulerAdapter` | Airflow, Dagster, Prefect | `datus_scheduler_{platform}` |

每种 adapter 类型都有定义好的接口契约，规定了需要实现的方法、返回类型和注册机制。

## 快速开始

启动 Datus CLI（`datus --database <namespace>`），然后调用 gen_adapter 子代理：

```bash
/gen_adapter 为 Cube 平台生成一个 semantic adapter
```

或者用自然语言描述你的集成需求——chat agent 会自动委派给 gen_adapter：

```bash
我需要把 Cube 作为语义层平台接入进来
```

## 工作原理

### 生成流程

```
用户描述平台 → Agent 确认意图 → 生成项目骨架 → 辅助实现 →
静态校验 → 运行合约测试 → 迭代修复直到通过 → 总结
```

工作流分为五个阶段：

### 阶段 1：确认意图

Agent 从你的请求中识别 adapter 类型、平台名称和输出目录，在生成任何文件之前先请求确认。

**交互示例：**

```
用户：为 Cube 生成一个 semantic adapter

Agent：我将创建一个 Cube 平台的 semantic adapter：
  - Adapter 类型: semantic
  - 平台: cube
  - 输出目录: ./datus-semantic-cube
  - 包名: datus_semantic_cube

是否继续？[通过 ask_user 确认]
```

### 阶段 2：生成骨架

确认后，agent 调用 `scaffold_adapter` 生成完整项目：

```
datus-semantic-cube/
├── datus_semantic_cube/
│   ├── __init__.py          # register() 函数 + 导出
│   ├── adapter.py           # CubeAdapter 类（方法桩）
│   ├── config.py            # CubeConfig（Pydantic 模型）
│   └── py.typed             # PEP 561 标记
├── tests/
│   ├── conftest.py          # 共享测试 fixtures
│   └── unit/
│       ├── test_adapter.py  # 基础实例化测试
│       └── test_contract.py # 合约测试套件（仅 semantic）
├── pyproject.toml           # 包元数据 + 入口点
└── README.md
```

生成的 `adapter.py` 包含抛出 `NotImplementedError` 的方法桩：

```python
from datus_semantic_core import BaseSemanticAdapter, DimensionInfo, MetricDefinition, QueryResult, ValidationResult

class CubeAdapter(BaseSemanticAdapter):
    """Cube adapter for Datus."""

    def __init__(self, config: "CubeConfig"):
        super().__init__(config)

    async def list_metrics(self, path=None, limit=100, offset=0) -> List[MetricDefinition]:
        raise NotImplementedError("TODO: Implement list_metrics")

    async def get_dimensions(self, metric_name, path=None) -> List[DimensionInfo]:
        raise NotImplementedError("TODO: Implement get_dimensions")

    async def query_metrics(self, metrics, dimensions=None, ...) -> QueryResult:
        raise NotImplementedError("TODO: Implement query_metrics")

    async def validate_semantic(self) -> ValidationResult:
        raise NotImplementedError("TODO: Implement validate_semantic")
```

### 阶段 3：辅助实现

Agent 逐个帮助实现方法桩：

1. **收集平台知识** — 通过 `web_search_document` 搜索官方 API 文档
2. **提出实现方案** — 将平台 API 映射到 Datus 接口要求
3. **写入代码** — 用户确认后，通过 `write_file` / `edit_file` 更新 `adapter.py` 和 `config.py`
4. **填写合约测试 factory**（仅 semantic）— 在 `tests/unit/test_contract.py` 中填入 mock fixtures

### 阶段 4：验证

验证分两步进行：

**4a. 静态校验**（`validate_adapter`）：

- 模块可成功导入
- 导出了 `register()` 函数
- 存在 Adapter 类
- 没有方法仍然抛出 `NotImplementedError`

**4b. 合约测试执行**（仅 semantic，`run_adapter_pytest`）：

```bash
# Agent 内部执行：
run_adapter_pytest(
    project_dir="/absolute/path/to/datus-semantic-cube",
    test_subpath="tests/unit/test_contract.py"
)
```

合约测试套件（来自 `datus_semantic_core.testing`）强制执行以下检查：

| 合约 | 断言 |
|------|------|
| `list_metrics` 返回类型 | `list[MetricDefinition]` |
| `MetricDefinition.dimensions` | `list[str]`（非 DimensionInfo） |
| `get_dimensions` 返回类型 | `list[DimensionInfo]` |
| `query_metrics` 返回类型 | `QueryResult`，`.data` 为 `list[dict]` |
| `query_metrics(dry_run=True)` | 设置 `metadata['dry_run']` 或包含 `'sql'` 列 |
| `validate_semantic` 返回类型 | `ValidationResult` |

如果测试失败，agent 读取失败输出、修复代码并重新运行——循环直到所有测试通过。

### 阶段 5：总结

Agent 展示已完成 adapter 的摘要：创建/修改的文件、合约测试结果以及剩余的 TODO。

## 配置

大部分配置是内置的。在 `agent.yml` 中只需最少配置：

```yaml
agentic_nodes:
  gen_adapter:
    model: claude        # 可选：默认使用已配置的模型
    max_turns: 30        # 可选：默认为 30
```

**内置配置**（自动启用）：

- **脚手架工具**：`scaffold_adapter`、`validate_adapter`、`list_adapter_types`
- **文件系统工具**：`read_file`、`read_multiple_files`、`write_file`、`edit_file`、`list_directory`
- **文档工具**：`list_document_nav`、`get_document`、`search_document`、`web_search_document`
- **测试运行器**：`run_adapter_pytest`
- **交互模式**：`ask_user`（仅交互模式）

### 配置选项

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `model` | 否 | 使用的 LLM 模型 | 使用默认配置的模型 |
| `max_turns` | 否 | 最大对话轮次 | 30 |

## 可用工具

| 工具 | 说明 |
|------|------|
| `list_adapter_types` | 列出所有支持的 adapter 类型及元数据 |
| `scaffold_adapter` | 生成完整的 adapter 项目骨架 |
| `validate_adapter` | 静态校验：导入、register()、adapter 类、桩检测 |
| `run_adapter_pytest` | 在生成的 adapter 项目中运行 pytest（受限范围） |
| `read_file` / `write_file` / `edit_file` | 读取和修改生成的 adapter 代码 |
| `list_directory` | 浏览项目文件结构 |
| `web_search_document` | 搜索外部平台 API 文档 |
| `ask_user` | 与用户确认决策（仅交互模式） |

## 使用示例

### 示例 1：Semantic Adapter（Cube）

**用户输入：**
```bash
/gen_adapter 为 Cube 平台生成一个 semantic adapter
```

**Agent 执行：**
1. 确认：semantic adapter，平台 "cube"，输出目录
2. 生成 `datus-semantic-cube/`（11 个文件）
3. 搜索 Cube REST API 文档
4. 使用 Cube 的 `/v1/meta` 端点实现 `list_metrics`
5. 使用 `/v1/meta` dimension 元数据实现 `get_dimensions`
6. 使用 Cube 的 `/v1/load` 端点实现 `query_metrics`
7. 使用 `/v1/meta` 健康检查实现 `validate_semantic`
8. 在 `test_contract.py` factory 中填入 mock HTTP 响应
9. 运行合约测试——全部通过
10. 报告总结

### 示例 2：BI Adapter（Metabase）

**用户输入：**
```bash
/gen_adapter 创建一个 Metabase 的 BI adapter
```

**Agent 执行：**
1. 确认：bi adapter，平台 "metabase"
2. 生成 `datus-bi-metabase/`（10 个文件）
3. 实现 `platform_name`、`list_dashboards`、`get_dashboard_info`、`list_charts`、`list_datasets`
4. 校验——所有方法已实现
5. 报告总结

### 示例 3：数据库 Adapter

**用户输入：**
```bash
/gen_adapter 为 ClickHouse 生成一个数据库 adapter
```

**Agent 执行：**
1. 确认：db adapter，平台 "clickhouse"
2. 生成 `datus-clickhouse/`（10 个文件）
3. 实现 `execute`、`test_connection`、`get_databases`
4. 校验——所有方法已实现

## Adapter 接口契约

### Semantic Adapter（4 个方法）

| 方法 | 签名 | 返回类型 |
|------|------|---------|
| `list_metrics` | `(self, path=None, limit=100, offset=0)` | `List[MetricDefinition]` |
| `get_dimensions` | `(self, metric_name, path=None)` | `List[DimensionInfo]` |
| `query_metrics` | `(self, metrics, dimensions=None, path=None, time_start=None, time_end=None, time_granularity=None, where=None, limit=None, order_by=None, dry_run=False)` | `QueryResult` |
| `validate_semantic` | `(self)` | `ValidationResult` |

### BI Adapter（5 个方法）

| 方法 | 签名 | 返回类型 |
|------|------|---------|
| `platform_name` | `(self)` | `str` |
| `list_dashboards` | `(self, search='', page_size=20)` | `list` |
| `get_dashboard_info` | `(self, dashboard_id)` | `object` |
| `list_charts` | `(self, dashboard_id)` | `list` |
| `list_datasets` | `(self, dashboard_id)` | `list` |

### DB Adapter（3 个方法）

| 方法 | 签名 | 返回类型 |
|------|------|---------|
| `execute` | `(self, input_params, result_format=None)` | `object` |
| `test_connection` | `(self)` | `bool` |
| `get_databases` | `(self, catalog_name='', include_sys=False)` | `list` |

### Scheduler Adapter（5 个方法）

| 方法 | 签名 | 返回类型 |
|------|------|---------|
| `platform_name` | `(self)` | `str` |
| `test_connection` | `(self)` | `bool` |
| `submit_job` | `(self, payload)` | `object` |
| `get_job` | `(self, job_id)` | `object` |
| `list_jobs` | `(self, project=None, status=None, limit=50, offset=0)` | `list` |

## 合约测试（仅 Semantic）

Semantic adapter 会自动生成接入 `datus_semantic_core.testing.make_semantic_contract_suite` 的合约测试。这个共享测试套件在不重复断言逻辑的情况下强制执行接口规范。

实现 adapter 后，要让合约测试通过：

1. 在 `tests/unit/test_contract.py` 中填写 `factory()` 函数——构造一个 mock 了 HTTP/SDK 层的测试用 adapter
2. 将 `SAMPLE_METRIC_NAME` 和 `SAMPLE_DIMENSION_NAME` 设置为你的 mock fixtures 实际暴露的值
3. 运行：`pytest tests/unit/test_contract.py -v`

### factory 示例：

```python
from unittest.mock import AsyncMock

async def factory() -> CubeAdapter:
    config = CubeConfig(api_url="http://localhost:4000", api_token="test-token")
    adapter = CubeAdapter(config)
    # Mock HTTP 层，填入 fixture 响应
    adapter._http_get = AsyncMock(return_value={
        "cubes": [{"name": "orders", "measures": [...], "dimensions": [...]}]
    })
    return adapter
```

## 安全约束

`run_adapter_pytest` 工具被有意限制了范围：

- `project_dir` 必须是绝对路径且包含 `pyproject.toml`
- `test_subpath` 必须以 `tests/` 开头，不能包含 `..`
- 不接受自由 pytest 参数——参数固定为 `-q --tb=short --no-header`
- 每次调用硬超时 120 秒
- 输出截断为 8KB 尾部，防止上下文溢出

## 总结

Adapter 生成功能提供：

- 4 种 adapter 类型的自动化项目脚手架（semantic、bi、db、scheduler）
- AI 辅助实现，支持平台 API 文档搜索
- 静态校验（导入、register、桩检测）
- 合约测试生成和执行（semantic adapter）
- 迭代修复闭环：生成 → 实现 → 测试 → 修复 → 重测
- 安全范围限定的 pytest 运行器
- 交互式工作流，在关键决策点进行用户确认
