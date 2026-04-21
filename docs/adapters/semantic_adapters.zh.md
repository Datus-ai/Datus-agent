# 语义层适配器

Datus Agent 通过插件化的适配器系统支持连接各种语义层服务。本文档介绍可用的适配器、安装方法以及语义层连接配置。

## 概述

Datus 采用模块化的语义层适配器架构，支持连接不同的语义层后端：

- **MetricFlow**: dbt 的语义层，用于指标和维度管理

这种设计为不同的语义层实现提供了统一的指标发现、查询和验证接口。

## 架构

```text
datus-agent (核心)
├── 语义层工具层
│   ├── BaseSemanticAdapter (抽象基类)
│   ├── SemanticAdapterRegistry (工厂)
│   └── 数据模型 (MetricDefinition, QueryResult 等)
│
└── 插件系统 (Entry Points)
    └── datus-semantic-metricflow
        └── MetricFlowAdapter
```

适配器系统使用 Python 的 entry points 机制进行自动发现。安装适配器包后，它会自动注册到 Datus Agent 并可供使用。

## 支持的语义层

| 语义层 | 包名 | 安装方式 | 状态 |
|--------|------|---------|------|
| MetricFlow | datus-semantic-metricflow | `pip install datus-semantic-metricflow` | 可用 |

## 安装

### MetricFlow 适配器

```bash
# 安装 MetricFlow 适配器
pip install datus-semantic-metricflow

# 或从源码安装
pip install -e ../datus-semantic-adapter/datus-semantic-metricflow
```

安装后，Datus Agent 会自动检测并加载适配器。

## 配置

在 `agent.yml` 的 `agent.services.semantic_layer` 中配置语义层适配器：

如果你使用的是 MetricFlow 的默认配置，整个 `semantic_layer` 段也可以省略；此时 Datus 会自动默认使用 `metricflow`。

### MetricFlow

```yaml
agent:
  services:
    semantic_layer:
      metricflow:
        timeout: 300  # 可选，默认 300 秒
        config_path: /path/to/agent.yml  # 可选的高级覆盖项

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow
    gen_metrics:
      semantic_adapter: metricflow
```

**语义模型文件位置**：
默认情况下，Datus 会把 MetricFlow 指向当前项目的语义模型目录：
```text
{project_root}/subject/semantic_models/
```
- `project_root` 是当前 Datus 项目的根目录。

### 选择规则

- `services.semantic_layer` 下的 key **必须等于 adapter type**（例如 `metricflow`）。如果同时写了 `type:` 字段，其值必须与 key 一致，否则 Datus 会在启动时抛出配置错误。比较时会先对 key 与 `type` 做 lowercase + trim 处理，因此 `MetricFlow` 或 ` metricflow ` 也会被视为与 `metricflow` 匹配。
- 语义相关节点通过 `semantic_adapter` 选择适配器。
- 如果 `services.semantic_layer` 和 `semantic_adapter` 都省略，Datus 会默认使用 `metricflow`。
- 如果只配置了一个 semantic layer，省略 `semantic_adapter` 时会自动使用它。
- 如果配置了多个 semantic layer，则必须显式填写 `semantic_adapter`。

### 关于 `config_path`

`config_path` 是可选项。正常运行时，Datus 会从以下上下文构造 MetricFlow 配置：

1. `services.databases` 中当前选中的数据库
2. 当前项目的语义模型目录
3. 当前生效的 `agent.home`

只有在你明确希望 MetricFlow 从另一份 agent 配置文件初始化时，才需要填写 `config_path`。

## 核心接口

### 指标接口

所有语义层适配器实现以下核心异步方法：

| 方法 | 描述 | 返回类型 |
|------|------|---------|
| `list_metrics(path, limit, offset)` | 列出可用指标，支持过滤 | `List[MetricDefinition]` |
| `get_dimensions(metric_name, path)` | 获取指标的维度 | `List[DimensionInfo]` |
| `query_metrics(metrics, dimensions, ...)` | 查询指标，支持过滤、时间范围、where 子句 | `QueryResult` |
| `validate_semantic()` | 验证语义层配置 | `ValidationResult` |

### 语义模型接口（可选）

| 方法 | 描述 | 返回类型 |
|------|------|---------|
| `get_semantic_model(table_name, ...)` | 获取表的语义模型 | `Optional[Dict]` |
| `list_semantic_models(...)` | 列出可用的语义模型 | `List[str]` |

## 数据模型

| 模型 | 主要字段 |
|------|---------|
| `MetricDefinition` | `name`, `description`, `type`, `dimensions`, `measures`, `unit`, `format`, `path` |
| `QueryResult` | `columns`, `data`, `metadata` |
| `ValidationResult` | `valid`, `issues` |
| `ValidationIssue` | `severity`, `message`, `location` |
| `DimensionInfo` | `name`, `description` |

## 使用示例

### 直接使用适配器

```python
import asyncio
from datus.tools.semantic_tools import semantic_adapter_registry
from datus_semantic_metricflow.config import MetricFlowConfig

async def main():
    config = MetricFlowConfig(namespace="my_project")
    adapter = semantic_adapter_registry.create_adapter("metricflow", config)

    metrics = await adapter.list_metrics(limit=10)
    dimensions = await adapter.get_dimensions(metric_name="revenue")
    result = await adapter.query_metrics(
        metrics=["revenue"], dimensions=["date"], time_start="2024-01-01"
    )

asyncio.run(main())
```

### Dry Run（SQL 预览）

```python
async def dry_run_example():
    result = await adapter.query_metrics(metrics=["revenue"], dry_run=True)
    print(result.data[0]["sql"])

asyncio.run(dry_run_example())
```

### 从适配器同步数据

```bash
datus-agent bootstrap-kb --database my_project --components metrics \
  --from_adapter metricflow --kb-update-strategy overwrite
```

## 适配器功能

### 通用功能

所有语义层适配器支持：

- 指标发现和列表
- 按指标获取维度
- 带过滤条件的指标查询
- 配置验证
- 存储同步缓存

### MetricFlow 适配器

- 完整的 MetricFlow API 集成
- 基于 YAML 的语义模型文件
- 三阶段验证（lint、parse、semantic）
- SQL 生成和执行计划
- 支持时间粒度的时间范围过滤

## 实现自定义适配器

你可以通过继承 `BaseSemanticAdapter` 并通过 Python entry points 注册来实现自己的语义层适配器。

### 必须实现的方法

你的适配器必须实现以下抽象方法：

| 方法 | 描述 | 返回类型 |
|------|------|---------|
| `list_metrics()` | 列出可用指标，支持过滤 | `List[MetricDefinition]` |
| `get_dimensions()` | 获取指标的可查询维度 | `List[DimensionInfo]` |
| `query_metrics()` | 执行带过滤条件的指标查询 | `QueryResult` |
| `validate_semantic()` | 验证语义层配置 | `ValidationResult` |

### 可选方法

| 方法 | 描述 | 默认行为 |
|------|------|---------|
| `get_semantic_model()` | 获取表的语义模型 | 返回 `None` |
| `list_semantic_models()` | 列出可用的语义模型 | 返回 `[]` |

### 包结构

```text
datus_semantic_myservice/
├── pyproject.toml
└── datus_semantic_myservice/
    ├── __init__.py    # register() 函数
    ├── adapter.py     # MyServiceAdapter
    └── config.py      # MyServiceConfig
```

### Entry Point 配置

```toml
# pyproject.toml
[project.entry-points."datus.semantic_adapters"]
myservice = "datus_semantic_myservice:register"
```

### 参考实现

完整示例请参考 MetricFlow 适配器实现：
- [datus-semantic-metricflow](https://github.com/Datus-ai/datus-semantic-adapter)

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| 适配器未找到 | 安装适配器：`pip install datus-semantic-metricflow` |
| 连接问题 | 验证 `agent.yml` 配置，检查当前数据库选择与语义模型目录 |
| 验证错误 | 运行 `adapter.validate_semantic()` 检查配置 |

## 下一步

- [MetricFlow 配置](../metricflow/introduction.md) - MetricFlow 详细配置
- [配置参考](../configuration/introduction.md) - 通用配置选项
