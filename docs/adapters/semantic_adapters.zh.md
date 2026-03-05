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
pip install -e ../datus-semantic-adapter/datus_semantic_metricflow
```

安装后，Datus Agent 会自动检测并加载适配器。

## 配置

在 `config.yaml` 文件的 `semantic` 部分配置语义层：

### MetricFlow

```yaml
semantic:
  type: metricflow
  namespace: my_project
  timeout: 300  # 可选，默认 300 秒
  config_path: /path/to/agent.yml  # 可选，未指定时使用默认查找路径
```

**语义模型文件位置**：
MetricFlow 自动在以下位置查找语义模型文件：
```
{agent.home}/semantic_models/{namespace}/
```
- `agent.home` 从 `agent.yml` 读取（默认为 `~/.datus`）

### 配置查找优先级

初始化 MetricFlow 适配器时：

1. `config_path` 参数（如果显式提供）
2. `./conf/agent.yml`（当前目录）
3. `~/.datus/conf/agent.yml`（主目录）

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
from datus.tools.semantic_tools import semantic_adapter_registry
from datus_semantic_metricflow.config import MetricFlowConfig

config = MetricFlowConfig(namespace="my_project")
adapter = semantic_adapter_registry.create_adapter("metricflow", config)

metrics = await adapter.list_metrics(limit=10)
dimensions = await adapter.get_dimensions(metric_name="revenue")
result = await adapter.query_metrics(
    metrics=["revenue"], dimensions=["date"], time_start="2024-01-01"
)
```

### Dry Run（SQL 预览）

```python
result = await adapter.query_metrics(metrics=["revenue"], dry_run=True)
print(result.data[0]["sql"])
```

### 从适配器同步数据

```bash
datus-agent bootstrap-kb --namespace my_project --components metrics \
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

```
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
| 连接问题 | 验证 `agent.yml` 配置，检查 namespace 与语义模型目录匹配 |
| 验证错误 | 运行 `adapter.validate_semantic()` 检查配置 |

## 下一步

- [MetricFlow 配置](../metricflow/introduction.md) - MetricFlow 详细配置
- [配置参考](../configuration/introduction.md) - 通用配置选项
