# 语义层适配器

Datus Agent 通过语义层适配器，把指标生成、校验、发现和查询连接到具体的语义层实现。

本文是适配器总览。具体适配器请看：

- [MetricFlow 语义适配器](metricflow_semantic_adapter.zh.md)
- [OSI 语义适配器](osi_semantic_adapter.zh.md)

## 概述

语义层适配器提供统一接口，用于：

- 列出可执行指标
- 获取指标可用维度
- 查询指标值
- 发布前校验语义资产
- 将已校验的语义资产同步到 Datus Knowledge Base

当前支持两个适配器：

| 适配器 | 包名 | Authoring format | 执行后端 | 状态 |
|--------|------|------------------|----------|------|
| MetricFlow | `datus-semantic-metricflow` | MetricFlow YAML | MetricFlow | 可用 |
| OSI | `datus-semantic-osi` | strict OSI core YAML + DATUS custom extensions | 默认 MetricFlow | 可用 |

MetricFlow 和 OSI 是并列的 semantic adapter。区别在于用户和生成 agent 维护的源格式：

- MetricFlow 模式直接编写 MetricFlow YAML。
- OSI 模式编写 OSI core YAML，由 `datus-semantic-osi` 编译到 Datus Semantic IR，再降低到 MetricFlow。

## 架构

```text
datus-agent
├── Semantic tools
│   ├── list_metrics
│   ├── get_dimensions
│   ├── query_metrics
│   └── validate_semantic
│
├── SemanticAdapterRegistry
│
└── Adapter packages
    ├── datus-semantic-metricflow
    │   └── MetricFlowAdapter
    └── datus-semantic-osi
        └── DatusOSIAdapter
```

适配器通过 `datus.semantic_adapters` Python entry point 自动发现。

## 配置

在 `agent.yml` 的 `agent.services.semantic_layer` 下配置语义层适配器。

```yaml
agent:
  services:
    semantic_layer:
      metricflow:
        default: true

      osi:
        execution_backend: metricflow

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow

    gen_metrics:
      semantic_adapter: metricflow

    ask_metrics:
      semantic_adapter: metricflow
```

`services.semantic_layer` 下的 key 必须等于 adapter type，例如 `metricflow` 或 `osi`。如果同时写了 `type:` 字段，其值必须与 key 一致。

选择规则、默认适配器和项目级 pin 见 [语义层配置](../configuration/semantic_layer.zh.md)。

## 核心接口

所有语义层适配器都实现以下方法：

| 方法 | 作用 |
|------|------|
| `list_metrics(path, limit, offset)` | 列出可执行指标。 |
| `get_dimensions(metric_name, path)` | 返回指标可用维度。 |
| `query_metrics(metrics, dimensions, ...)` | 查询指标，或通过 `dry_run=True` 渲染 SQL。 |
| `validate_semantic(scope)` | 校验语义资产和后端兼容性。 |

可选语义模型接口包括 `get_semantic_model()` 和 `list_semantic_models()`。

## 如何选择适配器

适合使用 MetricFlow 的情况：

- 已经有 MetricFlow YAML
- 希望生成文件就是 MetricFlow 原生格式
- 团队会直接查看或维护 MetricFlow 资产

适合使用 OSI 的情况：

- 希望源文件遵循 OSI core schema
- 希望 Datus 执行提示隔离在 `custom_extensions` 中
- 希望 LLM 生成时避免 `measure_proxy`、`type_params`、`data_source` 等后端 YAML 字段
- 当前仍希望通过 MetricFlow 执行

## 实现自定义适配器

可以通过继承 `BaseSemanticAdapter` 并注册 entry point 来实现自定义语义层适配器：

```toml
[project.entry-points."datus.semantic_adapters"]
myservice = "datus_semantic_myservice:register"
```

必须实现的方法：

| 方法 | 返回类型 |
|------|----------|
| `list_metrics()` | `List[MetricDefinition]` |
| `get_dimensions()` | `List[DimensionInfo]` |
| `query_metrics()` | `QueryResult` |
| `validate_semantic()` | `ValidationResult` |
