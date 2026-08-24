# 语义层适配器

Datus Agent 通过语义层适配器，把语义创作、校验、发现和查询连接到具体的语义层实现。Dosi 是默认适配器，也是唯一支持创作的适配器；MetricFlow 与 OSI 仅在显式配置时用于查询兼容。

本文是适配器总览。具体适配器请看：

- [MetricFlow 语义适配器](metricflow_semantic_adapter.zh.md)
- [OSI 语义适配器](osi_semantic_adapter.zh.md)
- [Dosi 语义适配器](dosi_semantic_adapter.zh.md)

## 概述

语义层适配器提供统一接口，用于：

- 列出可执行指标
- 获取指标可用维度
- 查询指标值
- 发布前校验语义资产
- 将已校验的语义资产同步到 Datus Knowledge Base

当前支持三个适配器：

| 适配器 | 包名 | 源格式 | 执行后端 | Datus 模式 |
|--------|------|--------|----------|------------|
| Dosi | `datus-semantic-dosi` | strict OSI core YAML + DATUS custom extensions | 原生 Dosi engine | 默认；创作 + 查询 |
| MetricFlow | `datus-semantic-metricflow` | MetricFlow YAML | MetricFlow | 显式配置；仅查询 |
| OSI | `datus-semantic-osi[metricflow]` | strict OSI core YAML + DATUS custom extensions | MetricFlow | 显式配置；仅查询 |

三个 adapter 共用查询接口，但不再共用创作入口：

- Dosi 创作 strict OSI-compatible YAML，并由 Rust engine 直接编译、规划和执行。
- MetricFlow 加载已有 MetricFlow YAML，用于校验、发现和查询。
- OSI 加载已有 OSI core YAML，编译到 Datus Semantic IR，再降低到 MetricFlow 查询。

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
    ├── datus-semantic-osi
    │   └── DatusOSIAdapter
    └── datus-semantic-dosi
        └── DosiAdapter
```

适配器通过 `datus.semantic_adapters` Python entry point 自动发现。

## 配置

在 `agent.yml` 的 `agent.services.semantic_layer` 下配置语义层适配器。

```yaml
agent:
  services:
    semantic_layer:
      dosi:
        default: true

      metricflow: {}  # 可选的查询兼容

      osi: {}         # 可选的查询兼容；安装 osi[metricflow]
```

`services.semantic_layer` 下的 key 必须等于 adapter type，例如 `metricflow`、`osi` 或 `dosi`。如果同时写了 `type:` 字段，其值必须与 key 一致。
语义适配器选择是全局的。旧的 node 级 `semantic_adapter` 和 `authoring_format` 字段会被忽略。

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
- 团队在 Datus 之外维护这些资产
- 迁移期间仍需通过 Datus 查询它们

适合使用 OSI 的情况：

- 希望源文件遵循 OSI core schema
- 希望 Datus 执行提示隔离在 `custom_extensions` 中
- 已有 OSI 资产需要继续通过 MetricFlow 查询

适合使用 Dosi 的情况：

- 要开始新的语义创作或修改已有模型
- 希望继续使用 strict OSI authoring 流程
- 模型以 aggregate、ratio 和 expression 指标为主
- 希望使用原生 Join 规划、fan-out 防护和执行，不依赖 MetricFlow

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
