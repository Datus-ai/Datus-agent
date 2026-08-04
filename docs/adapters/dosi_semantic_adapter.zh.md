# Dosi 语义适配器

Dosi adapter 使用原生 Rust engine 直接执行 OSI 语义模型。它复用现有 OSI
adapter 的 strict OSI authoring 流程，但不会再降低到 MetricFlow。

## 安装

```bash
pip install datus-semantic-dosi
```

该命令会同时安装所需的 `dosi-engine` wheel。在 `/services` 的 Semantic tab
中选择 `dosi` 时，CLI 也会自动执行相同安装。

## 配置

```yaml
agent:
  services:
    datasources:
      warehouse:
        type: duckdb
        uri: /absolute/path/to/warehouse.db
    semantic_layer:
      dosi:
        default: true
        # semantic_model_path: /absolute/path/to/model.yaml
```

Datus 会注入当前数据源连接，并默认把 `semantic_models_path` 指向
`subject/semantic_models/<datasource>/`。目录中只有一个 YAML 或 JSON 文件时
会自动选择；存在多个模型文件时，需要配置 `semantic_model_path`。

## 当前支持范围

Dosi 当前可执行 aggregate、ratio、expression 指标，维度、many-to-one
关系、复合 Join、query-backed dataset、过滤，以及 day 到 year 的时间粒度。
规划器会拒绝歧义 Join 路径和可能产生 fan-out 的查询，避免静默重复计算。

derived、累计或滚动窗口、metric offset 和 period-over-period 指标，在 Dosi
补齐等价能力前仍应使用 OSI + MetricFlow 路径。

## Authoring 与校验

选择 `dosi` 后仍会启用 Datus 的 OSI 语义模型和指标 authoring skills。
`validate_semantic`、`list_metrics`、`get_dimensions` 和 `query_metrics` 会通过
Dosi adapter 执行，源 OSI 文档仍是唯一事实来源。
