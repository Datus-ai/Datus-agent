# `gen_metrics`（已退役）

`gen_metrics` 仅为兼容旧配置而保留。它不会出现在 subagent 列表中，直接调用会返回错误并推荐使用 [`semantic_modeling`](semantic_modeling.md)。

## 替代方案

在 Dosi 项目中使用 `semantic_modeling`，统一创作 metric 及其依赖的 dataset 和 relationship。该流程会校验完整的 Dosi 模型、检查生成的指标，并将 YAML 源文件与 Knowledge Base 完整对账。

```text
根据这些 SQL 证据定义收入和订单数指标。@Agent semantic_modeling
```

MetricFlow 和 OSI 项目继续支持已有指标的执行和查询，但不能生成新指标。要修改已有 OSI YAML，请先把项目的 semantic type 改为 Dosi，再使用 `semantic_modeling`。不支持迁移 MetricFlow YAML。

## Bootstrap 兼容行为

在 Dosi 项目中，`bootstrap-kb --components metrics` 会运行完整的 `semantic_modeling` 流程。受支持的旧格式仍可使用 YAML 导入。将 `metrics` 与 `semantic_model` 或 `semantic_modeling` 组合时，仍只执行一次完整创作流程。

支持的流程见 [Semantic Modeling](semantic_modeling.md) 和[指标](../knowledge_base/metrics.md)。
