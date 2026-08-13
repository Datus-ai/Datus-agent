# `gen_semantic_model`（已退役）

`gen_semantic_model` 仅为兼容旧配置而保留。它不会出现在 subagent 列表中，直接调用会返回错误并推荐使用 [`semantic_modeling`](semantic_modeling.md)。

## 替代方案

在 Dosi 项目中使用 `semantic_modeling` 创建或修改 dataset、field、relationship、模型元数据和 metric。该流程会修改 Dosi YAML、校验选中的模型，并将 YAML 源文件与 Knowledge Base 完整对账。

```text
/semantic_modeling 为 orders 和 customers 建模并定义二者的关系。
```

MetricFlow 和 OSI 项目继续支持查询，但不支持语义写入。要修改已有 OSI 项目，请先把 semantic type 改为 Dosi，再使用 `semantic_modeling` 修复并校验已有 YAML。不支持迁移 MetricFlow YAML。

## Bootstrap 兼容行为

在 Dosi 项目中，`bootstrap-kb --components semantic_model` 会以 datasets-only scope 运行 `semantic_modeling`。它可以修改 dataset、field、relationship 和模型元数据，但会保留所有已有 metric 定义。受支持的旧格式仍可使用 YAML 导入和 profile 解析。

支持的流程见 [Semantic Modeling](semantic_modeling.md) 和[语义模型](../knowledge_base/semantic_model.md)。
