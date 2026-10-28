# 基准测试（Benchmark）

配置基准数据集以评估 Datus Agent 的 SQL 生成效果，用于准确性度量、配置对比与迭代验证。

## 支持的数据集
- **BIRD-DEV**：复杂 SQL 场景的综合评测
- **Spider2**：多数据库高级评测
- **Semantic Layer**：业务指标与语义理解评测

## 内置基准测试

`bird_dev`、`spider2` 和 `semantic_layer` 的路径固定在 `{agent.home}/benchmark` 下，不需要也不能在 `agent.yml` 中覆盖其 `benchmark_path`。运行 Spider2 前，请[单独下载数据](../benchmark/benchmark_manual.zh.md#spider2-data)。

自定义基准测试可在 `agent.benchmark` 中配置。更多用法参见[基准测试手册](../benchmark/benchmark_manual.zh.md)。
