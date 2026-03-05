# 基准测试（Benchmark）

配置基准数据集以评估 Datus Agent 的 SQL 生成效果，用于准确性度量、配置对比与迭代验证。

## 支持的数据集
- **BIRD-DEV**：复杂 SQL 场景的综合评测
- **Spider2**：多数据库高级评测
- **Semantic Layer**：业务指标与语义理解评测

## 配置结构
```yaml
benchmark:
  bird_dev:
    benchmark_path: benchmark/bird/dev_20240627

  spider2:
    benchmark_path: benchmark/spider2/spider2-snow

  semantic_layer:
    benchmark_path: benchmark/semantic_layer
```

## BIRD-DEV
```yaml
benchmark:
  bird_dev:
    benchmark_path: benchmark/bird/dev_20240627
```

## Spider2
```yaml
benchmark:
  spider2:
    benchmark_path: benchmark/spider2/spider2-snow
```

## Semantic Layer
```yaml
benchmark:
  semantic_layer:
    benchmark_path: benchmark/semantic_layer
```

更多用法与进阶配置，参见 [Benchmarks](../benchmark/benchmark_manual.md)。
