# 语义层配置（Semantic Layer）

语义适配器统一配置在 `agent.services.semantic_layer` 下。

如果你使用的是 MetricFlow 的默认配置，整个 `semantic_layer` 段可以省略；Datus 会默认使用 `metricflow`。

## 配置结构

```yaml
agent:
  services:
    semantic_layer:
      metricflow:
        timeout: 300
        config_path: ./conf/agent.yml   # 可选的高级覆盖项

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow

    gen_metrics:
      semantic_adapter: metricflow
```

## 选择规则

- `services.semantic_layer` 下的 key **必须等于 adapter type**（例如 `metricflow`）。如果同时写了 `type:` 字段，其值必须与 key 一致，否则 Datus 会在启动时抛出配置错误。比较时会先对 key 与 `type` 做 lowercase + trim 处理，因此 `MetricFlow` 或 ` metricflow ` 也会被视为与 `metricflow` 匹配。
- 语义相关节点通过 `semantic_adapter` 选择适配器。
- 语义适配器不支持 `default: true`。
- 如果 `services.semantic_layer` 和 `semantic_adapter` 都省略，Datus 会默认使用 `metricflow`。
- 如果只配置了一个 semantic layer，省略 `semantic_adapter` 时会自动使用它。
- 如果配置了多个 semantic layer，则必须显式填写 `semantic_adapter`。

## MetricFlow 说明

- `config_path` 是可选项。
- Datus 默认会基于当前 `services.databases` 中选中的数据库和项目语义模型目录自动构建运行时配置。
- 仅当你需要 MetricFlow 直接读取某个指定的 `agent.yml` 时才需要设置 `config_path`。
