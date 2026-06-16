# MetricFlow 语义适配器

MetricFlow 语义适配器把 Datus Agent 连接到 MetricFlow 原生的 semantic model 和 metric YAML 文件。

当你希望生成的语义资产就是 MetricFlow YAML，并且团队会直接维护 MetricFlow 源文件时，适合使用这个适配器。

## 安装

```bash
pip install datus-semantic-metricflow
```

从源码安装：

```bash
pip install -e ../datus-semantic-adapter/datus-semantic-core
pip install -e ../datus-semantic-adapter/datus-semantic-metricflow
```

## 配置

```yaml
agent:
  services:
    semantic_layer:
      metricflow:
        timeout: 300
        config_path: ./conf/agent.yml   # 可选高级覆盖项
        default: true

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow

    gen_metrics:
      semantic_adapter: metricflow

    ask_metrics:
      semantic_adapter: metricflow
```

`config_path` 是可选项。正常情况下，Datus 会从以下信息构造 MetricFlow 运行时配置：

1. `services.datasources` 中选中的数据源
2. 当前项目的 semantic model 目录
3. 当前生效的 `agent.home`

## 语义模型目录

默认情况下，Datus 会把 MetricFlow 指向当前项目的语义模型目录：

```text
{project_root}/subject/semantic_models/
```

该目录下生成的 YAML 都会参与验证，即使这些文件是项目本地文件或被 gitignore 忽略。

## Authoring 模型

MetricFlow 模式直接编写 MetricFlow YAML。

语义模型文件使用 `data_source` 文档：

```yaml
data_source:
  name: orders
  sql_table: public.orders
  identifiers:
    - name: order_id
      type: primary
      expr: order_id
  dimensions:
    - name: order_date
      type: time
      type_params:
        is_primary: true
        time_granularity: day
  measures:
    - name: revenue_sum
      agg: sum
      expr: amount
```

指标文件使用 `metric` 文档：

```yaml
metric:
  name: revenue
  type: measure_proxy
  type_params:
    measures:
      - revenue_sum
```

## 生成流程

配置 `semantic_adapter: metricflow` 后：

1. `gen_semantic_model` 写入 MetricFlow semantic model YAML。
2. `gen_metrics` 写入 MetricFlow metric YAML。
3. `validate_semantic()` 校验完整 MetricFlow model。
4. `query_metrics(..., dry_run=True)` 确认生成指标可以编译成 SQL。
5. `end_semantic_model_generation` 和 `end_metric_generation` 将通过校验的资产同步到 Knowledge Base。

## 支持的查询能力

该适配器支持通用 semantic adapter 方法：

- `list_metrics`
- `get_dimensions`
- `query_metrics`
- `validate_semantic`

MetricFlow 按自身模型处理 SQL 生成、join、时间粒度、metric constraint、cumulative metric、ratio metric、expression metric 和 derived metric。

底层 MetricFlow 引擎概念和支持的数据仓库见 [Datus-MetricFlow 介绍](../metricflow/introduction.zh.md)。

## 什么时候改用 OSI

如果你希望源文件是 strict OSI core YAML，并且希望 Datus / MetricFlow 执行提示隔离在 `custom_extensions` 中，请使用 [OSI 语义适配器](osi_semantic_adapter.zh.md)。
