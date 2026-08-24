# MetricFlow 语义适配器

MetricFlow 语义适配器把 Datus Agent 连接到已有的 MetricFlow 原生 semantic model 和 metric YAML 文件。

> **仅查询兼容：**已有 MetricFlow 项目仍可查询，但 Datus 不再生成或编辑 MetricFlow 语义 YAML。
> 新的语义创作只允许在 Dosi 项目中通过 `semantic_modeling` 完成。

当团队仍通过外部工具或流程维护 MetricFlow 源文件时，适合使用这个适配器。
MetricFlow 不再默认安装或选择，需要显式配置。

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
        default: true                   # 多个 adapter 并存时显式选择
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

该目录下已有的 YAML 都会参与验证，即使这些文件是项目本地文件或被 gitignore 忽略。

## 旧格式源模型

MetricFlow 项目直接加载 MetricFlow YAML。

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

## 查询流程

当 MetricFlow 是 active semantic layer 时，已有资产继续支持：

1. `validate_semantic()` 校验完整 MetricFlow model。
2. `query_metrics(...)` 编译并执行已有指标。
3. `ask_metrics`、指标预览、API 指标查询、report 和 dashboard 复用同一查询链路。

已退役的 `gen_semantic_model` 和 `gen_metrics` 会返回明确错误，提示迁移到 Dosi 并使用 `semantic_modeling`。

## 支持的查询能力

该适配器支持通用 semantic adapter 方法：

- `list_metrics`
- `get_dimensions`
- `query_metrics`
- `validate_semantic`

MetricFlow 按自身模型处理 SQL 生成、join、时间粒度、metric constraint、cumulative metric、ratio metric、expression metric 和 derived metric。

底层 MetricFlow 引擎概念和支持的数据仓库见 [Datus-MetricFlow 介绍](../metricflow/introduction.zh.md)。

## 其他语义格式

如需查询已有 strict OSI core YAML，请使用 [OSI 语义适配器](osi_semantic_adapter.zh.md)。新的语义创作请使用 Dosi。
