# OSI 语义适配器

本文面向使用 Datus 生成和查询指标的用户，说明当前 Datus 对 OSI（Open Semantic Interchange）语义模型的支持方式。这里的 OSI 支持由两部分组成：

- `datus-agent`：负责用 LLM 生成严格 OSI core YAML，把生成结果校验、dry-run，并同步到 Knowledge Base，供 `ask_metrics` 查询。
- `datus-semantic-adapter`：提供 `datus-semantic-osi` 适配器，负责加载 OSI YAML、校验 OSI core schema、编译到 Datus Semantic IR，再降低到执行后端。目前执行后端是 MetricFlow。

## 当前定位

OSI 在 Datus 里是语义模型和指标的 authoring format，也就是用户和 LLM 书写的源格式。MetricFlow 是默认 execution backend，也就是实际生成 SQL 和执行查询的后端。

```text
gen_semantic_model / gen_metrics
        |
        v
strict OSI core YAML + DATUS custom_extensions
        |
        v
datus-semantic-osi: validate -> compile IR -> lower to MetricFlow
        |
        v
validate_semantic / query_metrics / ask_metrics
```

用户不需要、也不应该在 OSI 模式下手写 MetricFlow YAML 字段，例如 `data_source`、`measures`、`measure_proxy`、`type_params`。这些字段由 OSI adapter 在后端产物里生成，生成物是 disposable artifact，不是用户维护的源文件。

## 安装

使用 OSI 需要安装 OSI adapter。当前 OSI adapter 位于 `datus-semantic-adapter` 仓库的 `datus-semantic-osi` 包。

从发布包安装时：

```bash
pip install "datus-semantic-osi[metricflow]"
```

从源码安装时：

```bash
pip install -e ../datus-semantic-adapter/datus-semantic-core
pip install -e "../datus-semantic-adapter/datus-semantic-metricflow"
pip install -e "../datus-semantic-adapter/datus-semantic-osi[metricflow]"
```

`metricflow` extra 会安装 MetricFlow 执行后端需要的依赖。只做 OSI 编译或静态校验时可以不安装 extra，但要通过 Datus 查询指标时需要可用的执行后端。

## 配置

在 `agent.yml` 里把 semantic layer 配成 `osi`，并让相关节点使用 `semantic_adapter: osi`。

```yaml
agent:
  services:
    datasources:
      starrocks:
        type: starrocks
        host: 127.0.0.1
        port: "9030"
        username: admin
        password: ${STARROCKS_PASSWORD}
        database: ac_manage
        default: true

    semantic_layer:
      osi:
        execution_backend: metricflow
        default: true

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: osi
      # authoring_format 可省略；semantic_adapter=osi 时会自动使用 OSI authoring。
      # authoring_format: osi

    gen_metrics:
      semantic_adapter: osi

    ask_metrics:
      semantic_adapter: osi
```

`datus-agent` 的 authoring format 解析规则是：

1. 如果节点显式配置 `authoring_format: osi`，使用 OSI。
2. 如果节点使用的 `semantic_adapter` 解析为 `osi`，自动使用 OSI。
3. 否则保持默认 MetricFlow authoring。

## 生成语义模型

在 OSI 模式下，`gen_semantic_model` 生成的是严格 OSI core document。根结构固定为：

```yaml
version: 0.2.0.dev0
semantic_model:
  - name: shop
    datasets:
      - name: orders
        source: orders
        primary_key: [order_id]
        fields:
          - name: order_date
            expression:
              dialects:
                - dialect: ANSI_SQL
                  expression: order_date
            dimension:
              is_time: true
            custom_extensions:
              - vendor_name: DATUS
                data: '{"type":"time","time_granularity":"day"}'
          - name: channel
            expression:
              dialects:
                - dialect: ANSI_SQL
                  expression: channel
            dimension:
              is_time: false
            description: "Order channel"
            custom_extensions:
              - vendor_name: DATUS
                data: '{"type":"categorical"}'
```

关键规则：

- 一个物理表对应一个 canonical dataset。不要为不同 SQL 或不同指标重复声明同一张表。
- dataset 使用 OSI core 的 `fields`，不是 MetricFlow 的 `dimensions`。
- dataset `source` 是表名字符串，不是 `{table: ...}`。
- 主键写在 `primary_key`。
- 时间字段用 `dimension.is_time: true` 标记，Datus 的时间粒度写进 `custom_extensions`。
- 关系写在 semantic model 对象的 `relationships` 下，不要写进 dataset。

## 生成指标

在 OSI 模式下，`gen_metrics` 会把指标追加到 OSI core document 的 `semantic_model[0].metrics` 下。指标的业务表达写在 OSI core 的 `expression` 中，Datus 执行提示写在 `custom_extensions`。

```yaml
version: 0.2.0.dev0
semantic_model:
  - name: shop
    datasets:
      - name: orders
        source: orders
        primary_key: [order_id]
        fields: [...]
    metrics:
      - name: revenue
        description: "Total paid order revenue"
        expression:
          dialects:
            - dialect: ANSI_SQL
              expression: "SUM(amount)"
        custom_extensions:
          - vendor_name: DATUS
            data: '{"dataset":"orders","time_dimension":"order_date","unit":"CNY"}'
```

支持的常见指标类型：

| 类型 | OSI 写法 | 后端 lowering |
|------|----------|---------------|
| 基础聚合 | `SUM`、`COUNT`、`COUNT(DISTINCT)`、`AVG`、`MIN`、`MAX` | MetricFlow `measure_proxy` |
| 条件聚合 | `COUNT(DISTINCT CASE WHEN ... THEN id END)` | backing measure + metric |
| 表达式指标 | 聚合结果之间的加减乘除 | MetricFlow `expr` |
| 比率 | 聚合除法，或 DATUS hints 中的 `numerator` / `denominator` | MetricFlow `ratio` |
| 累计 / 滚动窗口 | 基础聚合 + DATUS `window` 或 `grain_to_date` | MetricFlow `cumulative` |
| 同比 / 环比 | derived metric + input metric `offset_window` | MetricFlow `derived` |
| 多表维度查询 | OSI relationship + joined dimension | MetricFlow join identifier |

## 同比、环比和 offset_window

OSI 指标里不要直接写 SQL 窗口函数：

```sql
LAG(revenue) OVER (...)
ROW_NUMBER() OVER (...)
```

环比和同比应拆成基础指标和 derived 指标。比如月收入和上月收入：

```yaml
metrics:
  - name: revenue
    description: "Monthly revenue"
    expression:
      dialects:
        - dialect: ANSI_SQL
          expression: "SUM(amount)"
    custom_extensions:
      - vendor_name: DATUS
        data: '{"dataset":"orders","time_dimension":"order_month"}'

  - name: revenue_previous_month
    description: "Revenue in previous month"
    expression:
      dialects:
        - dialect: ANSI_SQL
          expression: "revenue_previous_month"
    custom_extensions:
      - vendor_name: DATUS
        data: '{"metric_kind":"derived","inputs":[{"name":"revenue","alias":"revenue_previous_month","offset_window":"1 month"}]}'

  - name: revenue_mom_delta
    description: "Revenue month-over-month delta"
    expression:
      dialects:
        - dialect: ANSI_SQL
          expression: "revenue - revenue_previous_month"
    custom_extensions:
      - vendor_name: DATUS
        data: '{"metric_kind":"derived","inputs":[{"name":"revenue"},{"name":"revenue","alias":"revenue_previous_month","offset_window":"1 month"}]}'
```

原因是 OSI 指标表达的是可复用的业务指标，而不是一段查询 SQL。`offset_window` 把“取上一周期同一指标”表达为语义关系，执行后端再负责生成等价查询计划。

## 多表 join

多表 join 通过 OSI core relationship 表达：

```yaml
relationships:
  - name: orders_to_customers
    from: orders
    to: customers
    from_columns: [customer_id]
    to_columns: [customer_id]
```

当前 Datus execution profile 支持单列关系，类型主要是 `many_to_one` 和 `one_to_one`。降低到 MetricFlow 时，adapter 会在 `from` dataset 上生成 foreign identifier，使 fact 指标可以按 dimension dataset 的字段分组或过滤。

查询时 joined dimension 的名字通常会带上目标 dataset 的主 identifier 前缀，例如：

```text
customer_id__country
```

`ask_metrics` 会通过 `list_metrics` 和 `get_dimensions` 发现这些可查询维度。

## 校验和发布流程

OSI 模式下，Datus 在发布前会做硬性校验：

1. `validate_semantic(scope="semantic_model")` 校验语义模型。
2. `validate_semantic(scope="all")` 校验完整语义层。
3. `query_metrics(..., dry_run=True)` 对生成的指标做 SQL dry-run。
4. `end_semantic_model_generation` / `end_metric_generation` 把 OSI semantic objects 和 metrics 同步到 Knowledge Base。

如果 validation 或 dry-run 失败，Datus 不会发布该指标到 Knowledge Base。这样 `ask_metrics` 只会查询已经通过语义层验证的指标。

## ask_metrics 查询

`ask_metrics` 面向统一的 semantic adapter 接口工作。使用 OSI adapter 时，它仍然调用同一组工具：

- `list_metrics`
- `get_dimensions`
- `query_metrics`
- `validate_semantic`

OSI adapter 返回的 metric metadata 会包含 Datus hints，例如：

- `dataset`
- `time_dimension`
- `metric_kind`
- `expr`
- `inputs`
- `offset_window`
- `window`
- `grain_to_date`
- `format`
- `unit`

这些 metadata 用来帮助 `ask_metrics` 选择正确指标、维度和查询参数。

## 不适合作为指标生成的 SQL

以下 SQL 类型不会被 `gen_metrics` 强行生成为 OSI metric：

- 明细列表：`SELECT col1, col2 ...`
- 去重明细：`SELECT DISTINCT ...`
- 排名列表：`ROW_NUMBER()`、`RANK()`、`DENSE_RANK()` 输出明细行
- TopN per group，例如“每个渠道排名前 N 的活动”
- 主要产出是行级记录，而不是聚合指标的查询

这些查询可以作为后续的 derived dataset、物化视图或 query layer 能力来表达，但不属于当前 `gen_metrics` 的指标生成范围。

## 当前限制

- OSI adapter 当前默认执行后端是 MetricFlow。
- relationship 当前执行 profile 支持单列 join。多列复合 join 需要后续扩展。
- SQL 窗口函数不能直接写进 OSI metric expression；周期对比使用 `offset_window`，排名/TopN 明细需要 query layer 或预计算数据集。
- 当 semantic model 包含多个 dataset 时，metric 必须通过 DATUS custom extension 明确 `dataset`。
- OSI core 之外的 Datus 执行信息必须写入 `custom_extensions[{vendor_name: DATUS}]`，不要写成 OSI core 顶层字段。
