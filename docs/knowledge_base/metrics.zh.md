# 指标

指标是可复用、可执行的业务计算，例如收入、活跃客户数或倒闭银行资产。它为一套计算逻辑提供稳定的名称和定义，让用户无需反复编写聚合 SQL 就能提出业务问题。

指标属于[语义模型](semantic_model.md)。定义保存在模型的 OSI YAML 中；Datus 将其投影到独立的 Knowledge Base store 供检索，再由内置 Dosi 引擎编译并执行指标查询。

## 指标定义什么

一个指标通常包含：

- datasource 内唯一的 `name`；
- 业务 `description` 和 `ai_context.instructions`；
- 聚合、比率或算术 `expression`；
- 所属 dataset 与业务时间；
- 用于检索的三级 `subject_path`；
- `unit`、`format` 等可选展示信息；
- 滚动、累计或同期比较等可选的可复用 window 行为。

指标所需的 dataset、field 和 relationship 都定义在同一份语义模型文件中。因此，计算逻辑和执行它所需的模型只有一个权威 artifact。

## 定义、检索与执行

指标数据有三个不同角色：

1. **OSI YAML 是定义。** 表达式、时间行为、描述和 extension 都以它为准。
2. **`metrics` Knowledge Base store 是检索投影。** 它按当前 datasource 索引指标名、描述、subject path、所属模型、可用维度、源 YAML 等检索信息。
3. **Dosi 负责执行。** `query_metrics` 把指标名、维度、时间范围、过滤和排序交给 Dosi，由引擎编译并运行查询。

把指标移动到其他 `subject_path` 只会改变分类，不会改变指标身份。因此，同一个 datasource 中的指标名必须保持唯一。

## 使用自然语言快速查询

推荐的用户流程是直接提业务问题。主 agent 会识别指标问题，并自动派发给 `ask_metrics`：

```bash
datus --datasource duckdb_demo
```

```text
看看倒闭银行数量和倒闭资产总额按年份的变化趋势。
```

用户不需要知道指标名。AskMetrics 会根据主题树和指标描述推断该问题需要 `bank_failure_count` 与 `failed_assets_million`，发现可用时间维度，再按年查询。

该流程要求当前 datasource 已经存在可执行指标。如果没有现有指标能回答问题，AskMetrics 会直接说明限制，不会切换成无关的原始 SQL 回答。路由方式、配置和输出行为见 [AskMetrics](../subagent/ask_metrics.md)。

## 当前指标 YAML

下面是 Datus 示例 DuckDB 数据库上真实生成的 `bank_failures.yml` 中 `semantic_model[0].metrics` 的片段，使用了[语义模型](semantic_model.md)页面中的 dataset 和 field。代码块保留了它在完整文件中的缩进；它属于 semantic model 内部，不是一份独立的 YAML 文档。

```yaml
    metrics:
      - name: bank_failure_count
        expression:
          dialects:
            - dialect: DUCKDB
              expression: COUNT(*)
        description: 倒闭银行数量，即银行倒闭事件的记录数。
        ai_context:
          instructions: 按 date 作为业务时间，统计倒闭事件条数；可按 state 或时间粒度分组。
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"1.4","dataset":"bank_failures","time_dimension":"bank_failures.date","subject_path":["banking","bank_failures","count"],"unit":"banks"}'

      - name: failed_assets_million
        expression:
          dialects:
            - dialect: DUCKDB
              expression: SUM(bank_failures.assets_million)
        description: 倒闭银行资产总额（单位：百万美元）。
        ai_context:
          instructions: 按 date 作为业务时间，对倒闭时资产求和；单位是百万美元。
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"1.4","time_dimension":"bank_failures.date","subject_path":["banking","bank_failures","assets"],"unit":"USD million"}'
```

Metric expression 与 field expression 一样使用 `expression.dialects[]`。字段应带 dataset 限定，例如 `SUM(bank_failures.assets_million)`。`COUNT(*)` 这类聚合无法根据限定字段判断所属 dataset，因此在 DATUS extension 中显式提供 `dataset`。

不要把 `dataset`、`time_dimension`、`subject_path`、`unit` 或 window 设置写在 metric 顶层。这些属于 Datus 行为，应放入 metric 的 DATUS `custom_extensions` entry。

## Metric extension

DATUS extension 是一个 OSI `custom_extensions` entry，其中 `data` 的值是 JSON 字符串：

```yaml
custom_extensions:
  - vendor_name: DATUS
    data: '{"v":"1.4","time_dimension":"orders.order_date","subject_path":["sales","revenue","total"],"unit":"USD"}'
```

常用 key 包括：

| Key | 作用 |
| --- | --- |
| `dataset` | 当表达式本身无法确定 dataset 时指定归属，例如 `COUNT(*)`。 |
| `time_dimension` | 选择指标的业务时间字段；字段名可能冲突时应带 dataset 限定。 |
| `subject_path` | 把指标放入三级检索分类。 |
| `unit`、`format` | 描述结果的单位和展示格式。 |
| `fill_nulls_with` | 定义明确的空值填充行为。 |
| `window` | 定义可复用的滚动、累计、同期比较、排名或取值窗口。 |

`semantic_modeling` 会写入已安装 engine 所需的 extension 版本。应保留生成的版本值，并保证每个对象最多只有一个 DATUS entry。

## 查询指标

AskMetrics 是面向用户的常规查询入口。集成或高级 agent 工作流可以使用底层工具，将检索与执行分开：

```python
# 从 Knowledge Base 中发现候选定义。
search_metrics(
    query_text="按年统计倒闭银行数量和资产",
    top_n=5,
)

# 每个指标怎么算、读哪些数据集、有哪些时间粒度。
get_metric(name="bank_failure_count")

# 指标所读数据集的字段及其定义。
get_dataset(name="bank_failures")

# 按年执行两个指标。
query_metrics(
    metrics=["bank_failure_count", "failed_assets_million"],
    dimensions=["metric_time"],
    time_granularity="year",
    order_by=["metric_time__year"],
)
```

### 检索工具返回什么

几个检索工具都返回文本，一行一项，对象没有的项就不写。

- `list_metrics` 按主题路径分组列出指标，每个指标一行 `- 名字 -- 描述`。这一页不是全部指标时，最后一行 `more:` 给出下次调用要传的 `offset`。
- `get_metric` 描述一个指标：
    - `expr`：指标怎么由其他指标或字段算出来。可能是一个聚合（`COUNT(DISTINCT orders.customer_id)`）、一个过滤指标（`base FILTER (WHERE 条件)`）、组合指标的算式、细节层次声明，或者窗口指标的 base。
    - `datasets`：指标读的数据集。`dimension_datasets`：可以用来分组和过滤的字段所在的数据集；只有部分字段可用时写成 `名字(f1, f2)`。
    - `window` 和 `compares_across`：窗口指标的规则，以及排名或占比在哪些维度之间比较。
    - `params`：每个参数写成 `名字 类型 = 默认值 [取值范围] -- 含义`；参数由成员指标或所读的 fixed 值声明时，后面带 `[declared_on 指标名]`。
    - `required`：每次查询都必须分组的维度。`requires_time_axis: true` 表示查询还必须按 `metric_time` 加粒度分组。
    - `time`：`metric_time` 和它的粒度，从细到粗；指标所在表没有时间字段时写 `none`。
- `get_dataset` 描述一个数据集：第一行是数据集描述、`source`、`time` 字段和主键，之后每个字段一行。普通列写 `- 名字`；计算字段写 `- 表达式 AS 名字`；fixed 字段写 `- fixed(base, keys) AS 名字`，表示按键在自己的一批行上算出的值。方括号里标出时间字段的格式、不是维度的字段，以及 fixed 字段读的是哪个数据集。最后的 `relationships:` 列出每个关联，写成 `关联类型 数据集 on 列=列`。

`query_metrics` 当前支持以下参数：

| 参数 | 含义 |
| --- | --- |
| `metrics` | 一个或多个准确的指标名。 |
| `dimensions` | 写成 `dataset.field` 的字段，取自 `get_metric` 在 `dimension_datasets` 里列出的数据集（字段见 `get_dataset`）。`metric_time` 表示指标的时间轴，粒度用 `time_granularity` 指定。 |
| `time_start`、`time_end` | 可选的左闭右开时间范围：start 包含，end 不包含。使用 ISO 日期（`YYYY-MM-DD`）。每个指标、每个 fixed 字段都按自己所在表的时间字段取这个时段。 |
| `time_granularity` | `day`、`week`、`month`、`quarter` 或 `year`。 |
| `where` | 可选过滤表达式，不包含 `WHERE` 关键字。引用已选指标的条件（如 `revenue > 1000`）在聚合之后过滤结果行，相当于 `HAVING`；含窗口指标的查询里，只引用分组字段的条件同样过滤结果行，保留的行仍带着全体人群算出的名次和分区计数。其余条件在聚合前过滤。 |
| `context_filter` | 可选过滤表达式，是所有数都要遵守的范围，在查询中每个读得到这些列的部分里于聚合前生效；读不到的部分会列在结果的 `warnings` 里。它限定窗口指标排名、计数或加权的人群：`where="market = 'North'"` 看的是 North 在所有市场里的名次，`context_filter="market = 'North'"` 则只在 North 内部重新排名。它也会限定固定粒度（fixed）值计算所用的行，`where` 不会。 |
| `order_by` | 排序所用的结果列；名称前加 `-` 表示降序。 |
| `limit` | 最大行数；只在用户明确要求 Top N、预览或其他行数限制时使用。 |
| `params` | 按名字绑定 `get_metric` 在 `params` 里列出的参数，包括指标自己的、成员指标的，以及所读 fixed 值的。传列表时一次查询绑定多个值，每个值各返回一列：`{"k": [1, 7]}` 得到 `<metric>__k_1` 和 `<metric>__k_7`。未传的参数取默认值。 |
| `dry_run` | 返回编译后的 SQL，不执行指标查询；配置了数据仓库 dry-run 时还会校验该 SQL。结果和实际执行一样带有查询的 `warnings`。 |

例如，要包含 2024 年 1 月的全部数据，应使用 `time_start="2024-01-01"` 和 `time_end="2024-02-01"`。

实际执行的查询返回文本：先是 `result_id` 一行和写着行数的 `rows` 一行；有需要说明的事时每件一行，比如引擎的 `warning` 或 `info`、`IN (...)` 里没有匹配到任何行的值（`note`）、按参数绑定命名的列对应哪个指标（`column`）；然后空一行，接着是 CSV 格式的结果行。结果较长时只显示开头和结尾几行，中间用 `... N rows omitted` 一行代替；完整结果已经缓存，可以用 `get_query_metrics_result` 分页读取，最终答案用的也是它。文本里不带编译后的 SQL，需要 SQL 时用 `dry_run`。

### 条件放在哪里

问题里的每个条件只放在一个位置：

- `time_start`/`time_end`：时段。每个指标、每个 fixed 字段（按实体取的值，比如某个客户的订单数）都按自己所在表的时间字段取它。所在表没有时间字段的指标或 fixed 字段会读全表，结果的 `warnings` 会说明；`get_metric` 显示 `time: none` 的指标读的就是这种表，查询时不传 `time_start`/`time_end`。
- `context_filter`：所有数都要遵守的范围，比如只看某个平台、某个活动。查询的每个部分读得到这些列就应用它，读不到的部分列在 `warnings` 里。
- `where`：要哪些结果行。字段上的条件不会改变 fixed 字段的值；引用已选指标的条件在聚合之后过滤。快照表取哪一天的快照写在这里。

不该跟着时段走的 fixed 值（比如首次购买日、累计值），把问题里的日期写在 `where` 里、用指标所在表自己的日期字段，不写 `time_start`/`time_end`：日期只挑选指标计数的行，所有 fixed 值都按全部历史计算。

```python
query_metrics(
    metrics=["active_customers"],
    dimensions=["visits.customer_first_order_day"],
    where="visits.visit_date >= '2024-01-01' AND visits.visit_date < '2024-02-01'",
)
```

只有当一个值依赖它所在分组以外的行时，放在不同位置才会算出不同的数：固定粒度（fixed）值、`include`/`exclude` 粒度，以及窗口。`get_metric` 和 `get_dataset` 都能看出来：`get_dataset` 把 fixed 字段写成 `fixed(order_count, customers.customer_id) AS 名字`（每个客户一个值；`fixed(revenue)` 表示全表一个值）；细节层次指标的 `expr` 就是它的声明；窗口指标有一行 `window`。某个部分读不到的条件，`query_metrics` 会在 `warnings` 里说明，所以 dry run 时就能看出条件放错了位置。

| 部分 | 各位置的作用 |
| --- | --- |
| fixed 值 | 时段按它自己所在表的时间字段取；`context_filter` 在读得到列时缩小它，`where` 不会。针对 fixed 字段本身的条件放 `where`。base 带窗口参数的 fixed 字段，按 `params` 绑定的窗口取值。 |
| `include`、`exclude` | 每个位置都会缩小它。 |
| 同比环比、累计窗口 | 日期放 `time_start`/`time_end`；放 `where` 会截掉回看，比较值为空。 |
| 移动窗口 | `time_start`/`time_end`、`context_filter` 和 `where` 结果相同。 |
| 排名 | `where` 条件字段参与分组时，值在全体中计算；`context_filter` 在子集内计算。 |

例如「1 月各地区占全公司的比例」：全公司总额应该随 1 月变化、不随地区变化，所以 1 月放 `time_start`/`time_end`，地区放 `where`。

## 创建和更新指标

日常创作使用 `semantic_modeling`。它可以在同一份 YAML 中补充所需 dataset 和 field、创建或更新 metric、校验完整模型、按需 dry-run 代表性查询，并同步两个 Knowledge Base 投影。

```text
在 sales 模型中增加月度收入增长指标，使用 order_date 作为业务时间，校验定义并 dry-run 一次按年查询。
```

如果要从包含 `question` 和 `sql` 的 CSV 批量创作，使用统一的 semantic-modeling component：

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components semantic_modeling \
  --success_story path/to/success_story.csv \
  --kb_update_strategy incremental \
  --metrics-batch-size 5
```

`--metrics-batch-size` 控制每个创作批次处理的历史记录数。`--subject_tree` 可以传入逗号分隔的允许分类；不传时，创作流程会复用已有分类或创建合适的分类。

默认的 `check` 策略不会创作指标，只报告当前 semantic-dataset 与 metric 投影数量。需要生成时应明确使用 `incremental` 或 `overwrite`。

## 同步 Knowledge Base

`semantic_modeling` 成功完成后会自动同步目标文件。如果手工编辑了 YAML，可以用下面的命令协调全部 semantic-object 与 metric 投影：

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components semantic_model \
  --kb_update_strategy sync-yaml
```

`sync-yaml` 默认读取当前 datasource 下的全部模型，替换各 artifact 的投影行，删除该 artifact 中已不再声明的指标，并在同步整个目录时清理已删除文件留下的行；不会调用 LLM，也不会查询数仓。

如果只想查看已索引指标数量，不做任何修改：

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components metrics \
  --kb_update_strategy check
```

## 相关文档

- [AskMetrics](../subagent/ask_metrics.md)：自然语言指标问答和能力边界
- [语义建模](../subagent/semantic_modeling.md)：完整创作流程与真实生成 YAML
- [语义模型](semantic_model.md)：dataset、field、relationship 与 Knowledge Base 投影
- [Dosi 引擎](../semantic/dosi_engine.zh.md)：模型位置与查询行为
