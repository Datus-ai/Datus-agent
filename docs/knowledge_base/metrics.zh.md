# 指标

指标是可复用、可执行的业务计算，例如收入、活跃客户数或倒闭银行资产。它为一套计算逻辑提供稳定的名称和定义，让用户无需反复编写聚合 SQL 就能提出业务问题。

指标属于[语义模型](semantic_model.md)。定义保存在模型的 OSI YAML 中；Datus 将其投影到独立的 Knowledge Base store 供检索，再由当前 semantic adapter 编译并执行指标查询。Dosi 是内置的默认 adapter。

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
3. **Semantic adapter 负责执行。** `query_metrics` 把指标名、维度、时间范围、过滤和排序交给当前 adapter，由 adapter 编译并运行查询。

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

下面的定义来自 Datus 示例 DuckDB 数据库上真实生成的 `bank_failures.yml`，使用了[语义模型](semantic_model.md)页面中的 dataset 和 field：

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

# 获取可查询维度和指标的主要时间轴。
get_dimensions(metric_name="bank_failure_count")

# 按年执行两个指标。
query_metrics(
    metrics=["bank_failure_count", "failed_assets_million"],
    dimensions=["metric_time"],
    time_granularity="year",
    order_by=["metric_time__year"],
)
```

`query_metrics` 当前支持以下参数：

| 参数 | 含义 |
| --- | --- |
| `metrics` | 一个或多个准确的指标名。 |
| `dimensions` | `get_dimensions` 返回的维度；在 Dosi 中，`metric_time` 表示指标的主要时间轴。 |
| `path` | 可选 subject-tree 路径，用于消除指标歧义。 |
| `time_start`、`time_end` | 可选的左闭右开时间范围：start 包含，end 不包含。Adapter 支持 ISO 日期以及 `-7d`、`now` 等相对值。 |
| `time_granularity` | `day`、`week`、`month`、`quarter` 或 `year`。 |
| `where` | 可选过滤表达式，不包含 `WHERE` 关键字。 |
| `order_by` | 排序所用的结果列；名称前加 `-` 表示降序。 |
| `limit` | 最大行数；只在用户明确要求 Top N、预览或其他行数限制时使用。 |
| `dry_run` | 编译并校验查询计划，不返回实时指标值。 |

例如，要包含 2024 年 1 月的全部数据，应使用 `time_start="2024-01-01"` 和 `time_end="2024-02-01"`。

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
- [语义层配置](../configuration/semantic_layer.md)：semantic adapter 配置
