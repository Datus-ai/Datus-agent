# 语义模型

语义模型把数据库的物理结构整理成可复用的业务概念。它告诉 Datus：哪些表或查询代表一个业务 dataset、字段的业务含义是什么、分析应该使用哪个时间字段，以及不同 dataset 可以如何关联。指标也定义在同一个模型中，详见[指标](metrics.md)。

Datus 通过 Dosi 编写严格的 OSI YAML。YAML 文件是唯一事实来源；Knowledge Base 保存的是由 YAML 派生的可搜索投影，供 agent 和产品界面使用。

## 语义模型包含什么

| 对象 | 作用 |
| --- | --- |
| **Semantic model** | 把相关的 dataset、relationship 和 metric 组织为一个业务域。 |
| **Dataset** | 将业务实体或事件集合映射到物理表或可复用 SQL 查询。 |
| **Field** | 为源列或表达式提供稳定名称和业务描述，也可以标记时间维度。 |
| **Relationship** | 定义两个 dataset 之间允许使用的等值关联。 |
| **Metric** | 定义可复用的业务计算。Metric 与 dataset 位于同一份 YAML，但使用独立的 Knowledge Base 投影和查询流程。 |

例如，银行业务模型可以把 `main.bank_failures` 映射为 `bank_failures` dataset，描述银行、州、倒闭日期和资产字段，再定义倒闭银行数量与倒闭资产总额等指标。

## 源文件与 Knowledge Base 投影

模型文件按 datasource 存放：

```text
subject/semantic_models/<datasource>/<semantic_model>.yml
```

每个文件只包含一个 semantic model。`semantic_modeling` 成功完成后，会校验完整文件、写入磁盘，并把内容同步到 Knowledge Base。

`semantic_dataset` 投影为每个 dataset、field 和 relationship 各保存一行：

- Dataset 由 `(semantic_model, dataset)` 标识，记录物理来源或可复用查询。
- Field 在此基础上增加 dataset 和 field 名、表达式、时间/键标记及描述。
- Relationship 由 `(semantic_model, relationship)` 标识，记录两端 dataset 与按位置配对的列。
- Query-backed dataset 只有 `source_query`，没有 `source_table`，因此不会与同名物理表混淆。

Metric 不保存在 `semantic_dataset` 中，而是投影到 `metrics` Knowledge Base store。两个投影都会保留源 YAML 路径，并按当前 datasource 隔离。

## 快速上手

用需要建模的 datasource 启动 Datus，然后直接用自然语言描述业务概念。主 agent 会自动把创作请求派发给 `semantic_modeling`：

```bash
datus --datasource duckdb_demo
```

```text
为 bank_failures 建模。添加银行、州、倒闭日期和资产字段，将倒闭日期设为主要时间维度，并定义倒闭银行数量和倒闭资产总额两个指标。校验模型并实际查询代表性的指标。
```

生成的文件位于：

```text
subject/semantic_models/duckdb_demo/bank_failures.yml
```

这个示例所用的 DuckDB datasource 配置见[语义建模指南](../subagent/semantic_modeling.md)。该指南还介绍了支持的数据库、datasource 准备和交互式 agent 选择方式。

## 当前 YAML 结构

下面的 dataset 来自 Datus `duckdb-demo.duckdb` 示例数据库上的真实生成结果，展示了 Dosi 当前使用的 OSI 结构：

```yaml
version: 0.2.0.dev0
semantic_model:
  - name: bank_failures
    datasets:
      - name: bank_failures
        source: main.bank_failures
        description: 银行倒闭事件事实表，每一行记录一家倒闭银行。
        ai_context: 用于按日期和州分析银行倒闭事件。
        fields:
          - name: bank
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: Bank
            description: 倒闭银行名称
          - name: state
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: State
            description: 银行所在州（美国州代码）
          - name: date
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: Date
            dimension:
              is_time: true
            description: 银行倒闭日期
            custom_extensions:
              - vendor_name: DATUS
                data: '{"v":"1.4","time_granularity":"day"}'
          - name: assets_million
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: '"Assets ($mil.)"'
            label: Assets ($mil.)
            description: 倒闭时资产总额（百万美元）
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"1.4","time_dimension":"date"}'
    relationships: []
    metrics: []
```

需要遵守以下规则：

- 根节点只有 `version` 和 `semantic_model`；`semantic_model` 是列表。
- 物理 dataset 的 `source` 使用带限定的表名。只有查询结果本身是稳定业务 dataset 时，才使用可复用查询作为 source。
- 表达式使用 `expression.dialects[]`，dialect 应与当前 datasource 匹配，例如 `DUCKDB`、`POSTGRESQL`、`SNOWFLAKE` 或 `STARROCKS`。
- `primary_key` 和 `unique_keys` 只记录经过确认的键；Datus 不会根据列名或单条查询臆测主键。
- 时间 field 使用 `dimension.is_time: true`，源数据粒度由 DATUS `time_granularity` extension 表达。
- Relationship 属于 semantic model，而不是某个 dataset。`from_columns` 与 `to_columns` 按位置配对，目标列必须是完整且经过确认的键。
- Metric 定义放在同一 semantic model 的 `metrics` 列表中，当前结构见[指标](metrics.md)。

## Relationship

Relationship 声明 Dosi 在跨 dataset 查询时可以使用的关联路径：

```yaml
relationships:
  - name: order_customer
    from: orders
    to: customers
    from_columns: [customer_id]
    to_columns: [customer_id]
    custom_extensions:
      - vendor_name: DATUS
        data: '{"v":"1.4","join_type":"left"}'
```

`from` 通常是多的一侧，`to` 是一的一侧。使用复合键时，两侧需要按相同顺序列出全部组成列。

## DATUS extension

`custom_extensions` 是 OSI 为厂商行为预留的扩展机制。DATUS entry 在保持文档符合 OSI 的同时，增加 Dosi 使用的执行或展示提示。

```yaml
custom_extensions:
  - vendor_name: DATUS
    data: '{"v":"1.4","time_dimension":"date"}'
```

`data` 是 JSON 字符串，不是嵌套 YAML 对象。`semantic_modeling` 会写入已安装 engine 所需的 extension 版本，因此应保留生成的版本值。

常见的语义模型 extension 包括：

| 对象 | Key | 作用 |
| --- | --- | --- |
| Dataset | `time_dimension`、`source_type` | 选择主要业务时间，或标记 query-backed source。 |
| 时间 field | `time_granularity` | 记录源数据粒度：day、week、month、quarter 或 year。 |
| Relationship | `join_type` | 选择 `left` 或 `inner` 关联。 |

Metric 专用 extension 见[指标](metrics.md)。

## 创建和更新模型

日常交互中，直接用自然语言让 Datus 创建或更新模型即可。`semantic_modeling` 会检查真实 schema、编辑一个目标文件、完成校验，并在成功后同步 semantic object 和 metric。

如果要从“问题 + SQL”历史批量创作，准备包含 `question` 和 `sql` 列的 CSV：

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components semantic_modeling \
  --success_story path/to/success_story.csv \
  --kb_update_strategy incremental \
  --metrics-batch-size 5
```

如果批处理只需要 dataset 和 relationship，不生成 metric，可把 component 改为 `semantic_model`。`overwrite` 与 `incremental` 都会协调所选语义 YAML artifact；它们都不是只读检查。默认的 `check` 只报告当前投影数量。

## 同步已有 YAML

如果 YAML 不是通过 `semantic_modeling` 修改的，可用下面的命令重建 Knowledge Base 投影：

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components semantic_model \
  --kb_update_strategy sync-yaml
```

该命令读取当前 datasource 下的全部语义 YAML，检查其结构，并协调 `semantic_dataset` 与 `metrics` 两个投影；不会调用 LLM，也不会查询数仓。只有明确想同步某个文件或目录时，才传 `--semantic_yaml path/to/model.yml`。

如果要刷新某个已有模型中由 profile 生成的描述，可同时提供该 YAML 和历史 SQL CSV。`refresh-profile` 会执行有界的只读 profile、更新 YAML 描述并同步结果：

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components semantic_model \
  --kb_update_strategy refresh-profile \
  --semantic_yaml path/to/model.yml \
  --success_story path/to/success_story.csv
```

## 检索和使用

Agent 通过下面的工具搜索语义对象：

```python
search_semantic_objects(
    query_text="银行倒闭日期和州",
    kinds=["dataset", "field", "relationship"],
    top_n=5,
)
```

该搜索不返回 metric；指标应使用 `search_metrics`。查看物理表时，Datus 也可以把匹配的 dataset 投影合并到表元数据中；Catalog 中的语义模型视图是只读的。需要修改时应更新 YAML 或使用 `semantic_modeling`，再通过同步刷新所有使用方。

## 相关文档

- [语义建模](../subagent/semantic_modeling.md)：完整创作流程、配置、支持的数据库和真实生成 YAML
- [指标](metrics.md)：指标定义、Knowledge Base 投影和查询参数
- [AskMetrics](../subagent/ask_metrics.md)：自然语言指标问答
- [语义层配置](../configuration/semantic_layer.md)：semantic adapter 配置
