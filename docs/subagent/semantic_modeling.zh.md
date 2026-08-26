# 语义建模

`semantic_modeling` 把数据库的物理结构和业务定义整理成可复用的语义层。建模以后，表之间如何关联、收入如何计算、时间字段如何使用等规则只需定义一次，Datus 就能据此生成一致的 SQL、回答指标问题，并把模型索引到 Knowledge Base 中。

该 subagent 是 Dosi 的语义创作工作流。它可以根据数据库元数据创建一个业务域模型、更新已有模型、把可复用的 SQL 逻辑沉淀为 dataset 和 metric，并在结果对其他 agent 可见之前完成校验。

## 语义建模在定义什么

一个语义模型会把以下四类信息连接起来：

- **Dataset**：把订单、客户等业务实体映射到物理表或可复用查询。
- **Field**：为列定义稳定的名称、表达式、说明和时间维度信息。
- **Relationship**：定义允许的关联方式。`from` 是多的一侧，`to` 是一的一侧。
- **Metric**：为收入、订单数等业务度量定义可复用的聚合表达式和业务语境。

YAML 文件是唯一事实来源。Knowledge Base 中的记录是从 YAML 派生出的可搜索索引，不是另一份需要单独维护的模型。

## 可用范围和支持的数据库

只有当前语义适配器为 `dosi` 时才能进行语义创作。直接支持的数据库范围取 `osi-engine` 已实现 SQL 方言与 `datus-db-adapters` 可安装数据库产品的交集：

| 数据库适配器 | Dosi SQL 方言 |
| --- | --- |
| StarRocks | `starrocks` |
| Snowflake | `snowflake` |
| PostgreSQL | `postgres` |
| Greenplum | `postgres` |
| MySQL | `mysql` |
| Doris | `doris` |
| ClickHouse | `clickhouse` |
| Trino | `trino` |
| Redshift | `redshift` |
| Hologres | `hologres` |
| GaussDB | `gaussdb` |
| Oracle | `oracle` |

这里共有 12 个数据库产品、11 种 engine 方言，因为 Greenplum 复用 PostgreSQL renderer。DuckDB 和 SQLite 还可以通过 Datus 内置 datasource 路径使用，但它们不属于外部 `datus-db-adapters` 的交集。只有 engine 方言或只有数据库 adapter，都不代表已经具备端到端的语义建模支持。

## 快速上手：使用内置 DuckDB 示例

Datus 安装包中包含 `duckdb-demo.duckdb`。首次启动 `datus` 且全局配置尚不存在时，Datus 会将它复制到 Datus home 的 `sample/` 目录。Datus home 默认为当前操作系统用户的 `~/.datus`，因此数据库通常位于 `~/.datus/sample/duckdb-demo.duckdb`；如果自定义了 Datus home，请使用对应目录下的 `sample/duckdb-demo.duckdb`。正常情况下无需另行下载数据，但仍需将该文件注册为 datasource；如果文件尚不存在，先启动一次 `datus` 完成初始化。

```yaml
agent:
  services:
    datasources:
      duckdb_demo:
        type: duckdb
        uri: ~/.datus/sample/duckdb-demo.duckdb
        default: true
```

`dosi` 是内置的默认语义适配器，因此没有配置其他语义适配器时不需要增加 `semantic_layer` 配置。启动 Datus：

```bash
datus --datasource duckdb_demo
```

然后直接用自然语言描述需要的业务模型。主 agent 会识别语义创作需求，并自动委派给 `semantic_modeling`：

```text
为 bank_failures 建模。添加银行、州、倒闭日期和资产字段，将倒闭日期设为主要时间维度，并定义倒闭银行数量 bank_failure_count 和倒闭银行资产总额 failed_assets_million 两个指标。校验模型并实际查询这两个指标。
```

在一次使用 DeepSeek 的手动 REPL 运行中，`semantic_modeling` 检查了包含 545 行、7 个物理字段的源表，选择请求中的 4 个业务字段建模，并生成 `subject/semantic_models/duckdb_demo/bank_failures.yml`。模型没有声明主键或唯一键，完整校验通过且没有 issue。

`semantic_modeling` 成功返回 `generated` 后，目标 YAML 已完成校验，指标也已同步到 Knowledge Base；保持在 `duckdb_demo` datasource 即可立即查询，无需另行发布、导入或重启 Datus。在主 chat 中直接提问：

```text
按年份查询倒闭银行数量和倒闭银行资产总额。
```

主 agent 会把这种指标查询自动委派给 `ask_metrics`。如果之前通过 `/agent semantic_modeling` 将语义建模设为当前 agent，先使用 `/agent chat` 返回主 chat；也可以在问题末尾添加 `@Agent ask_metrics` 单次指定。完整用法见 [AskMetrics](ask_metrics.md)。

实际执行结果包含 14 个年份；以下为部分结果：

| 年份 | `bank_failure_count` | `failed_assets_million` |
| --- | ---: | ---: |
| 2008 | 26 | 768,576.8 |
| 2009 | 140 | 169,507.4 |
| 2010 | 157 | 95,975.0 |
| 2023 | 6 | 572,650.0 |
| 2024 | 2 | 6,107.8 |

不带维度查询时，两个指标分别为 `545` 和约 `1,695,997.0`。全部连接字段和 datasource 选择规则见[数据源配置](../configuration/datasources.md)与[语义层配置](../configuration/semantic_layer.md)。

## 使用 subagent

在 Datus REPL 中，通常直接描述期望得到的业务模型即可。主 agent 会把创建或更新语义 dataset 和 metric 的请求委派给 `semantic_modeling`。表名、目标模型、指标和原始 SQL 都可以按需要写进请求。

```text
更新 sales.yml，基于 orders.order_date 添加收入月环比指标，并用 dry-run 查询验证。
```

```text
把下面 SQL 沉淀为可复用的收入指标，并保留其中的过滤条件：SELECT ...
```

如果要把单次请求明确交给该 subagent，可以增加 agent 引用：

```text
更新 sales.yml 并校验新增指标。@Agent semantic_modeling
```

如果要连续和该 subagent 对话，先将它设为当前 agent，再输入普通消息；使用 `/agent chat` 返回主 agent：

```text
/agent semantic_modeling
```

旧的 `/<subagent> <message>` 形式已经不再支持，因此 `/semantic_modeling ...` 会被识别为未知命令。通过 API 调用时，使用 `subagent_id: semantic_modeling` 选择它。

一次运行会执行以下流程：

1. 查看已有模型，通过明确的文件名、唯一事实表或业务域选择一个目标。
2. 在写入前读取真实 datasource 的 schema 和 relationship 证据。
3. 按依赖顺序创建或更新 dataset、field、relationship 和 metric。
4. 校验完整目标文件，并在用户要求时 dry-run 代表性的指标查询。
5. 将通过校验的 YAML 完整同步到 Knowledge Base。

每次只编辑一个语义模型。如果现有模型都不合适，则按业务域创建新模型，而不是为每张物理表各建一个文件。只有查询结果本身是稳定、可复用的业务实体，或用户明确要求复现某条查询时，才会把 SQL 查询作为 dataset 的 `source`。

最终结果会报告 `generated`、`skipped` 或 `blocked`。如果结果为 blocked，会说明缺少 schema 证据、目标不明确、定义无效或其他无法安全写入的原因。

## 生成文件的位置和组织方式

新文件写入当前项目和 datasource 对应的目录：

```text
subject/semantic_models/<datasource>/<semantic_model>.yml
```

每个创作文件的根 `semantic_model` 列表只放一个模型对象。建议按业务域组织相关的事实表和维表，例如 `sales.yml` 或 `marketing.yml`。`source` 应优先引用物理表；只有稳定且值得复用的逻辑 dataset 才使用 SQL。

## YAML 规范

Dosi 生成 OSI Core `0.2.0.dev0` YAML。core schema 不允许未知字段，因此 Datus 特有行为必须写在 `custom_extensions` 中，不能随意增加 YAML key。

主要字段如下：

| 对象 | 必需或重要字段 | 含义 |
| --- | --- | --- |
| 根对象 | `version`、`semantic_model` | OSI 版本和模型列表 |
| Semantic model | `name`、`datasets`；可选 `description`、`ai_context`、`relationships`、`metrics`、`custom_extensions` | 一个业务域 |
| Dataset | `name`、`source`；可选 `primary_key`、`unique_keys`、`fields`、说明和 extension | 物理表或可复用查询 |
| Field | `name`、`expression`；可选 `dimension`、`label`、说明和 extension | 可分组或过滤的属性 |
| Relationship | `name`、`from`、`to`、`from_columns`、`to_columns` | 等值关联；两组列按位置配对 |
| Metric | `name`、`expression`；可选说明、`ai_context` 和 extension | 可复用的聚合度量 |
| Expression | 包含 `dialect` 和 `expression` 的 `dialects` | 一种或多种 SQL 方言实现 |

表达式应使用当前 datasource 对应的方言标签，例如 `DUCKDB`、`POSTGRESQL`、`SNOWFLAKE`、`STARROCKS`、`DORIS`、`MYSQL`、`CLICKHOUSE`、`TRINO`、`REDSHIFT`、`HOLOGRES`、`GAUSSDB` 或 `ORACLE`。只有表达式确实可以跨数据库使用时才应写 `ANSI_SQL`。

### 完整示例

下面是这次手动 REPL 生成并通过 Dosi 原生校验的模型；仅调整了 YAML 缩进和长行换行，模型内容与真实产物一致：

```yaml
version: 0.2.0.dev0
semantic_model:
  - name: bank_failures
    datasets:
      - name: bank_failures
        source: main.bank_failures
        description: 银行倒闭事件事实表，每一行记录一家倒闭银行及其倒闭日期与倒闭时资产。
        ai_context: 每一行代表一家银行的倒闭事件；用于按时间（date）、州（state）等维度分析银行倒闭数量与倒闭资产规模。
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

这个结果来自真实 schema 和数据检查，而不是预先编写的模板。它只为提示词要求的 4 个字段建模，也没有声明 key。DuckDB 可以不加引号解析 `Bank`、`State` 和 `Date` 这些简单标识符；`Assets ($mil.)` 含空格和符号，因此使用双引号表达式。

## DATUS custom extension 简介

`custom_extensions` 是 OSI 规范为厂商行为预留的扩展机制。DATUS entry 可以在保持文档符合 OSI 的同时，增加 Dosi 使用的计算行为。不理解 DATUS 的 consumer 可以忽略该 entry，仍然读取 core model。

```yaml
custom_extensions:
  - vendor_name: DATUS
    data: '{"v":"1.4","join_type":"left"}'
```

使用时需要注意：

- `data` 是 JSON **字符串**，不是嵌套 YAML 对象。
- 一个 OSI 对象最多放一个 DATUS entry。
- 生成的 payload 会携带当前 engine 的 extension 版本。自动化流程应由 `semantic_modeling` 写入版本，不要自行固定。
- 每个 key 只能写在允许的对象上：dataset 或 metric 使用 `time_dimension`，时间 field 使用 `time_granularity`，relationship 使用 `join_type`，metric 计算行为写在 metric 上。

参与实际计算的 extension key 包括：

| 承载对象 | Key | 用途 |
| --- | --- | --- |
| Dataset | `time_dimension` | 选择 dataset 的主要时间字段 |
| Field | `time_granularity` | 声明 `day`、`week`、`month`、`quarter` 或 `year` 粒度 |
| Relationship | `join_type` | 选择 `left` 或 `inner` 关联 |
| Metric | `dataset`、`time_dimension`、`fill_nulls_with` | 解析指标归属、时间字段和空值行为 |
| Metric | `window` | 定义同比环比、滚动、累计、frame、rank 或 value window |
| Metric | `derive` | 基于同一模型内的基础指标定义过滤指标或组合指标 |

Datus 还会写入 `subject_path`、`unit`、`format`、`metric_kind`、`source_type`、`uid`、`owner` 等展示和来源元数据。这些 key 用于组织和展示模型，不改变 Dosi 原生指标计算。

即使 metric 使用了 `window` 或 `derive`，core `expression` 也必须保留合法的聚合表达式。Dosi 会检查 fallback expression 与 extension 语义是否一致。更多说明见 [Dosi 语义适配器](../adapters/dosi_semantic_adapter.md)、[语义模型](../knowledge_base/semantic_model.md)和[指标](../knowledge_base/metrics.md)。

## 校验和兼容性

手工修改后应始终校验整个文件。仅通过 YAML 结构校验还不够：dataset 和 field 引用、relationship key、方言表达式、extension 承载对象、派生指标和 window 都必须保持语义一致。`semantic_modeling` 会执行最终校验，并可在需要执行证据时 dry-run 指标查询。

现有 MetricFlow 和 OSI 项目仍可查询，但只提供查询兼容。要创作现有 OSI 项目，请先将 semantic type 改为 Dosi，再使用 `semantic_modeling` 原地修复和校验 YAML。不支持迁移 MetricFlow YAML。

已退役的 `gen_semantic_model` 和 `gen_metrics` 名称只为配置兼容而保留。它们不会出现在发现列表中，直接调用时会推荐 `semantic_modeling`。旧自定义 agent 的 `node_class` 或 `type` 使用其中任一名称时，在 Dosi 项目中会自动路由到 `semantic_modeling`。

对于 `bootstrap-kb`，历史 component 名称作为兼容别名保留：

- `--components semantic_model` 以 datasets-only 范围运行，可以更新模型元数据、dataset、field 和 relationship，但保护已有 metric。
- `--components metrics` 和 `--components semantic_modeling` 运行完整工作流。
- 同时指定这些 component 时只运行一次；只要包含 `metrics` 或 `semantic_modeling` 就以完整范围为准。

现有 YAML import 和 `refresh-profile` 仍是非 LLM 操作，不会重新启用任何已退役的 subagent。
