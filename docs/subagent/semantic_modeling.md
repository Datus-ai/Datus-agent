# Semantic Modeling

`semantic_modeling` turns physical database structures and business definitions into a reusable semantic layer. Instead of embedding table joins and metric formulas in every question, you define datasets, fields, relationships, and metrics once. Datus can then use the model to generate consistent SQL, answer metric questions, and index the definitions in the Knowledge Base.

The subagent is an authoring workflow for Dosi. It can create a business-domain model from database metadata, update an existing model, translate reusable SQL logic into datasets and metrics, and validate the result before it becomes available to other agents.

## What it models

A semantic model connects four kinds of information:

- **Datasets** map business entities such as orders and customers to physical tables or reusable queries.
- **Fields** give columns stable names, expressions, descriptions, and time-dimension metadata.
- **Relationships** describe valid joins. The `from` dataset is the many side and the `to` dataset is the one side.
- **Metrics** give business measures such as revenue and order count a reusable aggregate expression and business context.

The YAML file is the source of truth. Knowledge Base records are a searchable index derived from that file, not a second model that should be edited independently.

## Availability and supported databases

Semantic authoring is available when the active semantic adapter is `dosi`. The directly supported warehouse scope is the intersection of SQL dialects implemented by `osi-engine` and installable products in `datus-db-adapters`:

| Database adapter | Dosi SQL dialect |
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

This is 12 adapter products backed by 11 engine dialects because Greenplum uses the PostgreSQL renderer. DuckDB and SQLite are also supported as built-in Datus datasource paths; they are not part of the external `datus-db-adapters` intersection. An engine dialect alone, or a database adapter alone, does not imply end-to-end semantic-modeling support.

## Quickstart with the built-in DuckDB sample

The Datus package includes `duckdb-demo.duckdb`. On the first `datus` startup, if no global configuration exists yet, Datus copies it into the `sample/` directory under Datus home. Datus home defaults to `~/.datus` for the current operating-system user, so the database is normally at `~/.datus/sample/duckdb-demo.duckdb`; if Datus home is customized, use `sample/duckdb-demo.duckdb` under that directory. No separate download is normally needed, but the file must still be registered as a datasource. If it does not exist yet, start `datus` once to complete initialization.

```yaml
agent:
  services:
    datasources:
      duckdb_demo:
        type: duckdb
        uri: ~/.datus/sample/duckdb-demo.duckdb
        default: true
```

`dosi` is the built-in semantic-layer default, so no `semantic_layer` entry is needed when another semantic adapter is not configured. Start Datus:

```bash
datus --datasource duckdb_demo
```

Then ask for the business model in plain language. The main agent recognizes semantic-authoring requests and delegates them to `semantic_modeling` automatically:

```text
Model bank_failures. Add bank, state, failure date, and assets fields; use failure date as the primary time dimension; and define bank_failure_count and failed_assets_million. Validate the model and execute both metrics.
```

In a manual REPL run with DeepSeek, `semantic_modeling` inspected a source table with 545 rows and seven physical columns, modeled the four requested business fields, and generated `subject/semantic_models/duckdb_demo/bank_failures.yml`. It did not assert a primary or unique key. Full-model validation passed with no issues.

After `semantic_modeling` successfully returns `generated`, the target YAML is validated and its metrics are synced to the Knowledge Base. Keep using the `duckdb_demo` datasource and query them immediately; no separate publish, import, or Datus restart is needed. Ask in the main chat:

```text
Show bank failure count and failed assets by year.
```

The main agent delegates this metric-first question to `ask_metrics` automatically. If `/agent semantic_modeling` was used to make the authoring agent current, run `/agent chat` first; alternatively, append `@Agent ask_metrics` to route one question explicitly. See [AskMetrics](ask_metrics.md#quick-start-query-newly-generated-metrics) for the complete flow.

The executed query returned 14 yearly buckets. Selected results are:

| Year | `bank_failure_count` | `failed_assets_million` |
| --- | ---: | ---: |
| 2008 | 26 | 768,576.8 |
| 2009 | 140 | 169,507.4 |
| 2010 | 157 | 95,975.0 |
| 2023 | 6 | 572,650.0 |
| 2024 | 2 | 6,107.8 |

Without dimensions, the two metrics return `545` and approximately `1,695,997.0`. See [Datasource Configuration](../configuration/datasources.md) and [Semantic Layer Configuration](../configuration/semantic_layer.md) for other connection types and selection rules.

## Use the subagent

In the Datus REPL, normally describe the business outcome directly. The main agent delegates requests that create or update semantic datasets and metrics to `semantic_modeling`. Name the tables, target model, metrics, or source SQL when those details matter.

```text
Update sales.yml. Add monthly revenue growth based on orders.order_date, and verify it with a dry-run query.
```

```text
Turn the following SQL into a reusable revenue metric while preserving its filters: SELECT ...
```

To force one request to this subagent, add an agent reference:

```text
Update sales.yml and validate the new metrics. @Agent semantic_modeling
```

For several consecutive turns, select it as the current agent first, then enter normal messages. Use `/agent chat` to return to the main agent:

```text
/agent semantic_modeling
```

The legacy `/<subagent> <message>` form is no longer supported, so `/semantic_modeling ...` is treated as an unknown command. API callers select the built-in agent with `subagent_id: semantic_modeling`.

During one run, the subagent:

1. Inspects existing models and chooses one target using an explicit filename hint, a unique fact table, or the business domain.
2. Reads the live datasource schema and relationship evidence before authoring.
3. Creates or updates datasets, fields, relationships, and metrics in dependency order.
4. Validates the complete target file and, when requested, dry-runs representative metric queries.
5. Reconciles the validated YAML into the Knowledge Base.

It edits exactly one semantic model per run. If no existing model fits, it creates a business-domain model rather than one file per physical table. A dataset uses a query as its `source` only when the result is a durable reusable entity or the request explicitly requires reproducing a query.

The result reports `generated`, `skipped`, or `blocked`. A blocked result explains the missing schema evidence, ambiguous target, invalid definition, or other condition that prevented a safe update.

## Generated files and organization

New files are written under the active project and datasource:

```text
subject/semantic_models/<datasource>/<semantic_model>.yml
```

Each authored file contains one object in the root `semantic_model` list. Group related fact and dimension datasets by business domain, such as `sales.yml` or `marketing.yml`. Use a physical table reference in `source` whenever possible; use SQL only for a stable logical dataset that deserves to be reused.

## YAML contract

Dosi authors OSI Core `0.2.0.dev0` YAML. Unknown core fields are rejected, so Datus-specific behavior belongs in `custom_extensions` rather than arbitrary YAML keys.

The principal fields are:

| Object | Required or important fields | Meaning |
| --- | --- | --- |
| Root | `version`, `semantic_model` | OSI version and model list |
| Semantic model | `name`, `datasets`; optional `description`, `ai_context`, `relationships`, `metrics`, `custom_extensions` | One business domain |
| Dataset | `name`, `source`; optional `primary_key`, `unique_keys`, `fields`, descriptions and extensions | A physical table or reusable query |
| Field | `name`, `expression`; optional `dimension`, `label`, descriptions and extensions | A groupable or filterable attribute |
| Relationship | `name`, `from`, `to`, `from_columns`, `to_columns` | An equi-join; column arrays pair by position |
| Metric | `name`, `expression`; optional descriptions, `ai_context`, and extensions | A reusable aggregate calculation |
| Expression | `dialects` containing `dialect` and `expression` | One or more SQL-dialect implementations |

Use the dialect tag for the active datasource, for example `DUCKDB`, `POSTGRESQL`, `SNOWFLAKE`, `STARROCKS`, `DORIS`, `MYSQL`, `CLICKHOUSE`, `TRINO`, `REDSHIFT`, `HOLOGRES`, `GAUSSDB`, or `ORACLE`. `ANSI_SQL` is appropriate only when the expression is genuinely portable.

### Complete example

The following model reflects that manual REPL output and passed native Dosi validation. Descriptions are translated for this English page; object names, expressions, and DATUS extensions match the generated structure.

```yaml
version: 0.2.0.dev0
semantic_model:
  - name: bank_failures
    datasets:
      - name: bank_failures
        source: main.bank_failures
        description: Each row records one bank failure, its date, and assets at failure.
        ai_context: Analyze bank failure counts and failed assets by time and state.
        fields:
          - name: bank
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: Bank
            description: Name of the failed bank.
          - name: state
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: State
            description: US state code for the bank.
          - name: date
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: Date
            dimension:
              is_time: true
            description: Date of the bank failure.
            custom_extensions:
              - vendor_name: DATUS
                data: '{"v":"1.4","time_granularity":"day"}'
          - name: assets_million
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: '"Assets ($mil.)"'
            label: Assets ($mil.)
            description: Bank assets at failure, in millions of US dollars.
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
        description: Number of bank failure events.
        ai_context:
          instructions: Use date as business time; group by state or time granularity when requested.
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"1.4","dataset":"bank_failures","time_dimension":"bank_failures.date","subject_path":["banking","bank_failures","count"],"unit":"banks"}'

      - name: failed_assets_million
        expression:
          dialects:
            - dialect: DUCKDB
              expression: SUM(bank_failures.assets_million)
        description: Total assets of failed banks, in millions of US dollars.
        ai_context:
          instructions: Use date as business time and sum assets at failure.
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"1.4","time_dimension":"bank_failures.date","subject_path":["banking","bank_failures","assets"],"unit":"USD million"}'
```

This output comes from live schema and data checks rather than a prewritten template. It models only the four fields requested in the prompt and does not declare a key. DuckDB resolves the simple identifiers `Bank`, `State`, and `Date` without quotes; the physical name `Assets ($mil.)` requires double quotes because it contains spaces and punctuation.

## DATUS custom extensions

`custom_extensions` is the OSI-standard escape hatch for vendor behavior. A DATUS entry keeps the document valid OSI while adding behavior used by Dosi. Consumers that do not understand DATUS may ignore the entry and still read the core model.

```yaml
custom_extensions:
  - vendor_name: DATUS
    data: '{"v":"1.4","join_type":"left"}'
```

Important rules:

- `data` is a JSON **string**, not a nested YAML object.
- Put at most one DATUS entry on an OSI object.
- The generated payload carries the engine's current extension version. Let `semantic_modeling` stamp it instead of hardcoding a version in automation.
- Put each key on its supported object: `time_dimension` on a dataset or metric, `time_granularity` on a time field, `join_type` on a relationship, and metric behavior on a metric.

The executable extension keys are:

| Carrier | Keys | Purpose |
| --- | --- | --- |
| Dataset | `time_dimension` | Select the dataset's primary time field |
| Field | `time_granularity` | Declare `day`, `week`, `month`, `quarter`, or `year` granularity |
| Relationship | `join_type` | Choose `left` or `inner` joining |
| Metric | `dataset`, `time_dimension`, `fill_nulls_with` | Resolve metric attribution, time, and null behavior |
| Metric | `window` | Define period-over-period, rolling, cumulative, frame, rank, or value windows |
| Metric | `derive` | Define a filtered or composed metric using same-model base metrics |

Datus also writes presentation and provenance metadata such as `subject_path`, `unit`, `format`, `metric_kind`, `source_type`, `uid`, and `owner`. These keys help Datus organize and display the model but do not change native metric computation.

Keep the core metric expression as a valid aggregate even when `window` or `derive` is present. Dosi validates the fallback expression against supported extension semantics. For more details, see the [Dosi Semantic Adapter](../adapters/dosi_semantic_adapter.md), [Semantic Models](../knowledge_base/semantic_model.md), and [Metrics](../knowledge_base/metrics.md).

## Validation and compatibility

Always validate the whole file after manual edits. Structural validity alone is not enough: dataset and field references, relationship keys, dialect expressions, extension carriers, derived metrics, and windows must also be semantically consistent. `semantic_modeling` performs this final validation and can dry-run metric queries when execution evidence is needed.

Existing MetricFlow and OSI projects remain queryable but are query-only. To author an existing OSI project, change its semantic type to Dosi, then use `semantic_modeling` to repair and validate the YAML in place. MetricFlow YAML migration is not supported.

The retired `gen_semantic_model` and `gen_metrics` names remain reserved only for configuration compatibility. They are hidden from discovery, and direct invocation recommends `semantic_modeling`. A custom agent whose legacy `node_class` or `type` uses either name transparently routes to `semantic_modeling` in a Dosi project.

For `bootstrap-kb`, historical component names are compatibility aliases:

- `--components semantic_model` runs datasets-only scope. It may update model metadata, datasets, fields, and relationships, but protects existing metrics.
- `--components metrics` and `--components semantic_modeling` run the full workflow.
- Combining these components runs the workflow once; full scope wins when `metrics` or `semantic_modeling` is present.

Existing YAML import and `refresh-profile` remain non-LLM operations. They do not reactivate either retired subagent.
