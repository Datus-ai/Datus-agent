# Metrics

A metric is a reusable, executable business calculation such as revenue, active customers, or failed-bank assets. It gives a calculation one stable name and definition so users can ask business questions without rewriting its aggregation logic.

Metrics are part of a [semantic model](semantic_model.md). Their definitions live in the model's OSI YAML; Datus projects them into a separate Knowledge Base store for discovery, and the active semantic adapter compiles and executes metric queries. Dosi is the built-in default adapter.

## What a metric defines

A metric normally includes:

- a datasource-wide unique `name`;
- a business `description` and `ai_context.instructions`;
- an aggregate, ratio, or arithmetic `expression`;
- the dataset and business time it uses;
- a three-level `subject_path` for discovery;
- optional display metadata such as `unit` and `format`;
- optional reusable window behavior, such as rolling, cumulative, or period comparison.

The dataset, fields, and relationships required by a metric are defined in the same semantic-model file. This keeps the calculation and the model needed to execute it in one authoritative artifact.

## Definition, discovery, and execution

Metric data has three distinct roles:

1. **OSI YAML is the definition.** It is the source of truth for expressions, time behavior, descriptions, and extensions.
2. **The `metrics` Knowledge Base store is the discovery projection.** It indexes the metric name, description, subject path, model, dimensions, source YAML, and other retrieval metadata for the active datasource.
3. **The semantic adapter executes queries.** `query_metrics` sends metric names, dimensions, time bounds, filters, and ordering to the active adapter, which compiles and runs the query.

Moving a metric to another `subject_path` changes its classification, not its identity. Metric names must therefore remain unique within one datasource.

## Quick start with natural language

The recommended user flow is to ask a business question. The main agent recognizes metric questions and automatically delegates them to `ask_metrics`:

```bash
datus --datasource duckdb_demo
```

```text
Show the annual trend in the number of failed banks and their total assets.
```

The user does not need to know the metric names. AskMetrics uses the subject tree and metric descriptions to infer that this question needs `bank_failure_count` and `failed_assets_million`, discovers the available time dimension, and queries them by year.

This flow assumes the metrics already exist on the active datasource. If no existing metric can answer the question, AskMetrics reports that limitation instead of switching to an unrelated raw-SQL answer. See [AskMetrics](../subagent/ask_metrics.md) for routing, configuration, and output behavior.

## Current metric YAML

The following definitions are from the `bank_failures.yml` file generated against Datus's sample DuckDB database. They use the dataset and fields shown in [Semantic Models](semantic_model.md):

```yaml
metrics:
  - name: bank_failure_count
    expression:
      dialects:
        - dialect: DUCKDB
          expression: COUNT(*)
    description: Number of failed banks, measured as bank failure event rows.
    ai_context:
      instructions: Use date as business time; group by state or a time grain when requested.
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

Metric expressions use the same `expression.dialects[]` structure as field expressions. Qualify fields with their dataset name, as in `SUM(bank_failures.assets_million)`. An aggregate such as `COUNT(*)` cannot reveal its dataset from a qualified field, so its DATUS extension supplies `dataset` explicitly.

Do not place `dataset`, `time_dimension`, `subject_path`, `unit`, or window settings at the metric's top level. They are Datus behavior and belong in the metric's DATUS `custom_extensions` entry.

## Metric extensions

The DATUS extension is an OSI `custom_extensions` entry whose `data` value is a JSON string:

```yaml
custom_extensions:
  - vendor_name: DATUS
    data: '{"v":"1.4","time_dimension":"orders.order_date","subject_path":["sales","revenue","total"],"unit":"USD"}'
```

Common keys are:

| Key | Purpose |
| --- | --- |
| `dataset` | Anchors an expression when its dataset cannot be inferred, for example `COUNT(*)`. |
| `time_dimension` | Selects the metric's business time field. Qualify it when names may collide. |
| `subject_path` | Places the metric in a three-level discovery hierarchy. |
| `unit`, `format` | Describe how results should be labeled or displayed. |
| `fill_nulls_with` | Defines explicit null-filling behavior. |
| `window` | Defines a reusable rolling, cumulative, comparison, ranking, or value window. |

`semantic_modeling` writes the extension version required by the installed engine. Preserve generated version values and keep only one DATUS entry on each object.

## Querying metrics

AskMetrics is the normal user-facing query path. For integrations or advanced agent workflows, the underlying tools separate discovery from execution:

```python
# Discover candidate definitions from the Knowledge Base.
search_metrics(
    query_text="annual failed-bank count and assets",
    top_n=5,
)

# Discover queryable dimensions and the metric's primary time axis.
get_dimensions(metric_name="bank_failure_count")

# Execute both metrics by year.
query_metrics(
    metrics=["bank_failure_count", "failed_assets_million"],
    dimensions=["metric_time"],
    time_granularity="year",
    order_by=["metric_time__year"],
)
```

`query_metrics` accepts these current parameters:

| Parameter | Meaning |
| --- | --- |
| `metrics` | One or more exact metric names. |
| `dimensions` | Dimensions returned by `get_dimensions`; with Dosi, `metric_time` selects the primary time axis. |
| `path` | Optional subject-tree path used to disambiguate a metric. |
| `time_start`, `time_end` | Optional half-open time range: start is inclusive and end is exclusive. ISO dates and relative values such as `-7d` or `now` are supported by the adapter. |
| `time_granularity` | `day`, `week`, `month`, `quarter`, or `year`. |
| `where` | Optional filter expression without the `WHERE` keyword. |
| `order_by` | Result columns to sort; prefix a name with `-` for descending order. |
| `limit` | Maximum rows, used only when the user asks for Top N, a preview, or another explicit row limit. |
| `dry_run` | Compile and validate the query plan without returning live metric values. |

For example, to include all of January 2024, use `time_start="2024-01-01"` and `time_end="2024-02-01"`.

## Creating and updating metrics

Use `semantic_modeling` for normal authoring. It can create required datasets and fields, add or update metrics in the same YAML, validate the complete model, dry-run representative queries when requested, and synchronize both Knowledge Base projections.

```text
Add a monthly revenue growth metric to the sales model, use order_date as business time, validate it, and dry-run a yearly query.
```

For batch authoring from a CSV containing `question` and `sql`, use the unified semantic-modeling component:

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components semantic_modeling \
  --success_story path/to/success_story.csv \
  --kb_update_strategy incremental \
  --metrics-batch-size 5
```

`--metrics-batch-size` controls how many history rows are processed in one authoring batch. `--subject_tree` may provide allowed comma-separated categories; when omitted, authoring reuses existing categories or creates suitable ones.

The default `check` strategy does not author metrics. It only reports the current semantic-dataset and metric projection counts, so use `incremental` or `overwrite` when generation is intended.

## Synchronizing the Knowledge Base

A successful `semantic_modeling` run synchronizes its target file automatically. If YAML was edited manually, reconcile all semantic-object and metric projections with:

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components semantic_model \
  --kb_update_strategy sync-yaml
```

`sync-yaml` reads every model under the active datasource by default, replaces each artifact's projected rows, removes metrics no longer declared in that artifact, and prunes rows for deleted files during a directory-wide sync. It does not call an LLM or query the warehouse.

To check the number of indexed metrics without changing them:

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components metrics \
  --kb_update_strategy check
```

## Related guides

- [AskMetrics](../subagent/ask_metrics.md): natural-language metric questions and capability boundaries
- [Semantic Modeling](../subagent/semantic_modeling.md): end-to-end authoring and complete generated YAML
- [Semantic Models](semantic_model.md): datasets, fields, relationships, and Knowledge Base projection
- [Semantic layer configuration](../configuration/semantic_layer.md): semantic adapter configuration
