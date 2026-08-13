# MetricFlow Semantic Adapter

The MetricFlow semantic adapter connects Datus Agent to existing MetricFlow-native semantic model and metric YAML files.

> **Query-only compatibility:** existing MetricFlow projects remain queryable,
> but Datus no longer generates or edits MetricFlow semantic YAML. New semantic
> authoring is available only through `semantic_modeling` in Dosi projects.

Use this adapter when MetricFlow YAML remains the source format maintained by external users or automation.

## Installation

```bash
pip install datus-semantic-metricflow
```

From source:

```bash
pip install -e ../datus-semantic-adapter/datus-semantic-core
pip install -e ../datus-semantic-adapter/datus-semantic-metricflow
```

## Configuration

```yaml
agent:
  services:
    semantic_layer:
      metricflow:
        timeout: 300
        config_path: ./conf/agent.yml   # optional advanced override
        default: true
```

`config_path` is optional. In normal use, Datus builds the MetricFlow runtime config from:

1. the selected datasource in `services.datasources`
2. the current project semantic model directory
3. the active `agent.home`

## Semantic Model Directory

By default, Datus points MetricFlow at the current project's semantic model directory:

```text
{project_root}/subject/semantic_models/
```

Existing YAML under this directory is included in validation, even when the files are project-local or gitignored.

## Legacy Source Model

MetricFlow projects load MetricFlow YAML directly.

Semantic model files use `data_source` documents:

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

Metric files use `metric` documents:

```yaml
metric:
  name: revenue
  type: measure_proxy
  type_params:
    measures:
      - revenue_sum
```

## Query Flow

With MetricFlow as the active semantic layer, existing assets continue to support:

1. `validate_semantic()` validates the full MetricFlow model.
2. `query_metrics(...)` compiles and executes existing metrics.
3. `ask_metrics`, metric preview, API metric queries, reports, and dashboards use the same query path.

The retired `gen_semantic_model` and `gen_metrics` names return an error that recommends migrating to Dosi and using `semantic_modeling`.

## Supported Query Features

The adapter supports the common semantic adapter methods:

- `list_metrics`
- `get_dimensions`
- `query_metrics`
- `validate_semantic`

MetricFlow handles SQL generation, joins, time granularity, metric constraints, cumulative metrics, ratio metrics, expression metrics, and derived metrics according to the MetricFlow model.

For the underlying MetricFlow engine concepts and supported warehouses, see [Datus-MetricFlow Introduction](../metricflow/introduction.md).

## Other Semantic Formats

Use [OSI Semantic Adapter](osi_semantic_adapter.md) to query existing strict OSI core YAML. Use Dosi for new semantic authoring.
