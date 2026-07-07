# MetricFlow Semantic Adapter

The MetricFlow semantic adapter connects Datus Agent to MetricFlow-native semantic model and metric YAML files.

Use this adapter when you want generated semantic assets to be MetricFlow YAML and you expect MetricFlow to be the source format maintained by users or automation.

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

Generated YAML under this directory is included in validation, even when the files are project-local or gitignored.

## Authoring Model

MetricFlow mode authors MetricFlow YAML directly.

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

## Generation Flow

With MetricFlow as the active semantic layer:

1. `gen_semantic_model` writes MetricFlow semantic model YAML.
2. `gen_metrics` writes MetricFlow metric YAML.
3. `validate_semantic()` validates the full MetricFlow model.
4. `query_metrics(..., dry_run=True)` verifies generated metrics can compile to SQL.
5. `end_semantic_model_generation` and `end_metric_generation` sync validated assets to the Knowledge Base.

## Supported Query Features

The adapter supports the common semantic adapter methods:

- `list_metrics`
- `get_dimensions`
- `query_metrics`
- `validate_semantic`

MetricFlow handles SQL generation, joins, time granularity, metric constraints, cumulative metrics, ratio metrics, expression metrics, and derived metrics according to the MetricFlow model.

For the underlying MetricFlow engine concepts and supported warehouses, see [Datus-MetricFlow Introduction](../metricflow/introduction.md).

## When to Use OSI Instead

Use [OSI Semantic Adapter](osi_semantic_adapter.md) when you want the authored source files to be strict OSI core YAML and want Datus/MetricFlow-specific execution hints isolated in `custom_extensions`.
