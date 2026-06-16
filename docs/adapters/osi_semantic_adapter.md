# OSI Semantic Adapter

This page describes Datus support for OSI (Open Semantic Interchange) semantic models and metrics. The implementation spans two repositories:

- `datus-agent`: generates strict OSI core YAML with LLM agents, validates and dry-runs generated assets, then syncs them to the Knowledge Base for `ask_metrics`.
- `datus-semantic-adapter`: provides the `datus-semantic-osi` adapter. It loads OSI YAML, validates the OSI core schema, compiles the document into Datus Semantic IR, and lowers it to an execution backend. The current backend is MetricFlow.

## Positioning

In Datus, OSI is the authoring format for semantic models and metrics. MetricFlow is the default execution backend that renders SQL and executes metric queries.

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

Users and LLM agents should not write MetricFlow YAML fields in OSI mode, such as `data_source`, `measures`, `measure_proxy`, or `type_params`. The OSI adapter generates those backend artifacts internally. They are disposable execution artifacts, not the source files users maintain.

## Installation

Install the OSI adapter from the `datus-semantic-adapter` repository.

From a released package:

```bash
pip install "datus-semantic-osi[metricflow]"
```

From source:

```bash
pip install -e ../datus-semantic-adapter/datus-semantic-core
pip install -e "../datus-semantic-adapter/datus-semantic-metricflow"
pip install -e "../datus-semantic-adapter/datus-semantic-osi[metricflow]"
```

The `metricflow` extra installs dependencies required by the MetricFlow execution backend. Static OSI compilation and validation can work without the extra, but metric querying through Datus requires an available execution backend.

## Configuration

Configure the semantic layer as `osi` in `agent.yml`, and point semantic nodes to `semantic_adapter: osi`.

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
      # authoring_format is optional; semantic_adapter=osi enables OSI authoring.
      # authoring_format: osi

    gen_metrics:
      semantic_adapter: osi

    ask_metrics:
      semantic_adapter: osi
```

`datus-agent` resolves the authoring format in this order:

1. Use `authoring_format: osi` when explicitly configured on the node.
2. Use OSI authoring when the resolved `semantic_adapter` is `osi`.
3. Otherwise, keep the default MetricFlow authoring path.

## Semantic Model Generation

In OSI mode, `gen_semantic_model` writes a strict OSI core document. The root shape is fixed:

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

Key rules:

- Use one canonical dataset per physical table. Do not declare separate datasets for different queries or different metrics over the same table.
- Use OSI core `fields`, not MetricFlow `dimensions`.
- Dataset `source` is a table-name string, not `{table: ...}`.
- Declare primary keys in `primary_key`.
- Mark time fields with `dimension.is_time: true`; put Datus time-granularity hints in `custom_extensions`.
- Declare relationships under the semantic model object, not inside datasets.

## Metric Generation

In OSI mode, `gen_metrics` appends metrics under `semantic_model[0].metrics`. Business expressions live in OSI core `expression`; Datus execution hints live in `custom_extensions`.

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

Supported metric patterns:

| Pattern | OSI authoring | Backend lowering |
|---------|---------------|------------------|
| Basic aggregate | `SUM`, `COUNT`, `COUNT(DISTINCT)`, `AVG`, `MIN`, `MAX` | MetricFlow `measure_proxy` |
| Conditional aggregate | `COUNT(DISTINCT CASE WHEN ... THEN id END)` | Backing measure + metric |
| Expression metric | Arithmetic over aggregate results | MetricFlow `expr` |
| Ratio | Aggregate division, or DATUS `numerator` / `denominator` hints | MetricFlow `ratio` |
| Rolling / cumulative | Base aggregate + DATUS `window` or `grain_to_date` | MetricFlow `cumulative` |
| Period-over-period | Derived metric + input metric `offset_window` | MetricFlow `derived` |
| Joined dimensions | OSI relationship + joined dimension | MetricFlow join identifier |

## Period-over-period and offset_window

Do not write SQL window functions directly in OSI metric expressions:

```sql
LAG(revenue) OVER (...)
ROW_NUMBER() OVER (...)
```

Model period-over-period metrics as a base metric plus derived metrics. For example:

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

OSI metrics describe reusable business semantics, not one-off query SQL. `offset_window` expresses "the same metric shifted by a prior period" as semantic metadata; the execution backend is responsible for rendering an equivalent query plan.

## Multi-table Joins

Multi-table joins are represented with OSI core relationships:

```yaml
relationships:
  - name: orders_to_customers
    from: orders
    to: customers
    from_columns: [customer_id]
    to_columns: [customer_id]
```

The current Datus execution profile supports single-column relationships, primarily `many_to_one` and `one_to_one`. When lowering to MetricFlow, the adapter creates a foreign identifier on the `from` dataset so fact metrics can be grouped or filtered by fields from the dimension dataset.

Joined dimension names usually include the target dataset's primary identifier prefix, for example:

```text
customer_id__country
```

`ask_metrics` discovers these queryable dimensions through `list_metrics` and `get_dimensions`.

## Validation and Publishing

Datus enforces validation before publishing OSI assets:

1. `validate_semantic(scope="semantic_model")` validates generated semantic models.
2. `validate_semantic(scope="all")` validates the full semantic layer.
3. `query_metrics(..., dry_run=True)` validates generated metrics by rendering SQL.
4. `end_semantic_model_generation` / `end_metric_generation` sync semantic objects and metrics to the Knowledge Base.

If validation or dry-run fails, Datus does not publish the metric to the Knowledge Base. This ensures `ask_metrics` only queries validated metrics.

## ask_metrics Queries

`ask_metrics` uses the unified semantic adapter interface. With the OSI adapter, it still calls:

- `list_metrics`
- `get_dimensions`
- `query_metrics`
- `validate_semantic`

The OSI adapter returns Datus metadata for each metric, such as:

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

This metadata helps `ask_metrics` choose the correct metrics, dimensions, and query parameters.

## SQL That Should Not Become Metrics

`gen_metrics` does not force these SQL patterns into OSI metrics:

- Detail lists: `SELECT col1, col2 ...`
- Distinct detail lists: `SELECT DISTINCT ...`
- Ranking lists: `ROW_NUMBER()`, `RANK()`, or `DENSE_RANK()` producing row-level output
- TopN per group, such as "top N activities per channel"
- Queries whose main output is row-level records rather than aggregate metrics

These patterns may be modeled later with derived datasets, materialized views, or a query layer, but they are outside the current `gen_metrics` metric-generation scope.

## Current Limits

- The OSI adapter currently defaults to the MetricFlow execution backend.
- The current relationship execution profile supports single-column joins. Composite joins require future extension.
- SQL window functions cannot be written directly in OSI metric expressions. Use `offset_window` for period comparisons; ranking and TopN detail queries need a query layer or precomputed dataset.
- When a semantic model has multiple datasets, each metric must declare its owning `dataset` in the DATUS custom extension.
- Datus execution information outside OSI core must be encoded in `custom_extensions[{vendor_name: DATUS}]`, not as top-level OSI fields.
