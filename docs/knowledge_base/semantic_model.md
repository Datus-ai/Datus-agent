# Semantic Models

A semantic model turns physical database structures into reusable business concepts. It tells Datus which tables or queries represent a business dataset, what its fields mean, which time field should drive analysis, and how datasets may be joined. Metrics are defined in the same model and are covered separately in [Metrics](metrics.md).

Datus authors semantic models as strict OSI YAML through Dosi. The YAML file is the source of truth; the Knowledge Base contains a searchable projection of that file for agents and product interfaces.

## What a semantic model contains

| Object | Purpose |
| --- | --- |
| **Semantic model** | Groups related datasets, relationships, and metrics into one business domain. |
| **Dataset** | Maps a business entity or event set to a physical table or a reusable SQL query. |
| **Field** | Gives a source column or expression a stable name and business description. A field may also be a time dimension. |
| **Relationship** | Defines an allowed equality join between two datasets. |
| **Metric** | Defines a reusable business calculation. Metrics live in the same YAML but use a separate Knowledge Base projection and query workflow. |

For example, a banking model can map `main.bank_failures` to a `bank_failures` dataset, describe its bank, state, failure date, and asset fields, and define metrics that count failed banks or sum their assets.

## Source file and Knowledge Base projection

Authored files are stored by datasource:

```text
subject/semantic_models/<datasource>/<semantic_model>.yml
```

Each file contains one semantic model. A successful `semantic_modeling` run validates the complete file, writes it, and synchronizes its contents to the Knowledge Base.

The `semantic_dataset` projection stores one row for each authored dataset, field, or relationship:

- A dataset is identified by `(semantic_model, dataset)` and carries its physical source or reusable query.
- A field adds its dataset and field name, expression, time/key flags, and description.
- A relationship is identified by `(semantic_model, relationship)` and records both endpoints and their paired columns.
- A query-backed dataset has a `source_query` but no `source_table`, so it cannot be confused with a physical table of the same name.

Metrics are not stored in `semantic_dataset`; they are projected into the `metrics` Knowledge Base store. Both projections retain the source YAML path and are scoped to the active datasource.

## Quick start

Start Datus with the datasource you want to model, then describe the desired business concepts in natural language. The main agent automatically delegates authoring requests to `semantic_modeling`:

```bash
datus --datasource duckdb_demo
```

```text
Model bank_failures. Add the bank, state, failure date, and asset fields, use the failure date as the primary time dimension, and define metrics for the number of failed banks and their total assets. Validate the model and run representative metric queries.
```

The generated file is:

```text
subject/semantic_models/duckdb_demo/bank_failures.yml
```

The DuckDB datasource used in this example is configured in the [Semantic Modeling guide](../subagent/semantic_modeling.md). That guide also covers supported databases, datasource setup, and interactive agent selection.

## Current YAML structure

The following dataset is from the model generated against Datus's `duckdb-demo.duckdb` sample database. It shows the current OSI structure used by Dosi:

```yaml
version: 0.2.0.dev0
semantic_model:
  - name: bank_failures
    datasets:
      - name: bank_failures
        source: main.bank_failures
        description: Bank failure event fact table; each row records one failed bank.
        ai_context: Use this dataset to analyze bank failures by date and state.
        fields:
          - name: bank
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: Bank
            description: Name of the failed bank
          - name: state
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: State
            description: US state code for the failed bank
          - name: date
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: Date
            dimension:
              is_time: true
            description: Bank failure date
            custom_extensions:
              - vendor_name: DATUS
                data: '{"v":"1.4","time_granularity":"day"}'
          - name: assets_million
            expression:
              dialects:
                - dialect: DUCKDB
                  expression: '"Assets ($mil.)"'
            label: Assets ($mil.)
            description: Assets at failure, in millions of US dollars
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"1.4","time_dimension":"date"}'
    relationships: []
    metrics: []
```

Important rules:

- The root keys are `version` and `semantic_model`; `semantic_model` is a list.
- A physical dataset uses a qualified table name in `source`. A reusable query may be used only when the query result itself is a stable business dataset.
- Expressions use `expression.dialects[]`. Use the dialect selected for the active datasource, such as `DUCKDB`, `POSTGRESQL`, `SNOWFLAKE`, or `STARROCKS`.
- `primary_key` and `unique_keys` describe verified keys. Datus does not infer a key from a column name or a single query.
- A time field uses `dimension.is_time: true`; its stored grain is carried by the DATUS `time_granularity` extension.
- Relationships belong to the semantic model, not to an individual dataset. Their `from_columns` and `to_columns` lists are positionally paired, and the target columns must be a complete verified key.
- Metric definitions belong under the same semantic model's `metrics` list. See [Metrics](metrics.md) for their current shape.

## Relationships

A relationship declares the join path Dosi may use when a query combines fields from different datasets:

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

`from` is normally the many side and `to` the one side. For a composite key, list all components in the same order on both sides.

## DATUS extensions

`custom_extensions` is the OSI mechanism for Datus-specific behavior. A DATUS entry keeps the document valid OSI while adding execution or display hints used by Dosi.

```yaml
custom_extensions:
  - vendor_name: DATUS
    data: '{"v":"1.4","time_dimension":"date"}'
```

The `data` value is a JSON string, not a nested YAML object. `semantic_modeling` writes the extension version expected by the installed engine, so generated version values should be preserved.

Common semantic-model extensions are:

| Object | Keys | Purpose |
| --- | --- | --- |
| Dataset | `time_dimension`, `source_type` | Select the primary business time or mark a query-backed source. |
| Time field | `time_granularity` | Record the source grain: day, week, month, quarter, or year. |
| Relationship | `join_type` | Select `left` or `inner` join behavior. |

Metric-specific extensions are described in [Metrics](metrics.md).

## Creating and updating models

For normal interactive work, ask Datus to create or update the model in natural language. `semantic_modeling` inspects the live schema, edits one target file, validates it, and synchronizes both semantic objects and metrics after a successful run.

For batch authoring from question/SQL history, provide a CSV with `question` and `sql` columns:

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components semantic_modeling \
  --success_story path/to/success_story.csv \
  --kb_update_strategy incremental \
  --metrics-batch-size 5
```

Use `--components semantic_model` instead when the batch should author datasets and relationships only. `overwrite` and `incremental` both reconcile each selected semantic YAML artifact; neither should be mistaken for a read-only check. The default `check` strategy only reports the current projection counts.

## Synchronizing existing YAML

If YAML files were edited outside `semantic_modeling`, rebuild their Knowledge Base projections with:

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components semantic_model \
  --kb_update_strategy sync-yaml
```

This reads all semantic YAML files for the active datasource, validates their shape, and reconciles both the `semantic_dataset` and `metrics` projections. It does not call an LLM or query the warehouse. Pass `--semantic_yaml path/to/model.yml` only when you intentionally want to synchronize one file or directory.

To refresh profile-derived descriptions in one existing model, use `refresh-profile` with that YAML and the historical SQL CSV. This performs bounded read-only profiling, updates the YAML descriptions, and synchronizes the result:

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components semantic_model \
  --kb_update_strategy refresh-profile \
  --semantic_yaml path/to/model.yml \
  --success_story path/to/success_story.csv
```

## Retrieval and use

Agents search semantic objects with:

```python
search_semantic_objects(
    query_text="bank failure date and state",
    kinds=["dataset", "field", "relationship"],
    top_n=5,
)
```

This search does not return metrics; use `search_metrics` for those. Physical-table inspection can also merge the matching dataset projection into table metadata, while the Catalog semantic-model view is read-only. Edit the YAML through `semantic_modeling`, then let synchronization refresh every consumer.

## Related guides

- [Semantic Modeling](../subagent/semantic_modeling.md): end-to-end authoring, configuration, supported databases, and complete generated YAML
- [Metrics](metrics.md): metric definitions, Knowledge Base projection, and query parameters
- [AskMetrics](../subagent/ask_metrics.md): natural-language metric questions
- [Semantic layer configuration](../configuration/semantic_layer.md): semantic adapter configuration
