# Dosi Semantic Engine

Datus Agent includes the `dosi-engine` Python binding as a locked dependency. Agent's semantic tools call the native engine through a small runtime in `datus/tools/semantic_tools/dosi/`.

## Install

From a Datus-agent checkout:

```bash
uv sync --locked
```

The lockfile pins the engine version used by Agent and CI. Use the locked environment for development and deployment.

## Configure

```yaml
agent:
  services:
    datasources:
      warehouse:
        type: duckdb
        uri: /absolute/path/to/warehouse.db
```

Dosi is built in and needs no engine selection. Agent discovers OSI YAML or JSON files recursively under `subject/semantic_models/<datasource>/`, excluding per-metric fragments. Use `semantic_model_path` to select a specific file for a request. Old `agent.services.semantic_layer` and project `semantic` settings are ignored with a warning.

## Agent features

Agent uses Dosi to discover semantic models, metrics, and metric-specific dimensions; compile and execute metric queries; validate exact model artifacts; and run native attribution. The `semantic_modeling` workflow uses the native authoring contract and validation. Explorer reads, writes, and deletes metric nodes in OSI YAML files, then synchronizes the affected Knowledge Base entries. The external Dosi CLI, REST server, and MCP server are outside Agent's integration.

Each model file has its own engine instance. The runtime refreshes it after the model file changes. Dosi handles SQL dialects and query planning; Agent supplies the active datasource connection and keeps subject paths in its Knowledge Base.
