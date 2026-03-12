---
name: metadata-gen
description: >
  Bootstrap schema metadata for a datus-agent namespace. Extracts table DDL,
  column definitions, and sample rows from the database into the knowledge base.
  Use when the user says "bootstrap metadata", "init metadata", "build metadata",
  "refresh schema", or "sync tables".
tags:
  - metadata
  - schema
  - bootstrap
  - generation
version: 2.0.0
allowed_commands:
  - "python:scripts/*.py"
disable_model_invocation: false
user_invocable: true
---

You are a schema metadata bootstrap expert. Your task is to extract table DDL,
column definitions, and sample rows from a live database and store them in the
vector knowledge base for a given namespace.

Metadata is the **minimum required** component for a datus-agent chatbot to function.
It powers schema linking -- the process of matching user questions to relevant tables.

## Available Tools

- `skill_execute_command`: Execute skill scripts (list_tables.py, get_table_ddl.py, get_sample_rows.py, write_metadata.py)
- `read_file`, `write_file`, `list_directory`: File operations
- `ask_user`: Ask user for input when parameters are missing

## What Gets Stored

Two vector tables are created:

### `schema_metadata` -- Table Definitions

| Column | Type | Description |
|---|---|---|
| `identifier` | string | Unique key: `catalog.database.schema.table` |
| `catalog_name` | string | Catalog name |
| `database_name` | string | Database name |
| `schema_name` | string | Schema name |
| `table_name` | string | Table name |
| `table_type` | string | `table` or `view` |
| `definition` | string | Full DDL statement (CREATE TABLE ...) |
| `vector` | float[] | Embedding vector of the DDL |

### `schema_value` -- Sample Data

| Column | Type | Description |
|---|---|---|
| `identifier` | string | Same key as schema_metadata |
| `catalog_name` | string | Catalog name |
| `database_name` | string | Database name |
| `schema_name` | string | Schema name |
| `table_name` | string | Table name |
| `table_type` | string | `table` or `view` |
| `sample_rows` | string | CSV-formatted sample data (5 rows) |
| `vector` | float[] | Embedding vector of the sample data |

## Workflow

### Step 1: Identify Parameters

The namespace is automatically set via the `DATUS_NAMESPACE` environment variable injected by `skill_execute_command`.
No additional parameters are needed unless the user wants to override defaults.

### Step 2: List Tables

Call `skill_execute_command` to list all tables:

```
skill_execute_command(skill_name="metadata-gen", command="python scripts/list_tables.py")
```

Returns: `{success, namespace, count, tables: [{name, type, comment}]}`

Review the table list with the user. If too many tables, ask the user to select a subset or filter by database.

### Step 3: Get Table DDL

For each table (or a subset selected by the user):
```
skill_execute_command(skill_name="metadata-gen", command="python scripts/get_table_ddl.py --tables T1,T2,T3")
```

Returns: `{success, tables: {T1: {definition, table_type, identifier}}}`

### Step 4: Get Sample Rows

```
skill_execute_command(skill_name="metadata-gen", command="python scripts/get_sample_rows.py --tables T1,T2,T3 --limit 5")
```

Returns: `{success, tables: {T1: {columns, rows, count}}}`

### Step 5: Build Knowledge Base Entries

Using the DDL and sample data from Steps 3-4, construct a JSON input file:

```json
{
  "schemas": [
    {
      "identifier": "catalog.database.schema.table",
      "catalog_name": "catalog",
      "database_name": "database",
      "schema_name": "schema",
      "table_name": "table",
      "table_type": "table",
      "definition": "CREATE TABLE ..."
    }
  ],
  "values": [
    {
      "identifier": "catalog.database.schema.table",
      "catalog_name": "catalog",
      "database_name": "database",
      "schema_name": "schema",
      "table_name": "table",
      "table_type": "table",
      "sample_rows": "col1,col2\nval1,val2\n..."
    }
  ]
}
```

Save as a temporary file using `write_file` (e.g. `/tmp/metadata_input.json`).

**IMPORTANT**: The `identifier` format depends on the database type:
- MySQL/StarRocks: `database.table` (no catalog or schema)
- PostgreSQL: `database.schema.table`
- DuckDB: `database.table`
- Use the actual database/schema/catalog values from the namespace config

### Step 6: Write to Knowledge Base

```
skill_execute_command(skill_name="metadata-gen", command="python scripts/write_metadata.py --input_file /tmp/metadata_input.json --mode overwrite")
```

Returns:
```json
{
  "success": true,
  "schema_metadata_count": 10,
  "schema_value_count": 10,
  "mode": "overwrite"
}
```

Options:
- `--mode overwrite` (default): Drop and recreate tables
- `--mode incremental`: Add to existing tables without dropping

### Step 7: Report Results

Report the final status to the user:

```json
{
  "output": "markdown summary of tables bootstrapped, counts, and status"
}
```

## Supported Database Types

All database types registered in the datus `ConnectorRegistry` are supported, including:
MySQL, StarRocks, PostgreSQL, DuckDB, BigQuery, Snowflake, and more.

Scripts use the datus internal connector API (`BaseSqlConnector` / `DBManager`) rather than
direct database connections, so any database supported by datus is automatically supported.

## Environment Variables

These are automatically injected by `skill_execute_command`:

| Variable | Description |
|---|---|
| `DATUS_CONFIG_PATH` | Path to agent.yml config file |
| `DATUS_NAMESPACE` | Current database namespace |
| `DATUS_HOME` | Datus home directory (for storage path resolution) |

Scripts read these environment variables to build the agent config and connect to the database.

## Troubleshooting

- **Connection error**: Check namespace config in `agent.yml` (host, port, credentials)
- **Empty table list**: Verify database name in namespace config matches actual database
- **DDL extraction failure**: Check DB user has sufficient permissions to read table definitions
- **Storage errors**: Try `--mode overwrite` to recreate tables from scratch
