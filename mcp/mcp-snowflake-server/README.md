# Snowflake MCP Server

[![smithery badge](https://smithery.ai/badge/mcp_snowflake_server)](https://smithery.ai/server/mcp_snowflake_server) [![PyPI - Version](https://img.shields.io/pypi/dm/mcp-snowflake-server?color&logo=pypi&logoColor=white&label=PyPI%20downloads)](https://pypi.org/project/mcp-snowflake-server/)

---

## Overview
A Model Context Protocol (MCP) server implementation that provides database interaction with Snowflake. This server enables running SQL queries via tools and exposes data insights and schema context as resources.

---

## Components

### Resources

- **`memo://insights`**  
  A continuously updated memo aggregating discovered data insights.  
  Updated automatically when new insights are appended via the `append_insight` tool.

- **`context://table/{table_name}`**  
  (If prefetch enabled) Per-table schema summaries, including columns and comments, exposed as individual resources.

---

### Tools

The server exposes the following tools:

#### Query Tools

- **`read_query`**  
  Execute `SELECT` queries to read data from the database.  
  **Input:**  
  - `query` (string): The `SELECT` SQL query to execute  
  **Returns:** Query results as array of objects

- **`write_query`** (enabled only with `--allow-write`)  
  Execute `INSERT`, `UPDATE`, or `DELETE` queries.  
  **Input:**  
  - `query` (string): The SQL modification query  
  **Returns:** Number of affected rows or confirmation

- **`create_table`** (enabled only with `--allow-write`)  
  Create new tables in the database.  
  **Input:**  
  - `query` (string): `CREATE TABLE` SQL statement  
  **Returns:** Confirmation of table creation

#### Schema Tools

- **`list_databases`**  
  List all databases in the Snowflake instance.  
  **Returns:** Array of database names

- **`list_schemas`**  
  List all schemas within a specific database.  
  **Input:**  
  - `database` (string): Name of the database  
  **Returns:** Array of schema names

- **`list_tables`**  
  List all tables within a specific database and schema.  
  **Input:**  
  - `database` (string): Name of the database  
  - `schema` (string): Name of the schema  
  **Returns:** Array of table metadata

- **`describe_table`**  
  View column information for a specific table.  
  **Input:**  
  - `table_name` (string): Fully qualified table name (`database.schema.table`)  
  **Returns:** Array of column definitions with names, types, nullability, defaults, and comments

#### Analysis Tools

- **`append_insight`**  
  Add new data insights to the memo resource.  
  **Input:**  
  - `insight` (string): Data insight discovered from analysis  
  **Returns:** Confirmation of insight addition  
  **Effect:** Triggers update of `memo://insights` resource

---

## Usage with Claude Desktop

### Installing via Smithery

To install Snowflake Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/mcp_snowflake_server):

```bash
npx -y @smithery/cli install mcp_snowflake_server --client claude
```

---

### Installing via UVX

```json
"mcpServers": {
  "snowflake_pip": {
    "command": "uvx",
    "args": [
      "--python=3.12",  // Optional: specify Python version <=3.12
      "mcp_snowflake_server",
      "--account", "your_account",
      "--warehouse", "your_warehouse",
      "--user", "your_user",
      "--password", "your_password",
      "--role", "your_role",
      "--database", "your_database",
      "--schema", "your_schema"
      // Optionally: "--allow_write"
      // Optionally: "--log_dir", "/absolute/path/to/logs"
      // Optionally: "--log_level", "DEBUG"/"INFO"/"WARNING"/"ERROR"/"CRITICAL"
      // Optionally: "--exclude_tools", "{tool_name}", ["{other_tool_name}"]
    ]
  }
}
```

---

### Installing Locally

1. Install [Claude AI Desktop App](https://claude.ai/download)

2. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create a `.env` file with your Snowflake credentials:

```bash
SNOWFLAKE_USER="xxx@your_email.com"
SNOWFLAKE_ACCOUNT="xxx"
SNOWFLAKE_ROLE="xxx"
SNOWFLAKE_DATABASE="xxx"
SNOWFLAKE_SCHEMA="xxx"
SNOWFLAKE_WAREHOUSE="xxx"
SNOWFLAKE_PASSWORD="xxx"
# Alternatively, use external browser authentication:
# SNOWFLAKE_AUTHENTICATOR="externalbrowser"
```

4. [Optional] Modify `runtime_config.json` to set exclusion patterns for databases, schemas, or tables.

5. Test locally:

```bash
uv --directory /absolute/path/to/mcp_snowflake_server run mcp_snowflake_server
```

6. Add the server to your `claude_desktop_config.json`:

```json
"mcpServers": {
  "snowflake_local": {
    "command": "/absolute/path/to/uv",
    "args": [
      "--python=3.12",  // Optional
      "--directory", "/absolute/path/to/mcp_snowflake_server",
      "run", "mcp_snowflake_server"
      // Optionally: "--allow_write"
      // Optionally: "--log_dir", "/absolute/path/to/logs"
      // Optionally: "--log_level", "DEBUG"/"INFO"/"WARNING"/"ERROR"/"CRITICAL"
      // Optionally: "--exclude_tools", "{tool_name}", ["{other_tool_name}"]
    ]
  }
}
```

---

## Notes

- By default, **write operations are disabled**. Enable them explicitly with `--allow-write`.
- The server supports filtering out specific databases, schemas, or tables via exclusion patterns.
- The server exposes additional per-table context resources if prefetching is enabled.
- The `append_insight` tool updates the `memo://insights` resource dynamically.

---

## License

MIT
