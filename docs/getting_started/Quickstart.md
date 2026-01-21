This guide introduces three supported usage **modes** that allow you to adopt Datus based on your needs and environment:

### **🗨️ 1. Datus CLI (Chat Mode)**

Use Datus like a chatbot: type natural language questions, get SQL or summaries back. Ideal for **ad hoc queries** and *
*fast metric exploration**.

1. Install Datus:

```bash
pip install datus
```

- Configure your environment

```bash
datus-init
cp ~/.datus/conf/agent.yml.qs ~/.datus/conf/agent.yml
```

Export environment variables (e.g., default model is DeepSeek-v3)

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

- Launch the CLI

```bash
datus-cli --namespace local_duckdb
```

- Start chat with Datus.

```bash
Datus> /Ask any questions you'd like to know
```

For example:

```bash
Datus> /list all tables
Using existing session: chat_session_97e067fb (tokens: 31586, actions: 24)
Processing chat request...
╭─────────────────────────────────────────────────────────────────────────────────── Action Stream ────────────────────────────────────────────────────────────────────────────────────╮
│ 🟡 User: list all tables                                                                                                                                                             │
│ 💬 Generated response: # All Tables in the Database                                                                                                                                  │
│                                                                                                                                                                                      │
│ Here is the complete list of all tables available in the main schema:                                                                                                                │
│ ...                                                                                                                                                                                  │
│ 🔧 Tool call - list_tables('{"catalog": "", "database": "", "schema_name": "ma...') - ✓                                                                                              │
│     ✓ Success (0.5s)                                                                                                                                                                 │
│ 💬 Thinking: ```json                                                                                                                                                                 │
│ {                                                                                                                                                                                    │
│   "output": "# All Tables in the Database\n\nHere is the complete list of all tables available in the main schema:\n\n## Tables List\n\n| Table Name | Type                          │
│ |\n|------------|------|\n| bank_failures | table |\n| boxplot | table |\n| calendar | table |\n| candle | table |\n| christmas_cost | table |\n| companies | table |\n|             │
│ country_stats_scatter | table |\n| gold_vs_bitcoin | table |\n| japan_births_deaths | table |\n| japan_population | table |\n| metrics | table |\n| niger_population | table |\n|    │
│ quotes | table |\n| radar | table |\n| sankey | table |\n| search_trends | table |\n| tree | table |\n\n## Total Tables: 17\n\nThe database contains a diverse set of tables         │
│ covering various topics including:\n- Financial data (bank_failures, gold_vs_bitcoin)\n- Demographic data (japan_population, niger_population)\n- Company information (companies)\n- │
│ Holiday/seasonal data (christmas_cost)\n- Search trends and metrics\n- Various visualization-ready datasets (boxplot, candle, radar, sankey, tree)\n\nWould you like me to explore   │
│ any specific table in more detail or help you with queries on any of these datasets?"                                                                                                │
│ }                                                                                                                                                                                    │
│ ```                                                                                                                                                                                  │
│ 💬 Chat interaction completed successfully                                                                                                                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                              All Tables in the Database                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Here is the complete list of all tables available in the main schema:                                                                                                                   

                                                                                      Tables List                                                                                       

                                 
  Table Name              Type   
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  bank_failures           table  
  boxplot                 table  
  calendar                table  
  candle                  table  
  christmas_cost          table  
  companies               table  
  country_stats_scatter   table  
  gold_vs_bitcoin         table  
  japan_births_deaths     table  
  japan_population        table  
  metrics                 table  
  niger_population        table  
  quotes                  table  
  radar                   table  
  sankey                  table  
  search_trends           table  
  tree                    table  
                                 

                                                                                    Total Tables: 17                                                                                    

The database contains a diverse set of tables covering various topics including:                                                                                                        

 • Financial data (bank_failures, gold_vs_bitcoin)                                                                                                                                      
 • Demographic data (japan_population, niger_population)                                                                                                                                
 • Company information (companies)                                                                                                                                                      
 • Holiday/seasonal data (christmas_cost)                                                                                                                                               
 • Search trends and metrics                                                                                                                                                            
 • Various visualization-ready datasets (boxplot, candle, radar, sankey, tree)                                                                                                          

Would you like me to explore any specific table in more detail or help you with queries on any of these datasets? 
Datus>  
```

For more command references and options, see the documentation: [Cli references](./cli/reference.md)

### **🧪 2. Datus Benchmark (Docker Mode)**

Run benchmark tests in a pre-configured Docker image to evaluate Datus using standard benchmark datasets: Bird and
Spider-snow

- pull image

```bash
docker pull luochen2025/datus-agent
```

- start container

```bash
docker run --name datus \
--env DEEPSEEK_API_KEY=<your_api_key>  \
--env SNOWFLAKE_ACCOUNT=<your_snowflake_acount>  \
--env SNOWFLAKE_USERNAME=<your_snowflake_username>  \
--env SNOWFLAKE_PASSWORD=<your_snowflake_password>  \
-d luochen2025/datus-agent
```

- running benchmark
    - Bird

  Run a specific task by ID

   ```bash
   docker exec -it datus python -m datus.main benchmark  \
   --namespace bird_sqlite \
   --benchmark bird_dev \
   --benchmark_task_ids 14
   ```

  Run all tasks

   ```bash
   docker exec -it datus python -m datus.main benchmark  \
   --namespace bird_sqlite \
   --benchmark bird_dev
   ```

    - Spider-snow

  Run a specific task by ID

   ```bash
   docker exec -it datus python -m datus.main benchmark  \
   --namespace snowflake \
   --benchmark spider2 \
   --benchmark_task_ids sf_bq104
   ```

  Run all tasks

   ```bash
   docker exec -it datus python -m datus.main benchmark  \
   --namespace snowflake \
   --benchmark spider2
   ```

For more detailed information about Datus benchmarking: [Benchmark](./benchmark/benchmark_manual.md)

### **📊 3. Datus Metric (MetricFlow Integration)**

Connect Datus to **MetricFlow** and a data warehouse (e.g., StarRocks) to enable **semantic understanding of metrics** —
with support for model-based reasoning, date interpretation, and domain code mapping.

**Prerequisites**

- **Python:**
    - Datus Agent already installed
    - MetricFlow CLI itself: **Python 3.9** (separate venv via Poetry)
- install metricflow mcp server

```bash
pip show datus_agent           # note the installed path
cd <datus_install_path>/mcp/mcp-metricflow-server
uv sync
```

- Set up MetricFlow. Since MetricFlow requires Python 3.9, open a new terminal

```bash
# Before continuing, create a dedicated directory for your MetricFlow project. This path will be reused in later steps — if you change it here, be sure to update it consistently throughout the guide.
mkdir -p ~/mf
cd ~/mf
```

<aside>
💡You’ll reference this directory later as MF_PROJECT_DIR. Adjust paths accordingly if you use a different location.

</aside>

- Clone the repository and set up the environment

```bash
git clone https://github.com/Datus-ai/metricflow.git
cd metricflow
poetry lock
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
```

- Verify MetricFlow installation

```bash
mf setup
mf tutorial
mf validate-configs
```

- Set up MetricFlow for Datus Demo

Edit the `~/.metricflow/config.yml` file

```bash
model_path: ~/mf/metricflow/semantic_models   # Path to directory containing defined models (Leave until after DWH setup)
email: ''  # Optional
dwh_schema: demo
dwh_dialect: duckdb
dwh_database: ~/.datus/demo/demo.duckdb  # For DuckDB, this is the data file.
```

---

<aside>
💡Return to the Datus terminal, which uses Python version 3.12

</aside>

- Install the filesystem MCP server

```bash
npm install -g @modelcontextprotocol/server-filesystem
```

- Configure Datus for metrics integration.

Add the following section to `~/.datus/conf/agent.yml`

```bash
  metrics:
    demo:
      subject_path: economic/bank/bank_failures
```

- Set up environment variables

```bash
# MetricFlow MCP server path + CLI
export METRICFLOW_MCP_DIR=~/mf/metricflow
export MF_PATH=~/mf//metricflow/.venv/bin/mf
export MF_PROJECT_DIR=~/mf/metricflow
export MF_VERBOSE=true

# Filesystem MCP server root for semantic models
mkdir -p ~/mf/metricflow/semantic_models
export FILESYSTEM_MCP_DIRECTORY=~/mf/metricflow/semantic_models
```

- Start the Datus CLI and generate metrics

```bash
datus-cli --namespace local_duckdb
```

- Ask Datus a natural language question, and it will automatically generate the appropriate SQL query for you.

```bash
Datus> /which state  has the highest total asset value of failure bank?

#run the sql generated
Datus>SELECT 
    State,
    SUM("Assets ($mil.)") as Total_Assets_Millions,
    COUNT(*) as Number_of_Failures
FROM demo.main.bank_failures 
GROUP BY State 
ORDER BY Total_Assets_Millions DESC 
LIMIT 1
```

- Generate metrics based on your question

```bash
Datus> !gen_metrics
```

- View the generated metric definitions

Navigate to the directory specified in `FILESYSTEM_MCP_DIRECTORY`

```bash
cd ~/mf/metricflow/semantic_models

less bank_failures.yaml 
```

```yaml
data_source:
  name: bank_failures
  description: Bank failures data with state and asset information

  sql_table: demo.main.bank_failures

  measures:
    - name: total_assets_millions
      agg: SUM
      expr: "Assets ($mil.)"
      create_metric: true

    - name: number_of_failures
      agg: COUNT
      expr: "1"
      create_metric: true

  dimensions:
    - name: state
      type: CATEGORICAL
      expr: State

  identifiers:
    - name: bank_failure
      type: PRIMARY
      expr: "CONCAT(State, '-', "Bank Name")"

  mutability:
    type: APPEND_ONLY

---
metric:
  name: state_failure_count_highest_assets
  description: Number of failures in the state with highest total assets
  type: measure_proxy
  type_params:
    measure: number_of_failures
  constraint: "{{ Dimension('state__state') }} = (
    SELECT State 
    FROM demo.main.bank_failures 
    GROUP BY State 
    ORDER BY SUM("Assets ($mil.)") DESC
                LIMIT 1
  )"
  locked_metadata:
    display_name: "Failure Count in Highest Asset State"
    value_format: ",.0f"
    unit: "failures"
    tags:
      - "Banking"
      - "Risk Analysis"
```

For more details about metrics: [Metrics](./metrics/metrics.md)

### **🔌 4. Datus MCP Server (Claude Desktop Integration)**

Expose Datus's database and context search tools via the **Model Context Protocol (MCP)**, enabling integration with
Claude Desktop, Claude Code, and other MCP-compatible clients.

**Supported Transport Modes:**

- `http`: Streamable HTTP (bidirectional, default)
- `sse`: Server-Sent Events over HTTP (for web clients)
- `stdio`: Standard input/output (for Claude Desktop and CLI tools)

#### Quick Start

- Install Datus:

```bash
pip install datus-agent
```

- Run the MCP server:

```bash
# Run with uvx (recommended)
uvx --from datus-agent datus-mcp --namespace local_duckdb

# Or run with python directly
python -m datus.mcp_server --namespace local_duckdb
```

#### Claude Desktop Configuration

Add the following to your `claude_desktop_config.json`:

- **Using uvx (recommended):**

```json
{
  "mcpServers": {
    "datus": {
      "command": "uvx",
      "args": [
        "--from",
        "datus-agent",
        "datus-mcp",
        "--namespace",
        "local_duckdb",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

- **Using python directly:**

```json
{
  "mcpServers": {
    "datus": {
      "command": "python",
      "args": [
        "-m",
        "datus.mcp_server",
        "--namespace",
        "local_duckdb",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

<aside>
💡 The config file location:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
</aside>

#### HTTP Server Mode

For web clients or multi-client scenarios, run the server in HTTP mode:

```bash
# Streamable HTTP (default, bidirectional)
datus-mcp --namespace local_duckdb --host 0.0.0.0 --port 8000

# SSE mode (for web clients)
datus-mcp --namespace local_duckdb --transport sse --port 8000
```

Connect to:

- Streamable HTTP: `http://localhost:8000/mcp`
- SSE: `http://localhost:8000/sse`

#### Available Tools

The MCP server exposes the following tools:

| Category           | Tools                                                                                                                                                             |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Database**       | `list_databases`, `list_schemas`, `list_tables`, `search_table`, `describe_table`, `get_table_ddl`, `read_query`                                                  |
| **Context Search** | `list_subject_tree`, `search_metrics`, `get_metrics`, `search_reference_sql`, `get_reference_sql`, `search_semantic_objects`, `search_knowledge`, `get_knowledge` |

#### Command Line Options

```bash
datus-mcp --help

Options:
  --namespace, -n    Database namespace to use (required)
  --sub-agent, -s    Sub-agent name for scoped context
  --database, -d     Database name override
  --config           Path to agent configuration file
  --transport, -t    Transport type: http (default), sse, stdio
  --host             Host to bind for HTTP transports (default: 0.0.0.0)
  --port, -p         Port to bind for HTTP transports (default: 8000)
  --debug            Enable debug logging
```

