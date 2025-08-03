
# Datus-Agent User Manual

## Installation

```bash
# You can also use venv or other package management tools; here a Python 3.12 environment is required.
conda create -n datus-agent python=3.12.8

conda activate datus-agent

pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ datus-agent==0.1.1
```

## Configuring the LLM and Database

### agent.yml

```bash
cp ~/.datus/conf/agent.yml.build ~/.datus/conf/agent.yml
```

Edit `~/.datus/conf/agent.yml`:

By default, it uses deepseek v3, with the storage directory at `~/.datus/data`.

However, you need to manually `export` environment variables or add them to `.bashrc` and re-`source` it, as `.env` is not currently supported.

```yaml
agent:
  target: deepseek-v3
  models:
    deepseek-v3:
      type: deepseek
      vendor: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-chat

    deepseek-r1:
      type: deepseek
      vendor: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-reasoner

  # RAG storage base path; final data path examples: 'data/datus_db_spider2', 'data/datus_db_bird_dev', 'data/datus_db_local1', etc.
  storage_path: ~/.datus/data

  # Benchmark configuration
  benchmark:
    bird_dev:
      benchmark_path: benchmark/bird/dev_20240627
    spider2:
      benchmark_path: benchmark/spider2/spider2-snow

  namespace:
    spidersnow:
      type: snowflake
      username: ${SNOWFLAKE_USER}
      account: ${SNOWFLAKE_ACCOUNT}
      warehouse: ${SNOWFLAKE_WAREHOUSE}
      password: ${SNOWFLAKE_PASSWORD}
    bird_sqlite:
      type: sqlite
      path_pattern: benchmark/bird/dev_20240627/dev_databases/**/*.sqlite
    local_duckdb:
      type: duckdb
      uri: ~/.datus/sample/duckdb-demo.duckdb

  nodes:
    schema_linking:
      model: deepseek-v3
      matching_rate: fast
    generate_sql:
      model: deepseek-v3
      prompt_version: "1.0"
    reasoning:
      model: deepseek-v3
    reflect:
      prompt_version: "2.1"
```

- **Orange** parts require setting environment variables.
- **Green** parts are auto-generated during installation.
- **Yellow** parts require downloading for benchmarks.

## Executables

After installation, two executables are available: `datus-agent` and `datus-cli`.

You can run:

```bash
datus-agent --help
```

Sample output:

```
usage: datus-agent [-h] [--debug] [--config CONFIG] {probe-llm,check-db,bootstrap-kb,benchmark,run} ...

Datus: AI-powered SQL Agent for data engineering

positional arguments:
  {probe-llm,check-db,bootstrap-kb,benchmark,run}
                        Action to perform
    probe-llm           Test LLM connectivity
    check-db            Check database connectivity
    bootstrap-kb        Initialize knowledge base
    benchmark           Run benchmarks
    run                 Run SQL agent

options:
  -h, --help            Show this help message and exit
  --debug               Enable debug level logging
  --config CONFIG       Path to configuration file (default: conf/agent.yml)
```

## Connectivity Test

If you've configured `DEEPSEEK_API_KEY`:

```bash
datus-agent probe-llm
```

Sample output:
```
2025-05-31 07:44:06 [info] Storage modules initialized: [] [sql_agent]
2025-05-31 07:44:06 [info] Testing LLM model connectivity [sql_agent]
2025-05-31 07:44:06 [info] Using model type: deepseek, model name: deepseek-chat [sql_agent]
HTTP Request: POST https://api.deepseek.com/chat/completions "HTTP/1.1 200 OK"
2025-05-31 07:44:11 [info] LLM model test successful [sql_agent]
2025-05-31 07:44:11 [info] Final Result: {'status': 'success', 'message': 'LLM model test successful', 'response': 'Yes, I can "hear" you! Well, technically, I’m reading your message since I don’t have audio capabilities, but I’m here and ready to help. What’s on your mind? 😊'} [main]
```

## Prompt Template Directory

You can modify Jinja templates as needed to adjust prompts to your business and model. Combined with workflow and node configurations, this allows flexible customization.

```bash
ls ~/.datus/template
```

Example templates: `evaluation_1.0.j2`, `gen_sql_system_1.0.j2`, etc.

## Local Testing Data

A 5MB DuckDB file is uploaded; you can try it with:

```bash
datus-cli --namespace local_duckdb
```

## Benchmark

### Bird

Download the Bird dataset:

```bash
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip

mkdir -p benchmark/bird
mv dev_20240627 benchmark/bird
cd benchmark/bird/dev_20240627
unzip dev_databases
cd ../../..
```

Edit `agent.yml`:

```yaml
benchmark:
  bird_dev:
    benchmark_path: benchmark/bird/dev_20240627

namespace:
  bird_sqlite:
    type: sqlite
    path_pattern: benchmark/bird/dev_20240627/dev_databases/**/*.sqlite
```

If in the current directory, run:

```bash
datus-agent bootstrap-kb --namespace bird_sqlite --benchmark bird_dev --kb_update_strategy overwrite
```

This builds a LanceDB vector database at `~/.datus/data/datus_db_bird_sqlite`.

Run the benchmark:

```bash
datus-agent benchmark --namespace bird_sqlite --benchmark bird_dev --plan fixed --schema_linking_rate medium --benchmark_task_ids 1 2
```

### Spider

Like Bird, download the dataset:

```bash
mkdir benchmark
git clone https://github.com/xlang-ai/Spider2.git
mv Spider2 benchmark/spider2
```

Edit `agent.yml`:

```yaml
benchmark:
  spider2:
    benchmark_path: benchmark/spider2/spider2-snow

namespace:
  spidersnow:
    type: snowflake
    username: ${SNOWFLAKE_USER}
    account: ${SNOWFLAKE_ACCOUNT}
    warehouse: ${SNOWFLAKE_WAREHOUSE}
    password: ${SNOWFLAKE_PASSWORD}
```

## Exploring Datus-cli with StarRocks

Initialize the knowledge base (`bootstrap-kb`).

Add a StarRocks namespace in `~/.datus/conf/agent.yml`:

```yaml
sr:
  type: starrocks
  username: ${STARROCKS_USER}
  password: ${STARROCKS_PASSWORD}
  host: ${STARROCKS_HOST}
  port: ${STARROCKS_PORT}
  database: ${STARROCKS_DATABASE}
```

Check connectivity:

```bash
datus-agent check-db --namespace sr
```

Initialize vector DB (builds vector storage of schema and sample values):

```bash
datus-agent bootstrap-kb --namespace sr --kb_update_strategy overwrite
```

Run NL2SQL:

```bash
datus-agent run --namespace sr --task_db_name ssb_1 --task "how many parts are there?"
```

## Exploring Datus-cli

```bash
datus-cli --namespace sr
```

