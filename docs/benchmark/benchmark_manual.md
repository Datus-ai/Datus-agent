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

However, you need to manually `export` environment variables or add them to `.bashrc` and re-`source` it, as `.env` is
not currently supported.

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
usage: datus-agent [-h] [--debug] [--config CONFIG] {probe-llm,check-db,bootstrap-kb,platform-doc,benchmark,run} ...

Datus: AI-powered SQL Agent for data engineering

positional arguments:
  {probe-llm,check-db,bootstrap-kb,platform-doc,benchmark,run}
                        Action to perform
    probe-llm           Test LLM connectivity
    check-db            Check database connectivity
    bootstrap-kb        Initialize knowledge base
    platform-doc        Initialize platform documentation
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

You can modify Jinja templates as needed to adjust prompts to your business and model. Combined with workflow and node
configurations, this allows flexible customization.

Templates are stored in `{agent.home}/template/` (default: `~/.datus/template/`). Configure `agent.home` in `agent.yml`
to use a different root directory.

```bash
ls ~/.datus/template
# Or if you configured a custom home:
# ls /your/custom/home/template
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

mkdir -p ~/.datus/benchmark/bird
mv dev_20240627 ~/.datus/benchmark/bird
cd ~/.datus/benchmark/bird/dev_20240627
unzip dev_databases
```

Edit `agent.yml`:

```yaml

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
datus-agent benchmark --namespace bird_sqlite --benchmark bird_dev --workflow fixed --schema_linking_rate medium --benchmark_task_ids 1 2
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

### Custom Benchmarks

You can register any benchmark by adding an entry under `agent.benchmark` in `agent.yml`. Each entry is mapped into a
`BenchmarkConfig` object; the core fields are:

- `question_file`: Relative path to the question manifest (supports `.json`, `.jsonl`, `.csv`, `.tsv`).
- `question_id_key`: Optional column/key that contains task IDs. When omitted the evaluator uses row order.
- `question_key`: Field containing the natural language question.
- `db_key`: Field that indicates which database to run against; falls back to the namespace default.
- `ext_knowledge_key`: Optional field that carries extra knowledge (e.g. documentation snippets).
- `use_tables_key`: Optional field that limits the tables exposed to the agent for that task.

Evaluation relies on the gold SQL/result configuration. Depending on the source format you can choose either a directory
layout (one file per task) or a single manifest.

- `gold_sql_path`: Relative path to either a directory (`{gold_sql_path}/{task_id}.sql`) or a single JSON/CSV file.
- `gold_sql_key`: Required when `gold_sql_path` points to a JSON/CSV/JSONL file; names the column that stores SQL.
- `gold_result_path`: Optional path to pre-computed results (directory or single JSON/CSV/JSONL file). If omitted the
  evaluator executes `gold_sql` against the configured databases.
- `gold_result_key`: Required when `gold_result_path` is a JSON/CSV/JSONL file and stores the expected result payload.

Example configuration:

```yaml
agent:
  benchmark:
    sales_demo:
      question_file: tasks.jsonl
      question_id_key: task_id
      question_key: prompt
      db_key: db_id
      ext_knowledge_key: knowledge
      use_tables_key: allowed_tables
      gold_sql_path: gold/sql
      gold_result_path: gold/results.jsonl
      gold_sql_key: reference_sql
      gold_result_key: expected_answer
```

With this configuration in place you can run and evaluate the benchmark exactly like the built-in suites:

```bash
datus-agent benchmark --namespace my_namespace --benchmark sales_demo
datus-agent eval --namespace my_namespace --benchmark sales_demo
```

## Evaluating Results

After running a benchmark (whether partially via `--benchmark_task_ids` or the entire suite), use the new `eval`
action to compare your generated outputs against the gold answers and generate a report.

Typical usage:

```bash
# Evaluate the previously run Bird tasks 1 and 2 and write a JSON report
datus-agent eval \
  --namespace bird_sqlite \
  --benchmark bird_dev \
  --task_ids 1 2 \
  --output_file bird_eval_report.json
```

Key flags:

- `--task_ids`: Restrict evaluation to specific benchmark tasks; omit to process the full benchmark.

For custom benchmarks, You can at the gold result file containing the columns `task_id`, `question`, `gold_sql`,
`expected_answer`, `answer_rows`,
`expected_file`, `expected_table`, `expected_sql`, `expected_semantic_model`, `expected_metrics`, and
`expected_knowledge`.

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

## Multi-round Benchmark And Evaluation

Use `datus/multi_round_benchmark.py` to execute repeated benchmark + evaluation cycles and compare the
outcomes.

```shell
python -m datus.multi_round_benchmark \
  --namespace bird_sqlite \
  --benchmark bird_dev \
  --workflow chat_agentic \
  --max_round 4 \
  --max_workers 2
```

Key options:

- `--workflow`: workflow/plan name passed directly to the agent.
- `--max_round`: number of benchmark/evaluation loops (default 4).
- `--task_ids`: explicit task id list (space or comma separated).
- `--max_workers`: same as the regular benchmark flag to control concurrency.

For each round the script rewrites the agent's output directories to `{agent.home}/integration/{workflow}_{round}/` so
the following artifacts stay isolated:

- `save/` – namespace CSV answers.
- `trajectory/` – YAML workflow traces.
- `evaluation_round_{round}.json` – raw evaluation report.

After all rounds finish, the tool aggregates every task's status into
`{agent.home}/integration/{workflow}_summary.xlsx` with the schema:

| task_id                  | round_0        | round_1         | ... | Matching Rate |
|--------------------------|----------------|-----------------|-----|---------------|
| 0                        | Matched        | Result Mismatch | ... | 50%           |
| 1                        | Colum Mismatch | Matched         | ... | 50%           |
| Summary of Matching Rate | 50%            | 50%             | ... | 50%           |

Statuses currently map to(Priority starts from front to back):

1. `Not Executed`: Benchmark or assessment encounters an anomaly.
2. `Matched`: Successful matches with standard answers
3. `Gen SQL Failed`: The SQL generated by the agent fails to execute in the database
4. `Gold SQL Failed`: Gold SQL execution in the database has failed. Check whether the configuration is incorrect or the
   Gold SQL query is faulty.
5. `Match Failed`: The evaluation result was not obtained, possibly due to an uncaught exception.
6. `Table Mismatch`: Indicating that the table located by the Agent is inaccurate (incorrect or missing).
7. `Result Mismatch`: Table matching successful, but row or column matching failed.
8. `Column Mismatch`: Tables, rows, and certain columns matched successfully
