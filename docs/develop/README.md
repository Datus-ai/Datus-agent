# Datus-agent

---

# Deployment

## Pulling submodule code locally

```bash
git submodule update --init
```

## Install the Necessary Dependencies

### Using `uv`

```bash
# Create a virtual environment
uv venv -p 3.12

# Synchronize all dependencies
uv sync
# with dev dependencies
uv sync --dev

# Use current environment
source .venv/bin/activate
```

### Using `conda`

```bash
conda create -n datus-agent python=3.12
conda activate datus-agent
pip install -r requirements.txt
```

### Using `venv`

```bash
virtualenv datus-agent --python=python3.12
source datus-agent/bin/activate
pip install -r requirements.txt
```

---

## Configuration

### Agent.yml

```bash
cp conf/agent.yml.qs conf/agent.yml
```

Then modify `conf/agent.yml` as needed:

```yaml
agent:
  target: deepseek-v3
  models:
    deepseek-v3:
      type: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-chat

    deepseek-r1:
      type: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-reasoner

  storage_path: data

  # Benchmark data is stored at {agent.home}/benchmark/{name}
  # Supported benchmarks: bird_dev, spider2, semantic_layer

  namespace: # namespace is a set of database connections
    local_duckdb:
      type: duckdb
      uri: ./sample_data/duckdb-demo.duckdb
    spider-snow:
      type: snowflake
      warehouse: ${SNOWFLAKE_WAREHOUSE}
      account: ${SNOWFLAKE_ACCOUNT}
      username: ${SNWOFLAKE_USER}
      password: ${SNOWFLAKE_PASSWORD}
    bird_sqlite:
      type: sqlite
      path_pattern: benchmark/bird/dev_20240627/dev_databases/**/*.sqlite

  storage:
    base_path: data
    # Local model recommendations:
    # 1. For extreme performance: all-MiniLM-L6-v2 (~100M) or intfloat/multilingual-e5-small (~460M)
    # 2. For balanced performance and quality: intfloat/multilingual-e5-large-instruct (~1.2G)
    # 3. For optimal retrieval quality: BAAI/bge-large-en-v1.5 or BAAI/bge-large-zh-v1.5 (~3.6G)
    # You can also select any model that suits your requirements.
    # Default: all-MiniLM-L6-v2 if no model is configured.
    # Claude model suggestions: Now we just support openai.
    database:
      registry_name: sentence-transformers # default is sentence-transformers, now just support sentence-transformers and openai.
      model_name: text-embedding-v3
      dim_size: 1024
      # batch_size: 10 # This configuration is required when the registration mode is openai
    document:
      model_name: intfloat/multilingual-e5-large-instruct
      dim_size: 1024
    metric:
      model_name: all-MiniLM-L6-v2
      dim_size: 384
```

You can configure multiple models and databases. The `target` is the default model to use.

---

### Langsmith (Optional)

Set the following environment variables for Langsmith integration:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=xxx
LANGSMITH_PROJECT=Datus-agent
```

---

# Have a Try

## Test Connection

```bash
python -m datus.main probe-llm
```

Example Output:

```
LLM model test successful
Final Result: {"status": "success", "message": "LLM model test successful", "response": "Yes, I can 'hear' you! 😊 How can I assist you today?"}
```

```bash
python -m datus.main check-db --namespace local_duckdb
```

---

## Run SQL

```bash
python -m datus.cli.main --namespace local_duckdb --config conf/agent.yml

Datus> select * from tree;
┏━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
┃ Continent ┃ TradingBloc ┃ Country    ┃ GDP    ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
│ NA        │ US          │ US         │ 19.485 │
│ Asia      │ China       │ China      │ 12.238 │
│ Asia      │ Japan       │ Japan      │ 4.872  │
│ Europe    │ EU          │ Germany    │ 3.693  │
│ Asia      │ India       │ India      │ 2.651  │
│ Europe    │ UK          │ UK         │ 2.638  │
│ Europe    │ EU          │ France     │ 2.583  │
│ SA        │ Brazil      │ Brazil     │ 2.054  │
│ Europe    │ EU          │ Italy      │ 1.944  │
│ NA        │ US          │ Canada     │ 1.647  │
│ Europe    │ Russia      │ Russia     │ 1.578  │
│ Asia      │ SouthKorea  │ SouthKorea │ 1.531  │
│ Australia │ Australia   │ Australia  │ 1.323  │
│ Europe    │ EU          │ Spain      │ 1.314  │
│ NA        │ US          │ Mexico     │ 1.151  │
└───────────┴─────────────┴────────────┴────────┘
Returned 15 rows in 0.03 seconds

Datus> .help
```

---

# Spider2 Benchmark

## Initialization

Update `agent.yml` if needed:

```yaml
# Benchmark data is stored at {agent.home}/benchmark/spider2/

namespace:
  spidersnow:
    type: snowflake
    username: ${SNOWFLAKE_USER}
    account: ${SNOWFLAKE_ACCOUNT}
    warehouse: ${SNOWFLAKE_WAREHOUSE}
    password: ${SNOWFLAKE_PASSWORD}
```

### Bootstrap Knowledge Base

```bash
python -m datus.main bootstrap-kb --namespace spidersnow --benchmark spider2 --kb_update_strategy overwrite
```

> ⚠️ May take hours (approx. 14,000 tables).

### Run Test by IDs

```bash
python -m datus.main benchmark --namespace spidersnow --benchmark spider2 --benchmark_task_ids sf_bq104
```

```bash
python -m datus.cli.main --namespace spidersnow  --config conf/agent.yml

Datus> !darun_screen
Creating a new SQL task
Enter task ID (49856268):
Enter task description (): Based on the most recent refresh date, identify the top-ranked rising search term for the week that is exactly one year prior to the latest available week in the dataset.
Enter database name: GOOGLE_TRENDS
Enter output directory (output):
Enter external knowledge (optional) ():
SQL Task created: 49856268
Database: snowflake - GOOGLE_TRENDS

```

---

# Bird Benchmark

## Initialization

Update Configuration:

```yaml
# Benchmark data is stored at {agent.home}/benchmark/bird/

namespace:
  bird_sqlite:
    type: sqlite
    path_pattern: { agent.home }/benchmark/bird/dev_databases/**/*.sqlite
```

### Download and Extract Bird Dev

```bash
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip
mkdir -p ~/.datus/benchmark/bird
mv dev_20240627 benchmark/bird
cd benchmark/bird/dev_20240627
unzip dev_databases
cd ../../..
```

### Bootstrap Knowledge Base

```bash
python -m datus.main bootstrap-kb --namespace bird_sqlite --benchmark bird_dev --kb_update_strategy overwrite
```

### Run Tests

```bash
python -m datus.main benchmark --namespace bird_sqlite --benchmark bird_dev --workflow fixed --schema_linking_rate medium --benchmark_task_ids 14 15
```

```bash
python -m datus.main benchmark --namespace bird_sqlite --benchmark bird_dev --schema_linking_rate fast --benchmark_task_ids 32
```

```bash
python -m datus.main benchmark --benchmark bird_dev --workflow fixed --schema_linking_model medium
```

### Using cli to develop

```bash
python -m datus.cli.main --namespace bird_sqlite  --config conf/agent.yml
```

# Semantic Layer Benchmark

## Initialization

Install Metricflow:

```bash
# poetry config virtualenvs.in-project true
poetry lock
poetry install
source .venv/bin/activate

# make sure these commands succeeded
mf setup
mf tutorial
mf validate-configs
```

Update Configuration `~/.metricflow/config.yml`:

```yml
model_path: </path/to/semantic-models-dir>
dwh_schema: mf_demo
dwh_dialect: duckdb
dwh_database: <home dir>/.metricflow/duck.db
```

Update Configuration conf/agent.yml:

```yaml
namespace:
  duckdb:
    type: duckdb
    name: duck
    uri: ~/.metricflow/duck.db

# Benchmark data is stored at {agent.home}/benchmark/semantic_layer/

metrics:
  tutorial:
    subject_path: sale/layer1/layer2
    ext_knowledge: ""
```

Export Environment Variables:

```bash
export MF_PATH=</path/to/metricflow>/.venv/bin/mf
export MF_VERBOSE=true
export MF_MODEL_PATH=</path/to/semantic-models-dir>
```

### Bootstrap Metrics Generation

```bash
python -m datus.main bootstrap-kb --namespace duckdb --components metrics --metric_meta tutorial --kb_update_strategy overwrite
```

### Run Tests

```bash
python -m datus.main benchmark --namespace duckdb --benchmark semantic_layer --workflow metric_to_sql --metric_meta tutorial
```