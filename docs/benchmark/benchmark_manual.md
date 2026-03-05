# Benchmark

Evaluate Datus Agent's performance and capabilities using industry-standard benchmarks. Run comprehensive tests against datasets like BIRD and Spider 2.0-Snow to assess accuracy, execution success rate, and query generation quality.

## Overview

Datus Agent benchmark mode enables you to:

- **Measure Accuracy**: Evaluate how well the agent generates correct SQL from natural language
- **Track Success Rates**: Monitor query execution success across different database types
- **Compare Results**: Validate generated queries against expected outputs
- **Identify Improvements**: Discover areas for optimization and refinement

## Quick Start

Get started quickly with the built-in benchmark datasets.

### Step 1: Install Datus Agent

!!! tip
    See the [Quick Start Guide](../getting_started/Quickstart.md) for detailed installation and setup instructions.

```bash title="Terminal"
pip install datus-agent
datus-agent init
```

### Step 2: Configure Environment

Set the required environment variables for your benchmark:

=== "BIRD"
    ```bash title="Terminal"
    export DEEPSEEK_API_KEY=<your_api_key>
    ```

=== "Spider 2.0-Snow"
    ```bash title="Terminal"
    export DEEPSEEK_API_KEY=<your_api_key>
    export SNOWFLAKE_ACCOUNT=<your_snowflake_account>
    export SNOWFLAKE_USERNAME=<your_snowflake_username>
    export SNOWFLAKE_PASSWORD=<your_snowflake_password>
    ```

### Step 3: Download and Prepare the BIRD Dataset

!!! note
    This step is only required for the BIRD benchmark. If you are running Spider 2.0-Snow, skip to [Step 4](#step-4-run-benchmark-tests).

Download and extract the BIRD dev dataset into the Datus home directory (`~/.datus` by default):

```bash title="Terminal"
cd ~/.datus
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip
mkdir -p benchmark/bird
mv dev_20240627 benchmark/bird
cd benchmark/bird/dev_20240627
unzip dev_databases
cd ~
```

After extraction, the directory structure should look like:

```text
~/.datus/
└── benchmark/
    └── bird/
        └── dev_20240627/
            ├── dev_databases/
            │   ├── california_schools/
            │   │   └── california_schools.sqlite
            │   ├── ...
            │   └── <other_databases>/
            └── ...
```

Then bootstrap the knowledge base for the BIRD dataset:

```bash title="Terminal"
datus-agent bootstrap-kb --namespace bird_sqlite --benchmark bird_dev
```

### Step 4: Run Benchmark Tests

!!! warning
    Each task may take several minutes to complete. Running all tasks may require hours or days depending on your system configuration.

**BIRD Dataset**

!!! info
    Task ID range: 0-1533

=== "Run by Task ID"
    ```bash title="Terminal"
    datus-agent benchmark \
    --namespace bird_sqlite \
    --benchmark bird_dev \
    --benchmark_task_ids <task_id1> <task_id2>
    ```

=== "Run All Tasks"
    ```bash title="Terminal"
    datus-agent benchmark \
    --namespace bird_sqlite \
    --benchmark bird_dev
    ```

**Spider 2.0-Snow Dataset**

!!! info
    You can find the task ID (instance ID) in the [spider2-snow.jsonl](https://github.com/xlang-ai/Spider2/blob/main/spider2-snow/spider2-snow.jsonl) file.

!!! note
    Ensure you have configured the Snowflake environment variables before running Spider 2.0-Snow benchmarks.

=== "Run by Task ID"
    ```bash title="Terminal"
    datus-agent benchmark \
    --namespace snowflake \
    --benchmark spider2 \
    --benchmark_task_ids <task_id1> <task_id2>
    ```

=== "Run All Tasks"
    ```bash title="Terminal"
    datus-agent benchmark \
    --namespace snowflake \
    --benchmark spider2
    ```

### Step 5: Evaluate Results

#### Run Evaluation
=== "Evaluate by Task IDs"
```bash title="Run Evaluation"
datus-agent eval \
  --namespace snowflake \
  --benchmark spider2 \
  --output_file evaluation.json \
  --task_ids <task_id1> <task_id2>
```
=== "Evaluate All"
```bash title="Run Evaluation"
datus-agent eval \
  --namespace snowflake \
  --output_file evaluation.json \
  --benchmark spider2
```

#### Evaluation Results

=== "Summary"
```text title="Evaluation Result" hl_lines="8-10"
────────────────────────────────────────────────────────────────────────────────
 Datus Evaluation Summary (Total: 30 Queries)
────────────────────────────────────────────────────────────────────────────────
 ✅ Passed:                                 19    (63%)
 ⚠️  No SQL / Empty Result:                 0    (0%)
 ❌ Failed:                                 11    (37%)
     • Table Mismatch:                      4    (13%)
     • Table Matched (Result Mismatch):     7    (23%)
         - Row Count Mismatch:              1    (3%)
         - Column Value Mismatch:           6    (20%)
────────────────────────────────────────────────────────────────────────────────

 Passed Queries:                        12, 40, 63, 80, 209, 279, 340, 436, 738, 774, 778, 853, 864, 1013, 1069, 1177, 1329, 1439, 1488
 No SQL / Empty Result Queries:         None
 Failed (Table Mismatch):               263, 388, 484, 584
 Failed (Row Count Mismatch):           455
 Failed (Column Value Mismatch):        76, 131, 617, 642, 1011, 1205

────────────────────────────────────────────────────────────────────────────────
```

=== "Details"
```json
{
  "status": "success",
  "generated_time": "2025-11-07T15:56:40.050678",
  "summary": {
    "total_files": 1,
    "total_output_nodes": 1,
    "total_output_success": 1,
    "total_output_failure": 0,
    "success_rate": 100.0,
    "comparison_summary": {
      "total_comparisons": 1,
      "match_count": 1,
      "mismatch_count": 0,
      "comparison_error_count": 0,
      "empty_result_count": 0,
      "match_rate": 100.0
    }
  },
  "task_ids": {
    "failed_task_ids": "",
    "matched_task_ids": "0",
    "mismatched_task_ids": "",
    "empty_result_task_ids": ""
  },
  "details": {
    "0": {
      "total_node_count": 4,
      "output_node_count": 1,
      "output_success_count": 1,
      "output_failure_count": 0,
      "errors": [],
      "node_types": {
        "start": 1,
        "chat": 1,
        "execute_sql": 1,
        "output": 1
      },
      "tool_calls": {
        "list_tables": 1,
        "describe_table": 2,
        "search_files": 1,
        "read_query": 3
      },
      "completion_time": 1762173198.4857101,
      "status": "completed",
      "comparison_results": [
        {
          "task_id": "0",
          "actual_file_exists": true,
          "gold_file_exists": true,
          "actual_path": "<USER_HOME>/.datus/save/bird_school/0.csv",
          "gold_path": "<USER_HOME>/.datus/benchmark/california_schools/california_schools.csv",
          "comparison": {
            "match_rate": 1.0,
            "matched_columns": [
              [
                "eligible_free_rate",
                "`Free Meal Count (K-12)` / `Enrollment (K-12)`"
              ]
            ],
            "missing_columns": [],
            "extra_columns": [
              "School Name",
              "District Name",
              "Academic Year"
            ],
            "actual_shape": [1, 4],
            "expected_shape": [1, 1],
            "actual_preview": "School Name | District Name | eligible_free_rate | Academic Year\n       ----------------------------------------------------------------\n       Oakland Community Day Middle | Oakland Unified | 1.0 | 2014-2015",
            "expected_preview": "`Free Meal Count (K-12)` / `Enrollment (K-12)`\n       ----------------------------------------------\n       1.0",
            "actual_tables": ["frpm"],
            "expected_tables": ["frpm"],
            "matched_tables": ["frpm"],
            "actual_sql_error": null,
            "sql_error": null,
            "actual_sql": "SELECT \n    \"School Name\",\n    \"District Name\", \n    \"Percent (%) Eligible Free (K-12)\" as eligible_free_rate,\n    \"Academic Year\"\nFROM frpm \nWHERE \"County Name\" = 'Alameda'\n    AND \"Percent (%) Eligible Free (K-12)\" IS NOT NULL\nORDER BY \"Percent (%) Eligible Free (K-12)\" DESC\nLIMIT 1;",
            "gold_sql": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",
            "tools_comparison": {
              "expected_file": {"expected": "", "actual": [], "matched_actual": [], "match": true},
              "expected_sql": {"expected": "", "actual": [], "matched_actual": [], "match": true},
              "expected_semantic_model": {"expected": "", "actual": [], "matched_actual": [], "match": true},
              "expected_metrics": {
                "expected": ["Eligible free rate for K-12"],
                "actual": [],
                "matched_expected": [],
                "matched_actual": [],
                "missing_expected": ["Eligible free rate for K-12"],
                "match": false
              }
            },
            "error": null
          }
        }
      ]
    }
  }
}
```

**Key fields under `details`:**
- **comparison_results**: Comparison results for each task
    - `actual_sql`: SQL generated by Datus Agent
    - `actual_path`: Executed result produced by Datus Agent
    - `gold_sql`: Reference (gold) SQL
    - `gold_path`: Reference (gold) result path
    - `comparison`: Side-by-side comparison results
        - `match_rate`: Matching ratio
        - `matched_columns`: Columns that match
        - `missing_columns`: Columns missing compared to reference
        - `extra_columns`: Extra columns not present in the reference
        - `actual_tables`: Tables used by the generated result
        - `expected_tables`: Tables used by the reference answer
        - `matched_tables`: Intersected tables
- **tool_calls**: Count of each tool invocation

## Built-in Dataset Quick Trial

Try out benchmarking and evaluation with the pre-packaged California Schools dataset — no extra downloads needed.

### Step 1: Initialize the Dataset
```bash
datus-agent tutorial
```

![tutorial](../assets/tutorial.png)

This step does the following:

1. Add database configuration and benchmark configuration for California Schools to agent.yml
2. Initialize the metadata information of the tables for California Schools
3. Use `benchmark/california_schools/success_story.csv` to build metric information
4. Build reference SQL using the SQL files in `benchmark/california_schools/reference_sql`

### Benchmarking and evaluation
```bash
datus-agent benchmark --namespace california_schools --benchmark california_schools --benchmark_task_ids 0 1 2 --workflow <your workflow>
```
👉 See [Step 4: Run Benchmark Tests](#step-4-run-benchmark-tests)

```bash
datus-agent eval --namespace california_schools --benchmark california_schools --task_ids 0 1 2 
```
👉 See [Step 5: Evaluate Results](#step-5-evaluate-results)

## Custom Benchmark

### Add Benchmark Configuration

```yaml
agent:
  namespace:
    california_schools:
      type: sqlite
      name: california_schools
      uri: sqlite:///benchmark/bird/dev_20240627/dev_databases/california_schools/california_schools.sqlite # Database file path. Use sqlite:/// for relative paths; sqlite://// for absolute paths.
  benchmark:
    california_schools:              # Benchmark name
      question_file: california_schools.csv       # File containing benchmark questions
      question_id_key: task_id                    # Field name for question ID
      question_key: question                      # Field name for question text
      ext_knowledge_key: expected_knowledge       # Field name for external knowledge or additional context

      # Configuration for evaluation phase
      gold_sql_path: california_schools.csv       # File path for reference (gold) SQL
      gold_sql_key: gold_sql                      # Field name for reference SQL
      gold_result_path: california_schools.csv    # File path for reference (gold) results
      gold_result_key: ""                         # Field name for results when using a single file
```

---

📖 **Benchmark Configuration Field Description**

| Field                 | Description                                                                                                                                                                                                                                                                                                         |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **question_file**     | Path to the question file, supporting `.csv`, `.json`, and `.jsonl` formats. The path is relative to `{agent.home}/benchmark/{benchmark_name}`.                                                                                                                                                                     |
| **question_id_key**   | Unique identifier field name for each benchmark question.                                                                                                                                                                                                                                                           |
| **question_key**      | Field name for the natural language question.                                                                                                                                                                                                                                                                       |
| **ext_knowledge_key** | Field name for additional knowledge or problem description.                                                                                                                                                                                                                                                         |
| **gold_sql_path**     | Standard SQL file path, supporting two scenarios:<br/>1. A single file, e.g., BIRD_DEV; supporting `.csv`, `.json`, and `.jsonl` formats. <br/>2. A separate SQL file for each task (e.g., spider2).                                                                                                                |
| **gold_sql_key**      | When `gold_sql_path` is a single file, this specifies the field name containing the reference SQL.                                                                                                                                                                                                                  |
| **gold_result_path**  | Standard result file path, with each result being a string in CSV format. Three scenarios are supported:<br/>1. A single file in `.csv`, `.json`, or `.jsonl` format;<br/>2. A separate CSV file for each task (e.g., spider2);<br/>3. If not configured, the system will execute standard SQL to retrieve results. |
| **gold_result_key**   | Standard result field. When `gold_result_path` is a single file, this field is meaningful and must be configured.                                                                                                                                                                                                   |

---

### Build Knowledge Base

Construct the metadata, metrics, and reference SQL knowledge bases according to your dataset.  
👉 See [knowledge_base/introduction](../knowledge_base/introduction.md) for details.

### Run Benchmark

👉 See [Step 4: Run Benchmark Tests](#step-4-run-benchmark-tests)

### Evaluate Results

👉 See [Step 5: Evaluate Results](#step-5-evaluate-results)


## Multi-round Benchmark and Evaluation

Execute repeated benchmark + evaluation cycles (as described in [Step 4](#step-4-run-benchmark-tests) and [Step 5](#step-5-evaluate-results)) and compare the outcomes across rounds. This is useful for measuring the stability and consistency of agent performance.

### Usage

=== "datus-agent subcommand"
    ```bash title="Terminal"
    datus-agent multi-round-benchmark \
      --namespace bird_sqlite \
      --benchmark bird_dev \
      --workflow chat_agentic \
      --round 4 \
      --workers 2
    ```

=== "Standalone CLI"
    ```bash title="Terminal"
    datus-multi-benchmark \
      --namespace bird_sqlite \
      --benchmark bird_dev \
      --workflow chat_agentic \
      --round 4 \
      --workers 2
    ```

=== "Python module"
    ```bash title="Terminal"
    python -m datus.multi_round_benchmark \
      --namespace bird_sqlite \
      --benchmark bird_dev \
      --workflow chat_agentic \
      --round 4 \
      --workers 2
    ```

### Options

| Option                  | Required | Default                                       | Description                                                          |
|-------------------------|----------|-----------------------------------------------|----------------------------------------------------------------------|
| `--namespace`           | Yes      | —                                             | Namespace to benchmark, e.g. `bird_sqlite`                           |
| `--benchmark`           | Yes      | —                                             | Benchmark name, e.g. `bird_dev`                                      |
| `--workflow`            | No       | `reflection`                                  | Workflow plan to execute                                             |
| `--round`               | No       | `4`                                           | Number of benchmark iterations to run                                |
| `--max_steps`           | No       | `30`                                          | Maximum steps per workflow execution                                 |
| `--workers`             | No       | `1`                                           | Number of parallel workers for task execution                        |
| `--task_ids`            | No       | all tasks                                     | Explicit task IDs to benchmark (space or comma separated)            |
| `--group_name`          | No       | workflow name                                 | Name of the integration test group, used for output directory naming |
| `--delete_history`      | No       | `false`                                       | Delete existing round output directory before each round starts      |
| `--summary_report_file` | No       | —                                             | Path to summary report file. Reports will be appended for each round |
| `--debug`               | No       | `false`                                       | Enable debug level logging                                           |
| `--config`              | No       | `~/.datus/conf/agent.yml` or `conf/agent.yml` | Path to agent config file                                            |

> **Note:** `--round`, `--max_steps`, `--workers`, `--task_ids`, `--group_name`, `--delete_history`, and
> `--summary_report_file` also accept kebab-case (e.g. `--max-steps`) and legacy names
> (e.g. `--max_round`, `--max_workers`) for backward compatibility.

### Output Structure

`{timestamp}` is the time when the multi-round benchmark starts, formatted as `YYYYMMDD_HHmm` (e.g. `20260129_1530`). `{ts}` is a Unix epoch timestamp in seconds (e.g. `1769611836`).

For each round the tool creates an isolated output directory under `{agent.home}/integration/`. After all rounds finish, a summary Excel file is also exported:

```text
{agent.home}/integration/
├── {group_name}_0/
│   ├── save/{namespace}/{timestamp}/
│   │   ├── 0.json                                     # Task metadata
│   │   ├── 0.sql                                      # Generated SQL
│   │   ├── 0.csv                                      # Query execution result
│   │   ├── 1.json
│   │   ├── 1.sql
│   │   ├── 1.csv
│   │   └── ...
│   ├── trajectory/{namespace}/{timestamp}/
│   │   ├── 0_{ts}.yaml                                # Workflow trace (e.g. 0_1769611836.yaml)
│   │   ├── 1_{ts}.yaml
│   │   └── ...
│   └── evaluation_round_{timestamp}_0.json            # Evaluation report for round 0
├── {group_name}_1/
│   ├── save/{namespace}/{timestamp}/
│   │   ├── 0.json
│   │   ├── 0.sql
│   │   ├── 0.csv
│   │   ├── 1.json
│   │   ├── 1.sql
│   │   ├── 1.csv
│   │   └── ...
│   ├── trajectory/{namespace}/{timestamp}/
│   │   ├── 0_{ts}.yaml
│   │   ├── 1_{ts}.yaml
│   │   └── ...
│   └── evaluation_round_{timestamp}_1.json
├── ...
└── {group_name}_summary_{timestamp}.xlsx              # Summary report across all rounds
```

The summary Excel file contains a cross-round comparison table:

| task_id                  | round_0         | round_1         | ... | Matching Rate |
|--------------------------|-----------------|-----------------|-----|---------------|
| 0                        | Matched         | Result Mismatch | ... | 50.00%        |
| 1                        | Column Mismatch | Matched         | ... | 50.00%        |
| Summary of Matching Rate | 50.00%          | 50.00%          | ... | 50.00%        |
| Round Duration           | 5m 30s          | 5m 12s          | ... |               |

### Task Status Definitions

Each task in each round is classified into one of the following statuses, listed from highest to lowest evaluation priority (the first matching status is assigned):

| Status            | Description                                                                |
|-------------------|----------------------------------------------------------------------------|
| `Not Executed`    | Benchmark or evaluation encountered an anomaly; the task was not processed |
| `Matched`         | Generated SQL results match the gold standard answers                      |
| `Gen SQL Failed`  | The SQL generated by the agent fails to execute in the database            |
| `Gold SQL Failed` | Gold SQL execution failed — check configuration or gold SQL correctness    |
| `Match Failed`    | Evaluation result was not obtained, possibly due to an uncaught exception  |
| `Table Mismatch`  | The tables located by the agent are inaccurate (incorrect or missing)      |
| `Result Mismatch` | Tables matched, but row or column values differ from the expected output   |
| `Column Mismatch` | Tables and rows matched, but some columns differ or are missing            |