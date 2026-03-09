## Integration Testing

This document outlines testing workflows and utilities for benchmarks.

## Automated Benchmark Testing

The Bird, Spider2, and Semantic Layer benchmarks now have fully automated testing integrated into the main benchmark
functions. Simply run:

### Bird Benchmark

```shell
# Basic usage
python -m datus.main benchmark --namespace bird_sqlite --benchmark bird_dev

# With specific options
python -m datus.main benchmark --namespace bird_sqlite --benchmark bird_dev --workflow fixed --schema_linking_rate medium --benchmark_task_ids <id1> <id2>...

# With parallel execution (3 worker threads)
python -m datus.main benchmark --namespace bird_sqlite --benchmark bird_dev --max_workers 3
```

### Spider2 Benchmark

```shell
# Basic usage
python -m datus.main benchmark --namespace snowflake --benchmark spider2

# With specific task IDs
python -m datus.main benchmark --namespace snowflake --benchmark spider2 --benchmark_task_ids <id1> <id2>...

# With parallel execution (2 worker threads)
python -m datus.main benchmark --namespace snowflake --benchmark spider2 --max_workers 2
```

### Semantic Layer Benchmark

```shell
# Basic usage
python -m datus.main benchmark --namespace duckdb --benchmark semantic_layer

# With custom testing set file
python -m datus.main benchmark --namespace duckdb --benchmark semantic_layer --testing_set /path/to/custom_test.csv

# With specific task IDs(line numbers)
python -m datus.main benchmark --namespace duckdb --benchmark semantic_layer --benchmark_task_ids <id1>  <id2>...
```

These commands will automatically:

1. Generate gold standard results (Bird and Semantic Layer)
2. Run benchmark tests (Bird/Spider2 support parallel execution)
3. Evaluate accuracy and generate reports

### Parallel Execution Options

The `--max_workers` parameter controls the number of concurrent threads for **Bird and Spider2** benchmarks:

- **Default**: `--max_workers 1` (single-threaded, safest)
- **Recommended**: `--max_workers 2-3` for most systems
- **Note**: Higher concurrency may cause API rate limits or resource contention
- **Semantic Layer**: Does not support parallel execution (runs sequentially)

## Multi-agent testing

Create a folder named `multi` under the `conf` directory and prepare `agent{i}.yml` files. For example:

```
conf/multi/agent1.yml
conf/multi/agent2.yml
conf/multi/agent3.yml
```

# Generate multi bird tests

```shell
python gen_multi_benchmark.py --namespace bird_sqlite --benchmark bird_dev --workdir=${path to datus agent} --agent_num=3 --task_limit=100
```

# Run the tests concurrently with 3 threads for each agent

```shell
cat run_integration_agent{i}.sh | parallel -j 3
```

If using the Claude model, you need to reduce the parallelism, or set the parallelism to only 1.

# Select the best answer

```shell
python select_answer.py --workdir=${path to datus agent} --namespace bird_sqlite --agent=3
```

# Evaluate the agent1 answer

```shell
python evaluation.py --gold-path=benchmark/bird/dev_20240627/gold --namespace bird_sqlite --workdir=${path to datus agent} --save-dir multi/agent1_save --result-dir multi/agent1_output --enable-comparison
```

# Evaluate the best answer

```shell
python evaluation.py --gold-path=benchmark/bird/dev_20240627/gold --namespace bird_sqlite --workdir=${path to datus agent} --save-dir multi/best_agent_save --result-dir multi/best_agent_output --enable-comparison
```
