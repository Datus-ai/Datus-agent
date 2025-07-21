## Integration Testing

This document outlines the complete testing workflows for three different benchmarks: Bird, Spider2, and DuckDB Semantic Layer.

## Bird Testing Workflow

### 1. Generate Bird Tests
```shell
python gen_benchmark.py --namespace bird_sqlite --benchmark bird_dev --workdir=${path to datus agent} --extra_option '--plan fixed --schema_linking_rate medium'
```

### 2. Generate Bird Gold SQL Results
```shell
python gen_exec_result.py --namespace bird_sqlite --benchmark bird_dev --type bird --workdir=${path to datus agent}
```

### 3. Run Bird Tests
```shell
sh run_integration.sh
```

### 4. Evaluate Bird Tests
```shell
python evaluation.py --namespace bird_sqlite --gold-path=benchmark/bird/dev_20240627/gold --workdir=${path to datus agent}
```

## Spider2 Testing Workflow

### 1. Generate Spider2 Tests
```shell
python gen_benchmark.py --namespace snowflake --benchmark spider2 --workdir=${path to datus agent}
```

### 2. Run Spider2 Tests
```shell
sh run_integration.sh
```

### 3. Evaluate Spider2 Tests
```shell
python evaluation.py --namespace snowflake --gold-path=${path to gold} --workdir=${path to datus agent}
```

## DuckDB Semantic Layer Testing Workflow

### 1. Generate Semantic Models and Metrics
```shell
cd ${path to datus project dir} && python -m datus.main bootstrap-kb --namespace duckdb --components metrics --success_story benchmark/semantic_layer/success_story.csv --domain sale --layer1 duckdb --layer2 duck --kb_update_strategy overwrite
```

### 2. Generate Semantic Layer Test Commands
```shell
python gen_benchmark.py --namespace duckdb --benchmark semantic_layer --workdir=${path to datus agent} --extra_option '--plan metric2SQL --task_db_name duck --task_schema mf_demo --domain sale --layer1 duckdb --layer2 duck --task_ext_knowledge "${external knowledge}"'
```

### 3. Generate Semantic Layer Gold SQL Results
```shell
python gen_exec_result.py --namespace duckdb --benchmark semantic_layer --type semantic_layer --workdir=${path to datus agent}
```

### 4. Run Semantic Layer Tests
```shell
sh run_integration.sh
```

### 5. Evaluate Semantic Layer Tests
```shell
python evaluation.py --namespace duckdb --gold-path=benchmark/semantic_layer/gold --workdir=${path to datus agent} --enable-comparison
```

## General Testing Options

### Run Tests Concurrently
```shell
cat run_integration.sh | xargs -I {} -P 3 bash -c "{}"
```

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
