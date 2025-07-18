## Integration testing

# Generate spider2 tests.
```shell
python gen_benchmark.py --namespace snowflake --benchmark spider2 --workdir=${path to datus agent}
```
# Generate bird tests.
```shell
python gen_benchmark.py --namespace bird_sqlite --benchmark bird_dev --workdir=${path to datus agent} --extra_option '--plan fixed --schema_linking_rate medium'
```

# Generate semantic layer tests.

## generate semantic models and metrics
```shell
cd ${path to datus project dir} && python -m datus.main bootstrap-kb --namespace duckdb --components metrics --success_story benchmark/semantic_layer/success_story.csv --domain sale --layer1 duckdb --layer2 duck --kb_update_strategy overwrite
```
## generate semantic layer test commands
```shell
python gen_benchmark.py --namespace duckdb --benchmark semantic_layer --workdir=${path to datus agent} --extra_option '--plan metric2SQL --task_db_name duck --task_schema mf_demo --domain sale --layer1 duckdb --layer2 duck --task_ext_knowledge "${external knowledge}"'
```

# Generate bird gold sql result.
```shell
python gen_exec_result.py --namespace bird_sqlite --benchmark bird_dev --type bird --workdir=${path to datus agent}
```

# Generate semantic layer gold sql result.
```shell
python gen_exec_result.py --namespace duckdb --benchmark semantic_layer --type semantic_layer --workdir=${path to datus agent}
```

# Run all tests

```shell
sh run_integration.sh
```

# Run the tests concurrently with 3 threads.
```shell
cat run_integration.sh | xargs -I {} -P 3 bash -c "{}"
```
# Evaluate spider2 tests

```shell
python evaluation.py --namespace snowflake --gold-path=${path to gold} --workdir=${path to datus agent}
```
# Evaluate bird tests

```shell
python evaluation.py --namespace bird_sqlite --gold-path=benchmark/bird/dev_20240627/gold --workdir=${path to datus agent}
```

# Evaluate semantic layer tests

```shell
python evaluation.py --namespace duckdb --gold-path=benchmark/semantic_layer/gold --workdir=${path to datus agent} --enable-comparison
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
