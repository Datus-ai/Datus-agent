## Integration testing

# Generate spider2 tests.
```python
python gen_benchmark.py --namespace snowflake --benchmark spider2 --workdir=${path to datus agents}
```
# Generate bird tests.
```python
python gen_benchmark.py --namespace bird_sqlite --benchmark bird_dev --workdir=${path to datus agents} --extra_option '--plan fixed --schema_linking_rate medium'
```

# Generate bird gold sql result.
```python
python gen_exec_result.py --namespace bird_sqlite --benchmark bird_dev --type bird --workdir=${path to datus agents}
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

```python
python evaluation.py --namespace snowflake --gold-path=${path to gold} --workdir=${path to datus agents}
```
# Evaluate bird tests

```python
python evaluation.py --namespace bird_sqlite --gold-path=benchmark/bird/dev_20240627/gold --workdir=${path to datus agents}
```

## Multi-agent testing

Create a folder named `multi` under the `conf` directory and prepare `agent{i}.yml` files. For example:

```
conf/multi/agent1.yml  
conf/multi/agent2.yml  
conf/multi/agent3.yml
```

# Generate multi bird tests

```python
python gen_multi_benchmark.py --namespace bird_sqlite --benchmark bird_dev --workdir=${path to datus agents} --agent_num=3 --task_limit=100
```

# Run the tests concurrently with 3 threads for each agent
```shell
cat run_integration_agent{i}.sh | parallel -j 3
```
If using the Claude model, you need to reduce the parallelism, or set the parallelism to only 1.

# Select the best answer

```python
python select_answer.py --workdir=${path to datus agents} --namespace bird_sqlite --agent=3
```

# Evaluate the agent1 answer
```python
python evaluation.py --gold-path=benchmark/bird/dev_20240627/gold --namespace bird_sqlite --workdir=${path to datus agents} --save-dir multi/agent1_save --result-dir multi/agent1_output --enable-comparison
```

# Evaluate the best answer

```python
python evaluation.py --gold-path=benchmark/bird/dev_20240627/gold --namespace bird_sqlite --workdir=${path to datus agents} --save-dir multi/best_agent_save --result-dir multi/best_agent_output --enable-comparison
```
