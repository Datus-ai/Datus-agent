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
  
  storage:
    base_path: ~/.datus/data #rag storage base path, meta data path is 'data/datus_db_{namespace}'

  # benchmark configuration
  benchmark:
    bird_dev: # this is namespace of benchmark
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
      name: demo
      uri: ~/.datus/duckdb-demo.duckdb

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
