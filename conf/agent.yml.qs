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

  namespace:
    local_duckdb:
      type: duckdb
      name: duckdb-demo
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
