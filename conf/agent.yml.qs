agent:
  target: deepseek-v3
  models:
    deepseek-v3:
      type: deepseek
      vendor: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-chat
    kimi-k2-turbo:
      type: openai
      vendor: openai
      base_url: https://api.moonshot.cn/v1
      api_key: ${KIMI_API_KEY}
      model: kimi-k2-turbo-preview

  storage:
    base_path: ~/.datus/data #rag storage base path, meta data path is 'data/datus_db_{namespace}'
    workspace_root: ~/.datus/workspace

  namespace:
    local_duckdb:
      type: duckdb
      name: duckdb-demo
      uri: ~/.datus/sample/duckdb-demo.duckdb

  nodes:
    schema_linking:
      matching_rate: fast
    generate_sql:
      prompt_version: "1.0"
    reasoning:
      prompt_version: "1.0"
    reflect:
      prompt_version: "2.1"
    chat:
      prompt_version: "1.0"