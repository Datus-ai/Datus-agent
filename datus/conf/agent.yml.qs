agent:
  target: ""
  models: {}

  storage:
    # Data path is now fixed at {agent.home}/data (e.g., ~/.datus/data/datus_db_{namespace})
    workspace_root: ~/.datus/workspace
    embedding_device_type: cpu
  benchmark:
    california_schools:
      question_file: california_schools.csv
      question_id_key: task_id
      question_key: question
      ext_knowledge_key: evidence
      gold_sql_path: california_schools.csv
      gold_sql_key: gold_sql
      gold_result_path: california_schools.csv

  namespace:
    local_duckdb:
      type: duckdb
      name: duckdb-demo
      uri: ~/.datus/sample/duckdb-demo.duckdb
    california_schools:
      type: sqlite
      name: california_schools
      uri: ~/.datus/benchmark/california_schools/california_schools.sqlite

  export:
      max_lines: 1000
  nodes:
    schema_linking:
      matching_rate: fast