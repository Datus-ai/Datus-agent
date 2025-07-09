# Use Python 3.12 as base image from Aliyun mirror
FROM m.daocloud.io/docker.io/library/python:3.12-slim

# Set working directory
WORKDIR /app

# Install git and other necessary packages
RUN apt-get update && \
  apt-get install -y build-essential && \
  apt-get install -y git wget unzip libsqlite3-dev default-libmysqlclient-dev pkg-config &&\
  rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com
# Install uv
RUN pip install uv

RUN pip install -U huggingface_hub
ENV HF_ENDPOINT=https://hf-mirror.com
RUN huggingface-cli download intfloat/multilingual-e5-large-instruct

# # Configure git to use token authentication
# ARG GITHUB_TOKEN
# RUN git config --global url."https://${GITHUB_TOKEN}:x-oauth-basic@github.com/".insteadOf "https://github.com/"

# # Clone the repository with depth=1 to speed up
# RUN git clone --recursive https://github.com/Louis-Law/Datus-agent.git .

# Copy source code to the container (excluding tests directory)
COPY ./datus /app/datus
COPY ./benchmark /app/benchmark
COPY ./mcp /app/mcp
COPY ./pyproject.toml /app/pyproject.toml
COPY ./README.md /app/README.md
COPY ./LICENSE /app/LICENSE
COPY ./requirements.txt /app/requirements.txt
COPY ./conf/agent.yml.example /app/.datus/agent.example.yml


# Download and extract benchmark data
RUN mkdir -p /app/benchmark && \
  cd /app/benchmark && \
  wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip && \
  unzip dev.zip && \
  rm dev.zip && \
  cd dev_20240627 && \
  unzip dev_databases.zip && \
  rm dev_databases.zip

RUN cd /app
# Create .datus directory and agent.yml
RUN mkdir -p .datus && \
    echo 'agent:\n\
  target: deepseek\n\
  models:\n\
    deepseek:\n\
      type: deepseek\n\
      base_url: https://api.deepseek.com\n\
      api_key: ${DEEPSEEK_API_KEY}\n\
      model: deepseek-v3\n\
  nodes:\n\
    schema_linking:\n\
      model: qwen3\n\
      matching_rate: fast\n\
      prompt_version: "1.0"\n\
    generate_sql:\n\
      model: qwen3\n\
      prompt_version: "1.0"\n\
      max_table_schemas_length: 10000\n\
      max_data_details_length: 8000\n\
      max_context_length: 8000\n\
      max_value_length: 500\n\
    reasoning:\n\
      model: qwen3\n\
      prompt_version: "1.0"\n\
      max_table_schemas_length: 10000\n\
      max_data_details_length: 8000\n\
      max_context_length: 8000\n\
      max_value_length: 500\n\
    reflect:\n\
      prompt_version: "1.0"\n\
    output:\n\
      prompt_version: "1.0"\n\
      check_result: true\n\
  schema_linking_rate: medium\n\
  benchmark:\n\
    bird_dev: # this is namespace of benchmark\n\
      benchmark_path: /app/benchmark/dev_20240627\n\
    spider2:\n\
      benchmark_path: /app/benchmark/spider2/spider2-snow\n\
  # local databases configuration\n\
  namespace:\n\
    snowflake:\n\
      type: snowflake\n\
      account:  ${SNOWFLAKE_ACCOUNT}\n\
      username: ${SNOWFLAKE_USERNAME}\n\
      password: ${SNOWFLAKE_PASSWORD}\n\
    bird_dev:\n\
      type: sqlite\n\
      path_pattern: /app/benchmark/dev_20240627/dev_databases/**/*.sqlite # just support glob pattern\n\
  storage:\n\
    base_path: data\n\
    database:\n\
      registry_name: sentence-transformers\n\
      model_name: intfloat/multilingual-e5-large-instruct\n\
      dim_size: 1024\n\
    document:\n\
      model_name: intfloat/multilingual-e5-small\n\
      dim_size: 384\n\
    metric:\n\
      model_name: intfloat/multilingual-e5-small\n\
      dim_size: 384' > .datus/agent.yml

RUN apt-get install --reinstall gcc
# Run uv sync
RUN uv venv -p 3.12 && uv sync

RUN uv run datus/main.py bootstrap-kb --config .datus/agent.yml --namespace bird_dev --benchmark bird_dev --kb_update_strategy overwrite --pool_size 6

RUN uv run datus/main.py bootstrap-kb --config .datus/agent.yml --namespace snowflake --benchmark spider2 --kb_update_strategy overwrite --pool_size 8

# Set the working directory
WORKDIR /app

# Keep the container running
CMD ["tail", "-f", "/dev/null"]


