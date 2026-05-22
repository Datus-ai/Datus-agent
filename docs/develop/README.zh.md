# 开发者指南

本文面向从源码 checkout 运行 Datus Agent 的贡献者。如果你安装的是发布版，请从[快速开始](../getting_started/Quickstart.zh.md)开始。

## 源码环境

克隆仓库后初始化 submodule：

```bash
git submodule update --init
```

使用 Python 3.12。推荐用 `uv` 管理开发环境：

```bash
uv venv -p 3.12
uv sync --dev
source .venv/bin/activate
```

`uv sync --dev` 会安装运行时包、测试工具，以及开发时使用的 tracing 集成。

源码 checkout 暴露的 console entry point 与安装后的包一致：

```bash
uv run datus --version
uv run datus-agent --version
```

文档和脚本里优先使用这些入口：

| 命令 | 用途 |
| --- | --- |
| `datus` | 交互式 REPL 与 TUI |
| `datus-agent` | `probe-llm`、`check-db`、`bootstrap-kb`、`benchmark` 等批处理命令 |
| `datus-api` | REST API 服务 |
| `datus-mcp` | MCP 服务 |

`python -m datus.main` 和 `python -m datus.cli.main` 仍可用于底层调试，但不推荐作为面向用户的命令写法。

从干净源码 checkout 构建文档时，需要为 `mkdocs.yml` 提供所需插件：

```bash
uv run --with mkdocs-material --with mike --with mkdocs-static-i18n mkdocs build --strict
```

## 配置

配置文件查找顺序：

1. 显式传入的 `--config <path>`。
2. 当前工作目录下的 `./conf/agent.yml`。
3. `~/.datus/conf/agent.yml`。

本地源码开发时，如果还没有本地配置，可以从示例复制：

```bash
cp conf/agent.yml.example conf/agent.yml
```

不要把明文 API Key 提交到仓库，敏感信息应通过环境变量注入。

### 模型

当前模型配置采用 provider 方式。把凭证放在 `agent.providers` 下，然后在 REPL 里用 `/model` 为当前项目选择活跃 provider/model。选择结果会写入 `./.datus/config.yml`。

```yaml
agent:
  home: ~/.datus
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
    deepseek:
      api_key: ${DEEPSEEK_API_KEY}
    claude:
      api_key: ${ANTHROPIC_API_KEY}
    gemini:
      api_key: ${GEMINI_API_KEY}
```

只有自托管或私有模型端点不在 `conf/providers.yml` 覆盖范围内时，才使用 `agent.models`。

### 数据源

本地 smoke test 可以直接使用仓库自带的 DuckDB 示例库：

```yaml
agent:
  services:
    datasources:
      local_duckdb:
        type: duckdb
        uri: duckdb:///datus/sample_data/duckdb-demo.duckdb
```

验证连接并启动 REPL：

```bash
uv run datus-agent check-db --config conf/agent.yml --datasource local_duckdb
uv run datus --config conf/agent.yml --datasource local_duckdb
```

REPL 会自动识别 SQL：

```text
> show tables;
> select * from tree;
> /help
```

也可以在 REPL 中通过 `/datasource` 交互式配置数据源；默认会写入 `~/.datus/conf/agent.yml`。

### 存储布局

新配置不要再使用 `storage_path`。数据路径由 `agent.home` 推导：

| 路径 | 内容 |
| --- | --- |
| `{agent.home}/data/` | RDB/vector 存储后端 |
| `{agent.home}/sessions/` | 持久化 chat session |
| `{agent.home}/benchmark/` | 内置与自定义 benchmark 数据 |
| `{agent.home}/trajectory/` | Workflow checkpoint 与本地 LLM trace YAML |
| `{cwd}/subject/` | 项目语义模型、SQL summary、外部知识 |
| `{cwd}/.datus/config.yml` | 项目级模型、数据源与服务 pin |

## 冒烟测试

配置至少一个 provider 后，测试模型连通性：

```bash
uv run datus-agent probe-llm --config conf/agent.yml
```

测试数据源连通性：

```bash
uv run datus-agent check-db --config conf/agent.yml --datasource local_duckdb
```

启动 REPL：

```bash
uv run datus --config conf/agent.yml --datasource local_duckdb
```

一次性执行 workflow 时使用 `datus-agent run`：

```bash
uv run datus-agent run \
  --config conf/agent.yml \
  --datasource local_duckdb \
  --task_db_name duckdb-demo \
  --task "List the top 5 rows from the tree table"
```

## 基准测试

`bird_dev`、`spider2`、`semantic_layer` 是内置 benchmark 名称。它们的路径由 Datus 固定解析到 `{agent.home}/benchmark` 下；不要在 `agent.yml` 中覆盖这些内置 benchmark 的 `benchmark_path`。

内置路径应为：

```text
~/.datus/benchmark/bird/dev_20240627/
~/.datus/benchmark/spider2/spider2-snow/
~/.datus/benchmark/semantic_layer/
```

只有自定义 benchmark 需要在 `agent.benchmark` 下添加配置。

### BIRD

把 BIRD dev 数据集下载到 Datus home：

```bash
cd ~/.datus
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip
mkdir -p benchmark/bird
mv dev_20240627 benchmark/bird/
cd benchmark/bird/dev_20240627
unzip dev_databases
```

配置指向解压后 SQLite 数据库的数据源：

```yaml
agent:
  services:
    datasources:
      bird_sqlite:
        type: sqlite
        path_pattern: ~/.datus/benchmark/bird/dev_20240627/dev_databases/**/*.sqlite
```

初始化 metadata 并运行指定任务：

```bash
uv run datus-agent bootstrap-kb \
  --config conf/agent.yml \
  --datasource bird_sqlite \
  --benchmark bird_dev \
  --kb_update_strategy overwrite

uv run datus-agent benchmark \
  --config conf/agent.yml \
  --datasource bird_sqlite \
  --benchmark bird_dev \
  --workflow fixed \
  --schema_linking_rate medium \
  --benchmark_task_ids 14 15
```

### Spider 2.0 Snow

配置 Snowflake 数据源。数据源名称可以自定义，示例里使用 `snowflake`。

```yaml
agent:
  services:
    datasources:
      snowflake:
        type: snowflake
        account: ${SNOWFLAKE_ACCOUNT}
        username: ${SNOWFLAKE_USER}
        password: ${SNOWFLAKE_PASSWORD}
        warehouse: ${SNOWFLAKE_WAREHOUSE}
```

初始化并运行指定任务：

```bash
uv run datus-agent bootstrap-kb \
  --config conf/agent.yml \
  --datasource snowflake \
  --benchmark spider2 \
  --kb_update_strategy overwrite

uv run datus-agent benchmark \
  --config conf/agent.yml \
  --datasource snowflake \
  --benchmark spider2 \
  --benchmark_task_ids sf_bq104
```

Spider metadata 初始化可能需要数小时，因为 benchmark 包含大量表。

### 语义层

MetricFlow 通过 semantic adapter 系统配置，不再需要手动运行 `poetry lock`、`mf setup`，也不需要直接编辑 `~/.metricflow/config.yml`。

至少需要配置一个数据源，并显式配置 MetricFlow semantic adapter：

```yaml
agent:
  services:
    datasources:
      duckdb:
        type: duckdb
        uri: duckdb:///path/to/duck.db
    semantic_layer:
      metricflow: {}
```

`/services semantic` TUI 可以添加 `metricflow` 条目；如果缺少适配器包，也会安装 `datus-semantic-metricflow`。

把 semantic-layer benchmark 数据放到：

```text
~/.datus/benchmark/semantic_layer/
```

然后运行：

```bash
uv run datus-agent bootstrap-kb \
  --config conf/agent.yml \
  --datasource duckdb \
  --components metrics \
  --kb_update_strategy overwrite

uv run datus-agent benchmark \
  --config conf/agent.yml \
  --datasource duckdb \
  --benchmark semantic_layer \
  --workflow metric_to_sql
```

## 可观测性

Datus 提供三种互补的执行观测方式：

| 范围 | 机制 | 适用场景 |
| --- | --- | --- |
| 当前 REPL turn | 按 `Ctrl+O` | 本地交互时查看工具调用、SQL、原始输出 |
| 本地文件 | `--save_llm_trace` 或 `save_llm_trace: true` | 不接入托管 tracing 系统时，调试精确 prompt 与模型输出 |
| 托管 trace | LangSmith 和/或 Langfuse | 跨运行查看 workflow、benchmark、chat、LiteLLM 与 OpenAI Agents SDK trace |

### REPL 内联 Trace

在 `datus` REPL 中，运行中或运行后按 `Ctrl+O` 可切换 verbose trace 详情。再次按下，或按 `q`，回到 compact 视图。

### 本地 YAML Trace

使用 `--save_llm_trace` 持久化模型输入输出：

```bash
uv run datus-agent --save_llm_trace run \
  --config conf/agent.yml \
  --datasource local_duckdb \
  --task_db_name duckdb-demo \
  --task "Summarize the tree table"
```

也可以在 custom model 条目上启用：

```yaml
agent:
  models:
    my-internal:
      type: openai
      base_url: https://internal.example.com/v1
      api_key: ${MY_KEY}
      model: internal-gpt-4
      save_llm_trace: true
```

Trace YAML 会写入 `{agent.home}/trajectory/...`。Workflow checkpoint 也保存在同一棵 trajectory 目录下；配置外部 observability 后，保存的 workflow metadata 中可能包含 `trace_id`、`trace_span_id`、`trace_run_id`、`trace_provider` 等稳定 trace 引用字段。

### 外部 Observability

外部 trace export 只通过 `agent.observability.tracing.enabled: true` 启用。只设置 provider API key 不会自动打开 tracing。

`adapters` 可省略，默认使用 `langfuse`。内置的 `otlp`、`langsmith`、`langfuse`、`braintrust`、`datadog` adapter 都发送 OpenTelemetry trace。平台 adapter 是薄 preset：负责解析平台 endpoint 和鉴权，然后复用共享 OTLP exporter。基础 trace export 不需要安装平台 SDK。

```yaml
agent:
  observability:
    tracing:
      enabled: true
      capture_content: true
```

上面的配置等价于：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      adapters:
        - type: langfuse
```

LangSmith 使用 `LANGSMITH_API_KEY` 或 `LANGCHAIN_API_KEY`；`LANGSMITH_PROJECT` 可选。Langfuse 使用 `LANGFUSE_PUBLIC_KEY` 与 `LANGFUSE_SECRET_KEY`；`LANGFUSE_HOST` 可选，默认使用 Langfuse Cloud US。Datus 会在内部生成 Langfuse Basic Auth header。

需要同时发送到多个后端时，显式配置多个 adapter：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      adapters:
        - type: langsmith
        - type: langfuse
```

需要完全控制 endpoint/header 时，继续使用通用 OTLP：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      adapters:
        - type: otlp
          endpoint: https://collector.example/v1/traces
          headers:
            x-api-key: ${OTLP_API_KEY}
```

Braintrust 使用 `BRAINTRUST_API_KEY`，并需要 `BRAINTRUST_PARENT`、`BRAINTRUST_PROJECT_ID` 或 `BRAINTRUST_PROJECT_NAME` 之一。Datadog 默认发到本地 OTLP agent endpoint `http://localhost:4318/v1/traces`；其他部署可设置 `endpoint` 和 `DD_API_KEY`。

运行任意被 trace 的路径：

```bash
uv run datus-agent benchmark \
  --config conf/agent.yml \
  --datasource bird_sqlite \
  --benchmark bird_dev \
  --benchmark_task_ids 14
```

在配置的 provider project 中查看 trace。Trace 名称按操作组织，例如：

| 操作 | Trace 名称形态 |
| --- | --- |
| Workflow run | `cli/run/<workflow>` |
| Benchmark task | `benchmark/<benchmark>/<context>/task-<id>` |
| 知识库初始化 | `bootstrap-kb/<datasource>/<components>` |
| Chat/API session | `chat/<agent>` |

Tag 与 metadata 会包含 datasource、workflow、benchmark、task id、run id，以及可用时的 `agent.home`。

### 同时使用多个后端

多个 adapter 可以在同一进程中同时启用。Datus 会创建一个 OpenTelemetry provider，并为每个 adapter 挂一个 exporter/span processor，因此同一批 span 可以同时发送到 LangSmith、Langfuse、Braintrust、Datadog 和通用 OTLP collector。

Datus 不会在 workflow metadata 中持久化后端特有的 UI URL。使用 `trace_id`、`trace_span_id`、`trace_run_id`、`trace_provider` 等稳定 metadata 字段，将 workflow checkpoint 关联到对应 backend trace。

修改 tracing 环境变量后，需要重启 Datus 进程。Tracing setup 每个进程只初始化一次。

## 参考

- [快速开始](../getting_started/Quickstart.zh.md)
- [配置指南](../configuration/introduction.zh.md)
- [Agent 配置](../configuration/agent.zh.md)
- [Benchmark 手册](../benchmark/benchmark_manual.zh.md)
- [语义层配置](../configuration/semantic_layer.zh.md)
- [LLM Trace 使用](../training/llm_trace_usage.zh.md)
