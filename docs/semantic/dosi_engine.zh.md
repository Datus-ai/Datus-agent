# Dosi 语义引擎

Datus Agent 将 `dosi-engine` Python binding 作为锁定依赖内置。语义工具通过 `datus/tools/semantic_tools/dosi/` 中的运行时代码直接调用原生引擎。

## 安装

在 Datus-agent checkout 中运行：

```bash
uv sync --locked
```

锁文件固定了 Agent 和 CI 使用的引擎版本。开发和部署应使用该锁定环境。

## 配置

```yaml
agent:
  services:
    datasources:
      warehouse:
        type: duckdb
        uri: /absolute/path/to/warehouse.db
```

Dosi 是内置语义引擎，无需选择。Agent 递归发现 `subject/semantic_models/<datasource>/` 下的 OSI YAML 或 JSON 文件，跳过单独的指标片段。可用 `semantic_model_path` 为单次请求指定文件。旧的 `agent.services.semantic_layer` 和项目级 `semantic` 配置会被忽略并输出警告。

## Agent 使用的能力

Agent 使用 Dosi 发现语义模型、指标及指标可用维度，编译和执行指标查询，校验指定模型，以及运行原生归因。`semantic_modeling` 使用引擎的创作规范和校验。Explorer 在 OSI YAML 中读取、修改和删除指标节点，并同步受影响的 Knowledge Base 条目。独立的 Dosi CLI、REST 和 MCP 服务不属于 Agent 集成范围。

每个模型文件对应一个引擎实例，文件变化后运行时会刷新。Dosi 负责 SQL 方言和查询规划；Agent 提供当前数据源连接，并在 Knowledge Base 中维护主题路径。
