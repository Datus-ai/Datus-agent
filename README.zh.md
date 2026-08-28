<p align="center">
  <strong>Datus · 开源数据工程 Agent</strong>
</p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache&logoColor=white" alt="License: Apache 2.0"></a>
  <a href="https://pypi.org/project/datus-agent/"><img src="https://img.shields.io/pypi/v/datus-agent?logo=pypi&logoColor=white&color=654FF0" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <a href="https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg"><img src="https://img.shields.io/badge/Slack-join%20chat-4A154B?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZiI%2BPHBhdGggZD0iTTUuMDQyIDE1LjE2NWEyLjUyOCAyLjUyOCAwIDAgMS0yLjUyIDIuNTIzQTIuNTI4IDIuNTI4IDAgMCAxIDAgMTUuMTY1YTIuNTI3IDIuNTI3IDAgMCAxIDIuNTIyLTIuNTJoMi41MnYyLjUyek02LjMxMyAxNS4xNjVhMi41MjcgMi41MjcgMCAwIDEgMi41MjEtMi41MiAyLjUyNyAyLjUyNyAwIDAgMSAyLjUyMSAyLjUydjYuMzEzQTIuNTI4IDIuNTI4IDAgMCAxIDguODM0IDI0YTIuNTI4IDIuNTI4IDAgMCAxLTIuNTIxLTIuNTIydi02LjMxM3pNOC44MzQgNS4wNDJhMi41MjggMi41MjggMCAwIDEtMi41MjEtMi41MkEyLjUyOCAyLjUyOCAwIDAgMSA4LjgzNCAwYTIuNTI4IDIuNTI4IDAgMCAxIDIuNTIxIDIuNTIydjIuNTJIOC44MzR6TTguODM0IDYuMzEzYTIuNTI4IDIuNTI4IDAgMCAxIDIuNTIxIDIuNTIxIDIuNTI4IDIuNTI4IDAgMCAxLTIuNTIxIDIuNTIxSDIuNTIyQTIuNTI4IDIuNTI4IDAgMCAxIDAgOC44MzRhMi41MjggMi41MjggMCAwIDEgMi41MjItMi41MjFoNi4zMTJ6TTE4Ljk1NiA4LjgzNGEyLjUyOCAyLjUyOCAwIDAgMSAyLjUyMi0yLjUyMUEyLjUyOCAyLjUyOCAwIDAgMSAyNCA4LjgzNGEyLjUyOCAyLjUyOCAwIDAgMS0yLjUyMiAyLjUyMWgtMi41MjJWOC44MzR6TTE3LjY4OCA4LjgzNGEyLjUyOCAyLjUyOCAwIDAgMS0yLjUyMyAyLjUyMSAyLjUyNyAyLjUyNyAwIDAgMS0yLjUyLTIuNTIxVjIuNTIyQTIuNTI3IDIuNTI3IDAgMCAxIDE1LjE2NSAwYTIuNTI4IDIuNTI4IDAgMCAxIDIuNTIzIDIuNTIydjYuMzEyek0xNS4xNjUgMTguOTU2YTIuNTI4IDIuNTI4IDAgMCAxIDIuNTIzIDIuNTIyQTIuNTI4IDIuNTI4IDAgMCAxIDE1LjE2NSAyNGEyLjUyNyAyLjUyNyAwIDAgMS0yLjUyLTIuNTIydi0yLjUyMmgyLjUyek0xNS4xNjUgMTcuNjg4YTIuNTI3IDIuNTI3IDAgMCAxLTIuNTItMi41MjMgMi41MjYgMi41MjYgMCAwIDEgMi41Mi0yLjUyaDYuMzEzQTIuNTI3IDIuNTI3IDAgMCAxIDI0IDE1LjE2NWEyLjUyOCAyLjUyOCAwIDAgMS0yLjUyMiAyLjUyM2gtNi4zMTN6Ii8%2BPC9zdmc%2B" alt="Slack"></a>
</p>

<p align="center">
  <a href="https://datus.ai">官网</a> ·
  <a href="https://docs.datus.ai/zh/latest/">文档</a> ·
  <a href="https://docs.datus.ai/zh/latest/getting_started/Quickstart/">快速开始</a> ·
  <a href="https://dosi.datus.ai/">Dosi</a> ·
  <a href="https://docs.datus.ai/zh/latest/release_notes/">发布日志</a>
</p>

<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>

---

**Datus** 是面向现代数据栈的开源数据工程 Agent：用一个 Agent 连接数据仓库、数据目录、语义层和 BI，底座是一套沉淀在团队自己手里的可演进上下文引擎(evolvable context engine)。

Datus 可以完成 SQL 编写与验证、语义模型与指标构建，以及数据管道、报告和看板的生成；每一次执行与修正都会沉淀为上下文，持续提升后续输出的准确性。整个体系在生态上保持开放与灵活：数据库、BI、调度、LLM 乃至团队自己的工具，都能以标准方式接入。

## 架构

![Datus 架构](docs/assets/datus_architecture.svg)

整体架构自上而下分三段：谁在用、Agent 由什么组成、连接哪些系统，与上图对应：

- **按角色划分的三个入口**：数据工程师在 [Datus-CLI](https://docs.datus.ai/zh/latest/cli/introduction/) 里探索数据、构建资产；分析师通过 [Datus-Chat](https://docs.datus.ai/zh/latest/web_chatbot/introduction/)(Web、Slack/飞书、VS Code)提问，使用中的反馈会回流进 Agent；其他 Agent 和应用经 [Datus-API](https://docs.datus.ai/zh/latest/API/introduction/)(REST、MCP)消费。
- **Agent 核心**：[Subagent](https://docs.datus.ai/zh/latest/subagent/introduction/) 为单个业务域打包配好的上下文、工具和规则，[Skill](https://docs.datus.ai/zh/latest/skills/introduction/) 提供打包的扩展工具；底座是[上下文引擎](https://docs.datus.ai/zh/latest/knowledge_base/introduction/)：元数据、指标、参考 SQL、知识与本地文件，检索用业务域树加向量召回，[存储](https://docs.datus.ai/zh/latest/configuration/storage/)默认内嵌 LanceDB 和 SQLite，团队共享上下文时可换 PostgreSQL。
- **连接的系统**：LLM、数据仓库、[Dosi](https://dosi.datus.ai/) 语义层、作业调度、BI 工具与 MCP 服务端/客户端，经适配器接入；[Plugin](https://docs.datus.ai/zh/latest/plugin/introduction/) 则把第三方平台和公司内部工具接进 Agent。

## 核心能力

### 语义层

- **语义建模自动化**：Agent 读取数据库 schema 和历史 SQL，自动生成 [OSI](https://dosi.datus.ai/) 格式的语义模型与指标定义，不需要手写 YAML。
- **[Dosi](https://dosi.datus.ai/) 执行引擎**：把同一份语义模型编译成 13+ 种数据库方言的 SQL；它是一个独立程序，也可以单独以 CLI、REST 服务或 MCP server 的方式部署。
- **指标问答与归因**：[AskMetrics](https://docs.datus.ai/zh/latest/subagent/ask_metrics/) 依据指标定义回答业务问题，而不是临时拼 SQL；指标出现波动时，维度归因能定位变化来自哪个维度。

### Agent 与上下文

- **越用越准**：[上下文引擎](https://docs.datus.ai/zh/latest/getting_started/contextual_data_engineering/)汇集 schema、参考 SQL 和业务规则，使用中的每次修正都会写回，让后续回答持续变准。
- **[Subagent](https://docs.datus.ai/zh/latest/subagent/introduction/) 交付**：为一个业务领域配好上下文、工具和规则，打包成专属聊天机器人，通过 Web、API、MCP、Slack/飞书或 VS Code 提供给分析师。
- **数据工程自动化**：[内置 subagent](https://docs.datus.ai/zh/latest/subagent/builtin_subagents/) 承担跨库迁移、ETL 作业生成和宽表构建，可编排 [Airflow](https://docs.datus.ai/zh/latest/adapters/scheduler_adapters/) 调度，读写 Superset 和 Grafana 看板。
- **报告与看板生成**：在对话里直接生成自包含的 [HTML 报告和可交互看板](https://docs.datus.ai/zh/latest/subagent/gen_visual_report/)，本地即可预览，不依赖任何 SaaS 后端。

### 开放与治理

- **开放生态**：[17 种数据库适配器](https://docs.datus.ai/zh/latest/adapters/db_adapters/)、10+ LLM 提供商，以及 [MCP](https://docs.datus.ai/zh/latest/integration/mcp/) 服务端与客户端。
- **外部生态对接**：[Plugin](https://docs.datus.ai/zh/latest/plugin/introduction/) 框架把第三方平台和公司内部工具接入 Agent，一份 `datus-plugin.yml` 清单即可声明 CLI 命令、Skill 和 prompt 上下文，按项目启用。
- **[Skill](https://docs.datus.ai/zh/latest/skills/introduction/)**：遵循 agentskills.io 约定的打包工具，支持从 marketplace 安装。
- **企业级治理**：权限分级，SQL 按语句类型授权并由 AI 预审，bash 运行在 OS 级沙箱中，[trace](https://docs.datus.ai/zh/latest/develop/observability/) 可导出到任意 OTLP 平台。

## 快速开始

Linux 或 macOS：

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | sh
```

打开新终端，运行 `datus`，然后：

1. `/model` 配置模型
2. `/datasource` 添加数据源
3. `/init`(可选)扫描当前项目

也可以用 `pip install datus-agent` 手动安装(Python 3.12+)，更多安装方式见[快速开始指南](https://docs.datus.ai/zh/latest/getting_started/Quickstart/)。如果 `pip` 因回溯 `litellm` 版本而耗时数分钟(0.3.9 及更早版本会遇到)，改用 uv 几秒即可完成解析：`pip install uv && uv pip install datus-agent --system`。[端到端教程](https://docs.datus.ai/zh/latest/getting_started/contextual_data_engineering/)用示例数据集演示了完整使用流程。配置分两级：全局 `agent.yml` 存放主配置，项目下的 `.datus/config.yml` 保存当前模型、默认数据源等项目级覆盖，详见[配置文档](https://docs.datus.ai/zh/latest/configuration/introduction/)。

## 接入方式

以下示例使用名为 `demo` 的数据源，请先用 `/datasource` 创建。

| 接入方式 | 命令 | 适用场景 |
|-----------|---------|----------|
| **CLI**(交互式 REPL) | `datus --datasource demo` | 数据工程师探索数据、构建上下文、创建 subagent |
| **Web 聊天**(FastAPI + React) | `datus --web --datasource demo` | 分析师通过浏览器与 subagent 对话(`http://localhost:8501`) |
| **REST API**(FastAPI) | `datus-api --datasource demo` | 应用通过 REST 消费数据服务(`http://localhost:8000`) |
| **MCP 服务端** | `datus-mcp --datasource demo` | MCP 客户端(Claude Desktop、Cursor 等) |
| [**IM 网关**](https://docs.datus.ai/zh/latest/gateway/introduction/) | `datus-gateway` | 分析师在 Slack 或飞书中与 subagent 对话 |
| [**VS Code**](https://docs.datus.ai/zh/latest/vscode_extension/introduction/)(Datus Studio) | 连接 `datus --web` | IDE 内的目录浏览器、聊天面板、SQL 结果与 AI 图表 |

> **提示：** Print 模式向 stdout 流式输出 JSON，适合脚本与 CI：`datus -p "你的问题" --datasource demo`。

## 开发

### 开发 Datus

参与本体开发从这里开始：用 uv 安装依赖，提交前跑一遍 PR 测试与格式检查。

```bash
uv sync                                                                    # 安装依赖
uv run python ci/run-pr-tests.py upstream/main                             # PR CI 测试(无外部依赖)
uv run ruff format datus/ tests/ && uv run ruff check --fix datus/ tests/  # 格式化与 Lint
```

开发规范、架构模式与测试规则见 [CLAUDE.md](CLAUDE.md)。

### 开发 Plugin

为 Datus 编写扩展不需要改动本体代码：在 `datus-plugin.yml` 清单里声明 CLI 命令、Skill 和 prompt 上下文，打包后即可分发安装、按项目启用。完整流程见[插件开发指南](https://docs.datus.ai/zh/latest/plugin/development/)。

## 许可证

[Apache 2.0](LICENSE)
