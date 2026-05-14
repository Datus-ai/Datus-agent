# 发布说明

## 0.3

### 0.3.0

**新功能**

***Datus API***

- **FastAPI REST API** - 基于 FastAPI 的 REST API，包含 service/model 分层、CLI 入口、流式 Chat、任务追踪、SQL 执行停止、多选 `ask_user`、success story 持久化、知识库 bootstrap API，以及 API 请求侧的 proxy source / interactive mode 控制。[#520](https://github.com/Datus-ai/Datus-agent/pull/520) [#538](https://github.com/Datus-ai/Datus-agent/pull/538) [#539](https://github.com/Datus-ai/Datus-agent/pull/539) [#551](https://github.com/Datus-ai/Datus-agent/pull/551) [#553](https://github.com/Datus-ai/Datus-agent/pull/553) [#555](https://github.com/Datus-ai/Datus-agent/pull/555) [#606](https://github.com/Datus-ai/Datus-agent/pull/606) [#610](https://github.com/Datus-ai/Datus-agent/pull/610) [文档](https://docs.datus.ai/zh/API/introduction/)
- **模型发现 API** - 模型发现、单请求模型覆盖、current model 元数据和 ISO-8601 UTC 时间戳格式。[#643](https://github.com/Datus-ai/Datus-agent/pull/643) [#649](https://github.com/Datus-ai/Datus-agent/pull/649) [#700](https://github.com/Datus-ai/Datus-agent/pull/700) [文档](https://docs.datus.ai/zh/API/models/)
- **图表推荐与可视化 API** - 支持 Datus Chat 与外部应用生成 dashboard-ready 的可视化结果。[#545](https://github.com/Datus-ai/Datus-agent/pull/545) [文档](https://docs.datus.ai/zh/getting_started/dashboard_copilot/)

***Datus Chat 与 IM 网关***

- **Datus Chat（FastAPI 替换 Streamlit）** - 用 FastAPI + `@datus/web-chatbot` 替换旧的 Streamlit chatbot，并新增 Datus Chat 模块。[#543](https://github.com/Datus-ai/Datus-agent/pull/543) [#554](https://github.com/Datus-ai/Datus-agent/pull/554) [文档](https://docs.datus.ai/zh/web_chatbot/introduction/)
- **Slack 与 Feishu/Lark 网关** - 新增 IM 网关，支持 channel 配置、daemon mode、流式回复、feedback action，并将 `datus-claw` 统一改名为 `datus-gateway`。[#559](https://github.com/Datus-ai/Datus-agent/pull/559) [#562](https://github.com/Datus-ai/Datus-agent/pull/562) [#565](https://github.com/Datus-ai/Datus-agent/pull/565) [#616](https://github.com/Datus-ai/Datus-agent/pull/616) [#623](https://github.com/Datus-ai/Datus-agent/pull/623) [#593](https://github.com/Datus-ai/Datus-agent/pull/593) [文档](https://docs.datus.ai/zh/gateway/introduction/)

***项目与工作区配置***

- **项目感知的 Configure/Init 流程** - 将 setup 拆为项目感知的 configure/init 流程，新增项目级 `.datus/config.yml`、项目级 memory、自动 datasource/service setup，以及重建后的 `.datus` 目录结构。[#542](https://github.com/Datus-ai/Datus-agent/pull/542) [#578](https://github.com/Datus-ai/Datus-agent/pull/578) [#592](https://github.com/Datus-ai/Datus-agent/pull/592) [#608](https://github.com/Datus-ai/Datus-agent/pull/608) [文档](https://docs.datus.ai/zh/cli/init_command/)
- **统一运行时服务配置** - 围绕 `services.datasources`、`services.bi_platforms`、semantic layer、scheduler 建模；CLI 统一使用 `--datasource`。[#614](https://github.com/Datus-ai/Datus-agent/pull/614) [#633](https://github.com/Datus-ai/Datus-agent/pull/633) [#636](https://github.com/Datus-ai/Datus-agent/pull/636) [#642](https://github.com/Datus-ai/Datus-agent/pull/642) [文档](https://docs.datus.ai/zh/configuration/datasources/)
- **一行安装脚本** - 新增 Linux/macOS `curl | sh` 一行安装脚本，并更新 quickstart 与 service 文档。[#613](https://github.com/Datus-ai/Datus-agent/pull/613) [#611](https://github.com/Datus-ai/Datus-agent/pull/611) [#667](https://github.com/Datus-ai/Datus-agent/pull/667) [文档](https://docs.datus.ai/zh/getting_started/Quickstart/)

***CLI 体验***

- **统一 `/` 命令前缀** - 将交互命令统一到 `/` 前缀，新增 `/model`、`/skill`、`/mcp`、`/agent`、`/subagent`、交互式输入和流式 `/bootstrap` TUI。[#621](https://github.com/Datus-ai/Datus-agent/pull/621) [#635](https://github.com/Datus-ai/Datus-agent/pull/635) [#650](https://github.com/Datus-ai/Datus-agent/pull/650) [#655](https://github.com/Datus-ai/Datus-agent/pull/655) [#656](https://github.com/Datus-ai/Datus-agent/pull/656) [#659](https://github.com/Datus-ai/Datus-agent/pull/659) [#683](https://github.com/Datus-ai/Datus-agent/pull/683) [文档](https://docs.datus.ai/zh/cli/reference/)
- **`/language` 与 `/effort` 命令** - 用 `/language` 固定响应语言，`/effort` 控制 reasoning 强度，以及 `/<service>.<method>` 只读服务调用分发。[#641](https://github.com/Datus-ai/Datus-agent/pull/641) [#653](https://github.com/Datus-ai/Datus-agent/pull/653) [#631](https://github.com/Datus-ai/Datus-agent/pull/631) [文档](https://docs.datus.ai/zh/cli/language_command/)
- **CLI Print Mode 与体验优化** - CLI print mode、proxy tools、重做底部状态栏、固定 streaming/tool 状态行、改进 markdown streaming，并恢复 `@` reference 自动补全。[#489](https://github.com/Datus-ai/Datus-agent/pull/489) [#501](https://github.com/Datus-ai/Datus-agent/pull/501) [#583](https://github.com/Datus-ai/Datus-agent/pull/583) [#586](https://github.com/Datus-ai/Datus-agent/pull/586) [#654](https://github.com/Datus-ai/Datus-agent/pull/654) [#664](https://github.com/Datus-ai/Datus-agent/pull/664) [#661](https://github.com/Datus-ai/Datus-agent/pull/661) [#662](https://github.com/Datus-ai/Datus-agent/pull/662) [文档](https://docs.datus.ai/zh/cli/introduction/)
- **新增模型与订阅计划** - Codex OAuth、Claude Subscription、Coding Plan、OpenRouter、MiniMax、GLM、BigModel、Z.AI 等模型/计划支持，并重建 provider-based 模型配置和 provider catalog。[#487](https://github.com/Datus-ai/Datus-agent/pull/487) [#635](https://github.com/Datus-ai/Datus-agent/pull/635) [#687](https://github.com/Datus-ai/Datus-agent/pull/687) [#693](https://github.com/Datus-ai/Datus-agent/pull/693) [文档](https://docs.datus.ai/zh/cli/model_command/)
- **权限 Profile** - 新增 `normal` / `auto` / `dangerous` 权限 profile，支持 subagent-aware permission hooks，并放宽正常模式下的安全发现类工具。[#646](https://github.com/Datus-ai/Datus-agent/pull/646) [#652](https://github.com/Datus-ai/Datus-agent/pull/652) [文档](https://docs.datus.ai/zh/integration/skills/)

***数据工程 Subagent 与 Skills***

- **数据工程 Agents 与 Skills** - 新增跨库迁移、ETL/job 生成、scheduler workflow、表生成、dashboard 生成、BI/scheduler 编排等内置 agent 与技能。[#494](https://github.com/Datus-ai/Datus-agent/pull/494) [#525](https://github.com/Datus-ai/Datus-agent/pull/525) [#564](https://github.com/Datus-ai/Datus-agent/pull/564) [#575](https://github.com/Datus-ai/Datus-agent/pull/575) [#580](https://github.com/Datus-ai/Datus-agent/pull/580) [#639](https://github.com/Datus-ai/Datus-agent/pull/639) [文档](https://docs.datus.ai/zh/subagent/builtin_subagents/)
- **交付物 Validation Hook** - 表交付物 validation hook、共享 deliverable node、validation skills，以及 semantic/metric generation 的 publish gate。[#657](https://github.com/Datus-ai/Datus-agent/pull/657) [#663](https://github.com/Datus-ai/Datus-agent/pull/663) [#665](https://github.com/Datus-ai/Datus-agent/pull/665) [文档](https://docs.datus.ai/zh/integration/validation/)
- **自然语言指标与 Skill Creator** - 新增自然语言指标创建、wheel 内置 skills 打包、skill frontmatter scope，以及用于交互式 skill 创作的 `skill-creator` subagent。[#504](https://github.com/Datus-ai/Datus-agent/pull/504) [#526](https://github.com/Datus-ai/Datus-agent/pull/526) [#627](https://github.com/Datus-ai/Datus-agent/pull/627) [#645](https://github.com/Datus-ai/Datus-agent/pull/645) [#676](https://github.com/Datus-ai/Datus-agent/pull/676) [文档](https://docs.datus.ai/zh/subagent/customized_subagent/)

***记忆与 Reference Template***

- **Auto Memory** - 基于 `MEMORY.md` 的 Auto Memory、emergent topic tree、空 memory prompt、project/session isolation。[#498](https://github.com/Datus-ai/Datus-agent/pull/498) [#620](https://github.com/Datus-ai/Datus-agent/pull/620) [#595](https://github.com/Datus-ai/Datus-agent/pull/595) [#523](https://github.com/Datus-ai/Datus-agent/pull/523) [#594](https://github.com/Datus-ai/Datus-agent/pull/594) [文档](https://docs.datus.ai/zh/integration/memory/)
- **Reference Template** - 新增 Reference Template 机制，并修复 bootstrap 中 reference template 解析问题。[#508](https://github.com/Datus-ai/Datus-agent/pull/508) [#574](https://github.com/Datus-ai/Datus-agent/pull/574) [#677](https://github.com/Datus-ai/Datus-agent/pull/677) [文档](https://docs.datus.ai/zh/knowledge_base/reference_template/)

***生态与适配器***

- **Datus Studio（VSCode 插件）** - 官方 VSCode 插件，把 Datus 能力带进 IDE：Object Explorer（Catalog/Context 树）、SubAgent 创建向导、带 `@` 引用和 Plan 模式的 Datus Chat 面板、可切换 datasource/subagent、SQL Result & AI Chart 面板（ECharts），以及绑定 workspace 的 FileSystem 工具。通过单一 Endpoint 连接任意 Datus-agent Web Server（`datus-cli --web`）。[#713](https://github.com/Datus-ai/Datus-agent/pull/713) [#717](https://github.com/Datus-ai/Datus-agent/pull/717) [文档](https://docs.datus.ai/zh/vscode_extension/introduction/)
- **数据库适配器：Greenplum 与 Migration Mixin** - `datus-db-adapters` 新增 Greenplum，提升 metadata robustness、thread-safe connector isolation、dialect-specific identifier quoting，并新增 migration workflow 所需的 `MigrationTargetMixin`。[datus-db-adapters#40](https://github.com/Datus-ai/datus-db-adapters/pull/40) [#43](https://github.com/Datus-ai/datus-db-adapters/pull/43) [#45](https://github.com/Datus-ai/datus-db-adapters/pull/45) [#46](https://github.com/Datus-ai/datus-db-adapters/pull/46) [#47](https://github.com/Datus-ai/datus-db-adapters/pull/47) [#48](https://github.com/Datus-ai/datus-db-adapters/pull/48) [文档](https://docs.datus.ai/zh/adapters/db_adapters/)
- **BI 适配器：Superset 与 Grafana** - `datus-bi-adapters` 新增 `datus-bi-core`、Superset 和 Grafana adapters，支持 list API、chart data retrieval、dashboard/chart 写入校验、分页 envelope、datasource metadata 修复和 dashboard layout 改进。[datus-bi-adapters#1](https://github.com/Datus-ai/datus-bi-adapters/pull/1) [#2](https://github.com/Datus-ai/datus-bi-adapters/pull/2) [#3](https://github.com/Datus-ai/datus-bi-adapters/pull/3) [#7](https://github.com/Datus-ai/datus-bi-adapters/pull/7) [#8](https://github.com/Datus-ai/datus-bi-adapters/pull/8) [#9](https://github.com/Datus-ai/datus-bi-adapters/pull/9) [文档](https://docs.datus.ai/zh/adapters/bi_adapters/)
- **Scheduler 适配器：Airflow** - `datus-scheduler-adapters` 新增 `datus-scheduler-core` 与 Airflow adapter，支持 DuckDB DAG 执行、多租户 DAG folder、job/run list result envelope、inactive DAG 删除语义，并发布到 `datus-scheduler-airflow` 0.1.2。[datus-scheduler-adapters#2](https://github.com/Datus-ai/datus-scheduler-adapters/pull/2) [#3](https://github.com/Datus-ai/datus-scheduler-adapters/pull/3) [#4](https://github.com/Datus-ai/datus-scheduler-adapters/pull/4) [#8](https://github.com/Datus-ai/datus-scheduler-adapters/pull/8) [#9](https://github.com/Datus-ai/datus-scheduler-adapters/pull/9) [文档](https://docs.datus.ai/zh/adapters/scheduler_adapters/)
- **语义层适配器拆分** - `datus-semantic-adapter` 拆分出 `datus-semantic-core`，迁移 MetricFlow adapter，支持 dict config injection、语义 adapter contract tests、datasource 术语、可配置 semantic model 路径和更严格的 MetricFlow validation。[datus-semantic-adapter#6](https://github.com/Datus-ai/datus-semantic-adapter/pull/6) [#7](https://github.com/Datus-ai/datus-semantic-adapter/pull/7) [#9](https://github.com/Datus-ai/datus-semantic-adapter/pull/9) [#10](https://github.com/Datus-ai/datus-semantic-adapter/pull/10) [文档](https://docs.datus.ai/zh/adapters/semantic_adapters/)

**增强**

- **流式与会话稳定性** - 修复并增强 web/chat/gateway streaming、compact/resume、群聊 thread 处理、Feishu 权限、Slack 回复、API node 创建、session persistence 和时间戳格式。[#531](https://github.com/Datus-ai/Datus-agent/pull/531) [#548](https://github.com/Datus-ai/Datus-agent/pull/548) [#567](https://github.com/Datus-ai/Datus-agent/pull/567) [#568](https://github.com/Datus-ai/Datus-agent/pull/568) [#638](https://github.com/Datus-ai/Datus-agent/pull/638) [#674](https://github.com/Datus-ai/Datus-agent/pull/674) [#680](https://github.com/Datus-ai/Datus-agent/pull/680) [#689](https://github.com/Datus-ai/Datus-agent/pull/689) [#700](https://github.com/Datus-ai/Datus-agent/pull/700) [文档](https://docs.datus.ai/zh/API/chat/)
- **生成稳定性** - 提升 semantic、metric、reference-template、dashboard、SQL prompt、query-metric 生成稳定性。[#596](https://github.com/Datus-ai/Datus-agent/pull/596) [#604](https://github.com/Datus-ai/Datus-agent/pull/604) [#690](https://github.com/Datus-ai/Datus-agent/pull/690) [#691](https://github.com/Datus-ai/Datus-agent/pull/691) [#692](https://github.com/Datus-ai/Datus-agent/pull/692) [#697](https://github.com/Datus-ai/Datus-agent/pull/697) [文档](https://docs.datus.ai/zh/subagent/gen_semantic_model/)
- **文件系统与数据隔离** - 通过 `filesystem_strict`、project-root zone policy、安全 search、credential redaction、严格 FuncTool result handling 增强文件系统与数据隔离。[#588](https://github.com/Datus-ai/Datus-agent/pull/588) [#597](https://github.com/Datus-ai/Datus-agent/pull/597) [#603](https://github.com/Datus-ai/Datus-agent/pull/603) [#681](https://github.com/Datus-ai/Datus-agent/pull/681) [#694](https://github.com/Datus-ai/Datus-agent/pull/694) [文档](https://docs.datus.ai/zh/configuration/agent/)
- **Storage 重构** - 统一 `datus_db`、datasource 隔离、singleton registry、可插拔 RDB/vector backend，并支持通过 `datus-storage-postgresql` 使用 PostgreSQL 后端。[#493](https://github.com/Datus-ai/Datus-agent/pull/493) [#499](https://github.com/Datus-ai/Datus-agent/pull/499) [文档](https://docs.datus.ai/zh/configuration/storage/)
- **CI 流程重构** - 拆分 PR acceptance/nightly 流程，新增 docker-backed adapter integration tests 与 test-quality audit workflow，并修复多项 nightly、unit、integration regressions。[#589](https://github.com/Datus-ai/Datus-agent/pull/589) [#600](https://github.com/Datus-ai/Datus-agent/pull/600) [#601](https://github.com/Datus-ai/Datus-agent/pull/601) [#634](https://github.com/Datus-ai/Datus-agent/pull/634)

**文档补全**

- **REST API、IM 网关与 CLI 文档** - 新增 REST API deployment / chat / KB / models 文档，Slack 与 Feishu IM gateway 文档，以及 `/model`、`/language`、`/effort`、`/init`、`/bootstrap`、service command、`--datasource` 流程文档。[文档](https://docs.datus.ai/zh/API/deployment/)
- **配置文档** - 新增 datasources、semantic layer、BI platforms、schedulers、PostgreSQL-backed storage 配置文档。[文档](https://docs.datus.ai/zh/configuration/datasources/)
- **Subagent 文档** - 新增 dashboard generation、table generation、scheduler workflow、data pipeline、metrics、semantic model、SQL summary 等 subagent 文档。[文档](https://docs.datus.ai/zh/subagent/builtin_subagents/)
- **Adapter、Memory 与 Reference Template 文档** - 刷新 adapter、memory、reference template、quickstart、benchmark 与 docs deployment 文档。[#530](https://github.com/Datus-ai/Datus-agent/pull/530) [#536](https://github.com/Datus-ai/Datus-agent/pull/536) [#549](https://github.com/Datus-ai/Datus-agent/pull/549) [#556](https://github.com/Datus-ai/Datus-agent/pull/556) [#611](https://github.com/Datus-ai/Datus-agent/pull/611) [#622](https://github.com/Datus-ai/Datus-agent/pull/622) [#667](https://github.com/Datus-ai/Datus-agent/pull/667) [文档](https://docs.datus.ai/zh/adapters/db_adapters/)

## 0.2

### 0.2.6

**新功能**

- **Ask User Tool** - 引入交互式 `ask_user` 工具，支持内联自由文本输入和批量提问能力，已集成进 GenSQL 与 GenReport node，支持 human-in-the-loop workflow。[#457](https://github.com/Datus-ai/Datus-agent/pull/457) [#460](https://github.com/Datus-ai/Datus-agent/pull/460) [#481](https://github.com/Datus-ai/Datus-agent/pull/481)
- **Skill Marketplace CLI** - 内置 marketplace，可直接从 CLI 发现、安装、管理社区 skills。[#416](https://github.com/Datus-ai/Datus-agent/pull/416) [文档](https://docs.datus.ai/zh/integration/skills/)
- **General Chat Agent** - 通用聊天 agent，支持 SQL 生成以外的灵活会话场景。[#452](https://github.com/Datus-ai/Datus-agent/pull/452)
- **Explore Task Tool** - 新增 exploration 工具,用于在 agent 内导航与管理任务。[#455](https://github.com/Datus-ai/Datus-agent/pull/455)
- **Storage Adapter** - 可插拔 storage adapter 层，便于灵活接入后端。[#446](https://github.com/Datus-ai/Datus-agent/pull/446)
- **4 个新数据库适配器** - 在 [datus-db-adapters](https://github.com/Datus-ai/datus-db-adapters) 仓库新增 ClickHouse、Hive、Spark、Trino 适配器，均可通过 `pip install datus-<database>` 独立安装。[文档](https://docs.datus.ai/zh/adapters/db_adapters/)

**增强**

- **Session Resume/Rewind** - 新增 `/resume`、`/rewind`、`.interrupt` 命令，配合交互式方向键选择器浏览会话历史。[#438](https://github.com/Datus-ai/Datus-agent/pull/438) [#470](https://github.com/Datus-ai/Datus-agent/pull/470) [文档](https://docs.datus.ai/zh/cli/chat_command/)
- **Scoped Context Filter** - 基于 filter 的作用域上下文,SQL 生成时知识检索更精准。[#441](https://github.com/Datus-ai/Datus-agent/pull/441)
- **Subagent 直连 Web** - 新增 `--subagent` CLI 参数，直接通过 web 界面启动 subagent。[#447](https://github.com/Datus-ai/Datus-agent/pull/447)
- **CLI 交互体验** - 增强多行输入支持与省略号截断,可读性更好。[#468](https://github.com/Datus-ai/Datus-agent/pull/468)
- **Subagent 指引简化** - 精简 subagent 使用说明,上手流程更清晰。[#469](https://github.com/Datus-ai/Datus-agent/pull/469)
- **Function Tool 加固** - 强制 read-only SQL 执行,去重 tool 注册,改进 docstring。[#474](https://github.com/Datus-ai/Datus-agent/pull/474)
- **当前日期注入** - 直接把 `current_date` 注入 system prompt,移除独立的 `get_current_date` 工具。[#473](https://github.com/Datus-ai/Datus-agent/pull/473)
- **数据压缩** - 为 `query_metrics` 增加响应压缩,并修复 `DataCompressor` 的 model_name 处理,降低 token 消耗。[#435](https://github.com/Datus-ai/Datus-agent/pull/435) [#472](https://github.com/Datus-ai/Datus-agent/pull/472)

**Bug 修复**

- **Kimi-K2.5 与 Qwen3-Coder-Plus 初始化** - 修复交互式初始化时这些模型的 temperature/top_p 支持。[#483](https://github.com/Datus-ai/Datus-agent/pull/483)
- **Generation Hooks 条件** - 修复 `generation_hooks` 使用正确的 `where` 表达式。[#482](https://github.com/Datus-ai/Datus-agent/pull/482)
- **Ctrl+O 切换** - 修复 Ctrl+O 切换时早先回合响应未显示的问题。[#477](https://github.com/Datus-ai/Datus-agent/pull/477)
- **缺失 tabulate 依赖** - 在 pyproject.toml 与 requirements.txt 补上 `tabulate` 依赖。[#476](https://github.com/Datus-ai/Datus-agent/pull/476)
- **Skill 扫描路径** - 从默认扫描路径中移除 `~/.claude/skills`,并改进 ChatAgenticNode 的配置传递。[#475](https://github.com/Datus-ai/Datus-agent/pull/475)

**文档**

- 新增 Hive、Spark、ClickHouse、Trino 数据库适配器文档。[#464](https://github.com/Datus-ai/Datus-agent/pull/464) [文档](https://docs.datus.ai/zh/adapters/db_adapters/)
- 新增 resume/rewind 命令文档。[#465](https://github.com/Datus-ai/Datus-agent/pull/465)

### 0.2.5

**新功能**

- **OpenAI Agent SDK 0.7.0 升级，支持 Kimi-2.5 与 Gemini-3** - 用 `litellm_adapter` 和 `sdk_patches` 重建模型层，无缝接入最新的 Kimi-2.5 与 Gemini-3 系列模型。
- **AgentSkills 支持** - 引入完整的 Skill 系统，包含 skill 配置、注册、管理与权限控制，同时支持 bash 与 function 形态的 skill 工具。[文档](https://docs.datus.ai/zh/integration/skills/)
- **Tools as MCP Server** - 将 Datus 的数据库工具与 context search 暴露为 MCP server，可对接 Claude Desktop、Claude Code 等 MCP 兼容客户端。[文档](https://docs.datus.ai/zh/integration/mcp/)

**增强**

- **知识生成迭代** - 增强 external knowledge node，改进知识存储并提升 context search 准确率。[文档](https://docs.datus.ai/zh/knowledge_base/ext_knowledge/)
- **语义工具优化** - 优化语义工具与 context search，在 CLI 中获得更快、更相关的结果。
- **生成 Prompt 字符串校验** - 加强多个 prompt template 的字符串校验，提升生成结果可靠性。
- **基于 Action 的用户交互模型** - 重做 CLI 交互层，统一以 action-based 模型驱动 execution、generation 与 planning。
- **Reference SQL 并行化与日期支持** - 并行化 reference SQL 初始化加速 bootstrap，并增强日期表达式解析。[文档](https://docs.datus.ai/zh/knowledge_base/reference_sql/)
- **Bootstrap Markdown 摘要** - bootstrap 完成后展示格式化的 Markdown 摘要，便于快速浏览生成结果。[文档](https://docs.datus.ai/zh/getting_started/dashboard_copilot/)
- **Subject 条目删除** - 可直接在 `@subject` 界面删除 semantic models、metrics 与 SQL summaries。[文档](https://docs.datus.ai/zh/cli/context_command/#subject)

**Bug 修复**

- **Subject Node 竞争条件** - 修复并行创建多个 subject node 时的竞争条件,提升并发安全性。
- **多轮 Benchmark 评估** - 修复多轮评估中 agent state、workflow runner、配置处理相关的问题。[文档](https://docs.datus.ai/zh/benchmark/benchmark_manual/)
- **归因分析** - 简化归因分析逻辑，结果更清晰、可靠。

### 0.2.4

**Dashboard Copilot（自动生成）**

- Dashboard 转 Sub-Agent：从 BI dashboard 配置自动生成 sub-agent [#339](https://github.com/Datus-ai/Datus-agent/pull/339)
- BI dashboard bootstrap 时自动生成 semantic model [#368](https://github.com/Datus-ai/Datus-agent/pull/368)
- 直接从 Dashboard 组件生成 metrics 定义 [#363](https://github.com/Datus-ai/Datus-agent/pull/363)

**更好的语义层集成**

- Semantic Adapter：可插拔的外部 metric layer 适配器 [#355](https://github.com/Datus-ai/Datus-agent/pull/355)
- External Knowledge Storage：基于向量的知识检索增强 SQL 生成上下文 [#359](https://github.com/Datus-ai/Datus-agent/pull/359)
- 在 metrics schema 中新增 SQL 字段 [#364](https://github.com/Datus-ai/Datus-agent/pull/364)

**增强**

- 优化 reference SQL 搜索：去重并简化格式 [#348](https://github.com/Datus-ai/Datus-agent/pull/348) [#358](https://github.com/Datus-ai/Datus-agent/pull/358) [#375](https://github.com/Datus-ai/Datus-agent/pull/375)
- 增强 ContextSearch 方法与展示 [#347](https://github.com/Datus-ai/Datus-agent/pull/347)
- 改进 Plan Mode：Chat node 继承自 GenSQL agentic node [#334](https://github.com/Datus-ai/Datus-agent/pull/334)
- Catalog 界面改进：列注释与嵌套表行样式 [#345](https://github.com/Datus-ai/Datus-agent/pull/345) [#378](https://github.com/Datus-ai/Datus-agent/pull/378)
- 工具执行反馈：context 与 start 事件 [#340](https://github.com/Datus-ai/Datus-agent/pull/340) [#341](https://github.com/Datus-ai/Datus-agent/pull/341)
- 改进 prompt version 处理 [#367](https://github.com/Datus-ai/Datus-agent/pull/367) [#379](https://github.com/Datus-ai/Datus-agent/pull/379)
- 覆盖写入时清理废弃的 metric metadata 与 YAML 目录 [#362](https://github.com/Datus-ai/Datus-agent/pull/362) [#365](https://github.com/Datus-ai/Datus-agent/pull/365)

**重构**

- 语义模型与 metrics 架构重构 [#350](https://github.com/Datus-ai/Datus-agent/pull/350)
- 统一 subject tree 管理 [#349](https://github.com/Datus-ai/Datus-agent/pull/349)
- 可插拔的 DB 适配器架构 [#353](https://github.com/Datus-ai/Datus-agent/pull/353)
- Namespace 配置重构 [#346](https://github.com/Datus-ai/Datus-agent/pull/346)

**Bug 修复**

- 修复 Superset chart 中 query_context 为空的问题 [#372](https://github.com/Datus-ai/Datus-agent/pull/372)
- chatbot 中 tool call 跳过 render 处理 [#360](https://github.com/Datus-ai/Datus-agent/pull/360) [#380](https://github.com/Datus-ai/Datus-agent/pull/380)
- 修复 semantic model 与 metrics 去重问题 [#369](https://github.com/Datus-ai/Datus-agent/pull/369)
- 修复 context_search 中 subject_path 解析 [#357](https://github.com/Datus-ai/Datus-agent/pull/357)
- 改进 sample row 错误处理 [#354](https://github.com/Datus-ai/Datus-agent/pull/354)

### 0.2.3

**新功能**

- **内置教程数据集** - California Schools 数据集随安装包打包，并集成进 `datus-agent init` 流程，方便上手学习上下文数据工程。[#277](https://github.com/Datus-ai/Datus-agent/issues/277) [教程](https://docs.datus.ai/zh/getting_started/Datus_tutorial/)
- **增强的评测框架** - 新的 evaluation 命令，新增 Exact Match、Same Result Count（值不同）、Schema/Table Usage Match、Semantic/Metric Layer Correctness 等评测类别。[#264](https://github.com/Datus-ai/Datus-agent/issues/264)
- **基于插件的数据库连接器** - 数据库连接器重构为插件化架构，便于扩展与自定义适配器开发。[#284](https://github.com/Datus-ai/Datus-agent/issues/284)

**增强**

- **简化安装** - 默认安装中移除老的 transformers 依赖，加快安装、减小包体积。[#247](https://github.com/Datus-ai/Datus-agent/issues/247)
- **MetricFlow 配置简化** - MetricFlow 已原生支持 Datus 配置格式，简化配置。[#243](https://github.com/Datus-ai/Datus-agent/issues/243)
- **内置生成命令** - `/gen_semantic_model`、`/gen_metrics`、`/gen_sql_summary` subagent 开箱即用，无需额外配置。[#250](https://github.com/Datus-ai/Datus-agent/issues/250)
- **Agentic Node 集成** - 基于 workflow 的评测支持 agentic node,支持更复杂的测试场景。[#262](https://github.com/Datus-ai/Datus-agent/issues/262)
- **代码质量改进** - 重构 tool 模块、增强 node 逻辑，统一 `bootstrap-kb` 与 `gen_semantic_model` 使用同一实现。[#245](https://github.com/Datus-ai/Datus-agent/issues/245) [#250](https://github.com/Datus-ai/Datus-agent/issues/250)
- **Embedding 存储优化** - 重构 embedding model 存储并更新依赖，性能更好。[#247](https://github.com/Datus-ai/Datus-agent/issues/247)

**Bug 修复**

- **Schema 元数据处理** - 修复 schema_linking 命令中 definition 字段为空的问题,确保 schema 元数据正确传递给下游 node。[#327](https://github.com/Datus-ai/Datus-agent/issues/327)
- **初始化问题** - 修复多个初始化 bug,并修正 tutorial 模式下的配置文件校验。[#304](https://github.com/Datus-ai/Datus-agent/issues/304) [#303](https://github.com/Datus-ai/Datus-agent/issues/303)
- **环境变量兼容性** - 修复跨平台的环境变量处理，提升部署兼容性。[#294](https://github.com/Datus-ai/Datus-agent/issues/294)
- **评测摘要生成** - 修复 benchmark 评测中摘要生成的问题,评测报告更准确。[#314](https://github.com/Datus-ai/Datus-agent/issues/314)
- **FastEmbed 缓存目录** - 修复 fastembed 在不同平台上的缓存目录路径。[#251](https://github.com/Datus-ai/Datus-agent/issues/251)

### 0.2.2
跳过

### 0.2.1

**新功能**

- **Web Chatbot 升级** - 新增反馈收集、问题上报、流式输出，以及 `&hide_sidebar=true` 嵌入参数。[文档](https://docs.datus.ai/zh/web_chatbot/introduction/)
- **上下文生成命令** - subagent 中新增 `/gen_semantic_model`、`/gen_metrics`、`/gen_sql_summary` 命令，用于动态丰富知识库。[#192](https://github.com/Datus-ai/Datus-agent/issues/192) [文档](https://docs.datus.ai/zh/subagent/builtin_subagents/)
- **交互式上下文编辑** - `@catalog`、`@subject` 命令支持可视化编辑 semantic models、metrics 与 SQL summaries。[#219](https://github.com/Datus-ai/Datus-agent/issues/219) [#199](https://github.com/Datus-ai/Datus-agent/issues/199) [#175](https://github.com/Datus-ai/Datus-agent/issues/175) [文档](https://docs.datus.ai/zh/cli/context_command/#subject)
- **作用域知识库** - subagent 支持作用域 KB 初始化，提升上下文隔离与管理能力。[#217](https://github.com/Datus-ai/Datus-agent/issues/217)

**增强**

- **MetricFlow 集成** - 从 `env_settings.yml` 加载配置，改进项目检测，输出格式更整洁。[#214](https://github.com/Datus-ai/Datus-agent/issues/214) [#216](https://github.com/Datus-ai/Datus-agent/issues/216) [文档](https://docs.datus.ai/zh/metricflow/introduction/)
- **灵活的模型配置** - 在 agent 配置中支持多个模型 provider 与 specification。[#195](https://github.com/Datus-ai/Datus-agent/issues/195)
- **CLI 展示改进** - 优化表格宽度渲染，SQL 查询可读性更好。[#200](https://github.com/Datus-ai/Datus-agent/issues/200)
- **改进的初始化流程** - 增强 `datus-agent init` 的错误处理与初始化流程。[#194](https://github.com/Datus-ai/Datus-agent/issues/194)

**依赖变更**

- `openai-agents` 升级到 0.3.2（需手动更新：`pip install -U openai-agents`）
- `datus-metricflow` 更新到 0.1.2

### 0.2.0

**增强的 Chat 功能**

- 高级多轮对话，体验更流畅。[#91](https://github.com/Datus-ai/Datus-agent/issues/91)
- agentic 执行数据库工具、文件系统操作，并自动生成 to-do list。
- 支持自动与手动 compaction（.compact）。[#125](https://github.com/Datus-ai/Datus-agent/issues/125)
- 会话管理，支持 .resume 与 .clear 命令。
- 通过 @ Table、@ file、@ metrics、@sql_history 命令注入专门的上下文。[#134](https://github.com/Datus-ai/Datus-agent/issues/134) [#152](https://github.com/Datus-ai/Datus-agent/issues/152)
- Token 消耗跟踪与预估，资源使用更可见。[#119](https://github.com/Datus-ai/Datus-agent/issues/119)
- 执行敏感 tool action 前进行写操作确认。
- Plan Mode：AI 辅助规划，生成并管理 to-do list。[#147](https://github.com/Datus-ai/Datus-agent/issues/147)

**自动构建知识库**

- 从历史 success story 自动生成 MetricFlow 格式的 Metric YAML 文件。[#10](https://github.com/Datus-ai/Datus-agent/issues/10)
- 自动从 workspace 的 *.sql 文件总结、标注 SQL history 文件。[#132](https://github.com/Datus-ai/Datus-agent/issues/132)
- 使用 metrics 与 SQL history 提升 SQL 准确率与生成速度。

**MCP 扩展**

- 新增 .mcp 命令，可添加、删除、列出与调用 MCP server 与 tool。[#54](https://github.com/Datus-ai/Datus-agent/issues/54)

**灵活的 Workflow 配置**

- 通过 agent.yml 完整自定义 workflow 定义。
- 可配置的 node、模型与数据库连接。
- 支持 sub-workflow 与结果选择，提升准确率。[#88](https://github.com/Datus-ai/Datus-agent/issues/88)

**上下文探索**

- 改进 @catalogs，展示多个数据库下的所有 database、schema、table。
- 新增 @subject,展示用 MetricFlow 构建的所有 metrics。[#165](https://github.com/Datus-ai/Datus-agent/issues/165)
- 集成 context search 工具，提升元数据与 metrics 的召回。[#138](https://github.com/Datus-ai/Datus-agent/issues/138)

**用户行为日志**

- 自动采集用户行为日志。
- 把人机交互数据转化为可训练的数据集，用于后续改进。


## 0.1

### 0.1.0

**Datus-cli**

- 支持连接 SQLite、DuckDB、StarRocks 与 Snowflake，并执行常见的命令行操作。
- 支持三类命令扩展：!run_command、@context 与 /chat，提升开发效率。

**Datus-agent**

- 支持基于 React 范式的自动 NL2SQL 生成。
- 支持检索数据库元数据并基于元数据构建向量检索。
- 支持通过 MCP server 进行深度推理。
- 支持接入 bird-dev 与 spider2-snow benchmark。
- 支持保存与恢复 workflow，可记录执行上下文与 node 输入输出。
- 提供灵活配置：可在 Agent.yaml 中定义多个模型、数据库与 node 执行策略。

### 0.1.2

**Datus-cli**

- 新增 fix node，使用 !fix 快速修复上一条出错的 SQL，模板让 LLM 专注此任务。

**Datus-agent**

- 多线程优化 bootstrap-kb 性能。
- 其他细节 bug 修复。

### 0.1.3

**Datus-cli**

- 新增 datus-init 初始化 ~/.datus/ 目录。
- 在 ~/.datus/sample 中提供 DuckDB 示例数据库。

**Datus-agent**

- 在 output node 中新增 check_result 选项（默认 False）。

### 0.1.4

**Datus-agent**

- 新增 check-mcp 命令，用于确认 MCP server 的配置与可用性。
- 支持 DuckDB 与 SQLite MCP server。
- 实现 MCP server 自动安装到 datus-mcp 目录。

### 0.1.5

**Datus-agent**

- 自动化语义层生成。
- 新增内部 workflow：metrics2SQL。
- 新增 save_llm_trace,便于收集训练数据集。

**Datus-cli**

- 增强 !reason 与 !gen_semantic_model 命令，体验更 agentic、更直观。
