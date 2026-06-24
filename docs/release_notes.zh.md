# 发布说明

## 0.3

### 0.3.5

**新功能**

- **SQL Policy Framework** - 新增请求级 SQL 读查询策略框架，支持从 API 请求的 `X-Datus-Principal` 读取结构化调用方身份，并在数据库读查询执行前调用自定义 provider 改写或拒绝 SQL；改写后的 SQL 会再次经过只读校验，生产部署可接入自有鉴权或策略服务。[#1020](https://github.com/Datus-ai/Datus-agent/pull/1020) [#1028](https://github.com/Datus-ai/Datus-agent/pull/1028) [文档](https://docs.datus.ai/0.3/zh/configuration/sql_policy/)
- **Strict OSI 语义创作路径** - `gen_semantic_model` / `gen_metrics` 现可按 OSI adapter 自动进入严格 OSI authoring 模式，生成 OSI core YAML，通过 adapter 校验、dry-run 后发布，并把可查询指标同步回 Knowledge Base，避免 MetricFlow-only 字段泄漏到源语义模型中。[#1007](https://github.com/Datus-ai/Datus-agent/pull/1007) [文档](https://docs.datus.ai/0.3/zh/adapters/osi_semantic_adapter/)
- **Metric Preview API** - 新增已保存指标的维度发现与 SQL 预览接口，SaaS metric editor 可在 UI 中列出可查询维度、dry-run 编译指标 SQL，并用结构化 preflight 错误提示不兼容维度组合。[#992](https://github.com/Datus-ai/Datus-agent/pull/992)
- **轻量化 `/init` + 全新 `/build-kb`** - `/init` 现在只做快速项目扫描、`AGENTS.md` 清单和文件类 knowledge/memory 写入；重型向量知识库构建迁移到新的 `/build-kb` 命令，支持按文件、表、datasource 或业务域限定范围，并通过内置 `storage-classify` skill 路由 semantic models、metrics、reference SQL、knowledge、memory、skills 和 `AGENTS.md`。[#997](https://github.com/Datus-ai/Datus-agent/pull/997) [#1022](https://github.com/Datus-ai/Datus-agent/pull/1022) [文档](https://docs.datus.ai/0.3/zh/cli/build_kb_command/)

**增强**

- **系统 Prompt 前缀缓存** - AgenticNode 在 session 首次 LLM 调用时快照系统 prompt，后续轮次复用同一前缀以稳定命中 Anthropic ephemeral cache / OpenAI prompt cache；当前 datasource 等运行时信息改走 user turn 的 `<system_reminder>` 注入，切换模型时自动重建快照。[#996](https://github.com/Datus-ai/Datus-agent/pull/996)
- **AskMetrics 查询流升级** - `ask_metrics` 支持可选的最终结果选择、完整 `query_metrics` 结果缓存和压缩预览展示，并能在同环比问题中自动扩展当前值、上一周期值和差值指标，提升 benchmark 与复杂指标问答的稳定性。[#1005](https://github.com/Datus-ai/Datus-agent/pull/1005) [文档](https://docs.datus.ai/0.3/zh/subagent/ask_metrics/)
- **同环比指标生成与查询** - `gen_metrics` 可从 LAG SQL 自动生成 `offset_window` 派生指标，`ask_metrics` 会按 offset metadata 匹配正确时间粒度维度，减少因粒度不匹配导致的首期数据缺失。[#989](https://github.com/Datus-ai/Datus-agent/pull/989)
- **非交互式 Plan Mode** - `datus -p` 新增 `--plan-mode`，print mode 与 benchmark 场景中计划生成后可自动确认执行，不再因等待人工确认而卡住。[#993](https://github.com/Datus-ai/Datus-agent/pull/993)

**Bug 修复**

- **并发权限弹窗不再互相阻塞** - 权限 prompt 锁从 event loop 作用域改为 broker 作用域，独立会话和 sub-agent 在同一 API worker 上不会再互相卡住，同时保留单次运行内一次只展示一个权限弹窗的行为。[#1035](https://github.com/Datus-ai/Datus-agent/pull/1035)
- **交互式 API 会话保持在线** - 当 `AgentConfig` fingerprint 变化但旧 `DatusService` 仍有活跃任务时，缓存不再提前替换实例，避免用户回复 `/chat/user_interaction` 时命中新的空 task manager 并返回 `SESSION_NOT_FOUND`。[#1032](https://github.com/Datus-ai/Datus-agent/pull/1032)
- **Claude 原生 SDK 路径完整执行 Hook 生命周期** - Claude 订阅 OAuth token 路径现在与 OpenAI Agents SDK Runner 路径一致，会完整触发 permission、compact、generation 等 hook，不再绕过权限策略或生命周期回调。[#980](https://github.com/Datus-ai/Datus-agent/pull/980)
- **指标 queryability 与预览更稳** - 修复 metric preview 返回 datasource id 而不是物理 database、queryability publish gate 不接受 canonical `metric_time` evidence、period-offset metric discovery 不稳定，以及 offset-derived metric bootstrap 依赖 authoring format 的问题。[#1019](https://github.com/Datus-ai/Datus-agent/pull/1019) [#1018](https://github.com/Datus-ai/Datus-agent/pull/1018) [#1017](https://github.com/Datus-ai/Datus-agent/pull/1017) [#1011](https://github.com/Datus-ai/Datus-agent/pull/1011)
- **Anthropic SDK 认证头冲突** - 修复环境变量凭证回退时 Anthropic SDK 发送双重 `Authorization` header 的问题。[#991](https://github.com/Datus-ai/Datus-agent/pull/991)
- **Python 3.12 CLI 帮助显示恢复可读** - 将 `DBType`、`LLMProvider`、`EmbeddingProvider`、`SQLType` 迁移为 `StrEnum`，避免 argparse 在 Python 3.12 下把枚举候选值显示成原始类名。[#1001](https://github.com/Datus-ai/Datus-agent/pull/1001)
- **非 DB 节点也能拿到当前日期** - 日期注入从 datasource catalog 注入中拆出，`gen_visual_report`、`skill_creator` 等没有数据库工具的节点也能在 system prompt 中获得当前日期上下文。[#1026](https://github.com/Datus-ai/Datus-agent/pull/1026)
- **旧版 JSON 数组用户消息可正确解析** - `extract_user_input()` 现在能识别旧版本保存的 JSON 数组消息内容，Web 侧边栏不再直接显示原始 JSON 字符串。[#888](https://github.com/Datus-ai/Datus-agent/pull/888)
- **API 关闭时不再丢弃后台任务** - 新增统一 background task 注册与 drain 机制，FastAPI lifespan shutdown 前会等待 fire-and-forget 任务完成并记录异常，同时补齐 `DatusServiceCache.shutdown()` 调用。[#1003](https://github.com/Datus-ai/Datus-agent/pull/1003)
- **CLI 退出不再触发 RuntimeWarning** - 修复 prompt_toolkit teardown 期间 `run_in_terminal_sync()` 创建的 coroutine 被 loop close 丢弃导致的 `RuntimeWarning: coroutine was never awaited`。[#952](https://github.com/Datus-ai/Datus-agent/pull/952)
- **DBManager 连接状态更健壮** - `DBManager` 现在能容忍被置空的 `_conn_dict` 条目，并在 `list_databases` 失败时记录 traceback，减少长会话或异常恢复中的连接状态崩溃。[#994](https://github.com/Datus-ai/Datus-agent/pull/994)

### 0.3.4

**新功能**

- **AskMetrics 子代理** - 面向 KPI、趋势、group-by 和 attribution 问题的专属 metric QA agent，会优先使用 metric 专用工具与提示词而非裸 SQL，并支持自定义路由、模板和工具。[#954](https://github.com/Datus-ai/Datus-agent/pull/954) [文档](https://docs.datus.ai/0.3/zh/subagent/ask_metrics/)
- **OpenRouter Provider** - 用一个 `OPENROUTER_API_KEY` 即可访问完整的 vendor/model 目录；`/model` 选择器新增搜索过滤，并隐藏 Anthropic Claude 的非 canonical `-fast` 别名。[#973](https://github.com/Datus-ai/Datus-agent/pull/973) [#936](https://github.com/Datus-ai/Datus-agent/pull/936) [文档](https://docs.datus.ai/0.3/zh/cli/model_command/)
- **自更新命令** - `datus upgrade` / `datus update` 可在 CLI 中检查并安装 `datus-*` 包的新版本；交互式会话启动时会提示可用新版本，`--check` 则只查看不安装。[#949](https://github.com/Datus-ai/Datus-agent/pull/949)
- **Print 模式 Orchestrator 工具** - `--orchestrator-tools` 让外部 orchestrator 通过代理工具请求 issue 评论、状态更新、人工输入、blocked 标记和 mission 完成，且每个工具的 strict JSON schema 都被完整保留。[#950](https://github.com/Datus-ai/Datus-agent/pull/950)

**增强**

- **默认可查询的指标** - 指标生成现在会提取一份 queryability contract，并用 `query_metrics(dry_run=True)` 校验，因此不再产出结构合法、却无法按原始粒度查询的指标。[#943](https://github.com/Datus-ai/Datus-agent/pull/943) [#962](https://github.com/Datus-ai/Datus-agent/pull/962)
- **隔离的 Agent Memory** - 每个 agent 拥有一份独立的 2000 字节 `MEMORY.md`，仅能通过 `add_memory` / `edit_memory` 写入；sub-agent 只以只读方式继承。[#975](https://github.com/Datus-ai/Datus-agent/pull/975) [文档](https://docs.datus.ai/0.3/zh/integration/memory/)
- **内置知识抽取** - 外部知识生成从旧的 `ext_knowledge` 向量子系统切换为内置的 `extract-knowledge` skill，精简了遗留存储与工具面，同时保留 knowledge-base API。[#932](https://github.com/Datus-ai/Datus-agent/pull/932)
- **更强的 Snowflake 支持** - Snowflake 现支持 inline PEM 私钥、拒绝不支持的 catalog 参数，并在 DB 与 MetricFlow adapter 中规范化 time-grain 查询；文档也明确 password 与私钥二选一、warehouse 必填，且通常不应设置 catalog。[#937](https://github.com/Datus-ai/Datus-agent/pull/937) [datus-db-adapters#70](https://github.com/Datus-ai/datus-db-adapters/pull/70) [datus-db-adapters#72](https://github.com/Datus-ai/datus-db-adapters/pull/72) [datus-semantic-adapter#27](https://github.com/Datus-ai/datus-semantic-adapter/pull/27) [datus-semantic-adapter#28](https://github.com/Datus-ai/datus-semantic-adapter/pull/28) [datus-semantic-adapter#29](https://github.com/Datus-ai/datus-semantic-adapter/pull/29) [文档](https://docs.datus.ai/0.3/zh/configuration/datasources/)
- **统一的 `gen_sql` 命名** - SQL 生成节点、工作流、配置、CLI 命令（`/gen_sql`）、prompt 模板和 sub-agent 名称统一为同一套 `gen_sql` 命名；使用旧 `generate_sql` / `sql_system` 配置的项目需要迁移。[#935](https://github.com/Datus-ai/Datus-agent/pull/935) [文档](https://docs.datus.ai/0.3/zh/configuration/nodes/)
- **可配置的指标批量大小** - `bootstrap-kb --components metrics` 新增 `--metrics-batch-size`；需要逐条 success-story provenance 时设为 `1`，否则保留默认值 `5` 以维持原有吞吐。[#976](https://github.com/Datus-ai/Datus-agent/pull/976)

**Bug 修复**

- **按 Datasource 隔离的指标 Bootstrap** - 多 datasource 项目中指标 bootstrap 现按 datasource 做行级隔离，`overwrite` 不再删除其他 datasource 的 KB 数据，同时加固了 MetricFlow YAML merge、批次刷新、最终指标去重和 CLI 长输出截断。[#974](https://github.com/Datus-ai/Datus-agent/pull/974) [文档](https://docs.datus.ai/0.3/zh/qa/metric-bootstrap-generation-qa/)
- **更稳健的语义指标查询与校验** - Snowflake 表坐标现能正确进入 semantic model 生成提示，多指标维度兼容性有 preflight 与拆分建议，临时 YAML 校验不再扫描无关父目录，非字符串/可空 PyArrow 结果也能安全展示。[#941](https://github.com/Datus-ai/Datus-agent/pull/941) [#946](https://github.com/Datus-ai/Datus-agent/pull/946)
- **跨 Datasource 的 Subject 节点迁移** - 升级旧版 SQLite 存储时会迁移陈旧的 `subject_nodes` UNIQUE 约束，使同名 subject 节点可在不同 datasource 间共存，同时保持同一 datasource 内的去重。[#964](https://github.com/Datus-ai/Datus-agent/pull/964)
- **一个 Datasource 路由到多个 Database** - 单个 datasource 现可路由到多个 database——文件 glob datasource、benchmark 任务、`/database` 切换、DB 工具和节点执行都按 `(datasource, database)` 选择连接，benchmark SQL 任务也可显式指定 datasource。[#961](https://github.com/Datus-ai/Datus-agent/pull/961) [#934](https://github.com/Datus-ai/Datus-agent/pull/934) [datus-db-adapters#71](https://github.com/Datus-ai/datus-db-adapters/pull/71)

### 0.3.3

**新功能**

- **会话自动压缩** - 长对话现在可以自动整理上下文，减少因上下文占满而中断的情况。对话变长时，较早的工具调用记录会归档到磁盘，当前任务仍保留在上下文中；接近上限（约 90%）时，会生成会话摘要并继续运行。`/compact` 也新增了进度提示和摘要面板，完整历史会持久化保存，方便恢复和回溯。[#871](https://github.com/Datus-ai/Datus-agent/pull/871) [#919](https://github.com/Datus-ai/Datus-agent/pull/919) [#933](https://github.com/Datus-ai/Datus-agent/pull/933) [文档](https://docs.datus.ai/0.3/zh/configuration/compact/)
- **实时 Token 用量** - 模型响应时会实时显示本次新增与累计 token 数；状态栏同步展示当前上下文占用和总容量。用量历史会随会话保存，恢复会话或审计成本时都能查看。[#920](https://github.com/Datus-ai/Datus-agent/pull/920)

**增强**

- **Snowflake 密钥对认证** - Snowflake 启用 MFA 后，可改用 RSA 密钥对认证，不必依赖密码连接。配置 `private_key_file`、可选的 `private_key_file_pwd` 和 `role` 后，同一套认证信息会用于 SQL 执行，以及 MetricFlow 指标生成、校验和查询；日志中的凭据会自动脱敏。[#926](https://github.com/Datus-ai/Datus-agent/pull/926) [datus-db-adapters#66](https://github.com/Datus-ai/datus-db-adapters/pull/66) [datus-db-adapters#67](https://github.com/Datus-ai/datus-db-adapters/pull/67) [datus-semantic-adapter#25](https://github.com/Datus-ai/datus-semantic-adapter/pull/25) [文档](https://docs.datus.ai/0.3/zh/configuration/datasources/)
- **离线 Embedding 降级** - 离线、内网或无法访问 Hugging Face 时，Datus 不再卡在 embedding 模型下载上。上下文搜索和 `@` 引用补全会临时降级，并提示模型名、缓存路径、相关环境变量和修复方法；数据库工具和普通聊天仍可继续使用。文档也补充了预先缓存模型、或改用 OpenAI 兼容 embedding 服务的离线部署方案。[#870](https://github.com/Datus-ai/Datus-agent/pull/870)
- **Codex 缓存策略优化** - 使用 ChatGPT 订阅版 Codex 后端时，多步运行现在可以像官方客户端一样复用 prompt 缓存，降低等待时间和 token 成本。[#918](https://github.com/Datus-ai/Datus-agent/pull/918)
- **`datus -p` 与 `--resume` 支持多轮对话** - Print 模式现在也能恢复并继续指定会话。通过 `datus -p '...' --resume <session_id>` 即可在命令行中接续对话；REPL 与 API chat 也支持用 `--resume` 恢复既有会话。[#914](https://github.com/Datus-ai/Datus-agent/pull/914)

**Bug 修复**

- **自定义 Ask 子代理遵守工具白名单** - `ask_report` / `ask_dashboard` 现在只会看到子代理 `tools` 白名单中允许的工具。未授权的数据库、bash、skill 工具不再暴露给模型，prompt 也不再混入完整聊天工具目录，减少越权调用和上下文污染风险。[#877](https://github.com/Datus-ai/Datus-agent/pull/877) [#878](https://github.com/Datus-ai/Datus-agent/pull/878) [#881](https://github.com/Datus-ai/Datus-agent/pull/881)

### 0.3.2

**新功能**

- **Agent 可观测性（可配置追踪）** - 在 `agent.observability.tracing` 中开启后，运行追踪即可导出到 Langfuse、LangSmith、Datadog、Braintrust 或通用 OTLP collector。可按需采集运行内容并对敏感信息脱敏，同时生成稳定的追踪引用、在运行期对追踪分组，把 benchmark、bootstrap、CLI、chat 等运行归到同一条链路下，便于关联与排查。[#833](https://github.com/Datus-ai/Datus-agent/pull/833) [#864](https://github.com/Datus-ai/Datus-agent/pull/864) [文档](https://docs.datus.ai/0.3/zh/develop/observability/)
- **Visual Artifact 升级：支持 Dashboard** - 新增 `gen_visual_dashboard`，可在 Chat 里像 `gen_visual_report` 一样直接生成 dashboard 风格的 HTML 看板（多图表卡片布局），并可由 **本地 `datus --web` 直接预览**——切换筛选器即按当前条件重跑 SQL、实时刷新，无需 SaaS 后端。报告与看板的生成质量也明显提升：版面更整齐、图表渲染更稳定（多图表并发查询不再触发 DuckDB 竞态）、图表组件统一使用运行时 ChartCard、数据查询更准确，能应对更复杂的报告并支持多轮修改。生成完成后，HTML 的绝对路径会输出到 CLI 消息流中，关闭浏览器标签后仍可凭路径再次打开。[#829](https://github.com/Datus-ai/Datus-agent/pull/829) [#835](https://github.com/Datus-ai/Datus-agent/pull/835) [#842](https://github.com/Datus-ai/Datus-agent/pull/842) [#847](https://github.com/Datus-ai/Datus-agent/pull/847) [#848](https://github.com/Datus-ai/Datus-agent/pull/848) [#849](https://github.com/Datus-ai/Datus-agent/pull/849) [#853](https://github.com/Datus-ai/Datus-agent/pull/853) [#855](https://github.com/Datus-ai/Datus-agent/pull/855) [#863](https://github.com/Datus-ai/Datus-agent/pull/863) [#866](https://github.com/Datus-ai/Datus-agent/pull/866) [#867](https://github.com/Datus-ai/Datus-agent/pull/867) [#869](https://github.com/Datus-ai/Datus-agent/pull/869) [#894](https://github.com/Datus-ai/Datus-agent/pull/894) [#895](https://github.com/Datus-ai/Datus-agent/pull/895) [#901](https://github.com/Datus-ai/Datus-agent/pull/901) [#905](https://github.com/Datus-ai/Datus-agent/pull/905) [#907](https://github.com/Datus-ai/Datus-agent/pull/907) [文档](https://docs.datus.ai/0.3/zh/subagent/gen_visual_dashboard/)
- **运行中追加指令** - agent 还在流式输出时，你可以在 CLI / TUI 继续输入、或通过 API 发送新的指令；它会在模型的下一步被读到并写入会话记录，界面上也会实时显示这条插入——不用打断当前运行。[#824](https://github.com/Datus-ai/Datus-agent/pull/824)

**增强**

- **语义 SQL 指标提取** - 从历史 SQL 提取指标时，能区分新建指标、基于已有指标计算的指标、以及对已有指标的直接引用，从而避免重复创建，并保留时间粒度、筛选条件、字面量等信息。支持的指标类型包括计数、去重计数、求和、平均、最大 / 最小、条件统计、比率、表达式、累计和派生；遇到跨表、非等值关联、合并等多表场景时，会先把数据组合成一个数据源，再在其上定义指标。已在真实数仓（StarRocks）端到端验证：生成指标的取数结果与原始 SQL 完全一致。[#811](https://github.com/Datus-ai/Datus-agent/pull/811)
- **CLI 聊天显示更顺滑** - 修复使用 Claude 原生接口时终端里最后一段文字重复刷新的问题（并修复会话恢复时的相关解析）；同时统一了聊天记录的显示，历史消息、恢复 / 回退、运行中插话都呈现一致，用户消息以带边框面板更清晰区分。[#837](https://github.com/Datus-ai/Datus-agent/pull/837) [#852](https://github.com/Datus-ai/Datus-agent/pull/852)

**Bug 修复**

- **语义模型可分阶段校验** - 在还没生成指标时就能单独校验语义模型；此时预期内的「暂无指标」不再中断流程，真正的模型错误仍会被拦下。[#827](https://github.com/Datus-ai/Datus-agent/pull/827) [#850](https://github.com/Datus-ai/Datus-agent/pull/850)
- **Bootstrap 正确识别生成结果** - bootstrap 流程能正确识别各步骤的生成结果（包括失败），不再漏掉成功结果，也不再悄悄吞掉失败。[#831](https://github.com/Datus-ai/Datus-agent/pull/831)
- **参考 SQL 摘要路径解析** - 生成参考 SQL 摘要时，文件路径在各种写法下都能正确解析；越界路径会被安全跳过，而不再报错中断。[#840](https://github.com/Datus-ai/Datus-agent/pull/840)
- **Print 模式不再被权限弹窗挂住** - `datus -p`（非交互的脚本 / CI 模式）现在走 workflow 执行模式，与 `/bootstrap` 等非交互流程一致；原本会挂起等待人工答复的权限 ASK/EXTERNAL 弹窗现在直接短路返回，跑批不会再卡住。[#891](https://github.com/Datus-ai/Datus-agent/pull/891)

### 0.3.1

**新功能**

- **HTML 报告生成** - 新增 `gen_visual_report` subagent，把一个问题、一个指标引用或一条 SQL 直接变成自包含的 HTML 报告（KPI 卡片、图表、表格、叙事段落齐全），并支持按章节单独修改，可针对单个图表微调而无需重写整篇报告。[#783](https://github.com/Datus-ai/Datus-agent/pull/783) [#821](https://github.com/Datus-ai/Datus-agent/pull/821) [文档](https://docs.datus.ai/0.3/zh/subagent/gen_visual_report/)
- **Plan Mode 持久化** - Plan Mode 现在会把 `plan.md` 落盘并在会话恢复时一并恢复，CLI 中途关闭也不会丢失正在制定的计划。[#772](https://github.com/Datus-ai/Datus-agent/pull/772) [文档](https://docs.datus.ai/0.3/zh/cli/plan_mode/)
- **CLI / TUI 体验升级** - 新增实时 todo sidebar 直观跟踪任务进度，配合内联命令向导、滚动回查搜索、鼠标拖选复制和可拖拽 scrollbar，使终端交互更接近原生体验。[#772](https://github.com/Datus-ai/Datus-agent/pull/772)

**增强**

- **`/permission` 命令** - 将 `/profile` 更名为 `/permission`，支持 `normal` / `auto` / `dangerous` 三档模式，适配不同的开发场景。[#769](https://github.com/Datus-ai/Datus-agent/pull/769) [文档](https://docs.datus.ai/0.3/zh/cli/reference/)
- **自定义 Subagent 管理** - 自定义 subagent 支持通过 API 或 TUI 删除；不同 agent 类型的可用工具改由后端统一返回，SaaS 与 standalone UI 创建和编辑 subagent 的行为保持一致。[#807](https://github.com/Datus-ai/Datus-agent/pull/807) [#812](https://github.com/Datus-ai/Datus-agent/pull/812) [文档](https://docs.datus.ai/0.3/zh/subagent/customized_subagent/)
- **按次指定权限模式** - Chat 请求支持按次指定 `normal` / `auto` / `dangerous` 权限模式，多租户 SaaS 部署中不会再互相污染共享默认配置。[#822](https://github.com/Datus-ai/Datus-agent/pull/822) [文档](https://docs.datus.ai/0.3/zh/integration/skills/)

**Bug 修复**

- **Claude / Anthropic 参数冲突** - 修复 Claude / Anthropic 路由下同时传递 `temperature` 与 `top_p` 时请求失败的问题。[#817](https://github.com/Datus-ai/Datus-agent/pull/817)
- **缺少 Subject Path 时的 Metric ID 冲突** - 修复 metric id 未包含 `subject_path` 时，不同 subject 树下同名 metric 出现冲突的问题。[#819](https://github.com/Datus-ai/Datus-agent/pull/819)

### 0.3.0

**新功能**

***Datus API***

- **FastAPI REST API** - 基于 FastAPI 的 REST API，包含 service/model 分层、CLI 入口、流式 Chat、任务追踪、SQL 执行停止、多选 `ask_user`、success story 持久化、知识库 bootstrap API，以及 API 请求侧的 proxy source / interactive mode 控制。[#520](https://github.com/Datus-ai/Datus-agent/pull/520) [#538](https://github.com/Datus-ai/Datus-agent/pull/538) [#539](https://github.com/Datus-ai/Datus-agent/pull/539) [#551](https://github.com/Datus-ai/Datus-agent/pull/551) [#553](https://github.com/Datus-ai/Datus-agent/pull/553) [#555](https://github.com/Datus-ai/Datus-agent/pull/555) [#606](https://github.com/Datus-ai/Datus-agent/pull/606) [#610](https://github.com/Datus-ai/Datus-agent/pull/610) [文档](https://docs.datus.ai/0.3/zh/API/introduction/)
- **模型发现 API** - 模型发现、单请求模型覆盖、current model 元数据和 ISO-8601 UTC 时间戳格式。[#643](https://github.com/Datus-ai/Datus-agent/pull/643) [#649](https://github.com/Datus-ai/Datus-agent/pull/649) [#700](https://github.com/Datus-ai/Datus-agent/pull/700) [文档](https://docs.datus.ai/0.3/zh/API/models/)
- **图表推荐与可视化 API** - 支持 Datus Chat 与外部应用生成 dashboard-ready 的可视化结果。[#545](https://github.com/Datus-ai/Datus-agent/pull/545) [文档](https://docs.datus.ai/0.3/zh/getting_started/dashboard_copilot/)

***Datus Chat 与 IM 网关***

- **Datus Chat（FastAPI 替换 Streamlit）** - 用 FastAPI + `@datus/web-chatbot` 替换旧的 Streamlit chatbot，并新增 Datus Chat 模块。[#543](https://github.com/Datus-ai/Datus-agent/pull/543) [#554](https://github.com/Datus-ai/Datus-agent/pull/554) [文档](https://docs.datus.ai/0.3/zh/web_chatbot/introduction/)
- **Slack 与 Feishu/Lark 网关** - 新增 IM 网关，支持 channel 配置、daemon mode、流式回复、feedback action，并将 `datus-claw` 统一改名为 `datus-gateway`。[#559](https://github.com/Datus-ai/Datus-agent/pull/559) [#562](https://github.com/Datus-ai/Datus-agent/pull/562) [#565](https://github.com/Datus-ai/Datus-agent/pull/565) [#616](https://github.com/Datus-ai/Datus-agent/pull/616) [#623](https://github.com/Datus-ai/Datus-agent/pull/623) [#593](https://github.com/Datus-ai/Datus-agent/pull/593) [文档](https://docs.datus.ai/0.3/zh/gateway/introduction/)

***项目与工作区配置***

- **项目感知的 Configure/Init 流程** - 将 setup 拆为项目感知的 configure/init 流程，新增项目级 `.datus/config.yml`、项目级 memory、自动 datasource/service setup，以及重建后的 `.datus` 目录结构。[#542](https://github.com/Datus-ai/Datus-agent/pull/542) [#578](https://github.com/Datus-ai/Datus-agent/pull/578) [#592](https://github.com/Datus-ai/Datus-agent/pull/592) [#608](https://github.com/Datus-ai/Datus-agent/pull/608) [文档](https://docs.datus.ai/0.3/zh/cli/init_command/)
- **统一运行时服务配置** - 围绕 `services.datasources`、`services.bi_platforms`、semantic layer、scheduler 建模；CLI 统一使用 `--datasource`。[#614](https://github.com/Datus-ai/Datus-agent/pull/614) [#633](https://github.com/Datus-ai/Datus-agent/pull/633) [#636](https://github.com/Datus-ai/Datus-agent/pull/636) [#642](https://github.com/Datus-ai/Datus-agent/pull/642) [文档](https://docs.datus.ai/0.3/zh/configuration/datasources/)
- **一行安装脚本** - 新增 Linux/macOS `curl | sh` 一行安装脚本，并更新 quickstart 与 service 文档。[#613](https://github.com/Datus-ai/Datus-agent/pull/613) [#611](https://github.com/Datus-ai/Datus-agent/pull/611) [#667](https://github.com/Datus-ai/Datus-agent/pull/667) [文档](https://docs.datus.ai/0.3/zh/getting_started/Quickstart/)

***CLI 体验***

- **统一 `/` 命令前缀** - 将交互命令统一到 `/` 前缀，新增 `/model`、`/skill`、`/mcp`、`/agent`、`/subagent`、交互式输入和流式 `/bootstrap` TUI。[#621](https://github.com/Datus-ai/Datus-agent/pull/621) [#635](https://github.com/Datus-ai/Datus-agent/pull/635) [#650](https://github.com/Datus-ai/Datus-agent/pull/650) [#655](https://github.com/Datus-ai/Datus-agent/pull/655) [#656](https://github.com/Datus-ai/Datus-agent/pull/656) [#659](https://github.com/Datus-ai/Datus-agent/pull/659) [#683](https://github.com/Datus-ai/Datus-agent/pull/683) [文档](https://docs.datus.ai/0.3/zh/cli/reference/)
- **`/language` 与 `/effort` 命令** - 用 `/language` 固定响应语言，`/effort` 控制 reasoning 强度，以及 `/<service>.<method>` 只读服务调用分发。[#641](https://github.com/Datus-ai/Datus-agent/pull/641) [#653](https://github.com/Datus-ai/Datus-agent/pull/653) [#631](https://github.com/Datus-ai/Datus-agent/pull/631) [文档](https://docs.datus.ai/0.3/zh/cli/language_command/)
- **CLI Print Mode 与体验优化** - CLI print mode、proxy tools、重做底部状态栏、固定 streaming/tool 状态行、改进 markdown streaming，并恢复 `@` reference 自动补全。[#489](https://github.com/Datus-ai/Datus-agent/pull/489) [#501](https://github.com/Datus-ai/Datus-agent/pull/501) [#583](https://github.com/Datus-ai/Datus-agent/pull/583) [#586](https://github.com/Datus-ai/Datus-agent/pull/586) [#654](https://github.com/Datus-ai/Datus-agent/pull/654) [#664](https://github.com/Datus-ai/Datus-agent/pull/664) [#661](https://github.com/Datus-ai/Datus-agent/pull/661) [#662](https://github.com/Datus-ai/Datus-agent/pull/662) [文档](https://docs.datus.ai/0.3/zh/cli/introduction/)
- **新增模型与订阅计划** - Codex OAuth、Claude Subscription、Coding Plan、OpenRouter、MiniMax、GLM、BigModel、Z.AI 等模型/计划支持，并重建 provider-based 模型配置和 provider catalog。[#487](https://github.com/Datus-ai/Datus-agent/pull/487) [#635](https://github.com/Datus-ai/Datus-agent/pull/635) [#687](https://github.com/Datus-ai/Datus-agent/pull/687) [#693](https://github.com/Datus-ai/Datus-agent/pull/693) [文档](https://docs.datus.ai/0.3/zh/cli/model_command/)
- **权限 Profile** - 新增 `normal` / `auto` / `dangerous` 权限 profile，支持 subagent-aware permission hooks，并放宽正常模式下的安全发现类工具。[#646](https://github.com/Datus-ai/Datus-agent/pull/646) [#652](https://github.com/Datus-ai/Datus-agent/pull/652) [文档](https://docs.datus.ai/0.3/zh/integration/skills/)

***数据工程 Subagent 与 Skills***

- **数据工程 Agents 与 Skills** - 新增跨库迁移、ETL/job 生成、scheduler workflow、表生成、dashboard 生成、BI/scheduler 编排等内置 agent 与技能。[#494](https://github.com/Datus-ai/Datus-agent/pull/494) [#525](https://github.com/Datus-ai/Datus-agent/pull/525) [#564](https://github.com/Datus-ai/Datus-agent/pull/564) [#575](https://github.com/Datus-ai/Datus-agent/pull/575) [#580](https://github.com/Datus-ai/Datus-agent/pull/580) [#639](https://github.com/Datus-ai/Datus-agent/pull/639) [文档](https://docs.datus.ai/0.3/zh/subagent/builtin_subagents/)
- **交付物 Validation Hook** - 表交付物 validation hook、共享 deliverable node、validation skills，以及 semantic/metric generation 的 publish gate。[#657](https://github.com/Datus-ai/Datus-agent/pull/657) [#663](https://github.com/Datus-ai/Datus-agent/pull/663) [#665](https://github.com/Datus-ai/Datus-agent/pull/665) [文档](https://docs.datus.ai/0.3/zh/integration/validation/)
- **自然语言指标与 Skill Creator** - 新增自然语言指标创建、wheel 内置 skills 打包、skill frontmatter scope，以及用于交互式 skill 创作的 `skill-creator` subagent。[#504](https://github.com/Datus-ai/Datus-agent/pull/504) [#526](https://github.com/Datus-ai/Datus-agent/pull/526) [#627](https://github.com/Datus-ai/Datus-agent/pull/627) [#645](https://github.com/Datus-ai/Datus-agent/pull/645) [#676](https://github.com/Datus-ai/Datus-agent/pull/676) [文档](https://docs.datus.ai/0.3/zh/subagent/customized_subagent/)

***记忆与 Reference Template***

- **Auto Memory** - 基于 `MEMORY.md` 的 Auto Memory、emergent topic tree、空 memory prompt、project/session isolation。[#498](https://github.com/Datus-ai/Datus-agent/pull/498) [#620](https://github.com/Datus-ai/Datus-agent/pull/620) [#595](https://github.com/Datus-ai/Datus-agent/pull/595) [#523](https://github.com/Datus-ai/Datus-agent/pull/523) [#594](https://github.com/Datus-ai/Datus-agent/pull/594) [文档](https://docs.datus.ai/0.3/zh/integration/memory/)
- **Reference Template** - 新增 Reference Template 机制，并修复 bootstrap 中 reference template 解析问题。[#508](https://github.com/Datus-ai/Datus-agent/pull/508) [#574](https://github.com/Datus-ai/Datus-agent/pull/574) [#677](https://github.com/Datus-ai/Datus-agent/pull/677) [文档](https://docs.datus.ai/0.3/zh/knowledge_base/reference_template/)

***生态与适配器***

- **Datus Studio（VSCode 插件）** - 官方 VSCode 插件，把 Datus 能力带进 IDE：Object Explorer（Catalog/Context 树）、SubAgent 创建向导、带 `@` 引用和 Plan 模式的 Datus Chat 面板、可切换 datasource/subagent、SQL Result & AI Chart 面板（ECharts），以及绑定 workspace 的 FileSystem 工具。通过单一 Endpoint 连接任意 Datus-agent Web Server（`datus-cli --web`）。[#713](https://github.com/Datus-ai/Datus-agent/pull/713) [#717](https://github.com/Datus-ai/Datus-agent/pull/717) [文档](https://docs.datus.ai/0.3/zh/vscode_extension/introduction/)
- **数据库适配器：Greenplum 与 Migration Mixin** - `datus-db-adapters` 新增 Greenplum，提升 metadata robustness、thread-safe connector isolation、dialect-specific identifier quoting，并新增 migration workflow 所需的 `MigrationTargetMixin`。[datus-db-adapters#40](https://github.com/Datus-ai/datus-db-adapters/pull/40) [#43](https://github.com/Datus-ai/datus-db-adapters/pull/43) [#45](https://github.com/Datus-ai/datus-db-adapters/pull/45) [#46](https://github.com/Datus-ai/datus-db-adapters/pull/46) [#47](https://github.com/Datus-ai/datus-db-adapters/pull/47) [#48](https://github.com/Datus-ai/datus-db-adapters/pull/48) [文档](https://docs.datus.ai/0.3/zh/adapters/db_adapters/)
- **BI 适配器：Superset 与 Grafana** - `datus-bi-adapters` 新增 `datus-bi-core`、Superset 和 Grafana adapters，支持 list API、chart data retrieval、dashboard/chart 写入校验、分页 envelope、datasource metadata 修复和 dashboard layout 改进。[datus-bi-adapters#1](https://github.com/Datus-ai/datus-bi-adapters/pull/1) [#2](https://github.com/Datus-ai/datus-bi-adapters/pull/2) [#3](https://github.com/Datus-ai/datus-bi-adapters/pull/3) [#7](https://github.com/Datus-ai/datus-bi-adapters/pull/7) [#8](https://github.com/Datus-ai/datus-bi-adapters/pull/8) [#9](https://github.com/Datus-ai/datus-bi-adapters/pull/9) [文档](https://docs.datus.ai/0.3/zh/adapters/bi_adapters/)
- **Scheduler 适配器：Airflow** - `datus-scheduler-adapters` 新增 `datus-scheduler-core` 与 Airflow adapter，支持 DuckDB DAG 执行、多租户 DAG folder、job/run list result envelope、inactive DAG 删除语义，并发布到 `datus-scheduler-airflow` 0.1.2。[datus-scheduler-adapters#2](https://github.com/Datus-ai/datus-scheduler-adapters/pull/2) [#3](https://github.com/Datus-ai/datus-scheduler-adapters/pull/3) [#4](https://github.com/Datus-ai/datus-scheduler-adapters/pull/4) [#8](https://github.com/Datus-ai/datus-scheduler-adapters/pull/8) [#9](https://github.com/Datus-ai/datus-scheduler-adapters/pull/9) [文档](https://docs.datus.ai/0.3/zh/adapters/scheduler_adapters/)
- **语义层适配器拆分** - `datus-semantic-adapter` 拆分出 `datus-semantic-core`，迁移 MetricFlow adapter，支持 dict config injection、语义 adapter contract tests、datasource 术语、可配置 semantic model 路径和更严格的 MetricFlow validation。[datus-semantic-adapter#6](https://github.com/Datus-ai/datus-semantic-adapter/pull/6) [#7](https://github.com/Datus-ai/datus-semantic-adapter/pull/7) [#9](https://github.com/Datus-ai/datus-semantic-adapter/pull/9) [#10](https://github.com/Datus-ai/datus-semantic-adapter/pull/10) [文档](https://docs.datus.ai/0.3/zh/adapters/semantic_adapters/)

**增强**

- **流式与会话稳定性** - 修复并增强 web/chat/gateway streaming、compact/resume、群聊 thread 处理、Feishu 权限、Slack 回复、API node 创建、session persistence 和时间戳格式。[#531](https://github.com/Datus-ai/Datus-agent/pull/531) [#548](https://github.com/Datus-ai/Datus-agent/pull/548) [#567](https://github.com/Datus-ai/Datus-agent/pull/567) [#568](https://github.com/Datus-ai/Datus-agent/pull/568) [#638](https://github.com/Datus-ai/Datus-agent/pull/638) [#674](https://github.com/Datus-ai/Datus-agent/pull/674) [#680](https://github.com/Datus-ai/Datus-agent/pull/680) [#689](https://github.com/Datus-ai/Datus-agent/pull/689) [#700](https://github.com/Datus-ai/Datus-agent/pull/700) [文档](https://docs.datus.ai/0.3/zh/API/chat/)
- **生成稳定性** - 提升 semantic、metric、reference-template、dashboard、SQL prompt、query-metric 生成稳定性。[#596](https://github.com/Datus-ai/Datus-agent/pull/596) [#604](https://github.com/Datus-ai/Datus-agent/pull/604) [#690](https://github.com/Datus-ai/Datus-agent/pull/690) [#691](https://github.com/Datus-ai/Datus-agent/pull/691) [#692](https://github.com/Datus-ai/Datus-agent/pull/692) [#697](https://github.com/Datus-ai/Datus-agent/pull/697) [文档](https://docs.datus.ai/0.3/zh/subagent/gen_semantic_model/)
- **文件系统与数据隔离** - 通过 `filesystem_strict`、project-root zone policy、安全 search、credential redaction、严格 FuncTool result handling 增强文件系统与数据隔离。[#588](https://github.com/Datus-ai/Datus-agent/pull/588) [#597](https://github.com/Datus-ai/Datus-agent/pull/597) [#603](https://github.com/Datus-ai/Datus-agent/pull/603) [#681](https://github.com/Datus-ai/Datus-agent/pull/681) [#694](https://github.com/Datus-ai/Datus-agent/pull/694) [文档](https://docs.datus.ai/0.3/zh/configuration/agent/)
- **Storage 重构** - 统一 `datus_db`、datasource 隔离、singleton registry、可插拔 RDB/vector backend，并支持通过 `datus-storage-postgresql` 使用 PostgreSQL 后端。[#493](https://github.com/Datus-ai/Datus-agent/pull/493) [#499](https://github.com/Datus-ai/Datus-agent/pull/499) [文档](https://docs.datus.ai/0.3/zh/configuration/storage/)
- **CI 流程重构** - 拆分 PR acceptance/nightly 流程，新增 docker-backed adapter integration tests 与 test-quality audit workflow，并修复多项 nightly、unit、integration regressions。[#589](https://github.com/Datus-ai/Datus-agent/pull/589) [#600](https://github.com/Datus-ai/Datus-agent/pull/600) [#601](https://github.com/Datus-ai/Datus-agent/pull/601) [#634](https://github.com/Datus-ai/Datus-agent/pull/634)

**文档补全**

- **REST API、IM 网关与 CLI 文档** - 新增 REST API deployment / chat / KB / models 文档，Slack 与 Feishu IM gateway 文档，以及 `/model`、`/language`、`/effort`、`/init`、`/bootstrap`、service command、`--datasource` 流程文档。[文档](https://docs.datus.ai/0.3/zh/API/deployment/)
- **配置文档** - 新增 datasources、semantic layer、BI platforms、schedulers、PostgreSQL-backed storage 配置文档。[文档](https://docs.datus.ai/0.3/zh/configuration/datasources/)
- **Subagent 文档** - 新增 dashboard generation、table generation、scheduler workflow、data pipeline、metrics、semantic model、SQL summary 等 subagent 文档。[文档](https://docs.datus.ai/0.3/zh/subagent/builtin_subagents/)
- **Adapter、Memory 与 Reference Template 文档** - 刷新 adapter、memory、reference template、quickstart、benchmark 与 docs deployment 文档。[#530](https://github.com/Datus-ai/Datus-agent/pull/530) [#536](https://github.com/Datus-ai/Datus-agent/pull/536) [#549](https://github.com/Datus-ai/Datus-agent/pull/549) [#556](https://github.com/Datus-ai/Datus-agent/pull/556) [#611](https://github.com/Datus-ai/Datus-agent/pull/611) [#622](https://github.com/Datus-ai/Datus-agent/pull/622) [#667](https://github.com/Datus-ai/Datus-agent/pull/667) [文档](https://docs.datus.ai/0.3/zh/adapters/db_adapters/)

## 0.2

### 0.2.6

**新功能**

- **Ask User Tool** - 引入交互式 `ask_user` 工具，支持内联自由文本输入和批量提问能力，已集成进 GenSQL 与 GenReport node，支持 human-in-the-loop workflow。[#457](https://github.com/Datus-ai/Datus-agent/pull/457) [#460](https://github.com/Datus-ai/Datus-agent/pull/460) [#481](https://github.com/Datus-ai/Datus-agent/pull/481)
- **Skill Marketplace CLI** - 内置 marketplace，可直接从 CLI 发现、安装、管理社区 skills。[#416](https://github.com/Datus-ai/Datus-agent/pull/416) [文档](skills/introduction.md)
- **General Chat Agent** - 通用聊天 agent，支持 SQL 生成以外的灵活会话场景。[#452](https://github.com/Datus-ai/Datus-agent/pull/452)
- **Explore Task Tool** - 新增 exploration 工具,用于在 agent 内导航与管理任务。[#455](https://github.com/Datus-ai/Datus-agent/pull/455)
- **Storage Adapter** - 可插拔 storage adapter 层，便于灵活接入后端。[#446](https://github.com/Datus-ai/Datus-agent/pull/446)
- **4 个新数据库适配器** - 在 [datus-db-adapters](https://github.com/Datus-ai/datus-db-adapters) 仓库新增 ClickHouse、Hive、Spark、Trino 适配器，均可通过 `pip install datus-<database>` 独立安装。[文档](adapters/db_adapters.md)

**增强**

- **Session Resume/Rewind** - 新增 `/resume`、`/rewind`、`.interrupt` 命令，配合交互式方向键选择器浏览会话历史。[#438](https://github.com/Datus-ai/Datus-agent/pull/438) [#470](https://github.com/Datus-ai/Datus-agent/pull/470) [文档](cli/chat_command.md)
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

- 新增 Hive、Spark、ClickHouse、Trino 数据库适配器文档。[#464](https://github.com/Datus-ai/Datus-agent/pull/464) [文档](adapters/db_adapters.md)
- 新增 resume/rewind 命令文档。[#465](https://github.com/Datus-ai/Datus-agent/pull/465)

### 0.2.5

**新功能**

- **OpenAI Agent SDK 0.7.0 升级，支持 Kimi-2.5 与 Gemini-3** - 用 `litellm_adapter` 和 `sdk_patches` 重建模型层，无缝接入最新的 Kimi-2.5 与 Gemini-3 系列模型。
- **AgentSkills 支持** - 引入完整的 Skill 系统，包含 skill 配置、注册、管理与权限控制，同时支持 bash 与 function 形态的 skill 工具。[文档](skills/introduction.md)
- **Tools as MCP Server** - 将 Datus 的数据库工具与 context search 暴露为 MCP server，可对接 Claude Desktop、Claude Code 等 MCP 兼容客户端。[文档](integration/mcp.md)

**增强**

- **语义工具优化** - 优化语义工具与 context search，在 CLI 中获得更快、更相关的结果。
- **生成 Prompt 字符串校验** - 加强多个 prompt template 的字符串校验，提升生成结果可靠性。
- **基于 Action 的用户交互模型** - 重做 CLI 交互层，统一以 action-based 模型驱动 execution、generation 与 planning。
- **Reference SQL 并行化与日期支持** - 并行化 reference SQL 初始化加速 bootstrap，并增强日期表达式解析。[文档](knowledge_base/reference_sql.md)
- **Bootstrap Markdown 摘要** - bootstrap 完成后展示格式化的 Markdown 摘要，便于快速浏览生成结果。[文档](getting_started/dashboard_copilot.md)
- **Subject 条目删除** - 可直接在 `@subject` 界面删除 semantic models、metrics 与 SQL summaries。[文档](cli/context_command.md#subject)

**Bug 修复**

- **Subject Node 竞争条件** - 修复并行创建多个 subject node 时的竞争条件,提升并发安全性。
- **多轮 Benchmark 评估** - 修复多轮评估中 agent state、workflow runner、配置处理相关的问题。[文档](benchmark/benchmark_manual.md)
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

- **内置教程数据集** - California Schools 数据集随安装包打包，并集成进 `datus-agent init` 流程，方便上手学习上下文数据工程。[#277](https://github.com/Datus-ai/Datus-agent/issues/277) [教程](getting_started/contextual_data_engineering.md)
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

- **Web Chatbot 升级** - 新增反馈收集、问题上报、流式输出，以及 `&hide_sidebar=true` 嵌入参数。[文档](web_chatbot/introduction.md)
- **上下文生成命令** - subagent 中新增 `/gen_semantic_model`、`/gen_metrics`、`/gen_sql_summary` 命令，用于动态丰富知识库。[#192](https://github.com/Datus-ai/Datus-agent/issues/192) [文档](subagent/builtin_subagents.md)
- **交互式上下文编辑** - `@catalog`、`@subject` 命令支持可视化编辑 semantic models、metrics 与 SQL summaries。[#219](https://github.com/Datus-ai/Datus-agent/issues/219) [#199](https://github.com/Datus-ai/Datus-agent/issues/199) [#175](https://github.com/Datus-ai/Datus-agent/issues/175) [文档](cli/context_command.md#subject)
- **作用域知识库** - subagent 支持作用域 KB 初始化，提升上下文隔离与管理能力。[#217](https://github.com/Datus-ai/Datus-agent/issues/217)

**增强**

- **MetricFlow 集成** - 从 `env_settings.yml` 加载配置，改进项目检测，输出格式更整洁。[#214](https://github.com/Datus-ai/Datus-agent/issues/214) [#216](https://github.com/Datus-ai/Datus-agent/issues/216) [文档](metricflow/introduction.md)
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
