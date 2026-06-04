# Release notes

## 0.3

### 0.3.3

**New Features**

- **Automatic Session Compaction** - Long conversations now clean up context automatically, reducing interruptions when the context fills up. As a conversation grows, older tool-call records are archived to disk while the active task stays in context; near the limit (about 90%), Datus summarizes the session and keeps the run moving. The `/compact` command also shows progress and a summary panel, with full history persisted for restore and review. [#871](https://github.com/Datus-ai/Datus-agent/pull/871) [#919](https://github.com/Datus-ai/Datus-agent/pull/919) [#933](https://github.com/Datus-ai/Datus-agent/pull/933) [docs](configuration/compact.md)
- **Live Token Usage** - Model responses now show newly used and cumulative tokens as they stream, while the status bar tracks current context usage and total capacity. Usage history is saved with the session for restore, cost review, and context audits. [#920](https://github.com/Datus-ai/Datus-agent/pull/920)

**Enhancements**

- **Snowflake Key-Pair Authentication** - Snowflake setups with MFA can now use RSA key-pair authentication instead of passwords. Configure `private_key_file`, optional `private_key_file_pwd`, and `role`; the same credentials are used for SQL execution and MetricFlow metric generation, validation, and querying, with credentials redacted from logs. [#926](https://github.com/Datus-ai/Datus-agent/pull/926) [datus-db-adapters#66](https://github.com/Datus-ai/datus-db-adapters/pull/66) [datus-db-adapters#67](https://github.com/Datus-ai/datus-db-adapters/pull/67) [datus-semantic-adapter#25](https://github.com/Datus-ai/datus-semantic-adapter/pull/25) [docs](configuration/datasources.md)
- **Offline Embedding Fallback** - Datus no longer stalls on embedding model downloads in offline, intranet, or Hugging Face-blocked environments. Context search and `@` reference autocomplete temporarily degrade with diagnostics for the model name, cache path, environment variables, and fix steps; database tools and normal chat continue to work. The docs also cover pre-caching models or using an OpenAI-compatible embedding service for offline deployments. [#870](https://github.com/Datus-ai/Datus-agent/pull/870)
- **Codex Prompt Cache Optimization** - When using the ChatGPT subscription-backed Codex backend, multi-step runs can now reuse prompt cache like the official client, reducing wait time and token cost. [#918](https://github.com/Datus-ai/Datus-agent/pull/918)
- **Multi-Turn `datus -p` and `--resume`** - Print mode can now restore and continue a specific session. Use `datus -p '...' --resume <session_id>` to continue from the command line; REPL and API chat also support `--resume` for existing sessions. [#914](https://github.com/Datus-ai/Datus-agent/pull/914)

**Bug Fixes**

- **Custom Ask Subagents Honor Tool Whitelists** - `ask_report` / `ask_dashboard` now only see tools allowed by the subagent's `tools` whitelist. Unauthorized database, bash, and skill tools are no longer exposed to the model, and prompts no longer include the full chat tool directory, reducing unauthorized-tool and context-pollution risk. [#877](https://github.com/Datus-ai/Datus-agent/pull/877) [#878](https://github.com/Datus-ai/Datus-agent/pull/878) [#881](https://github.com/Datus-ai/Datus-agent/pull/881)

### 0.3.2

**New Features**

- **Agent Observability (Configurable Tracing)** - Turn on `agent.observability.tracing` to export run traces to Langfuse, LangSmith, Datadog, Braintrust, or a generic OTLP collector. Optionally capture run content and redact sensitive data, and get stable trace references plus runtime trace grouping that tie benchmark, bootstrap, CLI, and chat runs into a single trace for easy correlation and debugging. [#833](https://github.com/Datus-ai/Datus-agent/pull/833) [#864](https://github.com/Datus-ai/Datus-agent/pull/864) [docs](develop/observability.md)
- **Visual Artifacts: Dashboard Support** - The new `gen_visual_dashboard` generates dashboard-style HTML (multi-chart card layouts) right from Chat, just like `gen_visual_report`, with **local interactive preview served by `datus --web`** — filters re-run live against your database, no SaaS backend required. Report and dashboard generation is also noticeably better: cleaner layouts, more reliable chart rendering (concurrent multi-chart queries to DuckDB no longer race), consistent use of the runtime ChartCard primitive, more accurate underlying data queries, support for more complex reports, and multi-round refinement. The compiled HTML's absolute path is now surfaced in the CLI message stream, so you can reopen the artifact after closing the browser tab. [#829](https://github.com/Datus-ai/Datus-agent/pull/829) [#835](https://github.com/Datus-ai/Datus-agent/pull/835) [#842](https://github.com/Datus-ai/Datus-agent/pull/842) [#847](https://github.com/Datus-ai/Datus-agent/pull/847) [#848](https://github.com/Datus-ai/Datus-agent/pull/848) [#849](https://github.com/Datus-ai/Datus-agent/pull/849) [#853](https://github.com/Datus-ai/Datus-agent/pull/853) [#855](https://github.com/Datus-ai/Datus-agent/pull/855) [#863](https://github.com/Datus-ai/Datus-agent/pull/863) [#866](https://github.com/Datus-ai/Datus-agent/pull/866) [#867](https://github.com/Datus-ai/Datus-agent/pull/867) [#869](https://github.com/Datus-ai/Datus-agent/pull/869) [#894](https://github.com/Datus-ai/Datus-agent/pull/894) [#895](https://github.com/Datus-ai/Datus-agent/pull/895) [#901](https://github.com/Datus-ai/Datus-agent/pull/901) [#905](https://github.com/Datus-ai/Datus-agent/pull/905) [#907](https://github.com/Datus-ai/Datus-agent/pull/907) [docs](subagent/gen_visual_dashboard.md)
- **Mid-Run User Input** - Send additional instructions while an agent run is still streaming — type in the CLI/TUI or POST to the API; the text is picked up on the model's next step, saved to the session, and shown live as it is inserted, without interrupting the run. [#824](https://github.com/Datus-ai/Datus-agent/pull/824)

**Enhancements**

- **Semantic SQL Metric Extraction** - When mining metrics from historical SQL, Datus tells apart brand-new metrics, metrics derived from existing ones, and plain references to existing metrics, so it won't create duplicates; it also keeps time grain, filters, and literal values. Supported metric types include count, distinct count, sum, average, min/max, conditional (filtered), ratio, expression, cumulative, and derived; for multi-table cases (cross-table calculations, non-equi joins, unions) it first combines the data into a single source and then defines metrics on top. End-to-end verified on a real warehouse (StarRocks): the generated metrics return the same values as the original SQL. [#811](https://github.com/Datus-ai/Datus-agent/pull/811)
- **Smoother CLI Chat & Streaming** - Fixes a duplicated last paragraph in the terminal when streaming through Claude's native API (plus a related session-resume parsing fix); chat history now renders through one unified path, so history, resume/rewind, and mid-run user inserts all display consistently, with user messages shown in a bordered panel for clearer separation. [#837](https://github.com/Datus-ai/Datus-agent/pull/837) [#852](https://github.com/Datus-ai/Datus-agent/pull/852)

**Bug Fixes**

- **Phased Semantic-Model Validation** - A semantic model can be validated before any metrics exist: the expected "no metrics yet" issue no longer aborts the flow, while genuine model errors still fail validation. [#827](https://github.com/Datus-ai/Datus-agent/pull/827) [#850](https://github.com/Datus-ai/Datus-agent/pull/850)
- **Reliable Bootstrap Result Handling** - Bootstrap flows now correctly recognize each step's generation result (including failures), so successful output is no longer dropped and failures are no longer silently swallowed. [#831](https://github.com/Datus-ai/Datus-agent/pull/831)
- **Reference SQL Summary Path Resolution** - Generated reference-SQL summary paths now resolve correctly across the different path forms; out-of-sandbox paths are safely skipped instead of crashing. [#840](https://github.com/Datus-ai/Datus-agent/pull/840)
- **Print Mode No Longer Hangs on Permission Prompts** - `datus -p` (the non-interactive print mode, used in CI and scripting) now runs under the workflow execution mode and dangerous profile — consistent with `/bootstrap` and other non-interactive flows — so ASK/EXTERNAL permission prompts that previously blocked waiting for a human responder short-circuit cleanly. [#891](https://github.com/Datus-ai/Datus-agent/pull/891)

### 0.3.1

**New Features**

- **HTML Report Generation** - New `gen_visual_report` subagent turns a question, a metric reference, or your own SQL into a self-contained HTML report (KPI cards, charts, tables, narrative) with section-by-section editing so you can refine a single chart without rewriting the whole report. [#783](https://github.com/Datus-ai/Datus-agent/pull/783) [#821](https://github.com/Datus-ai/Datus-agent/pull/821) [docs](subagent/gen_visual_report.md)
- **Persistent Plan Mode** - Plan Mode now writes `plan.md` to disk and restores it on session resume, so closing the CLI mid-plan no longer loses your work. [#772](https://github.com/Datus-ai/Datus-agent/pull/772) [docs](cli/plan_mode.md)
- **CLI / TUI Polish** - Live todo sidebar tracks task progress at a glance, plus an inline command wizard, scroll-back search, mouse-drag selection copy, and a draggable scrollbar for a more native terminal feel. [#772](https://github.com/Datus-ai/Datus-agent/pull/772)

**Enhancements**

- **`/permission` Command** - Renamed `/profile` to `/permission`, with `normal` / `auto` / `dangerous` modes for matching different development workflows. [#769](https://github.com/Datus-ai/Datus-agent/pull/769) [docs](cli/reference.md)
- **Custom Subagent Management** - Custom subagents can now be deleted via API or TUI, and the available-tools list per agent type is returned by a single backend source so SaaS and standalone UIs build and edit subagents consistently. [#807](https://github.com/Datus-ai/Datus-agent/pull/807) [#812](https://github.com/Datus-ai/Datus-agent/pull/812) [docs](subagent/customized_subagent.md)
- **Per-Request Permission Mode** - Chat requests can pick `normal` / `auto` / `dangerous` per call, so multi-tenant SaaS deployments stop polluting a shared default. [#822](https://github.com/Datus-ai/Datus-agent/pull/822) [docs](integration/skills.md)

**Bug Fixes**

- **Claude / Anthropic Parameter Conflict** - Requests on the Claude / Anthropic route no longer fail when both `temperature` and `top_p` are sent in the same call. [#817](https://github.com/Datus-ai/Datus-agent/pull/817)
- **Metric ID Collision Under Missing Subject Path** - Same-named metrics across different subject trees now stay distinct when `subject_path` was previously absent from the metric id. [#819](https://github.com/Datus-ai/Datus-agent/pull/819)

### 0.3.0

**New Features**

***Datus API***

- **FastAPI REST API** - Layered service/model REST API with CLI entry, streaming Chat, task tracking, SQL execution stop, multi-select `ask_user`, success story persistence, knowledge base bootstrap, and request-side proxy source / interactive mode controls. [#520](https://github.com/Datus-ai/Datus-agent/pull/520) [#538](https://github.com/Datus-ai/Datus-agent/pull/538) [#539](https://github.com/Datus-ai/Datus-agent/pull/539) [#551](https://github.com/Datus-ai/Datus-agent/pull/551) [#553](https://github.com/Datus-ai/Datus-agent/pull/553) [#555](https://github.com/Datus-ai/Datus-agent/pull/555) [#606](https://github.com/Datus-ai/Datus-agent/pull/606) [#610](https://github.com/Datus-ai/Datus-agent/pull/610) [docs](API/introduction.md)
- **Model Discovery API** - Model discovery, per-request model override, current model metadata, and ISO-8601 UTC timestamps. [#643](https://github.com/Datus-ai/Datus-agent/pull/643) [#649](https://github.com/Datus-ai/Datus-agent/pull/649) [#700](https://github.com/Datus-ai/Datus-agent/pull/700) [docs](API/models.md)
- **Chart Recommendation & Visualization API** - Generate dashboard-ready visualizations from Datus Chat and external applications. [#545](https://github.com/Datus-ai/Datus-agent/pull/545) [docs](getting_started/dashboard_copilot.md)

***Datus Chat & IM Gateways***

- **Datus Chat (FastAPI Chatbot)** - Replaced the legacy Streamlit chatbot with FastAPI + `@datus/web-chatbot`, adding the Datus Chat module. [#543](https://github.com/Datus-ai/Datus-agent/pull/543) [#554](https://github.com/Datus-ai/Datus-agent/pull/554) [docs](web_chatbot/introduction.md)
- **Slack & Feishu/Lark Gateways** - New IM gateways with channel configuration, daemon mode, streaming replies, and feedback actions; `datus-claw` renamed to `datus-gateway`. [#559](https://github.com/Datus-ai/Datus-agent/pull/559) [#562](https://github.com/Datus-ai/Datus-agent/pull/562) [#565](https://github.com/Datus-ai/Datus-agent/pull/565) [#616](https://github.com/Datus-ai/Datus-agent/pull/616) [#623](https://github.com/Datus-ai/Datus-agent/pull/623) [#593](https://github.com/Datus-ai/Datus-agent/pull/593) [docs](gateway/introduction.md)

***Project & Workspace Configuration***

- **Project-aware Configure/Init Flow** - Split `setup` into project-aware `configure`/`init` flows with project-level `.datus/config.yml`, project memory, automatic datasource/service setup, and a redesigned `.datus` directory. [#542](https://github.com/Datus-ai/Datus-agent/pull/542) [#578](https://github.com/Datus-ai/Datus-agent/pull/578) [#592](https://github.com/Datus-ai/Datus-agent/pull/592) [#608](https://github.com/Datus-ai/Datus-agent/pull/608) [docs](cli/init_command.md)
- **Unified Runtime Services Config** - Unified configuration around `services.datasources`, `services.bi_platforms`, semantic layer, and scheduler; CLI now uses `--datasource`. [#614](https://github.com/Datus-ai/Datus-agent/pull/614) [#633](https://github.com/Datus-ai/Datus-agent/pull/633) [#636](https://github.com/Datus-ai/Datus-agent/pull/636) [#642](https://github.com/Datus-ai/Datus-agent/pull/642) [docs](configuration/datasources.md)
- **One-line Installer** - New Linux/macOS `curl | sh` install script with refreshed quickstart and service docs. [#613](https://github.com/Datus-ai/Datus-agent/pull/613) [#611](https://github.com/Datus-ai/Datus-agent/pull/611) [#667](https://github.com/Datus-ai/Datus-agent/pull/667) [docs](getting_started/Quickstart.md)

***CLI Experience***

- **Unified `/` Command Prefix** - All interactive commands moved to `/` prefix; added `/model`, `/skill`, `/mcp`, `/agent`, `/subagent`, interactive input, and a streaming `/bootstrap` TUI. [#621](https://github.com/Datus-ai/Datus-agent/pull/621) [#635](https://github.com/Datus-ai/Datus-agent/pull/635) [#650](https://github.com/Datus-ai/Datus-agent/pull/650) [#655](https://github.com/Datus-ai/Datus-agent/pull/655) [#656](https://github.com/Datus-ai/Datus-agent/pull/656) [#659](https://github.com/Datus-ai/Datus-agent/pull/659) [#683](https://github.com/Datus-ai/Datus-agent/pull/683) [docs](cli/reference.md)
- **`/language` and `/effort` Commands** - Pin response language with `/language`, control reasoning effort with `/effort`, plus `/<service>.<method>` dispatch for read-only service calls. [#641](https://github.com/Datus-ai/Datus-agent/pull/641) [#653](https://github.com/Datus-ai/Datus-agent/pull/653) [#631](https://github.com/Datus-ai/Datus-agent/pull/631) [docs](cli/language_command.md)
- **CLI Print Mode & UX Polish** - Print mode, proxy tools, reworked bottom status bar, fixed streaming/tool status line, improved markdown streaming, and restored `@` reference auto-completion. [#489](https://github.com/Datus-ai/Datus-agent/pull/489) [#501](https://github.com/Datus-ai/Datus-agent/pull/501) [#583](https://github.com/Datus-ai/Datus-agent/pull/583) [#586](https://github.com/Datus-ai/Datus-agent/pull/586) [#654](https://github.com/Datus-ai/Datus-agent/pull/654) [#664](https://github.com/Datus-ai/Datus-agent/pull/664) [#661](https://github.com/Datus-ai/Datus-agent/pull/661) [#662](https://github.com/Datus-ai/Datus-agent/pull/662) [docs](cli/introduction.md)
- **New Model & Plan Providers** - Codex OAuth, Claude Subscription, Coding Plan, OpenRouter, MiniMax, GLM, BigModel, Z.AI support, with rebuilt provider-based model configuration and provider catalog. [#487](https://github.com/Datus-ai/Datus-agent/pull/487) [#635](https://github.com/Datus-ai/Datus-agent/pull/635) [#687](https://github.com/Datus-ai/Datus-agent/pull/687) [#693](https://github.com/Datus-ai/Datus-agent/pull/693) [docs](cli/model_command.md)
- **Permission Profiles** - New `normal` / `auto` / `dangerous` permission profiles with subagent-aware permission hooks; safe discovery tools relaxed in normal mode. [#646](https://github.com/Datus-ai/Datus-agent/pull/646) [#652](https://github.com/Datus-ai/Datus-agent/pull/652) [docs](integration/skills.md)

***Data Engineering Subagents & Skills***

- **Data Engineering Agents & Skills** - Built-in agents and skills for cross-database migration, ETL/job generation, scheduler workflows, table generation, dashboard generation, and BI/scheduler orchestration. [#494](https://github.com/Datus-ai/Datus-agent/pull/494) [#525](https://github.com/Datus-ai/Datus-agent/pull/525) [#564](https://github.com/Datus-ai/Datus-agent/pull/564) [#575](https://github.com/Datus-ai/Datus-agent/pull/575) [#580](https://github.com/Datus-ai/Datus-agent/pull/580) [#639](https://github.com/Datus-ai/Datus-agent/pull/639) [docs](subagent/builtin_subagents.md)
- **Deliverable Validation Hooks** - Table deliverable validation hook, shared deliverable node, validation skills, and a publish gate for semantic/metric generation. [#657](https://github.com/Datus-ai/Datus-agent/pull/657) [#663](https://github.com/Datus-ai/Datus-agent/pull/663) [#665](https://github.com/Datus-ai/Datus-agent/pull/665) [docs](integration/validation.md)
- **Natural-language Metrics & Skill Creator** - Natural-language metric creation, wheel-bundled built-in skills, skill frontmatter scope, and a `skill-creator` subagent for interactive skill authoring. [#504](https://github.com/Datus-ai/Datus-agent/pull/504) [#526](https://github.com/Datus-ai/Datus-agent/pull/526) [#627](https://github.com/Datus-ai/Datus-agent/pull/627) [#645](https://github.com/Datus-ai/Datus-agent/pull/645) [#676](https://github.com/Datus-ai/Datus-agent/pull/676) [docs](subagent/customized_subagent.md)

***Memory & Reference Template***

- **Auto Memory** - New `MEMORY.md`-based Auto Memory with an emergent topic tree, empty-memory prompt, and project/session isolation. [#498](https://github.com/Datus-ai/Datus-agent/pull/498) [#620](https://github.com/Datus-ai/Datus-agent/pull/620) [#595](https://github.com/Datus-ai/Datus-agent/pull/595) [#523](https://github.com/Datus-ai/Datus-agent/pull/523) [#594](https://github.com/Datus-ai/Datus-agent/pull/594) [docs](integration/memory.md)
- **Reference Template** - New Reference Template mechanism, with bootstrap reference template parsing fixes. [#508](https://github.com/Datus-ai/Datus-agent/pull/508) [#574](https://github.com/Datus-ai/Datus-agent/pull/574) [#677](https://github.com/Datus-ai/Datus-agent/pull/677) [docs](knowledge_base/reference_template.md)

***Ecosystem & Adapters***

- **Datus Studio (VSCode Extension)** - Official VSCode extension that brings Datus into the IDE: Object Explorer (Catalog/Context trees), SubAgent wizard, Datus Chat panel with `@`-references, plan mode, datasource/subagent switching, SQL Result & AI Chart panel (ECharts), and workspace-scoped FileSystem tools. Connects to any Datus-agent Web Server (`datus-cli --web`) via a single Endpoint. [#713](https://github.com/Datus-ai/Datus-agent/pull/713) [#717](https://github.com/Datus-ai/Datus-agent/pull/717) [docs](vscode_extension/introduction.md)
- **Database Adapters: Greenplum & Migration Mixin** - `datus-db-adapters` added Greenplum, improved metadata robustness, thread-safe connector isolation, dialect-specific identifier quoting, and a `MigrationTargetMixin` for migration workflows. [datus-db-adapters#40](https://github.com/Datus-ai/datus-db-adapters/pull/40) [#43](https://github.com/Datus-ai/datus-db-adapters/pull/43) [#45](https://github.com/Datus-ai/datus-db-adapters/pull/45) [#46](https://github.com/Datus-ai/datus-db-adapters/pull/46) [#47](https://github.com/Datus-ai/datus-db-adapters/pull/47) [#48](https://github.com/Datus-ai/datus-db-adapters/pull/48) [docs](adapters/db_adapters.md)
- **BI Adapters: Superset & Grafana** - New `datus-bi-core` with Superset and Grafana adapters, list API, chart data retrieval, dashboard/chart write validation, paginated envelope, datasource metadata fixes, and dashboard layout improvements. [datus-bi-adapters#1](https://github.com/Datus-ai/datus-bi-adapters/pull/1) [#2](https://github.com/Datus-ai/datus-bi-adapters/pull/2) [#3](https://github.com/Datus-ai/datus-bi-adapters/pull/3) [#7](https://github.com/Datus-ai/datus-bi-adapters/pull/7) [#8](https://github.com/Datus-ai/datus-bi-adapters/pull/8) [#9](https://github.com/Datus-ai/datus-bi-adapters/pull/9) [docs](adapters/bi_adapters.md)
- **Scheduler Adapters: Airflow** - New `datus-scheduler-core` and Airflow adapter with DuckDB DAG execution, multi-tenant DAG folder, job/run list result envelope, and inactive DAG deletion semantics; published as `datus-scheduler-airflow` 0.1.2. [datus-scheduler-adapters#2](https://github.com/Datus-ai/datus-scheduler-adapters/pull/2) [#3](https://github.com/Datus-ai/datus-scheduler-adapters/pull/3) [#4](https://github.com/Datus-ai/datus-scheduler-adapters/pull/4) [#8](https://github.com/Datus-ai/datus-scheduler-adapters/pull/8) [#9](https://github.com/Datus-ai/datus-scheduler-adapters/pull/9) [docs](adapters/scheduler_adapters.md)
- **Semantic Adapter Split** - `datus-semantic-adapter` split out `datus-semantic-core` and migrated the MetricFlow adapter, with dict config injection, adapter contract tests, datasource terminology, configurable semantic model paths, and stricter MetricFlow validation. [datus-semantic-adapter#6](https://github.com/Datus-ai/datus-semantic-adapter/pull/6) [#7](https://github.com/Datus-ai/datus-semantic-adapter/pull/7) [#9](https://github.com/Datus-ai/datus-semantic-adapter/pull/9) [#10](https://github.com/Datus-ai/datus-semantic-adapter/pull/10) [docs](adapters/semantic_adapters.md)

**Enhancements**

- **Streaming & Session Stability** - Fixed and enhanced web/chat/gateway streaming, compact/resume, group chat thread handling, Feishu permissions, Slack replies, API node creation, session persistence, and timestamp formats. [#531](https://github.com/Datus-ai/Datus-agent/pull/531) [#548](https://github.com/Datus-ai/Datus-agent/pull/548) [#567](https://github.com/Datus-ai/Datus-agent/pull/567) [#568](https://github.com/Datus-ai/Datus-agent/pull/568) [#638](https://github.com/Datus-ai/Datus-agent/pull/638) [#674](https://github.com/Datus-ai/Datus-agent/pull/674) [#680](https://github.com/Datus-ai/Datus-agent/pull/680) [#689](https://github.com/Datus-ai/Datus-agent/pull/689) [#700](https://github.com/Datus-ai/Datus-agent/pull/700) [docs](API/chat.md)
- **Generation Stability** - Improved semantic, metric, reference-template, dashboard, SQL prompt, and query-metric generation. [#596](https://github.com/Datus-ai/Datus-agent/pull/596) [#604](https://github.com/Datus-ai/Datus-agent/pull/604) [#690](https://github.com/Datus-ai/Datus-agent/pull/690) [#691](https://github.com/Datus-ai/Datus-agent/pull/691) [#692](https://github.com/Datus-ai/Datus-agent/pull/692) [#697](https://github.com/Datus-ai/Datus-agent/pull/697) [docs](subagent/gen_semantic_model.md)
- **Filesystem & Data Isolation** - Strengthened via `filesystem_strict`, project-root zone policy, safe search, credential redaction, and strict FuncTool result handling. [#588](https://github.com/Datus-ai/Datus-agent/pull/588) [#597](https://github.com/Datus-ai/Datus-agent/pull/597) [#603](https://github.com/Datus-ai/Datus-agent/pull/603) [#681](https://github.com/Datus-ai/Datus-agent/pull/681) [#694](https://github.com/Datus-ai/Datus-agent/pull/694) [docs](configuration/agent.md)
- **Storage Refactor** - Unified `datus_db`, datasource isolation, singleton registry, pluggable RDB/vector backends, and PostgreSQL support via `datus-storage-postgresql`. [#493](https://github.com/Datus-ai/Datus-agent/pull/493) [#499](https://github.com/Datus-ai/Datus-agent/pull/499) [docs](configuration/storage.md)
- **CI Restructure** - Split PR acceptance and nightly pipelines, added docker-backed adapter integration tests and a test-quality audit workflow, and resolved multiple nightly/unit/integration regressions. [#589](https://github.com/Datus-ai/Datus-agent/pull/589) [#600](https://github.com/Datus-ai/Datus-agent/pull/600) [#601](https://github.com/Datus-ai/Datus-agent/pull/601) [#634](https://github.com/Datus-ai/Datus-agent/pull/634)

**Documentation**

- **REST API, IM Gateway & CLI Docs** - New docs for REST API deployment / chat / KB / models, Slack & Feishu IM gateways, and `/model` / `/language` / `/effort` / `/init` / `/bootstrap` / service / `--datasource` flows. [docs](API/deployment.md)
- **Configuration Docs** - Added datasources, semantic layer, BI platforms, schedulers, and PostgreSQL-backed storage configuration docs. [docs](configuration/datasources.md)
- **Subagent Docs** - Dashboard generation, table generation, scheduler workflow, data pipeline, metrics, semantic model, and SQL summary subagent docs. [docs](subagent/builtin_subagents.md)
- **Adapter, Memory & Reference Template Docs** - Refreshed adapter, memory, reference template, quickstart, benchmark, and docs-deployment documentation. [#530](https://github.com/Datus-ai/Datus-agent/pull/530) [#536](https://github.com/Datus-ai/Datus-agent/pull/536) [#549](https://github.com/Datus-ai/Datus-agent/pull/549) [#556](https://github.com/Datus-ai/Datus-agent/pull/556) [#611](https://github.com/Datus-ai/Datus-agent/pull/611) [#622](https://github.com/Datus-ai/Datus-agent/pull/622) [#667](https://github.com/Datus-ai/Datus-agent/pull/667) [docs](adapters/db_adapters.md)

## 0.2

### 0.2.6

**New Features**

- **Ask User Tool** - Introduced an interactive `ask_user` tool with inline free-text support and batch question capabilities, integrated into GenSQL and GenReport nodes for human-in-the-loop workflows. [#457](https://github.com/Datus-ai/Datus-agent/pull/457) [#460](https://github.com/Datus-ai/Datus-agent/pull/460) [#481](https://github.com/Datus-ai/Datus-agent/pull/481)
- **Skill Marketplace CLI** - Built-in marketplace for discovering, installing, and managing community skills directly from the CLI. [#416](https://github.com/Datus-ai/Datus-agent/pull/416) [docs](integration/skills.md)
- **General Chat Agent** - A general-purpose chat agent for flexible conversational workflows beyond SQL generation. [#452](https://github.com/Datus-ai/Datus-agent/pull/452)
- **Explore Task Tool** - New exploration tool for navigating and managing tasks within the agent. [#455](https://github.com/Datus-ai/Datus-agent/pull/455)
- **Storage Adapter** - Pluggable storage adapter layer for flexible backend integration. [#446](https://github.com/Datus-ai/Datus-agent/pull/446)
- **4 New Database Adapters** - Added ClickHouse, Hive, Spark, and Trino adapters in the [datus-db-adapters](https://github.com/Datus-ai/datus-db-adapters) repository, all installable as independent packages via `pip install datus-<database>`. [docs](adapters/db_adapters.md)

**Enhancements**

- **Session Resume/Rewind** - Added `/resume`, `/rewind`, and `.interrupt` commands with interactive arrow-key selector for navigating conversation history. [#438](https://github.com/Datus-ai/Datus-agent/pull/438) [#470](https://github.com/Datus-ai/Datus-agent/pull/470) [docs](cli/chat_command.md)
- **Scoped Context Filter** - Filter-based scoped context for more precise knowledge retrieval during SQL generation. [#441](https://github.com/Datus-ai/Datus-agent/pull/441)
- **Direct Subagent Web Access** - New `--subagent` CLI parameter for launching subagents directly via the web interface. [#447](https://github.com/Datus-ai/Datus-agent/pull/447)
- **CLI Interaction UX** - Improved multiline input support and ellipsis truncation for better readability. [#468](https://github.com/Datus-ai/Datus-agent/pull/468)
- **Simplified Subagent Guidance** - Streamlined subagent usage instructions for clearer onboarding workflows. [#469](https://github.com/Datus-ai/Datus-agent/pull/469)
- **Hardened Function Tools** - Enforced read-only SQL execution, deduplicated tool registration, and improved docstrings. [#474](https://github.com/Datus-ai/Datus-agent/pull/474)
- **Current Date Injection** - Injected `current_date` directly into system prompts, removing the separate `get_current_date` tool. [#473](https://github.com/Datus-ai/Datus-agent/pull/473)
- **Data Compression** - Added response compression for `query_metrics` and fixed `DataCompressor` model_name handling to reduce token consumption. [#435](https://github.com/Datus-ai/Datus-agent/pull/435) [#472](https://github.com/Datus-ai/Datus-agent/pull/472)

**Bug Fixes**

- **Kimi-K2.5 & Qwen3-Coder-Plus Init** - Fixed temperature/top_p support for these models during interactive initialization. [#483](https://github.com/Datus-ai/Datus-agent/pull/483)
- **Generation Hooks Condition** - Fixed `generation_hooks` to use correct `where` expression condition. [#482](https://github.com/Datus-ai/Datus-agent/pull/482)
- **Ctrl+O Toggle** - Fixed missing response display for previous turns when toggling with Ctrl+O. [#477](https://github.com/Datus-ai/Datus-agent/pull/477)
- **Missing Tabulate Dependency** - Added missing `tabulate` dependency to pyproject.toml and requirements.txt. [#476](https://github.com/Datus-ai/Datus-agent/pull/476)
- **Skill Scan Paths** - Removed `~/.claude/skills` from default scan paths and improved config passing for ChatAgenticNode. [#475](https://github.com/Datus-ai/Datus-agent/pull/475)

**Documentation**

- Added Hive, Spark, ClickHouse, Trino database adapter docs. [#464](https://github.com/Datus-ai/Datus-agent/pull/464) [docs](adapters/db_adapters.md)
- Added resume/rewind command documentation. [#465](https://github.com/Datus-ai/Datus-agent/pull/465)

### 0.2.5

**New Features**

- **OpenAI Agent SDK 0.7.0 Upgrade with Kimi-2.5 & Gemini-3 Support** - Rebuilt the model layer with `litellm_adapter` and `sdk_patches`, enabling seamless integration with the latest Kimi-2.5 and Gemini-3 series models.
- **AgentSkills Support** - Introduced a complete Skill system with skill configuration, registration, management, and permission control, supporting both bash and function-based skill tools. [docs](integration/skills.md)
- **Tools as MCP Server** - Expose Datus database tools and context search as an MCP server, enabling integration with Claude Desktop, Claude Code, and other MCP-compatible clients. [docs](integration/mcp.md)

**Enhancements**

- **Semantic Tools Optimization** - Optimized semantic tools and context search for faster, more relevant results in the CLI.
- **Generation Prompt String Validation** - Strengthened string validation across multiple prompt templates for more reliable generation output.
- **Action-Based User Interaction Model** - Redesigned the CLI interaction layer to use a unified action-based model for execution, generation, and planning.
- **Reference SQL Parallelization & Date Support** - Parallelized reference SQL initialization for faster bootstrap, and enhanced date expression parsing. [docs](knowledge_base/reference_sql.md)
- **Bootstrap Markdown Summary** - Displays a formatted Markdown summary after bootstrap completion for quick review of generated results. [docs](getting_started/dashboard_copilot.md)
- **Subject Entry Deletion** - Added the ability to delete semantic models, metrics, and SQL summaries directly from the `/subject` screen. [docs](cli/context_command.md#subject)

**Bug Fixes**

- **Subject Node Race Condition** - Fixed a race condition when creating multiple subject nodes in parallel, improving concurrency safety.
- **Multi-Round Benchmark Evaluation** - Resolved issues in agent state, workflow runner, and configuration handling during multi-round evaluations. [docs](benchmark/benchmark_manual.md)
- **Attribution Analysis** - Simplified attribution analysis logic for clearer and more reliable results.

### 0.2.4

**Dashboard Copilot (Auto-generation)**

- Dashboard to Sub-Agent: Automatically generate sub-agents from BI dashboard configurations [#339](https://github.com/Datus-ai/Datus-agent/pull/339)
- Automatic semantic model generation during BI dashboard bootstrap [#368](https://github.com/Datus-ai/Datus-agent/pull/368)
- Generate metrics definitions directly from Dashboard components [#363](https://github.com/Datus-ai/Datus-agent/pull/363)

**Better Semantic Layer Integration**

- Semantic Adapter: Pluggable adapter for external metric layer integration [#355](https://github.com/Datus-ai/Datus-agent/pull/355)
- External Knowledge Storage: Vector-based knowledge retrieval for enhanced SQL generation context [#359](https://github.com/Datus-ai/Datus-agent/pull/359)
- Added SQL field to metrics schema definition [#364](https://github.com/Datus-ai/Datus-agent/pull/364)

**Enhancements**

- Optimized reference SQL search with deduplication and simplified format [#348](https://github.com/Datus-ai/Datus-agent/pull/348) [#358](https://github.com/Datus-ai/Datus-agent/pull/358) [#375](https://github.com/Datus-ai/Datus-agent/pull/375)
- Enhanced ContextSearch methods and display [#347](https://github.com/Datus-ai/Datus-agent/pull/347)
- Improved Plan Mode: Chat node inherits from GenSQL agentic node [#334](https://github.com/Datus-ai/Datus-agent/pull/334)
- Catalog screen improvements: column comments and nested table row styles [#345](https://github.com/Datus-ai/Datus-agent/pull/345) [#378](https://github.com/Datus-ai/Datus-agent/pull/378)
- Tool execution feedback with context and start events [#340](https://github.com/Datus-ai/Datus-agent/pull/340) [#341](https://github.com/Datus-ai/Datus-agent/pull/341)
- Enhanced prompt version handling [#367](https://github.com/Datus-ai/Datus-agent/pull/367) [#379](https://github.com/Datus-ai/Datus-agent/pull/379)
- Clean deprecated metric metadata and YAML directory on overwrite [#362](https://github.com/Datus-ai/Datus-agent/pull/362) [#365](https://github.com/Datus-ai/Datus-agent/pull/365)

**Refactoring**

- Semantic model and metrics architecture refactor [#350](https://github.com/Datus-ai/Datus-agent/pull/350)
- Unified subject tree management [#349](https://github.com/Datus-ai/Datus-agent/pull/349)
- Pluggable DB adapter architecture [#353](https://github.com/Datus-ai/Datus-agent/pull/353)
- Namespace config refactor [#346](https://github.com/Datus-ai/Datus-agent/pull/346)

**Bug Fixes**

- Fixed empty query_context in Superset charts [#372](https://github.com/Datus-ai/Datus-agent/pull/372)
- Skip render processing for tool calls in chatbot [#360](https://github.com/Datus-ai/Datus-agent/pull/360) [#380](https://github.com/Datus-ai/Datus-agent/pull/380)
- Fixed semantic model and metrics deduplication [#369](https://github.com/Datus-ai/Datus-agent/pull/369)
- Fixed subject_path parsing in context_search [#357](https://github.com/Datus-ai/Datus-agent/pull/357)
- Improved sample row error handling [#354](https://github.com/Datus-ai/Datus-agent/pull/354)

### 0.2.3

**New Features**

- **Embedded Tutorial Dataset** - California Schools dataset now bundled with installation and integrated into `datus-agent init` workflow for hands-on learning of contextual data engineering. [#277](https://github.com/Datus-ai/Datus-agent/issues/277) [tutorial](getting_started/contextual_data_engineering.md#part-2--hands-on-tutorial-california-schools)
- **Enhanced Evaluation Framework** - New evaluation command with expanded categories: Exact Match, Same Result Count (different values), Schema/Table Usage Match, and Semantic/Metric Layer Correctness. [#264](https://github.com/Datus-ai/Datus-agent/issues/264)
- **Plugin-Based Database Connector** - Refactored database connector to plugin-based architecture for easier extensibility and custom adapter development. [#284](https://github.com/Datus-ai/Datus-agent/issues/284)

**Enhancements**

- **Simplified Installation** - Removed legacy transformers dependency from default installation for faster setup and reduced package size. [#247](https://github.com/Datus-ai/Datus-agent/issues/247)
- **Streamlined MetricFlow Configuration** - Simplified configuration as MetricFlow now natively supports Datus config format. [#243](https://github.com/Datus-ai/Datus-agent/issues/243)
- **Built-in Generation Commands** - `/gen_semantic_model`, `/gen_metrics`, and `/gen_sql_summary` subagents now work out of the box without additional setup. [#250](https://github.com/Datus-ai/Datus-agent/issues/250)
- **Agentic Node Integration** - Workflow-based evaluations now support agentic nodes for more sophisticated testing scenarios. [#262](https://github.com/Datus-ai/Datus-agent/issues/262)
- **Code Quality Improvements** - Refactored tool modules and enhanced node logic. Unified `bootstrap-kb` and `gen_semantic_model` to use the same implementation. [#245](https://github.com/Datus-ai/Datus-agent/issues/245) [#250](https://github.com/Datus-ai/Datus-agent/issues/250)
- **Optimized Embedding Storage** - Refactored embedding model storage and updated dependencies for better performance. [#247](https://github.com/Datus-ai/Datus-agent/issues/247)

**Bug Fixes**

- **Schema Metadata Handling** - Fixed empty definition field in schema_linking command to ensure proper schema metadata is passed to downstream nodes. [#327](https://github.com/Datus-ai/Datus-agent/issues/327)
- **Initialization Issues** - Resolved multiple initialization bugs and corrected configuration file validation for tutorial mode. [#304](https://github.com/Datus-ai/Datus-agent/issues/304) [#303](https://github.com/Datus-ai/Datus-agent/issues/303)
- **Environment Variable Compatibility** - Fixed environment variable handling across different platforms for improved deployment compatibility. [#294](https://github.com/Datus-ai/Datus-agent/issues/294)
- **Evaluation Summary Generation** - Fixed summary generation in benchmark evaluation for more accurate evaluation reports. [#314](https://github.com/Datus-ai/Datus-agent/issues/314)
- **FastEmbed Cache Directory** - Fixed cache directory path for fastembed to resolve caching issues on different platforms. [#251](https://github.com/Datus-ai/Datus-agent/issues/251)

### 0.2.2

skipped

### 0.2.1

**New Features**

- **Web Chatbot Upgrade** - Added feedback collection, issue reporting, stream output, and `&hide_sidebar=true` parameter for embedding. [docs](web_chatbot/introduction.md)
- **Context Generation Commands** - New `/gen_semantic_model`, `/gen_metrics`, and `/gen_sql_summary` commands in subagents for dynamic knowledge base enrichment. [#192](https://github.com/Datus-ai/Datus-agent/issues/192) [docs](subagent/builtin_subagents.md)
- **Interactive Context Editing** - Visual editing support for `/catalog` and `/subject` commands to modify semantic models, metrics, and SQL summaries. [#219](https://github.com/Datus-ai/Datus-agent/issues/219) [#199](https://github.com/Datus-ai/Datus-agent/issues/199) [#175](https://github.com/Datus-ai/Datus-agent/issues/175) [docs](cli/context_command.md#subject)
- **Scoped Knowledge Base** - Subagents now support scoped KB initialization for better context isolation and management. [#217](https://github.com/Datus-ai/Datus-agent/issues/217)

**Enhancements**

- **MetricFlow Integration** - Load configuration from `env_settings.yml`, improved project detection, and cleaner output formatting. [#214](https://github.com/Datus-ai/Datus-agent/issues/214) [#216](https://github.com/Datus-ai/Datus-agent/issues/216) [docs](metricflow/introduction.md)
- **Flexible Model Configuration** - Support for multiple model providers and specifications in agent configuration. [#195](https://github.com/Datus-ai/Datus-agent/issues/195)
- **CLI Display Improvements** - Enhanced table width rendering for better SQL query readability. [#200](https://github.com/Datus-ai/Datus-agent/issues/200)
- **Improved Initialization** - Enhanced `datus-agent init` command with better error handling and setup flow. [#194](https://github.com/Datus-ai/Datus-agent/issues/194)

**Dependency Changes**

- `openai-agents` upgraded to 0.3.2 (requires manual update: `pip install -U openai-agents`)
- `datus-metricflow` updated to 0.1.2

### 0.2.0

**Enhanced Chat Functionality**

- Advanced multi-turn conversations for seamless interactions. [#91](https://github.com/Datus-ai/Datus-agent/issues/91)
- Agentic execution of database tools, file system operations, and automatic to-do list generation.
- Support for both automatic and manual compaction (/compact). [#125](https://github.com/Datus-ai/Datus-agent/issues/125)
- Session management with /resume and /clear commands.
- Provide dedicated context by introducing it with the `@table`, `@file`, `@metrics`, `@sql_history` commands. [#134](https://github.com/Datus-ai/Datus-agent/issues/134) [#152](https://github.com/Datus-ai/Datus-agent/issues/152)
- Token consumption tracking and estimation for better resource visibility. [#119](https://github.com/Datus-ai/Datus-agent/issues/119)
- Write-capability confirmations before executing sensitive tool actions.
- Plan Mode: An AI-assisted planning feature that generates and manages a to-do list. [#147](https://github.com/Datus-ai/Datus-agent/issues/147)

**Automatic Knowledge Base Building**

- Automatic generation of Metric YAML files in MetricFlow format from historical success stories. [#10](https://github.com/Datus-ai/Datus-agent/issues/10)
- Automatic summary and labeling SQL history files from *.sql files in workspace. [#132](https://github.com/Datus-ai/Datus-agent/issues/132)
- Improves SQL accuracy and generation speed using metrics & SQL history.

**MCP Extension**

- New /mcp commands to add, remove, list, and call MCP servers and tools. [#54](https://github.com/Datus-ai/Datus-agent/issues/54)

**Flexible Workflow Configuration**

- Fully customizable workflow definitions via agent.yml.
- Configurable nodes, models, and database connections.
- Support for sub-workflows and result selection to improve accuracy. [#88](https://github.com/Datus-ai/Datus-agent/issues/88)

**Context Exploration**

- Improve `/catalog` to display all databases, schemas, and tables across multiple databases.
- New /subject to show all metrics built with MetricFlow. [#165](https://github.com/Datus-ai/Datus-agent/issues/165)
- Context search tools integration to enhance recall of metadata and metrics. [#138](https://github.com/Datus-ai/Datus-agent/issues/138)

**User Behavior Logging**

- Automatic collection of user behavior logs.
- Transforms human–computer interaction data into trainable datasets for future improvements.

## 0.1

### 0.1.0

**Datus-cli**

- Supports connecting to SQLite, DuckDB, StarRocks, and Snowflake, and performing common command-line operations.
- Supports three types of command extensions: !run_command, @context, and /chat to enhance development efficiency.

**Datus-agent**

- Supports automatic NL2SQL generation using the React paradigm.
- Supports retrieving database metadata and building vector-based search on metadata.
- Supports deep reasoning via the MCP server.
- Supports integration with bird-dev and spider2-snow benchmarks.
- Supports saving and restoring workflows, allowing execution context and node inputs/outputs to be recorded.
- Offers flexible configuration: you can define multiple models, databases, and node execution strategies in Agent.yaml.

### 0.1.2

**Datus-cli**

- Added a fix node: use `!fix` to quickly fix the last SQL error, with a focused template for the LLM.

**Datus-agent**

- Performance improvement for bootstrap-kb with multi-threading.
- Other minor bug fixes.

### 0.1.3

**Datus-cli**

- Added datus-init to initialize the ~/.datus/ directory.
- Included a sample DuckDB database in ~/.datus/sample.

**Datus-agent**

- Added the check_result option to the output node (default: False).

### 0.1.4

**Datus-agent**

- Added the check-mcp command to confirm the MCP server configuration and availability.
- Added support for both DuckDB and SQLite MCP servers.
- Implemented automatic installation of the MCP server into the datus-mcp directory.

### 0.1.5

**Datus-agent**

- Automated semantic layer generation.
- Introduced a new internal workflow: metrics2SQL.
- Added save_llm_trace to facilitate training dataset collection.

**Datus-cli**

- Enhanced !reason and !gen_semantic_model commands for a more agentic and intuitive experience.
