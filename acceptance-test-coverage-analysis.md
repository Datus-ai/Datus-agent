# Acceptance 测试覆盖分析

> 分析日期：2026-02-09

## 现状概览

项目共有 **30 个 acceptance 测试**，分布在 8 个测试文件中。

当前测试基础设施：
- pytest.ini 中仅定义了 `acceptance` 一个自定义 marker
- CI（`run-ut.yml`）通过 `pytest -m acceptance tests/` 执行
- 集成测试（`run-integration.yml`、`run-integration-bi.yml`）按计划每日运行
- `tests/e2e/` 目录存在但为空
- 外部依赖通过 `@pytest.mark.skipif(not os.getenv("API_KEY"))` 做条件跳过

---

## Nightly 测试实施进度（P0~P3 全部完成）

> 更新日期：2026-02-13

### 总览：21 个模块、214 个 nightly 测试全部收集

运行命令：`pytest --collect-only -m nightly tests/`（214/1134 tests collected）

---

### P0+P1：新建 nightly 测试 + 现有文件 marker（25 tests）

#### P0+P1 新建文件

| 文件 | 测试数 | 覆盖场景 | 说明 |
|------|--------|---------|------|
| `tests/nightly/__init__.py` | — | 基础设施 | 空文件，标识 nightly 包 |
| `tests/nightly/conftest.py` | — | 基础设施 | 公共 fixtures：`agent_config`（含 `deepcopy` 隔离）、`snowflake_config` |
| `tests/nightly/test_nightly_api.py` | 10 | N8-01~N8-08 | FastAPI TestClient + httpx.AsyncClient，覆盖健康检查、JWT 认证、同步/异步/流式 workflow、feedback、错误场景 |
| `tests/nightly/test_nightly_bootstrap_kb.py` | 7 | N1-01~N1-09 | 直接调用 `SubAgentBootstrapper`，覆盖 metadata/metrics/reference_sql/overwrite/incremental/wildcard |
| `tests/nightly/test_nightly_init.py` | 3 | N4-01/03/04 | mock `InteractiveInit`，覆盖 LLM 配置探测、配置文件生成、可选组件初始化 |
| `tests/nightly/test_nightly_chat_agentic.py` | 3 | N5-02/03/05 | DatusCLI + 真实 DeepSeek LLM，覆盖多轮对话上下文、多工具组合调用、流式响应 |

#### P0+P1 修改的现有文件

| 文件 | 变更 | 说明 |
|------|------|------|
| `tests/test_sub_agent.py` | 重写 `test_plan` + `test_overwrite` | 修复 `SUPPORTED_COMPONENTS` 数量断言（3→动态）、位置索引→名称过滤、scoped context 路径修正、添加 `@pytest.mark.nightly` |
| `tests/test_init_util.py` | 添加 `@pytest.mark.nightly` | `TestDatabaseConnectivity` 3 个测试 |
| `tests/test_cli_rich.py` | 添加 `@pytest.mark.nightly` | `test_chat_command` |
| `tests/conf/agent.yml` | chatbot 节点 `max_turns: 5` | 限制 agent 工具调用轮次，防止 DeepSeek 131K 上下文溢出 |
| `pyproject.toml` | 注册 `nightly` marker | 避免 pytest 未知 marker 警告 |

#### P0+P1 关键技术决策

1. **conftest `deepcopy` 隔离**：bootstrap 测试向 `agentic_nodes` 添加 `SubAgentConfig` 对象会污染 `configuration_manager` 缓存，导致后续 chat 测试报 `AttributeError: 'SubAgentConfig' object has no attribute 'get'`。通过 `copy.deepcopy(config.agentic_nodes)` 隔离。

2. **Scoped context 路径**：测试数据 subject tree 根节点为 `california_schools`（非 `education`），children 包括 `Charter-Fund`、`SAT_Score`、`Students_K-12` 等。匹配计数：metadata ≥ 3 tables、metrics = 1、reference_sql = 13。

3. **DeepSeek 上下文限制**：bird_school 数据集 + 多工具调用易超出 DeepSeek 131K token 限制。解决方案：`max_turns: 5` + `parse_at_context` mock（返回空列表避免预加载大量 schema）。

4. **名称过滤替代位置索引**：`SUPPORTED_COMPONENTS` 从 3 增到 4（添加 `semantic_model`），所有组件断言改用 `[r for r in results if r.component == "name"][0]` 替代 `results[0]`。

---

### P2：补充 N7/N10/N11/N12 新建测试（22 tests）

#### P2 新建文件

| 文件 | 测试数 | 覆盖场景 | 说明 |
|------|--------|---------|------|
| `tests/nightly/test_nightly_sub_agent.py` | 3 | N7-01/02/04 | Stub 模式测试 SubAgentManager：创建、列表获取、删除 + scoped KB 清理 |
| `tests/nightly/test_nightly_mcp_client.py` | 5 | N10-03/04/07/08/09 | MCP HTTP 客户端端到端：read_query、describe_table、错误处理、大结果集、并发调用 |
| `tests/nightly/test_nightly_func_tools.py` | 10 | N11-01~N11-10 | 真实 DuckDB/SQLite 上的 DBFuncTool 端到端：list_tables、describe_table、get_table_ddl、read_query、失败场景、search_table、scoped 过滤、multi-connector |
| `tests/nightly/test_nightly_cli_search.py` | 4 | N12-04/05/06/07 | AgentCommands CLI 搜索：`!sd` 文档搜索、`!sl` 无结果处理、`!sq` subject_path 过滤、`!sd` 无效 platform |

#### P2 关键技术决策

1. **StubAgentConfig 兼容**：`SubAgentManager` 依赖 `agent_config.db_type` 和 `agent_config.agentic_nodes`，Stub 需要补充 `db_type = "sqlite"` 和 `agentic_nodes = {}`。

2. **describe_table 非严格错误**：SQLite 对不存在的表调用 `describe_table` 返回 `success=1` + 0 列（空结果），而非 `success=0` 错误。断言需适配此行为。

3. **MCP HTTP 会话管理**：使用 `mcp_http_session` 上下文管理器包装 HTTP streamable 传输层，复用 `test_integration_mcp_server.py` 的 server 启动工具函数。

---

### P3：现有文件添加 `@pytest.mark.nightly` marker（167 tests from 10 files）

#### P3 N2（bootstrap-bi）marker 添加

| 文件 | 测试函数/类 | 新增测试数 |
|------|-----------|-----------|
| `tests/test_bi_dashboard.py` | `test_workflow_without_llm`（alongside existing `@pytest.mark.acceptance`） | 1 |
| `tests/integration/test_integration_bi_dashboard.py` | `test_complete_workflow` | 1 |

#### P3 N3（tutorial）marker 添加

| 文件 | 测试函数 | 新增测试数 |
|------|---------|-----------|
| `tests/test_tutorial.py` | `test_run_returns_1_when_ensure_config_fails`、`test_run_returns_1_on_exception`、`test_run_completes_successfully`、`test_init_success_story_metrics_returns_error_on_exception`、`test_init_reference_sql_reports_process_errors`（均 alongside existing `@pytest.mark.acceptance`） | 5 |

#### P3 N6（MCP Server）marker 添加

| 文件 | 测试类 | 新增测试数 |
|------|--------|-----------|
| `tests/test_mcp_server.py` | `TestMCPServerCreation`、`TestMCPToolRegistration`、`TestDynamicRouterPathParsing` | 10 |
| `tests/integration/test_integration_mcp_server.py` | `TestStaticModeHTTPStreamable`、`TestStaticModeSSE`、`TestStaticModeStdio`（alongside existing `@pytest.mark.asyncio`） | 21 |

#### P3 N9（Skill）marker 添加

| 文件 | 方式 | 新增测试数 |
|------|------|-----------|
| `tests/skills/test_skill_registry.py` | 模块级 `pytestmark = pytest.mark.nightly` | 17 |
| `tests/skills/test_permission_manager.py` | 模块级 `pytestmark = pytest.mark.nightly` | 20 |
| `tests/skills/test_skill_bash_tool.py` | 模块级 `pytestmark = pytest.mark.nightly` | 27 |
| `tests/skills/test_skill_manager.py` | 模块级 `pytestmark = pytest.mark.nightly` | 30 |
| `tests/integration/test_integration_skill.py` | 类级 `@pytest.mark.nightly` 添加到 6 个测试类 | 30 |

> **注**：N9 skill 文件原本使用 `pytestmark = pytest.mark.nightly` 模块级标记（单元测试）或 `@pytest.mark.nightly` 类级标记（集成测试，alongside existing `@pytest.mark.acceptance`）。

---

### 全场景覆盖率总览

| 场景组 | 总场景数 | 实施前覆盖 | 当前实现 | 新覆盖率 | 实施批次 |
|--------|---------|-----------|---------|---------|---------|
| N1 bootstrap-kb | 9 | 22% | 7 新建 + 2 修改 | **100%** | P0+P1 |
| N2 bootstrap-bi | 3 | 100% | 2 marker 添加 | **100%** | P3 |
| N3 tutorial | 3 | 100% | 5 marker 添加 | **100%** | P3 |
| N4 init | 4 | 25% | 3 新建 + 1 修改 | **100%** | P0+P1 |
| N5 chat_agentic | 5 | 40% | 3 新建 + 1 修改 | **80%** | P0+P1 |
| N6 MCP Server | 6 | 83% | 31 marker 添加 | **100%** | P3 |
| N7 Sub-agent | 6 | 50% | 3 新建 + 2 修改 | **83%** | P2 |
| N8 API | 8 | 0% | 10 新建 | **100%** | P0+P1 |
| N9 Skill | 7 | 100% | 124 marker 添加 | **100%** | P3 |
| N10 MCP Client | 10 | 60% | 5 新建 + 6 修改 | **100%** | P2 |
| N11 Func Tools | 26 | 65% | 10 新建 | **75%** | P2 |
| N12 CLI Search | 7 | 43% | 4 新建 + 1 修改 | **71%** | P2 |
| **合计** | **94** | **55%** | **214 tests / 21 modules** | **92%** |

### 按文件汇总

| 模块 | 测试数 | 来源 |
|------|--------|------|
| `tests/nightly/test_nightly_api.py` | 10 | P0 新建 |
| `tests/nightly/test_nightly_bootstrap_kb.py` | 7 | P1 新建 |
| `tests/nightly/test_nightly_chat_agentic.py` | 3 | P1 新建 |
| `tests/nightly/test_nightly_init.py` | 3 | P1 新建 |
| `tests/nightly/test_nightly_sub_agent.py` | 3 | P2 新建 |
| `tests/nightly/test_nightly_mcp_client.py` | 5 | P2 新建 |
| `tests/nightly/test_nightly_func_tools.py` | 10 | P2 新建 |
| `tests/nightly/test_nightly_cli_search.py` | 4 | P2 新建 |
| `tests/test_sub_agent.py` | 2 | P1 marker |
| `tests/test_init_util.py` | 3 | P1 marker |
| `tests/test_cli_rich.py` | 2 | P1 marker |
| `tests/test_bi_dashboard.py` | 1 | P3 marker |
| `tests/integration/test_integration_bi_dashboard.py` | 1 | P3 marker |
| `tests/test_tutorial.py` | 5 | P3 marker |
| `tests/test_mcp_server.py` | 10 | P3 marker |
| `tests/integration/test_integration_mcp_server.py` | 21 | P3 marker |
| `tests/skills/test_skill_registry.py` | 17 | P3 marker |
| `tests/skills/test_permission_manager.py` | 20 | P3 marker |
| `tests/skills/test_skill_bash_tool.py` | 27 | P3 marker |
| `tests/skills/test_skill_manager.py` | 30 | P3 marker |
| `tests/integration/test_integration_skill.py` | 30 | P3 marker |
| **合计** | **214** | |

---

## 第一部分：已覆盖场景

### 1. CLI 交互命令 (9 tests) — `tests/test_cli_rich.py`

| 测试函数 | 场景 |
|----------|------|
| `test_schema_linking` | `!sl` schema linking 命令执行与结果表展示 |
| `test_search_reference_sql` | `!sq` / `!search_sql` 搜索参考 SQL |
| `test_search_metrics` | `!sm` 搜索 metrics |
| `test_bash_command_allowed` | `!bash ls` 白名单命令正常执行 |
| `test_bash_command_denied` | `!bash rm` 危险命令被拦截 |
| `test_databases_command` | `.databases` 显示数据库信息 |
| `test_tables_command` | `.tables` 显示表信息 |
| `test_chat_command` | `/chat` 多轮对话 + 上下文记忆 + tool call |
| `test_chat_info` | `.chat_info` 查看上次会话信息 |

### 2. DuckDB 连接器 (5 tests) — `tests/test_connector_duckdb.py`

| 测试函数 | 场景 |
|----------|------|
| `test_get_table_with_ddl` | 获取表 DDL 定义 + 样例行 |
| `test_get_views_with_ddl` | 获取视图 DDL 定义 |
| `test_get_table_schema` | 获取表 schema + 不存在表的异常处理 |
| `test_execute_query` | SQL 查询执行（成功 + 不存在表的失败场景） |
| `test_get_schemas` | 获取 schema 列表（存在/不存在的数据库） |

### 3. DeepSeek 模型 (4 tests) — `tests/test_deepseek_model.py`

| 测试函数 | 场景 |
|----------|------|
| `test_generate_acceptance` | 基础文本生成能力（多 prompt） |
| `test_generate_with_mcp_acceptance` | MCP 工具调用（SSB 业务分析：收入、折扣、利润率） |
| `test_generate_with_mcp_stream_acceptance` | MCP 流式调用 + action 生成验证 |
| `test_generate_with_mcp_token_consumption` | Token 消耗追踪（stream vs non-stream 对比） |

### 4. 核心 Node 执行 (4 tests) — `tests/test_node.py`

| 测试函数 | 场景 |
|----------|------|
| `test_generation_node` | SQL 生成节点（DeepSeek + Snowflake，含输入校验） |
| `test_reflection_node` | 反思节点（从 YAML 加载测试用例，执行反思分析） |
| `test_execution_node` | SQL 执行节点（Snowflake，多测试用例） |
| `test_compare_with_mcp_node` | 对比节点 + MCP 流式（JOIN 识别 + 数据库建议） |

### 5. Tutorial / 初始化 (5 tests) — `tests/test_tutorial.py`

| 测试函数 | 场景 |
|----------|------|
| `test_run_returns_1_when_ensure_config_fails` | 配置文件缺失时 benchmark 返回错误码 |
| `test_run_returns_1_on_exception` | RuntimeError 时 benchmark 返回错误码 |
| `test_run_completes_successfully` | benchmark 完整流程（config → files → metadata → SQL → metrics） |
| `test_init_success_story_metrics_returns_error_on_exception` | metrics 初始化异常处理 |
| `test_init_reference_sql_reports_process_errors` | reference SQL 初始化错误上报 |

### 6. 其他 (3 tests)

| 文件 | 测试函数 | 场景 |
|------|----------|------|
| `tests/test_bi_dashboard.py` | `test_workflow_without_llm` | BI Dashboard workflow 集成（mock LLM，真实 Superset API） |
| `tests/test_tools_output.py` | `test_output` | 跨 benchmark（平台/数据库）的输出生成 |
| `tests/test_configuration_load.py` | `test_config_exception` | 配置加载异常（文件缺失、无效 node 类型、不支持的值） |

---

## 第二部分：覆盖缺口分析

### 缺口 1：数据库类型覆盖

项目支持多种数据库（`DBType` 枚举），acceptance 测试仅覆盖 2 种。以下列出主要数据库的覆盖状态：

| 数据库 | 覆盖状态 |
|--------|---------|
| DuckDB | 有专门的 connector 测试 |
| Snowflake | node 测试中使用 |
| PostgreSQL | 未覆盖 |
| MySQL | 未覆盖 |
| StarRocks | 未覆盖 |
| SQLite (作为 connector) | 未覆盖 |

**影响**：无法验证各数据库 connector 的 DDL 解析、schema 获取、查询执行等功能在不同数据库上的兼容性。

### 缺口 2：Agentic Node 体系

`datus/agent/node/` 下共有 8 个 agentic node（不含基类 `agentic_node.py`），acceptance 测试仅通过 `test_compare_with_mcp_node` 覆盖了 `compare_agentic_node`，其余 7 个完全无覆盖：

| Node | 功能 | 重要程度 |
|------|------|---------|
| `chat_agentic_node` | 聊天式 agent 交互 | 高 |
| `gen_sql_agentic_node` | agent 驱动的 SQL 生成 | 高 |
| `sql_summary_agentic_node` | SQL 结果摘要 | 中 |
| `gen_semantic_model_agentic_node` | 语义模型生成 | 中 |
| `gen_metrics_agentic_node` | metrics 生成 | 中 |
| `gen_ext_knowledge_agentic_node` | 外部知识生成 | 低 |
| `gen_report_agentic_node` | 报告生成 | 低 |

### 缺口 3：API / MCP 接口

| 接口 | 覆盖状态 | 说明 |
|------|---------|------|
| FastAPI (`datus/api/`) | 无 acceptance 测试 | 端点可用性、请求/响应格式、错误处理完全未覆盖 |
| MCP Server (`datus/mcp_server.py`) | 有测试但未标记 acceptance | 工具注册、调用、多 namespace 支持未作为 acceptance 验证 |
| Web UI (Streamlit) | 无测试 | 交互流程无覆盖 |

### 缺口 4：Storage / 知识库

| 模块 | 未覆盖场景 |
|------|-----------|
| `schema_metadata/` | 向量搜索精度、embedding 质量验证 |
| `semantic_model/` | 语义模型创建、更新、查询 |
| `reference_sql/` | few-shot SQL 检索质量、相关性排序 |
| `ext_knowledge/` | 外部知识检索排序、LanceDB 向量搜索 |
| `feedback/` | 反馈闭环（upvote → 优化 → 验证） |
| `subject_tree/` | 层级导航、目录结构展示 |

### 缺口 5：Tool 系统

| 工具类别 | 未覆盖场景 |
|---------|-----------|
| `skill_tools/` | 自定义 skill 完整执行流程 |
| `permission/` | 细粒度权限控制（除 bash 白名单外） |
| 工具链 | 多工具链式调用 |
| 错误恢复 | 工具执行失败后的恢复策略 |


### 缺口 6：LLM 模型覆盖

仅测试了 DeepSeek 模型，以下模型无 acceptance 测试：

| 模型 | 实现文件 | 覆盖状态 |
|------|---------|---------|
| Claude (Anthropic) | `datus/models/claude_model.py` | 无 |
| OpenAI (GPT) | `datus/models/openai_model.py` | 无 |
| Gemini | `datus/models/gemini_model.py` | 无 |
| Qwen | `datus/models/qwen_model.py` | 无 |
| Kimi | `datus/models/kimi_model.py` | 无 |

---

## 第三部分：测试分层分类建议

### 分层设计原则

```
                    ┌─────────────────────────────┐
                    │     Regression (发版前)       │  全量，含外部数据库 + 多模型
                    │     @pytest.mark.regression  │
                    ├─────────────────────────────┤
                    │     E2E / Nightly (每日)      │  真实 LLM + 本地数据库
                    │     @pytest.mark.nightly     │
                    ├─────────────────────────────┤
                    │     CI (每次 PR)              │  mock LLM，本地数据库，快速
                    │     @pytest.mark.ci          │
                    └─────────────────────────────┘
```

> 关于术语：发版前的全量测试一般称为 **回归测试 (Regression Testing)**，目标是确保新变更没有破坏已有功能。也有人称之为 **发布验证测试 (Release Qualification Testing)**。

### 第一层：CI 测试（PR 触发，每次必跑）

#### 设计原则

CI 测试的本质是**回归守护**——每次代码变更时快速验证核心逻辑没有被破坏。一个测试是否适合放在 CI，用以下 5 条标准判断：

1. **零外部依赖**：不调用 LLM API，不访问远程数据库，不下载模型文件
2. **零预构建数据**：不需要 LanceDB 索引、不需要 FastEmbed 向量化、不需要运行 `build_test_data.sh`
3. **确定性**：相同代码必须产生相同结果，不存在随机性或时序依赖
4. **快速**：单个测试 < 5 秒，总套件 < 3 分钟
5. **仅依赖 pip install**：安装完 Python 依赖包即可运行，加上项目自带的少量测试数据文件（如 `sample_data/duckdb-demo.duckdb`）

**触发时机**：每次 PR 提交 / push

**运行方式**：`pytest -m ci tests/`

#### 当前问题

项目已有 **~278 个 unit tests**（`tests/unit_tests/`），全部使用 mock，完全满足上述 5 条标准，但**没有被纳入 CI**。当前 CI 通过 `pytest -m acceptance` 运行，这些 unit tests 没有 `acceptance` 标记。

与此同时，当前 32 个 acceptance 测试中混入了不适合 CI 的测试：
- `test_cli_rich.py` 的 `!sl`/`!sq`/`!sm` 命令——需要预构建的 LanceDB 索引（违反原则 2）
- `test_node.py` 全部 4 个测试——调用真实 LLM API（违反原则 1）
- `test_deepseek_model.py` 全部 4 个测试——调用真实 DeepSeek API（违反原则 1）

#### A. 已有测试纳入 CI

> **注意**：不应将所有已有测试无差别地标记为 `ci`。需要先梳理出核心流程（配置加载、数据库连接、CLI 基础命令、workflow 定义解析等）对应的测试，确认这些核心路径已被覆盖。对于覆盖不足的核心路径，应在 B 节中补齐。非核心的辅助测试（如边缘格式转换、冷门工具函数）可暂缓纳入。

以下为满足 CI 五原则且覆盖核心路径的现有测试（~295 个）：

| 来源 | 测试数 | 覆盖的核心路径 |
|------|--------|--------------|
| `tests/unit_tests/` | ~278 | DB tools (55)、Subject tree (63)、BI adapter (75)、SQL utils (21)、PyArrow (19)、Context search (14) 等。100% mock，无外部依赖 |
| `test_configuration_load.py` | 10 | 配置解析、namespace 校验、benchmark 路径、异常路径 |
| `test_tutorial.py` | 5 | Tutorial 初始化逻辑（Dummy stub，完全 mock） |
| `test_connector_duckdb.py` | 5 | DuckDB connector：DDL / schema / 查询执行 |
| `test_cli_rich.py` 部分 | ~4 | CLI 基础命令：`!bash` (mock subprocess)、`.databases`/`.tables` (本地 connector) |

#### B. 建议新增的 CI 测试

| 编号 | 测试场景 | 对应缺口 | 说明 |
|------|---------|---------|------|
| CI-01 | Workflow 定义校验 | 缺口 1 | 验证 `workflow.yml` 解析正确、所有引用的 node 已注册、节点顺序合法 |
| CI-02 | Context 传递逻辑 | 缺口 1 | 验证 SQLContext/DOCContext/MetricContext 在节点间正确传递和合并 |
| CI-03 | 配置加载：namespace 切换 | — | 不存在的 namespace 报错、环境变量 `${VAR}` 展开 |
| CI-04 | 配置加载：模型定义校验 | — | 无效模型类型、缺失字段等异常路径 |
| CI-05 | Tool 权限系统 | 缺口 6 | permission 模块对各工具类型的细粒度控制 |
| CI-06 | 日期解析 | 缺口 3 | "去年Q3"、"最近7天"、"2024年1月" 等表达式解析（纯逻辑） |
| CI-07 | Node 输入校验 | 缺口 3 | Pydantic 模型校验：缺失必填字段、类型错误的异常处理 |
| CI-08 | FastAPI 端点契约 | 缺口 4 | TestClient 验证请求/响应格式、认证、错误码（mock Agent） |
| CI-09 | MCP Server 工具注册 | 缺口 4 | 验证工具列表、schema 定义、参数校验（不启动完整 Agent） |
| CI-10 | SQLite connector | 缺口 2 | DDL、schema、查询执行（与 DuckDB 测试对齐，使用本地 SQLite 文件） |
| CI-11 | Skill 定义加载 | 缺口 6 | 自定义 skill 的 YAML 定义加载、参数校验 |

#### C. 从当前 acceptance 移出 CI → 移入 Nightly

| 当前测试 | 测试数 | 移出原因 |
|---------|--------|---------|
| `test_cli_rich.py` 的 `!sl`/`!sq`/`!sm` | 3 | 需要预构建 LanceDB 索引（违反原则 2） |
| `test_cli_rich.py` 的 `/chat` + `.chat_info` | 2 | 涉及 chat 上下文管理，依赖存储初始化 |
| `test_node.py` 全部 | 4 | 调用真实 LLM API（违反原则 1） |
| `test_deepseek_model.py` 全部 | 4 | 调用真实 DeepSeek API（违反原则 1） |
| `test_bi_dashboard.py` | 1 | 需要 BI superset 配置（违反原则 2） |
| `test_tools_output.py` | 1 | 跨 benchmark 输出，依赖复杂初始化 |

#### CI 总览

| 类别 | 测试数 | 状态 |
|------|--------|------|
| A. 已有测试纳入 | ~302 | 仅需添加 `@pytest.mark.ci` 标记 |
| B. 新增测试 | ~11 | 需要新编写 |
| **CI 总计** | **~313** | |
| C. 移出至 Nightly | ~15 | 需要改标记 |

### 第二层：Nightly 测试（每日定时，端到端）

**目标**：验证 12 大核心功能的端到端质量。使用真实 LLM + 本地数据库。

**触发时机**：每日定时（如 UTC 14:00，与现有 `run-integration.yml` 对齐）

**运行方式**：`pytest -m nightly tests/`

**依赖约束**：
- 使用真实 LLM API（需配置 API Key）
- 使用本地数据库（DuckDB / SQLite 内置测试数据）
- 不依赖外部数据库实例（Snowflake / PostgreSQL 等）
- 例外：N2（bootstrap-bi）需要连接 BI 平台（如 Superset），属于外部服务依赖

**执行方式**：以 Python 代码级调用为主，不使用 shell 包装。原因：

1. 项目 CLI 解析层很薄（argparse → 核心类），不值得通过 subprocess 测试
2. 代码级调用可以断言返回对象和中间状态（如 `SQLContext`、LanceDB 索引文件），shell 只能检查 returncode/stdout
3. 现有测试（`test_bi_dashboard.py`）已采用此模式——直接构造 `BiDashboardCommands` 调用方法

各场景的调用入口：

| 场景 | 调用入口 | 断言目标 |
|------|---------|---------|
| N1 bootstrap-kb | `Agent.bootstrap_kb()` | LanceDB 索引文件、embedding 数量 |
| N2 bootstrap-bi | `BiDashboardCommands` 方法 | chart 提取数、SQL 生成结果 |
| N3 tutorial | `BenchmarkTutorial.run()` | SQL 输出、执行结果 |
| N4 init | `InteractiveInit`（mock stdin） | 配置文件生成、namespace 写入 |
| N5 chat_agentic | `Agent.run()` + chat workflow | 回复内容、tool call 记录 |
| N6 MCP Server | MCP client SDK 连接 | 工具列表、调用响应 |
| N7 Sub-agent | `Agent` 子 agent 创建 | sub-agent 配置、隔离性 |
| N8 API | FastAPI `TestClient` | HTTP 响应体、状态码 |
| N9 Skill | `AgentCommands` skill 执行 | skill 输出、上下文传递 |
| N10 MCP Client | `MCPToolManager` | 工具发现、调用结果 |
| N11 Func Tools | `Tool.execute()` | 工具返回值、副作用 |
| N12 CLI search | `AgentCommands` 搜索方法 | 返回结果列表、排序 |

#### N1. bootstrap-kb 知识库初始化

入口：`datus-agent bootstrap-kb`（`Agent.bootstrap_kb()`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N1-01 | metadata 初始化 | `--components metadata`：表/列 embedding 构建，验证 LanceDB 索引生成 |
| N1-02 | metrics 初始化（success story） | `--components metrics --success_story`：从 CSV 导入 metrics |
| N1-03 | metrics 初始化（semantic YAML） | `--components metrics --semantic_yaml`：从 YAML 导入 metrics |
| N1-04 | semantic_model 初始化 | `--components semantic_model`：MetricFlow 语义模型构建 |
| N1-05 | reference_sql 初始化 | `--components reference_sql --sql_dir`：SQL 文件索引 + validate-only 模式 |
| N1-06 | ext_knowledge 初始化 | `--components ext_knowledge --ext_knowledge`：外部知识文档导入 |
| N1-07 | 多组件联合初始化 | `--components metadata,metrics,reference_sql`：多组件一次性初始化 |
| N1-08 | incremental 更新策略 | `--kb_update_strategy incremental`：增量更新不破坏已有数据 |
| N1-09 | overwrite 更新策略 | `--kb_update_strategy overwrite`：全量重建 |

#### N2. bootstrap-bi BI 看板集成

入口：`datus-agent bootstrap-bi`（`BiDashboardCommands.cmd()`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N2-01 | BI 看板解析 | 连接 BI 平台 → 提取图表列表 → 解析 SQL 和上下文 |
| N2-02 | Sub-agent 创建 | 从看板自动创建 scoped sub-agent + 初始化 reference_sql 和 metrics |
| N2-03 | Dashboard Assembler | 图表 SQL 提取 → 表上下文映射 → 系统 prompt 生成 |

#### N3. datus-agent tutorial 教程流程

入口：`datus-agent tutorial`（`BenchmarkTutorial.run()`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N3-01 | 完整教程流程 | 6 步全流程：数据准备 → metadata → semantic_model → metrics → reference_sql → 执行任务 |
| N3-02 | 教程配置自动创建 | 验证 california_schools namespace 自动添加、SQLite 数据库正确复制 |
| N3-03 | 教程任务执行 | 验证样例 SQL 任务在 california_schools 上端到端执行成功 |

#### N4. datus-agent init 初始化向导

入口：`datus-agent init`（`InteractiveInit.run()`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N4-01 | LLM 配置 + 连通性 | 选择 LLM provider → 输入 API Key → `probe_llm()` 验证连通 |
| N4-02 | 数据库配置 + 连通性 | 添加 namespace → 输入连接信息 → `check_db()` 验证连通 |
| N4-03 | 配置文件生成 | 验证 `conf/agent.yml` 正确生成，含 model 定义 + namespace 定义 |
| N4-04 | 可选组件初始化 | 向导中选择初始化 metadata/metrics 等组件，验证流程衔接 |

#### N5. chat_agentic workflow 聊天式交互

入口：workflow.yml `chat_agentic: [chat, execute_sql, output]`（`ChatAgenticNode`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N5-01 | 单轮 SQL 生成 | 自然语言问题 → tool call（schema 搜索 + metrics 搜索）→ SQL 生成 → 执行 |
| N5-02 | 多轮对话上下文 | 追问 / 修改条件，验证会话上下文保持 |
| N5-03 | Tool call 组合 | 验证 `db_func_tool` + `context_tool` 的组合调用（search_table → describe_table → read_query） |
| N5-04 | Skill 加载与执行 | 聊天中通过 `skill_func_tool` 加载自定义 skill 并执行 |
| N5-05 | 流式响应 | 验证 streaming 模式下的 action 生成和结果输出 |

#### N6. MCP Server 管理功能

入口：`datus-mcp`（`DatusMCPServer` / `LightweightDynamicMCPServer`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N6-01 | 静态模式工具注册 | 单 namespace 启动，验证 DB 工具 + Context 工具全部注册 |
| N6-02 | DB 工具端到端 | `list_tables` → `describe_table` → `get_table_ddl` → `read_query` 链路 |
| N6-03 | Context 工具端到端 | `search_metrics` → `get_metrics`、`search_reference_sql` → `get_reference_sql` |
| N6-04 | 动态模式多 namespace | 动态路由 `/mcp/{namespace}`，验证不同 namespace 返回不同工具上下文 |
| N6-05 | 动态模式 subagent 路由 | `?subagent={name}` 参数，验证 scoped 工具上下文 |
| N6-06 | Transport 模式 | 分别测试 stdio / sse / http 三种传输模式的可用性 |

#### N7. Sub-agent 管理功能

入口：`SubAgentManager`（`datus/utils/sub_agent_manager.py`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N7-01 | 创建 sub-agent | `save_agent(config)` → 验证 `conf/agent.yml` 的 `agentic_nodes` 写入 |
| N7-02 | 列表与获取 | `list_agents()` → `get_agent(name)` → 验证配置完整性 |
| N7-03 | Sub-agent KB 初始化 | `bootstrap_agent(config, components)` → 验证 scoped LanceDB 在 `sub_agents/{name}/` 下生成 |
| N7-04 | 删除 sub-agent | `remove_agent(name)` → 验证配置移除 + scoped KB 清理 |
| N7-05 | 重命名 sub-agent | `save_agent(config, previous_name)` → 验证配置更新 + KB 迁移 |
| N7-06 | Sub-agent 执行 | 通过 chat_agentic workflow 使用 scoped sub-agent 执行查询 |

#### N8. Agent API 调用

入口：`DatusAPIService`（`datus/api/service.py`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N8-01 | 健康检查 | `GET /health` → 验证 DB + LLM 连通性状态返回 |
| N8-02 | 认证流程 | `POST /auth/token` → 获取 JWT → 后续请求携带 Bearer token |
| N8-03 | 同步 workflow 执行 | `POST /workflows/run` mode=sync → 验证 SQL 生成 + 执行 + 结果返回 |
| N8-04 | 异步 workflow 执行 | `POST /workflows/run` mode=async → 验证 task_id 返回 + 状态查询 |
| N8-05 | 流式 workflow 执行 | `POST /workflows/stream` → 验证 SSE 事件序列：started → sql_generated → execution_complete → done |
| N8-06 | Feedback 记录 | `POST /workflows/feedback` → 验证 task_id 关联的反馈存储 |
| N8-07 | 多 namespace 切换 | 同一 API 实例，不同 namespace 参数 → 验证 agent 池正确路由 |
| N8-08 | 错误场景 | 无效 namespace / 无效 workflow / SQL 执行失败 → 验证错误码和错误信息 |

#### N9. Skill 管理与调用

入口：`SkillManager` + `SkillRegistry`（`datus/tools/skill_tools/`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N9-01 | Skill 发现与扫描 | `registry.scan_directories()` → 扫描 `~/.datus/skills` 等目录，解析 SKILL.md frontmatter |
| N9-02 | Skill 元数据获取 | `registry.get_skill(name)` → 验证 name / description / tags / version / allowed_commands |
| N9-03 | Skill 内容加载 | `registry.load_skill_content(name)` → 验证完整 markdown 内容加载 |
| N9-04 | Skill 权限控制 | `manager.load_skill()` + permission check → 验证 DENY / ASK / ALLOW 三种权限路径 |
| N9-05 | Skill Bash 执行 | `SkillBashTool` → 验证 allowed_commands 限制（`python:scripts/*.py` 允许，其他拒绝） |
| N9-06 | Skill XML 生成 | `generate_available_skills_xml()` → 验证输出格式正确，可注入系统 prompt |
| N9-07 | 重复 skill 检测 | 多目录存在同名 skill → 验证 warn_duplicates 警告 |

#### N10. Datus as MCP Server（客户端视角）

入口：`datus-mcp`（`datus/mcp_server.py`），从 MCP 客户端视角验证完整的协议交互。

> 与 N6 的区别：N6 侧重服务端内部的工具注册、路由配置、transport 模式；N10 侧重模拟真实 MCP 客户端（如 Claude Desktop）连接 datus-mcp 后的端到端交互。

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N10-01 | 协议握手与初始化 | MCP 客户端连接 → initialize 请求 → 验证 server info / capabilities / protocol version 返回 |
| N10-02 | 工具列表发现 | `tools/list` 请求 → 验证返回的工具名称、描述、JSON Schema 参数定义完整且符合 MCP 规范 |
| N10-03 | 单工具调用：read_query | `tools/call(read_query, {sql: "SELECT ..."})` → 验证 SQL 执行结果格式正确返回 |
| N10-04 | 单工具调用：describe_table | `tools/call(describe_table, {table: "..."})` → 验证列定义、类型、注释返回 |
| N10-05 | 单工具调用：search_metrics | `tools/call(search_metrics, {query: "..."})` → 验证 metrics 搜索结果返回 |
| N10-06 | 工具链式调用 | 客户端顺序调用 `list_tables` → `describe_table` → `read_query`，模拟 LLM agent 的典型交互模式 |
| N10-07 | 错误处理 | 调用不存在的工具 / 缺失必填参数 / SQL 执行失败 → 验证 MCP error response 格式和错误码 |
| N10-08 | 大结果集处理 | `read_query` 返回大量行 → 验证截断策略和结果大小限制 |
| N10-09 | 并发工具调用 | 多个 `tools/call` 并发发送 → 验证服务端正确处理不互相干扰 |
| N10-10 | 会话生命周期 | 连接 → 多次工具调用 → 断开 → 重连 → 验证状态隔离 |

#### N11. Func Tools 工具函数测试

`datus/tools/func_tool/` 下共有 10 个工具类，是 agent 与数据库/知识库/文件系统交互的核心接口层。现有测试分散在 `tests/unit_tests/test_db_func_tools.py`（~50 个 unit test）、`tests/unit_tests/test_context_search_tools.py`（~14 个 unit test）、`tests/test_func_tools_db.py`（~30 个集成测试），但均未标记 acceptance。

> 现有 unit test 覆盖了 mock 级别的接口正确性，此处补充的是真实数据库 + 真实知识库的端到端 acceptance 测试。

**DBFuncTool** — 数据库操作（`func_tool/database.py`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N11-01 | list_databases | 真实 DuckDB/SQLite 上列出数据库，验证返回格式和过滤 |
| N11-02 | list_schemas | 列出 schema，验证包含/排除系统 schema |
| N11-03 | list_tables（含 views） | 列出表和视图，验证 view 标记正确 |
| N11-04 | describe_table | 获取列定义 + 类型 + semantic 注释，验证完整性 |
| N11-05 | get_table_ddl | 获取 CREATE TABLE 语句，验证 DDL 可回放 |
| N11-06 | read_query | 执行 SQL 返回结果，验证行数 / 列名 / 数据类型 |
| N11-07 | read_query 失败 | 无效 SQL → 验证错误信息格式（不泄露内部异常） |
| N11-08 | search_table | 模糊搜索表名，验证 RAG 检索结果的相关性 |
| N11-09 | Scoped tables 过滤 | sub-agent 场景下 scoped_tables 限制生效，越界访问被拦截 |
| N11-10 | Multi-connector 切换 | 多 namespace 配置下 connector LRU 缓存和切换正确 |

**ContextSearchTools** — 知识库搜索（`func_tool/context_search.py`）

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N11-11 | list_subject_tree | 列出层级目录，验证树结构完整 |
| N11-12 | search_metrics | 关键词搜索 metrics，验证返回结果的相关性排序 |
| N11-13 | get_metrics | 按 subject_path + name 获取具体 metric 定义 |
| N11-14 | search_reference_sql | 搜索历史 SQL，验证 few-shot 候选的质量 |
| N11-15 | get_reference_sql | 获取完整 SQL 示例 |
| N11-16 | search_semantic_objects | 搜索语义模型实体 |
| N11-17 | search_knowledge / get_knowledge | 外部知识检索 + 内容获取 |

**其他 Func Tools**

| 编号 | 测试场景 | 工具类 | 说明 |
|------|---------|--------|------|
| N11-18 | 日期解析 | `DateParsingTools` | "去年Q3"、"最近7天"、"2024-01" 等表达式 → 验证解析为正确的日期范围 |
| N11-19 | get_current_date | `DateParsingTools` | 验证返回当前日期格式 |
| N11-20 | 文件读取 | `FilesystemFuncTool` | read_file / read_multiple_files → 验证内容正确和错误路径处理 |
| N11-21 | 目录列出与搜索 | `FilesystemFuncTool` | list_directory / search_files → 验证结果过滤和 exclude_patterns |
| N11-22 | 文档导航与搜索 | `PlatformDocSearchTool` | list_document_nav / search_document / get_document → 平台文档检索 |
| N11-23 | 语义模型工具 | `GenSemanticModelTools` | get_multiple_tables_ddl → 批量获取表 DDL 用于语义模型生成 |
| N11-24 | 计划工具 | `PlanTool` | todo list CRUD → get_todo_list / get_item / get_completed_items |
| N11-25 | 语义工具 | `SemanticTools` | search_metrics / list_metrics / get_dimensions → MetricFlow 层的语义搜索 |
| N11-26 | 生成工具 | `GenerationTools` | SQL 生成辅助工具链 |

#### N12. CLI 搜索与内省命令

入口：`DatusCLI` REPL（`datus/cli/repl.py`），通过 `AgentCommands`（`datus/cli/agent_commands.py`）执行。

> 这些命令在 CI 重构中被移出（依赖预构建 LanceDB 索引），需在 Nightly 层端到端验证。现有 `test_cli_rich.py` 中 `!sl`/`!sq`/`!sm` 的 3 个测试可迁移合并到此组。

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| N12-01 | `!sl` schema linking | 输入自然语言问题 → 搜索相关表/列 → 验证返回表结构和 top_n 匹配 |
| N12-02 | `!sq` search reference SQL | 输入问题 → 搜索历史 SQL → 验证返回 SQL 和相关性排序 |
| N12-03 | `!sm` search metrics | 输入关键词 → 搜索 metrics → 验证返回 metric 定义和 subject_path |
| N12-04 | `!sd` search document | 输入 platform + keywords → 搜索平台文档 → 验证返回文档片段 |
| N12-05 | `!sl` 无结果处理 | 不匹配的查询 → 验证友好提示而非异常 |
| N12-06 | `!sq` subject_path 过滤 | 带 subject_path 参数搜索 → 验证按目录范围过滤 |
| N12-07 | `!sd` 无效 platform | 不存在的 platform → 验证错误提示 |

#### N-Deprecated. 建议移除的现有测试

以下现有测试在新分层体系下已被更完整的场景覆盖，在新 Nightly 测试（N1~N12）实现后可移除。过渡期间先按第四部分建议迁移标记至 nightly：

| 来源 | 现有测试 | 移除原因 | 被覆盖于 |
|------|---------|---------|---------|
| `test_deepseek_model.py` (4 tests) | `test_generate_acceptance` 等 4 个 | 单独测试 LLM 裸调用，不验证业务场景 | N5-01/05、N10-03/06 |
| `test_node.py` (4 tests) | `test_generation_node`、`test_reflection_node`、`test_execution_node`、`test_compare_with_mcp_node` | 单独测试孤立 node，新体系通过 workflow 端到端测试覆盖了完整节点链路 | N1~N5 workflow 端到端、N11 func tools |
| `test_tools_output.py` (1 test) | `test_output` | Output Tool 已在各 workflow 端到端测试的输出环节中隐式覆盖 | N3-03、N5-01、N8-03 |
| `test_integration_benchmark.py` | benchmark 执行 | Tutorial 流程（N3）已覆盖完整 benchmark 场景，且 benchmark 本身更适合作为独立的评估流程而非 acceptance 测试 | N3-01/03 |
| `test_bi_dashboard.py` (1 test) | `test_workflow_without_llm` | Superset API 交互 + DashboardAssembler 组装 + workflow 编排均已被 N2 端到端覆盖，且 N2 使用真实 LLM 而非 mock | N2-01/02/03 |

### 第三层：Regression 测试（发版前，全量回归）

**目标**：验证 Nightly 层无法覆盖的**外部服务兼容性**和**跨平台矩阵**。

**设计原则**

Regression 测试与 Nightly 的核心区别在于：它依赖**外部基础设施**。一个测试放在 Regression 而非 Nightly，必须满足以下条件之一：

1. **需要外部数据库实例**：PostgreSQL / MySQL / Snowflake / StarRocks 等远程数据库（注：外部 DB connector 测试在 datus-adapters 包的专门测试中覆盖，不在本回归测试套件中实现）
2. **需要多 LLM 厂商 API**：OpenAI / Claude / Gemini / Qwen / Kimi（Nightly 仅需一个 LLM）
3. **需要完整部署环境**：Web UI、多服务联动等

不满足以上条件的测试（如仅使用本地 DB + 单 LLM），应放在 Nightly 层。

**触发时机**：发版前手动触发 / Release 分支 push

**运行方式**：`pytest -m "ci or nightly or regression" tests/`

> 清理说明：以下项目原属 Regression，因可在本地环境完成而移入 Nightly：
> - 多 namespace 切换 → N8-07
> - Skill 执行流程 → N9
> - 工具链式调用 → N5-03、N10-06
> - parallel / subworkflow / selection node → 可用本地 DB，纳入 N5 workflow 测试扩展
> - Semantic model / ext_knowledge / feedback / subject_tree → N1、N11
> - init 向导 → N4
> - bootstrap-kb → N1

#### R1. 多 LLM 模型兼容性

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| R-01 | OpenAI 模型 | generate + tool call + streaming |
| R-02 | Claude 模型 | generate + MCP tool call |
| R-03 | Gemini 模型 | generate + tool call |
| R-04 | Qwen 模型 | generate + tool call |
| R-05 | Kimi 模型 | generate + tool call |
| R-06 | 运行时模型切换 | 运行中切换 `agent.target`，验证模型热切换不丢上下文 |

#### R2. 外部数据库 Connector

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| R-07 | PostgreSQL connector | DDL / schema / 查询执行 / schema 列表 |
| R-08 | MySQL connector | DDL / schema / 查询执行 |
| R-09 | StarRocks connector | DDL / catalog / 查询执行 |

#### R4. Web UI

| 编号 | 测试场景 | 说明 |
|------|---------|------|
| R-10 | Streamlit 基础流程 | 页面加载 → 查询提交 → 结果展示 → 会话管理 |

---

## 第四部分：现有测试迁移建议

现有 30 个 acceptance 测试应根据其依赖情况重新标记：

| 现有测试文件 | 当前 marker | 建议归入 | 理由 |
|------------|------------|---------|------|
| `test_cli_rich.py` — `!bash`、`.databases`、`.tables` (4 tests) | acceptance | **ci** | mock subprocess / 本地 connector 查询，无外部依赖 |
| `test_cli_rich.py` — `!sl`、`!sq`、`!sm`、`/chat`、`.chat_info` (5 tests) | acceptance | **nightly** | 需要预构建 LanceDB 索引 / chat 存储初始化 |
| `test_connector_duckdb.py` (5 tests) | acceptance | **ci** | 仅用本地 DuckDB |
| `test_configuration_load.py` (10 tests) | acceptance | **ci** | 纯配置解析，无外部依赖 |
| `test_tutorial.py` (5 tests) | acceptance | **ci** | 使用 Dummy stub，完全 mock |
| `test_deepseek_model.py` (4 tests) | acceptance | **nightly** | 需要真实 DeepSeek API |
| `test_node.py` (4 tests) | acceptance | **nightly** | 需要真实 LLM + Snowflake |
| `test_bi_dashboard.py` (1 test) | acceptance | **nightly** | 需要真实 Superset API |
| `test_tools_output.py` (1 test) | acceptance | **nightly** | 需要真实 LLM |

迁移后：CI 层可获得 ~21 个快速测试 + ~278 个 unit tests，Nightly 层保留 ~15 个需外部服务的测试。

---

## 第五部分：pytest 配置建议

在 `pytest.ini` 中注册新 marker：

```ini
[pytest]
markers =
    ci: CI tests - run on every PR, no external dependencies, < 3 min
    nightly: Nightly tests - real LLM + local DB, run daily
    regression: Regression tests - full matrix, run before release
    acceptance: (legacy) acceptance tests, to be migrated to ci/nightly/regression
```

CI workflow 配置示例：

```yaml
# PR 触发
- run: pytest -m ci tests/ --timeout=180

# 每日定时
- run: pytest -m "ci or nightly" tests/ --timeout=600

# 发版前
- run: pytest -m "ci or nightly or regression" tests/ --timeout=1800
```

---

## 第六部分：按优先级的实施路线

### 阶段一：建立 CI 层（最高优先级）

重点是将现有可快速运行的测试标记为 `ci`，并补充 workflow 端到端测试（mock LLM）：

1. 为 `tests/unit_tests/` ~278 个测试 + 现有 ~21 个可快速运行的 acceptance 测试添加 `@pytest.mark.ci`
2. 新增 CI-01 ~ CI-02：Workflow 定义校验 + Context 传递逻辑
3. 新增 CI-08 ~ CI-09：FastAPI 端点契约 + MCP Server 工具注册
4. 注册 pytest markers，更新 CI workflow

### 阶段二：充实 Nightly 层

1. 新增 N1（bootstrap-kb）、N3（tutorial）、N4（init）：基础设施初始化验证
2. 新增 N5（chat_agentic workflow）：核心交互路径端到端
3. 新增 N6 ~ N10（MCP Server、Sub-agent、API、Skill、MCP Client）：平台功能覆盖
4. 新增 N11 ~ N12（Func Tools、CLI 搜索命令）：工具层端到端

### 阶段三：建立 Regression 层

1. 新增 R-01 ~ R-06：多 LLM 模型兼容性 + 运行时模型切换
2. 新增 R-07 ~ R-09：外部数据库 connector 测试（PostgreSQL / MySQL / StarRocks）
3. 新增 R-10 ~ R-12：外部环境稳健性（连接恢复、并发、MCP × 外部 DB）
4. 新增 R-13：Streamlit Web UI 基础流程
