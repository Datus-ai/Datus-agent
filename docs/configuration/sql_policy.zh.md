# SQL Policy

可信上游完成用户认证和权限计算，Agent 只执行生效策略，不管理 role。
策略通过可选的 `sql-policy` plugin 加载；没有配置策略时，Agent / Dosi 保持原有独立运行方式，不依赖私有包。

## 配置

### 用 DEBUG 日志查看 SQL

以 `datus --debug` 启动，或配置 `agent.logging.level: DEBUG`。
Agent 常规日志中会出现以下结构化事件：

| 事件 | SQL 字段 | 记录时机 |
|---|---|---|
| `sql.policy.rewrite` | `original_sql`、`rewritten_sql`、`rewritten` | 策略处理完成、执行之前；覆盖 SQL 工具、API 读取和 CLI 手工 SQL。 |
| `sql.execute` | `sql` | SQL 工具或 CLI 向 connector 提交最终 SQL 之前。 |
| `sql.dosi.execute` | `sql` | 原生指标/明细执行成功后，记录该次引擎调用实际返回的 SQL。 |
| `sql.dosi.compile` | `sql` | 指标 dry-run 编译完成；该 SQL 尚未执行。 |

事件包含 datasource、dialect；Dosi 另包含 connection 和 operation。
打印日志不会额外编译查询，也不会序列化 policy_context；SQL 字段沿用现有日志脱敏配置。
INFO 及以上级别不输出这些事件。

### 策略规则

原生 `datus-sql-policies` 使用同一份物理表 `row_filter` 保护 SQL 和指标读取。
旧 `metric_row_filter` 配置需要迁移，不再往指标的业务 `where` 中拼接条件。

```yaml
agent:
  plugins:
    sql-policy:
      default:
        default: true
        policies:
          - id: tenant_scope
            name: tenant_scope
            type: row_filter
            applies_to:
              datasources: [warehouse]
              tables: [analytics.main.orders]
            condition:
              column: tenant_id
              operator: in
              value_from: policy_context.row_filter.tenant_ids
        column_masking:
          - column: analytics.main.orders.phone
            mask: last4
```

Profile 目标支持短名、部分限定名和完整名称：`orders`、`orders.phone` 等省略的前缀，在绑定时按实际连接
的当前坐标补齐，不使用 datasource 初始默认值；显式坐标保持不变。切换命名空间后，旧绑定失效，
重新绑定的短目标跟随新命名空间，完整目标仍指向原来指定的表。必要的当前坐标不可用时拒绝绑定。

完整目标按 adapter 层级书写：DuckDB、PostgreSQL、Greenplum、GaussDB 用 `database.schema.table`；
StarRocks、Doris 用 `catalog.database.table`；Trino 用 `catalog.schema.table`；Oracle、HANA 用 `schema.table`；
MySQL、TiDB、ClickHouse、Hive、Spark 用 `database.table`。Oracle 服务名和 HANA 租户属于连接身份，
不作为 SQL 对象名前缀；datasource 路由名单独指定。名称含点时需要 SQL 引号。
策略默认值只指向当前命名空间，查询短名仍按 SQL 的 search path 解析。
MySQL/TiDB 会读取实际连接的 `lower_case_table_names`，物理目标和 CTE 按该设置匹配大小写。

行过滤支持 `eq`、`in`。四种固定 masking 为 `nullify`、`redact`、`last4`、`email`，只配置目标列和方式。
文本方式需要兼容的文本类型和已支持的方言。目前已完成 DuckDB，以及 13 个 Docker adapter 的真实执行验收：PostgreSQL、MySQL、TiDB、StarRocks、
Doris、ClickHouse、Greenplum、GaussDB/openGauss、Trino、Oracle、Hive、Spark（HiveServer2）、HANA，
覆盖已测试的原生类型和查询形态。Oracle 指标路径需要 OCI，HANA 指标路径需要 SAP HANA ODBC；
本地 HANA 验收使用直接明文 SQL 端口，租户路由及 TLS 另行验收。
SQLite 也已验证：原生 SQL 使用 SQLite 方言，Dosi 通过 DuckDB SQLite 扩展执行。
目标使用 `main.table` 或 `attach_alias.table`，不使用文件名。Dosi 仅支持文件库，
存在临时对象或额外附加库时拒绝执行；元数据要求 SQLite 3.37+，`last4` 遇到内嵌 NUL 返回 NULL。
SQLite 文本列的行规则要求字符串值，避免两种执行器的隐式类型转换差异。
其他文本方言（包括 Inceptor）尚未开放。

## 可信请求上下文

项目可以使用命名脱敏方案替代静态 `column_masking` 列表。以下片段放在现有
`agent.plugins.sql-policy.<profile>` 内：

```yaml
policy_revision: "r17"
masking_profiles:
  finance_team:
    - column: analytics.main.orders.phone
      mask: last4
  external_partner:
    - column: analytics.main.orders.phone
      mask: redact
  approved_raw: []
```

ID 是自定义且区分大小写的配置 key；可信宿主每请求选择一个方案：

```json
{"policy_revision":"r17","masking_profile":"finance_team","row_filter":{"access_mode":"scoped","tenant_ids":[42]}}
```

请求只传 ID 和版本，列规则保存在项目共享配置中。明确选中空列表方案时不执行列脱敏，
行过滤仍然执行，方案名称没有特殊授权含义。ID 缺失、未知或版本不匹配时拒绝读取。
命名方案与静态 masking 规则互斥；请求中的方案选择与完整 `sql_policy` 快照也互斥。
只有 masking 的项目同样需要方案 ID 和版本。选择过程不修改共享配置，
方案及版本会进入执行和缓存指纹，并显示在 `sql.policy.rewrite` DEBUG 事件中。

可信宿主可以通过 `PolicyRuntime(config).requires_request_context()` 判断是否需要用户上下文，
包括只有命名脱敏方案的项目。SaaS/Backend 的用户选择和配置刷新尚未接入，当前需由可信宿主明确传入。

配置中的行规则继续使用现有执行输入：

```json
{"row_filter":{"access_mode":"scoped","tenant_ids":[42]}}
```

`denied` 拒绝所有读取。可选 `policy_modes` 可将某个已配置规则 ID 标为 `unrestricted`，只跳过该行规则；其他规则及 masking 仍然生效。缺少输入、未知 ID、非法模式都会拒绝。

可信宿主也可直接提供 `policy_context.sql_policy`，包含 `version` 和
`policies: {row_filters: [...], column_masks: [...]}`。目标为规范化的结构化对象：
`datasource`、可选 `catalog`、`database`、`schema`、`table`。
行规则增加 `column`、`operator`、`values`，masking 增加 `column`、`mask`。
该快照替代本次请求的 profile 规则，属于可信授权输入，不能由模型的工具参数提供。
默认 API provider 可从 `X-Datus-Policy-Context` 解析 JSON，部署方必须保证 header 来源可信。
`X-Datus-User-Id` 仍只用于会话隔离。

## 生效流程

1. Agent 校验 context，将本次策略快照绑定到实际连接和元数据。
2. 普通 SQL 经 `DBFuncTool.execute_read_enforced`；Dosi 经可选 Python `policy` / `policy_required` 参数注入同一 provider。
3. 完整 SQL lowering 后，共享原生核心逐个替换物理表读取：先按原值过滤行，再投影脱敏值，外层业务计算使用这些值。子查询、重复引用、query-backed dataset、衍生指标、FIXED/context_filter 和归因查询都经过这个边界。
4. 执行前再次校验连接上下文；指标/明细结果缓存绑定策略快照和执行上下文。返回结果后不再做二次 masking。

要求策略但缺失实现、绑定失败或语法不支持时拒绝执行。暂不支持的视图、表函数等不透明来源会被拒绝；不会退回旧的指标 WHERE 注入。进程外 proxied executor 需自行保护；新增原生接口不会自动保护所有已有 workflow 或外部执行入口。

`datus sql-policy status` 查看配置。离线 `check` 可以校验 context，但缺少实际连接时拒绝策略改写；查看改写后的 SQL 应使用 Agent / Dosi dry-run。
本实现需要配套原生 wheel，SaaS/Backend 的接入和发布另行处理。

报表/看板指标先编译，在最终 SQL 读取入口统一应用一次策略；直接 Dosi 执行仍使用原生 provider。
配置了策略时，包含读取的写语句会被拒绝；仅配置静态 `column_masking`、请求上下文为空时也会生效。
命名脱敏方案需要请求明确提供方案 ID 和版本。

## 只读硬开关（`agent.sql_read_only`）


`agent.sql_read_only` 是另一套更简单的机制，不要与上面的 policy plugin 混淆。

```yaml
agent:
  sql_read_only: true   # 默认 false
```

置为 `true` 后，该配置服务的所有 SQL 入口都不允许执行非只读语句：

- `DBFuncTool.execute_sql`（暴露给 agentic node 与 MCP server 的工具）会硬拒绝 SELECT / SHOW / DESCRIBE / EXPLAIN 之外的任何语句。这与权限档位无关，在 `PermissionHooks` 被完全绕过的路径上同样生效（LLM validator 以 `hooks=None` 运行，MCP server 的工具实例根本不经过 hooks）。
- 该工具的其它写入口同样拒绝，而不只是 `execute_sql` 这个分发入口：`execute_write`、`execute_ddl`、`transfer_query_result`。最后一个尤其关键——它从一个数据源读、向**另一个**数据源写（`CREATE TABLE` / `TRUNCATE` / `INSERT`），由 `gen_job` 单独挂载为工具，且完全不经过 `execute_sql`。
- workflow 流水线同样拒绝。它的 `execute_sql` 节点和 output 工具的「改写后 SQL」检查都把 SQL 直接交给 connector，不经过 `DBFuncTool`，因此上面那些闸门都覆盖不到；而 `POST /workflows/run` 让这条流水线经 API 可达。两处都通过共用的 `deployment_read_only_refusal` 查询该开关。
- `EXPLAIN` 只有在它所解释的语句本身是只读时才算只读。`EXPLAIN ANALYZE <写语句>` 在 PostgreSQL 和 MySQL 上**会真正执行**该写操作，因此内层语句会被单独归类，不是只读就拒绝 —— 无论是否带 `ANALYZE`，因为按选项关键字判断意味着要一直追平各 dialect 的所有拼法，漏掉一种就 fail open。
- 顶层是读的语句，内部仍可能写：PostgreSQL 的数据修改型 CTE（`WITH d AS (DELETE ... RETURNING *) SELECT * FROM d`）从外面看就是一条 SELECT。因此会遍历整条语句查找写节点，而不只看根节点。
- `POST /sql/execute` 拒绝任何不是**单条**只读语句的输入。多语句（`SELECT 1; DROP TABLE t`）、可写 `PRAGMA`、`USE` / `SET`，以及解析器无法归类的语句一律拒绝——判定是 fail-closed 的。需要切换数据库时请使用请求体的 `database_name` 字段，而不是 `USE`。

`DBFuncTool.read_only` 返回的是**生效后**的姿态，因此构造时没有传 `read_only` 的实例——MCP server 的 `create_dynamic` / `create_static` 工厂正是这样构造的——在加固过的部署上读出来仍是 `True`。

与 policy plugin 的区别：

| | `agent.sql_read_only` | policy plugin |
|---|---|---|
| 需要 plugin | 否 | 是 |
| 需要请求上下文 | 否 | 是——每请求的 `policy_context` |
| 行为 | 直接拒绝该语句 | 按请求上下文重写或拒绝 |
| 粒度 | 全局、非黑即白 | 行 / 表 / 列，按调用方 |
| 覆盖 `POST /sql/execute` | 是 | 否（仅只读工具） |

该开关只能收紧：它以只读属性加一个单向的 `AgentConfig.harden_sql_read_only()` 暴露——每请求的配置副本可以自行加固，但下游任何代码都无法把 `true` 改回去；同时它也绝不会放松本就以只读运行的组件（Explore、`ask_report`、LLM validator）。

适用场景：进程需要针对自有数据源运行第三方作者的 agent 内容（skill、subagent、reference template）。两套机制可以叠加使用——`sql_read_only` 限定「允许执行哪一类语句」，policy plugin 限定「某个调用方能读到什么」。

### 如何验证一个部署

自动化覆盖位于 `tests/unit_tests/tools/func_tool/test_database.py`、`tests/integration/tools/test_func_tools_db.py` 与 `tests/integration/tools/test_mcp_server.py`。

另有两个手工脚本，通过 MCP 端到端探测真实服务。两者都不在 CI 中运行，都会打印判定表格并在失败时以非零码退出：

| 脚本 | 检查内容 |
|---|---|
| `scripts/e2e_sql_read_only_mcp.py` | 自包含。搭建一次性 SQLite 工作区，跑「开关打开 / 关闭」对照矩阵，报告开关拒绝了哪些语句。无需参数。 |
| `scripts/e2e_sql_read_only_mcp_project.py` | 指向真实的打包项目（`--project`）。`--sqlite-standin` 用一次性 SQLite 替换数据源，使开关关闭时写入真正执行；`--endpoint` 探测已在运行的服务；`--live-writes` 会写入**真实**数据源（仅限本次运行自建的临时表），`--dry-run` 只打印语句不执行。 |
