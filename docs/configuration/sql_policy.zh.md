# SQL Policy

Datus Agent 不负责用户认证，也不判断用户具有什么权限。可信上游服务完成认证和鉴权后，只把策略执行所需的输入作为 `policy_context` 传给 Agent；启用的 policy plugin 负责解释这些输入并保护数据读取。

## 配置

SQL policy 只通过 plugin profile 配置；原来的 `agent.sql_policy` provider 配置不再支持。

```yaml
agent:
  plugins:
    sql-policy:
      default:
        default: true
        policies:
          - name: store_scope_sql
            type: row_filter
            applies_to:
              datasources: ["warehouse"]
              tables: ["orders", "store_sales"]
            condition:
              column: store_id
              operator: in
              value_from: policy_context.row_filter.store_ids
```

policy type 及其字段由 policy plugin 定义。Agent 只加载 plugin 在 `datus-plugin.yml` 中声明的运行时。

## 请求上下文

默认 API provider 从 `X-Datus-Policy-Context` 读取 JSON object：

```http
X-Datus-Policy-Context: {"row_filter":{"access_mode":"scoped","store_ids":[1,2]}}
```

约定结构按 policy family 分两层，不包含用户身份或 groups：

```json
{
  "row_filter": {
    "access_mode": "scoped",
    "store_ids": [1, 2]
  },
  "column_mask": {
    "customers.email": {"strategy": "email_partial"}
  }
}
```

当前 sql-policy plugin 支持三种 row-filter 模式：

| `access_mode` | 行为 |
|---|---|
| `denied` | 拒绝所有数据读取。 |
| `scoped` | 执行已配置的行过滤，并解析其 `policy_context.*` 输入；缺少输入时 fail closed。 |
| `unrestricted` | 仅跳过行过滤；未来的 column masking 等其他 policy family 仍会执行。 |

配置了 row policy 时，缺少 `access_mode` 或传入未知值都会被拒绝；没有配置 row policy 时，空 context 可以通过。

`X-Datus-User-Id` 只用于会话隔离。Agent 不会把它合并进 `policy_context`，也不会把 context 中的任何字段当作已认证身份。

## 运行流程

1. 上游服务完成认证、鉴权并生成 `policy_context`。
2. API 将 header 解析为 `AppContext.policy_context`，并在启动任何内置或用户自定义 subagent 前调用启用的 policy runtime 校验。
3. 请求级 `AgentConfig` 副本把同一 context 传给所有 subagent 和 tool transformer。
4. `DBFuncTool.execute_read_enforced` 先校验原始 SQL，再调用 `before_sql_read`，然后重新校验改写后的 SQL 并执行。
5. 查询成功后，原始结果会先经过 `after_read_result`，之后才允许压缩、保存 artifact、渲染或返回；column masking 将在这个扩展点实现。
6. 语义指标工具也调用同一个 plugin runtime，在聚合前写入 `where` 条件。

无效 runtime 声明、格式错误的 decision、策略异常、策略拒绝和不安全 SQL 改写都会 fail closed。proxied tool 在 Agent 进程外执行，需要由外部 executor 自行保护。

手工检查时显式传入同一个对象：

```bash
datus sql-policy check --sql "SELECT * FROM orders" \
  --policy-context '{"row_filter":{"access_mode":"scoped","store_ids":[1,2]}}'
```

## 只读硬开关（`agent.sql_read_only`）

`agent.sql_read_only` 是另一套更简单的机制，不要与上面的 policy plugin 混淆。

```yaml
agent:
  sql_read_only: true   # 默认 false
```

置为 `true` 后，该配置服务的所有 SQL 入口都不允许执行非只读语句：

- `DBFuncTool.execute_sql`（暴露给 agentic node 与 MCP server 的工具）会硬拒绝 SELECT / SHOW / DESCRIBE / EXPLAIN 之外的任何语句。这与权限档位无关，在 `PermissionHooks` 被完全绕过的路径上同样生效（LLM validator 以 `hooks=None` 运行，MCP server 的工具实例根本不经过 hooks）。
- `POST /sql/execute` 拒绝任何不是**单条**只读语句的输入。多语句（`SELECT 1; DROP TABLE t`）、可写 `PRAGMA`、`USE` / `SET`，以及解析器无法归类的语句一律拒绝——判定是 fail-closed 的。需要切换数据库时请使用请求体的 `database_name` 字段，而不是 `USE`。

与 policy plugin 的区别：

| | `agent.sql_read_only` | policy plugin |
|---|---|---|
| 需要 plugin | 否 | 是 |
| 需要请求上下文 | 否 | 是——每请求的 `policy_context` |
| 行为 | 直接拒绝该语句 | 按请求上下文重写或拒绝 |
| 粒度 | 全局、非黑即白 | 行 / 表 / 列，按调用方 |
| 覆盖 `POST /sql/execute` | 是 | 否（仅只读工具） |

该开关只能收紧：每请求的配置副本可以自行加固，但下游任何代码都无法把 `true` 改回去；同时它也绝不会放松本就以只读运行的组件（Explore、`ask_report`、LLM validator）。

适用场景：进程需要针对自有数据源运行第三方作者的 agent 内容（skill、subagent、reference template）。两套机制可以叠加使用——`sql_read_only` 限定「允许执行哪一类语句」，policy plugin 限定「某个调用方能读到什么」。
