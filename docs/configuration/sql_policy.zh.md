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
            enforcement:
              on_read: filter
              on_unhandled: deny
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
