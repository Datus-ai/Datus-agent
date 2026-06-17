# SQL Policy

SQL policy 是一个扩展框架，用于在请求级别控制 SQL 读查询。适用于 API 请求只能访问调用方业务范围内数据的场景，例如租户、市场、区域或门店列表。

开源 Agent 提供框架边界和运行时 hook 点，不内置具体策略引擎。需要执行策略的部署应提供自己的 plugin 包。

## 框架边界

开源 Agent 负责：

- 从 Agent 配置中加载 `agent.sql_policy`。
- 从 `agent.sql_policy.provider` 加载配置的 plugin class。
- 将完整的原始 `agent.sql_policy` mapping 传给 plugin。
- 将请求 principal 字段解析到 `AppContext.principal`。
- 对必需的 `principal.*` 值执行通用 API pre-check。
- 在读查询到达数据库前调用 plugin。
- 在 plugin 改写 SQL 后重新校验 SQL。
- 将策略拒绝原因返回给 tool/model 层。

plugin 负责：

- 定义 `agent.sql_policy` 下的策略 schema。
- 校验 plugin 自己的策略字段。
- 将策略匹配到 datasource、表、列或其他业务概念。
- 从请求 principal 中解析策略值。
- 改写读 SQL 或拒绝查询。
- 返回清晰的拒绝原因，说明 model 或调用方缺少什么。

## 运行时流程

启用 SQL policy 后，请求处理流程如下：

1. API auth provider 创建 `AppContext`。
2. 请求级属性被写入 `AppContext.principal`。
3. Agent 配置被加载，其中包括 `agent.sql_policy`。
4. 如果原始策略配置引用了 `value_from: principal.<path>`，chat API 会在 Agent 启动前检查每个 principal path 是否存在。
5. Agent 正常运行，直到调用数据库读工具。
6. `DBFuncTool.read_query` 校验原始 SQL 是否只读。
7. 通过 `enforce_read(...)` 加载并调用配置的 plugin。
8. 如果 plugin 拒绝查询，tool 返回 plugin 的拒绝原因，不执行 SQL。
9. 如果 plugin 返回改写后的 SQL，Agent 会重新校验改写后的 SQL 仍然只读。
10. 校验通过的 SQL 被发送到目标 datasource 执行。

第 4 步的 pre-check 是通用逻辑。它会扫描原始策略 mapping 中以 `principal.` 开头的 `value_from` 字符串，不假设特定 policy type、列名或业务字段。

## 配置契约

开源 Agent 只解释这些字段：

```yaml
agent:
  sql_policy:
    enabled: true
    provider: my_company.sql_policies:SqlPolicyProvider
```

`enabled` 用于打开框架。`provider` 必须是 `module:Class` 格式的 Python class path。

`agent.sql_policy` 下的其他字段会原样放进 `SqlPolicyConfig.raw` 传给 plugin。你的 plugin 可以定义自己需要的 schema：

```yaml
agent:
  sql_policy:
    enabled: true
    provider: my_company.sql_policies:SqlPolicyProvider
    policies:
      - name: tenant_scope
        type: row_filter
        applies_to:
          datasources: ["warehouse"]
          tables: ["orders"]
        condition:
          column: tenant_id
          operator: eq
          value_from: principal.tenant.id
        enforcement:
          on_read: filter
          on_unhandled: deny
```

在这个例子里，开源 Agent 使用 `enabled`、`provider`，并使用 `principal.tenant.id` 引用做 pre-check。`policies`、`type`、`applies_to`、`condition` 和 `enforcement` 的含义都由 plugin 决定。

## Plugin 接口

plugin 是一个安装在 `datus-api` 同一 Python 环境中的普通 Python 包。provider class 必须接收 `SqlPolicyConfig` 参数，并实现 `enforce_read(...)`。

```python
from typing import Any, Dict, Optional

from datus.tools.sql_policy import SqlPolicyConfig, EnforcementResult


class SqlPolicyProvider:
    def __init__(self, config: Optional[SqlPolicyConfig] = None) -> None:
        self.config = config or SqlPolicyConfig()
        self.policies = self.config.raw.get("policies", []) or []
        self._validate_policy_config()

    def enforce_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        principal: Optional[Dict[str, Any]],
    ) -> EnforcementResult:
        principal = principal or {}

        # 在这里实现策略选择和执行逻辑：
        # - 解析 SQL 并识别引用的表
        # - 为 datasource 和表匹配策略
        # - 从 principal 中解析配置值
        # - 返回改写后的读查询，或带清晰原因地拒绝查询
        return EnforcementResult(allowed=True, sql=sql)

    def _validate_policy_config(self) -> None:
        # 如果 plugin 自己需要的配置不合法，在这里抛出异常。
        pass
```

返回值决定后续行为：

```python
return EnforcementResult(
    allowed=True,
    sql=rewritten_sql,
    applied_policies=["tenant_scope"],
)
```

```python
return EnforcementResult(
    allowed=False,
    reason="Missing required principal path: principal.tenant.id",
)
```

改写 SQL 时应使用 SQL parser 或数据库安全的 query builder，不要用字符串拼接生成策略谓词。

## 请求 Principal

principal 是策略 plugin 使用的请求级调用方属性。默认 API auth provider 会从 `X-Datus-Principal` header 读取 principal 字段，header 值必须是 JSON object：

```http
X-Datus-Principal: {"tenant":{"id":"tenant_001"},"market_codes":["MKT300","MKT301"]}
```

plugin 会收到解析后的对象：

```python
{
    "tenant": {"id": "tenant_001"},
    "market_codes": ["MKT300", "MKT301"],
}
```

策略配置可以通过 `principal.<path>` 引用嵌套字段：

```yaml
condition:
  column: tenant_id
  operator: eq
  value_from: principal.tenant.id
```

API pre-check 会把缺失 key、`null`、空字符串和空数组都视为缺失值。

`X-Datus-User-Id` 和 SQL policy principal 是两件事。它用于调用方会话隔离，不会被复制进 `AppContext.principal`。

## Provider 契约

| 项 | 契约 |
|----|------|
| `agent.sql_policy.enabled` | 启用 SQL policy 框架。 |
| `agent.sql_policy.provider` | Python class path，格式为 `module:Class`。启用时必填。 |
| `SqlPolicyConfig.raw` | 完整的原始 `agent.sql_policy` mapping，会传给 plugin。 |
| `enforce_read(sql, datasource, dialect, principal)` | 在读 SQL 执行前调用。 |
| `EnforcementResult.allowed=True` | 查询可以继续。`sql` 可以是原始 SQL 或改写后的 SQL。 |
| `EnforcementResult.allowed=False` | 查询被拒绝。`reason` 会返回给 tool/model 层。 |
| `applied_policies` | 可选的策略名称列表，用于日志和诊断。 |
| `value_from: principal.*` | 可选约定，用于 API pre-check 检查缺失的 principal 字段。 |

## 错误行为

如果 SQL policy 已启用但没有配置 provider，enforcement 会在 SQL 执行前失败。

如果配置的 provider class 无法 import、无法初始化，或没有实现可调用的 `enforce_read`，enforcement 会返回 SQL policy provider error。

如果策略配置引用了缺失的 `principal.*` 值，chat API 会在 Agent 启动前失败：

```text
SQL_POLICY_PRINCIPAL_REQUIRED
```

错误信息会包含缺失的 principal path，例如 `principal.tenant.id`。

如果 plugin 拒绝查询，SQL 不会执行，tool 会返回 plugin 的 `reason`。

如果 plugin 将 SQL 改写成非读语句或多语句查询，读查询校验器会在执行前拒绝它。

## 限制

- 开源 Agent 不包含内置 row-filter 或 SQL 注入审查 plugin。
- 除了 `enabled`、`provider` 和可选的 `principal.*` pre-check 约定，框架不定义强制策略 schema。
- CLI 请求没有 HTTP header。请求级 API principal 输入来自 API auth context。对于 CLI 或自定义部署，需要通过对应 auth 或运行时集成填充 `AppContext.principal`。
- 不要把 `user_id` 放进 `X-Datus-Principal`；`user_id` 是 `X-Datus-User-Id` 的保留字段。
