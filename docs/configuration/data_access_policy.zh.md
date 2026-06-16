# 数据访问策略

数据访问策略会在 SQL 到达数据库前限制读查询。适用于 API 需要按调用方的业务范围回答问题的场景，例如市场、租户、区域或门店列表。

开源 Agent 提供 data-access 扩展点、配置加载、请求 principal 处理和 API fail-fast 检查。开源版不包含私有策略实现。要真正执行策略，需要实现并安装自己的 provider 包。

## 开源 Agent 提供什么

当 `agent.data_access.enabled` 为 `true` 时，Datus 会：

- 加载 `agent.data_access.provider` 配置的 Python class。
- 将完整的 `agent.data_access` mapping 作为 `DataAccessConfig.raw` 传给 provider。
- 在读查询执行前调用 provider。
- 将 API 请求中的 request-scoped principal 字段传给 provider。
- 当策略配置引用了缺失的 `principal.*` 字段时，在 Agent 启动前拒绝 chat 请求。

provider 自己决定策略 schema 和 enforcement 行为。

## 实现 Provider

创建一个能被 `datus-api` 所在 Python 环境 import 到的包。provider class 必须实现 `enforce_read(...)`。

```python
from typing import Any, Optional

from datus.tools.data_access_policy import DataAccessConfig, EnforcementResult


class MyDataAccessProvider:
    def __init__(self, config: Optional[DataAccessConfig] = None):
        self.config = config or DataAccessConfig()
        self.policies = self.config.raw.get("policies", []) or []

    def enforce_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        principal: Optional[dict[str, Any]],
    ) -> EnforcementResult:
        principal = principal or {}

        # 在这里实现你的策略逻辑：
        # - 检查 datasource、sql 和 self.policies
        # - 从 principal 中解析策略值
        # - 用 SQL parser 改写 SQL，或拒绝查询
        # - 返回最终应该执行的 SQL
        return EnforcementResult(allowed=True, sql=sql)
```

返回值示例：

```python
EnforcementResult(allowed=True, sql=rewritten_sql, applied_policies=["market_scope"])
EnforcementResult(allowed=False, reason="Missing principal.market_code")
```

改写 SQL 时建议使用 SQL parser 或数据库安全的 query builder，不要用简单字符串拼接来生成策略 SQL。

## 配置 Provider

将 `agent.data_access.provider` 指向你自己的 provider class：

```yaml
agent:
  data_access:
    enabled: true
    provider: my_company.datus_policies:MyDataAccessProvider
    policies:
      - name: market_scope
        type: row_filter
        applies_to:
          datasources: ["starrocks"]
          tables: ["v_udata_ac_info"]
        condition:
          column: market_code
          operator: eq
          value_from: principal.market_code
        enforcement:
          on_read: filter
          on_unhandled: deny
```

开源 Agent 只解释 `enabled` 和 `provider`。其他 policy 结构会原样放进 `DataAccessConfig.raw`，由你的 provider 自己解释。

如果策略配置中包含 `value_from: principal.market_code`，API pre-check 会要求请求 principal 中必须包含 `market_code`，然后才允许 Agent 启动。这个递归 `value_from` 扫描是通用逻辑，不依赖特定 policy type。

## 发送请求 Principal

默认 API auth provider 通过 `X-Datus-Principal` header 接收 data-access 属性。header 值必须是 JSON object：

```bash
curl -N "http://127.0.0.1:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -H 'X-Datus-Principal: {"market_code":"MKT300"}' \
  -d '{
    "message": "查询6月新产品活动",
    "subagent_id": "baisheng_metadata"
  }'
```

`X-Datus-Principal` 会变成 `AppContext.principal`：

```python
{"market_code": "MKT300"}
```

你的 provider 会在 `enforce_read(..., principal=...)` 中收到这个 dict。

`X-Datus-User-Id` 是另一件事。它是可选的调用者身份字段，只用于会话隔离；它不会写入 data-access principal，也不能满足 `principal.market_code`。

## 多值范围

如果你的 provider 支持多值范围，可以使用 principal array：

```yaml
condition:
  column: market_code
  operator: in
  value_from: principal.market_codes
```

请求里发送：

```bash
curl -N "http://127.0.0.1:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -H 'X-Datus-Principal: {"market_codes":["MKT300","MKT301"]}' \
  -d '{"message":"查询6月新产品活动"}'
```

开源 Agent 只负责传递这个数组。`operator: in` 如何执行，由你的 provider 决定。

## 嵌套 Principal 字段

request principal 可以包含嵌套 JSON object：

```http
X-Datus-Principal: {"tenant":{"id":"tenant_001"}}
```

策略配置可以引用这个路径：

```yaml
condition:
  column: tenant_id
  operator: eq
  value_from: principal.tenant.id
```

API pre-check 会把空字符串、空数组和 `null` 当作缺失值。

## Provider 契约

| 项 | 契约 |
|----|------|
| `agent.data_access.enabled` | 启用 data-access enforcement。 |
| `agent.data_access.provider` | Python class path，格式为 `module:Class`。启用时必填。 |
| `DataAccessConfig.raw` | 完整的原始 `agent.data_access` mapping，会传给 provider。 |
| `enforce_read(sql, datasource, dialect, principal)` | 在读查询执行前调用。 |
| `EnforcementResult.allowed=True` | 查询可以继续。通过 `sql` 返回原始或改写后的 SQL。 |
| `EnforcementResult.allowed=False` | 查询被拒绝。通过 `reason` 返回给 tool/model 的错误原因。 |
| `X-Datus-Principal` | API 请求中会被解析为 `AppContext.principal` 的 JSON object。 |
| `value_from: principal.*` | 可选约定，用于 API pre-check 检查缺失的 principal 字段。 |

## 错误行为

如果 data access 已启用，而策略配置需要的 principal 字段缺失，chat API 会在 Agent 启动前失败：

```text
DATA_ACCESS_PRINCIPAL_REQUIRED
```

错误信息会包含缺失字段，例如 `principal.market_code`。

如果 `X-Datus-Principal` 不是合法 JSON，API 会返回 HTTP 400，因为请求上下文本身格式错误。

如果 provider 拒绝查询，tool 会把 provider 返回的 `reason` 传给 Agent。

## 限制

- 开源版 Datus 不包含内置 row-filter 实现。
- 当前内置请求 principal 入口是 API header `X-Datus-Principal`。
- CLI 请求没有 HTTP header。需要请求级 data-access principal enforcement 时请使用 REST API，或者在部署中实现能填充 `AppContext.principal` 的 auth provider。
- 不要把 `user_id` 放进 `X-Datus-Principal`；`user_id` 是 `X-Datus-User-Id` 的保留字段。
