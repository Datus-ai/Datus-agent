# Data Access Policy

Data access policies restrict SQL reads before they reach the database. Use them when the API must answer questions only within a caller's business scope, such as a market, tenant, region, or store list.

The open-source agent provides the data-access extension point, configuration loading, request-principal handling, and API fail-fast checks. It does not ship private policy implementations. To enforce a policy, implement and install your own provider package.

## What the Open-Source Agent Provides

When `agent.data_access.enabled` is `true`, Datus:

- Loads the Python class configured in `agent.data_access.provider`.
- Passes the full `agent.data_access` mapping to the provider as `DataAccessConfig.raw`.
- Calls the provider before read queries are executed.
- Passes request-scoped principal fields from the API request to the provider.
- Fails chat requests before the agent starts when policy config references missing `principal.*` fields.

The provider decides the policy schema and enforcement behavior.

## Implement a Provider

Create a Python package importable by the same environment that runs `datus-api`. The provider class must implement `enforce_read(...)`.

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

        # Implement your policy logic here:
        # - inspect datasource, sql, and self.policies
        # - resolve values from principal
        # - rewrite SQL with a parser, or deny the query
        # - return the SQL that should be executed
        return EnforcementResult(allowed=True, sql=sql)
```

Return values:

```python
EnforcementResult(allowed=True, sql=rewritten_sql, applied_policies=["market_scope"])
EnforcementResult(allowed=False, reason="Missing principal.market_code")
```

Use a SQL parser or database-safe query builder for rewrites. Avoid ad hoc string concatenation for policy SQL.

## Configure the Provider

Point `agent.data_access.provider` at your provider class:

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

Only `enabled` and `provider` are interpreted by the open-source agent. The rest of the policy shape is passed through to your provider in `DataAccessConfig.raw`.

If your policy config contains `value_from: principal.market_code`, the API pre-check requires the request principal to contain `market_code` before the agent starts. This recursive `value_from` scan is generic and does not depend on a specific policy type.

## Send the Request Principal

For the default API auth provider, send data-access attributes in the `X-Datus-Principal` header as a JSON object:

```bash
curl -N "http://127.0.0.1:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -H 'X-Datus-Principal: {"market_code":"MKT300"}' \
  -d '{
    "message": "Show June new product campaigns",
    "subagent_id": "baisheng_metadata"
  }'
```

`X-Datus-Principal` becomes `AppContext.principal`:

```python
{"market_code": "MKT300"}
```

Your provider receives that dict in `enforce_read(..., principal=...)`.

`X-Datus-User-Id` is separate. It is optional and only identifies the caller for session isolation. It does not populate data-access principal fields and cannot satisfy `principal.market_code`.

## Multi-Value Scope

If your provider supports multi-value scopes, use a principal array:

```yaml
condition:
  column: market_code
  operator: in
  value_from: principal.market_codes
```

Send:

```bash
curl -N "http://127.0.0.1:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -H 'X-Datus-Principal: {"market_codes":["MKT300","MKT301"]}' \
  -d '{"message":"Show June new product campaigns"}'
```

The open-source agent only passes this array through. Your provider decides how `operator: in` is enforced.

## Nested Principal Fields

The request principal can contain nested JSON objects:

```http
X-Datus-Principal: {"tenant":{"id":"tenant_001"}}
```

Policy config can reference that path:

```yaml
condition:
  column: tenant_id
  operator: eq
  value_from: principal.tenant.id
```

The API pre-check treats empty strings, empty arrays, and `null` as missing.

## Provider Contract

| Item | Contract |
|------|----------|
| `agent.data_access.enabled` | Enables data-access enforcement. |
| `agent.data_access.provider` | Python class path in `module:Class` format. Required when enabled. |
| `DataAccessConfig.raw` | Full raw `agent.data_access` mapping passed to the provider. |
| `enforce_read(sql, datasource, dialect, principal)` | Called before read queries execute. |
| `EnforcementResult.allowed=True` | Query may continue. Return `sql` to execute the original or rewritten SQL. |
| `EnforcementResult.allowed=False` | Query is denied. Return `reason` for the tool/model-facing error. |
| `X-Datus-Principal` | JSON object parsed into `AppContext.principal` for API requests. |
| `value_from: principal.*` | Optional convention used by API pre-check to detect missing principal fields. |

## Error Behavior

If data access is enabled and policy config requires a missing principal field, the chat API fails before the agent starts:

```text
DATA_ACCESS_PRINCIPAL_REQUIRED
```

The error message includes the missing field, for example `principal.market_code`.

If `X-Datus-Principal` is not valid JSON, the API returns HTTP 400 because the request context is malformed.

If the provider denies a query, the tool returns the provider's `reason` to the agent.

## Limits

- Open-source Datus does not include a built-in row-filter implementation.
- Current built-in request-principal input is the API header `X-Datus-Principal`.
- CLI requests do not have HTTP headers. Use the REST API for request-scoped data-access principal enforcement, or implement an auth provider that populates `AppContext.principal` for your deployment.
- Do not put `user_id` inside `X-Datus-Principal`; `user_id` is reserved for `X-Datus-User-Id`.
