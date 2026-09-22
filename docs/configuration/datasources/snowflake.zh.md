# Snowflake datasource

Snowflake adapter 使用原生 Snowflake Python connector，支持密码和 RSA key-pair 认证，并保留 `database → schema → table` 层级。

## 安装

```bash
pip install datus-snowflake
```

## 连接配置

密码认证：

```yaml
agent:
  services:
    datasources:
      snowflake_prod:
        type: snowflake
        account: ${SNOWFLAKE_ACCOUNT}
        username: ${SNOWFLAKE_USER}
        password: ${SNOWFLAKE_PASSWORD}
        warehouse: ${SNOWFLAKE_WAREHOUSE}
        database: ANALYTICS
        schema: PUBLIC
        role: ANALYST
```

使用内存 secret 的 key-pair 认证：

```yaml
snowflake_ci:
  type: snowflake
  account: ${SNOWFLAKE_ACCOUNT}
  username: ${SNOWFLAKE_USER}
  private_key: ${SNOWFLAKE_PRIVATE_KEY}
  warehouse: ${SNOWFLAKE_WAREHOUSE}
  database: ANALYTICS
  schema: PUBLIC
```

本地密钥文件可把 `private_key` 换成 `private_key_file`；仅当 PEM 已加密时才设置 `private_key_file_pwd`。

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `account` | string | 是 | — | Snowflake account identifier。 |
| `username` | string | 是 | — | Snowflake 用户。 |
| `password` | secret string | 条件必填 | — | 密码认证；没有 `private_key` 时与 `private_key_file` 互斥。 |
| `private_key` | secret string | 条件必填 | — | Inline PEM；优先级高于其他凭证。 |
| `private_key_file` | string | 条件必填 | — | PEM 编码 RSA 私钥路径。 |
| `private_key_file_pwd` | secret string | 否 | — | 加密私钥的 passphrase。 |
| `warehouse` | string | 是 | — | 计算 warehouse。 |
| `database` | string | 否 | — | 初始 database。 |
| `schema` | string | 否 | — | 初始 schema。 |
| `role` | string | 否 | — | 会话启用的 role。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## 认证、TLS 与命名空间

可以配置 `private_key`；如果没有它，则必须在 `password` 与 `private_key_file` 中恰好选择一个。同时提供其他凭证时，`private_key` 优先。Key-pair 认证内部使用 Snowflake JWT，适合启用 MFA 的用户和 CI。

`private_key` 与 `private_key_file` 是 Snowflake JWT 用户认证所用的 RSA 私钥，不是 TLS 证书。当前 adapter profile 不暴露自定义 CA bundle 或双向 TLS 客户端证书；HTTPS 服务端证书校验由原生 Snowflake connector 管理。

Snowflake 使用 `database` 与 `schema`，不要配置 `catalog`。登录 role 需要 warehouse、database、schema 的 `USAGE`，以及 Datus 要发现或查询对象的权限。

## 验证连接

```bash
datus --config conf/agent.yml --datasource snowflake_prod
```

运行 `/databases`、`/schemas` 和 `/tables`，验证 role 激活和授权。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 提示必须恰好选择一个凭证 | 使用 `private_key`，或在 `password`/`private_key_file` 中二选一。 |
| 无法加载私钥 | 使用 PEM 编码；只有加密私钥才配置 `private_key_file_pwd`。 |
| 看不到 warehouse 或对象 | 检查 `role`、`USAGE`/对象授权；带引号的对象名可能区分大小写。 |

## 参考

- [Snowflake key-pair 认证](https://docs.snowflake.com/en/user-guide/key-pair-auth)
