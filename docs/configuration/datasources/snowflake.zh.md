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
agent:
  services:
    datasources:
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
| `username` | string | 条件必填 | — | Snowflake 用户；也可从 Secret 映射中读取。 |
| `auth_mode` | string | 配置 `secret_ref` 时必填 | — | `key_pair` 或 `password`；静态配置保留原有凭据选择方式。 |
| `secret_ref` | string | 否 | — | AWS Secrets Manager Secret 名称或完整 ARN。 |
| `secret_key_mapping` | object | 否 | — | 为 `username`、`private_key`、`private_key_file_pwd` 或 `password` 指定顶层 JSON 键名。 |
| `secret_cache_ttl_seconds` | integer | 否 | `300` | 凭据缓存时长，范围 1–300 秒。 |
| `aws` | object | 否 | SDK 默认配置 | 可选 `region`、`profile` 和 AWS HTTPS `ca_bundle`。 |
| `password` | secret string | 条件必填 | — | 密码认证；没有 `private_key` 时与 `private_key_file` 互斥。 |
| `private_key` | secret string | 条件必填 | — | Inline PEM；优先级高于其他凭证。 |
| `private_key_file` | string | 条件必填 | — | PEM 编码 RSA 私钥路径。 |
| `private_key_file_pwd` | secret string | 否 | — | 加密私钥的 passphrase。 |
| `warehouse` | string | 是 | — | 计算 warehouse。 |
| `database` | string | 否 | — | 初始 database。 |
| `schema` | string | 否 | — | 初始 schema。 |
| `role` | string | 否 | — | 会话启用的 role。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## AWS Secrets Manager 凭据

需要 `datus-snowflake` 0.1.11 或更高版本，并安装可选 AWS 依赖。0.1.11 尚未发布；发布前请从包含 Secrets Manager 实现的源码 checkout 安装：

```bash
pip install '/path/to/datus-db-adapters/datus-db-core[aws]' \
  /path/to/datus-db-adapters/datus-snowflake
```

发布后可安装正式版本：

```bash
pip install 'datus-snowflake[aws]>=0.1.11'
```

Secret 的 `SecretString` 可以保存 JSON，例如 `{"sf_user":"READER","sf_key":"<PEM 私钥>","sf_passphrase":"<私钥口令>"}`，再配置顶层键名映射：

```yaml
agent:
  services:
    datasources:
      snowflake_prod:
        type: snowflake
        account: ${SNOWFLAKE_ACCOUNT}
        warehouse: ${SNOWFLAKE_WAREHOUSE}
        database: ANALYTICS
        schema: PUBLIC
        role: READER
        auth_mode: key_pair
        secret_ref: ${SNOWFLAKE_SECRET_ARN}
        secret_cache_ttl_seconds: 300
        secret_key_mapping:
          username: sf_user
          private_key: sf_key
          private_key_file_pwd: sf_passphrase
        aws:
          region: us-east-1
          profile: datus-prod
```

未加密 PEM 不设置口令映射。Secret 的无关字段会被忽略，映射值必须是非空字符串；键名只匹配顶层 JSON 字段。账号、warehouse、role 和命名空间始终来自连接配置。同时配置用户名和用户名映射时，两者必须一致。有效用户名变化时，需替换 connector，并刷新旧身份下缓存的元数据。

不配置映射或映射全部留空时，整个 `SecretString` 就是所选凭据。例如使用 `auth_mode: key_pair`、`username: READER` 和 `secret_ref` 读取原始未加密 PEM；原始密码使用 `auth_mode: password`，JSON 密码则把 `password` 映射到对应键。密码认证仍受 Snowflake 用户 MFA 策略约束。`secret_ref` 与直接配置的密码、私钥、私钥文件和解密口令互斥。不支持 `SecretBinary`。

只有新建物理连接时才获取凭据。缓存到期后，已打开的 Snowflake 会话继续正常使用；不会后台轮询或定时关闭会话。后续登录在缓存过期后读取 `AWSCURRENT`。发布并验证新密钥时，旧密钥至少保留一个缓存窗口；提前撤销会使窗口内使用旧缓存的新登录失败。SDK 调用方可用 `connector.refresh_credentials()` 更新下次登录所用缓存，当前连接保持打开；`close()` 后 `connect()` 使用当前缓存新建会话。重连恢复受管理的 database/schema 以及配置的 role/warehouse，不恢复临时对象或任意会话设置。业务 SQL 不会自动重放。

[AWS Secrets Manager 的更新存在传播延迟](https://docs.aws.amazon.com/secretsmanager/latest/userguide/troubleshoot.html#troubleshoot_general_eventual-consistency)。撤销旧密钥前，应确认替换版本已作为 `AWSCURRENT` 可见；300 秒上限仅指 adapter 缓存，不代表 AWS 全链路传播时限。

SDK 会话意外关闭时会返回连接错误。调用方须显式调用 `connect()` 开启新会话后再提交 SQL，adapter 不会在新会话中静默继续已经丢失的事务。

AWS 认证复用 SDK 凭据链。跨账号读取时，在共享 AWS 配置中设置 Role 和 External ID，再由数据源选择 profile：

```ini
[profile datus-prod]
role_arn = arn:aws:iam::123456789012:role/DatusSecretReader
source_profile = default
external_id = customer-external-id
region = us-east-1
```

调用身份需要 AssumeRole 权限，目标 Role 的信任策略须接受该身份和 External ID。读取身份需要 Secret 的 `secretsmanager:GetSecretValue` 权限；使用自管 KMS 密钥时还需要 `kms:Decrypt`。临时 AWS 凭据续签由 SDK 负责。`aws.ca_bundle` 只用于 AWS HTTPS API。此凭据来源不增加 OIDC issuer 或 Snowflake OAuth 流程。

## 认证、TLS 与命名空间

静态凭据可以配置 `private_key`；如果没有它，则必须在 `password` 与 `private_key_file` 中恰好选择一个。同时提供其他凭证时，`private_key` 优先。Key-pair 认证内部使用 Snowflake JWT，适合启用 MFA 的用户和 CI。

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
- [AWS SDK 凭据与 profile](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)
- [AWS GetSecretValue 权限与版本阶段](https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html)
