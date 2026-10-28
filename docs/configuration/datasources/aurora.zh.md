# Amazon Aurora 数据源

Aurora MySQL 使用 [MySQL adapter](mysql.zh.md)（`type: mysql`）；Aurora PostgreSQL 使用 [PostgreSQL adapter](postgresql.zh.md)（`type: postgresql`）。通过 `aurora: true` 开启 TLS 验证和 writer endpoint 检查。连接使用普通 MySQL/PostgreSQL 数据库协议，并保留对应 SQL 方言和元数据层级。

## 安装 {#install}

把所选引擎的 adapter 安装到运行 `datus` 的同一个 Python 环境。以下已发布版本支持本文的 Aurora 配置，并兼容共享凭据处理：

```bash
# Aurora MySQL
pip install --upgrade 'datus-mysql[aws]>=0.1.10' \
  'datus-sqlalchemy>=0.1.12' 'datus-db-core>=0.1.10' 'SQLAlchemy==2.0.23'

# Aurora PostgreSQL
pip install --upgrade 'datus-postgresql[aws]>=0.1.11' \
  'datus-sqlalchemy>=0.1.12' 'datus-db-core>=0.1.10' 'SQLAlchemy==2.0.23'
```

`aws` extra 安装 IAM 和 Secrets Manager 所需的 SDK；仅使用密码时可以省略。升级后重启 Datus。如果从源码 checkout 通过 uv 启动，安装 adapter 后使用 `uv run --no-sync`，避免同步时恢复较旧的共享包。

## 连接配置

### Aurora MySQL

```yaml
agent:
  services:
    datasources:
      aurora_mysql:
        type: mysql
        host: ${AURORA_MYSQL_ENDPOINT}
        port: 3306
        username: ${AURORA_DB_USER}
        database: analytics
        aurora: true
        auth_mode: iam
        aws:
          region: us-east-1
          # profile: analytics
        ssl_ca: /absolute/path/rds-ca-bundle.pem
        ssl_verify_cert: true
        ssl_verify_identity: true
        connect_timeout_seconds: 10
        read_timeout_seconds: 60
        write_timeout_seconds: 60
```

### Aurora PostgreSQL

```yaml
agent:
  services:
    datasources:
      aurora_postgresql:
        type: postgresql
        host: ${AURORA_POSTGRESQL_ENDPOINT}
        port: 5432
        username: ${AURORA_DB_USER}
        database: analytics
        schema: public
        aurora: true
        auth_mode: iam
        aws:
          region: us-east-1
          # profile: analytics
        sslrootcert: /absolute/path/rds-ca-bundle.pem
        sslmode: verify-full
        connect_timeout_seconds: 10
```

`host` 使用实际 Aurora endpoint，并确保 Datus 运行环境能够访问对应数据库 TCP 端口。能访问 AWS API 不代表数据库网络已连通。使用 SDK 默认凭据链时省略 `aws.profile`。MySQL 命名空间为 `database.table`；PostgreSQL 为 `database → schema → table`。

### 密码认证

在上面的任一完整配置中，把 `auth_mode: iam` 替换为以下设置，保留 `aurora: true` 和当前引擎的数据库 CA 字段：

```yaml
auth_mode: password
password: ${AURORA_PASSWORD}
```

### Secrets Manager 认证

使用密码模式，以 `secret_ref` 替代内联密码：

```yaml
auth_mode: password
secret_ref: ${AURORA_SECRET_ARN}
secret_cache_ttl_seconds: 300
```

SecretString 必须是包含字符串 `password` 的 JSON 对象。如果包含 `username`，它必须与配置中的数据库用户一致。运行时 AWS 身份需要 `secretsmanager:GetSecretValue` 和相应 KMS 权限。缓存过期后读取当前 Secret 版本，版本变化时释放旧连接池；已经借出的连接继续工作。更换数据库用户或 AWS 身份时，应替换 connector，而不是仅轮换密码。

## 参数与 TLS

`host`、`database`、`schema` 等连接字段遵循所选 [MySQL](mysql.zh.md) 或 [PostgreSQL](postgresql.zh.md) adapter。Aurora 认证额外使用以下字段：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `aurora` | boolean | `false` | 强制 Aurora TLS 验证，并对可识别的 cluster writer endpoint 检查角色。 |
| `auth_mode` | string | `password` | `password` 或 `iam`；IAM 不能同时配置 `password` 或 `secret_ref`。 |
| `secret_ref` | string | — | 密码模式使用的 Secrets Manager 名称/ARN，不能同时提供内联密码。 |
| `secret_cache_ttl_seconds` | integer | `300` | Secret 缓存时间，范围为 1–86400 秒。 |
| `aws` | object | SDK 默认值 | `region`、`profile`、`ca_bundle`；SDK 提供方管理凭据与刷新。 |
| `connect_timeout_seconds` | integer | — | 正整数数据库建连超时覆盖值。 |

Aurora MySQL 必须设置 `ssl_ca`；`ssl_verify_cert` 和 `ssl_verify_identity` 会设为 `true`，显式关闭任一开关都会报错。Aurora PostgreSQL 必须设置 `sslrootcert`，且 `sslmode` 必须是 `verify-full`（省略时自动选择）。下载适用的 [RDS CA bundle](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/UsingWithRDS.SSL.html)，确保 Datus 可读取，并使用与证书匹配的 endpoint。数据库 CA 设置与 `aws.ca_bundle` 相互独立，后者只影响 AWS HTTPS API 信任。

## IAM 前提条件

1. 在 Aurora cluster 上启用 IAM 数据库认证。
2. 数据库用户配置 MySQL `AWSAuthenticationPlugin` 或 PostgreSQL `rds_iam` role，并授予所需 database/schema/table 权限。
3. 运行时 AWS 身份获得数据库用户资源的 `rds-db:connect` 权限。ARN 使用 **cluster resource ID**，而不是 cluster 名称：`arn:aws:rds-db:<region>:<account-id>:dbuser:<DbClusterResourceId>/<db-user-name>`。
4. IAM 签名使用原始 AWS endpoint、实际端口和准确的数据库用户名，不能用任意 DNS alias 替代签名 endpoint。

跨账号访问时，选择能代入数据库所属账号角色的 SDK AssumeRole profile。SDK 管理临时 AWS 凭据刷新。IAM 数据库 token 按每个新物理连接生成；已建立的会话不需要因为登录 token 到期而重连。

## 超时与故障切换

MySQL 的 `timeout_seconds` 默认是 30 秒，用于连接池和驱动等待；`connect_timeout_seconds`、`read_timeout_seconds`、`write_timeout_seconds` 分别覆盖对应边界。PostgreSQL 的 `timeout_seconds` 提供连接池、socket 操作和查询截止时间；`connect_timeout_seconds`、`read_timeout_seconds`、`query_timeout_seconds` 可分别覆盖对应设置。这些是不同控制项，客户端/socket 超时不代表已确认服务端查询被取消。

对于会自动暂停的 Serverless 实例，应为恢复预留足够的建连时间。针对标准 Aurora cluster writer endpoint，adapter 在借出连接时检查服务端角色，避免 writer 连接池继续使用已降级的 writer 执行业务 SQL。Reader、instance 和 custom endpoint 保留各自角色语义。DNS 或连接失败后应通过新请求显式恢复，SQL 和事务不会自动重放。

## 验证连接

```bash
datus --config conf/agent.yml --datasource aurora_mysql
# Or select the PostgreSQL profile:
datus --config conf/agent.yml --datasource aurora_postgresql
```

运行 `/databases` 和 `/tables`，PostgreSQL 另检查 `/schemas`。执行 `SELECT 1 AS connection_ok;` 验证 SQL 访问。AWS 登录、数据库登录和表权限是独立检查。

## 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| `aurora`、TLS 或 AWS 字段被拒绝 | 在 Agent 环境升级上面列出的 adapter 和共享包，再重启 Datus。 |
| IAM 登录失败 | 检查 cluster IAM 支持、数据库用户 IAM 设置、`rds-db:connect` 的 resource ID/用户名、region、endpoint 和运行时 AWS 身份。 |
| 认证选项冲突 | 从 IAM、内联密码或 Secret 密码中选择一种，不要组合使用。 |
| 证书校验失败 | 使用当前引擎的 CA 字段和原始的证书匹配 endpoint。 |
| 轮换后 Secret 登录失败 | 检查 Secret JSON 字段、用户名、AWS/KMS 权限与缓存时间。 |
| AWS API 正常但数据库连接超时 | 单独检查 VPC 路由、安全组、数据库 TCP 端口和 VPN/隧道。 |
| 故障切换后 writer 连接被拒绝 | 等待 endpoint DNS 收敛后显式重试，不要自动重放结果不确定的事务。 |

## 参考

- [Adapter Aurora 指南](https://github.com/Datus-ai/datus-db-adapters/blob/main/docs/aurora.md)
- [IAM 数据库用户配置](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/UsingWithRDS.IAMDBAuth.DBAccounts.html)
- [IAM 数据库访问策略](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/UsingWithRDS.IAMDBAuth.IAMPolicy.html)
- [Aurora TLS 与 CA 证书](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/UsingWithRDS.SSL.html)
