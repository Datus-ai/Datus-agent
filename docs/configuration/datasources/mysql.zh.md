# MySQL datasource

MySQL adapter 通过 PyMySQL 连接数据库。MySQL database 对应 Datus database，不再有独立的 schema 层。

## 安装

```bash
pip install --upgrade 'datus-mysql>=0.1.10' \
  'datus-sqlalchemy>=0.1.12' 'datus-db-core>=0.1.10' 'SQLAlchemy==2.0.23'
```

这些已发布版本包含下述 TLS 和 AWS 认证字段。Aurora 密码、IAM 和 Secrets Manager 配置见 [Aurora 指南](aurora.zh.md)；IAM 和 Secrets Manager 需要 adapter 的可选 `aws` extra。请安装到 Agent 环境，升级共享包后重启 Datus。

## 连接配置

```yaml
agent:
  services:
    datasources:
      mysql_prod:
        type: mysql
        host: ${MYSQL_HOST:-127.0.0.1}
        port: 3306
        username: ${MYSQL_USER}
        password: ${MYSQL_PASSWORD}
        database: analytics
        charset: utf8mb4
        autocommit: true
        timeout_seconds: 30
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --: | --- | --- |
| `host` | string | 否 | `127.0.0.1` | MySQL 服务地址。 |
| `port` | integer | 否 | `3306` | MySQL 协议端口。 |
| `username` | string | 是 | — | 登录用户。 |
| `password` | string | 否 | 空字符串 | 登录密码，应使用环境变量。 |
| `database` | string | 否 | — | 初始/默认 database。 |
| `charset` | string | 否 | `utf8mb4` | 连接字符集。 |
| `autocommit` | boolean | 否 | `true` | 是否启用自动提交。 |
| `timeout_seconds` | integer | 否 | `30` | 连接池等待及驱动默认建连/读/写超时。 |
| `connect_timeout_seconds` | integer | 否 | — | 正整数建连超时覆盖值。 |
| `read_timeout_seconds` / `write_timeout_seconds` | integer | 否 | — | 正整数 socket 读/写超时覆盖值。 |
| `query_timeout_seconds` | integer | 否 | — | 可选的正整数服务端 SELECT 截止时间，省略时保留服务端设置。 |
| `ssl_ca` | string | 否 | — | 数据库 CA 文件；Aurora/IAM 必填。 |
| `ssl_cert` / `ssl_key` | string | 否 | — | 客户端证书和私钥，必须成对提供。 |
| `ssl_verify_cert` / `ssl_verify_identity` | boolean | 否 | 未设置 | 证书和主机名校验；启用时必须提供 `ssl_ca`。Aurora/IAM 开启两项并拒绝显式 `false`。 |
| `aurora` | boolean | 否 | `false` | 强制 TLS 验证，并检查可识别的 Aurora cluster writer endpoint。 |
| `auth_mode` | string | 否 | `password` | `password` 或 AWS RDS `iam`。 |
| `aws` | object | 否 | SDK 默认值 | `region`、`profile` 和 AWS HTTPS `ca_bundle`。 |
| `secret_ref` | string | 否 | — | 密码模式使用的 Secrets Manager 名称/ARN，不能同时提供内联密码或 IAM。 |
| `secret_cache_ttl_seconds` | integer | 否 | `300` | Secret 缓存时间，1–86400 秒。 |

## 命名空间与认证

对象使用 `database.table`。Adapter 要求提供用户名；普通密码认证在服务端允许时可使用空密码。需要验证 TLS 时，在 datasource 下增加：

```yaml
ssl_ca: /absolute/path/mysql-ca.pem
ssl_verify_cert: true
ssl_verify_identity: true
```

mTLS 另需同时提供 `ssl_cert` 和 `ssl_key`。MySQL 使用文件路径，`aws.ca_bundle` 独立配置 AWS API 信任。[Aurora](aurora.zh.md) 使用同一个 `mysql` 类型并设置 `aurora: true`，支持密码、IAM 或 Secret 密码认证。IAM 不能同时提供 `password` 或 `secret_ref`。

元数据发现需要当前用户拥有目标 database 的访问权限。如果 Datus 不应修改数据，请在数据库侧为该账号只授予只读 SQL 权限。

## 验证连接

```bash
datus --config conf/agent.yml --datasource mysql_prod
```

运行 `/databases` 和 `/tables`，同时验证连接与元数据权限。

## 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| `Access denied for user` | 检查 `username`、`password`、允许的来源 host 与 MySQL grant。 |
| 连接被拒绝或超时 | 检查 `host`、3306 端口、bind address、防火墙和容器端口映射。 |
| 看不到其他 database 的表 | 为账号授予该 database 的权限，或将 `database` 改为预期初始命名空间。 |

## 参考

- [MySQL 文档](https://dev.mysql.com/doc/)
