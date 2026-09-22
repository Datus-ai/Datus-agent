# MySQL datasource

MySQL adapter 通过 PyMySQL 连接数据库。MySQL database 对应 Datus database，不再有独立的 schema 层。

## 安装

```bash
pip install datus-mysql
```

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
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | MySQL 服务地址。 |
| `port` | integer | 否 | `3306` | MySQL 协议端口。 |
| `username` | string | 是 | — | 登录用户。 |
| `password` | string | 否 | 空字符串 | 登录密码，应使用环境变量。 |
| `database` | string | 否 | — | 初始/默认 database。 |
| `charset` | string | 否 | `utf8mb4` | 连接字符集。 |
| `autocommit` | boolean | 否 | `true` | 是否启用自动提交。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## 命名空间与认证

对象使用 `database.table`。Adapter 要求提供用户名；如果服务端允许，密码可以为空。当前 profile schema 没有暴露 MySQL TLS 证书字段，`ssl_ca` 等未知字段会被拒绝。

元数据发现需要当前用户拥有目标 database 的访问权限。如果 Datus 不应修改数据，请在数据库侧为该账号只授予只读 SQL 权限。

## 验证连接

```bash
datus --config conf/agent.yml --datasource mysql_prod
```

运行 `/databases` 和 `/tables`，同时验证连接与元数据权限。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| `Access denied for user` | 检查 `username`、`password`、允许的来源 host 与 MySQL grant。 |
| 连接被拒绝或超时 | 检查 `host`、3306 端口、bind address、防火墙和容器端口映射。 |
| 看不到其他 database 的表 | 为账号授予该 database 的权限，或将 `database` 改为预期初始命名空间。 |

## 参考

- [MySQL 文档](https://dev.mysql.com/doc/)
