# Spark SQL datasource

Spark adapter 通过 HiveServer2/PyHive 连接 Spark Thrift Server。Spark 使用两层 `database → table` 结构。

## 安装

```bash
pip install datus-spark
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      spark_warehouse:
        type: spark
        host: ${SPARK_HOST:-127.0.0.1}
        port: 10000
        username: ${SPARK_USER}
        password: ${SPARK_PASSWORD}
        database: default
        auth_mechanism: LDAP
        timeout_seconds: 30
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | Spark Thrift Server 地址。 |
| `port` | integer | 否 | `10000` | Thrift Server 端口。 |
| `username` | string | 是 | — | Spark/Thrift 用户。 |
| `password` | string | 否 | 空字符串 | 所选认证模式的密码。 |
| `database` | string | 否 | 连接时使用 `default` | 初始 Spark database。 |
| `auth_mechanism` | string | 否 | `NONE` | `NONE`、`NOSASL`、`LDAP`、`KERBEROS` 或 `CUSTOM`。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## 认证与命名空间

`auth_mechanism` 会传给 PyHive。服务端要求基于密码的 SASL 时使用 `LDAP` 或 `CUSTOM`。`PLAIN` 是底层 SASL mechanism，不是该 profile 的合法值。

Spark database 就是命名空间，profile 不接受 `schema` 与 `catalog`。Adapter 连接的是 Spark Thrift Server，而不是 Spark driver 进程。

## 验证连接

```bash
datus --config conf/agent.yml --datasource spark_warehouse
```

运行 `/databases` 和 `/tables`，再在 SQL 模式执行 `SELECT 1`。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| `auth_mechanism` 校验失败 | 使用文档列出的五个值之一，不要使用 `PLAIN`。 |
| 启动时连接一直等待 | 检查 Spark Thrift Server 健康和 10000 端口连通性。 |
| Database 为空或不存在 | 检查 Thrift Server 的 catalog 实现与用户权限。 |

## 参考

- [Spark Thrift JDBC/ODBC server](https://spark.apache.org/docs/latest/sql-distributed-sql-engine.html)
