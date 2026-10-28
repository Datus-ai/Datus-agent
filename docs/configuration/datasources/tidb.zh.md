# TiDB datasource

TiDB adapter 使用兼容 MySQL 的协议，同时保留 TiDB 特有元数据与 SQL 行为。

## 安装

```bash
pip install datus-tidb
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      tidb_cluster:
        type: tidb
        host: ${TIDB_HOST:-127.0.0.1}
        port: 4000
        username: ${TIDB_USER}
        password: ${TIDB_PASSWORD}
        database: analytics
        charset: utf8mb4
        autocommit: true
        timeout_seconds: 30
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | TiDB 服务地址。 |
| `port` | integer | 否 | `4000` | TiDB 默认端口，不是 MySQL 的 `3306`。 |
| `username` | string | 是 | — | TiDB 用户。 |
| `password` | string | 否 | 空字符串 | 登录密码。 |
| `database` | string | 否 | — | 初始 database。 |
| `charset` | string | 否 | `utf8mb4` | 连接字符集。 |
| `autocommit` | boolean | 否 | `true` | 是否启用自动提交。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## TLS 与 TiDB 行为

当前 adapter profile 不支持 TLS 字段。要求 TLS 的 TiDB Cloud endpoint 因而不受当前版本支持；添加 MySQL SSL 字段只会触发 profile 校验错误，并不会启用 TLS。

对象使用 `database.table`。Adapter 能识别 TiDB/TiFlash 元数据，但生成或手写 SQL 仍须遵守 TiDB 限制，例如不支持 `FULL OUTER JOIN`。

## 验证连接

```bash
datus --config conf/agent.yml --datasource tidb_cluster
```

运行 `/databases` 和 `/tables`。如果需要 TiFlash，再用有权限的账号查询 `information_schema.TIFLASH_REPLICA`。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 连接 3306 被拒绝 | TiDB 默认端口是 4000；使用实际 TiDB server/listener 端口。 |
| 要求 TLS 的云 endpoint 失败 | 当前 adapter 没有 TLS profile 字段；需使用允许当前连接方式的 endpoint。 |
| MySQL 兼容 SQL 仍报错 | 检查 TiDB 自身不支持的语法与版本差异。 |

## 参考

- [TiDB 客户端连接](https://docs.pingcap.com/tidb/stable/connect-to-tidb/)
