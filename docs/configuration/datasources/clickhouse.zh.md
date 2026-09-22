# ClickHouse datasource

ClickHouse adapter 通过 ClickHouse HTTP 接口连接。ClickHouse database 直接作为命名空间，不再有独立 schema 层。

## 安装

```bash
pip install datus-clickhouse
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      clickhouse_prod:
        type: clickhouse
        host: ${CLICKHOUSE_HOST:-localhost}
        port: 8123
        username: ${CLICKHOUSE_USER}
        password: ${CLICKHOUSE_PASSWORD}
        database: analytics
        timeout_seconds: 30
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `localhost` | ClickHouse 服务地址。 |
| `port` | integer | 否 | `8123` | HTTP 接口端口。 |
| `username` | string | 是 | — | ClickHouse 用户。 |
| `password` | string | 否 | 空字符串 | 登录密码。 |
| `database` | string | 否 | — | 初始 database。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## 协议与命名空间

应连接 HTTP endpoint（默认 `8123`），不要使用 native TCP 端口（默认 `9000`）。对象使用 `database.table`，不要添加 `schema`。

当前 adapter profile 没有 TLS/HTTPS 开关。不要通过不可信的明文网络传输凭证；应使用外部支持 HTTPS 的代理或加密隧道，或将连接限制在隔离的可信网络内。

## 验证连接

```bash
datus --config conf/agent.yml --datasource clickhouse_prod
```

运行 `/databases`、`/tables`，并执行一个小型 `SELECT 1` 查询。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 连接 9000 出现协议/handshake 错误 | 使用 HTTP 接口，通常是 8123 端口。 |
| 未知字段 `schema` | ClickHouse database 就是命名空间，移除 `schema`。 |
| 认证失败 | 检查 ClickHouse 用户、密码、允许的网络与 database grant。 |

## 参考

- [ClickHouse HTTP 接口](https://clickhouse.com/docs/interfaces/http)
