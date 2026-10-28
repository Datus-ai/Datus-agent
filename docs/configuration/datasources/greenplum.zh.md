# Greenplum datasource

Greenplum 使用 PostgreSQL wire protocol，连接字段与 PostgreSQL 相同。Adapter 额外处理 Greenplum 系统 schema、分布策略元数据和存储信息。

## 安装

```bash
pip install datus-greenplum
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      greenplum_warehouse:
        type: greenplum
        host: ${GREENPLUM_HOST}
        port: 5432
        username: ${GREENPLUM_USER}
        password: ${GREENPLUM_PASSWORD}
        database: analytics
        schema: public
        sslmode: require
        timeout_seconds: 30
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | Greenplum coordinator 地址。 |
| `port` | integer | 否 | `5432` | PostgreSQL 兼容端口。 |
| `username` | string | 是 | — | 登录 role。 |
| `password` | string | 否 | 空字符串 | 登录密码。 |
| `database` | string | 否 | 连接时使用 `postgres` | 初始 database。 |
| `schema` | string | 否 | `public` | 初始 schema。 |
| `sslmode` | string | 否 | `prefer` | PostgreSQL/libpq SSL 模式。 |
| `timeout_seconds` | integer | 否 | `30` | 连接与连接池超时秒数。 |

## Greenplum 行为

对象使用 `database.schema.table`。Datus 会在可用时读取 Greenplum 分布策略和存储元数据，并从常规发现中排除内部 schema。连接 profile 与 [PostgreSQL](postgresql.md) 保持一致，不包含 coordinator/segment 管理字段。

Adapter 默认值是 `sslmode: prefer`；上面的含凭据基础示例使用 `require`，避免 TLS 不可用时静默回退到明文传输。

## 验证连接

```bash
datus --config conf/agent.yml --datasource greenplum_warehouse
```

运行 `/schemas`、`/tables`，再查看一张分布式表，确认账号可以读取 Greenplum catalog。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 登录到了错误服务 | 确认配置的是 coordinator，而不是 segment endpoint。 |
| 缺少分布策略元数据 | 为登录 role 授予相关 Greenplum catalog view 的读取权限。 |
| TLS 或认证失败 | 按 PostgreSQL 相同的 libpq/`pg_hba.conf` 路径排查。 |

## 参考

- [Greenplum 文档](https://greenplum.org/)
- [PostgreSQL datasource](postgresql.md)
