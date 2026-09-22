# PostgreSQL datasource

PostgreSQL adapter 通过 PostgreSQL wire protocol 与 psycopg2 连接，保留原生 `database → schema → table` 层级，并支持发现 table、view 和 materialized view。

## 安装

```bash
pip install datus-postgresql
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      warehouse_pg:
        type: postgresql
        host: ${POSTGRES_HOST}
        port: 5432
        username: ${POSTGRES_USER}
        password: ${POSTGRES_PASSWORD}
        database: analytics
        schema: public
        sslmode: prefer
        timeout_seconds: 30
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | PostgreSQL 服务地址。 |
| `port` | integer | 否 | `5432` | PostgreSQL wire-protocol 端口。 |
| `username` | string | 是 | — | 登录 role。 |
| `password` | string | 否 | 空字符串 | 登录密码。 |
| `database` | string | 否 | 连接时使用 `postgres` | 初始 database。 |
| `schema` | string | 否 | `public` | 初始 schema，是 adapter `schema_name` 的 alias。 |
| `sslmode` | string | 否 | `prefer` | `disable`、`allow`、`prefer`、`require`、`verify-ca` 或 `verify-full`。 |
| `timeout_seconds` | integer | 否 | `30` | 连接与连接池超时秒数。 |

## TLS 与命名空间

`sslmode` 会传给 psycopg2/libpq。当前 profile 暴露 `sslmode`，但没有 `sslrootcert` 字段；证书校验模式依赖 libpq 的标准证书目录和环境配置。不要在 datasource 中加入 adapter 不支持的证书字段。

`database` 是初始 PostgreSQL database，`schema` 默认是 `public`。常规发现会过滤系统 database/schema；访问其他 database 仍取决于网络连通性和 role 权限。

## 验证连接

```bash
datus --config conf/agent.yml --datasource warehouse_pg
```

运行 `/databases`、`/schemas` 和 `/tables`，可以区分“登录成功”和“缺少 catalog 权限”。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| `password authentication failed` | 检查 role/password 与匹配的 `pg_hba.conf` 规则。 |
| `no pg_hba.conf entry` | 放行 Datus 客户端地址，并选择正确的 TLS/认证规则。 |
| 证书校验失败 | 使用合法 `sslmode`，并通过 libpq 标准环境/文件配置可信 CA；adapter profile 没有 `sslrootcert` 字段。 |
| 找不到 relation | 检查 `database`、`schema`、`search_path`、标识符大小写与权限。 |

## 参考

- [PostgreSQL libpq SSL 支持](https://www.postgresql.org/docs/current/libpq-ssl.html)
