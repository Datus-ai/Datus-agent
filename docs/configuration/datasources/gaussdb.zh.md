# GaussDB / openGauss datasource

GaussDB adapter 通过 PostgreSQL wire protocol 连接 GaussDB/openGauss，同时处理 GaussDB 认证、兼容模式、分布式表元数据与 TLS 行为。

## 安装

```bash
pip install datus-gaussdb
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      gaussdb_prod:
        type: gaussdb
        host: ${GAUSSDB_HOST}
        port: 5432
        username: ${GAUSSDB_USER}
        password: ${GAUSSDB_PASSWORD}
        database: postgres
        schema: public
        # driver: pg8000
        sslmode: verify-ca
        sslrootcert: /etc/datus/certs/gaussdb-ca.pem
        timeout_seconds: 30
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | GaussDB/openGauss 地址。 |
| `port` | integer | 否 | `5432` | PostgreSQL 兼容端口。 |
| `username` | string | 是 | — | 数据库用户。 |
| `password` | string | 否 | 空字符串 | 密码；支持的认证方式取决于 `driver`。 |
| `database` | string | 否 | 连接时使用 `postgres` | 初始 database。 |
| `schema` | string | 否 | `public` | 初始 schema。 |
| `driver` | string | 否 | 按平台选择 | `gaussdb`、`pg8000` 或 `psycopg2`。 |
| `sslmode` | string | 否 | `prefer` | `disable`、`allow`、`prefer`、`require`、`verify-ca` 或 `verify-full`。 |
| `sslrootcert` | string | 否* | — | CA 路径或 inline PEM。`pg8000` 的证书校验模式必填；libpq 驱动也可使用标准 CA 目录。 |
| `timeout_seconds` | integer | 否 | `30` | 连接与连接池超时秒数。 |

## 驱动与认证

| Driver | 默认平台 | 认证 | 适用场景 |
|---|---|---|---|
| `gaussdb` | Linux | SHA-256、MD5、SM3 | 官方驱动，可连接原生默认服务端。 |
| `pg8000` | macOS | SHA-256、MD5 | 纯 Python，可在任意平台显式选择。 |
| `psycopg2` | 无 | 仅 MD5 | 兼容性兜底。 |

官方驱动没有 macOS build，因此 macOS 自动选择 `pg8000`。`psycopg2` 不仅要求 `pg_hba.conf` 使用 MD5，还要求 role 密码以 GaussDB 的 MD5 兼容设置存储；修改服务端设置不会自动重编码现有密码。

## TLS

| `sslmode` | 加密 | 证书校验 |
|---|---|---|
| `disable` | 关闭 | 无 |
| `allow` | 明文失败后尝试 | 无 |
| `prefer` | 优先 TLS，允许回退明文 | 无 |
| `require` | 强制 | 无；`pg8000` 在提供 `sslrootcert` 时会校验 CA |
| `verify-ca` | 强制 | 用 `sslrootcert` 校验证书链 |
| `verify-full` | 强制 | 校验证书链与 hostname |

生产环境基线建议使用 `verify-ca`；配置 hostname 与证书匹配时使用 `verify-full`。`sslrootcert` 接受路径或 inline PEM。Adapter 只支持服务端校验，不暴露双向 TLS 所需的客户端证书字段。由于 API 限制，`pg8000` 会把 `allow` 按 `prefer` 处理。

## 兼容模式

GaussDB database 可使用 `A`（Oracle）、`B`（MySQL）或 `PG` 兼容模式。Adapter 会在运行时探测兼容模式以及集中式/分布式形态。`A` 模式会把空字符串变为 `NULL`；分布式部署的表 DDL 会补回探测到的 `DISTRIBUTE BY`。

## 验证连接

```bash
datus --config conf/agent.yml --datasource gaussdb_prod
```

运行 `/schemas` 和 `/tables`，并在依赖空字符串、boolean 或算术语义前确认探测到的兼容模式。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| `psycopg2` 无法通过 SHA-256 认证 | 使用 `gaussdb`/`pg8000`，或正确把 role 改为 MD5。 |
| macOS 无法使用官方驱动 | 省略 `driver` 使用 macOS 默认值，或显式设置 `driver: pg8000`。 |
| `verify-ca`/`verify-full` 失败 | 通过 `sslrootcert` 提供签发 CA；`verify-full` 还必须使用证书中的 hostname。 |
| 查询结果与 PostgreSQL 不同 | 检查 database 的 `A`/`B`/`PG` 兼容模式。 |

## 参考

- [openGauss 客户端连接安全](https://docs.opengauss.org/en/docs/latest/docs/DatabaseAdministrationGuide/configuring-client-connection-security.html)
