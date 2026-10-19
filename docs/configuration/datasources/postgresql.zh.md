# PostgreSQL datasource

PostgreSQL adapter 通过 PostgreSQL wire protocol 与 psycopg2 连接，保留原生 `database → schema → table` 层级，并支持发现 table、view 和 materialized view。

## 安装

```bash
pip install datus-postgresql
```

下述证书和 AWS 认证字段面向待发布的 `datus-postgresql>=0.1.10`、`datus-sqlalchemy>=0.1.10`、`datus-db-core>=0.1.8`。这些版本尚未发布；发布前需安装对应源码包。较旧公开版本只暴露 `sslmode`，证书仍需通过 libpq 标准文件或环境配置。

发布前可把以下三个兼容源码包安装到 Agent 环境。固定的 adapter 提交包含本页的 TLS／AWS 字段；请把 Python 路径替换为实际运行 Datus 的环境：

```bash
git clone https://github.com/Datus-ai/datus-db-adapters.git
git -C datus-db-adapters checkout 66fadf1bffc1710c6b85e4149d6bb08d3458d464
uv pip install --python /path/to/Datus-agent/.venv/bin/python \
  -e ./datus-db-adapters/datus-db-core \
  -e ./datus-db-adapters/datus-sqlalchemy \
  -e './datus-db-adapters/datus-postgresql[aws]' \
  'SQLAlchemy==2.0.23'
```

`aws` extra 安装 IAM／Secrets Manager 所需 SDK，普通密码连接可省略。SQLAlchemy 保持 Agent 固定的 `2.0.23`。源码安装后通过 uv 启动时使用 `uv run --no-sync`，普通同步可能按 Agent 锁文件重新安装旧 core。

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
        sslmode: verify-full
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
| `sslrootcert` | string | 否 | — | 传给 libpq 的 CA bundle 文件路径；Aurora/IAM 必填。 |
| `sslcert` / `sslkey` | string | 否 | — | mTLS 客户端证书和私钥文件路径，必须成对提供。 |
| `aurora` | boolean | 否 | `false` | 对 Aurora endpoint 强制 CA 和 `verify-full`。 |
| `auth_mode` | string | 否 | `password` | `password` 或 AWS RDS `iam`。 |
| `aws` | object | 否 | — | SDK 身份设置：`region`、`profile`、`ca_bundle`（仅 AWS API 信任）。 |
| `secret_ref` | string | 否 | — | Secrets Manager 名称/ARN；不能与内联密码或 IAM 同用。 |
| `secret_cache_ttl_seconds` | integer | 否 | `300` | Secret 刷新周期，1–86400 秒。 |
| `connect_timeout_seconds` | integer | 否 | — | 可选正整数 DBAPI 建连超时。 |
| `timeout_seconds` | integer | 否 | `30` | 连接与连接池超时秒数。 |

## TLS 与命名空间

`sslmode` 会传给 psycopg2/libpq。已有 profile 的 adapter 默认值仍为 `prefer`，可能回退明文，且不提供服务端身份验证。新生产配置应显式选择 `verify-full`，使用与证书匹配的 hostname，并配置签发 CA。0.1.10 起可增加：

```yaml
sslrootcert: /absolute/path/postgresql-ca.pem
# mTLS 必须同时提供以下两个客户端文件：
# sslcert: /absolute/path/client.crt
# sslkey: /absolute/path/client.key
```

这些字段接受文件路径，不接受内联 PEM。普通 PostgreSQL 省略 `sslrootcert` 时仍可使用 libpq 标准 CA 文件或环境。旧 `prefer` profile 迁移前应先准备可信 CA 和匹配 hostname；仅升级 adapter 不会改变它的连接策略。

### AWS 认证

发布后安装 `datus-postgresql[aws]>=0.1.10`，发布前可安装源码包的 `aws` extra。数据库 TLS 信任使用 `sslrootcert`；`aws.ca_bundle` 只影响 AWS HTTPS API。使用 SDK 凭据链或 AWS shared profile，包括由 SDK 自动刷新的 role；不要把临时 AWS 凭据粘贴到 datasource。

RDS/Aurora IAM 配置应省略 `password` 与 `secret_ref`，并增加：

```yaml
aurora: true  # 仅 Aurora endpoint 设置
auth_mode: iam
aws:
  region: us-east-1
  profile: analytics
sslmode: verify-full
sslrootcert: /absolute/path/rds-ca-bundle.pem
connect_timeout_seconds: 10
```

`aurora: true` 和 `auth_mode: iam` 都要求显式 CA 路径；省略 `sslmode` 时自动使用 `verify-full`，显式选择其他模式会报错。IAM token 按新物理连接生成。密码认证也可用 `auth_mode: password` + `secret_ref`：Secret 必须包含 JSON 字符串 `password`，若含 `username` 必须与配置一致。Secret 按 TTL 刷新，版本变化会释放旧连接池后再建连。数据库 IAM 登录和 AWS 权限仍需单独启用。

[GaussDB](gaussdb.zh.md)／[DWS](dws.zh.md) 有独立 adapter 和证书限制，应参照各自 datasource 页面；不能用 PostgreSQL 的 `verify-full` 模板代替这些产品的主机名验证。

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
| 证书校验失败 | 0.1.10 起可通过 `sslrootcert` 提供签发 CA 文件，并让 `host` 匹配证书；旧包使用 libpq 标准文件/环境。 |
| 找不到 relation | 检查 `database`、`schema`、`search_path`、标识符大小写与权限。 |

## 参考

- [PostgreSQL libpq SSL 支持](https://www.postgresql.org/docs/current/libpq-ssl.html)
