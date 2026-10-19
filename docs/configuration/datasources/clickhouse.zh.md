# ClickHouse datasource

ClickHouse adapter 通过 ClickHouse HTTP(S) 接口连接。ClickHouse database 直接作为命名空间，不再有独立 schema 层。

## 安装

```bash
pip install datus-clickhouse
```

该命令安装已发布的 adapter，适用于下方 HTTP 配置。本页 HTTPS 字段面向即将发布的 `0.1.4`，所需版本见后文说明。

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
| `port` | integer | 否 | `8123` / `8443` | 省略时按 `protocol` 选择；显式自定义端口保持不变。 |
| `username` | string | 是 | — | ClickHouse 用户。 |
| `password` | string | 否 | 空字符串 | 登录密码。 |
| `database` | string | 否 | — | 初始 database。 |
| `timeout_seconds` | integer | 否 | `30` | SQLAlchemy 连接池等待时间，不是查询执行截止时间。 |
| `protocol` | string | 否 | `http` | `http` 或 `https`；端口号本身不会切换协议。 |
| `verify` | boolean | 否 | `true` | 验证 HTTPS 证书链和主机名。 |
| `ca_cert` | string | 否 | — | 验证 HTTPS 使用的 PEM CA bundle 文件路径。 |

## 协议与命名空间

应连接 HTTP endpoint（默认 `8123`），不要使用 native TCP 端口（默认 `9000`）。对象使用 `database.table`，不要添加 `schema`。

TLS 需要 `datus-clickhouse>=0.1.4` 及其共享依赖（`datus-sqlalchemy>=0.1.10`、`datus-db-core>=0.1.8`）。这些版本尚未在 PyPI 发布；请在发布后使用下方 HTTPS 配置。旧配置继续使用 HTTP。

```yaml
agent:
  services:
    datasources:
      clickhouse_prod:
        type: clickhouse
        host: clickhouse.example.com
        protocol: https
        port: 8443
        username: ${CLICKHOUSE_USER}
        password: ${CLICKHOUSE_PASSWORD}
        database: analytics
        verify: true
        ca_cert: /absolute/path/clickhouse-ca.pem
```

省略 `ca_cert` 时使用 Requests 的默认信任配置。自建 CA 应提供可读的 PEM bundle 文件；支持展开 `~`，相对路径以进程工作目录为基准。上述字段不支持内联 PEM 或客户端证书认证。

HTTPS 默认校验证书链和主机名。`ca_cert` 必须与 HTTPS、开启验证一起使用；冲突配置和无效文件在配置校验时失败。错误 CA／主机名会导致连接失败，不会回退重试 HTTP。`verify: false` 仅用于显式关闭验证进行开发诊断。应使用服务器的 HTTPS 端点（通常为 8443），不要误用 native TLS 端口 9440。

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
| 证书校验失败 | 检查 CA bundle，并使用证书 SAN 中列出的主机名。 |
| HTTPS 连接明文 HTTP 端口失败 | 同时检查 `protocol` 与 HTTP(S) endpoint 端口。 |
| 认证失败 | 检查 ClickHouse 用户、密码、允许的网络与 database grant。 |

## 参考

- [ClickHouse HTTP 接口](https://clickhouse.com/docs/interfaces/http)
