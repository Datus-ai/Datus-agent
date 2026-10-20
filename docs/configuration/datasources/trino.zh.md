# Trino datasource

Trino adapter 通过 Trino HTTP 协议连接，并保留 `catalog → schema → table` 三层结构。

## 安装

```bash
pip install datus-trino
```

下述 CA bundle 字段面向待发布的 `datus-trino>=0.1.4`，共享依赖为 `datus-sqlalchemy>=0.1.10`、`datus-db-core>=0.1.8`。这些版本尚未发布；发布前需安装包含 CMP-TLS-03 的源码。已有不含 `ca_cert` 的 HTTPS profile 继续可用。

## 连接配置

```yaml
agent:
  services:
    datasources:
      trino_prod:
        type: trino
        host: ${TRINO_HOST}
        port: 8443
        username: ${TRINO_USER}
        password: ${TRINO_PASSWORD}
        catalog: hive
        schema_name: analytics
        http_scheme: https
        verify: true
        timeout_seconds: 30
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | Trino coordinator 地址。 |
| `port` | integer | 否 | `8080` | HTTP/HTTPS 端口，范围 1–65535。 |
| `username` | string | 是 | — | Trino 用户，空值会被拒绝。 |
| `password` | string | 否 | 空字符串 | Trino client 使用的密码。 |
| `catalog` | string | 否 | `hive` | 初始 catalog。 |
| `schema_name` | string | 否 | `default` | 初始 schema。必须使用这个字段名，不能写 `schema`。 |
| `http_scheme` | string | 否 | `http` | `http` 或 `https`。 |
| `verify` | boolean | 否 | `true` | 校验证书链和主机名。 |
| `ca_cert` | string | 否 | — | 可读 PEM CA bundle 文件；要求 HTTPS 且 `verify: true`，从 0.1.4 提供。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时，必须大于 0。 |

## TLS 与命名空间

启用 TLS 时设置 `http_scheme: https`，生产环境保持 `verify: true`。私有 CA 可在上述 profile 中增加 `ca_cert: /absolute/path/trino-ca.pem`；省略时使用 Python client 的 Requests 信任配置。`verify` 仍是布尔值，不能填文件路径。

`ca_cert` 接受可读 PEM bundle 文件：`~` 展开为用户目录，相对路径基于进程工作目录解析。无效文件、HTTP + CA、`verify: false` + CA 会在连接前被拒绝。此字段不支持内联 PEM 或客户端证书。`verify: false` 会关闭证书链和主机名校验，仅用于临时开发排查。HTTPS 失败不会重试 HTTP；端口不会自动选择协议，TLS 应显式设置 `port` 和 `http_scheme`。

Trino profile 必须使用 `schema_name`；与部分 SQL adapter 不同，它没有为 `schema` 定义 alias。运行时可以切换 catalog，但仍受 Trino access-control 规则限制。

## 验证连接

```bash
datus --config conf/agent.yml --datasource trino_prod
```

运行 `/databases` 和 `/tables`；Datus 会在保留 catalog 上下文的同时，把 Trino schema 映射到 database 列表。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 未知字段 `schema` | 改为 `schema_name`。 |
| HTTPS 证书失败 | 检查 endpoint hostname 和签发 CA；0.1.4 起可通过 `ca_cert` 提供私有 CA。生产环境保持 `verify: true`。 |
| 看不到 catalog 或 schema | 检查准确名称、coordinator connector 配置和 Trino access-control 授权。 |

## 参考

- [Trino TLS 与 HTTPS](https://trino.io/docs/current/security/tls.html)
- [Trino Python client TLS 配置](https://github.com/trinodb/trino-python-client#ssl)
- [Trino client protocol](https://trino.io/docs/current/develop/client-protocol.html)
