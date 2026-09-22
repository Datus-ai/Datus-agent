# Trino datasource

Trino adapter 通过 Trino HTTP 协议连接，并保留 `catalog → schema → table` 三层结构。

## 安装

```bash
pip install datus-trino
```

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
| `verify` | boolean | 否 | `true` | 是否校验 HTTPS 证书。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时，必须大于 0。 |

## TLS 与命名空间

启用 TLS 时设置 `http_scheme: https`，生产环境应保持 `verify: true`。当前 adapter 只暴露布尔校验开关，没有自定义 CA bundle 路径。

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
| HTTPS 证书失败 | 使用系统信任证书或可信代理终止 TLS；生产环境不要使用 `verify: false`。 |
| 看不到 catalog 或 schema | 检查准确名称、coordinator connector 配置和 Trino access-control 授权。 |

## 参考

- [Trino client protocol](https://trino.io/docs/current/develop/client-protocol.html)
