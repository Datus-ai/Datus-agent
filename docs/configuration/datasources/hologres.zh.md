# Hologres datasource

Hologres adapter 使用 PostgreSQL wire protocol，同时保留 Hologres 特有命名空间、foreign table、表属性与 SQL 规则。

## 安装

```bash
pip install datus-hologres
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      hologres_prod:
        type: hologres
        host: ${HOLOGRES_ENDPOINT}
        port: 80
        username: ${HOLOGRES_ACCESS_KEY_ID}
        password: ${HOLOGRES_ACCESS_KEY_SECRET}
        database: ${HOLOGRES_DATABASE}
        schema: public
        sslmode: prefer
        timeout_seconds: 30
```

`host` 可以是 hostname，也可以直接使用控制台的 `hostname:port`；已经内嵌端口时可省略 `port`。

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 是 | — | Endpoint hostname，可内嵌端口。 |
| `port` | integer | 否 | `80` | Endpoint 端口，范围 1–65535；必须与内嵌端口一致。 |
| `username` | string | 是 | — | AccessKey ID；也接受 `access_key_id`。 |
| `password` | string | 是 | — | AccessKey secret；也接受 `access_key_secret`。 |
| `database` | string | 是 | — | Hologres database。 |
| `schema` | string | 否 | `public` | 初始 schema。 |
| `sslmode` | string | 否 | `prefer` | `disable`、`allow`、`prefer`、`require`、`verify-ca` 或 `verify-full`。 |
| `timeout_seconds` | integer | 否 | `30` | 连接与连接池超时，必须大于 0。 |

## Endpoint、TLS 与命名空间

`host` 中不要包含 URI scheme、凭证、路径、query 或 fragment。`host` 和 `port` 同时携带端口时必须一致。Hologres 公网 endpoint 常用 80 端口，但应以当前实例显示的值为准。

Adapter 暴露 `sslmode`，但没有自定义 `sslrootcert` profile 字段。对象可写成 `table`、`schema.table` 或 `database.schema.table`；常规发现会过滤 Hologres 内部 schema。

## 验证连接

```bash
datus --config conf/agent.yml --datasource hologres_prod
```

运行 `/schemas` 和 `/tables`；如果项目使用外部存储，还应检查一张已知 foreign table。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| Endpoint 端口与显式 port 冲突 | 移除 `port`，或让它与 `host` 内嵌端口一致。 |
| Host 校验拒绝 endpoint | 移除 `http://`、`https://`、路径和用户信息。 |
| 登录被拒绝 | 检查 AccessKey、database、endpoint 类型与实例网络白名单。 |

## 参考

- [Hologres 客户端连接](https://www.alibabacloud.com/help/en/hologres/user-guide/connect-to-hologres)
