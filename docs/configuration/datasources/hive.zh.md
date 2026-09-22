# Hive datasource

Hive adapter 通过 PyHive/Thrift 连接 HiveServer2。Hive database 是表的命名空间，可通过 `configuration` 传递会话属性。

## 安装

```bash
pip install datus-hive
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      hive_warehouse:
        type: hive
        host: ${HIVE_HOST:-127.0.0.1}
        port: 10000
        username: ${HIVE_USER}
        password: ${HIVE_PASSWORD}
        database: default
        auth: LDAP
        configuration:
          hive.execution.engine: tez
          hive.vectorized.execution.enabled: true
        timeout_seconds: 30
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | HiveServer2 地址。 |
| `port` | integer | 否 | `10000` | HiveServer2 Thrift 端口。 |
| `username` | string | 是 | — | Hive 用户。 |
| `password` | string | 否 | 空字符串 | 密码认证使用。 |
| `database` | string | 否 | 连接时使用 `default` | 初始 Hive database。 |
| `auth` | string | 否 | 驱动默认值 | 支持 `NONE`、`LDAP`、`CUSTOM`、`KERBEROS` 等模式。 |
| `configuration` | mapping | 否 | `{}` | Hive 会话属性，值会被规范化为字符串。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## 认证与会话

HiveServer2 部署要求密码时使用 `LDAP` 或 `CUSTOM`。Kerberos 仍需要外围 Kerberos client/ticket 配置，这些设置不通过额外 datasource 字段表达。

`configuration` 会作为 Hive session configuration 发送。YAML boolean 转为 `true`/`false` 字符串，其他 scalar 会字符串化，`null` 会变成空字符串。

## 验证连接

```bash
datus --config conf/agent.yml --datasource hive_warehouse
```

运行 `/databases` 和 `/tables`，再查询一张小表，验证 HiveServer2 与底层 metastore 访问。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| Thrift 连接失败 | 检查 HiveServer2 host/port、服务健康和网络路由。 |
| SASL/认证错误 | 让 `auth`、用户名/密码与服务端 HiveServer2 认证模式一致。 |
| 会话选项不生效 | 在 `configuration` 中使用准确 Hive property key，并确认服务端允许覆盖。 |

## 参考

- [HiveServer2 clients](https://cwiki.apache.org/confluence/display/Hive/HiveServer2+Clients)
