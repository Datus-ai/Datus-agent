# Apache Doris datasource

Doris adapter 使用 MySQL 协议，通过 `catalog.database.table` 同时支持原生和外部 catalog。

## 安装

```bash
pip install datus-doris
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      doris_prod:
        type: doris
        host: ${DORIS_HOST}
        port: 9030
        username: ${DORIS_USER}
        password: ${DORIS_PASSWORD}
        catalog: internal
        database: analytics
        charset: utf8mb4
        autocommit: true
        timeout_seconds: 30
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | FE 查询地址。 |
| `port` | integer | 否 | `9030` | MySQL 协议查询端口。 |
| `username` | string | 是 | — | Doris 用户。 |
| `password` | string | 否 | 空字符串 | 登录密码。 |
| `catalog` | string | 否 | `internal` | 初始 catalog。 |
| `database` | string | 否 | — | 初始 database。 |
| `charset` | string | 否 | `utf8mb4` | 连接字符集。 |
| `autocommit` | boolean | 否 | `true` | 是否启用自动提交。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## Catalog 与对象

原生 Doris 表使用 `catalog: internal`。Hive Metastore 等外部 catalog 可以直接按名称选择。元数据调用和全限定 SQL 都会保留三层命名空间。

应连接 FE MySQL-protocol 查询端口，不要使用 FE HTTP 端口（许多部署中为 `8030`）。

## 验证连接

```bash
datus --config conf/agent.yml --datasource doris_prod
```

运行 `/databases` 和 `/tables`；外部 catalog 还要确认其 metastore 健康，并且 Doris 能访问。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 连接到了 HTTP 服务而不是 MySQL | 使用 FE 9030 端口，不要使用 8030。 |
| 看不到外部 catalog 对象 | 检查 `catalog`、Doris catalog 权限与外部 metastore。 |
| 看不到原生对象 | 设置 `catalog: internal` 和正确的 `database`。 |

## 参考

- [Apache Doris 连接文档](https://doris.apache.org/docs/dev/db-connect/database-connect/)
