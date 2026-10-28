# StarRocks datasource

StarRocks adapter 使用 MySQL 协议，并增加 StarRocks catalog、物化视图和三段命名空间支持。

## 安装

```bash
pip install datus-starrocks
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      starrocks_prod:
        type: starrocks
        host: ${STARROCKS_HOST}
        port: 9030
        username: ${STARROCKS_USER}
        password: ${STARROCKS_PASSWORD}
        catalog: default_catalog
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
| `username` | string | 是 | — | StarRocks 用户。 |
| `password` | string | 否 | 空字符串 | 登录密码。 |
| `catalog` | string | 否 | `default_catalog` | 初始 StarRocks catalog。 |
| `database` | string | 否 | — | 初始 database。 |
| `charset` | string | 否 | `utf8mb4` | 连接字符集。 |
| `autocommit` | boolean | 否 | `true` | 是否启用自动提交。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## Catalog 与对象

StarRocks 对象使用 `catalog.database.table`。原生表使用 `default_catalog`，外部表则选择对应 catalog（例如 Hive catalog）。登录账号不仅要能查询对象，还要能列出 catalog 与 database。

应连接 FE 查询端口（默认 `9030`），不要使用 FE HTTP 端口或 BE 端口。

## 验证连接

```bash
datus --config conf/agent.yml --datasource starrocks_prod
```

运行 `/databases` 和 `/tables`；如果工作流依赖物化视图，再查询一张 MV 验证发现能力。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| MySQL handshake 或连接失败 | 使用 FE MySQL-protocol endpoint 和 9030 端口。 |
| 看不到外部 catalog | 将 `catalog` 设为准确名称，并授予 catalog/database 权限。 |
| 看不到原生表 | 使用 `catalog: default_catalog` 和正确的 `database`。 |

## 参考

- [StarRocks 客户端连接](https://docs.starrocks.io/docs/sql-reference/System_variable/)
