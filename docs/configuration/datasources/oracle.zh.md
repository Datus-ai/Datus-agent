# Oracle datasource

Oracle adapter 使用 `python-oracledb` Thin mode，不需要安装 Oracle Client。它生成兼容 Oracle Database 19c 的 SQL，可连接 Oracle Database 12.1 及以上版本。

## 安装

```bash
pip install datus-oracle
```

## 连接配置

现代部署推荐使用 service/PDB name：

```yaml
agent:
  services:
    datasources:
      oracle_prod:
        type: oracle
        host: ${ORACLE_HOST:-127.0.0.1}
        port: 1521
        username: ${ORACLE_USER}
        password: ${ORACLE_PASSWORD}
        service_name: FREEPDB1
        schema: ANALYTICS
        timeout_seconds: 30
```

传统数据库可将 `service_name` 换成 `sid`；TNS alias 或完整 connect descriptor 则使用 `dsn`。

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | Listener 地址，与 `service_name` 或 `sid` 一起使用。 |
| `port` | integer | 否 | `1521` | Listener 端口。 |
| `username` | string | 是 | — | Oracle 用户。 |
| `password` | string | 否 | 空字符串 | 登录密码。 |
| `service_name` | string | 三选一 | — | 推荐的 service/PDB 目标。 |
| `sid` | string | 三选一 | — | 传统 SID 目标。 |
| `dsn` | string | 三选一 | — | TNS alias 或完整 connect descriptor。 |
| `database` | string | 否 | — | `service_name` 的兼容 alias，建议使用显式字段。 |
| `schema` | string | 否 | 登录用户 | 默认对象命名空间。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## 连接目标与命名空间

`service_name`、`sid`、`dsn` 必须且只能配置一个。Service/PDB 只选择连接目标，不进入 SQL 对象名；Oracle 对象使用 `SCHEMA.TABLE`。未设置 `schema` 时，Datus 使用登录用户名的大写形式。

Adapter 运行在 Thin mode。不要向 profile 添加 Thick mode client library 字段。

## 验证连接

```bash
datus --config conf/agent.yml --datasource oracle_prod
```

连接测试会执行 Oracle 要求的 `SELECT 1 FROM DUAL`；随后运行 `/schemas` 和 `/tables` 验证 data dictionary 权限。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 连接目标校验失败 | `service_name`、`sid`、`dsn` 只保留一个。 |
| `ORA-12514` 或 `DPY-6001` | Listener 不认识目标 service；检查服务注册与拼写。 |
| 看不到表 | 把 `schema` 设为对象 owner，并为用户授权；Oracle 未加引号对象名通常显示为大写。 |

## 参考

- [python-oracledb 连接管理](https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html)
