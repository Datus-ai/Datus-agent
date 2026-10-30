# SAP HANA 数据源

HANA adapter 使用 SAP 官方 Python `hdbcli` 客户端。SQL 对象使用 `schema.table`；可选的 tenant database 用于选择连接目标。客户端安装 Python adapter，HANA 服务运行在独立服务器上。

## 安装

使用 Python 3.12 或更新版本，以及包含 HANA SQL 分析支持的 Datus 构建。在 `datus-hana` 发布到 PyPI 前，请把以下固定源码版本安装到运行 `datus` 的同一个 Python 环境：

```bash
pip install --upgrade 'datus-db-core>=0.1.10' \
  'datus-hana @ git+https://github.com/Datus-ai/datus-db-adapters.git@6f5fb395cee2fa0217f281aa9a0af27f33dac68d#subdirectory=datus-hana'
```

源码安装需要 Git，并会安装 `hdbcli` 依赖。升级 adapter 或共享 core 后请重启 Datus。仅向旧 Agent 安装 SDK 不会增加 HANA 查询分析能力。如果从源码 checkout 通过 uv 启动，安装这些包后使用 `uv run --no-sync`。

## 连接配置

使用密码认证并验证 TLS：

```yaml
agent:
  services:
    datasources:
      hana_prod:
        type: hana
        host: ${HANA_HOST}
        port: 39041
        username: ${HANA_USER}
        password: ${HANA_PASSWORD}
        database: HXE
        schema: ANALYTICS
        encrypt: true
        ssl_validate_certificate: true
        ssl_trust_store: /absolute/path/hana-ca.pem
        timeout_seconds: 30
        communication_timeout_seconds: 60
```

请使用管理员提供的实际 tenant SQL endpoint 和端口。HANA Cloud 通常使用 `443`；`39041` 只是 adapter 默认值。如果 endpoint 已选择目标 tenant 且不需要显式 database 名称，可以省略 `database`。

### 安全用户存储

已有 HANA secure user-store key 时，用 `userkey` 替代用户名和密码：

```yaml
agent:
  services:
    datasources:
      hana_userstore:
        type: hana
        userkey: DATUS_READER
        schema: ANALYTICS
        encrypt: true
        ssl_validate_certificate: true
        ssl_trust_store: /absolute/path/hana-ca.pem
```

使用 SAP `hdbuserstore` 工具，为运行 Datus 的 OS 用户配置 key。安装 Python wheel 不会自动安装该工具或创建 key。省略 `host` 时使用 key 保存的 endpoint。`userkey` 模式必须指定 `schema`，并且不能同时提供 `username` 或 `password`。TLS 设置继续生效。

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --: | --- | --- |
| `host` | string | 密码认证时 | — | Tenant SQL endpoint；`userkey` 模式可省略。 |
| `port` | integer | 否 | `39041` | SQL 端口，范围为 1–65535；提供 `host` 时使用。 |
| `username` | string | 密码认证时 | — | HANA 数据库用户。 |
| `password` | string | 密码认证时 | 空字符串 | 必须是非空密码，请使用环境变量。 |
| `userkey` | string | 否 | — | 已有 secure user-store key，与用户名/密码互斥。 |
| `database` | string | 否 | — | Tenant 连接目标，在当前 datasource 内固定。 |
| `schema` | string | `userkey` 模式 | 配置中的用户名 | 对象命名空间，是 `schema_name` 的 alias。 |
| `encrypt` | boolean | 否 | `true` | 开启数据库连接加密。 |
| `ssl_validate_certificate` | boolean | 否 | `true` | 验证服务端证书。 |
| `ssl_trust_store` | string | 否 | SDK 信任配置 | 签发 CA 的 trust-store 文件路径。 |
| `ssl_host_name_in_certificate` | string | 否 | SDK 默认值 | 需要显式指定时使用的预期证书主机名。 |
| `timeout_seconds` | integer | 否 | `30` | 正整数建连超时秒数。 |
| `communication_timeout_seconds` | integer | 否 | `60` | 正整数网络请求超时秒数。 |
| `max_result_rows` | integer | 否 | `10000` | 正整数结果行数上限。 |
| `max_result_bytes` | integer | 否 | `16777216` | 正整数结果字节上限。 |

## TLS 与命名空间

生产连接应保持加密和证书验证开启，配置部署所需 CA trust store，并使用与证书匹配的主机名。Adapter 会把超时秒数转换为 SAP 客户端要求的毫秒；communication timeout 限制网络等待，不代表已确认服务端 SQL 被取消。

`database` 在建连时选择 tenant。SQL 对象应写成 `schema.table`，例如 `"ANALYTICS"."ORDERS"`。HANA 将未加引号的标识符转为大写；混合大小写名称应按元数据返回的原始大小写加双引号。切换 tenant 需要另一个 datasource；单次查询上下文只选择 schema。

## 查询支持与结果限制

Datus 会在执行前检查常见的 SELECT、JOIN、CTE、聚合、窗口、集合操作和 LIMIT 查询。SQLScript、Calculation View 参数、序列访问、表函数、锁定查询和多语句输入不在本地检查支持范围内，会被拒绝。该 adapter 不支持完整 HANA SQL 格式化、policy 改写、跨方言转换和静态血缘。通过分析的 SQL 会原样交给 SAP 客户端。

结果支持 list、CSV、pandas 和 Arrow。超过行数或字节预算会明确失败，不会静默截断。DML 批次可以使用同一事务，但 HANA DDL 可能隐式提交。连接失败后不会自动重放写入。元数据 DDL 描述列信息，不是无损 schema 迁移导出。

## 验证连接

```bash
datus --config conf/agent.yml --datasource hana_prod
```

运行 `/schemas` 和 `/tables` 检查元数据权限，再执行：

```sql
SELECT 1 AS CONNECTION_OK FROM DUMMY;
```

## 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| 无法从 PyPI 安装 adapter | 在包发布前使用上面的固定源码安装命令。 |
| 认证或连接失败 | 检查实际 tenant endpoint、端口、用户名/密码或 secure user-store key，以及网络访问。 |
| `userkey` 校验失败 | 提供 `schema`、省略用户名/密码，并为运行 Datus 的 OS 用户配置 key。 |
| TLS 校验失败 | 检查签发 CA、trust-store 路径和预期证书主机名。 |
| 找不到表 | 使用 `schema.table`，匹配带引号标识符的大小写，并检查 schema/对象权限。 |
| SQL 在执行前被拒绝 | 使用受支持的查询分析子集；仅安装 `hdbcli` 不会扩展 Agent SQL 分析能力。 |
| 结果超过预算 | 减少选择的行/列，或配置合适的 `max_result_rows` 和 `max_result_bytes`。 |

## 参考

- [Adapter 配置与支持边界](https://github.com/Datus-ai/datus-db-adapters/blob/main/datus-hana/README.md)
- [SAP Python 客户端](https://pypi.org/project/hdbcli/)
- [SAP 客户端连接参数](https://help.sap.com/docs/hana-cloud-database/f1b440ded6144a54ada97ff95dac7adf/ee592e89dcce4480a99571a4ae7a702f.html)
