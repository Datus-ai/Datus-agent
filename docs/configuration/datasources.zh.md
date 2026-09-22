# Datasources

Datasource 是 Datus 用于执行 SQL、发现元数据和构建知识库索引的具名数据库连接。在 `agent.yml` 的 `agent.services.datasources` 下配置：

```yaml
agent:
  services:
    datasources:
      analytics:
        type: postgresql
        host: ${POSTGRES_HOST}
        port: 5432
        username: ${POSTGRES_USER}
        password: ${POSTGRES_PASSWORD}
        database: analytics
        schema: public
        default: true
```

`datasources` 下的每个 key 都是 datasource 名称，只能包含字母、数字、下划线和连字符。最多为一个条目设置 `default: true`；仅有一个 datasource 时，Datus 会自动选择它。

!!! note
    配置路径仍然是 `agent.services.datasources`。Semantic adapter 在[适配器](../adapters/semantic_adapters.md)中配置；Airflow 等外部系统优先使用 [Plugin](../plugin/introduction.md)。

## 选择数据源

SQLite 和 DuckDB 内置在 Datus 中。其他 datasource 均由可独立安装的 `datus-<type>` adapter 提供。在 `/datasource` 中新增数据源时，Datus 可以自动安装缺失的 adapter，也可以手动安装对应包。

| Datasource | `type` | 安装包 | 命名空间 |
|---|---|---|---|
| [SQLite](datasources/sqlite.md) | `sqlite` | 内置 | 数据库文件 → 表 |
| [DuckDB](datasources/duckdb.md) | `duckdb` | 内置 | database → schema → table |
| [MySQL](datasources/mysql.md) | `mysql` | `datus-mysql` | database → table |
| [PostgreSQL](datasources/postgresql.md) | `postgresql` | `datus-postgresql` | database → schema → table |
| [Greenplum](datasources/greenplum.md) | `greenplum` | `datus-greenplum` | database → schema → table |
| [Amazon Redshift](datasources/redshift.md) | `redshift` | `datus-redshift` | database → schema → table |
| [Snowflake](datasources/snowflake.md) | `snowflake` | `datus-snowflake` | database → schema → table |
| [Google BigQuery](datasources/bigquery.md) | `bigquery` | `datus-bigquery` | project → dataset → table |
| [StarRocks](datasources/starrocks.md) | `starrocks` | `datus-starrocks` | catalog → database → table |
| [Apache Doris](datasources/doris.md) | `doris` | `datus-doris` | catalog → database → table |
| [TiDB](datasources/tidb.md) | `tidb` | `datus-tidb` | database → table |
| [ClickHouse](datasources/clickhouse.md) | `clickhouse` | `datus-clickhouse` | database → table |
| [Trino](datasources/trino.md) | `trino` | `datus-trino` | catalog → schema → table |
| [Hive](datasources/hive.md) | `hive` | `datus-hive` | database → table |
| [Spark SQL](datasources/spark.md) | `spark` | `datus-spark` | database → table |
| [ClickZetta](datasources/clickzetta.md) | `clickzetta` | `datus-clickzetta` | instance/workspace → schema → table |
| [Hologres](datasources/hologres.md) | `hologres` | `datus-hologres` | database → schema → table |
| [MaxCompute](datasources/maxcompute.md) | `maxcompute` | `datus-maxcompute` | project → 可选 schema → table |
| [Oracle](datasources/oracle.md) | `oracle` | `datus-oracle` | schema → table |
| [GaussDB / openGauss](datasources/gaussdb.md) | `gaussdb` | `datus-gaussdb` | database → schema → table |
| [GaussDB(DWS)](datasources/dws.md) | `dws` | `datus-dws` | database → schema → table |

## 通用 profile 字段

各 datasource 子页面只列该类型专用的连接字段。下面两个 profile 字段由 Datus 自身处理，适用于全部 datasource：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `type` | string | 是 | — | 上表列出的 adapter 类型。 |
| `default` | boolean | 否 | `false` | 将当前条目标记为默认 datasource。 |

Adapter 配置模型会拒绝未知字段，因此字段名必须与文档完全一致。凭证应通过环境变量传入：

```yaml
username: ${DB_USERNAME}
password: ${DB_PASSWORD}
```

`${NAME:-fallback}` 可以提供默认值。`${NAME}` 未定义时会展开为明显的缺失标记，连接会失败，不会静默使用空凭证。

## 文件数据源批量匹配

SQLite 和 DuckDB 可以用 `path_pattern` 代替 `uri`，将多个文件绑定到一个 datasource：

```yaml
agent:
  services:
    datasources:
      benchmark:
        type: sqlite
        path_pattern: benchmark/databases/**/*.sqlite
        database: california_schools  # 可选：初始文件名（不含扩展名）
```

匹配到的文件会成为同一个 datasource 下的多个 database。启动时如果没有匹配到文件，Datus 会跳过该条目。

## 添加、测试与选择数据源

推荐使用交互式 datasource 管理器：

```bash
datus
```

在 CLI 中运行 `/datasource`。新增或编辑条目时会校验字段、安装缺失的 adapter、测试连通性，并写回 `--config` 或默认搜索顺序选中的当前 `agent.yml`。使用 `/datasource <name>` 可切换当前会话的数据源。

如果手工编辑配置文件，可以指定目标 datasource 启动 Datus。创建 connector 和首次加载元数据时会直接暴露配置、网络与认证错误：

```bash
datus --config conf/agent.yml --datasource analytics
```

项目级默认选择写在 `.datus/config.yml`：

```yaml
default_datasource: analytics
```

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| `Unsupported value ... for field datasource` | 选择的名称不是 `agent.services.datasources` 下的 key。检查拼写和实际加载的配置文件。 |
| Adapter 导入或安装失败 | 在运行 `datus` 的同一个 Python 环境中安装子页面标明的包。 |
| 提示未知或多余字段 | 已安装的 adapter 会按自身 schema 校验配置。移除该字段，或改用对应 datasource 页面中的准确字段名。 |
| 已连接但看不到目标对象 | 检查 adapter 的命名空间字段（`catalog`、`database`、`schema` 或 `schema_name`）以及数据库账号授权。 |
| 凭证显示为 `<MISSING:...>` | 启动 Datus 前先导出对应环境变量。 |

## 相关文档

- [数据库 adapter 架构](../adapters/db_adapters.md)
- [SQL policy 与部署级只读模式](sql_policy.md)
- [Plugin 系统](../plugin/introduction.md)
- [CLI 参考](../cli/reference.md)
