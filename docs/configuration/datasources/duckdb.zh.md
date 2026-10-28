# DuckDB datasource

DuckDB 内置在 Datus 中，可以连接持久化 `.duckdb` 文件、内存数据库，也可以在本地引擎中挂载 DuckDB Iceberg REST catalog。

## 连接配置

```yaml
agent:
  services:
    datasources:
      local_duckdb:
        type: duckdb
        uri: duckdb:///data/analytics.duckdb
        read_only: false
        enable_external_access: false
        memory_limit: 2GB
        default: true
```

进程内存数据库使用 `duckdb:///:memory:`。多文件发现使用 `path_pattern`，规则与 [SQLite](sqlite.md) 相同。

Adapter 为兼容性仍默认开启外部访问，但上面的基础示例会将其关闭。只有可信工作负载确实需要访问外部文件或加载扩展时才设置为 `true`。下方 Iceberg 配置因为需要加载扩展并访问外部存储，所以显式开启。

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `uri` | string | 是* | — | DuckDB URI 或文件路径；使用 `path_pattern` 时不填。 |
| `path_pattern` | string | 是* | — | 匹配多个 `.duckdb` 文件的 glob；使用 `uri` 时不填。 |
| `database` | string | 否 | 第一个匹配项 | 使用 `path_pattern` 时，按文件名（不含扩展名）选择初始 database。 |
| `read_only` | boolean | 否 | `false` | 以只读方式打开 DuckDB 文件。 |
| `enable_external_access` | boolean | 否 | `true` | 控制 DuckDB 是否可以访问外部文件和扩展。 |
| `memory_limit` | string | 否 | DuckDB 默认值 | 传给 `SET memory_limit`，例如 `2GB`。 |
| `iceberg` | mapping | 否 | — | 挂载 Iceberg REST catalog，见下文。 |

## Iceberg REST catalog

```yaml
agent:
  services:
    datasources:
      lakehouse:
        type: duckdb
        uri: "duckdb:///:memory:"
        enable_external_access: true
        iceberg:
          catalog_uri: ${ICEBERG_REST_URI}
          warehouse: s3://analytics-warehouse/
          catalog_alias: lake
          client_id: ${ICEBERG_CLIENT_ID}
          client_secret: ${ICEBERG_CLIENT_SECRET}
          oauth2_server_uri: ${ICEBERG_OAUTH_TOKEN_URI}
          s3_region: us-east-1
          s3_endpoint: ${S3_ENDPOINT}
          s3_access_key_id: ${S3_ACCESS_KEY_ID}
          s3_secret_access_key: ${S3_SECRET_ACCESS_KEY}
          read_only: true
```

`iceberg` 内的 `catalog_uri` 和 `warehouse` 必填，`catalog_alias` 默认是 `lake`。OAuth 字段用于创建 DuckDB Iceberg secret，也可以直接提供 `token`。S3 凭证既支持示例中的 `s3_*` 名称，也支持对应的 `aws_*` alias。高级挂载选项包括 `endpoint_type`、`authorization_type`、`access_delegation_mode`、`support_nested_namespaces`、`support_stage_create`、`max_table_staleness` 和 `purge_requested`。

!!! note
    顶层 `read_only` 控制本地 DuckDB 连接；`iceberg.read_only` 独立控制挂载的 catalog。

## 验证连接

```bash
datus --config conf/agent.yml --datasource local_duckdb
```

进入 CLI 后运行 `/databases`、`/tables`，或在 SQL 模式执行 `SELECT 1`。Iceberg 场景下，`httpfs` 与 `iceberg` 扩展加载成功后，`/databases` 应包含配置的 `catalog_alias`。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 外部文件或扩展访问被拒绝 | 设置 `enable_external_access: true`；如果数据源必须与文件系统/网络隔离，应保持关闭。 |
| `DuckDB iceberg config requires catalog_uri` | 添加 `iceberg.catalog_uri`，也接受 `iceberg_catalog_uri` 或 `endpoint` alias。 |
| `DuckDB iceberg config requires warehouse` | 添加逻辑 warehouse 标识，例如 S3 URI。 |
| 文件被锁或写入失败 | 不要用不兼容的读写模式重复打开同一文件，并检查 `read_only` 与文件系统权限。 |

## 参考

- [DuckDB 文档](https://duckdb.org/docs/)
