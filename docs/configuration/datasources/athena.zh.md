# Amazon Athena 数据源

Athena adapter 使用 AWS 官方 boto3 SDK 执行查询和发现 catalog 元数据。命名空间为 `catalog.database.table`，没有额外的 schema 层。请在数据 catalog 所在 AWS region 中配置 Athena SQL Workgroup。

## 安装

Adapter 要求 Python 3.12 或更新版本。在 `datus-athena` 发布到 PyPI 前，请把以下固定源码版本安装到运行 `datus` 的同一个 Python 环境：

```bash
pip install --upgrade 'datus-db-core>=0.1.10' \
  'datus-athena @ git+https://github.com/Datus-ai/datus-db-adapters.git@6f5fb395cee2fa0217f281aa9a0af27f33dac68d#subdirectory=datus-athena'
```

该命令需要 Git。安装后重启 Datus。如果从源码 checkout 通过 uv 启动，使用 `uv run --no-sync`，避免同步时恢复较旧的依赖集合。

## 连接配置

先为 Workgroup 配置 S3 查询结果位置或 Athena 托管查询结果：

```yaml
agent:
  services:
    datasources:
      athena_lake:
        type: athena
        aws:
          region: us-east-1
          # profile: analytics
        workgroup: analytics
        catalog: AwsDataCatalog
        database: analytics
        query_timeout_seconds: 600
        max_rows: 10000
```

使用实例、容器角色等 SDK 默认凭据链时，省略 `aws.profile`。如果 Workgroup 未提供结果位置，可在当前 datasource 下增加 `output_location: s3://your-results-bucket/datus/`。

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --: | --- | --- |
| `region` | string | 否 | SDK 配置 | AWS region，也可通过 `aws.region` 设置。 |
| `profile_name` | string | 否 | SDK 默认凭据链 | AWS profile 名称，也可通过 `aws.profile` 设置。 |
| `ca_bundle` | string | 否 | SDK 信任配置 | AWS HTTPS API 的 CA bundle，也可通过 `aws.ca_bundle` 设置；证书验证始终开启。 |
| `aws` | object | 否 | — | 仅接受 `region`、`profile`、`ca_bundle`；平铺字段与嵌套字段冲突时拒绝配置。 |
| `workgroup` | string | 否 | `primary` | Athena SQL Workgroup，不能是空字符串。 |
| `catalog` | string | 否 | `AwsDataCatalog` | 数据 catalog，不能是空字符串。 |
| `database` | string | 否 | — | catalog 中的初始 database。 |
| `output_location` | string | 否 | — | 可选的查询结果位置，格式为 `s3://bucket/prefix/`。 |
| `timeout_seconds` | number | 否 | `30` | 每次 SDK 建连/socket 读取的超时秒数。 |
| `query_timeout_seconds` | number | 否 | `600` | 前台等待查询的截止时间，单位为秒。 |
| `result_timeout_seconds` | number | 否 | `60` | 结果分页读取的截止时间，单位为秒。 |
| `cancel_timeout_seconds` | number | 否 | `10` | 等待取消确认的秒数；`0` 表示发起取消后不等待。 |
| `poll_interval_seconds` | number | 否 | `1` | 查询状态轮询间隔，单位为秒。 |
| `max_rows` | integer | 否 | `10000` | 最多返回的行数。 |
| `max_result_bytes` | integer | 否 | `16777216` | 类型转换前保留的 UTF-8 单元格内容字节上限。 |
| `page_size` | integer | 否 | `1000` | 结果 API 每页行数，范围为 1–1000，包含可能存在的表头行。 |

超时和预算必须为正数，只有 `cancel_timeout_seconds` 同时允许为零。

## AWS 身份与权限

凭据来自 [AWS SDK 凭据提供方](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)：环境变量、shared profile、SSO、web identity、容器或实例角色。长期运行的会话应使用支持刷新的提供方。可通过 `aws.profile` 选择 AssumeRole profile，由 SDK 管理凭据刷新。Access key 和 session token 应保存在 datasource profile 之外。

运行身份需要发现和执行所用的 Athena 操作：`GetWorkGroup`、`ListDataCatalogs`、`ListDatabases`、`GetDatabase`、`ListTableMetadata`、`GetTableMetadata`、`StartQueryExecution`、`GetQueryExecution`、`GetQueryResults` 和 `StopQueryExecution`。此外还需要 Glue catalog 与源 S3 数据读取权限。根据数据和结果配置，另行授予结果前缀、KMS 和 Lake Formation 权限。能读取元数据不代表具备查询数据的权限。

## 结果存储、超时与查询成本

每次查询都会读取 Workgroup 配置。[Workgroup 强制设置](https://docs.aws.amazon.com/athena/latest/ug/workgroups-settings-override.html)和 [Athena 托管查询结果](https://docs.aws.amazon.com/athena/latest/ug/managed-results.html)优先于 `output_location`。使用托管结果时，不需要配置客户端 S3 结果位置。两种模式都通过 Athena API 分页获取结果。

`max_rows` 和 `max_result_bytes` 限制返回结果，并报告截断状态，不能限制 Athena 扫描的数据量。请在 Workgroup 中配置单次查询扫描字节限制；SQL `LIMIT` 也不能保证扫描成本。

查询等待超时或轮询失败时，adapter 会发起原生取消，并最多等待 `cancel_timeout_seconds` 以确认结果。发起取消不等于查询已停止；执行诊断会区分取消请求与确认状态，并保留 query ID。截止时间在 SDK 调用之间检查，socket 超时和 SDK 重试可能延长实际等待时间。失败的查询不会自动重新提交。

## 命名空间与支持范围

SQL 对象使用 `catalog.database.table`。SELECT 标识符使用双引号，Hive 风格表 DDL 使用反引号。Adapter 支持发现表、视图、分区列与 DDL，并进行有界采样。DDL/DML 是否可用取决于表格式和权限；UPDATE、DELETE、MERGE 要求受支持的 Iceberg 表。事务、会话 `SET`/`USE`、Spark Workgroup 和通用 dataframe 迁移不在支持范围内。

## 验证连接

```bash
datus --config conf/agent.yml --datasource athena_lake
```

运行 `/databases` 和 `/tables` 检查元数据发现，再执行以下小查询，检查 Workgroup、结果存储和结果权限：

```sql
SELECT 1 AS connection_ok;
```

## 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| 无法从 PyPI 安装 adapter | 在包发布前使用上面的固定源码安装命令。 |
| AWS 凭据或 token 错误 | 检查运行时 SDK 身份、region、profile 与凭据刷新。 |
| 元数据正常但 SELECT 失败 | 分别检查源 S3、Glue/Lake Formation、结果存储和 KMS 权限。 |
| 缺少查询结果位置 | 配置 Workgroup 结果，或在 Workgroup 允许时设置 `output_location`。 |
| 结果位置与配置不符 | 检查 Workgroup 是否强制使用自身设置，或启用了托管结果。 |
| 查询超时或结果截断 | 检查 query ID、最终状态、取消确认和等待/行数/字节预算。 |
| `schema` 或凭据字段不被接受 | 使用 `catalog` 和 `database`；AWS 凭据由 SDK 提供方管理。 |

## 参考

- [Adapter 配置与支持边界](https://github.com/Datus-ai/datus-db-adapters/blob/main/datus-athena/README.md)
- [Athena Workgroup 设置](https://docs.aws.amazon.com/athena/latest/ug/workgroups-settings-override.html)
- [Athena 托管查询结果](https://docs.aws.amazon.com/athena/latest/ug/managed-results.html)
