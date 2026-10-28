# Amazon Athena datasource

The Athena adapter uses the official AWS boto3 SDK to execute queries and discover catalog metadata. Its namespace is `catalog.database.table`; there is no additional schema level. Use an Athena SQL Workgroup in the AWS region that contains your data catalog.

## Install

The adapter requires Python 3.12 or later. Until `datus-athena` is published on PyPI, install the following source revision into the same Python environment as `datus`:

```bash
pip install --upgrade 'datus-db-core>=0.1.10' \
  'datus-athena @ git+https://github.com/Datus-ai/datus-db-adapters.git@6f5fb395cee2fa0217f281aa9a0af27f33dac68d#subdirectory=datus-athena'
```

This command requires Git. Restart Datus after installation. If launching through uv from a checkout, use `uv run --no-sync` so synchronization does not restore an older dependency set.

## Connection profile

Configure a Workgroup that has either an S3 query-results location or Athena managed query results:

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

Omit `aws.profile` when using the SDK default credential chain, such as an instance or container role. If the Workgroup does not provide a result location, add `output_location: s3://your-results-bucket/datus/` to this datasource.

## Parameters

| Key | Type | Required | Default | Notes |
| --- | --- | --: | --- | --- |
| `region` | string | no | SDK configuration | AWS region; also accepted as `aws.region`. |
| `profile_name` | string | no | SDK default chain | Named AWS profile; also accepted as `aws.profile`. |
| `ca_bundle` | string | no | SDK trust configuration | CA bundle for AWS HTTPS APIs; also accepted as `aws.ca_bundle`. Certificate verification remains enabled. |
| `aws` | object | no | — | Accepts only `region`, `profile`, and `ca_bundle`. Conflicting flat and nested settings are rejected. |
| `workgroup` | string | no | `primary` | Athena SQL Workgroup; nonblank. |
| `catalog` | string | no | `AwsDataCatalog` | Data catalog; nonblank. |
| `database` | string | no | — | Initial database in the catalog. |
| `output_location` | string | no | — | Optional `s3://bucket/prefix/` for query results. |
| `timeout_seconds` | number | no | `30` | Timeout for each SDK connection/socket read, in seconds. |
| `query_timeout_seconds` | number | no | `600` | Foreground query wait deadline, in seconds. |
| `result_timeout_seconds` | number | no | `60` | Result pagination deadline, in seconds. |
| `cancel_timeout_seconds` | number | no | `10` | Wait for cancellation confirmation; `0` requests cancellation without waiting. |
| `poll_interval_seconds` | number | no | `1` | Query-state polling interval, in seconds. |
| `max_rows` | integer | no | `10000` | Maximum returned rows. |
| `max_result_bytes` | integer | no | `16777216` | Maximum retained UTF-8 cell bytes before type conversion. |
| `page_size` | integer | no | `1000` | Result API page size, from 1 to 1000, including any header row. |

Timeouts and budgets must be positive, except `cancel_timeout_seconds`, which also accepts zero.

## AWS identity and permissions

Credentials come from the [AWS SDK credential providers](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html): environment, shared profiles, SSO, web identity, or container/instance roles. Use a refreshable provider for long-running sessions. A named AssumeRole profile can be selected with `aws.profile`; the SDK owns credential refresh. Keep access keys and session tokens outside the datasource profile.

The runtime identity needs the Athena operations used for discovery and execution: `GetWorkGroup`, `ListDataCatalogs`, `ListDatabases`, `GetDatabase`, `ListTableMetadata`, `GetTableMetadata`, `StartQueryExecution`, `GetQueryExecution`, `GetQueryResults`, and `StopQueryExecution`. It also needs Glue catalog access and permission to read the source data in S3. Depending on the data and result configuration, grant result-prefix access, KMS permissions, and Lake Formation access as well. Metadata permissions alone do not establish permission to query the data.

## Result storage, timeouts, and query cost

The adapter reads the Workgroup configuration for each query. [Enforced Workgroup settings](https://docs.aws.amazon.com/athena/latest/ug/workgroups-settings-override.html) and [Athena managed query results](https://docs.aws.amazon.com/athena/latest/ug/managed-results.html) take precedence over `output_location`. With managed results, no client S3 result location is needed. Results are paged through the Athena API in either mode.

`max_rows` and `max_result_bytes` bound returned results and report truncation. They do not bound bytes scanned by Athena. Set per-query scanned-byte controls on the Workgroup; adding SQL `LIMIT` is not a scan-cost guarantee.

On a query wait timeout or polling failure, the adapter requests native cancellation and waits up to `cancel_timeout_seconds` for confirmation. A cancellation request does not prove the query stopped; execution diagnostics distinguish requested and confirmed cancellation and retain the query ID. Deadlines are checked between SDK calls, so socket timeouts and SDK retries can extend wall-clock waiting. Failed queries are not automatically resubmitted.

## Namespace and supported operations

Use `catalog.database.table` for SQL object names. SELECT identifiers use double quotes; Hive-style table DDL uses backticks. The adapter discovers tables, views, partition columns, and DDL and supports bounded sampling. DDL/DML availability depends on the table format and grants; UPDATE, DELETE, and MERGE require supported Iceberg tables. Transactions, session `SET`/`USE`, Spark Workgroups, and generic dataframe migration are outside the adapter's supported operations.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource athena_lake
```

Run `/databases` and `/tables` to check discovery, then execute this small query to check the Workgroup, result storage, and result permissions:

```sql
SELECT 1 AS connection_ok;
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Adapter cannot be installed from PyPI | Use the pinned source installation above until an adapter release is available. |
| AWS credential or token error | Check the runtime SDK identity, region, profile, and credential renewal. |
| Metadata works but SELECT fails | Check source S3, Glue/Lake Formation, result storage, and KMS permissions separately. |
| Query result location is missing | Configure the Workgroup's results or set `output_location` when the Workgroup permits it. |
| Unexpected result location | Check whether the Workgroup enforces its settings or uses managed results. |
| Query times out or results are truncated | Inspect the query ID, terminal state, cancellation confirmation, and configured wait/row/byte budgets. |
| Unknown `schema` or credential field | Use `catalog` and `database`; AWS credentials belong to SDK providers. |

## Reference

- [Adapter configuration and support boundaries](https://github.com/Datus-ai/datus-db-adapters/blob/main/datus-athena/README.md)
- [Athena Workgroup settings](https://docs.aws.amazon.com/athena/latest/ug/workgroups-settings-override.html)
- [Athena managed query results](https://docs.aws.amazon.com/athena/latest/ug/managed-results.html)
