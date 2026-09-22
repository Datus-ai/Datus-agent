# Google BigQuery datasource

BigQuery adapter 把 Google Cloud project 映射为 Datus `catalog`，把 BigQuery dataset 映射为 Datus `database`。Dataset 下没有额外 schema 层。

## 安装

```bash
pip install datus-bigquery
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      bigquery_prod:
        type: bigquery
        catalog: ${BIGQUERY_PROJECT}
        database: ${BIGQUERY_DATASET}
        location: ${BIGQUERY_LOCATION:-US}
        timeout_seconds: 60
```

这份基础配置使用 Google Application Default Credentials（ADC），并以 `catalog` 作为默认计费 project。需要使用 service-account 文件或独立计费 project 时，仅在对应环境变量已设置的情况下添加以下字段：

```yaml
credentials_path: ${GOOGLE_APPLICATION_CREDENTIALS}
billing_project_id: ${BIGQUERY_BILLING_PROJECT}
```

托管环境可以使用 base64 编码的 service-account 文档：

```yaml
credentials_base64: ${BIGQUERY_CREDENTIALS_BASE64}
```

也可以把 service-account JSON 写成 YAML mapping：

```yaml
credentials_info:
  type: service_account
  project_id: your-project
  private_key_id: ${BIGQUERY_PRIVATE_KEY_ID}
  private_key: ${BIGQUERY_PRIVATE_KEY}
  client_email: datus@your-project.iam.gserviceaccount.com
  token_uri: https://oauth2.googleapis.com/token
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `catalog` | string | 是 | — | Google Cloud project；也接受 `project`。 |
| `database` | string | 否 | — | 默认 dataset；也接受 `dataset`。 |
| `credentials_path` | string | 否 | ADC | Service-account JSON 文件路径。 |
| `credentials_info` | mapping | 否 | ADC | Service-account JSON object，以 YAML mapping 表示。 |
| `credentials_base64` | secret string | 否 | ADC | Base64 编码的 service-account JSON。 |
| `billing_project_id` | string | 否 | 已配置 project | 计费/配额 project。 |
| `location` | string | 否 | BigQuery/client 默认值 | 作业区域，例如 `US` 或 `EU`。 |
| `timeout_seconds` | integer | 否 | `60` | Datus 操作超时，必须大于 0。 |

## 凭证与命名空间

`credentials_path`、`credentials_info` 和 `credentials_base64` 最多配置一个；全部省略时使用 Google Application Default Credentials。`credentials_info` 必须是 YAML mapping，不能是加引号的 JSON 字符串。Base64 只是传输编码，不是加密。

建议使用 Datus 的 `catalog` 与 `database` 名称，使 profile 与运行时上下文一致。不要配置 `schema`。每个 datasource 绑定一个 project，并可指定默认 dataset。

## 验证连接

```bash
datus --config conf/agent.yml --datasource bigquery_prod
```

运行 `/databases` 和 `/tables`，并确认身份在目标 project/dataset 中拥有 BigQuery job 与元数据权限。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 提示配置了多个凭证来源 | 只保留一个显式凭证字段，或全部省略以使用 ADC。 |
| `credentials_info` 被拒绝 | 使用 YAML mapping，不要传 JSON 字符串。 |
| 找不到 dataset | 检查 `catalog`/project、`database`/dataset、`location` 与 IAM 授权。 |
| 作业计费到错误 project | 设置 `billing_project_id`，并授予所需 service-usage 权限。 |

## 参考

- [BigQuery 认证](https://cloud.google.com/bigquery/docs/authentication)
