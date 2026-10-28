# Google BigQuery datasource

The BigQuery adapter maps a Google Cloud project to Datus `catalog` and a BigQuery dataset to Datus `database`. BigQuery has no schema level below the dataset.

## Install

```bash
pip install datus-bigquery
```

## Connection profile

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

This base profile uses Google Application Default Credentials (ADC) and the configured `catalog` as the billing project. To use a service-account file or a separate billing project, add these fields only when the referenced environment variables are set:

```yaml
credentials_path: ${GOOGLE_APPLICATION_CREDENTIALS}
billing_project_id: ${BIGQUERY_BILLING_PROJECT}
```

Hosted deployments can use a base64-encoded service-account document:

```yaml
credentials_base64: ${BIGQUERY_CREDENTIALS_BASE64}
```

Or provide the service-account JSON as a YAML mapping:

```yaml
credentials_info:
  type: service_account
  project_id: your-project
  private_key_id: ${BIGQUERY_PRIVATE_KEY_ID}
  private_key: ${BIGQUERY_PRIVATE_KEY}
  client_email: datus@your-project.iam.gserviceaccount.com
  token_uri: https://oauth2.googleapis.com/token
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `catalog` | string | yes | — | Google Cloud project; `project` is also accepted. |
| `database` | string | no | — | Default dataset; `dataset` is also accepted. |
| `credentials_path` | string | no | ADC | Service-account JSON file path. |
| `credentials_info` | mapping | no | ADC | Service-account JSON object, represented as YAML mapping. |
| `credentials_base64` | secret string | no | ADC | Base64-encoded service-account JSON. |
| `billing_project_id` | string | no | configured project | Billing/quota project. |
| `location` | string | no | BigQuery/client default | Job location such as `US` or `EU`. |
| `timeout_seconds` | integer | no | `60` | Datus operation timeout; must be positive. |

## Credentials and namespaces

Configure at most one of `credentials_path`, `credentials_info`, and `credentials_base64`. When all are omitted, Google Application Default Credentials are used. `credentials_info` must be a YAML mapping—not a quoted JSON string. Base64 is transport encoding, not encryption.

Prefer the Datus names `catalog` and `database` so the profile matches runtime context. Do not set `schema`. Each datasource binds to one project and can use a default dataset.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource bigquery_prod
```

Run `/databases` and `/tables`. Also confirm the identity has BigQuery job and metadata permissions in the configured project and dataset.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| More than one credential source error | Keep exactly one explicit credential field, or omit all three for ADC. |
| `credentials_info` is rejected | Use a YAML mapping, not a JSON string. |
| Dataset is not found | Check `catalog`/project, `database`/dataset, `location`, and IAM grants. |
| Jobs bill the wrong project | Set `billing_project_id` and grant the required service-usage permission. |

## Reference

- [BigQuery authentication](https://cloud.google.com/bigquery/docs/authentication)
