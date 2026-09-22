# Amazon Redshift datasource

Redshift adapter 使用原生 `redshift_connector` 驱动，支持密码或 IAM 认证。对象遵循 `database → schema → table` 层级。

## 安装

```bash
pip install datus-redshift
```

## 连接配置

密码认证：

```yaml
agent:
  services:
    datasources:
      redshift_prod:
        type: redshift
        host: ${REDSHIFT_HOST}
        port: 5439
        username: ${REDSHIFT_USER}
        password: ${REDSHIFT_PASSWORD}
        database: analytics
        schema: public
        ssl: true
        timeout_seconds: 30
```

IAM 认证：

```yaml
redshift_iam:
  type: redshift
  host: ${REDSHIFT_HOST}
  port: 5439
  username: ${REDSHIFT_USER}
  database: analytics
  schema: public
  iam: true
  cluster_identifier: analytics-cluster
  region: us-east-1
  # access_key_id: ${AWS_ACCESS_KEY_ID}
  # secret_access_key: ${AWS_SECRET_ACCESS_KEY}
```

运行环境已有 AWS role 或标准 AWS credentials 时，不要再配置静态 access key。

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 是 | — | Redshift cluster endpoint。 |
| `port` | integer | 否 | `5439` | Redshift 服务端口。 |
| `username` | string | 是 | — | 数据库用户。 |
| `password` | string | 是* | — | `iam` 为 `false` 时必填。 |
| `database` | string | 否 | 连接时为 `dev` | 初始 database。 |
| `schema` | string | 否 | — | 初始 schema；省略时驱动使用 `public`。 |
| `ssl` | boolean | 否 | `true` | 是否启用 TLS。 |
| `iam` | boolean | 否 | `false` | 是否启用 IAM 认证。 |
| `cluster_identifier` | string | IAM | — | 获取临时凭证所需的 cluster identifier。 |
| `region` | string | IAM | — | AWS region。 |
| `access_key_id` | string | 否 | AWS credential chain | IAM 认证使用的静态 access key。 |
| `secret_access_key` | string | 否 | AWS credential chain | IAM 认证使用的静态 secret key。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## 认证与 TLS

设置 `iam: true` 后 `password` 可以省略。在 AWS 托管环境中，应优先使用附加 role，不要把 key 写入 `agent.yml`。Adapter 只暴露布尔字段 `ssl`，profile 中没有 CA bundle 或 hostname 校验字段。

## 验证连接

```bash
datus --config conf/agent.yml --datasource redshift_prod
```

运行 `/schemas` 和 `/tables`。IAM profile 还需要确认运行身份有权为指定 cluster 调用 Redshift 临时凭证 API。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 提示必须提供 password | 填写 `password`，或设置 `iam: true`。 |
| IAM 认证失败 | 检查 `cluster_identifier`、`region`、运行时 AWS credentials/role 与 IAM policy。 |
| 连接超时 | 检查 VPC 路由、安全组、endpoint 和 5439 端口。 |

## 参考

- [Amazon Redshift 连接指南](https://docs.aws.amazon.com/redshift/latest/mgmt/configure-jdbc-connection.html)
