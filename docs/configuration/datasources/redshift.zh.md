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
agent:
  services:
    datasources:
      redshift_iam:
        type: redshift
        host: ${REDSHIFT_HOST}
        port: 5439
        username: ${REDSHIFT_USER}
        database: analytics
        schema: public
        iam: true
        ssl: true
        cluster_identifier: analytics-cluster
        region: us-east-1
        # access_key_id: ${AWS_ACCESS_KEY_ID}
        # secret_access_key: ${AWS_SECRET_ACCESS_KEY}
```

运行环境已有 AWS role 或标准 AWS credentials 时，不要再配置静态 access key。

两种认证方式中的 `username` 都是已有的 Redshift **数据库用户**。使用 IAM 认证时，Adapter 将其作为驱动的 `db_user` 参数，用于获取临时数据库凭证。这里应填写数据库用户名，而不是 IAM 用户或 role ARN。`db_user` 是驱动内部参数，不是 `agent.yml` 支持的配置字段。

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 是 | — | Redshift cluster endpoint。 |
| `port` | integer | 否 | `5439` | Redshift 服务端口。 |
| `username` | string | 是 | — | 数据库用户；IAM 认证时也作为 `db_user` 传给驱动。 |
| `password` | string | 是* | — | `iam` 为 `false` 时必填。 |
| `database` | string | 否 | 连接时为 `dev` | 初始 database。 |
| `schema` | string | 否 | — | 初始 schema；省略时 Adapter 使用 `public`。 |
| `ssl` | boolean | 否 | `true` | 是否启用 TLS；IAM 认证必须启用。 |
| `iam` | boolean | 否 | `false` | 是否启用 IAM 认证。 |
| `cluster_identifier` | string | IAM | — | 获取临时凭证所需的 cluster identifier。 |
| `region` | string | IAM | — | AWS region。 |
| `access_key_id` | string | 否 | AWS credential chain | IAM 认证使用的静态 access key。 |
| `secret_access_key` | string | 否 | AWS credential chain | IAM 认证使用的静态 secret key。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## 认证与 TLS

设置 `iam: true` 后 `password` 可以省略，IAM 连接需保持 `ssl: true`。在 AWS 托管环境中，应优先使用附加 role，不要把 key 写入 `agent.yml`。Adapter 只暴露布尔字段 `ssl`，profile 中没有 CA bundle 或 hostname 校验字段。

Adapter 不传递 `sslmode`，主机名检查遵循已安装驱动的默认策略。`ssl: true` 并不保证严格的主机名校验，当前 profile 也无法显式选择 `sslmode: verify-full`。

## 从本机连接

AWS CLI 调用的是 AWS HTTPS API，Adapter 则单独建立数据库 TCP 连接。CLI 调用成功或已配置 HTTP 代理，并不代表数据库网络可达。需要检查 cluster endpoint 的实际数据库端口、VPC 路由、网络 ACL 和安全组。访问公网 endpoint 时，应允许数据库客户端实际使用的公网出口 IP；HTTP 代理的出口地址可能不同。

访问私网 endpoint 时，可使用 VPN，或通过已有的 Systems Manager 托管主机转发连接。托管主机需要能解析并访问数据库 endpoint，安全组需允许该主机的安全组或私网 IP 访问数据库端口。本机还需安装 [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)。

使用 SSM 隧道时，将 `REDSHIFT_HOST` 设为原始 cluster endpoint，`REDSHIFT_REGION` 设为所在 AWS region，`REDSHIFT_SSM_TARGET` 设为托管主机的 instance ID。将 `REDSHIFT_REMOTE_PORT` 设为 cluster 的实际数据库端口（`Endpoint.Port`，通常为 `5439`），然后在独立终端运行：

```bash
aws ssm start-session \
  --region "$REDSHIFT_REGION" \
  --target "$REDSHIFT_SSM_TARGET" \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters "{\"host\":[\"$REDSHIFT_HOST\"],\"portNumber\":[\"$REDSHIFT_REMOTE_PORT\"],\"localPortNumber\":[\"15439\"]}"
```

使用隧道时，datasource profile 的 `host` 仍填写原始 Redshift endpoint。隧道开启期间，通过本机 DNS 或 hosts 文件将该 endpoint 临时映射到 `127.0.0.1`，并将 profile 的 `port` 改为 `15439`。这只改变本机路由，TLS 主机名检查仍遵循上文说明的驱动策略。托管主机会独立解析远端 endpoint，不受本机映射影响。使用完毕后移除映射并关闭 SSM 会话。

## 验证连接

```bash
datus --config conf/agent.yml --datasource redshift_prod
```

运行 `/schemas` 和 `/tables`。IAM profile 还需要确认运行身份有权为指定 cluster 调用 Redshift 临时凭证 API。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 提示必须提供 password | 填写 `password`，或设置 `iam: true`。 |
| IAM 认证失败 | 检查数据库 `username`、`cluster_identifier`、`region`、运行时 AWS credentials/role 与 IAM policy。 |
| IAM 错误提示缺少 `db_user` | 使用包含 [IAM 用户名传递修复](https://github.com/Datus-ai/datus-db-adapters/pull/157)的 Adapter 发布版本。数据库用户仍填写在 `username` 中；在 profile 中添加 `db_user` 会导致校验失败。 |
| CLI 调用成功，但数据库连接超时 | 单独检查数据库 TCP 连通性：endpoint、实际数据库端口、VPC 路由、安全组以及 VPN 或隧道。 |
| Python 3.12 下出现 `SSLContext` / `PROTOCOL_TLS` 弃用告警 | 包括 `redshift_connector` 2.2.0 在内的驱动版本使用了已弃用的 SSL context 初始化方式。保持 TLS 开启，在[上游修复](https://github.com/aws/amazon-redshift-python-driver/pull/288)进入正式发布后使用对应的官方驱动版本。将这些告警作为错误处理时，可能无法建立连接。 |

## 参考

- [Amazon Redshift 连接指南](https://docs.aws.amazon.com/redshift/latest/mgmt/configure-jdbc-connection.html)
- [Session Manager 转发到远端主机](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-sessions-start.html#sessions-remote-port-forwarding)
- [Python SSLContext 弃用说明](https://docs.python.org/3.12/library/ssl.html#ssl.SSLContext)
