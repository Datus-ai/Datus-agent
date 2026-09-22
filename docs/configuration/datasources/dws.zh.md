# 华为云 GaussDB(DWS) datasource

DWS adapter 通过 PostgreSQL 兼容协议连接华为云 GaussDB(DWS)，并处理 DWS 兼容模式、原生表 DDL 与证书限制。

## 安装

```bash
pip install datus-dws
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      dws_analytics:
        type: dws
        host: ${DWS_HOST}
        port: 8000
        username: ${DWS_USER}
        password: ${DWS_PASSWORD}
        database: gaussdb
        schema: public
        sslmode: verify-ca
        sslrootcert: /etc/datus/certs/dws-cacert.pem
        timeout_seconds: 30
```

可以把控制台的 `hostname:port` 直接写入 `host`；此时省略 `port`。

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 是 | — | Coordinator endpoint，可内嵌端口。 |
| `port` | integer | 否 | `8000` | Coordinator 端口，范围 1–65535；必须与内嵌端口一致。 |
| `username` | string | 是 | — | DWS 数据库用户。 |
| `password` | string | 否 | 空字符串 | 登录密码。 |
| `database` | string | 是 | — | DWS database，集群默认通常是 `gaussdb`。 |
| `schema` | string | 否 | `public` | 初始 schema。 |
| `sslmode` | string | 否 | `prefer` | `disable`、`allow`、`prefer`、`require`、`verify-ca` 或 `verify-full`。 |
| `sslrootcert` | string | 否* | — | CA 路径或 inline PEM。除非已配置 libpq 标准 CA 目录，`verify-ca` 应提供该字段。 |
| `timeout_seconds` | integer | 否 | `30` | 连接与连接池超时，必须大于 0。 |

## TLS

使用 `verify-ca` 时，应选择控制台 `dws_ssl_cert` 压缩包中的 `v2/sslcert/cacert.pem`。v1 CA 与服务端证书签发者不匹配。`sslrootcert` 接受文件路径或 inline PEM。

!!! warning
    `verify-full` 无法用于默认 DWS 服务端证书：证书使用 `CN=server` 且没有 `subjectAltName`，不可能匹配真实集群 endpoint。`verify-ca` 只校验签发 CA，不校验集群 hostname；应通过可信 VPC 路径或经过核实的固定 endpoint 访问。

`require` 会加密但不认证服务端；`prefer` 会在集群提供/强制 TLS 时升级连接，但不验证服务端身份；集群强制 SSL 时，`disable` 会失败。

## 兼容模式

DWS database 可使用 `ORA`、`TD` 或 `MySQL` 兼容模式。新集群通常默认 `ORA`：`7/2` 得到 `3.5`，空字符串变为 `NULL`，字符串拼接会吸收 `NULL`，`DATE` 映射到 `timestamp(0)`。复用 PostgreSQL 假设前应先确认模式。

## 验证连接

```bash
datus --config conf/agent.yml --datasource dws_analytics
```

运行 `/schemas` 和 `/tables`，再确认 database 兼容模式；如果迁移依赖分布、分区、tablespace 或 resource-group clause，还要检查一张表的 DDL。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 内嵌端口与显式 port 冲突 | 移除 `port`，或让它与 `host` 内端口一致。 |
| `verify-ca` 报签发者错误 | 使用 DWS 证书包中的 v2 CA，不要使用 v1。 |
| `verify-full` 报 hostname 不匹配 | 默认 DWS 证书下这是预期行为；改用 `verify-ca` 和可信网络路径。 |
| SQL 语义与 PostgreSQL 不同 | 检查 database 是 `ORA`、`TD` 还是 `MySQL` 模式。 |

## 参考

- [华为云 DWS SSL 连接设置](https://support.huaweicloud.com/intl/en-us/mgtg-dws/dws_01_0038.html)
