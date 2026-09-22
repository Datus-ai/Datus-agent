# MaxCompute datasource

MaxCompute adapter 基于 PyODPS，同时支持传统 `project.table` 项目和启用 schema 的 `project.schema.table` 项目。

## 安装

```bash
pip install datus-maxcompute
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      maxcompute_prod:
        type: maxcompute
        database: ${MAXCOMPUTE_PROJECT}
        endpoint: ${MAXCOMPUTE_ENDPOINT}
        access_key_id: ${MAXCOMPUTE_ACCESS_KEY_ID}
        access_key_secret: ${MAXCOMPUTE_ACCESS_KEY_SECRET}
        namespace_mode: auto
        # schema: default
        # quota_name: ${MAXCOMPUTE_QUOTA_NAME}
        # tunnel_endpoint: ${MAXCOMPUTE_TUNNEL_ENDPOINT}
        timeout_seconds: 30
        query_timeout_seconds: 600
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `database` | string | 是 | — | MaxCompute project；也接受 `project`。 |
| `endpoint` | string | 是 | — | MaxCompute 服务 endpoint。 |
| `access_key_id` | secret string | 是 | — | 阿里云 AccessKey ID。 |
| `access_key_secret` | secret string | 是 | — | 阿里云 AccessKey secret。 |
| `schema` | string | 否 | 三层模式下为 `default` | Schema-enabled 项目的默认 schema。 |
| `namespace_mode` | string | 否 | `auto` | `auto`、`two_level` 或 `three_level`。 |
| `quota_name` | string | 否 | — | MaxCompute quota 名称。 |
| `tunnel_endpoint` | string | 否 | 沿用服务 endpoint 行为 | 单独的 Instance Tunnel endpoint。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时，必须大于 0。 |
| `query_timeout_seconds` | integer | 否 | `600` | SQL job 超时，必须大于 0。 |
| `default_hints` | mapping | 否 | `{}` | 默认 MaxCompute SQL hints。 |

## 命名空间模式

`database` 表示 MaxCompute project，adapter 也接受原生命名 `project`。除非当前身份无法探测 schema 支持，否则保持 `namespace_mode: auto`。

- `two_level`：对象为 `project.table`，不要设置 `schema`。
- `three_level`：对象为 `project.schema.table`；省略 `schema` 时使用 `default`。

每个 datasource 只在已配置 project 内生成 SQL，不生成跨 project SQL。仅当 Instance Tunnel 与 SQL 服务使用不同 endpoint 时设置 `tunnel_endpoint`。

## 验证连接

```bash
datus --config conf/agent.yml --datasource maxcompute_prod
```

运行 `/databases` 和 `/tables`；三层模式下还要确认当前身份可以发现 schema。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 无权探测 namespace | 将 `namespace_mode` 显式设为 `two_level` 或 `three_level`。 |
| 表解析到了错误 schema | 设置 `schema`；两层项目则应保持未设置。 |
| 查询成功但下载失败 | 配置正确的 `tunnel_endpoint` 和网络路由。 |
| Job 超时 | 增大 `query_timeout_seconds`；`timeout_seconds` 只控制连接。 |

## 参考

- [MaxCompute endpoints](https://www.alibabacloud.com/help/en/maxcompute/user-guide/endpoints)
