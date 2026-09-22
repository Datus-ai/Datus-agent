# ClickZetta datasource

ClickZetta adapter 连接服务 endpoint，并选择 instance、workspace、schema 与 virtual cluster 执行 SQL。

## 安装

```bash
pip install datus-clickzetta
```

## 连接配置

```yaml
agent:
  services:
    datasources:
      clickzetta_lakehouse:
        type: clickzetta
        service: ${CLICKZETTA_SERVICE}
        username: ${CLICKZETTA_USERNAME}
        password: ${CLICKZETTA_PASSWORD}
        instance: ${CLICKZETTA_INSTANCE}
        workspace: ${CLICKZETTA_WORKSPACE}
        schema: ${CLICKZETTA_SCHEMA:-PUBLIC}
        vcluster: ${CLICKZETTA_VCLUSTER:-DEFAULT_AP}
        secure: true
        hints:
          key: value
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `service` | string | 是 | — | ClickZetta 服务 endpoint。 |
| `username` | string | 是 | — | 登录用户，空值会被拒绝。 |
| `password` | string | 是 | — | 登录密码，空值会被拒绝。 |
| `instance` | string | 是 | — | Instance identifier。 |
| `workspace` | string | 是 | — | Workspace 名称。 |
| `schema` | string | 否 | `PUBLIC` | 默认 schema，是 `schema_name` 的 alias。 |
| `vcluster` | string | 否 | `DEFAULT_AP` | Virtual cluster。 |
| `secure` | boolean | 否 | connector 默认值 | 是否启用安全连接。 |
| `hints` | mapping | 否 | — | 其他 ClickZetta connection hints。 |

## Profile 边界

Adapter 模型还定义了名为 `extra` 的通用字段，但 Datus Agent 内部会占用 `extra` 来承载 adapter 顶层扩展字段。不要在 `agent.yml` 中配置 `extra:`；请使用上表中的显式字段，包括 `hints`。

Service、user、password、instance 和 workspace 都不能是空字符串。Schema 与 virtual-cluster 名称由 ClickZetta 服务处理，Datus 不会改写。

## 验证连接

```bash
datus --config conf/agent.yml --datasource clickzetta_lakehouse
```

运行 `/schemas` 和 `/tables`，验证 workspace、schema 与 virtual cluster 选择是否正确。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| `Required field cannot be empty` | 启动 Datus 前导出全部五个必填环境变量。 |
| 看不到 workspace 或 schema | 检查 `instance`、`workspace`、`schema`、`vcluster` 与账号授权。 |
| 安全连接失败 | 确认 endpoint 是否要求 TLS，并相应设置 `secure`。 |

## 参考

- [ClickZetta 文档](https://www.clickzetta.com/)
