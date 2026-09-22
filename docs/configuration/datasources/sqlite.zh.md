# SQLite datasource

SQLite 内置在 Datus 中，直接连接本地数据库文件，不需要服务端、账号、catalog 或 schema。

## 连接配置

```yaml
agent:
  services:
    datasources:
      local_sqlite:
        type: sqlite
        uri: sqlite:////absolute/path/to/analytics.sqlite
        read_only: true
        default: true
```

相对路径使用三个斜杠：

```yaml
uri: sqlite:///data/analytics.sqlite
```

如果要在一个 datasource 中暴露多个文件，用 glob 替换 `uri`：

```yaml
benchmark:
  type: sqlite
  path_pattern: benchmark/databases/**/*.sqlite
  database: california_schools  # 可选：初始文件名（不含扩展名）
```

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `uri` | string | 是* | — | SQLite URI 或文件路径；使用 `path_pattern` 时不填。 |
| `path_pattern` | string | 是* | — | 多文件 glob；使用 `uri` 时不填。 |
| `database` | string | 否 | 第一个匹配项 | 使用 `path_pattern` 时，按文件名（不含扩展名）选择初始 database。 |
| `read_only` | boolean | 否 | `false` | 通过 SQLite 只读 URI 模式打开文件。 |

## 路径与命名空间

- `sqlite:////tmp/orders.sqlite` 对应绝对路径 `/tmp/orders.sqlite`。
- `sqlite:///data/orders.sqlite` 相对于进程工作目录解析。
- 单文件 datasource 以文件名（不含扩展名）作为 database 名，表不再有 schema 层。
- `path_pattern` datasource 会把每个匹配文件列为独立 database。

!!! warning
    `read_only: true` 在连接层保护 SQLite 文件。部署级 `agent.sql_read_only` 是另一层 SQL 执行策略，可以同时启用。

## 验证连接

```bash
datus --config conf/agent.yml --datasource local_sqlite
```

进入 CLI 后运行 `/tables`，或在 SQL 模式执行 `SELECT 1`。通过 `/datasource` 编辑时也会在保存前测试文件。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 启动时跳过 datasource | `path_pattern` 没有匹配文件。检查工作目录和 glob。 |
| `unable to open database file` | 路径不存在，或 Datus 进程没有目录/文件权限。 |
| 写语句失败 | 配置了 `read_only: true`、文件不可写，或启用了 `agent.sql_read_only`。 |

## 参考

- [SQLite 文档](https://www.sqlite.org/docs.html)
