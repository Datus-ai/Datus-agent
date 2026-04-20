# 数据库配置（Databases）

配置 `agent.services.databases` 下的数据库连接。

## 概览

Datus Agent 在 `agent.yml` 中通过 `agent.services` 统一管理运行时集成。本页聚焦 `services.databases` 下的数据库连接；语义层、BI 工具和调度器分别在同级页面单独说明。

主要特性：

- **统一连接入口**：支持 Snowflake、StarRocks、SQLite、DuckDB 等数据库，以及语义层、BI、调度器服务
- **凭证安全**：支持 `${ENV_VAR}` 环境变量展开
- **默认数据库**：可通过 `default: true` 标记默认数据库
- **插件适配器**：可按需安装数据库适配器
- **动态发现**：支持通过 `path_pattern` 批量发现多个本地数据库文件

> **迁移说明**：旧的 `namespace:` 配置格式会在运行时自动迁移到 `services.databases`。你也可以执行 `python -m datus.configuration.config_migrator --config conf/agent.yml` 进行离线迁移。

## 配置结构

数据库统一配置在 `agent.services.databases` 下，每个条目都是一个独立数据库连接：

```yaml
agent:
  services:
    databases:
      my_snowflake:
        type: snowflake
        account: ${SNOWFLAKE_ACCOUNT}
        username: ${SNOWFLAKE_USER}
        password: ${SNOWFLAKE_PASSWORD}
        default: true

      my_duckdb:
        type: duckdb
        uri: ./data/analytics.duckdb

    semantic_layer:
      metricflow: {}

    bi_tools:
      superset:
        type: superset
        api_url: http://localhost:8088
        username: ${SUPERSET_USER}
        password: ${SUPERSET_PASSWORD}

    schedulers:
      airflow_prod:
        type: airflow
        api_base_url: ${AIRFLOW_URL}
        username: ${AIRFLOW_USER}
        password: ${AIRFLOW_PASSWORD}
        dags_folder: ${AIRFLOW_DAGS_DIR}
```

## 服务分组

| 配置段 | 用途 | 选择方式 |
|--------|------|----------|
| `services.databases` | SQL 与知识库操作使用的数据库连接 | `--database` / 当前数据库 / 默认数据库 |
| `services.semantic_layer` | 语义适配器配置，例如 MetricFlow | `semantic_adapter` |
| `services.bi_tools` | BI 平台凭据与数据集物化配置 | `bi_platform` |
| `services.schedulers` | 调度器服务实例，例如 Airflow | `scheduler_service` |

## 支持的数据库类型

### Snowflake

```yaml
my_snowflake:
  type: snowflake
  account: ${SNOWFLAKE_ACCOUNT}
  username: ${SNOWFLAKE_USER}
  password: ${SNOWFLAKE_PASSWORD}
  database: ${SNOWFLAKE_DATABASE}    # 可选
  schema: ${SNOWFLAKE_SCHEMA}        # 可选
  warehouse: ${SNOWFLAKE_WAREHOUSE}  # 可选
  default: true                      # 可选：设为默认数据库
```

### StarRocks

```yaml
my_starrocks:
  type: starrocks
  host: ${STARROCKS_HOST}
  port: ${STARROCKS_PORT}
  username: ${STARROCKS_USER}
  password: ${STARROCKS_PASSWORD}
  database: ${STARROCKS_DATABASE}
  catalog: ${STARROCKS_CATALOG}      # 可选
```

### SQLite

```yaml
my_sqlite:
  type: sqlite
  uri: sqlite:////Users/xxx/data/orders.db
```

### DuckDB

```yaml
my_duckdb:
  type: duckdb
  uri: duckdb:////Users/xxx/data/analytics.duckdb
```

### MySQL

```yaml
my_mysql:
  type: mysql
  host: localhost
  port: 3306
  username: ${MYSQL_USER}
  password: ${MYSQL_PASSWORD}
  database: analytics
```

### PostgreSQL

```yaml
my_postgresql:
  type: postgresql
  host: localhost
  port: 5432
  username: ${POSTGRES_USER}
  password: ${POSTGRES_PASSWORD}
  database: analytics
```

### 路径模式（批量发现多个文件）

使用 glob 模式自动发现数据库文件：

```yaml
bird_benchmark:
  type: sqlite
  path_pattern: benchmark/bird/dev_20240627/dev_databases/**/*.sqlite
```

常见模式包括：`*.sqlite`、`**/*.sqlite`、`data/2024/*.db`

## 配置参数

### 通用参数

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `type` | 是 | 数据库类型，例如 `sqlite`、`duckdb`、`snowflake`、`starrocks`、`mysql`、`postgresql` |
| `default` | 否 | 设为 `true` 后作为默认数据库 |
| `uri` | 文件型数据库必填 | SQLite / DuckDB 的连接 URI |
| `host` | 服务型数据库必填 | 数据库主机地址 |
| `port` | 服务型数据库必填 | 数据库端口 |
| `username` | 服务型数据库必填 | 用户名 |
| `password` | 服务型数据库必填 | 密码 |
| `database` | 否 | 数据库名 |

### 数据库特定参数

- **Snowflake**：`account`、`warehouse`、`role`、`schema`
- **StarRocks**：`catalog`
- **SQLite/DuckDB**：`path_pattern` 用于批量发现数据库文件
- **MySQL/PostgreSQL**：`host`、`port`、`username`、`password`、`database`

## 管理数据库

### 交互式配置

使用 `datus-agent configure` 交互式添加、删除或管理数据库：

```bash
datus-agent configure
```

它会先展示当前模型与数据库，然后提供菜单：

```text
Current Databases:
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Name         ┃ Type      ┃ Connection              ┃ Default ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ demo         │ duckdb    │ ./demo.duckdb           │ *       │
│ prod_sf      │ snowflake │ account=my_account      │         │
└──────────────┴───────────┴─────────────────────────┴─────────┘

What would you like to do?
  → [add_database] Add a database
    [delete_database] Delete a database
    [done] Done
```

当你选择未安装的数据库类型（例如 snowflake、mysql）时，适配器插件会自动安装。

### CLI 命令

```bash
# 列出所有数据库
datus-agent service list

# 交互式添加数据库
datus-agent service add

# 交互式删除数据库
datus-agent service delete
```

### 指定自定义配置文件

```bash
datus-agent service list --config /path/to/agent.yml
datus-agent configure --config /path/to/agent.yml
```

## 默认数据库选择

运行 CLI 命令时，可以显式指定要使用的数据库：

```bash
datus-cli --database my_duckdb
datus-agent run --database my_snowflake --task "..." --task_db_name ANALYTICS
```

如果没有指定 `--database`：

1. 若某个数据库设置了 `default: true`，则自动使用它
2. 若只配置了一个数据库，则自动使用该数据库
3. 若配置了多个数据库且都未设置默认值，则展示可选列表

## 安全建议

### 凭证管理

```yaml
# 推荐：使用环境变量
username: ${DB_USERNAME}
password: ${DB_PASSWORD}

# 不推荐：直接硬编码凭证
username: "actual_username"
password: "actual_password"
```

## 相关文档

- [数据库适配器](../adapters/db_adapters.md) - 安装 MySQL、Snowflake、StarRocks 等插件适配器
- [语义层配置](semantic_layer.md) - 配置语义适配器
- [BI 工具配置](bi_tools.md) - 配置 Superset 或 Grafana
- [调度器配置](schedulers.md) - 配置 Airflow 等调度器
- [CLI 命令](../cli-commands.md) - 查看 `configure`、`init`、`service` 等完整命令说明
