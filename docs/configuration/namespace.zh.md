# 命名空间（Namespace）

为不同数据源配置数据库命名空间与连接。

## 概览

命名空间为多数据库接入提供统一抽象：
- **通用连接**：支持云数仓（Snowflake、StarRocks）、本地数据库（SQLite、DuckDB）与基准数据集
- **环境隔离**：开发/预发/生产分离
- **凭证安全**：环境变量管理 + 安全协议
- **动态发现**：基于路径模式自动纳入多库文件
- **可扩展组织**：按层级组织命名空间

## 结构
```yaml
namespace:
  service:
    type: cloud_provider
    endpoint: ${SERVICE_ENDPOINT}
    access_key: ${ACCESS_KEY}
    secret_key: ${SECRET_KEY}
    region: ${SERVICE_REGION}

  local_resource:
    type: local_service
    resources:
      - name: primary
        uri: protocol://path/to/resource
      - name: secondary
        uri: protocol://path/to/backup
```

## 支持的数据库类型

### Snowflake
```yaml
snowflake:
  type: snowflake
  account: ${SNOWFLAKE_ACCOUNT}
  username: ${SNOWFLAKE_USER}
  password: ${SNOWFLAKE_PASSWORD}
  database: ${SNOWFLAKE_DATABASE}    # 可选
  schema: ${SNOWFLAKE_SCHEMA}        # 可选
  warehouse: ${SNOWFLAKE_WAREHOUSE}  # 可选
```

### StarRocks
```yaml
starrocks:
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
# 单库
local_sqlite:
  type: sqlite
  name: ssb
  uri: sqlite:////Users/xxx/benchmark/SSB.db

# 多库
local_sqlite_multi:
  type: sqlite
  dbs:
    - name: ssb
      uri: sqlite:////Users/xxx/benchmark/SSB.db
    - name: northwind
      uri: sqlite:////Users/xxx/data/northwind.db
```

### DuckDB
```yaml
# 单库
local_duckdb:
  type: duckdb
  name: analytics
  uri: duckdb:////absolute/path/to/analytics.db

# 多库
local_duckdb_multi:
  type: duckdb
  dbs:
    - name: ssb
      uri: duckdb:////absolute/path/to/ssb.db
    - name: tpch
      uri: duckdb:///relative/path/to/tpch.duckdb
```

## 配置参数

### 通用
- `type`：数据库类型（必填）
- `name`：数据库标识（SQLite/DuckDB 必填）
- `uri`：本地库连接 URI
- `host`、`port`、`username`、`password`、`database`

### 特定
- Snowflake：`account`、`warehouse`、`role`、`schema`
- StarRocks：`catalog`、`ssl`
- SQLite/DuckDB：`path_pattern`、`dbs`

## 完整示例
```yaml
namespace:
  production_snowflake:
    type: snowflake
    account: ${SNOWFLAKE_ACCOUNT}
    username: ${SNOWFLAKE_USER}
    password: ${SNOWFLAKE_PASSWORD}
    database: ANALYTICS
    schema: PUBLIC
    warehouse: COMPUTE_WH

  dev_starrocks:
    type: starrocks
    host: ${STARROCKS_HOST}
    port: ${STARROCKS_PORT}
    username: ${STARROCKS_USER}
    password: ${STARROCKS_PASSWORD}
    database: dev_analytics

  test_sqlite:
    type: sqlite
    dbs:
      - name: orders
        uri: sqlite:////Users/data/orders.db
      - name: customers
        uri: sqlite:////Users/data/customers.db
      - name: products
        uri: sqlite:////Users/data/products.db

  analytics_duckdb:
    type: duckdb
    dbs:
      - name: sales
        uri: duckdb:////opt/data/sales.db
      - name: marketing
        uri: duckdb:///data/marketing.duckdb

  bird_benchmark:
    type: sqlite
    path_pattern: benchmark/bird/dev_20240627/dev_databases/**/*.sqlite
```

## 多库配置

### SQLite 多库
```yaml
multi_sqlite:
  type: sqlite
  dbs:
    - name: sales_2023
      uri: sqlite:////data/sales_2023.db
    - name: sales_2024
      uri: sqlite:////data/sales_2024.db
    - name: customer_master
      uri: sqlite:////data/customers.db
```

### 路径模式
```yaml
benchmark_dbs:
  type: sqlite
  path_pattern: benchmarks/**/*.sqlite
```
**常用模式**：`*.sqlite`、`**/*.sqlite`、`data/2024/*.db`、`benchmark/bird/**/*.sqlite`

## URI 格式
```text
sqlite:////absolute/path/to/database.db
sqlite:///relative/path/to/database.db

duckdb:////absolute/path/to/database.db
duckdb:///relative/path/to/database.db
```

## 命名空间管理命令

Datus Agent 提供交互式 CLI 工具来管理命名空间配置，无需手动编辑 YAML 文件。

### 命令

#### 列出命名空间

查看所有已配置的命名空间及其连接详情：

```bash
datus-agent namespace list
```

输出示例：
```
Configured namespaces:

Namespace: production_snowflake
  Database: ANALYTICS
    Type: snowflake
    Account: my_account
    Warehouse: COMPUTE_WH
    Database: ANALYTICS
    Schema: PUBLIC
    Username: admin

Namespace: local_duckdb
  Database: analytics
    Type: duckdb
    URI: duckdb:////data/analytics.db
```

#### 添加命名空间

交互式添加新的命名空间配置：

```bash
datus-agent namespace add
```

命令会提示输入：

1. **命名空间名称**：命名空间的唯一标识符
2. **数据库类型**：从 sqlite、duckdb、snowflake、mysql、starrocks 中选择
3. **连接参数**：根据数据库类型不同而变化

**文件型数据库（SQLite、DuckDB）：**
- 连接字符串（文件路径）

**主机型数据库（MySQL、StarRocks）：**
- 主机地址
- 端口
- 用户名
- 密码
- 数据库名

**Snowflake：**
- 用户名
- 账户
- 仓库
- 密码
- 数据库（可选）
- Schema（可选）

输入配置后，工具会：
- 测试数据库连接
- 连接成功后保存配置到 `conf/agent.yml`

示例会话：
```text
Add New Namespace
- Namespace name: my_analytics
- Database type [sqlite/duckdb/snowflake/mysql/starrocks] (duckdb): snowflake
- Username: admin
- Account: my_account
- Warehouse: COMPUTE_WH
- Password: ********
- Database (optional): ANALYTICS
- Schema (optional): PUBLIC
→ Testing database connectivity...
✔ Database connection test successful

Configuration saved to conf/agent.yml
✔ Namespace 'my_analytics' added successfully
```

#### 删除命名空间

交互式删除现有命名空间：

```bash
datus-agent namespace delete
```

命令会：
1. 显示可用的命名空间
2. 提示输入要删除的命名空间名称
3. 删除前要求确认

示例会话：
```
Delete Namespace
Available namespaces:
  - production_snowflake
  - local_duckdb
  - test_sqlite
- Namespace name to delete: test_sqlite
Are you sure you want to delete namespace 'test_sqlite'? This action cannot be undone. [y/N]: y
Configuration saved to conf/agent.yml
✔ Namespace 'test_sqlite' deleted successfully
```

### 使用自定义配置文件

指定自定义配置文件：

```bash
datus-agent namespace list --config /path/to/agent.yml
datus-agent namespace add --config /path/to/agent.yml
datus-agent namespace delete --config /path/to/agent.yml
```

## 安全
```yaml
# 推荐
username: ${DB_USERNAME}
password: ${DB_PASSWORD}

# 避免
username: "actual_username"
password: "actual_password"
```

## 另请参阅

- [数据库适配器](../adapters/db_adapters.md) - 安装 MySQL、Snowflake、StarRocks 等插件适配器
