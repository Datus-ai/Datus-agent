# 数据库适配器

Datus Agent 通过基于插件的适配器系统支持连接各种数据库。本文档介绍可用的适配器、安装方法以及数据库连接配置。

## 概览

Datus 使用模块化适配器架构，允许连接不同的数据库：

- **内置适配器**：SQLite 和 DuckDB 包含在核心包中
- **插件适配器**：其他数据库（MySQL、Snowflake、StarRocks 等）可作为独立包安装

这种设计保持核心包轻量化，同时允许按需添加特定数据库支持。

## 支持的数据库

| 数据库 | 包名 | 安装方式 | 状态 |
|--------|------|----------|------|
| SQLite | 内置 | 已包含 | 可用 |
| DuckDB | 内置 | 已包含 | 可用 |
| MySQL | datus-mysql | `pip install datus-mysql` | 可用 |
| StarRocks | datus-starrocks | `pip install datus-starrocks` | 可用 |
| Snowflake | datus-snowflake | `pip install datus-snowflake` | 可用 |
| ClickZetta | datus-clickzetta | `pip install datus-clickzetta` | 可用 |

## 安装

### 内置数据库

SQLite 和 DuckDB 已包含在 Datus Agent 中，无需额外安装。

### 插件适配器

为您的数据库安装对应的适配器包：

```bash
# MySQL
pip install datus-mysql

# Snowflake
pip install datus-snowflake

# StarRocks
pip install datus-starrocks

# ClickZetta
pip install datus-clickzetta
```

安装后，Datus Agent 会自动检测并加载适配器。

## 配置

在 `config.yaml` 文件中配置数据库连接：

### SQLite

```yaml
namespace:
  mydata:
    type: sqlite
    uri: sqlite:///path/to/database.db
```

### DuckDB

```yaml
namespace:
  analytics:
    type: duckdb
    database: /path/to/database.duckdb
```

### MySQL

```yaml
namespace:
  production:
    type: mysql
    host: localhost
    port: 3306
    username: your_username
    password: your_password
    database: your_database
```

### Snowflake

```yaml
namespace:
  warehouse:
    type: snowflake
    account: your_account
    username: your_username
    password: your_password
    warehouse: your_warehouse
    database: your_database
    schema: your_schema
    role: your_role  # 可选
```

### StarRocks

```yaml
namespace:
  analytics:
    type: starrocks
    host: localhost
    port: 9030
    username: root
    password: your_password
    database: your_database
```

### ClickZetta

```yaml
namespace:
  lakehouse:
    type: clickzetta
    service: CLICKZETTA_SERVICE
    username: CLICKZETTA_USERNAME
    password: CLICKZETTA_PASSWORD
    instance: CLICKZETTA_INSTANCE
    workspace: CLICKZETTA_WORKSPACE
    schema: CLICKZETTA_SCHEMA
    vcluster: CLICKZETTA_VCLUSTER
```

## 多数据库连接

可以在同一命名空间下配置多个数据库：

```yaml
namespace:
  project:
    source_db:
      type: mysql
      host: source-server
      username: reader
      password: password
      database: source
    target_db:
      type: snowflake
      account: your_account
      username: writer
      password: password
      warehouse: compute_wh
      database: target
```

## 适配器功能

### 通用功能

所有适配器支持：

- SQL 查询执行（SELECT、INSERT、UPDATE、DELETE）
- DDL 操作（CREATE、ALTER、DROP）
- 元数据获取（表、视图、schema）
- 样本数据获取
- 连接池和超时管理

### 适配器特定功能

#### MySQL
- INFORMATION_SCHEMA 查询
- SHOW CREATE TABLE/VIEW 支持
- 完整 CRUD 操作

#### Snowflake
- 多数据库和 schema 支持
- 表、视图和物化视图
- Arrow 格式高效数据传输
- 原生 SDK 集成

#### StarRocks
- 多 Catalog 支持
- 物化视图支持
- MySQL 协议兼容

#### ClickZetta
- Workspace 和 schema 管理
- Volume/Stage 文件操作
- 原生 SDK 集成

## 故障排除

### 适配器未找到

如果看到 `Connector 'mysql' not found` 错误，请确保已安装对应的适配器包：

```bash
pip install datus-mysql
```

### 连接问题

检查以下内容：

1. **网络连接**：确保能够访问数据库服务器
2. **凭证**：验证用户名和密码是否正确
3. **端口**：确认指定的端口正确
4. **数据库名**：确保数据库存在

### 驱动依赖

部分适配器需要额外的系统依赖：

- **MySQL**：需要 `pymysql`（自动安装）
- **Snowflake**：需要 `snowflake-connector-python`（自动安装）

## 架构

```text
datus-agent (核心)
├── 内置适配器
│   ├── SQLite Connector
│   └── DuckDB Connector
│
└── 插件系统 (Entry Points)
    ├── datus-sqlalchemy (基础层)
    │   ├── datus-mysql
    │   └── datus-starrocks
    │
    └── 原生 SDK 适配器
        ├── datus-snowflake
        └── datus-clickzetta
```

适配器系统使用 Python 的 entry points 机制实现自动发现。当您安装适配器包时，它会自动注册到 Datus Agent 并可供使用。

## 下一步

- [快速开始](../getting_started/Quickstart.md) - 开始使用 Datus Agent
- [配置参考](../configuration/introduction.md) - 详细配置选项
