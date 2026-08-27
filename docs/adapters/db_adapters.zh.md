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
| PostgreSQL | datus-postgresql | `pip install datus-postgresql` | 可用 |
| StarRocks | datus-starrocks | `pip install datus-starrocks` | 可用 |
| Snowflake | datus-snowflake | `pip install datus-snowflake` | 可用 |
| ClickZetta | datus-clickzetta | `pip install datus-clickzetta` | 可用 |
| Hive | datus-hive | `pip install datus-hive` | 可用 |
| Spark | datus-spark | `pip install datus-spark` | 可用 |
| ClickHouse | datus-clickhouse | `pip install datus-clickhouse` | 可用 |
| Trino | datus-trino | `pip install datus-trino` | 可用 |
| Apache Doris | datus-doris | `pip install datus-doris` | 可用 |
| TiDB | datus-tidb | `pip install datus-tidb` | 可用 |
| Hologres | datus-hologres | `pip install datus-hologres` | 可用 |
| Oracle | datus-oracle | `pip install datus-oracle` | 可用 |
| GaussDB / openGauss | datus-gaussdb | `pip install datus-gaussdb` | 可用（Linux 和 macOS） |

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

# Hive
pip install datus-hive

# Spark
pip install datus-spark

# ClickHouse
pip install datus-clickhouse

# Trino
pip install datus-trino

# Apache Doris
pip install datus-doris

# Hologres
pip install datus-hologres

# Oracle
pip install datus-oracle

# GaussDB / openGauss
pip install datus-gaussdb
```

安装后，Datus Agent 会自动检测并加载适配器。

## 配置

在 `agent.yml` 的 `agent.services.datasources` 下配置数据源连接：

```yaml
agent:
  services:
    datasources:
      mydata:
        type: sqlite
        uri: sqlite:///path/to/database.db
```

`services.datasources` 下的每个条目都表示一个逻辑数据库连接。

### SQLite

```yaml
mydata:
  type: sqlite
  uri: sqlite:///path/to/database.db
```

### DuckDB

```yaml
analytics:
  type: duckdb
  uri: duckdb:///path/to/database.duckdb
```

### MySQL

```yaml
production:
  type: mysql
  host: localhost
  port: 3306
  username: your_username
  password: your_password
  database: your_database
```

### PostgreSQL

```yaml
production_pg:
  type: postgresql
  host: localhost
  port: 5432
  username: your_username
  password: your_password
  database: your_database
  schema: public  # 可选，默认为 public
  sslmode: prefer  # 可选，默认为 prefer
```

### Snowflake

```yaml
warehouse:
  type: snowflake
  account: your_account
  username: your_username
  password: your_password  # 可配置 private_key，或在没有 private_key 时 password 和 private_key_file 二选一
  # private_key: |
  #   -----BEGIN PRIVATE KEY-----
  #   ...
  #   -----END PRIVATE KEY-----
  # private_key_file: /path/to/rsa_key.p8
  # private_key_file_pwd: optional_key_passphrase
  warehouse: your_warehouse
  database: your_database
  schema: your_schema
  role: your_role  # 可选
```

Snowflake 支持密码认证和 key-pair 认证。可以配置 `private_key`，或在没有 `private_key` 时配置 `password` 和 `private_key_file` 其中一个。`private_key` 会优先于 `private_key_file` 和 `password`；私钥加密时再配置 `private_key_file_pwd`。Snowflake 使用 `database` 和 `schema`，不要为 Snowflake 配置 `catalog`。

### StarRocks

```yaml
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

### Hive

```yaml
hive_data:
  type: hive
  host: 127.0.0.1
  port: 10000
  username: hive
  database: default
  auth: NONE  # 可选：NONE、LDAP、CUSTOM、KERBEROS
  configuration:  # 可选 Hive session 配置
    hive.execution.engine: spark
```

### Spark

```yaml
spark_data:
  type: spark
  host: localhost
  port: 10000
  username: spark
  database: default
  auth_mechanism: NONE  # 可选：NONE、PLAIN、KERBEROS
```

### ClickHouse

```yaml
analytics:
  type: clickhouse
  host: localhost
  port: 8123
  username: default
  password: your_password
  database: your_database
```

### Trino

```yaml
trino_data:
  type: trino
  host: localhost
  port: 8080
  username: trino
  catalog: hive
  schema: default
  http_scheme: http  # 可选：http 或 https
```

### Apache Doris

```yaml
doris_data:
  type: doris
  host: localhost
  port: 9030
  username: root
  password: your_password
  catalog: internal      # 可选，默认为 internal
  database: your_database
  charset: utf8mb4       # 可选，默认为 utf8mb4
  autocommit: true       # 可选，默认为 true
  timeout_seconds: 30    # 可选，默认为 30
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `host` | `127.0.0.1` | FE 节点地址 |
| `port` | `9030` | FE 的 MySQL 协议查询端口，不是 FE HTTP 端口（8030） |
| `username` | 必填 | Doris 用户名 |
| `password` | 空 | Doris 密码 |
| `catalog` | `internal` | 连接建立时所在的 catalog |
| `database` | 无 | 连接建立时所在的数据库 |
| `charset` | `utf8mb4` | 连接字符集 |
| `autocommit` | `true` | 自动提交模式 |
| `timeout_seconds` | `30` | 连接超时时间（秒） |

Doris 使用 MySQL 协议，`port` 为 FE 查询端口（默认 9030）。对象以 `catalog.database.table` 三段式命名：
Doris 在数据库和表之间没有 schema 这一层，因此不要配置 `schema`，多出来的一层由 `catalog` 承担。

`internal` 是存放 Doris 自管理表的内置 catalog。如果希望连接建立时就位于某个外部 catalog（例如 Hive
Metastore catalog），直接在 `catalog` 中指定：

```yaml
doris_hive:
  type: doris
  host: localhost
  port: 9030
  username: root
  password: your_password
  catalog: hive_catalog
  database: warehouse
```

该 catalog 必须已经存在于 Doris 中——Datus 只负责选择 catalog，不会创建 catalog。请先在 Doris 侧创建
（`CREATE CATALOG hive_catalog PROPERTIES (...)`，并按 catalog 类型填入 metastore 地址和存储凭证），
确认它出现在 `SHOW CATALOGS` 的结果中。

在会话内，`SWITCH <catalog>` 用于切换 catalog，`USE [<catalog>.]<database>` 用于切换数据库。切换 catalog
会清空当前数据库上下文——同名数据库通常并不存在于新的 catalog 中，切换后需要再执行一次 `USE`。

### TiDB

```yaml
tidb_data:
  type: tidb
  host: 127.0.0.1
  port: 4000
  username: root
  password: your_password
  database: your_database
  charset: utf8mb4       # 可选，默认 utf8mb4
  autocommit: true       # 可选，默认 true
  timeout_seconds: 30    # 可选，默认 30
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `host` | `127.0.0.1` | TiDB 服务器地址 |
| `port` | `4000` | TiDB 自身的默认 SQL 端口，不是 MySQL 的 3306 |
| `username` | 必填 | TiDB 用户名 |
| `password` | 空 | TiDB 密码 |
| `database` | 无 | 连接初始数据库 |
| `charset` | `utf8mb4` | 连接字符集 |
| `autocommit` | `true` | 自动提交模式 |
| `timeout_seconds` | `30` | 连接超时 |

TiDB 使用 MySQL 通信协议，对象按 `database.table` 寻址：与 MySQL 一样没有 schema 层级，`catalog` 和
`schema` 都留空即可。请显式设置 `port`——TiDB 监听 4000，同一台机器上的 3306 通常是另一个完全不相干的
服务。

兼容 MySQL 不等于是 MySQL 的超集。TiDB 不支持 `FULL OUTER JOIN`、`JSON_TABLE`、`LATERAL`、
`CREATE TABLE ... AS SELECT`、`CORR`/`COVAR_*`、物化视图和可更新视图，默认排序规则是区分大小写的
`utf8mb4_bin`。另有两个子句会被接受但不生效：未开启 `tidb_enable_check_constraint` 时 `CHECK` 约束不
强制执行，`FULLTEXT` 索引会被静默丢弃。适配器自带的 SQL skill 会让生成的 SQL 避开这些构造。

**TiFlash。** TiDB 的列存副本引擎按表启用：执行 `ALTER TABLE t SET TIFLASH REPLICA 1` 后，优化器会自动
在行存与列存之间选择，查询本身无需改动，Datus 侧也无需任何配置。查询 `information_schema.TIFLASH_REPLICA`
可以查看哪些表已有同步完成的副本。通常无法从副本获益的是聚合类窗口函数——在 TiDB v8.5 上它们会回退到 TiDB 单节点计算，结果正确但失去
并行，因此在两种写法都可行时应优先使用 `GROUP BY` 聚合。下推范围随版本和调用周围的算子而变，判断前
请用 `EXPLAIN` 确认窗口算子是否标记为 `mpp[tiflash]`。

`datus-tidb` 适配器目前不支持 TLS 配置，因此无法连接需要 TLS 的 TiDB Cloud 端点。这是适配器的限制，不是 TiDB 本身的限制。

### Hologres

```yaml
hologres_data:
  type: hologres
  host: your-instance-cn-hangzhou.hologres.aliyuncs.com  # 控制台 endpoint，可内嵌 ":80"
  port: 80
  username: ${HOLOGRES_ACCESS_KEY_ID}
  password: ${HOLOGRES_ACCESS_KEY_SECRET}
  database: your_database
  schema: public   # 可选，默认为 public
  sslmode: prefer  # 可选，默认为 prefer
```

Hologres（阿里云）使用 PostgreSQL wire 协议和 AccessKey 凭证。`access_key_id`/`access_key_secret`
也可作为 `username`/`password` 的别名。`host` 支持纯 hostname 或 `hostname:port` 形式的控制台
endpoint；显式配置的 `port` 必须与 endpoint 中内嵌的端口一致。

### Oracle

```yaml
oracle_data:
  type: oracle
  host: localhost
  port: 1521
  username: datus_user
  password: your_password
  service_name: FREEPDB1
  schema: DATUS_USER
```

连接目标必须且只能配置一个：`service_name`（推荐）、`sid` 或 `dsn`。service/PDB 只用于选择连接目标，
不属于 SQL 对象名；对象应使用 `SCHEMA.TABLE` 限定。

### GaussDB

```yaml
gaussdb_data:
  type: gaussdb
  host: localhost
  port: 5432
  username: datus
  password: ${GAUSSDB_PASSWORD}
  database: postgres
  schema: public   # 可选，默认为 public
  # driver: pg8000  # 可选；省略时 Linux 默认 gaussdb，macOS 默认 pg8000
  sslmode: verify-ca
  sslrootcert: /etc/datus/certs/gaussdb-ca.pem
```

GaussDB 与 openGauss 使用 PostgreSQL wire 协议。驱动应按运行平台和服务端账号的密码认证方式选择：

| 驱动 | 平台 | 认证方式 |
|------|------|----------|
| `gaussdb`（Linux 默认） | Linux | sha256、md5、sm3 |
| `pg8000`（macOS 默认） | Linux 和 macOS | sha256、md5 |
| `psycopg2` | Linux 和 macOS | 仅 md5；作为兜底方案 |

所有驱动都接受 `disable`、`allow`、`prefer`（默认）、`require`、`verify-ca` 和 `verify-full`，但重试行为
并不完全相同。生产环境应以同时配置 `verify-ca` 与 `sslrootcert` 为基线：连接会加密，并验证服务端证书
是否由该 CA 签发。当配置的 `host` 能保证与服务端证书匹配时，使用 `verify-full` 可进一步验证 hostname；
否则继续使用 `verify-ca`。`require` 只加密、不验证服务端身份；`prefer` 允许回退到非加密连接。服务端
只是启用 TLS 时，客户端不需要自身证书；服务端强制 TLS 时，`prefer` 通常也会协商出 TLS，但应显式使用
`require` 或任一 `verify-*` 模式，避免客户端连接到配置不同的 endpoint 时回退到明文。其中只有
`verify-ca` 和 `verify-full` 需要在客户端配置服务端 CA。`pg8000` 会把 `allow` 按 `prefer` 处理（先尝试
TLS），因为其 API 无法表达 libpq 的明文优先重试顺序。当前适配器仅支持单向 TLS，不支持通过
`sslcert`/`sslkey` 配置双向 TLS。

集中式与分布式部署均受支持。连接器会自动探测数据库的 A（Oracle）、B（MySQL）或 PG 兼容模式，
使生成的 SQL 遵循对应语义。

## 多数据库连接

可以在 `agent.services.datasources` 下配置多个独立数据源连接：

```yaml
agent:
  services:
    datasources:
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

#### Hive
- HiveServer2/Thrift 协议连接
- Hive session 配置支持
- 多种认证机制（NONE、LDAP、CUSTOM、KERBEROS）
- 数据库上下文切换（USE 语句）

#### Spark
- 通过 HiveServer2 协议连接 Spark Thrift Server
- 多种认证机制（NONE、PLAIN、KERBEROS）
- Spark SQL 方言支持

#### ClickHouse
- HTTP 协议连接
- 无 schema 层（数据库即 schema）
- ClickHouse 特有的 DML 语法（ALTER TABLE UPDATE）
- 轻量级删除支持

#### Trino
- 三级层次结构：catalog → schema → table
- 跨 catalog 查询支持
- 内置 TPC-H 连接器用于基准测试
- HTTP/HTTPS 连接及 SSL 支持

#### Apache Doris
- 通过 FE 查询端口兼容 MySQL 协议
- 多 catalog 支持：`SHOW CATALOGS` 发现，以及 `SWITCH <catalog>`、`USE [catalog.]database` 上下文切换
- 元数据读取使用 catalog 限定的 `information_schema`，无需切换会话级 catalog，且线程安全
- `catalog.database.table` 三段式标识符，支持反引号引用；没有 schema 层
- 通过 `mv_infos()` 发现异步物化视图，并获取其 DDL
- 元数据区分 Doris 的三种 key 模型（Duplicate Key、Unique Key、Aggregate Key）的 key 列
- Catalog 感知的样本数据获取，支持 list、CSV、Pandas、Arrow 结果格式
- 内置 `db-doris-sql` Skill，覆盖表模型、分桶、物化视图，以及 Stream Load、Routine Load 和基于 TVF / catalog 的 `INSERT INTO SELECT` 数据导入
- 支持作为迁移目标：表结构布局建议、DDL 校验、源类型映射，以及在集群上试运行 `CREATE TABLE`

#### Hologres
- PostgreSQL wire 协议（PostgreSQL 兼容 SQL 方言）
- 阿里云 AccessKey 认证
- 多 schema 数据源支持
- 控制台 endpoint 归一化（`hostname` 或 `hostname:port`）
- SSL 连接模式（disable、allow、prefer、require、verify-ca、verify-full）

#### Oracle
- 通过 adapter 提供的 skill 注入 Oracle Database 19c SQL 和 PL/SQL 语法知识
- 支持 service name、SID 或 DSN 连接目标
- 通过 `ALL_*` 数据字典视图发现 schema 范围内的元数据
- Oracle 兼容的 profiling 和绑定参数数据传输

#### GaussDB
- PostgreSQL wire 协议（PostgreSQL 兼容 SQL 方言）
- 通过官方 `gaussdb` 驱动支持 sha256、md5 和 sm3 认证
- 通过纯 Python `pg8000` 驱动在 Linux 和 macOS 支持 sha256/md5 认证
- 支持到 `verify-full` 的 TLS 模式；生产环境以 `verify-ca` 为基线，`verify-full` 可增加 hostname 验证
- 自动探测 A / B / PG 兼容模式，使 SQL 生成遵循服务端语义
- 同时支持集中式与分布式部署，并生成分布键感知的建表 DDL
- 多 schema 数据源支持

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
- **Hive**：需要 `pyhive`、`thrift`、`thrift-sasl`、`pure-sasl`（自动安装）
- **Spark**：需要 `pyhive`、`thrift`、`thrift-sasl`、`pure-sasl`（自动安装）
- **ClickHouse**：需要 `clickhouse-sqlalchemy`（自动安装）
- **Trino**：需要 `trino`（自动安装）
- **Apache Doris**：需要 `datus-mysql` 和 `pymysql`（自动安装）
- **Hologres**：需要 `datus-postgresql` 和 `psycopg2-binary`（自动安装）
- **Oracle**：需要 `oracledb`（自动安装；Thin 模式不需要 Oracle Client）
- **GaussDB**：需要 `datus-postgresql`，相关依赖会自动安装。Linux 默认使用官方 `gaussdb` 驱动，
  发布 wheel 已内置其 GaussDB 系 libpq；macOS 没有兼容的原生 libpq，因此默认使用纯 Python `pg8000` 驱动。

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
    │   │   ├── datus-starrocks
    │   │   └── datus-doris
    │   ├── datus-postgresql
    │   │   ├── datus-hologres
    │   │   └── datus-gaussdb
    │   ├── datus-hive
    │   ├── datus-spark
    │   ├── datus-clickhouse
    │   ├── datus-trino
    │   └── datus-oracle
    │
    └── 原生 SDK 适配器
        ├── datus-snowflake
        └── datus-clickzetta
```

适配器系统使用 Python 的 entry points 机制实现自动发现。当您安装适配器包时，它会自动注册到 Datus Agent 并可供使用。

## 下一步

- [快速开始](../getting_started/Quickstart.md) - 开始使用 Datus Agent
- [配置参考](../configuration/introduction.md) - 详细配置选项
