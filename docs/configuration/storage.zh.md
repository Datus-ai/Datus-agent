# 存储（Storage）

配置嵌入模型与向量数据库，用于表结构/样例数据、文档与指标的嵌入与检索。

## 结构
```yaml
storage:
  base_path: data                # RAG 向量存储根目录
  embedding_device_type: cpu     # cpu/cuda/mps/auto

  database:
    registry_name: openai
    model_name: text-embedding-3-small
    dim_size: 1536
    batch_size: 10
    target_model: openai

  document:
    model_name: all-MiniLM-L6-v2
    dim_size: 384

  metric:
    model_name: all-MiniLM-L6-v2
    dim_size: 384
```

### 路径与设备
```yaml
storage:
  base_path: data
  embedding_device_type: auto
```

- 路径示例：`data/datus_db_<datasource>`（如 `data/datus_db_snowflake`）
- 设备选项：`cpu`、`cuda`、`mps`、`auto`

## 嵌入模型

### 数据库嵌入（表结构/样例）
```yaml
database:
  registry_name: openai                # openai 或 sentence-transformers
  model_name: text-embedding-3-small
  dim_size: 1536
  batch_size: 10
  target_model: openai                 # 关联 models 配置
```
**参数**：`registry_name`、`model_name`、`dim_size`、`batch_size`、`target_model`

### 文档嵌入
```yaml
document:
  model_name: all-MiniLM-L6-v2
  dim_size: 384
```

### 指标嵌入
```yaml
metric:
  model_name: all-MiniLM-L6-v2
  dim_size: 384
```

## 提供方选项

### OpenAI（云）
```yaml
database:
  registry_name: openai
  model_name: text-embedding-3-small   # 或 3-large
  dim_size: 1536                        # 3-small=1536, 3-large=3072
  batch_size: 10
  target_model: openai
```

### 自定义 OpenAI 兼容嵌入模型

自定义模型名需要通过 `dim_size` 声明输出向量宽度。Datus 默认不向自定义模型发送 API 的可选 `dimensions` 参数：声明向量宽度并不表示提供方支持调整输出维度。

```yaml
storage:
  database:
    registry_name: openai
    model_name: kinfra-text-embedding-0.6b
    dim_size: 1024                  # 示例：替换为提供方的真实输出宽度
    target_model: custom_provider  # 关联包含 API Key 和 base_url 的模型配置
    send_dimensions: false         # 可选：明确省略 API dimensions 参数
```

省略 `send_dimensions` 或设为 `null` 时采用自动策略：`text-embedding-3-small` 和 `text-embedding-3-large` 会发送配置的维度，自定义模型名则不发送。只有提供方支持该参数时才设置 `send_dimensions: true`，例如支持调整维度的 Azure 嵌入模型部署别名。设为 `false` 可对 OpenAI v3 模型也省略该参数，此时 `dim_size` 必须匹配原生输出宽度。`text-embedding-ada-002` 始终使用固定的 1536 维，不发送该参数。

自定义模型未配置正数 `dim_size` 时无法推断其宽度。改变模型的输出宽度后，需要按新宽度重建已有向量索引。

### Sentence-Transformers（本地）
```yaml
database:
  registry_name: sentence-transformers
  model_name: all-MiniLM-L6-v2
  dim_size: 384
```

!!! info "其它本地模型"
    - `intfloat/multilingual-e5-large-instruct`（~1.2GB，1024 维，多语种）
    - `BAAI/bge-large-en-v1.5` / `BAAI/bge-large-zh-v1.5`（~1.2GB，1024 维）

## 方案建议

=== "轻量本地"
```yaml
storage:
  base_path: data
  embedding_device_type: auto
  database:
    registry_name: sentence-transformers
    model_name: all-MiniLM-L6-v2
    dim_size: 384
  document:
    model_name: all-MiniLM-L6-v2
    dim_size: 384
  metric:
    model_name: all-MiniLM-L6-v2
    dim_size: 384
```

=== "混合云本地"
```yaml
storage:
  base_path: data
  embedding_device_type: cpu
  database:
    registry_name: openai
    model_name: text-embedding-3-small
    dim_size: 1536
    batch_size: 10
    target_model: openai
  document:
    model_name: intfloat/multilingual-e5-large-instruct
    dim_size: 1024
  metric:
    model_name: intfloat/multilingual-e5-large-instruct
    dim_size: 1024
```

=== "企业高质"
```yaml
storage:
  base_path: /opt/datus/embeddings
  embedding_device_type: cuda
  database:
    registry_name: openai
    model_name: text-embedding-3-large
    dim_size: 3072
    batch_size: 5
    target_model: openai
  document:
    model_name: BAAI/bge-large-en-v1.5
    dim_size: 1024
  metric:
    model_name: BAAI/bge-large-en-v1.5
    dim_size: 1024
```

## 与其它组件集成
```yaml
metrics:
  duckdb:
    domain: sale
    layer1: layer1
    layer2: layer2

storage:
  metric:
    model_name: all-MiniLM-L6-v2
    dim_size: 384
```

## 存储后端 {#storage-backends}

Datus Agent 采用双轨存储架构 — **向量数据库（Vector DB）** 用于 embedding 存储，**关系数据库（RDB）** 用于结构化元数据。两者均支持通过 Registry + entry-point 机制实现可插拔后端切换。

> 架构概览请参考 [知识库简介](../knowledge_base/introduction.zh.md#storage-backends)

### 默认后端

默认情况下（YAML 中无 `rdb` / `vector` 配置段），Datus Agent 使用：

- **向量存储**：LanceDB（基于文件，零配置）
- **关系存储**：SQLite（基于文件，零配置）

数据存储在 `data/datus_db_<datasource>/` 目录下。

### PostgreSQL 后端

#### 前置条件
- PostgreSQL 15+ 并启用 `pgvector` 扩展
- 安装适配器包：
  ```bash
  pip install datus-storage-postgresql
  # 或
  uv add datus-storage-postgresql
  ```
  entry-point 自动注册，无需修改代码。

#### 配置
```yaml
storage:
  rdb:
    type: postgresql
    host: ${PG_HOST:-localhost}
    port: 5432
    user: ${PG_USER:-postgres}
    password: ${PG_PASSWORD}
    dbname: datus
    pool_min_size: 1
    pool_max_size: 10

  vector:
    type: postgresql
    host: ${PG_HOST:-localhost}
    port: 5432
    user: ${PG_USER:-postgres}
    password: ${PG_PASSWORD}
    dbname: datus
    pool_min_size: 1
    pool_max_size: 10
```

**参数说明：**

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | 是 | — | `postgresql` |
| `host` | 是 | — | PostgreSQL 服务器主机名 |
| `port` | 是 | — | PostgreSQL 服务器端口 |
| `user` | 是 | — | 数据库用户 |
| `password` | 是 | — | 用户密码 |
| `dbname` | 是 | — | 数据库名称 |
| `pool_min_size` | 否 | 1 | 最小连接池大小 |
| `pool_max_size` | 否 | 10 | 最大连接池大小 |

#### 混合后端

可以混合使用不同后端 — 例如 RDB 使用 PostgreSQL，向量存储使用 LanceDB：

```yaml
storage:
  rdb:
    type: postgresql
    host: localhost
    port: 5432
    user: postgres
    password: ${PG_PASSWORD}
    dbname: datus

  vector:
    type: lance    # 默认，无需额外安装
```
