# 存储（Storage）

配置嵌入模型与向量数据库，用于表结构/样例数据、文档与指标的嵌入与检索。

## 结构
```yaml
storage:
  base_path: data                # RAG 向量存储根目录
  embedding_device_type: cpu     # cpu/cuda/mps/auto

  database:
    registry_name: openai
    model_name: text-embedding-v3-small
    dim_size: 1024
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

- 路径示例：`data/datus_db_<namespace>`（如 `data/datus_db_snowflake`）
- 设备选项：`cpu`、`cuda`、`mps`、`auto`

## 嵌入模型

### 数据库嵌入（表结构/样例）
```yaml
database:
  registry_name: openai                # openai 或 sentence-transformers
  model_name: text-embedding-v3-small
  dim_size: 1024
  batch_size: 10
  target_model: openai                 # 关联 agent.models
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
  model_name: text-embedding-v3-small   # 或 v3-large
  dim_size: 1536                        # small=1536, large=3072
  batch_size: 10
  target_model: openai
```

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
    model_name: text-embedding-v3-small
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
    model_name: text-embedding-v3-large
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
    ext_knowledge: ""

storage:
  metric:
    model_name: all-MiniLM-L6-v2
    dim_size: 384
```
