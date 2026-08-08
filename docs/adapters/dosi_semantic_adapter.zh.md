# Dosi 语义适配器

Dosi adapter 使用原生 Rust engine 直接执行 OSI 语义模型。authoring、校验、
catalog 加载和执行都由 `datus-semantic-dosi` 提供；它不会加载 Python OSI
adapter，也不会降低到 MetricFlow。

## 安装

本地开发不要求先发布 `dosi-engine`。在标准的同级仓库布局中，从
`Datus-agent` checkout 进入同一个 Python 环境后安装源码：

```bash
uv pip install -e ../osi-engine/crates/dosi-py
uv pip install -e ../datus-semantic-adapter/datus-semantic-core
uv pip install -e ../datus-semantic-adapter/datus-semantic-dosi
```

adapter 仍声明逻辑上的 `dosi-engine` 依赖，但本地 editable install 会直接
用源码满足它。只有通过包仓库安装或正式交付时才需要已发布版本。

## 配置

```yaml
agent:
  services:
    datasources:
      warehouse:
        type: duckdb
        uri: /absolute/path/to/warehouse.db
    semantic_layer:
      dosi:
        default: true
        # semantic_model_path: /absolute/path/to/model.yaml
```

Datus 会注入当前数据源连接，并默认把 `semantic_models_path` 指向
`subject/semantic_models/<datasource>/`。目录中只有一个 YAML 或 JSON 文件时
会自动选择；存在多个模型文件时，需要配置 `semantic_model_path`。

## 当前支持范围

Dosi 当前可执行 aggregate、ratio、expression 指标，维度、many-to-one
关系、复合 Join、query-backed dataset、过滤、day 到 year 的时间粒度、
D-TIME 主时间轴、D-JOIN/D-FILL，以及原生 D-WINDOW 的同比/环比、滚动和
累计指标。当前 datus-ext 1.2 的 D-WINDOW 使用单个普通 aggregate 作为
base expression，并在其上声明窗口派生语义。

## Authoring 与校验

选择 `dosi` 后会启用 strict OSI authoring 和 Dosi 原生 extension profile。
`validate_semantic`、`list_metrics`、`get_dimensions` 和 `query_metrics` 都由
原生 engine 执行。时间查询用 `metric_time` 作为 dimension，并把粒度单独
传入；返回的 `metric_time__<grain>` 是对应的结果列/order key。

切换已有 Python-OSI 模型前，先运行一次性检查工具。默认只读；`--write`
只转换语义明确的情况，使用已安装的原生 engine 校验，创建同目录 `.bak`
备份，并拒绝自动改写有歧义的文件：

```bash
datus-dosi-migrate subject/semantic_models/<datasource>
datus-dosi-migrate subject/semantic_models/<datasource> --write
```
