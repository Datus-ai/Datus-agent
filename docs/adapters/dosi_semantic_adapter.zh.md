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
`time_dimension` 主时间轴、relationship `join_type`、metric `fill_nulls_with`，
以及结构化窗口。datus-ext 1.3 支持期间比较、滚动、累计、排名与分布、
first/last/nth value、向前或向后 offset，以及基于 ROWS 或 RANGE 的统计 frame。

## Authoring 与校验

选择 `dosi` 后会启用 strict OSI authoring 和 Dosi 原生 extension profile。
`validate_semantic`、`list_metrics`、`get_dimensions` 和 `query_metrics` 都由
原生 engine 执行。时间查询用 `metric_time` 作为 dimension，并把粒度单独
传入；返回的 `metric_time__<grain>` 是对应的结果列/order key。

统一 `semantic_modeling` 会分别列出有效模型和可修复模型：有效模型直接绑定，
可修复模型按原名称原地规划和修改。最终由 Host 校验确切 YAML，并把该 artifact
完整同步到 Knowledge Base。
