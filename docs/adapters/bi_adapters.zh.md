# BI 适配器

Datus Agent 通过插件化适配器系统接入 BI 平台。这些适配器为 `gen_dashboard` 和 BI bootstrap 等工作流提供能力。

## 概览

BI 适配器让 Datus 能够连接外部仪表盘平台，并提供统一工具层来完成：

- 列出仪表盘、图表和数据集
- 创建仪表盘和图表
- 在平台支持时注册数据集

数据搬运不由 BI 适配器或 `gen_dashboard` 负责。先用 `gen_job` 或
`scheduler` 单独把 serving 表准备好，再调用 `gen_dashboard` 基于这些已存在的表
创建 BI dataset、chart 和 dashboard。

BI 运行时配置统一写在 `agent.yml` 的 `agent.services.bi_platforms` 下。

## 支持的平台

| 平台 | 包名 | 安装方式 | 状态 |
|------|------|----------|------|
| Apache Superset | `datus-bi-superset` | `pip install datus-bi-superset` | 可用 |
| Grafana | `datus-bi-grafana` | `pip install datus-bi-grafana` | 可用 |

## 安装

按目标平台安装对应的适配器包：

```bash
# Superset
pip install datus-bi-superset

# Grafana
pip install datus-bi-grafana
```

安装后，Datus 会通过 Python entry points 自动发现适配器。

## 配置

Serving DB 自身是一条普通的 `services.datasources` 记录；BI 平台条目通过
`dataset_db.datasource_ref` 按名引用它，从而复用 Datus 已有的连接池、schema
metadata 和凭据。

```yaml
agent:
  services:
    datasources:
      serving_pg:
        type: postgresql
        host: 127.0.0.1
        port: 5433
        database: superset_examples
        schema: bi_public
        username: ${SERVING_WRITE_USER}
        password: ${SERVING_WRITE_PASSWORD}

    bi_platforms:
      superset:
        type: superset
        api_base_url: http://localhost:8088
        username: ${SUPERSET_USER}
        password: ${SUPERSET_PASSWORD}
        dataset_db:
          datasource_ref: serving_pg
          bi_database_name: analytics_pg

      grafana:
        type: grafana
        api_base_url: http://localhost:3000
        api_key: ${GRAFANA_API_KEY}
        dataset_db:
          datasource_ref: serving_pg          # 可以共用同一个 serving DB
          bi_database_name: PostgreSQL        # Grafana 中的数据源名称

  agentic_nodes:
    gen_dashboard:
      bi_platform: superset
```

## 选择规则

- `bi_platform` 用来从 `services.bi_platforms` 中选择一个平台配置。
- 配置 key 应与平台名一致，例如 `superset` 或 `grafana`。
- 如果只配置了一个 BI 平台，省略 `bi_platform` 时 Datus 会自动选择它。
- 如果配置了多个 BI 平台，就应显式设置 `bi_platform`。

## 平台说明

### Superset

- 认证方式使用 `username` 和 `password`。
- `dataset_db.datasource_ref` 指向一个 `services.datasources` 条目；Datus
  用该 datasource 对应的 adapter（如 `datus-postgresql`）做 schema 探查
  和写入，凭据从 datasource 中读取。
- `dataset_db.bi_database_name` 与 Superset `list_bi_databases()` 返回的
  别名保持一致，`gen_dashboard` 用它解析 `create_dataset` 所需的
  `database_id`。

### Grafana

- 认证方式使用 `api_key`。
- `dataset_db.bi_database_name` 应与 Grafana 中已有的数据源名称一致
  （Grafana 的连接别名）。
- Grafana 面板直接嵌入 SQL，因此没有单独的数据集注册步骤。

## 工作流差异

| 维度 | Superset | Grafana |
|------|----------|---------|
| 认证方式 | 用户名 + 密码 | API Key |
| 数据集注册 | 有 | 无 |
| 创建图表前置条件 | `dataset_id` | `dashboard_id` + SQL |
| Serving DB 别名 | `dataset_db.bi_database_name` | `dataset_db.bi_database_name` |

## 运行时说明

- `services.bi_platforms` 是 BI 凭据的唯一运行时来源。
- 敏感值支持 `${ENV_VAR}` 环境变量替换。
- 不再支持旧的内联形式（`dataset_db: {uri: "..."}` 或
  `dataset_db: {type:..., host:..., ...}`），请把连接字段挪到
  `services.datasources.<name>` 下，再用 `dataset_db.datasource_ref`
  按名引用。

## 相关文档

- [BI 平台配置](../configuration/bi_platforms.md)
- [生成 Dashboard](../subagent/gen_dashboard.zh.md)
- [数据工程快速开始](../getting_started/data_engineering_quickstart.zh.md)
