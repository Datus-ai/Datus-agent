# BI 适配器

Datus Agent 通过插件化适配器系统接入 BI 平台。这些适配器为 `gen_dashboard` 和 BI bootstrap 等工作流提供能力。

## 概览

BI 适配器让 Datus 能够连接外部仪表盘平台，并提供统一工具层来完成：

- 列出仪表盘、图表和数据集
- 创建仪表盘和图表
- 将 SQL 结果物化到 BI 面向的数据库
- 在平台支持时注册数据集

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

在 `agent.services.bi_platforms` 下配置 BI 平台：

```yaml
agent:
  services:
    bi_platforms:
      superset:
        type: superset
        api_url: http://localhost:8088
        username: ${SUPERSET_USER}
        password: ${SUPERSET_PASSWORD}
        dataset_db:
          uri: ${SUPERSET_DB_URI}
          schema: public

      grafana:
        type: grafana
        api_url: http://localhost:3000
        api_key: ${GRAFANA_API_KEY}
        dataset_db:
          uri: ${GRAFANA_DB_URI}
          datasource_name: PostgreSQL

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
- `dataset_db.uri` 是 `write_query` 物化结果时写入的 SQLAlchemy 目标库。
- Superset 中应预先存在对应的数据库连接，这样 `create_dataset` 才能注册物化表。

### Grafana

- 认证方式使用 `api_key`。
- `dataset_db.uri` 是 `write_query` 物化结果时写入的 SQLAlchemy 目标库。
- `dataset_db.datasource_name` 应与 Grafana 中已有的数据源名称一致。
- Grafana 面板直接嵌入 SQL，因此没有单独的数据集注册步骤。

## 工作流差异

| 维度 | Superset | Grafana |
|------|----------|---------|
| 认证方式 | 用户名 + 密码 | API Key |
| 数据集注册 | 有 | 无 |
| 创建图表前置条件 | `dataset_id` | `dashboard_id` + SQL |
| 物化目标 | `dataset_db.uri` | `dataset_db.uri` |
| 额外选择字段 | 无 | `dataset_db.datasource_name` |

## 运行时说明

- `services.bi_platforms` 是 BI 凭据的唯一运行时来源。
- 敏感值支持 `${ENV_VAR}` 环境变量替换。

## 相关文档

- [BI 平台配置](../configuration/bi_platforms.md)
- [生成 Dashboard](../subagent/gen_dashboard.md)
- [数据工程快速开始](../getting_started/data_engineering_quickstart.md)
