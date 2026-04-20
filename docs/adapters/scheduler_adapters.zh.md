# Scheduler 适配器

Datus Agent 通过插件化适配器系统接入外部调度平台。这些适配器为 `scheduler` subagent 和定时作业生命周期管理提供能力。

## 概览

Scheduler 适配器让 Datus 能够在外部编排平台上提交、查看和管理定时作业。当前运行时主要面向 Airflow 驱动的 SQL 和 SparkSQL 作业。

Scheduler 运行时配置统一写在 `agent.yml` 的 `agent.services.schedulers` 下。

## 支持的平台

| 平台 | 包名 | 安装方式 | 状态 |
|------|------|----------|------|
| Apache Airflow | `datus-scheduler-airflow` | `pip install datus-scheduler-airflow` | 可用 |

## 安装

安装具体的 scheduler 适配器包：

```bash
pip install datus-scheduler-airflow
```

`datus-scheduler-airflow` 会自动拉取 `datus-scheduler-core`，因此不需要单独安装 core 包。

## 配置

在 `agent.services.schedulers` 下配置 scheduler 服务：

```yaml
agent:
  services:
    schedulers:
      airflow_prod:
        type: airflow
        api_base_url: ${AIRFLOW_URL}
        username: ${AIRFLOW_USER}
        password: ${AIRFLOW_PASSWORD}
        dags_folder: ${AIRFLOW_DAGS_DIR}
        default: true
        connections:
          postgres_prod: PostgreSQL production

      airflow_dev:
        type: airflow
        api_base_url: ${AIRFLOW_DEV_URL}
        username: ${AIRFLOW_DEV_USER}
        password: ${AIRFLOW_DEV_PASSWORD}
        dags_folder: /tmp/airflow-dags

  agentic_nodes:
    scheduler:
      scheduler_service: airflow_prod
```

## 选择规则

- `scheduler_service` 用来从 `services.schedulers` 中选择一个 scheduler 实例。
- 如果只配置了一个 scheduler，Datus 可以自动使用它。
- 如果配置了多个 scheduler，应显式设置 `scheduler_service`，或者只给其中一个服务设置 `default: true`。
- 不要给多个服务同时设置 `default: true`。

## Airflow 说明

- `api_base_url` 应指向 Airflow REST API，例如 `http://localhost:8080/api/v1`。
- `dags_folder` 是 Datus 在主机侧写入 DAG 文件的目录。
- 如果 Airflow 运行在 Docker 或 Kubernetes 中，需要把该目录挂载到 Airflow 的 DAG 目录，例如 `/opt/airflow/dags`。
- `connections` 是可选项，用于整理可供定时任务使用的目标连接。
- `submit_sql_job` 或 `update_job` 引用的 SQL 文件必须事先存在于 agent 主机上。

## 运行时说明

- `services.schedulers` 是 scheduler 配置的唯一运行时来源。
- 敏感值支持 `${ENV_VAR}` 环境变量替换。

## 支持的作业类型

当前 Airflow 适配器支持：

- 基于 `.sql` 文件的定时 SQL 作业
- 基于 `.sql` 文件的定时 SparkSQL 作业
- 触发、暂停、恢复、更新、删除和查看作业运行状态

## 相关文档

- [调度器配置](../configuration/schedulers.md)
- [Scheduler Subagent 指南](../subagent/scheduler.zh.md)
- [数据工程快速开始](../getting_started/data_engineering_quickstart.zh.md)
