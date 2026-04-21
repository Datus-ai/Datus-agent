# 数据工程快速开始

本指南使用开源的 DAComp 数据工程数据集，串起一条完整的 Datus 工作流：理解数仓分层设计、交互式建表、生成 ETL、产出 marts 数据、提交 Airflow 天级任务，并把结果写入 Superset 创建仪表盘。

## 步骤 0：先下载 DAComp

DAComp **不包含**在 `datus-agent` 仓库中，请先从 Hugging Face 下载：

- <https://huggingface.co/datasets/DAComp/dacomp-de>

本教程使用其中的 `dacomp-de-impl-001` 示例。

如果你当前只是通过 PyPI 安装了 Datus，还需要额外 clone 一份 `Datus-agent` 仓库，因为本文会引用仓库里的本地 compose 环境：

```bash
git clone https://github.com/Datus-ai/Datus-agent.git
export DATUS_AGENT_REPO=/absolute/path/to/Datus-agent
```

然后把 `DACOMP_HOME` 指向本地解压后的示例目录，并复制一份可写的 DuckDB 工作库：

```bash
export DACOMP_HOME=/absolute/path/to/dacomp-de-impl-001
cp "$DACOMP_HOME/lever_start.duckdb" "$DACOMP_HOME/lever_workbench.duckdb"
cd "$DACOMP_HOME"
```

后续步骤默认这个目录下至少有这些文件：

- `docs/data_contract.yaml`
- `config/layer_dependencies.yaml`
- `sql/staging/`
- `sql/intermediate/`
- `sql/marts/`
- `lever_start.duckdb`

## 步骤 1：理解数仓分层

这个 DAComp 示例已经给出了一套典型的分层数仓设计：

| 层级 | 表数量 | 作用 |
|---|---:|---|
| `staging` | 24 | 清洗原始 ATS 数据，统一类型和格式 |
| `intermediate` | 17 | 做实体关联和可复用业务逻辑 |
| `marts` | 14 | 产出可直接分析和出图的结果层 |

最关键的两个设计文件是：

- `docs/data_contract.yaml`：描述字段清洗、校验和标准化规则
- `config/layer_dependencies.yaml`：描述层级顺序与表依赖关系

在开始写 DDL 和 ETL 之前，先把这两份文件过一遍，后面给 `gen_table` 和 `gen_job` 的提示词就能更贴近原始设计。

## 步骤 2：启动本地 quickstart 环境

仓库中已经带了本文真正会用到的两套本地 demo 环境，目录在 `quickstart/data_engineering/`。

启动 Superset：

```bash
cd "$DATUS_AGENT_REPO/quickstart/data_engineering/superset"
export SUPERSET_DB_PASSWORD='superset'
export SUPERSET_SECRET_KEY='datus-test-secret-key-not-for-prod'
export SUPERSET_ADMIN_PASSWORD='admin'
docker compose up -d
```

启动 Airflow：

```bash
cd "$DATUS_AGENT_REPO/quickstart/data_engineering/airflow"
docker compose up -d
```

本地默认访问方式：

- Superset：`http://127.0.0.1:8088`，用户名 `admin`，密码 `admin`
- Airflow：`http://127.0.0.1:8080`，用户名 `admin`，密码 `admin`

这套 quickstart 的 Superset compose 会从环境变量读取数据库密码、管理员密码和 secret key。上面的 3 个 `export` 命令故意用了本地演示值，方便直接按文档复制运行。

Airflow 的 compose 文件会挂载 `${DACOMP_HOME}`，并自动注入一个名为 `duckdb_dacomp_lever` 的连接，指向 `/workspace/lever_workbench.duckdb`。

## 步骤 3：安装本文需要的适配器

安装 Datus 以及这条链路会用到的适配器：

```bash
pip install datus-bi-superset datus-postgresql datus-semantic-metricflow datus-scheduler-airflow
```

这些包在本文中的作用分别是：

- `datus-bi-superset`：`gen_dashboard` 使用的 Superset BI 适配器
- `datus-postgresql`：连接 Superset 物化目标 PostgreSQL 所需的驱动支持
- `datus-semantic-metricflow`：生成语义模型和指标
- `datus-scheduler-airflow`：Airflow 调度适配器，会自动安装 `datus-scheduler-core`

## 步骤 4：配置 `agent.yml`

把下面这段最小配置写入 `~/.datus/conf/agent.yml`：

```yaml
agent:
  services:
    databases:
      lever_duckdb:
        type: duckdb
        uri: "duckdb:///${DACOMP_HOME}/lever_workbench.duckdb"
        default: true

    semantic_layer:
      metricflow: {}

    bi_platforms:
      superset:
        type: superset
        api_url: http://127.0.0.1:8088
        username: admin
        password: admin
        dataset_db:
          uri: "postgresql+psycopg2://superset:superset@127.0.0.1:5433/superset_examples"
          schema: public

    schedulers:
      airflow_prod:
        type: airflow
        api_base_url: http://127.0.0.1:8080/api/v1
        username: admin
        password: admin
        dags_folder: "${DATUS_AGENT_REPO}/quickstart/data_engineering/airflow/dags"
        connections:
          duckdb_dacomp_lever: DAComp Lever DuckDB

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow
    gen_metrics:
      semantic_adapter: metricflow
    gen_dashboard:
      bi_platform: superset
    scheduler:
      scheduler_service: airflow_prod
```

然后连接到这个 DuckDB 工作库启动 Datus：

```bash
cd "$DACOMP_HOME"
datus-cli --database lever_duckdb
```

这里的 `dags_folder` 是 Datus 在主机上写入 DAG 文件的目录。`quickstart/data_engineering/airflow/docker-compose.yml` 会把这个目录挂载到 Airflow 容器内的 `/opt/airflow/dags`，所以 Datus 生成的新 DAG 会被 Airflow 自动发现。

## 步骤 5：交互式创建分层对象

当你希望先看 DDL、确认后再执行时，用 `gen_table` 最合适。它会先展示 SQL，再等你确认。

先创建目标 schema：

```text
/gen_table Create schemas staging, intermediate, and marts in the current DuckDB database. Keep the existing raw schema unchanged.
```

再根据数据契约创建一个代表性的 staging 表：

```text
/gen_table Read ./docs/data_contract.yaml and create staging.stg_lever__requisition from raw.requisition. Apply the validation and normalization rules from the contract and show me the DDL before execution.
```

如果你想逐张表把关键 DDL 看清楚，可以继续用同样方式创建更多 staging 表；如果要批量落地，下一步交给 `gen_job` 更合适。

## 步骤 6：生成 ETL 并产出 marts 层数据

接下来用 `gen_job` 把 DAComp 自带的 SQL 资产真正落到当前 DuckDB 工作库中。

先生成一个代表性的 intermediate 表：

```text
/gen_job Create intermediate.int_lever__requisition_users in the current database using ./sql/intermediate/int_lever__requisition_users.sql.
```

再生成一个面向分析的 marts 表：

```text
/gen_job Create marts.lever__recruitment_analytics_dashboard in the current database using ./sql/marts/lever__recruitment_analytics_dashboard.sql.
```

这条链路的基本顺序始终是：

```text
staging -> intermediate -> marts
```

生成完成后，可以直接验证 marts 表：

```sql
SELECT COUNT(*) FROM marts.lever__recruitment_analytics_dashboard;
```

## 步骤 7：为 marts 生成语义资产

拿到 marts 结果后，就可以继续生成可复用的语义资产。

先生成语义模型：

```text
/gen_semantic_model Generate a semantic model for marts.lever__recruitment_analytics_dashboard.
```

再生成仪表盘会用到的业务指标：

```text
/gen_metrics Create metrics for open requisitions, applications, interviews, offers, and hires from marts.lever__recruitment_analytics_dashboard. subject_tree: recruiting/hiring/dashboard
```

这些 YAML 文件会写入当前项目的语义模型存储目录，后续可以复用于分析、搜索和其他 agent 工作流。

## 步骤 8：提交天级 Airflow 任务

现在可以把 marts 刷新过程提交给 scheduler。quickstart 自带的 Airflow 已经预置好了 `duckdb_dacomp_lever` 连接。

提交一个每天早上 8 点运行的 SQL 任务：

```text
/scheduler Submit a daily SQL job named daily_lever_recruitment_dashboard from ./sql/marts/lever__recruitment_analytics_dashboard.sql at 8am every day using the duckdb_dacomp_lever connection
```

再手动触发一次做验证：

```text
/scheduler Trigger daily_lever_recruitment_dashboard once now and show me the latest run status
```

你应该会看到：

- `${DATUS_AGENT_REPO}/quickstart/data_engineering/airflow/dags` 下生成新的 DAG 文件
- 同一份文件会在 Airflow 容器内显示为 `/opt/airflow/dags/<dag_id>.py`
- scheduler 返回 `job_id`
- Airflow UI 中出现对应任务

## 步骤 9：写入 Superset 并创建 Dashboard

当 marts 表已经稳定可查，就可以把它交给 BI subagent：

```text
/gen_dashboard Create a recruiting operations dashboard in Superset from marts.lever__recruitment_analytics_dashboard. Include KPI tiles for requisitions, applications, interviews, offers, and hires, plus a funnel chart and a weekly trend chart.
```

这条 Superset 工作流内部会依次执行：

```text
write_query -> create_dataset -> create_chart -> create_dashboard -> add_chart_to_dashboard
```

因为 `bi_platforms.superset.dataset_db` 已经指向本地 Superset 所用的 PostgreSQL，Datus 会把查询结果物化进去，并自动注册成 Superset dataset。

## 步骤 10：验证端到端结果

走完整条链路后，你应该能确认：

- `lever_workbench.duckdb` 里已经有 `staging`、`intermediate`、`marts` 三层 schema
- 至少有一张 marts 表可以正常查询
- 已生成语义模型和指标 YAML
- Airflow 中能看到日常调度任务
- `gen_dashboard` 返回了 Superset dashboard URL

如果你还想分别深入某一段能力，请继续阅读：

- [数据库适配器](../adapters/db_adapters.md)
- [语义层适配器](../adapters/semantic_adapters.md)
- [Table Generation](../subagent/gen_table.md)
- [ETL Job](../subagent/gen_job.md)
- [Scheduler](../subagent/scheduler.md)
- [Generate Dashboard](../subagent/gen_dashboard.md)
