# 数据工程快速开始

本指南使用开源的 DAComp 数据工程数据集，串起一条完整的 Datus 工作流：理解数仓分层设计、交互式建表、生成 ETL、产出 marts 数据、提交 Airflow 天级任务，并把结果写入 Superset 创建仪表盘。

## 步骤 0：下载 quickstart 数据

DAComp **不包含**在 `datus-agent` 仓库中。本文使用一个从 DAComp Lever
示例整理出来的小型 quickstart 数据包，不需要下载完整 DAComp 压缩包。

下载并解压 quickstart 数据包和本地 Docker stack：

```bash
mkdir -p ~/datus-quickstart-data
cd ~/datus-quickstart-data

curl -L -o datus-de-lever-quickstart-v1.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v1/datus-de-lever-quickstart-v1.zip
curl -L -o datus-data-engineering-quickstart-stack-v1.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v1/datus-data-engineering-quickstart-stack-v1.zip

unzip datus-de-lever-quickstart-v1.zip
unzip datus-data-engineering-quickstart-stack-v1.zip
```

然后把 `DACOMP_HOME` 指向本地解压后的数据目录，把 `DATUS_QUICKSTART_STACK`
指向本地解压后的 Docker stack，并复制一份可写的 DuckDB 工作库：

```bash
export DACOMP_HOME=/absolute/path/to/datus-de-lever-quickstart
export DATUS_QUICKSTART_STACK=/absolute/path/to/datus-data-engineering-quickstart-stack
cp "$DACOMP_HOME/lever_start.duckdb" "$DACOMP_HOME/lever_workbench.duckdb"
cd "$DACOMP_HOME"
```

后续步骤默认这个目录下至少有这些文件：

- `docs/data_contract.yaml`
- `config/layer_dependencies.yaml`
- `lever_start.duckdb`

## 步骤 1：理解数仓分层

这个 DAComp 示例已经给出了一套典型的分层数仓设计：

| 层级 | 表数量 | 作用 |
|---|---:|---|
| `staging` | 24 | 清洗原始 ATS 数据，统一类型和格式 |
| `intermediate` | 17 | 做实体关联和可复用业务逻辑 |
| `marts` | 14 | 产出可直接分析、报表和出图的结果层 |

最关键的两个设计文件是：

- `docs/data_contract.yaml`：描述字段清洗、校验和标准化规则
- `config/layer_dependencies.yaml`：描述层级顺序与表依赖关系

在开始写 DDL 和 ETL 之前，先把这两份文件过一遍，后面给 agent 的提示词就能更贴近原始设计。

## 步骤 2：启动本地 quickstart 环境

下载的 stack 中已经包含本文真正会用到的两套本地 demo 环境。

启动 Superset：

```bash
cd "$DATUS_QUICKSTART_STACK/superset"
docker compose up -d
```

启动 Airflow：

```bash
cd "$DATUS_QUICKSTART_STACK/airflow"
docker compose up -d
```

本地默认访问方式：

- Superset：`http://127.0.0.1:8088`，用户名 `admin`，密码 `admin`
- Airflow：`http://127.0.0.1:8080`，用户名 `admin`，密码 `admin`

这套 quickstart 的 Superset compose 已经带了本地演示用的元数据库和管理员默认值。

Airflow 的 compose 文件会挂载 `${DACOMP_HOME}`，并暴露一个名为
`duckdb_dacomp_lever` 的连接，指向 `/workspace/lever_workbench.duckdb`。

## 步骤 3：安装本文需要的适配器

安装 Datus 以及这条链路会用到的适配器：

```bash
pip install datus-agent datus-bi-superset datus-postgresql datus-scheduler-airflow
```

## 步骤 4：配置 `agent.yml`

把下面这段 service 配置合并到 `~/.datus/conf/agent.yml` 现有的 `agent:`
下面。保留已有的 `agent.providers` 配置；`/model` 会使用这些凭据。路径会直接使用步骤
0 里导出的 `DACOMP_HOME` 和 `DATUS_QUICKSTART_STACK` 环境变量。

```yaml
agent:
  services:
    datasources:
      lever_duckdb:
        type: duckdb
        uri: "duckdb:///${DACOMP_HOME}/lever_workbench.duckdb"
        default: true
      superset_serving:
        type: postgresql
        host: 127.0.0.1
        port: 5433
        database: superset_examples
        schema: public
        username: superset
        password: superset

    bi_platforms:
      superset:
        type: superset
        api_base_url: http://127.0.0.1:8088
        username: admin
        password: admin
        dataset_db:
          datasource_ref: superset_serving
          bi_database_name: examples

    schedulers:
      airflow_prod:
        type: airflow
        api_base_url: http://127.0.0.1:8080/api/v1
        username: admin
        password: admin
        dags_folder: "${DATUS_QUICKSTART_STACK}/airflow/dags"
        connections:
          duckdb_dacomp_lever: DAComp Lever DuckDB

  agentic_nodes:
    gen_dashboard:
      bi_platform: superset
    scheduler:
      scheduler_service: airflow_prod
```

然后使用 `lever_duckdb` datasource 启动 Datus；它指向上面的 DuckDB 工作库：

```bash
cd "$DACOMP_HOME"
datus-cli --datasource lever_duckdb
```

如果 CLI 提示还没有配置模型，继续之前先在 CLI 内运行：

```text
/model
```

选择 provider/model，并按提示填写凭据。`/model` 会把 provider 凭据写入
`~/.datus/conf/agent.yml` 的 `agent.providers`，并把当前项目使用的
provider/model 写入 `./.datus/config.yml`。

这里的 `dags_folder` 是 Datus 在主机上写入 DAG 文件的目录。Airflow compose 会把这个目录挂载到 Airflow 容器内的 `/opt/airflow/dags`，所以 Datus 生成的新 DAG 会被 Airflow 自动发现。

## 步骤 5：创建必要的 staging 表

自然语言 agent 任务不要以 `CREATE`、`COPY` 这类 SQL 动词开头；CLI 会根据这些
开头关键字判断是否直接执行 SQL。

先要求 agent 创建目标 schema：

```text
Please set up the target schemas staging, intermediate, and marts in the current DuckDB database. Keep the existing raw schema unchanged.
```

这条教程只构建一条窄但完整的依赖链：`marts.lever__requisition_enhanced`。
字段选择、字段重命名和业务逻辑以 `docs/data_contract.yaml` 为准。

再要求 agent 根据 `lever__requisition_enhanced` 和
`intermediate.int_lever__requisition_users` 的 `source_models` 创建必需的
staging 表。agent 会把任务分发到建表流程：

```text
Read ./docs/data_contract.yaml and create the staging tables needed for marts.lever__requisition_enhanced: staging.stg_lever__requisition from raw.requisition, staging.stg_lever__user from raw.user, staging.stg_lever__requisition_posting from raw.requisition_posting, and staging.stg_lever__requisition_offer from raw.requisition_offer. Use the field design and source-to-target mapping from the contract.
```

这四张 staging 表就是 requisition enhanced 示例需要的最小 raw-to-staging 输入。

## 步骤 6：生成 intermediate 和 marts 表

先生成 intermediate 表。它应该按照 `docs/data_contract.yaml` 中
`int_lever__requisition_users` 的定义，把 requisition 字段和 user 字段关联起来。

创建 intermediate 表：

```text
Read ./docs/data_contract.yaml and create intermediate.int_lever__requisition_users from staging.stg_lever__requisition and staging.stg_lever__user. Use the contract's field design, joins, and source-to-target mapping.
```

再生成面向分析的 marts 表。契约中定义 `marts.lever__requisition_enhanced`
是一张按 `requisition_id` 一行的表，依赖：

- `intermediate.int_lever__requisition_users`
- `staging.stg_lever__requisition_posting`
- `staging.stg_lever__requisition_offer`

创建 marts 表：

```text
Read ./docs/data_contract.yaml and create marts.lever__requisition_enhanced from intermediate.int_lever__requisition_users, staging.stg_lever__requisition_posting, and staging.stg_lever__requisition_offer. Use the contract's business logic: keep all base requisition rows, count posting and offer links by requisition_id, fill missing counts with 0, and add has_posting and has_offer flags.
```

这条链路的基本顺序始终是：

```text
staging -> intermediate -> marts
```

生成完成后，可以直接验证 marts 表：

```sql
SELECT COUNT(*) FROM marts.lever__requisition_enhanced;
```

## 步骤 7：提交天级 Airflow 任务

现在可以要求 agent 把 marts 刷新过程提交给 scheduler。quickstart 自带的 Airflow 已经预置好了 `duckdb_dacomp_lever` 连接。

提交一个每天早上 8 点运行的 SQL 任务，刷新同一条从契约生成的链路：

```text
Submit a daily SQL job named daily_lever_requisition_enhanced that refreshes staging.stg_lever__requisition, staging.stg_lever__user, staging.stg_lever__requisition_posting, staging.stg_lever__requisition_offer, intermediate.int_lever__requisition_users, and marts.lever__requisition_enhanced at 8am every day using the duckdb_dacomp_lever connection. Use the SQL generated and validated from docs/data_contract.yaml in the previous steps.
```

再手动触发一次做验证：

```text
Trigger daily_lever_requisition_enhanced once now and show me the latest run status
```

你应该会看到：

- `${DATUS_QUICKSTART_STACK}/airflow/dags` 下生成新的 DAG 文件
- 同一份文件会在 Airflow 容器内显示为 `/opt/airflow/dags/<dag_id>.py`
- scheduler 返回 `job_id`
- Airflow UI 中出现对应任务

## 步骤 8：把 marts 表同步到 Superset serving DB

上面的 marts 表是通过 `lever_duckdb` datasource 生成的。创建仪表盘之前，需要先把它复制到
`dataset_db.datasource_ref` 指向的 BI 注册数据库 `superset_serving`（Postgres）。
这里的 `lever_duckdb` 和 `superset_serving` 都是 `agent.yml` 里的 Datus
datasource 名称，不是 DuckDB 或 Postgres 内部真实的 database/catalog 名。

```text
Please copy the source table marts.lever__requisition_enhanced from the lever_duckdb datasource into the superset_serving datasource as public.lever__requisition_enhanced, replacing the target table if it already exists. Then verify the source and target row counts.
```

完成后，这张表就位于 Superset 通过 `bi_database_name: examples` 识别的数据库中。

## 步骤 9：创建 Superset Dashboard

当表已经存在于 `superset_serving`，就可以要求 agent 创建仪表盘：

```text
Please create a requisition operations dashboard in Superset from public.lever__requisition_enhanced. Include KPI tiles for total requisitions, open requisitions, requisitions with postings, requisitions with offers, and total requested headcount. Add charts by status, team, location, employment_status, count_postings, and count_offers.
```

数据准备是单独的 ETL / scheduler 步骤。仪表盘生成流程期望目标表或
SQL dataset 已经存在于 BI 已注册的数据库中。

## 步骤 10：验证端到端结果

走完整条链路后，你应该能确认：

- `lever_workbench.duckdb` 里已经有 `staging`、`intermediate`、`marts` 三层 schema
- `marts.lever__requisition_enhanced` 是从 raw 数据经 staging 和 intermediate 逐层加工得到的
- Airflow 中能看到日常调度任务
- 仪表盘生成流程返回了 Superset dashboard URL
