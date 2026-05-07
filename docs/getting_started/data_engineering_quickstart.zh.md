# 数据工程快速开始

本指南使用开源的 DAComp 数据工程数据集，串起一条完整的 Datus 工作流：理解数仓分层设计、交互式建表、生成 ETL、产出 marts 数据、提交 Airflow 天级任务，并把结果写入 Superset 创建仪表盘。

## 步骤 0：下载 quickstart 数据

DAComp **不包含**在 `datus-agent` 仓库中。本文使用一个从 DAComp Lever
示例整理出来的小型 quickstart 数据包，不需要下载完整 DAComp 压缩包。

先创建并进入工作目录：

```bash
mkdir -p ~/datus-quickstart-data
cd ~/datus-quickstart-data
```

然后直接执行下面这段 bash，会下载并解压 quickstart 数据包和本地 Docker
stack，导出 `DACOMP_HOME` / `DATUS_QUICKSTART_STACK`，最后打印两个环境变量供后续步骤使用：

```bash
curl -L -o datus-de-lever-quickstart-v2.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v2/datus-de-lever-quickstart-v2.zip
curl -L -o datus-data-engineering-quickstart-stack-v2.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v2/datus-data-engineering-quickstart-stack-v2.zip

unzip -o datus-de-lever-quickstart-v2.zip
unzip -o datus-data-engineering-quickstart-stack-v2.zip

export DACOMP_HOME="$(pwd)/datus-de-lever-quickstart"
export DATUS_QUICKSTART_STACK="$(pwd)/datus-data-engineering-quickstart-stack"
cd "$DACOMP_HOME"

echo "export DACOMP_HOME=$DACOMP_HOME"
echo "export DATUS_QUICKSTART_STACK=$DATUS_QUICKSTART_STACK"
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

下载的 stack 中已经包含本文会用到的本地 demo 服务。

启动 Superset：

```bash
cd "$DATUS_QUICKSTART_STACK/superset"
docker compose up -d
```

启动本地 lakehouse stack。Datus 和 Airflow 都会访问这套共享存储：

```bash
cd "$DATUS_QUICKSTART_STACK/lakehouse"
docker compose up -d
docker logs datus-quickstart-lakehouse-seed
```

seed 日志最后应该看到：

```text
Seeded lake.demo_raw tables: requisition, user, requisition_posting, requisition_offer
```

lakehouse 网络创建完成后，再启动 Airflow：

```bash
cd "$DATUS_QUICKSTART_STACK/airflow"
docker compose up -d
```

本地默认访问方式：

- Superset：`http://127.0.0.1:8088`，用户名 `admin`，密码 `admin`
- Airflow：`http://127.0.0.1:8080`，用户名 `admin`，密码 `admin`
- MinIO Console：`http://127.0.0.1:9001`，用户名 `admin`，密码 `password`
- Iceberg REST catalog：`http://127.0.0.1:8181`

这套 quickstart 的 Superset compose 已经带了本地演示用的元数据库和管理员默认值。

lakehouse compose 会把需要的 Lever 源表写入共享 demo raw namespace
`lake.demo_raw`。Airflow 暴露一个名为 `lakehouse_demo` 的共享连接；调度任务会用
in-memory DuckDB 连接同一个 Iceberg REST catalog，因此 Datus 和 Airflow 不再争抢本地
`.duckdb` 文件锁。

在生产或 SaaS 部署里，用能写 raw 数据的系统账号初始化 `lake.demo_raw`；Datus 和
Airflow 运行时使用另一个账号，这个账号可以读 `lake.demo_raw`，并能写非 raw 的
workspace namespace，比如 `lake.ws_lever_demo`。Datus 本身不做 namespace
特判，真正的边界由 Iceberg catalog 和存储权限负责。

如果需要重置原始 demo 表，只重跑 seed service：

```bash
cd "$DATUS_QUICKSTART_STACK/lakehouse"
docker compose up -d --force-recreate seed-demo-raw
docker logs datus-quickstart-lakehouse-seed
```

## 步骤 3：配置 `agent.yml`

把下面这段 service 配置合并到 `~/.datus/conf/agent.yml` 现有的 `agent:`
下面。保留已有的 `agent.providers` 配置；`/model` 会使用这些凭据。路径会直接使用步骤
0 里导出的 `DACOMP_HOME` 和 `DATUS_QUICKSTART_STACK` 环境变量。

```yaml
agent:
  project_name: lever_demo

  services:
    datasources:
      demo_lakehouse:
        type: duckdb
        uri: "duckdb:///:memory:"
        default: true
        iceberg:
          catalog_alias: lake
          catalog_uri: http://127.0.0.1:8181
          warehouse: s3://warehouse/
          s3_region: us-east-1
          s3_endpoint: http://127.0.0.1:9000
          s3_access_key_id: admin
          s3_secret_access_key: password
          s3_url_style: path
          authorization_type: none
          access_delegation_mode: none
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
          lakehouse_demo:
            description: Shared DuckDB Iceberg demo lakehouse
            type: duckdb_iceberg
            default: true
            capabilities:
              - sql
              - lakehouse

    semantic_layer:
      metricflow:
        type: metricflow

  agentic_nodes:
    gen_dashboard:
      bi_platform: superset
    scheduler:
      scheduler_service: airflow_prod
```

上面的 YAML 是本地 demo stack 配置；本地 Iceberg REST catalog 不启用认证。
如果连接共享公共 catalog，请换成运行账号凭据，例如 `client_id`、
`client_secret`、`oauth2_server_uri` 和
`access_delegation_mode: vended_credentials`。Airflow connection 也要配置同一个
运行账号；系统/admin 账号只用于 Datus 之外的 raw 数据初始化。

然后使用 `demo_lakehouse` datasource 启动 Datus：

```bash
cd "$DACOMP_HOME"
datus-cli --datasource demo_lakehouse
```

如果这个 workspace 之前跑过旧版 quickstart，启动前先把 `./.datus/config.yml`
里的默认 datasource 改成：

```yaml
default_datasource: demo_lakehouse
```

如果 CLI 提示还没有配置模型，继续之前先在 CLI 内运行：

```text
/model
```

选择 provider/model，并按提示填写凭据。`/model` 会把 provider 凭据写入
`~/.datus/conf/agent.yml` 的 `agent.providers`，并把当前项目使用的
provider/model 写入 `./.datus/config.yml`。

这里的 `dags_folder` 是 Datus 在主机上写入 DAG 文件的目录。Airflow compose 会把这个目录挂载到 Airflow 容器内的 `/opt/airflow/dags`，所以 Datus 生成的新 DAG 会被 Airflow 自动发现。

继续之前，先确认 Datus 能读到已经 seed 好的 lakehouse 数据：

```sql
SELECT COUNT(*) FROM lake.demo_raw.requisition;
```

quickstart 数据中，`requisition` 应该返回 `200` 行。

## 步骤 4：创建必要的 staging 表

自然语言 agent 任务不要以 `CREATE`、`COPY` 这类 SQL 动词开头；CLI 会根据这些
开头关键字判断是否直接执行 SQL。

这套 quickstart 使用：

- 共享 demo 源 namespace：`lake.demo_raw`
- 当前 workspace 输出 namespace：`lake.ws_lever_demo`

先要求 agent 创建当前 workspace 的输出 schema：

```text
Please set up the current workspace output schema lake.ws_lever_demo. Treat lake.demo_raw as read-only source data.
```

这条教程只构建一条窄但完整的依赖链：`marts_lever__requisition_enhanced`。
字段选择、字段重命名和业务逻辑以 `docs/data_contract.yaml` 为准。

再要求 agent 根据 `lever__requisition_enhanced` 和
`intermediate.int_lever__requisition_users` 的 `source_models` 创建必需的
staging 表。agent 会把任务分发到建表流程：

```text
Read ./docs/data_contract.yaml and create the staging tables needed for marts_lever__requisition_enhanced in lake.ws_lever_demo: stg_lever__requisition from lake.demo_raw.requisition, stg_lever__user from lake.demo_raw.user, stg_lever__requisition_posting from lake.demo_raw.requisition_posting, and stg_lever__requisition_offer from lake.demo_raw.requisition_offer. Use the field design and source-to-target mapping from the contract.
```

这四张 staging 表就是 requisition enhanced 示例需要的最小 raw-to-staging 输入。

## 步骤 5：生成 intermediate 和 marts 表

先生成 intermediate 表。它应该按照 `docs/data_contract.yaml` 中
`int_lever__requisition_users` 的定义，把 requisition 字段和 user 字段关联起来。

创建 intermediate 表：

```text
Read ./docs/data_contract.yaml and create lake.ws_lever_demo.int_lever__requisition_users from lake.ws_lever_demo.stg_lever__requisition and lake.ws_lever_demo.stg_lever__user. Use the contract's field design, joins, and source-to-target mapping.
```

再生成面向分析的 marts 表。契约中定义 `marts_lever__requisition_enhanced`
是一张按 `requisition_id` 一行的表，依赖：

- `lake.ws_lever_demo.int_lever__requisition_users`
- `lake.ws_lever_demo.stg_lever__requisition_posting`
- `lake.ws_lever_demo.stg_lever__requisition_offer`

创建 marts 表：

```text
Read ./docs/data_contract.yaml and create lake.ws_lever_demo.marts_lever__requisition_enhanced from lake.ws_lever_demo.int_lever__requisition_users, lake.ws_lever_demo.stg_lever__requisition_posting, and lake.ws_lever_demo.stg_lever__requisition_offer. Use the contract's business logic: keep all base requisition rows, count posting and offer links by requisition_id, fill missing counts with 0, and add has_posting and has_offer flags.
```

这条链路的基本顺序始终是：

```text
staging -> intermediate -> marts
```

生成完成后，可以直接验证 marts 表：

```sql
SELECT COUNT(*) FROM lake.ws_lever_demo.marts_lever__requisition_enhanced;
```

## 步骤 6：提交天级 Airflow 任务

现在可以要求 agent 把 marts 刷新过程提交给 scheduler。quickstart 自带的 Airflow 已经预置好了 `lakehouse_demo` 连接。

提交一个每天早上 8 点运行的 SQL 任务，刷新同一条从契约生成的链路：

```text
Submit a daily SQL job named daily_lever_requisition_enhanced that refreshes lake.ws_lever_demo.stg_lever__requisition, lake.ws_lever_demo.stg_lever__user, lake.ws_lever_demo.stg_lever__requisition_posting, lake.ws_lever_demo.stg_lever__requisition_offer, lake.ws_lever_demo.int_lever__requisition_users, and lake.ws_lever_demo.marts_lever__requisition_enhanced at 8am every day using the lakehouse_demo connection. Use the SQL generated and validated from docs/data_contract.yaml in the previous steps.
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

## 步骤 7：把 marts 表同步到 Superset serving DB

上面的 marts 表是通过 `demo_lakehouse` datasource 生成的。创建仪表盘之前，需要先把它复制到
`dataset_db.datasource_ref` 指向的 BI 注册数据库 `superset_serving`（Postgres）。
这里的 `demo_lakehouse` 和 `superset_serving` 都是 `agent.yml` 里的 Datus
datasource 名称，不是 DuckDB 或 Postgres 内部真实的 database/catalog 名。

```text
Please copy the source table lake.ws_lever_demo.marts_lever__requisition_enhanced from the demo_lakehouse datasource into the superset_serving datasource as public.lever__requisition_enhanced, replacing the target table if it already exists. Then verify the source and target row counts.
```

如果 `public.lever__requisition_enhanced` 还不存在，传输工具会根据源查询结果列自动创建目标表。

完成后，这张表就位于 Superset 通过 `bi_database_name: examples` 识别的数据库中。

## 步骤 8：创建 Superset Dashboard

当表已经存在于 `superset_serving`，就可以要求 agent 创建仪表盘：

```text
Please create a requisition operations dashboard in Superset from public.lever__requisition_enhanced. Include KPI tiles for total requisitions, open requisitions, requisitions with postings, requisitions with offers, and total requested headcount. Add charts by status, team, location, employment_status, count_postings, and count_offers.
```

数据准备是单独的 ETL / scheduler 步骤。仪表盘生成流程期望目标表或
SQL dataset 已经存在于 BI 已注册的数据库中。

## 步骤 9：验证端到端结果

走完整条链路后，你应该能确认：

- `lake.demo_raw` 已作为共享 demo 源 namespace 完成初始化
- `lake.ws_lever_demo.marts_lever__requisition_enhanced` 是从 raw 数据经 staging 和 intermediate 表逐层加工得到的
- Airflow 中能看到日常调度任务
- 仪表盘生成流程返回了 Superset dashboard URL
