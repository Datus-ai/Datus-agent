# 端到端数据工程

本场景教程使用开源的 DAComp 数据工程数据集，串起一条完整的本地 Datus
工作流：理解数仓分层设计、在本地 DuckDB workbench 文件中交互式建表、
生成 ETL、产出 marts 数据、提交 Airflow 天级任务，并把结果写入 Superset
创建仪表盘。

!!! info "本教程从哪里开始"
    本教程从源数据开始，新建数据管道和 Dashboard。如果你已有 Superset Dashboard，希望把它转换成分析 Agent，请阅读[将 Dashboard 变成 Copilot](dashboard_copilot.zh.md)。第一次使用 Datus 时，建议先完成[安装并完成第一次提问](Quickstart.zh.md)。

本文通过 Datus plugin 使用 Airflow 和 Superset。Datus datasource 负责 SQL
执行与数据传输，plugin 则通过 Airflow 和 Superset API 发现、创建、运行和
检查资源。

本地开源 quickstart **不需要** Iceberg、MinIO 或 S3。SaaS Studio tour
使用托管的 DuckDB + Iceberg lakehouse；对应的 namespace 模型见文末
「SaaS Studio Tour 变体」。

## 步骤 0：下载 quickstart 数据

DAComp **不包含**在 `datus-agent` 仓库中。本文使用一个从 DAComp Lever
示例整理出来的小型 quickstart 数据包，不需要下载完整 DAComp 压缩包。

先创建并进入工作目录：

```bash
mkdir -p ~/datus-quickstart-data
cd ~/datus-quickstart-data
```

然后直接执行下面这段 bash，会下载并解压 quickstart 数据包和本地 Docker
stack，创建可写的 DuckDB workbench，导出 `DACOMP_HOME` /
`DATUS_QUICKSTART_STACK`，最后打印两个环境变量供后续步骤使用：

```bash
curl -L -o datus-de-lever-quickstart-v1.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v1/datus-de-lever-quickstart-v1.zip
curl -L -o datus-data-engineering-quickstart-stack-v1.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v1/datus-data-engineering-quickstart-stack-v1.zip

unzip -o datus-de-lever-quickstart-v1.zip
unzip -o datus-data-engineering-quickstart-stack-v1.zip

export DACOMP_HOME="$(pwd)/datus-de-lever-quickstart"
export DATUS_QUICKSTART_STACK="$(pwd)/datus-data-engineering-quickstart-stack"
cp "$DACOMP_HOME/lever_start.duckdb" "$DACOMP_HOME/lever_workbench.duckdb"
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

在开始写 DDL 和 ETL 之前，先把这两份文件过一遍，后面给 agent
的提示词就能更贴近原始设计。

## 步骤 2：启动本地 quickstart 环境

下载的 stack 中已经包含本文会用到的本地 demo 服务。

Superset 中名为 `examples` 的 Database 使用
`postgres:5432/superset_examples` 连接 PostgreSQL。Superset plugin 会根据
这个不包含凭据的连接标识解析对应的 Datus datasource。启动 Superset 前，
需要把同一个 endpoint 暴露给主机，并确保 Compose service 名称可以在主机
上解析：

```bash
cd "$DATUS_QUICKSTART_STACK/superset"

cat > docker-compose.override.yml <<'YAML'
services:
  postgres:
    ports:
      - "5432:5432"
YAML

if grep -qE '(^|[[:space:]])postgres([[:space:]]|$)' /etc/hosts && \
   ! grep -qxE '[[:space:]]*127\.0\.0\.1[[:space:]]+postgres[[:space:]]*' /etc/hosts; then
  echo 'Conflicting /etc/hosts entry for postgres; replace it with: 127.0.0.1 postgres' >&2
  exit 1
fi
grep -qxE '[[:space:]]*127\.0\.0\.1[[:space:]]+postgres[[:space:]]*' /etc/hosts || \
  echo '127.0.0.1 postgres' | sudo tee -a /etc/hosts

docker compose up -d
```

主机的 5432 端口必须可用。这个流程使用 `postgres:5432`，从而让 Datus
和 Superset 返回相同的连接标识。

启动 Airflow：

```bash
cd "$DATUS_QUICKSTART_STACK/airflow"
docker compose up -d
```

本地默认访问方式：

- Superset：`http://127.0.0.1:8088`，用户名 `admin`，密码 `admin`
- Airflow：`http://127.0.0.1:8080`，用户名 `admin`，密码 `admin`
- PostgreSQL serving database：`postgres:5432/superset_examples`，用户名/密码
  为 `superset/superset`

这套 quickstart 的 Superset compose 已经带了本地演示用的元数据库和管理员默认值。

Airflow compose 会把 `${DACOMP_HOME}` 挂载到容器中，并暴露一个名为
`duckdb_dacomp_lever` 的 Airflow connection，指向
`/workspace/lever_workbench.duckdb`。

即使这些本地 demo 凭据是公开默认值，也不要把它们直接写进 `agent.yml`。
请在每个运行 Datus 的 shell 中导出：

```bash
export AIRFLOW_PASSWORD=admin
export SUPERSET_PASSWORD=admin
export SUPERSET_PG_PASSWORD=superset
```

## 步骤 3：安装并配置 plugin

把两个已发布 plugin 安装到 Datus 所在的同一个环境中。裸包名默认使用
pip/PyPI 安装来源，不要添加 `pip:` 前缀。Superset plugin 内置了后续步骤
使用的 `superset-dashboard-authoring` skill：

```bash
datus plugin install datus-airflow-plugin
datus plugin install datus-superset-plugin
datus plugin info airflow
datus plugin info superset
```

如需用最新发布版本替换已有安装，请添加 `--force`：

```bash
datus plugin install datus-airflow-plugin --force
datus plugin install datus-superset-plugin --force
```

把下面这段配置合并到 `~/.datus/conf/agent.yml` 现有的 `agent:`
下面。保留已有的 `agent.providers` 配置；`/model` 会使用这些凭据。路径会直接使用步骤
0 里导出的 `DACOMP_HOME` 和 `DATUS_QUICKSTART_STACK` 环境变量。

```yaml
agent:
  filesystem:
    allow_write:
      - "${DATUS_QUICKSTART_STACK}/airflow/dags"

  services:
    datasources:
      lever_duckdb:
        type: duckdb
        uri: "duckdb:///${DACOMP_HOME}/lever_workbench.duckdb"
        default: true
      superset_serving:
        type: postgresql
        host: postgres
        port: 5432
        database: superset_examples
        schema: public
        username: superset
        password: ${SUPERSET_PG_PASSWORD}

  plugins:
    airflow:
      local:
        default: true
        api_base_url: http://127.0.0.1:8080/api/v1
        api_version: auto
        username: admin
        password: ${AIRFLOW_PASSWORD}
        verify_ssl: true
        timeout: 30
        dags_folder: "${DATUS_QUICKSTART_STACK}/airflow/dags"
        dag_id_prefix: daily_lever_
        allow_commands: dags,tasks,version,health

    superset:
      local:
        default: true
        api_base_url: http://127.0.0.1:8088
        auth_mode: login
        username: admin
        password: ${SUPERSET_PASSWORD}
        provider: db
        verify_ssl: "true"
        timeout: "30"
```

`filesystem.allow_write` 允许 agent 把 DAG 发布到 Airflow 挂载的主机目录。
`dags_folder` 告诉 agent 应把运行副本发布到哪里。DAG 发现、触发、运行状态
检查和日志读取由主 agent 通过 Airflow plugin 完成。

先为当前项目启用两个 profile，然后启动聊天会话：

```bash
cd "$DACOMP_HOME"
datus plugin enable airflow --profile local
datus plugin enable superset --profile local
datus --datasource lever_duckdb
```

不要自行运行具体 plugin 命令，让主 agent 验证两个服务：

```text
Using the enabled local profiles, ask the Airflow plugin for its server version and health, then ask the Superset plugin for health and the available databases. Perform read-only checks only and report any connectivity or authentication error.
```

始终在启动 Datus 前完成 plugin 配置和启用。plugin skill 和环境上下文会在
session 启动时准备好；修改 profile 后请重启 session。这里选择的
`lever_duckdb` datasource 指向可写的 workbench 文件。

quickstart 通过 Airflow 的 `AIRFLOW_CONN_DUCKDB_DACOMP_LEVER` 环境变量注入
`duckdb_dacomp_lever`。task 运行时可以通过 `BaseHook` 读取环境变量 connection，
但 Airflow REST connection endpoint 不会返回它。步骤 6 会通过实际运行 DAG
验证这个 connection。

如果 CLI 提示还没有配置模型，继续之前先在 CLI 内运行：

```text
/model
```

选择 provider/model，并按提示填写凭据。`/model` 会把 provider 凭据写入
`~/.datus/conf/agent.yml` 的 `agent.providers`，并把当前项目使用的
provider/model 写入 `./.datus/config.yml`。

## 步骤 4：创建必要的 staging 表

自然语言 agent 任务不要以 `CREATE`、`COPY` 这类 SQL 动词开头；CLI 会根据这些
开头关键字判断是否直接执行 SQL。

先要求 agent 创建目标 schema：

```text
Please set up the target schemas staging, intermediate, and marts in the current DuckDB database. Keep the existing raw schema unchanged.
```

这条教程只构建一条窄但完整的依赖链：`marts.lever__requisition_enhanced`。
字段选择、字段重命名和业务逻辑以 `docs/data_contract.yaml` 为准。

先要求 agent 检查物理源表字段，避免把 source-to-target 重命名误判为字段缺失：

```text
Inspect the schemas and sample rows for raw.requisition, raw.user, raw.requisition_posting, and raw.requisition_offer. Before generating SQL, confirm these source-to-target renames from the physical columns: raw.requisition.id to requisition_id, name to requisition_name, creator_id to creator_user_id, owner_id to owner_user_id, and hiring_manager_id to hiring_manager_user_id; raw.user.id to user_id, name to user_name, and external_directory_id to external_directory_user_id. Do not create NULL placeholders for columns that exist in the source tables.
```

再要求 agent 根据 `lever__requisition_enhanced` 和
`intermediate.int_lever__requisition_users` 的 `source_models` 创建必需的
staging 表。agent 会把任务分发到建表流程：

```text
Read ./docs/data_contract.yaml and create the staging tables needed for marts.lever__requisition_enhanced: staging.stg_lever__requisition from raw.requisition, staging.stg_lever__user from raw.user, staging.stg_lever__requisition_posting from raw.requisition_posting, and staging.stg_lever__requisition_offer from raw.requisition_offer. Use the field design and source-to-target mapping from the contract.
```

这四张 staging 表就是 requisition enhanced 示例需要的最小 raw-to-staging 输入。

## 步骤 5：生成 intermediate 和 marts 表

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

生成完成后，验证每一层以及仪表盘所需的维度：

```sql
SELECT 'stg_user' AS model, COUNT(*) AS row_count FROM staging.stg_lever__user
UNION ALL
SELECT 'stg_requisition', COUNT(*) FROM staging.stg_lever__requisition
UNION ALL
SELECT 'stg_requisition_posting', COUNT(*) FROM staging.stg_lever__requisition_posting
UNION ALL
SELECT 'stg_requisition_offer', COUNT(*) FROM staging.stg_lever__requisition_offer
UNION ALL
SELECT 'int_requisition_users', COUNT(*) FROM intermediate.int_lever__requisition_users
UNION ALL
SELECT 'marts_requisition_enhanced', COUNT(*) FROM marts.lever__requisition_enhanced;

SELECT
  COUNT(*) AS total_rows,
  COUNT(status) AS rows_with_status,
  COUNT(team) AS rows_with_team,
  COUNT(location) AS rows_with_location,
  SUM(count_postings) AS posting_links,
  SUM(count_offers) AS offer_links
FROM marts.lever__requisition_enhanced;
```

每个 model 都必须非空，`rows_with_status`、`rows_with_team`、
`rows_with_location`、`posting_links` 和 `offer_links` 都必须大于 0。使用
version 1 quickstart 数据包时，marts 表应有 146 行。如果维度意外全部为
NULL，请返回前面的 schema 检查，修正源字段映射后再继续。

保存并验证用于刷新同一条契约生成链路的 SQL；每天早上 8 点的调度将在步骤 6 中创建：

```text
Collect the exact SQL statements that successfully created the staging, intermediate, and marts schemas and built the four staging tables, intermediate.int_lever__requisition_users, and marts.lever__requisition_enhanced. Keep them in dependency order and write them to ./jobs/daily_lever_requisition_enhanced.sql. Do not replace validated statements with newly invented SQL. Execute the saved file once against lever_duckdb and confirm it reproduces the same non-zero validation results.
```

## 步骤 6：发布并运行天级 Airflow DAG

Airflow plugin 可以查询 DAG、检查源码和导入错误、触发运行，并读取 run
状态、task 状态和日志。agent 通过 filesystem 工具发布新 DAG，再使用 plugin
完成验证和运行。

对于本地 stack，发布就是把生成的文件写入 allowlist 中、并挂载到
`/opt/airflow/dags` 的主机目录。要求 agent 编写、发布并验证 DAG：

```text
Use the Airflow plugin with profile local and follow its airflow skill. Create ./dags/daily_lever_requisition_enhanced.py for DAG id daily_lever_requisition_enhanced with schedule 0 8 * * *, catchup disabled, and a fixed timezone-aware start date. At runtime, read /workspace/jobs/daily_lever_requisition_enhanced.sql, resolve the duckdb_dacomp_lever Airflow connection with BaseHook, reconstruct the DuckDB SQLAlchemy URL from the connection schema or host, and execute the validated SQL inside a committed transaction. Keep the project source file, then use the filesystem tools to write identical content to the local profile's configured dags_folder. Confirm the two files are identical. Wait until the Airflow plugin reports the DAG, check import errors and DAG details, then trigger it once and wait for completion. After the wait finishes, read the latest run again and show the final dag_run_id and state. If it fails, inspect task states and logs before reporting the error.
```

发布和触发操作可能需要确认。同一个 agent prompt 已包含必要的回查；如需重复
检查，请让主 agent 通过 Airflow plugin 列出匹配 DAG、import errors、DAG 详情
和最新 run。

你应该会看到：

- 维护中的源码位于 `$DACOMP_HOME/dags/daily_lever_requisition_enhanced.py`
- `${DATUS_QUICKSTART_STACK}/airflow/dags` 下出现内容完全相同的运行副本
- 同一个文件在 Airflow 容器内显示为
  `/opt/airflow/dags/daily_lever_requisition_enhanced.py`
- Airflow 返回 `dag_id`、成功的 `dag_run_id` 和运行状态

## 步骤 7：把 marts 表同步到 Superset serving DB

上面的 marts 表是通过 `lever_duckdb` datasource 生成的。创建仪表盘之前，需要先把它复制到
`superset_serving` Postgres datasource。
这里的 `lever_duckdb` 和 `superset_serving` 都是 `agent.yml` 里的 Datus
datasource 名称，不是 DuckDB 或 Postgres 内部真实的 database/catalog 名。

```text
Please copy the source table marts.lever__requisition_enhanced from the lever_duckdb datasource into the superset_serving datasource as public.lever__requisition_enhanced, replacing the target table if it already exists. Then verify the source and target row counts.
```

如果 `public.lever__requisition_enhanced` 还不存在，传输工具会根据源查询结果列
自动创建目标表。version 1 数据包的源表和目标表都应该返回 146 行。

完成后，这张表就位于 Superset 中注册为 `examples` 的 PostgreSQL 数据库。
两边都使用 `postgres:5432/superset_examples` 标识数据库，因此 plugin 可以
把它唯一解析到 `superset_serving` Datus datasource。

## 步骤 8：通过 plugin 创建 Superset Dashboard

当表已经存在于 `superset_serving`，要求 agent 使用 plugin 的 authoring
skill：

```text
Use the Superset plugin with profile local and follow the superset-dashboard-authoring skill. Discover the Superset Database named examples and resolve its credential-free connection identity uniquely to the superset_serving Datus datasource. Validate public.lever__requisition_enhanced and the planned queries on that Datus datasource first. Register it as a physical Superset Dataset, then create a requisition operations dashboard with KPI tiles for total requisitions, open requisitions, requisitions with postings, requisitions with offers, and total requested headcount. Add charts by status, team, location, employment_status, count_postings, and count_offers. Store only non-sensitive Database, Dataset, Dashboard, and Chart resource request payloads in project-local JSON files. Never persist authentication or login request bodies, tokens, cookies, passwords, or other secrets, and redact sensitive fields before writing any payload. Every chart must contain matching params and query_context JSON strings. Attach all charts and update a complete position_json layout so the dashboard is not blank. Read the Database, Dataset, Dashboard, and Charts back, confirm that the Database connection still identifies postgres:5432/superset_examples, and run representative chart data queries. Return the Database, Dataset, Dashboard, and Chart IDs plus the dashboard URL.
```

数据准备是单独的 ETL / 调度步骤。创建仪表盘前，目标表或 SQL dataset
必须已经存在于 Superset 所识别的数据库中。Superset 的创建、更新和查询
操作会根据 plugin 权限规则要求确认。

同一个 agent prompt 会回读新建资源并执行代表性 chart 查询。如需重复验证，请把
返回的 ID 交给主 agent，让它通过 Superset plugin 检查 Database、table metadata、
Dashboard、Charts 和 chart data；不要复制本文中的示例 ID。

仪表盘应该包含 11 个 chart。代表性的 total requisitions chart 查询应返回
146，分类查询应返回多个分组，Database connection 应标识为
`postgres:5432/superset_examples`。

## 步骤 9：验证端到端结果

走完整条链路后，你应该能确认：

- `lever_workbench.duckdb` 中已经有 `staging`、`intermediate` 和 `marts` schema
- `marts.lever__requisition_enhanced` 是从 raw 数据经 staging 和 intermediate 层逐层加工得到的
- `$DACOMP_HOME/jobs` 和 `$DACOMP_HOME/dags` 中保存了已验证 SQL 和维护中的 DAG 源码
- Airflow UI 中能看到成功运行的天级 DAG
- 获得 Superset Database、Dataset、Dashboard、Chart ID 以及 dashboard URL

## SaaS Studio Tour 变体

托管的 SaaS tour 使用同一条 Lever 工作流，但不使用本地
`lever_workbench.duckdb` 文件。平台会提供共享的 DuckDB + Iceberg lakehouse：

- 共享只读 raw namespace：`lake.demo_raw`
- 每个 workspace 独立可写 namespace：`lake.ws_<workspace_id>`
- SaaS Airflow connection：`duckdb_lever_workbench`

托管平台会提供受管理的 Airflow/Superset plugin profile 和自己的 DAG 部署
通道。不要把本地的 `filesystem.allow_write` 或 Compose DAG 挂载路径带入
SaaS。plugin 操作仍负责发现、触发和验证托管资源；下面的 namespace 规则
保持不变。

每个用户都应该在独立 workspace 中运行 tour。backend 会按当前 workspace
渲染 seed 进去的 `docs/data_contract.yaml`，所以输出会写到
`lake.ws_<workspace_id>`，源数据继续来自 `lake.demo_raw`。prompt 和 SQL
应该使用完整限定名，例如：

```text
lake.demo_raw.requisition
lake.ws_<workspace_id>.stg_lever__requisition
lake.ws_<workspace_id>.int_lever__requisition_users
lake.ws_<workspace_id>.marts_lever__requisition_enhanced
```

SaaS tour 中不要使用 `raw.*`、`staging.*`、`intermediate.*`、`marts.*`
这类未限定的物理 schema 名。它们只表示逻辑层级；真实可写边界是 workspace
namespace。

workspace namespace 发生变化时，需要重建 demo project 并重新生成 DAG，
确保 DAG 使用当前的 `lake.ws_<workspace_id>` namespace。

## 后续步骤

- [将 Dashboard 变成 Copilot](dashboard_copilot.zh.md) —— 从已有 Superset Dashboard 构建分析子代理。
- [构建上下文增强 Agent](contextual_data_engineering.zh.md) —— 构建可复用上下文，并比较回答质量。
- [Plugin](../plugin/introduction.zh.md) —— 配置 Airflow、Superset 和其他集成。
- [选择上手路径](index.md) —— 对比所有入门指南。
