# 数据开发 Demo

本指南是一个使用 Datus skills 构建数据 mart 的动手 demo。它使用
`product_adoption` 数据集，用 Pendo 功能使用事件模拟账号级产品采用度分析场景。demo
包的 `docs/` 目录中包含目标 mart 的业务需求文档。

## 教程概览

本教程展示一条由 Datus skills 驱动的端到端数据开发工作流。你会从原始 Pendo 数据出发，基于 Reference SQL 初始化项目知识，创建 ETL 实施计划，审查生成的 SQL，执行任务，并将最终 mart 与可信 expected-result 表做对账。

业务场景是产品采用度分析。产品、客户成功和增长团队需要理解客户账号如何在不同应用中使用产品功能。目标 mart 会在账号和应用粒度汇总功能使用情况，帮助下游用户识别试用、常规使用、高度采用，以及可能需要运营支持的低采用账号。

包内包含两个工作流输入：

| 输入 | 作用 |
|---|---|
| `docs/pendo_product_adoption_summary_requirements.md` | 目标产品采用度 summary mart 的业务需求。 |
| `ref_sql/` | 历史 SQL 参考，用于初始化项目知识、抽取血缘，并为实现决策提供依据。 |

主要源数据是 Pendo 功能交互数据：

| Source | 作用 |
|---|---|
| `raw.feature_event` | 功能使用事件，包括 visitor、account、application、feature、事件次数、使用分钟数和时间戳。 |
| `raw.feature_history` | 功能元数据，用于 Reference SQL 和项目知识初始化。 |
| `raw.page_history` | 页面元数据，用于 Reference SQL 和功能信息增强示例。 |

本教程要实现的目标表是：

```text
marts.pendo__product_adoption_summary
```

目标 mart 的粒度是：

```text
feature_id + account_id + app_id
```

这个 mart 会计算 total users、sessions、events、minutes、active days、average usage per session、adoption level 和 feature health score 等采用度指标。

expected-result 表已经存储在 DuckDB 中，并会在初始化时加载到 PostgreSQL：

```text
marts.pendo__product_adoption_summary_expected
```

目标是生成 `marts.pendo__product_adoption_summary`，并让它与 expected 表完全一致。

工作流概览：

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 启动 PostgreSQL | 本地数据库可用。 |
| 2 | 加载 DuckDB 数据 | `raw` 源表和 `marts` expected 表被复制到 PostgreSQL。 |
| 3 | 启动 Datus | Datus 使用已配置的模型和 datasource 就绪。 |
| 4 | 运行 `project-set-up` | 从 Reference SQL 生成项目知识文档。 |
| 5 | 运行 `etl-plan` | 创建并审批实施计划。 |
| 6 | 运行 `sql-review` | 在执行前审查并修正生成的 SQL。 |
| 7 | 运行 `execute-job` | 执行 staging 和 mart SQL 任务。 |
| 8 | 运行 `data-compare` | 将最终 mart 与 expected 表做对账。 |

完成后应满足：

- `marts.pendo__product_adoption_summary` 存在于 PostgreSQL。
- mart 有 24,995 行。
- 13 个输出字段全部匹配 `marts.pendo__product_adoption_summary_expected`。
- 数值比较在 sub-1e-9 tolerance 下通过。
- 不需要继续修正 SQL。

## 步骤 0：下载 demo 包

下载包：[product_adoption.zip](../assets/product_adoption.zip)。

解压后保持目录结构不变：

```text
product_adoption/
  README.md
  docker-compose.yml
  pendo_start.duckdb
  docs/
    pendo_product_adoption_summary_requirements.md
  docker/
    duckdb-loader/
      Dockerfile
      requirements.txt
      load_duckdb_to_postgres.py
  ref_sql/
    staging/
      stg_pendo__feature_event.sql
      stg_pendo__feature_history.sql
      stg_pendo__page_history.sql
    intermediate/
      int_pendo__latest_feature.sql
      int_pendo__latest_page.sql
      int_pendo__feature_info.sql
      int_pendo__feature_daily_metrics.sql
    marts/
      feature.sql
      feature_event.sql
      feature_daily_metrics.sql
  .datus/
    skills/
```

`docs/pendo_product_adoption_summary_requirements.md` 是计划阶段使用的业务需求文档。

## 步骤 1：启动 PostgreSQL

在 `product_adoption` 目录下启动 PostgreSQL：

```bash
cd product_adoption
docker compose up -d postgres
```

PostgreSQL 连接信息：

| 设置 | 值 |
|---|---|
| Host | `127.0.0.1` |
| Port | `5432` |
| Database | `pendo` |
| Username | `pendo` |
| Password | `pendo` |
| Default schema | `raw` |

## 步骤 2：将 DuckDB 数据加载到 PostgreSQL

执行一次性迁移：

```bash
docker compose --profile migration run --rm duckdb-loader
```

loader 会把下面两个 DuckDB schema 复制到 PostgreSQL：

```text
raw
marts
```

迁移后 expected baseline 表是：

```text
marts.pendo__product_adoption_summary_expected
```

期望行数：

```text
24995
```

## 步骤 3：启动 Datus

在同一目录下启动 Datus：

```bash
datus
```

Datus 打开后，先在界面中配置模型。

然后使用下面的值配置 datasource：

| 设置 | 值 |
|---|---|
| Datasource name | `pendo_pg` |
| Type | `PostgreSQL` |
| Host | `127.0.0.1` |
| Port | `5432` |
| Database | `pendo` |
| Username | `pendo` |
| Password | `pendo` |
| Default schema | `raw` |

## 步骤 4：初始化项目知识

使用 `project-set-up` skill 初始化项目知识库。

在 Datus 中输入：

```text
Initialize this project using skill project-set-up
```

预期输出文档：

| 文档 | 作用 |
|---|---|
| `AGENTS.md` | 项目概览、架构、核心资产索引和关键决策。 |
| `docs/business_knowledge.md` | 业务规则、强制过滤、SCD 语义、日粒度指标、首次/回访逻辑和除零保护。 |
| `docs/technical_standards.md` | full reload、schema bootstrap、时间戳解析、命名、CTE、窗口去重和 NULL 处理等 SQL 约定。 |
| `docs/table_lineage.md` | retained staging、intermediate 和 mart Reference SQL 的 DAG 与字段血缘。 |
| `docs/ref_sql_inventory.md` | 每个文件的用途、源表、目标表和 SQL 证据。 |

预期分析范围：

| 层级 | 文件 |
|---|---|
| Staging | `stg_pendo__feature_event`, `stg_pendo__feature_history`, `stg_pendo__page_history` |
| Intermediate | `int_pendo__latest_feature`, `int_pendo__latest_page`, `int_pendo__feature_info`, `int_pendo__feature_daily_metrics` |
| Marts | `feature`, `feature_event`, `feature_daily_metrics` |

需要关注的关键发现：

1. 所有 Reference SQL 都使用 full reload。
2. Reference SQL 使用 DuckDB 风格语法。
3. 最新记录逻辑使用 `ROW_NUMBER()`，按业务键分组并按 `last_updated_at` 排序。
4. 事件数据会被清洗，元数据文本基本透传。
5. 日粒度 ratio 会 round 到 3 位小数，除零返回 NULL。
6. 当时间戳相同时，previous-feature sequencing 没有额外 tie-breaker。

## 步骤 5：创建 ETL 计划

使用 `etl-plan` skill 创建实施计划。

在 Datus 中输入：

```text
Please create an ETL plan using skill etl-plan
```

预期计划：

| 项目 | 详情 |
|---|---|
| Plan file | `plans/build_product_adoption_summary.md` |
| Goal | 在 PostgreSQL 中构建 `marts.pendo__product_adoption_summary`，并与 `marts.pendo__product_adoption_summary_expected` 对账。 |
| Expected baseline | `marts.pendo__product_adoption_summary_expected` 包含 24,995 行。 |
| Out of scope | 本教程不构建 `pendo__product_adoption_analytics`。 |

预期计划任务：

| 任务 | 作用 |
|---|---|
| `jobs/stg_pendo__feature_event.sql` | 从 `raw.feature_event` 物化 `staging.stg_pendo__feature_event`。 |
| `jobs/pendo__product_adoption_summary.sql` | 构建产品采用度 summary mart。 |

审查计划后，用下面的输入批准实施：

```text
Approve, start implementation the plan
```

批准后会生成 SQL jobs。

## 步骤 6：审查生成的 SQL

执行前使用 `sql-review` skill。

在 Datus 中输入：

```text
Please review the ETL SQL using skill sql-review
```

预期审查结果：

| 项目 | 预期状态 |
|---|---|
| Modified file | `jobs/pendo__product_adoption_summary.sql` |
| Main fix | 在 `LEAST(...)` 之前增加显式 `CASE WHEN avg_events_per_session IS NULL THEN NULL`。 |
| Type check | `feature_health_score` 保持 `double precision`。 |
| Remaining low risks | 时间戳 regex guard，以及没有为无 `WHERE` filter 写 inline explanation。 |

## 步骤 7：执行 SQL jobs

审查通过后，使用 `execute-job` skill 执行任务。

在 Datus 中输入：

```text
Please execute the SQL jobs using skill execute-job
```

当需要 DDL 或 job generation 时，`execute-job` skill 可能会使用 `gen_table`、`gen_job` 等项目执行工具。

预期生成的表：

```text
staging.stg_pendo__feature_event
marts.pendo__product_adoption_summary
```

## 步骤 8：比较结果

任务完成后，使用 `data-compare` skill 将生成的 mart 与 expected 表对比。

在 Datus 中输入：

```text
Please compare the job result with the expected table using skill data-compare
```

预期对账结果：

```text
marts.pendo__product_adoption_summary reconciles perfectly with marts.pendo__product_adoption_summary_expected.
```

成功验证意味着：

- 24,995 行匹配。
- 13 个字段全部匹配。
- 数值比较在 sub-1e-9 tolerance 下通过。
- 双向 `EXCEPT` 检查通过。
- 不需要继续修正 SQL。

## Skill 参考

项目包含的 Datus skills 位于：

```text
.datus/skills/
```

| Skill | 用途 |
|---|---|
| `project-set-up` | 从 SQL、文档、血缘和业务规则初始化项目知识。 |
| `etl-plan` | 在生成 SQL 前创建并确认实施计划。 |
| `sql-review` | 基于已批准的计划审查生成的 ETL SQL。 |
| `execute-job` | 执行 SQL jobs，以及偏 DDL 的 table/job 操作。 |
| `data-compare` | 将生成结果与 expected 数据对比，并解释差异。 |

## 日常启动

环境初始化完成后，可以用下面的命令启动 PostgreSQL 和 Datus：

```bash
cd product_adoption
docker compose up -d postgres
datus
```

DuckDB 文件只在初始化或完整重建时需要。
