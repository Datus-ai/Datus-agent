# BI 平台配置

BI 平台连接配置在 `agent.services.bi_platforms` 下。

## 结构

承载 BI 平台读、Datus 写的 serving DB 注册为**普通 Datus datasource**，BI
平台条目通过 `dataset_db.datasource_ref` 引用它，连接池、schema 元数据、凭据
全部与 Datus 其它部分共享。

```yaml
agent:
  services:
    datasources:
      # 现有源数仓（BI 侧只读）
      src_warehouse:
        type: starrocks
        host: ${SRC_WAREHOUSE_HOST}
        port: 9030
        username: ${SRC_WAREHOUSE_USER}
        password: ${SRC_WAREHOUSE_PASSWORD}
        database: warehouse

      # serving DB —— Datus 写入、BI 平台读取
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
          datasource_ref: serving_pg          # ← 引用 services.datasources.serving_pg
          bi_database_name: examples          # Superset Settings > Database Connections 里看到的名字

      grafana:
        type: grafana
        api_base_url: http://localhost:3000
        api_key: ${GRAFANA_API_KEY}
        dataset_db:
          datasource_ref: serving_pg
          bi_database_name: PostgreSQL

  agentic_nodes:
    gen_dashboard:
      bi_platform: superset
```

## `dataset_db` 字段

`dataset_db` 是叠在 Datus datasource 之上的 BI 专属层，只承载 BI 相关字段：

| 字段 | 必填 | 说明 |
|------|------|------|
| `datasource_ref` | 是 | `services.datasources` 中某个条目的名字。Datus 用它进行 schema 内省与写入。 |
| `bi_database_name` | 推荐 | 同一份 DB 在 BI 平台内部的别名。`gen_dashboard` 通过 `list_bi_databases()` 匹配它来解出 `database_id`，给 `create_dataset` 用。 |

旧的 inline 写法（`dataset_db: {uri: "..."}` 或 `dataset_db: {type: ..., host: ..., ...}`）已经不再受支持，把连接字段挪到
`services.datasources` 后用 ref 引用即可。

## 选择规则

- `bi_platform` 决定使用 `services.bi_platforms` 中的哪一条。
- 配置 key 应与平台名一致（`superset`、`grafana`、…）。
- 若 `bi_platform` 留空，Datus 会先读 `./.datus/config.yml` 的 `dashboard:`
  字段（项目级别 pin），再退回到「唯一条目自动选」规则。
- 若配置了多个 BI 平台、且没有项目级 pin，调用处必须显式传 `bi_platform`。

## 通过 CLI 配置（`/services`）

在 Datus REPL 内执行 `/services` 即可直接进入交互式 TUI（默认 Dashboard
tab；`/services scheduler` 直接落到 Scheduler tab；`/services list` 退回
到原先的只读列表）：

- 列表最后一行 `+ Add new dashboard`，按 `Enter` 进入新增流程。选择 `type`
  时若对应 adapter 包（`datus-bi-superset` / `datus-bi-grafana` …）尚未安
  装，Datus 会自动 `pip install` 并热加载 registry，**无需重启**。
- `e` 编辑凭据，`x` 删除条目，`t` 触发一次连通性 probe。
- `p` 把当前光标项设为项目级 default。pin 会写到 `./.datus/config.yml` 的
  `dashboard: <name>` 字段，仅对当前项目生效，优先级高于「自动唯一项」规则。
  在已 pin 的行上再按一次 `p` 清除。

service 定义会写入 `~/.datus/conf/agent.yml`，跨项目共享凭据；只有 active
选择属于项目级。

## 所有权

仪表盘创建被拆成三步：

1. `gen_job` 或 `scheduler` 在 `dataset_db.datasource_ref` 对应的 serving DB 里
   准备 / 刷新数据。
2. `gen_dashboard` 基于 BI 已注册数据库中的现成表 / SQL dataset 创建 dataset /
   chart / dashboard。
3. `bi-validation` 通过 `ValidationHook.on_end` 自动跑创建后的校验。

源 DB 凭据不会泄露给 Superset / Grafana —— BI 端只看见以 `bi_database_name`
注册的 serving DB。

## 注意

- `services.bi_platforms` 是 BI 凭据唯一的运行时来源。
- 顶层 `dashboard:` 字段在运行时不再被读取。
- `services.bi_platforms.<x>.dataset_db.datasource_ref` 必须指向一个已存在的
  `services.datasources.<datasource_ref>` 条目；Datus 在启动时会校验，野指针
  会直接报错。
