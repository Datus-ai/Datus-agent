# 调度器配置

调度器服务配置在 `agent.services.schedulers` 下。

## 结构

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
          starrocks_default: StarRocks production

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

`get_scheduler_config(service_name)` 按下列顺序解析活动 scheduler：

1. 调用处显式传入的 `service_name`。
2. `./.datus/config.yml` 顶层 `scheduler:` 字段（项目级 pin）。
3. 全局 `default: true` 标记（在 `services.schedulers` 中必须唯一）。
4. 唯一条目兜底：只配置了一个 scheduler 时直接使用。

若项目级 pin 已失效（指向一个 agent.yml 里不再存在的 service），Datus 会打
warning 并退回到全局 default。

## 通过 CLI 配置（`/services`）

在 Datus REPL 内执行 `/services scheduler` 直接进入交互式 TUI 的 Scheduler
tab（裸 `/services` 落在 Dashboard tab；`/services list` 退回只读列表）：

- 列表最后一行 `+ Add new scheduler`，按 `Enter` 进入新增流程。当前仅
  `airflow`（`datus-scheduler-airflow`）受支持；若 adapter 包尚未安装，
  Datus 会自动 `pip install` 并热加载 registry，**无需重启**。
- `e` 编辑凭据，`x` 删除条目，`t` 触发一次连通性 probe。
- `d` 切换**全局** `default: true`：按 `d` 把光标项设为 workspace 级
  default 并自动清掉其他条目的该字段，避免出现「多 default」。
- `p` 设置**项目级** default：值写入 `./.datus/config.yml` 的
  `scheduler: <name>`，只对当前项目生效，优先级高于全局标记。在已 pin 的
  行上再按一次 `p` 清除。

service 定义会写入 `~/.datus/conf/agent.yml`，跨项目共享；只有 active 选择
属于项目级。

首次进入交互式 REPL 时,Datus 会对该 section 跑一遍 bootstrap:若尚无项目级 pin,而能从 YAML 中解析出明确的默认值(单条快捷或唯一标 `default: true` 的条目),Datus 会自动写入项目级 pin。若配置了多条但都未标 default,启动时会弹出一个轻量选择器。CI / Docker 等无人值守环境可设置 `DATUS_DISABLE_SERVICE_BOOTSTRAP=1` 关闭。

## 注意

- `services.schedulers` 是 scheduler 配置唯一的运行时来源。
- 顶层 `scheduler:` 字段在运行时不再被读取。
