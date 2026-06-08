# 指标生成与 Bootstrap KB QA 测试说明

本文档用于 QA 验证当前分支的指标生成、批量 `bootstrap-kb`、MetricFlow YAML 合并、Datasource 行级隔离和失败回滚行为。

## 当前实现状态

本次实现聚焦 `gen_metrics` 和 `bootstrap-kb --components metrics` 的稳定性与可测性，目标是让普通聚合 SQL、复杂 SQL、已有指标复用和不可发布指标都能被正确处理。

当前 `gen_metrics` 使用配置中的 `deepseek` 模型，配置位置为 `conf/agent.yml`：

```yaml
nodes:
  gen_metrics:
    model: deepseek
```

`deepseek` 的默认底层模型由 `conf/providers.yml` 决定，当前为 `deepseek-v4-pro`。

## 主要改动范围

### 1. SQL 指标候选抽取

`analyze_metric_candidates_from_history` 现在会把 SQL 输出拆成更明确的几类：

- `direct_metric_candidates`：可直接建模的业务指标，例如 `COUNT(DISTINCT ac_code)`、`AVG(sr_value)`、`MAX(sr_value)`。
- `base_measures`：指标依赖的基础 measure。
- `support_measure_candidates`：只作为辅助口径的 measure，不应发布成顶层 metric。
- `derived_metric_candidates`：基于已有 metric 的二阶段指标，例如 MoM delta。
- `identity_metric_references`：SQL 只是引用已有指标，不应重复生成。
- `non_metric_evidence`：明细查询、过滤查询、Top-N/ROW_NUMBER 查询等非指标证据。
- `derived_datasource_recommendations`：窗口函数、排名、复杂子查询等需要先建 derived data source 的场景。

关键行为：

- 普通聚合 SQL 仍按指标抽取。
- `COUNT(*)` 单独出现时仍可作为直接指标。
- 如果同一 SELECT 中同时有 `COUNT(*)` 和更明确的 `COUNT(DISTINCT ...)` 业务口径，`COUNT(*)` 会被降级为 support measure，避免生成 `product_row_count` 这类低价值指标。
- `ROW_NUMBER()` / Top-N 明细查询不会硬套成普通指标，会归类为明细/derived datasource 场景。

### 2. MetricFlow YAML 写入与合并

新增了面向 MetricFlow YAML 的文件工具逻辑：

- 写入已有 semantic model YAML 时，合并 `measures` / `dimensions` / `identifiers`，避免后续 batch 覆盖前面 batch 已写入的内容。
- 写入已有 metric YAML 时，按 metric name 合并或追加，避免最后一个 batch 重新生成文件并丢失已有 metric。
- 对 metric 文件中的 `subject_tree` tag 做 datasource 路径规范化。
- YAML 单引号转义修复改为按 YAML 报错位置局部修复，避免全局替换破坏合法内容。
- 严格模式下，文件读写会先做路径策略检查，再读取已有文件进行合并。

### 3. 批量指标生成与 KB 同步

`bootstrap-kb --components metrics` 现在按 batch 处理 SQL 后会刷新上下文：

- 已生成的 metric 会进入后续 batch 的 existing metric catalog，后续 batch 不再重复生成。
- `end_metric_generation` 根据本批新增 metric 作用域同步 KB，不再把同一个 metric 文件里已有 metric 重复计入。
- combined dry-run SQL 不再被错误归属给所有 metric；如果一条 dry-run 覆盖多个新 metric，会按当前新增 metric 范围同步。
- 最终 `metrics_count` 代表本次 datasource 下最终同步成功的唯一 metric 数，而不是每个 batch 的重复累计数。

### 4. Datasource 行级隔离

当前实现不再通过修改向量库目录来隔离 datasource，而是在同一 project 物理存储中增加行级 scope：

- scoped 行增加 `datasource_id`。
- scoped 行增加内部 upsert key `storage_key`，格式为 `{datasource_id}:{business_id}`。
- 覆盖更新时按 datasource 删除对应行，不 drop 整张 project 表。
- 没有 datasource 的 datasource-scoped storage 会 fail fast，避免写入 project 全局命名空间。
- Explorer 等无 datasource 入口做了降级处理，避免因为未绑定 datasource 直接 500。
- 旧数据兼容通过 best-effort scope 列补齐处理：旧行默认 `datasource_id=""`，`storage_key="legacy:{id}"`。

## QA 环境准备

### 必要配置

确认 `conf/agent.yml` 中有可用的 StarRocks datasource，并且 `gen_metrics` 使用 `deepseek`：

```bash
rg -n "gen_metrics|model: deepseek|starrocks" conf/agent.yml conf/providers.yml
```

确认 StarRocks 中测试库和测试表已准备好。本地验证使用：

- datasource：`starrocks`
- database：`ac_manage`
- table：`ac_manage.v_udata_ac_info`
- time spine：`ac_manage.mf_time_spine`
- success story：`seed_context.csv`

如果 QA 环境没有 `seed_context.csv`，请使用测试包随附的同等 CSV；该文件需要覆盖以下三类 SQL：

- 普通聚合 SQL：COUNT DISTINCT、条件计数、AVG、MAX、ratio。
- 复杂 SQL：ROW_NUMBER / Top-N / 明细查询。
- 已有 metric 复用：已有 `activity_count` 基础上识别 ratio、support measure、MoM derived candidate。

### 建议清理

为了保证从空 YAML 状态验证合并逻辑，运行前可清理当前 datasource 的本地 YAML：

```bash
rm -rf subject/semantic_models/starrocks
```

如果要验证 overwrite 的 storage 行级删除，请不要删除整个 `~/.datus/data/.../datus_db`，否则无法覆盖“同一 project 表内按 datasource 删除”的场景。

## 主流程测试

### 测试 1：指标 bootstrap 全流程

执行：

```bash
uv run datus-agent bootstrap-kb \
  --datasource starrocks \
  --components metrics \
  --kb_update_strategy overwrite \
  --success_story seed_context.csv \
  -y
```

预期结果：

```text
Final Result: {'status': 'success', 'message': 'metrics bootstrap completed, metrics_count=12', 'error': ''}
```

预期日志特征：

- `Processing 28 SQL queries in 6 batch(es)`
- `Batch 1/6 completed successfully` 到 `Batch 6/6 completed successfully`
- `Metrics extraction completed: 6/6 batch(es) succeeded`
- 不应出现最终失败：
  - `All batch(es) failed`
  - `No valid objects found to sync`
  - `Metrics extraction completed but produced no output`

### 测试 2：最终 YAML 内容

执行：

```bash
uv run python - <<'PY'
import glob, yaml

total = 0
for path in sorted(glob.glob("subject/semantic_models/starrocks/**/*.yml", recursive=True)):
    docs = list(yaml.safe_load_all(open(path)))
    metrics = []
    measures = []
    for doc in docs:
        if isinstance(doc, dict) and isinstance(doc.get("metric"), dict):
            metrics.append(doc["metric"].get("name"))
        if isinstance(doc, dict) and isinstance(doc.get("data_source"), dict):
            measures.extend([m.get("name") for m in doc["data_source"].get("measures") or []])
    if metrics or measures:
        print(path)
        print("metrics_count:", len(metrics), metrics)
        print("measures_count:", len(measures), measures)
        total += len(metrics)
print("TOTAL_METRICS", total)
PY
```

预期最终 metric 共 12 个：

```text
activity_count
avg_sr_value
avg_first_day_sr
delivery_activity_count
high_sr_activity_count
new_product_activity_count
new_product_activity_ratio
new_ip_activity_count
delivery_activity_ratio
major_activity_count
max_sr_value
avg_activity_days
```

预期 semantic model measures 共 10 个：

```text
activity_count
avg_sr_value
avg_first_day_sr
delivery_activity_count
high_sr_activity_count
new_product_activity_count
new_ip_activity_count
major_activity_count
max_sr_value
avg_activity_days
```

以下内容不应出现在最终 YAML 中：

```text
product_row_count
activity_count_mom_delta
```

### 测试 3：文件合并不覆盖

重点观察批次 4、5、6 后的最终文件：

- 批次 4 生成的 `avg_sr_value`、`avg_first_day_sr`、`high_sr_activity_count`、`new_product_activity_ratio` 应保留到最终文件。
- 批次 5 追加的 `new_ip_activity_count`、`delivery_activity_ratio`、`major_activity_count`、`max_sr_value`、`avg_activity_days` 应保留。
- 批次 6 不应因为 glob 为空或重新 `write_file` 覆盖 `v_udata_ac_info.yml` / `v_udata_ac_info_metrics.yml`。

可用命令：

```bash
rg -n "Metric file merged successfully|Semantic model merged successfully|File written successfully" ~/.datus/logs/agent.$(date +%F).log
```

预期：

- 已存在文件的后续写入应出现 `Metric file merged successfully` 或 `Semantic model merged successfully`。
- 最终 metric 文件仍为 12 个 metric block。

### 测试 4：support measure 不发布

测试 SQL 中包含类似：

```sql
SELECT
  COUNT(*) AS product_row_count,
  COUNT(DISTINCT ac_code) AS activity_count
FROM ac_manage.v_udata_ac_info
...
```

预期：

- `activity_count` 被识别为业务指标，且如果已存在则复用。
- `product_row_count` 被识别为 `support_measure_candidates`。
- 不生成 `metric: product_row_count`。
- 如果没有指标依赖它，也不需要把它追加到 semantic model measures。

### 测试 5：ROW_NUMBER / Top-N SQL 不硬套指标

测试 SQL 中包含 `ROW_NUMBER() OVER (...)` 或排名过滤，例如每个渠道 SR 最高的活动。

预期：

- 不生成直接 metric。
- 日志或模型输出应说明这是 detail ranking / derived datasource 场景。
- 该 SQL 不应导致 `max_sr_value` 之外的错误 Top-N 指标被发布。

### 测试 6：StarRocks MoM derived metric 失败回滚

测试 SQL 中包含月环比逻辑，例如：

```sql
activity_count - LAG(activity_count) OVER (ORDER BY month)
```

当前预期行为：

- 系统可以识别 `activity_count_mom_delta` 是 derived metric candidate。
- 模型可能尝试用 MetricFlow `derived` + `offset_window` 表达。
- 在 StarRocks 下，MetricFlow 生成的 offset SQL 可能出现重复列别名或无法解析别名，例如：

```text
Column 'activity_count' is ambiguous
```

验收标准：

- dry-run/validation 失败后必须回滚。
- 最终 YAML 不包含 `activity_count_mom_delta`。
- KB 中不包含 `metric:activity_count_mom_delta`。
- 整个 batch 应以 skipped/success 完成，不应让 bootstrap 失败。

### 测试 7：metrics_count 不重复计数

重复运行主流程命令。

预期：

- 最终仍为 `metrics_count=12`。
- 不应因为多个 batch 重复 sync 同一个 metric 文件导致 `metrics_count` 大于最终 YAML metric 数。
- 用 YAML 解析脚本得到 `TOTAL_METRICS 12`。

## Storage 隔离测试

### 测试 8：不同 datasource 不互相删除

准备两个 datasource，例如 `starrocks` 和另一个测试 datasource，在同一个 project 下分别写入 metric KB。

执行：

```bash
uv run datus-agent bootstrap-kb --datasource starrocks --components metrics --kb_update_strategy overwrite --success_story seed_context.csv -y
```

预期：

- 只删除并重建 `datasource_id='starrocks'` 的 metric 行。
- 其他 datasource 的 metric/semantic/reference rows 不被删除。
- 向量库目录不应新增 `{project}__ds__{datasource}` 这类 datasource 拆分目录。

### 测试 9：无 datasource 入口降级

打开 Explorer 或调用无 datasource 上下文的 Explorer API。

预期：

- 不应返回 500。
- 在无 datasource 绑定时，可以返回空列表或降级结果。
- 一旦 datasource 绑定后，应能读取对应 datasource scope 下的数据。

## 推荐回归测试命令

执行当前改动相关单测：

```bash
uv run pytest -q \
  tests/unit_tests/tools/func_tool/test_semantic_discovery_tools.py::TestAnalyzeMetricCandidatesFromHistory \
  tests/unit_tests/tools/func_tool/test_semantic_tools.py::TestGenerationEvidence \
  tests/unit_tests/tools/func_tool/test_generation_tools.py \
  tests/unit_tests/tools/func_tool/test_filesystem_tools.py::TestGlobSearch \
  tests/unit_tests/tools/func_tool/test_metric_filesystem_tools.py \
  tests/unit_tests/agent/node/test_gen_metrics_agentic_node.py \
  tests/unit_tests/storage/metric/test_metric_init.py
```

当前本地验证结果：

```text
200 passed
```

## 已知限制

1. StarRocks 下 MetricFlow `derived` + `offset_window` 对同一个 metric 做环比时，生成 SQL 可能存在重复别名或别名不可解析问题。当前实现不绕过 MetricFlow 生成 SQL，而是要求 validation/dry-run 失败后不发布该 derived metric。
2. `components=metrics` 只保证 metrics 相关 KB 行按 datasource overwrite；如果 QA 同时验证 semantic model storage，请使用 `--components semantic_model,metrics` 或对应全量流程。
3. 当前不做 subject path 级别清理；overwrite 的隔离维度是 datasource。
4. 旧向量表的 scope 字段补齐依赖后端支持 `ensure_columns`。如果后端不支持，需要重新 bootstrap 或执行后端迁移。

## QA 结论模板

验证通过时可记录：

```text
通过。bootstrap-kb metrics 在 StarRocks datasource 下 6/6 batch 成功，最终 metrics_count=12，YAML 中 12 个 metric 与 KB 计数一致。普通聚合、条件计数、AVG、MAX、ratio 均可生成；ROW_NUMBER Top-N 被识别为 derived datasource/detail 场景；product_row_count 未被发布；activity_count_mom_delta 在 StarRocks dry-run 失败后被回滚且未入库；overwrite 未出现跨 datasource 清理或文件覆盖丢失。
```
