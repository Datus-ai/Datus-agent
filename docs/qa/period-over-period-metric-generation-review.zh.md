# 同环比指标生成逻辑改造 Review 说明

本文档总结当前同环比相关 metric 生成逻辑的改造内容、目标效果、边界和验证结果，供代码 review 使用。

## 背景

success story / 历史 SQL 通常不是一次性查询样例，而是用户沉淀长期指标体系的重要输入。之前同环比类 SQL 容易被表达成依赖扩展字段的 derived metric，或者需要查询阶段临时补充上期值。这样会带来两个问题：

- 指标定义不够独立，跨格式转换或后端 lowering 时容易丢失语义。
- `ask_metrics` 使用时不够直接，用户问“月环比”“周同比”时还需要查询层做额外推断或兜底。

本次改造的目标是：从历史 SQL 中识别固定语义的同环比业务输出，并生成独立、可查询的长期指标。

## 目标效果

以“活动数量月环比增长率”为例，期望生成一个独立 metric：

```yaml
- name: activity_count_mom_percent_change
  description: "活动数量月环比增长率，本月活动数量相对于上月活动数量的变化百分比。"
    expression:
      dialects:
      - dialect: ANSI_SQL
        expression: "COUNT(DISTINCT ac_code)"
  custom_extensions:
    - vendor_name: DATUS
      data: '{"dataset":"activity","time_dimension":"start_date","period_over_period":{"time_grain":"month","offset_window":"1 month","calculation":"percent_change"},"format":"0.00%","unit":"%"}'
```

这里的语义拆分是：

- `expression` 表示基础聚合口径。
- `period_over_period` 表示固定的同环比执行语义。
- 查询时用户直接使用 `activity_count_mom_percent_change`，不需要额外传 `compare` 参数，也不需要手工指定上月指标。

## 主要修改

### 1. 生成固定语义的 period-over-period metric

当 success story / 历史 SQL 中出现明确的同环比最终业务输出时，会生成固定语义的独立指标，包括：

- previous period value，例如上月活动数量、上周活动数量。
- delta，例如月环比差值、周环比差值。
- percent change，例如月环比增长率、月同比增长率。
- ratio，例如月环比倍数。

生成结果使用 DATUS extension 中的结构化字段表达：

```json
{
  "period_over_period": {
    "time_grain": "month",
    "offset_window": "1 month",
    "calculation": "percent_change"
  }
}
```

### 2. 用固定比较语义表达同环比

新的同环比 metric 使用“基础聚合口径 + 固定比较语义”的表达方式：

```json
{
  "dataset": "activity",
  "time_dimension": "start_date",
  "period_over_period": {
    "time_grain": "month",
    "offset_window": "1 month",
    "calculation": "percent_change"
  }
}
```

这样可以让指标定义直接表达长期业务语义，跨格式转换和后端编译时也能保留固定的比较粒度、偏移窗口和计算方式。

### 3. 增强 SQL history 解析

`analyze_metric_candidates_from_history` 现在能从常见 SQL 写法中识别固定同环比语义：

- CTE 中先按时间粒度聚合，再用 `LAG()` 取上一期。
- inline 表达式中的 `current - previous`。
- `(current - previous) / previous`。
- `current / previous`。
- `LAG(metric, 2)` 这类显式 offset。

同时修复了一个通用表达式识别问题：

- `(current - previous) * 1.0 / previous`
- `current * 1.0 / previous`

这些 SQL 常用于强制浮点除法，现在会被识别成 `percent_change` / `ratio`，而不是退化成普通 ratio 或无法识别的表达式。

### 4. OSI prompt / skill 约束更新

OSI metric 生成 prompt 明确要求：

- 同环比类最终业务输出应生成固定独立 metric。
- OSI expression 只写基础聚合口径。
- 固定比较语义写入 DATUS `period_over_period` extension。
- 当一个 SQL 结果同时呈现当前值、上一期值和比较结果时，优先把可复用的比较结果发布为 metric；当前值和上一期值作为该比较语义的一组计算上下文理解。
- detail/list 查询、TopN per group、ranking window 仍然跳过，不强行生成 metric。

同时修正 OSI prompt 中 expression dialect 的示例：

- `expression.dialects[].dialect` 使用 OSI core schema 允许的 dialect label。
- StarRocks、MySQL 等当前 OSI schema 未单独枚举的 SQL datasource 写 `ANSI_SQL`。
- Snowflake、Databricks 等 OSI schema 已枚举的 datasource 写对应的 `SNOWFLAKE`、`DATABRICKS`。

### 5. Adapter 执行侧支持

OSI adapter 会把 `period_over_period` lowering 到后端可执行 metric。

查询时会根据 metric 固定配置完成：

- 根据 `offset_window` 扩展查询时间范围。
- 按 `time_grain` 查询 current / previous。
- 根据 `calculation` 计算 previous value、delta、percent change 或 ratio。
- 最终过滤回用户请求的当前时间区间。

如果一次查询混用了不同固定时间粒度的同环比 metric，会拒绝执行，避免结果语义不清。

## 非目标

本次没有实现 query-time `compare` 参数，也没有让 LLM 在 `query_metrics()` 中临时构造比较逻辑。

本次也没有加 node 层兜底逻辑，例如：

- 请求 delta 时自动补 previous value。
- 针对某个 task ID 或某个指标名特殊处理。
- 针对 Baisheng 数据集写硬编码分支。

当前方向是让指标定义本身完整表达长期语义，查询时只选择已有 metric。

## 是否有定制化逻辑

产品代码中没有针对具体 benchmark case 的定制化兜底。

通用逻辑包括：

- 从 `LAG()` 和固定 offset 识别 `period_over_period`。
- 从 `* 1.0` 浮点除法形态识别 percent change / ratio。
- OSI prompt dialect 跟随当前 datasource。
- Adapter 按 metadata 中的 `time_grain` / `offset_window` / `calculation` lowering。

benchmark 中有一处测试数据层面的格式适配：

- task 51 的 gold SQL 使用 `DATE_FORMAT(..., '%Y-%m-%d %H:%i:%s')` 对齐当前 evaluator 对时间字符串的比较方式。
- 这是为了避免 `2025-08-04` 与 `2025-08-04 00:00:00` 的 CSV 表示差异影响评分。
- 长期更干净的方向是 evaluator 对日期/时间列做标准化比较，而不是每个 gold SQL 自己 format。

## 新增测试覆盖

### datus-agent

新增或扩展单测覆盖：

- `LAG()` 识别 previous period value。
- delta metric 生成。
- percent change metric 生成。
- ratio metric 生成。
- `LAG(metric, 2)` 显式 offset 生成 `2 months` / `2 weeks` 等窗口。
- OSI prompt 在 `current_datasource=starrocks` 时使用 schema-valid 的 `ANSI_SQL` dialect。

### datus-semantic-adapter

新增或扩展测试覆盖：

- OSI `period_over_period` delta lowering。
- OSI `period_over_period` ratio lowering。
- weekly grain 查询。
- hidden base metric 生成。
- mixed fixed grain 查询拒绝。

### datus-benchmark

`baisheng_ask_metrics` 新增 task 49-52：

- 49：活动数量月环比增长率。
- 50：活动数量月环比倍数。
- 51：每周活动数量、上周活动数量、周环比差值。
- 52：按 `product_type` 统计每月活动数量和月环比增长率。

## 验证结果

本地验证结果：

- `baisheng_ask_metrics` task 49-52：query evaluation `Average Score: 1.000`，4/4 passed。
- `datus-agent` 相关测试：324 passed。
- OSI prompt 测试：5 passed。
- `datus-semantic-adapter` 测试：79 passed。
- `validate_scenarios.py --scenario baisheng_ask_metrics` passed。
- 三个 worktree 的 `git diff --check` passed。

## Review 重点

请重点 review：

- `period_over_period` 是否足以表达当前长期同环比指标语义。
- `expression` 只保留基础聚合口径是否清晰。
- SQL history parser 对 `LAG()`、delta、percent change、ratio 的识别是否通用。
- adapter lowering 是否正确处理时间范围扩展和最终区间过滤。
- 是否还有任何不必要的 node 层兜底或 case-specific 分支。
- benchmark 51 的时间字符串格式适配是否接受，或者是否应后续改 evaluator 做日期标准化。
