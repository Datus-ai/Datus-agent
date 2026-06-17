# 校验

Validation 是面向“已经创建或更新了持久资源”的 subagent 的后置保护层。
它和生成过程里的 `read_query`、`validate_ddl`、`validate_semantic` 不同：
这些工具帮助主 agent 在结束前生成正确 SQL 或 YAML，而 validation hook 会在写入类工具上报
deliverable 之后自动运行。

## 会在哪里运行

`ValidationHook` 会挂在会产出 deliverable 的 subagent 上：

| Subagent | 收集的 target | 内置检查 | Validator skill |
|----------|---------------|----------|-----------------|
| `gen_table` | `execute_ddl` 产出的表 | 表存在 | `table-validation` |
| `gen_job` | DDL、DML、`transfer_query_result` 产出的表和跨库传输 | 表存在、传输行数一致 | `table-validation`、`transfer-reconciliation` |
| `gen_dashboard` | BI 工具产出的 dashboard、chart、dataset | BI 资源存在 | `bi-validation` |
| `scheduler` | submit/update 工具产出的 scheduler job | job 存在且状态不是 failed；如果 runtime 工具可用，还会确定性触发并轮询一次 | `scheduler-validation` |

语义模型和指标生成走自己的发布门禁：`validate_semantic`，以及指标场景里的
`query_metrics(..., dry_run=True)`。这些流程不使用 `ValidationHook` target。

## 运行流程

1. 写入类工具返回 `FuncToolResult`，并在 `result.deliverable_target` 里声明它产出了什么。
2. `ValidationHook.on_tool_end` 读取这个 target，加入当前 session。hook 累积的是整次运行，不只是最后一次工具调用。
3. agent 结束时，`ValidationHook.on_end` 把所有 target 包成一个 `SessionTarget`。
4. Layer A 运行确定性的代码检查。即使关闭 LLM validator skill，这一层也始终执行。
5. 对 scheduler job，如果 scheduler 工具暴露 trigger 和 run-history API，hook 会额外执行确定性的 runtime 检查：触发刚提交的 job、轮询对应 run，并在失败时附上日志。
6. 当 `agent.validation.skill_validators_enabled` 为 true 时，Layer B 会运行匹配的 validator skill。
7. 只要有 blocking 检查失败，拥有这个 hook 的 subagent 会把一份精简的 validation 失败报告作为下一轮 prompt 继续重试。重试次数由 `agent.validation.max_retries` 限制。重试后仍失败时，节点返回 `success=false`，并带上 `validation_report`。

Advisory 级别的失败和 warning 会被报告出来，但不会触发重试。

## 内置检查和 validator skill 的区别

Layer A 是代码层基础设施，用来验证每次运行都应该满足的硬性不变量：

- table target 可以被 `describe_table`
- transfer 的目标表存在
- 当 transfer 工具上报 source/target 行数时，两边行数一致
- BI dashboard、chart、dataset 可达
- scheduler job 存在，且不是已失败状态

Layer B 是用户可扩展的规则层，由 frontmatter 中 `kind: validator` 的 skill 实现。
Validator skill 不会通过 `load_skill` 加载，也不会作为普通 skill 暴露给主 agent；
它只由 `ValidationHook` 在匹配的 subagent 运行结束时调用。

Validator 子 agent 只能使用只读工具：

- 数据库只读工具，例如 `describe_table`、`read_query`
- BI 只读工具，例如 `get_dashboard`、`get_chart`、`get_chart_data`、`get_dataset`
- Scheduler 只读工具，例如 `get_scheduler_job`、`list_job_runs`、`get_run_log`

写入工具和递归 subagent 工具不会暴露给 validator。Scheduler validator 也不应该调用
`trigger_scheduler_job`；确定性的触发和轮询由 hook 负责。

## 配置 validation

在 `agent.yml` 中配置：

```yaml
agent:
  validation:
    # 设为 false 时，只关闭 Layer B validator skill；Layer A 仍然会运行。
    skill_validators_enabled: true

    # 主 agent 总尝试次数，包含第一次尝试。
    max_retries: 3
```

如果只想保留低成本的确定性检查，可以设置 `skill_validators_enabled: false`。
如果只想关闭某一个 validator，保留其他 validator，可覆盖或编辑对应 skill，把
`severity` 设为 `off`。

## 添加项目级 validator

在项目下创建一个 skill：

```text
./.datus/skills/
└── finance-table-validation/
    └── SKILL.md
```

如果你自定义了 `skills.directories`，请把 validator 放到配置里的某个目录下。
默认扫描顺序下，项目级 `./.datus/skills` 会覆盖用户级 `~/.datus/skills`，二者又都会覆盖内置 skill。

示例：

```markdown
---
name: finance-table-validation
description: Validate finance mart tables after gen_job writes them
tags: [validation, finance, data-quality]
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
kind: validator
severity: blocking
mode: llm
allowed_agents:
  - gen_job
targets:
  - type: table
    schema: marts
    table_pattern: finance_*
---

# Finance Table Validation

ValidationHook 会在匹配的 `gen_job` 运行结束后调用这个 skill。你收到的是
`SessionTarget`；请遍历 `session.targets`，对每个匹配的表独立执行检查。

对每张 finance mart 表：

1. 使用目标 datasource 上的 `read_query` 检查最新业务日期是否存在。
2. 使用 `read_query` 确认 `amount` 没有负值。
3. 每条规则、每张表都返回一个独立 check。

不要修改数据。如果某张表失败，报告具体表名和失败规则，让重试 prompt 只修复有问题的 target。
```

hook 会自动追加 JSON 输出契约。validator 的回答里需要包含一个 fenced JSON block：

```json
{
  "checks": [
    {
      "name": "non_negative_amount",
      "passed": false,
      "severity": "blocking",
      "observed": {"table": "marts.finance_daily", "negative_rows": 3},
      "expected": {"negative_rows": 0}
    }
  ],
  "blocking_issues": ["marts.finance_daily has negative amount values"]
}
```

`blocking_issues` 里的每一项都会被转换成失败的 blocking check。它适合放简短的必修复问题。

## Target 过滤

`targets` 决定 validator 什么时候运行。空列表表示匹配 session 中的所有 target。
否则，只要任意 filter 匹配，就会激活这个 validator。

支持的 target 类型：

- `table`
- `transfer`
- `dashboard`
- `chart`
- `dataset`
- `scheduler_job`

表类 target 还支持：

- `database`
- `schema`
- `table`
- `table_pattern`，使用 `fnmatch` glob 语法

示例：

```yaml
# gen_table 创建的任意表。
allowed_agents: [gen_table]
targets:
  - type: table

# 只检查 marts 下名称以 rev_ 开头的表。
allowed_agents: [gen_job]
targets:
  - type: table
    schema: marts
    table_pattern: rev_*

# gen_dashboard 产出的任意 dashboard/chart/dataset。
allowed_agents: [gen_dashboard]
targets: []
```

`allowed_agents` 可以写具体配置出来的 subagent alias，也可以写规范的节点类名，
例如 `gen_job`、`gen_dashboard`、`scheduler`。

## 修改内置 validation 规则

如果只想在某个项目里修改内置 validator，把内置 skill 复制到项目级 skill 目录，并保持相同
`name`：

```text
./.datus/skills/table-validation/SKILL.md
./.datus/skills/bi-validation/SKILL.md
./.datus/skills/scheduler-validation/SKILL.md
./.datus/skills/transfer-reconciliation/SKILL.md
```

修改时保留 `kind: validator`、正确的 `allowed_agents`，以及符合预期的 `targets`
过滤条件。改完后重启 Datus 或重新打开对应 subagent，让 skill registry 重新扫描文件。

空值率、枚举取值、重复键、样本 diff、业务阈值等表级数据内容检查，建议通过项目级
validator skill 添加。内置 `table-validation` 刻意只覆盖显式 schema 契约；
对象存在和基础行数不变量属于 Layer A。
