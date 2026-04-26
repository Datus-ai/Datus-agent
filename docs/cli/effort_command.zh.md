# Effort 命令 `/effort`

## 概览

`/effort` 用于设置当前 LLM 使用的 **reasoning effort（推理强度）** 级别。
一套抽象覆盖所有支持的 provider：LiteLLM 会把级别映射成各家原生方言 —— 
OpenAI 的 `reasoning_effort`、Anthropic 的 `thinking.budget_tokens`、Gemini 的
`thinking_config.thinking_budget`、DeepSeek / Kimi 的 reasoning 等。

设置可以持久化到全局（`agent.yml`），也可以收敛到当前项目
（`.datus/config.yml`）；项目级取值在运行时优先级最高。

---

## 强度级别

| 级别 | 含义 |
|------|------|
| `off` | 关闭推理（不思考） |
| `minimal` | 极小推理（快速；gpt-5 系列） |
| `low` | 低强度 |
| `medium` | 中等强度（平衡） |
| `high` | 高强度（深度推理） |

如果当前模型不支持 reasoning，级别会被静默忽略。`/effort status` 会通过
`litellm.supports_reasoning` 检查并提示当前模型是否真的能消费该 hint。

---

## 基本用法

### 交互式 TUI

直接输入 `/effort` 打开 TUI。两步流程：

1. 选择 effort 级别
2. 选择持久化范围（项目 vs 全局）

```text
/effort
```

### 命令式快捷写法

```text
# 选级别后交互式选 scope
/effort high
/effort minimal

# 持久化到项目（.datus/config.yml）
/effort high --project

# 持久化到全局（agent.yml）
/effort high --global

# 关闭推理
/effort off

# 清除项目级 override（回退到全局或模型默认）
/effort --clear

# 显示当前生效级别与来源
/effort status
```

---

## 解析顺序

agent 在每一轮对话开始时按以下顺序解析有效级别（命中即停止）：

1. `.datus/config.yml` 的 `reasoning_effort:`（项目级 override）
2. `agent.yml` 顶层 `reasoning_effort:`（全局默认）
3. `agent.models.<active>.reasoning_effort`（模型级配置）
4. `agent.models.<active>.enable_thinking: true` → 视作 `medium`
5. 都未设置：使用模型自身的默认行为

`/effort status` 会同时打印生效级别与来源标签
（`project` / `global` / `model` / `not set`）。

---

## 示例

```bash
# 单次：将当前项目切换到 high
/effort high --project

# 在本机所有项目把 low 设为默认
/effort low --global

# 快速关闭当前项目的推理
/effort off --project

# 清除项目级 override，恢复模型默认
/effort --clear

# 查看当前生效情况
/effort status
```

参见：[`/model`](model_command.md) 用于切换当前 LLM。
