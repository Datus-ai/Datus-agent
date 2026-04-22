# Model 命令 `/model`

## 概述

`/model` 斜杠命令允许你在运行时切换活跃的 LLM 提供商和模型，无需编辑配置文件。它提供交互式 TUI 选择器和直接快捷语法以快速切换。

所有选择都会持久化到项目级 `.datus/config.yml`，在会话重启后仍然生效，并且作用域限定在当前工作目录。

---

## 基本用法

### 交互式选择器

输入 `/model`（或 `/models`）不带参数以打开完整的 TUI 选择器：

```text
/model
```

选择器有三个标签页，可通过 **Tab** / **Shift+Tab** 切换：

| 标签页 | 说明 |
|--------|------|
| **Providers** | 通用 LLM 提供商（OpenAI、DeepSeek、Claude 等） |
| **Plans** | 编程导向的 Plan 提供商（Alibaba Coding、GLM Coding 等） |
| **Custom** | 来自 `agent.models` 的自托管或私有部署模型 |

使用**上/下**箭头导航，**Enter** 选择，**Esc** / **q** 取消。

### 直接快捷方式

无需打开选择器即可切换：

```text
# 切换到指定提供商 + 模型
/model openai/gpt-4.1
/model deepseek/deepseek-chat
/model claude/claude-sonnet-4-5

# 切换到自定义模型条目
/model custom:my-internal-model

# 打开选择器并预先定位到某个提供商
/model openai
```

---

## 提供商认证

### API Key 提供商

对于需要 API Key 的提供商（OpenAI、DeepSeek、Claude API、Kimi、Qwen、Gemini 等）：

1. 在选择器中选择该提供商
2. 如果未配置凭据，将出现凭据表单
3. 输入你的 API Key（输入掩码显示）
4. Key 会持久化到 `agent.yml` 的 `agent.providers.<name>` 下

API Key 也可以通过环境变量预配置。每个提供商在 `conf/providers.yml` 中定义了默认环境变量名（例如 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`）。

### Claude Subscription

Claude Pro/Max 订阅用户：

1. 在选择器中选择 **claude_subscription**
2. 系统自动检测本地存储的订阅令牌
3. 如果自动检测失败，系统会提示你手动输入令牌（`sk-ant-oat01-...`）

### Codex (ChatGPT Plus/Pro) OAuth

ChatGPT Plus/Pro 用户通过 Codex：

1. 在选择器中选择 **codex**
2. 浏览器窗口打开进行 OAuth 认证
3. 完成浏览器流程后，系统自动验证连通性
4. 成功后返回选择器，Codex 模型变为可用

---

## 自定义模型

### 添加自定义模型

1. 导航到选择器的 **Custom** 标签页
2. 选择 **+ Add custom model**
3. 填写表单字段：
   - **Name** — 唯一标识符（用作 `agent.models.<name>` 的 key）
   - **Type** — 接口类型（`openai`、`claude`、`deepseek`、`kimi`、`gemini` 等）
   - **Base URL** — API 端点 URL
   - **API Key** — 认证凭据（掩码显示）
   - **Model** — 模型名称 / SKU
4. 按 **Enter** 保存

自定义模型存储在 `agent.yml` 的 `agent.models` 下，并出现在 **Custom** 标签页中。

### 删除自定义模型

1. 导航到 **Custom** 标签页
2. 选择要删除的模型
3. 按 **d** 两次（两次按键确认）以删除

---

## 配置文件

### 项目覆盖 (`.datus/config.yml`)

当你通过 `/model` 选择模型时，选择会保存到当前工作目录的 `.datus/config.yml`：

```yaml
# 提供商级选择
target:
  provider: openai
  model: gpt-4.1

# 或者，自定义模型选择
target:
  custom: my-internal-model
```

此文件仅支持三个 key：

| Key | 说明 |
|-----|------|
| `target` | 活跃的 LLM 选择（provider+model 或自定义名称） |
| `default_database` | 此项目的默认数据库 |
| `project_name` | session/data 分片名称覆盖 |

### 提供商凭据 (`agent.yml`)

提供商凭据位于 `agent.yml` 的 `agent.providers` 下：

```yaml
agent:
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
    deepseek:
      api_key: ${DEEPSEEK_API_KEY}
    claude_subscription:
      auth_type: subscription
    codex:
      auth_type: oauth
```

### 提供商目录 (`conf/providers.yml`)

提供商目录定义了可用的提供商、模型、base URL 和模型规格。它是只读的，随 Datus 一起分发。向此文件添加新提供商即可在 `/model` 选择器中使用，无需代码更改。

---

## 工作原理

模型解析遵循以下优先级：

1. **项目覆盖** — 当前工作目录中的 `.datus/config.yml`
2. **基础配置** — `agent.yml` 中的 `agent.target`

`/model` 命令更新项目覆盖。它**不会**重建 agent — 切换在同一会话内的下一次 LLM 调用时生效。
