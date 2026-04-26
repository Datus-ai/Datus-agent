# Language 命令 `/language`

## 概览

`/language` 用于固定所有 agentic 节点的 **响应语言** —— 包括回答文本、写入文件
的注释、调用子 agent 的 prompt、以及 `ask_user` 询问问题所用的自然语言。
代码、SQL 与标识符不受影响，保持原样。

设置可持久化到全局（`agent.yml` 的 `agent.language`），也可只作用于当前项目
（`.datus/config.yml` 的 `language`）；项目级取值在运行时优先级最高。

这是 [Agent 配置](../configuration/agent.md)中 `agent.language` 字段的运行时入口。

---

## 支持的语言

| 代码 | 语言 |
|------|------|
| `auto` | 让模型自行决定（清除 override） |
| `en` | English |
| `zh` | 中文 |
| `ja` | 日本語 |
| `ko` | 한국어 |
| `es` | Español |
| `fr` | Français |
| `de` | Deutsch |
| `pt` | Português |
| `ru` | Русский |
| `it` | Italiano |

不在列表中的代码会以 warning 形式接受，并原样写入 system prompt——便于
适配未列出的语言。

---

## 基本用法

### 交互式 TUI

直接输入 `/language` 打开 TUI。两步流程：

1. 选择语言代码
2. 选择持久化范围（项目 vs 全局）

```text
/language
```

### 命令式快捷写法

```text
# 选代码后交互式选 scope
/language zh
/language en

# 持久化到项目（.datus/config.yml）
/language zh --project

# 持久化到全局（agent.yml）
/language zh --global

# 清除 override（模型自行决定）
/language auto
/language --clear            # 等价：仅清除项目级 override
/language auto --global      # 清除全局 override
```

---

## 解析顺序

运行时解析顺序：

1. `.datus/config.yml` 的 `language:`（项目级 override）
2. `agent.yml` 的 `agent.language:`（全局默认）
3. 都未设置：不向 system prompt 注入语言指令，由模型自行决定

清除项目 override 后会回退到全局值（若有）；清除全局后则回退到"由模型决定"。

---

## 示例

```bash
# 把当前项目固定为中文
/language zh --project

# 在所有项目默认使用英文
/language en --global

# 删除项目 override，回退到全局默认
/language --clear

# 完全交还给模型
/language auto --global
```

参见：

- 配置字段：[`agent.language`](../configuration/agent.md)
- 单次请求级 override：[`POST /api/v1/chat/stream`](../API/chat.md#post-apiv1chatstream) 的 `language` 字段
