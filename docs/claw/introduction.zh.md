# IM 网关 (Claw)

Claw 是 Datus Agent 的 IM（即时通讯）网关模块。它将 Datus 连接到 Slack 和飞书（Lark）等聊天平台，使用户可以直接在团队沟通工具中与数据分析 Agent 进行交互。

## 核心设计理念

- **仅使用出站长连接** — Claw 通过 WebSocket、Socket Mode 或 Stream SDK 主动连接到各 IM 平台，无需 Webhook 端点或公网 URL。
- **实时流式响应** — Agent 的响应（思考过程、工具调用、SQL、Markdown）在生成时即刻流式推送到聊天中。
- **会话管理** — 每个对话（群聊/私聊/话题）自动获得持久化会话。用户可通过 `/new` 或 `/reset` 重置会话。
- **命名空间与子代理路由** — 每个频道可覆盖默认命名空间或将消息路由到特定的子代理。

## 支持的平台

| 平台 | 适配器 | SDK | 连接方式 | 状态 |
|------|--------|-----|----------|------|
| Slack | `slack` | `slack-sdk[socket_mode]` | Socket Mode (WebSocket) | 可用 |
| 飞书 (Lark) | `feishu` | `lark-oapi` | WebSocket 长连接 | 可用 |

## 架构

```text
┌──────────────┐     长连接              ┌──────────────────┐
│  IM 平台      │ ◄──────────────────────► │  ChannelAdapter   │
│  (Slack /    │                          │  (按平台实现)      │
│   飞书)      │                          └────────┬─────────┘
│              │                                   │
└──────────────┘                                   │ InboundMessage
                                                   ▼
                                          ┌──────────────────┐
                                          │  ChannelBridge    │
                                          │  - 会话管理        │
                                          │  - 聊天命令        │
                                          └────────┬─────────┘
                                                   │ StreamChatInput
                                                   ▼
                                          ┌──────────────────┐
                                          │ ChatTaskManager   │
                                          │  (Agent 循环)     │
                                          └────────┬─────────┘
                                                   │ SSE 事件
                                                   ▼
                                          ┌──────────────────┐
                                          │  OutboundMessage  │
                                          │  → adapter.send() │
                                          └──────────────────┘
```

每个适配器维护与其平台的持久连接。当消息到达时，`ChannelBridge` 将其转换为 `StreamChatInput`，送入 `ChatTaskManager`，并将每个 SSE 事件作为 `OutboundMessage` 流式推送给用户。

## 安装

为需要连接的平台安装对应的 SDK：

```bash
# Slack
pip install "slack-sdk[socket_mode]"

# 飞书 (Lark)
pip install lark-oapi
```

只需安装你使用的平台的 SDK。如果缺少必需的 SDK，Claw 会给出明确的错误提示。

## 配置

在 `agent.yml` 文件中添加 `channels` 部分。`channels` 下的每个键定义一个频道实例。

### 通用结构

```yaml
channels:
  my-channel:
    adapter: slack          # 必填: feishu | slack
    enabled: true           # 可选: 默认 true
    namespace: my_namespace # 可选: 覆盖默认命名空间
    subagent_id: agent_01   # 可选: 路由到特定子代理
    extra:                  # 必填: 适配器专用凭证
      # ... 平台专用配置项
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `adapter` | string | 是 | 适配器类型：`feishu` 或 `slack` |
| `enabled` | bool | 否 | 是否启用此频道（默认：`true`） |
| `namespace` | string | 否 | 覆盖网关的默认命名空间 |
| `subagent_id` | string | 否 | 将消息路由到特定子代理 |
| `extra` | dict | 是 | 适配器专用配置（令牌、密钥等） |

### Slack

```yaml
channels:
  slack-main:
    adapter: slack
    enabled: true
    extra:
      app_token: ${SLACK_APP_TOKEN}       # xapp-... (Socket Mode 令牌)
      bot_token: ${SLACK_BOT_TOKEN}       # xoxb-... (Bot User OAuth 令牌)
```

### 飞书 (Lark)

```yaml
channels:
  feishu-main:
    adapter: feishu
    enabled: true
    extra:
      app_id: ${FEISHU_APP_ID}
      app_secret: ${FEISHU_APP_SECRET}
```

!!! warning "请勿在配置文件中明文存储密钥"
    始终使用 `${ENV_VAR}` 替换来引用令牌和密钥，切勿将明文凭证提交到版本控制系统。

## 平台配置指南

各 IM 平台的详细配置步骤，请参阅各适配器专属指南：

- [Slack 配置指南](slack.zh.md) — 创建 Slack App、启用 Socket Mode、配置权限和事件订阅
- [飞书配置指南](feishu.zh.md) — 创建自建应用、启用机器人能力、配置 WebSocket 事件


## 启动网关

使用以下命令启动 Claw 网关：

```bash
python -m datus.claw.main --config conf/agent.yml
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `./conf/agent.yml` | Agent 配置文件路径 |
| `--namespace` | `default`（或 `DATUS_NAMESPACE` 环境变量） | 所有频道的默认命名空间 |
| `--host` | `0.0.0.0` | 健康检查服务绑定地址 |
| `--port` | `9000` | 健康检查服务绑定端口 |
| `--debug` | `false` | 启用调试日志 |
| `--log-level` | `INFO`（或 `DATUS_LOG_LEVEL` 环境变量） | 日志级别：DEBUG、INFO、WARNING、ERROR、CRITICAL |

## 功能特性

### 实时流式响应

Claw 将 Agent 处理的每个阶段实时推送到聊天中：

- **思考过程** — Agent 的推理过程
- **工具调用** — 正在调用哪些工具及其返回结果
- **SQL** — 生成的 SQL 查询以代码块形式展示
- **Markdown** — 最终回答和解释
- **错误** — 任何错误都会立即上报

### 聊天命令

Claw 提供了内置的斜杠命令，这些命令在消息进入 Agent 循环之前被拦截处理。输入时可以带或不带 `/` 前缀。

| 命令 | 别名 | 说明 |
|------|------|------|
| `/help` | — | 显示所有可用命令 |
| `/new` | `/reset`、`/clear` | 重置当前对话会话 |
| `/verbose [级别]` | — | 查看或设置当前对话的详细程度 |

#### `/help`

显示所有已注册命令及其描述。

#### `/new` / `/reset` / `/clear`

重置当前对话会话并开启新会话。机器人会确认重置并显示新的会话 ID。

#### `/verbose [级别]`

控制机器人在处理过程中展示的信息详细程度。不带参数调用时，显示当前的详细级别。

| 级别 | 别名 | 行为 |
|------|------|------|
| `quiet` | `off` | 仅显示思考过程和最终输出（不显示工具调用） |
| `brief` | `on` | 显示思考过程、工具调用摘要和最终输出 |
| `detail` | `full` | 显示思考过程、工具参数/结果和最终输出 |

示例：

```text
/verbose           # 查看当前详细级别
/verbose quiet     # 仅显示思考和最终回答
/verbose brief     # 显示工具调用摘要
/verbose detail    # 显示完整的工具调用详情
```

### 会话管理

每个对话（群聊、私聊或话题）会自动映射到一个持久化会话。要开启新会话，发送 `/new`、`/reset` 或 `/clear`。

机器人会确认重置并返回新的会话 ID。

### 命名空间覆盖

每个频道可在配置中指定 `namespace` 以覆盖网关的默认命名空间。这允许不同频道查询不同的数据库。

### 子代理路由

使用 `subagent_id` 字段将特定频道的消息路由到专用子代理，实现按频道定制行为。

## 故障排查

### SDK 未安装

```text
ImportError: slack_sdk is required for the Slack adapter. Install it with: pip install slack-sdk[socket_mode]
```

请安装目标平台所需的 SDK 包。参见[安装](#安装)部分。

### 凭证无效

如果机器人连接后立即断开或日志中出现认证错误，请检查：

- Slack：`app_token` 以 `xapp-` 开头，`bot_token` 以 `xoxb-` 开头
- 飞书：`app_id` 和 `app_secret` 与自建应用凭证匹配

### 机器人收不到消息

- **Slack**：确保机器人已被邀请到频道且事件订阅已启用。
- **飞书**：确保已选择 WebSocket 订阅模式且已添加 `im.message.receive_v1` 事件。

### 调试日志

启用详细日志以诊断连接问题：

```bash
python -m datus.claw.main --config conf/agent.yml --debug
```

或设置环境变量：

```bash
export DATUS_LOG_LEVEL=DEBUG
```

## 下一步

- [配置](../configuration/introduction.md) — 完整的 Agent 配置参考
- [Chatbot](../web_chatbot/introduction.md) — 基于 Web 的聊天界面
- [命名空间](../configuration/namespace.md) — 命名空间配置详情
- [子代理](../subagent/introduction.md) — 子代理配置与自定义
