# Slack 适配器

本指南将引导你完成 Slack 机器人的配置，通过 Datus Gateway IM 网关连接到 Datus Agent。Slack 使用 **Socket Mode**，从服务器建立出站 WebSocket 连接 —— 无需公网 URL 或 Webhook 端点。

## 前置条件

- 你有权限安装应用的 Slack 工作区
- 已安装并配置好 Datus Agent

## 步骤 1：安装 SDK

```bash
pip install "slack-sdk[socket_mode]"
```

## 步骤 2：创建 Slack App（使用 Manifest）

使用 Manifest 是创建正确配置的 Slack App 最快的方式。它会一步到位地预配置 Socket Mode、Bot 权限和事件订阅。

1. 前往 [api.slack.com/apps/new](https://api.slack.com/apps/new)。
2. 点击 **Create New App** → **From an app manifest**。
3. 选择你的工作区，点击 **Next**。
4. 粘贴以下 Manifest（选择 JSON 标签页）：

```json
{
  "display_information": {
    "name": "Datus Agent",
    "description": "Datus AI data analysis agent"
  },
  "features": {
    "bot_user": {
      "display_name": "Datus Agent",
      "always_online": true
    },
    "app_home": {
      "messages_tab_enabled": true,
      "messages_tab_read_only_enabled": false
    }
  },
  "oauth_config": {
    "scopes": {
      "bot": [
        "app_mentions:read",
        "channels:history",
        "channels:read",
        "chat:write",
        "groups:history",
        "groups:read",
        "im:history",
        "im:read",
        "im:write",
        "mpim:history",
        "mpim:read",
        "mpim:write",
        "users:read"
      ]
    }
  },
  "settings": {
    "socket_mode_enabled": true,
    "event_subscriptions": {
      "bot_events": [
        "app_mention",
        "message.channels",
        "message.groups",
        "message.im",
        "message.mpim"
      ]
    }
  }
}
```

5. 检查配置摘要，点击 **Create**。

!!! tip "自定义 Manifest"
    你可以将 `display_information.name` 修改为任意名称。如需额外功能，可添加更多权限范围（如 `reactions:write`、`files:write`）。

## 步骤 3：生成令牌

应用创建后，你需要两个令牌：

### App-Level Token（用于 Socket Mode）

1. 进入 **Settings → Basic Information**。
2. 向下滚动到 **App-Level Tokens**，点击 **Generate Token and Scopes**。
3. 输入名称（如 `gateway-socket-token`），添加权限范围 `connections:write`，点击 **Generate**。
4. 复制生成的令牌（以 `xapp-` 开头）。这就是你的 `SLACK_APP_TOKEN`。

### Bot Token（用于发送消息）

1. 在左侧导航栏，进入 **OAuth & Permissions**。
2. 点击 **Install to Workspace**，审批权限。
3. 复制 **Bot User OAuth Token**（以 `xoxb-` 开头）。这就是你的 `SLACK_BOT_TOKEN`。

!!! tip "令牌格式"
    - App-Level Token：以 `xapp-` 开头
    - Bot User OAuth Token：以 `xoxb-` 开头

    如果你的令牌以 `xoxp-` 开头，说明误复制了 User OAuth Token。

## 步骤 4：邀请机器人到频道

在 Slack 中，将机器人邀请到你希望它响应消息的频道：

```bash
/invite @YourBotName
```

机器人只会接收到它已被邀请的频道中的消息。

## 步骤 5：配置 Datus Agent

在 `agent.yml` 中添加 Slack 频道配置：

```yaml
channels:
  slack-main:
    adapter: slack
    enabled: true
    extra:
      app_token: ${SLACK_APP_TOKEN}       # xapp-... (App-Level Token)
      bot_token: ${SLACK_BOT_TOKEN}       # xoxb-... (Bot User OAuth Token)
```

设置环境变量：

```bash
export SLACK_APP_TOKEN="xapp-1-..."
export SLACK_BOT_TOKEN="xoxb-..."
```

## 配置参考

| 键 | 必填 | 说明 |
|----|------|------|
| `app_token` | 是 | Socket Mode 的 App-Level Token（`xapp-...`） |
| `bot_token` | 是 | Bot User OAuth Token（`xoxb-...`） |

## 验证连接

启动网关并查看日志：

```bash
python -m datus.gateway.main --config conf/agent.yml --debug
```

你应该看到：

```text
Slack adapter 'slack-main' connecting...
Slack adapter 'slack-main' started.
```

在已邀请的频道中发送消息，机器人应回复 Agent 的分析结果。

## 权限与事件清单

如果你使用了上述 Manifest 创建应用，所有权限和事件已自动配置。以下清单用于验证或手动调整。

### 必需的 Bot Token 权限

| 权限范围 | 用途 |
|----------|------|
| `chat:write` | 向频道发送消息 |
| `channels:history` | 读取公共频道消息 |
| `channels:read` | 查看公共频道基本信息 |
| `groups:history` | 读取私有频道消息 |
| `groups:read` | 查看私有频道基本信息 |
| `im:history` | 读取私聊消息 |
| `im:read` | 查看私聊基本信息 |
| `im:write` | 主动发起私聊 |
| `mpim:history` | 读取群组私聊消息 |
| `mpim:read` | 查看群组私聊信息 |
| `mpim:write` | 发起群组私聊 |
| `users:read` | 查看用户信息 |
| `app_mentions:read` | 接收 @提及 事件 |

### 必需的 Bot 事件

| 事件 | 触发条件 |
|------|----------|
| `message.channels` | 公共频道中有新消息 |
| `message.groups` | 私有频道中有新消息 |
| `message.im` | 机器人收到私聊消息 |
| `message.mpim` | 群组私聊中有新消息 |
| `app_mention` | 机器人被 @提及 |

??? info "手动配置（不使用 Manifest）"
    如果你更倾向于手动配置应用：

    1. 前往 [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From scratch**。
    2. **启用 Socket Mode**：**Settings → Socket Mode** → 开启 → 生成 App-Level Token（添加 `connections:write` 权限范围）。
    3. **添加 Bot 权限**：**OAuth & Permissions → Bot Token Scopes** → 逐一添加上表中的权限范围。
    4. **启用事件订阅**：**Event Subscriptions** → 开启 → 在 **Subscribe to bot events** 中添加上表中的事件 → **Save Changes**。
    5. **安装应用**：**OAuth & Permissions** → **Install to Workspace** → 复制 Bot User OAuth Token。

## 故障排查

### "invalid_auth" 错误

- 确认 `bot_token` 以 `xoxb-` 开头且未过期。
- 重新安装应用到工作区以重新生成令牌。

### 机器人连接成功但收不到消息

- 确保机器人已被邀请到频道（`/invite @BotName`）。
- 确认事件订阅已启用，且已添加四个 `message.*` 事件。
- 检查 Socket Mode 是否已启用。

### "missing_scope" 错误

- 前往 **OAuth & Permissions → Bot Token Scopes** 添加缺少的权限范围。
- 添加新权限后需要重新安装应用。

### 机器人消息触发循环

- 适配器会自动忽略带有子类型的消息（编辑、删除、机器人消息）。如果仍出现循环，请检查机器人是否在回复自己的消息。
