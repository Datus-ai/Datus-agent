# 飞书适配器

本指南将引导你完成飞书机器人的配置，通过 Claw IM 网关连接到 Datus Agent。飞书使用官方 `lark-oapi` SDK 的 **WebSocket 长连接**，机器人主动发起出站连接 —— 无需公网 URL 或 Webhook 端点。

## 前置条件

- 你有开发者权限的飞书租户（组织）
- 已安装并配置好 Datus Agent

## 步骤 1：安装 SDK

```bash
pip install lark-oapi
```

## 步骤 2：创建自建应用

1. 前往[飞书开放平台](https://open.feishu.cn)，使用飞书账号登录。
2. 点击右上角头像，选择**开发者后台**（或直接访问 [open.feishu.cn/app](https://open.feishu.cn/app)）。
3. 点击**创建企业自建应用**。
4. 填写应用名称（如 "Datus Agent"）和描述。
5. 点击**创建**。

!!! note "Lark（国际版）用户"
    如果你使用的是国际版（Lark），请前往 [open.larksuite.com](https://open.larksuite.com)。配置步骤完全相同。

## 步骤 3：获取应用凭证

1. 创建应用后，进入应用管理页面。
2. 在左侧导航栏，点击**凭证与基础信息**。
3. 复制 **App ID** 和 **App Secret**。它们分别是你的 `FEISHU_APP_ID` 和 `FEISHU_APP_SECRET`。

!!! warning "妥善保管 App Secret"
    App Secret 相当于密码，切勿公开分享或提交到版本控制系统。请使用环境变量引用。

## 步骤 4：启用机器人能力

1. 在左侧导航栏，进入**应用功能 → 机器人**。
2. 将机器人能力切换为**已启用**。
3. 可选：设置机器人名称和头像。

## 步骤 5：配置事件与回调（WebSocket 模式）

Claw 使用飞书 WebSocket 长连接接收事件，无需公网回调 URL。

1. 在左侧导航栏，进入**事件与回调**。
2. 在**订阅方式**中，选择**使用长连接接收事件**（WebSocket 模式）。

    !!! tip "为什么选择 WebSocket 模式？"
        WebSocket 模式非常适合 Claw，因为它不需要公网可达的服务器。机器人主动发起到飞书服务器的 WebSocket 出站连接，事件通过该连接推送。

3. 在**事件列表**中，点击**添加事件**，搜索并添加：

    | 事件名称 | 事件标识 | 说明 |
    |----------|----------|------|
    | 接收消息 | `im.message.receive_v1` | 机器人收到消息时触发 |
    | 消息被表情回复 | `im.message.reaction.created_v1` | 用户对消息添加表情回复时触发 |
    | 消息表情回复被删除 | `im.message.reaction.deleted_v1` | 用户移除消息表情回复时触发 |

4. 点击**保存**。

## 步骤 6：配置权限

1. 在左侧导航栏，进入**权限管理**。
2. 搜索并添加以下权限：

| 权限名称 | 权限标识 | 用途 |
|----------|----------|------|
| 读取发给机器人的消息 | `im:message` | 接收私聊消息 |
| 接收群聊中 @机器人消息 | `im:message.group_at_msg:readonly` | 接收群聊 @bot 消息 |
| 以机器人身份发送消息 | `im:message:send_as_bot` | 回复用户 |
| 读取卡片信息 | `cardkit:card:read` | 将消息 ID 转换为卡片 ID，用于流式卡片 |
| 写入卡片内容 | `cardkit:card:write` | 更新流式卡片内容并关闭流式模式 |
| 获取群组信息 | `im:chat:readonly` | 读取群聊元数据（可选） |
| 读取用户信息 | `contact:user.base:readonly` | 解析发送者名称（可选） |

3. 点击**保存**。

!!! tip "批量导入"
    在**权限管理**页面，点击**批量开通**，粘贴以下 JSON 即可一次性添加所有权限：

    ```json
    {
      "scopes": {
        "tenant": [
          "cardkit:card:read",
          "cardkit:card:write",
          "contact:user.base:readonly",
          "im:chat:readonly",
          "im:message",
          "im:message.group_at_msg:readonly",
          "im:message:send_as_bot",
          "im:resource"
        ]
      }
    }
    ```

!!! note "必需权限与可选权限"
    `im:message`、`im:message.group_at_msg:readonly`、`im:message:send_as_bot`、`cardkit:card:read` 和 `cardkit:card:write` 是完整功能（包括群聊 @bot 消息和流式卡片响应）所必需的。其他权限为可选，取决于你的使用场景。如果未开通 CardKit 权限，机器人会回退为每个响应片段发送独立消息。

## 步骤 7：发布应用

1. 在左侧导航栏，进入**版本管理与发布**。
2. 点击**创建版本**。
3. 填写版本号和更新说明。
4. 点击**提交审核**。

- **测试租户**：你可以在测试组织中自行审批。先在**测试**部分添加测试成员。
- **生产租户**：需要组织管理员审批通过。

!!! info "应用可用性"
    审批通过后，应用对组织内的用户可用。用户可以从飞书应用目录或搜索机器人名称来添加。

## 步骤 8：配置 Datus Agent

在 `agent.yml` 中添加飞书频道配置：

```yaml
channels:
  feishu-main:
    adapter: feishu
    enabled: true
    extra:
      app_id: ${FEISHU_APP_ID}
      app_secret: ${FEISHU_APP_SECRET}
```

设置环境变量：

```bash
export FEISHU_APP_ID="cli_xxxxxxxxxxxx"
export FEISHU_APP_SECRET="xxxxxxxxxxxxxxxxxxxxxxxx"
```

## 配置参考

| 键 | 必填 | 说明 |
|----|------|------|
| `app_id` | 是 | 飞书开放平台的 App ID |
| `app_secret` | 是 | 飞书开放平台的 App Secret |

## 验证连接

启动 Claw 网关并查看日志：

```bash
python -m datus.claw.main --config conf/agent.yml --debug
```

你应该看到：

```text
Feishu adapter 'feishu-main' connecting...
Feishu adapter 'feishu-main' started.
```

在飞书群聊或私聊中向机器人发送消息，机器人应回复 Agent 的分析结果。

## 故障排查

### "app_ticket is empty" 或认证失败

- 确认 `app_id` 和 `app_secret` 正确，且与飞书开放平台中的应用一致。
- 确保应用已发布并审核通过。

### 机器人连接成功但收不到消息

- 检查事件订阅中是否选择了 **WebSocket 模式**（而非 HTTP 回调模式）。
- 确认已添加 `im.message.receive_v1` 事件。
- 确保在**应用功能 → 机器人**中已启用机器人能力。

### 机器人不回复

- 检查是否已添加 `im:message:send_as_bot` 权限并已审批。
- 确认包含该权限的应用版本已发布。

### 日志中出现 "permission denied" 错误

- 进入**权限管理**，确保所有必需权限已添加。
- 添加权限后，必须**创建新版本**并获得审批，权限才会生效。

### WebSocket 连接频繁断开

- `lark-oapi` SDK 会自动处理重连。请检查网络连通性。
- 启用调试日志（`--debug`）查看详细的连接状态。
