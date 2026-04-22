# Feishu (Lark) Adapter

This guide walks you through setting up a Feishu (Lark) bot that connects to Datus Agent via the Datus Gateway IM module. Feishu uses the official `lark-oapi` SDK with **WebSocket long connection**, which means the bot connects outbound — no public URL or webhook endpoint is required.

## Prerequisites

- A Feishu tenant (organization) where you have developer permissions
- Datus Agent installed and configured

## Step 1: Install the SDK

```bash
pip install lark-oapi
```

## Step 2: Create a Self-Built App

1. Go to [Feishu Open Platform](https://open.feishu.cn) and log in with your Feishu account.
2. Click your avatar in the top-right corner, then select **Developer Console** (or go directly to [open.feishu.cn/app](https://open.feishu.cn/app)).
3. Click **Create Custom App**.
4. Fill in the app name (e.g., "Datus Agent") and description.
5. Click **Create**.

!!! note "Lark (International) users"
    If you are using the international version (Lark), go to [open.larksuite.com](https://open.larksuite.com) instead. The setup steps are the same.

## Step 3: Get App Credentials

1. After creating the app, you will be taken to the app dashboard.
2. In the left sidebar, click **Credentials & Basic Info**.
3. Copy the **App ID** and **App Secret**. These are your `FEISHU_APP_ID` and `FEISHU_APP_SECRET`.

!!! warning "Keep your App Secret safe"
    The App Secret is equivalent to a password. Never share it publicly or commit it to version control. Use environment variables instead.

## Step 4: Enable Bot Capability

1. In the left sidebar, go to **Features → Bot**.
2. Toggle the bot capability to **Enabled**.
3. Optionally, set a bot name and avatar.

## Step 5: Configure Event Subscription (WebSocket Mode)

Datus Gateway uses the Feishu WebSocket long connection to receive events. This eliminates the need for a public callback URL.

1. In the left sidebar, go to **Event Subscriptions**.
2. Under **Subscription Mode**, select **WebSocket Mode** (also labeled "Use long connection to receive events").

    !!! tip "Why WebSocket Mode?"
        WebSocket mode is ideal for Datus Gateway because it does not require a publicly accessible server. The bot initiates an outbound WebSocket connection to Feishu's servers, and events are pushed through this connection.

3. Under **Event List**, click **Add Event** and search for:

    | Event Name | Event Key | Description |
    |------------|-----------|-------------|
    | Receive messages | `im.message.receive_v1` | Triggered when the bot receives a message |
    | Message reaction created | `im.message.reaction.created_v1` | Triggered when a user adds a reaction to a message |
    | Message reaction deleted | `im.message.reaction.deleted_v1` | Triggered when a user removes a reaction from a message |

4. Click **Save**.

## Step 6: Configure Permissions

1. In the left sidebar, go to **Permissions & Scopes**.
2. Search for and add the following permissions:

| Permission | Scope Key | Purpose |
|------------|-----------|---------|
| Read messages sent to the bot | `im:message` | Receive messages (base permission) |
| Read p2p messages sent to the bot | `im:message.p2p_msg:readonly` | Receive direct messages |
| Read group messages | `im:message.group_msg` | Receive all group messages (required for thread replies without @bot) |
| Send messages as the bot | `im:message:send_as_bot` | Reply to users |
| Read card info | `cardkit:card:read` | Convert message ID to card ID for streaming cards |
| Write card content | `cardkit:card:write` | Update streaming card content and close streaming mode |
| Get chat info | `im:chat:readonly` | Read group chat metadata (optional) |
| Read user info | `contact:user.base:readonly` | Resolve sender names (optional) |

!!! note "Group message permission"
    `im:message.group_msg` allows the bot to receive **all** group messages, including thread replies without @bot. The bot automatically filters out irrelevant messages — only @bot messages and replies within an existing bot thread are processed. If you prefer a more restrictive scope, you can use `im:message.group_at_msg:readonly` instead, but thread replies without @bot will not be received.

3. Click **Save**.

!!! tip "Batch import"
    In the **Permissions & Scopes** page, click **Batch Enable** and paste the following JSON to add all permissions at once:

    ```json
    {
      "scopes": {
        "tenant": [
          "cardkit:card:read",
          "cardkit:card:write",
          "contact:user.base:readonly",
          "im:chat:readonly",
          "im:message",
          "im:message.p2p_msg:readonly",
          "im:message.group_msg",
          "im:message:send_as_bot",
          "im:resource"
        ]
      }
    }
    ```

!!! note "Required vs Optional"
    `im:message`, `im:message.p2p_msg:readonly`, `im:message.group_msg`, `im:message:send_as_bot`, `cardkit:card:read`, and `cardkit:card:write` are required for full functionality (including group @bot messages, thread replies, and streaming card responses). The other permissions are optional and depend on your use case. Without CardKit permissions, the bot falls back to sending separate messages for each response chunk.

## Step 7: Publish the App

1. In the left sidebar, go to **Version Management & Release**.
2. Click **Create Version**.
3. Fill in the version number and update notes.
4. Click **Submit for Review**.

- **Test tenants**: You can self-approve in test organizations. Go to the **Testing** section to add test members first.
- **Production tenants**: An admin of the organization must approve the app.

!!! info "App availability"
    After approval, the app becomes available to users in the organization. Users can find and add the bot from the Feishu app directory or by searching its name.

## Step 8: Configure Datus Agent

Add the Feishu channel to your `agent.yml`:

```yaml
channels:
  feishu-main:
    adapter: feishu
    enabled: true
    extra:
      app_id: ${FEISHU_APP_ID}
      app_secret: ${FEISHU_APP_SECRET}
```

Set the environment variables:

```bash
export FEISHU_APP_ID="cli_xxxxxxxxxxxx"
export FEISHU_APP_SECRET="xxxxxxxxxxxxxxxxxxxxxxxx"
```

## Configuration Reference

| Key | Required | Description |
|-----|----------|-------------|
| `app_id` | Yes | App ID from Feishu Open Platform |
| `app_secret` | Yes | App Secret from Feishu Open Platform |

## Verify the Connection

Start the gateway and check the logs:

```bash
python -m datus.gateway.main --config conf/agent.yml --debug
```

You should see:

```text
Feishu adapter 'feishu-main' connecting...
Feishu adapter 'feishu-main' started.
```

Send a message to the bot in a Feishu group or direct message. The bot should respond with the agent's analysis.

## Troubleshooting

### "app_ticket is empty" or authentication failure

- Verify that `app_id` and `app_secret` are correct and match the app in the Feishu Open Platform.
- Ensure the app has been published and approved.

### Bot connects but doesn't receive messages

- Check that the **WebSocket Mode** is selected (not HTTP callback mode) under Event Subscriptions.
- Verify the `im.message.receive_v1` event is added.
- Ensure the bot capability is enabled under **Features → Bot**.

### Bot doesn't reply

- Check that the `im:message:send_as_bot` permission has been added and approved.
- Verify the app version with this permission has been published.

### "permission denied" errors in logs

- Go to **Permissions & Scopes** and ensure all required permissions are added.
- After adding permissions, you must **create a new version** and get it approved for the permissions to take effect.

### WebSocket connection keeps dropping

- The `lark-oapi` SDK handles reconnection automatically. Check network connectivity.
- Enable debug logging (`--debug`) to see detailed connection status.
