# Slack Adapter

This guide walks you through setting up a Slack bot that connects to Datus Agent via the Claw IM gateway. Slack uses **Socket Mode**, which establishes an outbound WebSocket connection from your server — no public URL or webhook endpoint is required.

## Prerequisites

- A Slack workspace where you have permission to install apps
- Datus Agent installed and configured

## Step 1: Install the SDK

```bash
pip install "slack-sdk[socket_mode]"
```

## Step 2: Create a Slack App (From Manifest)

Using a manifest is the fastest way to create a correctly configured Slack App. It pre-configures Socket Mode, bot scopes, and event subscriptions in one step.

1. Go to [api.slack.com/apps/new](https://api.slack.com/apps/new).
2. Click **Create New App** → **From an app manifest**.
3. Select your workspace and click **Next**.
4. Paste the following manifest (JSON tab):

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

5. Review the summary and click **Create**.

!!! tip "Customize the manifest"
    You can change `display_information.name` to any name you prefer. Add extra scopes (e.g., `reactions:write`, `files:write`) if needed for your use case.

## Step 3: Generate Tokens

After the app is created, you need two tokens:

### App-Level Token (for Socket Mode)

1. Go to **Settings → Basic Information**.
2. Scroll down to **App-Level Tokens** and click **Generate Token and Scopes**.
3. Name it (e.g., `claw-socket-token`), add the scope `connections:write`, and click **Generate**.
4. Copy the token (starts with `xapp-`). This is your `SLACK_APP_TOKEN`.

### Bot Token (for sending messages)

1. Go to **OAuth & Permissions** in the left sidebar.
2. Click **Install to Workspace** and approve the permissions.
3. Copy the **Bot User OAuth Token** (starts with `xoxb-`). This is your `SLACK_BOT_TOKEN`.

!!! tip "Token formats"
    - App-Level Token: starts with `xapp-`
    - Bot User OAuth Token: starts with `xoxb-`

    If your token starts with `xoxp-`, you copied the User OAuth Token by mistake.

## Step 4: Invite the Bot to Channels

In Slack, invite the bot to each channel where you want it to respond:

```bash
/invite @YourBotName
```

The bot will only receive messages from channels it has been invited to.

## Step 5: Configure Datus Agent

Add the Slack channel to your `agent.yml`:

```yaml
channels:
  slack-main:
    adapter: slack
    enabled: true
    extra:
      app_token: ${SLACK_APP_TOKEN}       # xapp-... (App-Level Token)
      bot_token: ${SLACK_BOT_TOKEN}       # xoxb-... (Bot User OAuth Token)
```

Set the environment variables:

```bash
export SLACK_APP_TOKEN="xapp-1-..."
export SLACK_BOT_TOKEN="xoxb-..."
```

## Configuration Reference

| Key | Required | Description |
|-----|----------|-------------|
| `app_token` | Yes | App-Level Token for Socket Mode (`xapp-...`) |
| `bot_token` | Yes | Bot User OAuth Token (`xoxb-...`) |

## Verify the Connection

Start the Claw gateway and check the logs:

```bash
python -m datus.claw.main --config conf/agent.yml --debug
```

You should see:

```text
Slack adapter 'slack-main' connecting...
Slack adapter 'slack-main' started.
```

Send a message in an invited channel. The bot should respond with the agent's analysis.

## Scope and Event Checklist

If you created the app from the manifest above, all scopes and events are already configured. Use this checklist to verify or if you need to adjust manually.

### Required Bot Token Scopes

| Scope | Purpose |
|-------|---------|
| `chat:write` | Send messages to channels |
| `channels:history` | Read messages in public channels |
| `channels:read` | View basic channel info |
| `groups:history` | Read messages in private channels |
| `groups:read` | View basic private channel info |
| `im:history` | Read direct messages |
| `im:read` | View basic DM info |
| `im:write` | Start direct messages with users |
| `mpim:history` | Read group direct messages |
| `mpim:read` | View group DM info |
| `mpim:write` | Start group DMs |
| `users:read` | View user info |
| `app_mentions:read` | Receive @mention events |

### Required Bot Events

| Event | Trigger |
|-------|---------|
| `message.channels` | Message posted in a public channel |
| `message.groups` | Message posted in a private channel |
| `message.im` | Direct message sent to the bot |
| `message.mpim` | Message in a group DM |
| `app_mention` | Bot is @mentioned |

??? info "Manual setup (without manifest)"
    If you prefer to configure the app manually instead of using the manifest:

    1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From scratch**.
    2. **Enable Socket Mode**: **Settings → Socket Mode** → toggle on → generate App-Level Token with `connections:write` scope.
    3. **Add Bot Scopes**: **OAuth & Permissions → Bot Token Scopes** → add each scope from the table above.
    4. **Enable Events**: **Event Subscriptions** → toggle on → add each event from the table above under **Subscribe to bot events** → **Save Changes**.
    5. **Install**: **OAuth & Permissions** → **Install to Workspace** → copy the Bot User OAuth Token.

## Troubleshooting

### "invalid_auth" error

- Verify `bot_token` starts with `xoxb-` and is not expired.
- Re-install the app to the workspace to regenerate the token.

### Bot connects but doesn't receive messages

- Ensure the bot is invited to the channel (`/invite @BotName`).
- Verify event subscriptions are enabled and the four `message.*` events are added.
- Check that Socket Mode is enabled.

### "missing_scope" error

- Go to **OAuth & Permissions → Bot Token Scopes** and add the missing scope.
- Reinstall the app after adding new scopes.

### Messages from bot trigger loops

- The adapter automatically ignores messages with subtypes (edits, deletes, bot messages). If you still see loops, check that the bot is not responding to its own messages.
