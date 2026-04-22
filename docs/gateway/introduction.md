# IM Gateway

Datus Gateway is the IM (Instant Messaging) gateway module for Datus Agent. It connects Datus to chat platforms such as Slack and Feishu (Lark), enabling users to interact with the data analysis agent directly from their team communication tools.

## Key Design Principles

- **Outbound long connections only** — Datus Gateway actively connects to each IM platform using WebSocket, Socket Mode, or Stream SDK. No webhook endpoint or public URL is needed.
- **Real-time streaming** — Agent responses (thinking, tool calls, SQL, markdown) are streamed back to the chat as they are generated.
- **Session management** — Each conversation (group/DM/thread) automatically gets a persistent session. Users can reset it with `/new` or `/reset`.
- **Datasource & subagent routing** — Each channel can override the default datasource or route messages to a specific sub-agent.

## Supported Platforms

| Platform | Adapter | SDK | Connection Type | Status |
|----------|---------|-----|-----------------|--------|
| Slack | `slack` | `slack-sdk[socket_mode]` | Socket Mode (WebSocket) | Ready |
| Feishu (Lark) | `feishu` | `lark-oapi` | WebSocket long connection | Ready |

## Architecture

```text
┌──────────────┐     long connection     ┌──────────────────┐
│  IM Platform │ ◄──────────────────────► │  ChannelAdapter   │
│  (Slack /    │                          │  (per platform)   │
│   Feishu)    │                          └────────┬─────────┘
│              │                                   │
└──────────────┘                                   │ InboundMessage
                                                   ▼
                                          ┌──────────────────┐
                                          │  ChannelBridge    │
                                          │  - session mgmt   │
                                          │  - chat commands   │
                                          └────────┬─────────┘
                                                   │ StreamChatInput
                                                   ▼
                                          ┌──────────────────┐
                                          │ ChatTaskManager   │
                                          │  (agentic loop)   │
                                          └────────┬─────────┘
                                                   │ SSE events
                                                   ▼
                                          ┌──────────────────┐
                                          │  OutboundMessage  │
                                          │  → adapter.send() │
                                          └──────────────────┘
```

Each adapter maintains a persistent connection to its platform. When a message arrives, the `ChannelBridge` converts it into a `StreamChatInput`, feeds it to `ChatTaskManager`, and streams each SSE event back to the user as an `OutboundMessage`.

## Installation

Install the SDK for each platform you want to connect:

```bash
# Slack
pip install "slack-sdk[socket_mode]"

# Feishu (Lark)
pip install lark-oapi
```

You only need to install the SDK(s) for the platform(s) you use. Datus Gateway will raise a clear error if a required SDK is missing.

## Configuration

Add a `channels` section to your `agent.yml` file. Each key under `channels` defines a channel instance.

### General Structure

```yaml
channels:
  my-channel:
    adapter: slack          # Required: feishu | slack
    enabled: true           # Optional: default true
    datasource: my_datasource # Optional: override default datasource
    subagent_id: agent_01   # Optional: route to a specific sub-agent
    extra:                  # Required: adapter-specific credentials
      # ... platform-specific keys
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `adapter` | string | Yes | Adapter type: `feishu` or `slack` |
| `enabled` | bool | No | Whether this channel is active (default: `true`) |
| `datasource` | string | No | Override the gateway's default datasource |
| `subagent_id` | string | No | Route messages to a specific sub-agent |
| `extra` | dict | Yes | Adapter-specific configuration (tokens, keys, etc.) |

### Slack

```yaml
channels:
  slack-main:
    adapter: slack
    enabled: true
    extra:
      app_token: ${SLACK_APP_TOKEN}       # xapp-... (Socket Mode token)
      bot_token: ${SLACK_BOT_TOKEN}       # xoxb-... (Bot User OAuth Token)
```

### Feishu (Lark)

```yaml
channels:
  feishu-main:
    adapter: feishu
    enabled: true
    extra:
      app_id: ${FEISHU_APP_ID}
      app_secret: ${FEISHU_APP_SECRET}
```

!!! warning "Keep secrets out of config files"
    Always use `${ENV_VAR}` substitution for tokens and secrets. Never commit plaintext credentials to version control.

## Platform Setup Guides

For detailed step-by-step instructions on configuring each IM platform, see the dedicated adapter guides:

- [Slack Setup Guide](slack.md) — Create a Slack App, enable Socket Mode, configure scopes and events
- [Feishu (Lark) Setup Guide](feishu.md) — Create a self-built app, enable bot capability, configure WebSocket events


## Running the Gateway

### Foreground (default)

Start the gateway in the foreground:

```bash
datus-gateway --config conf/agent.yml

# Or via uv
uv run datus-gateway --config conf/agent.yml
```

### Daemon (background) Mode

Run the gateway as a background daemon:

```bash
# Start in background
datus-gateway --daemon

# Check status
datus-gateway --action status

# Stop
datus-gateway --action stop

# Restart
datus-gateway --action restart
```

All daemon commands also work with `uv run`, e.g. `uv run datus-gateway --daemon`.

By default, the PID file is stored at `~/.datus/run/datus-gateway.pid` and daemon logs are written to `logs/datus-gateway.log`. You can override these paths:

```bash
datus-gateway --daemon --pid-file /var/run/datus-gateway.pid --daemon-log-file /var/log/datus-gateway.log
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `./conf/agent.yml` | Path to agent configuration file |
| `--datasource` | `default` (or `DATUS_DATASOURCE` env) | Default datasource for all channels |
| `--host` | `0.0.0.0` | Health-check server bind host |
| `--port` | `9000` | Health-check server bind port |
| `--debug` | `false` | Enable debug logging |
| `--log-level` | `INFO` (or `DATUS_LOG_LEVEL` env) | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `--daemon` | `false` | Run in background as a daemon |
| `--action` | `start` | Daemon action: `start`, `stop`, `restart`, `status` |
| `--pid-file` | `~/.datus/run/datus-gateway.pid` | PID file path |
| `--daemon-log-file` | `logs/datus-gateway.log` | Daemon log file path |

## Features

### Real-Time Streaming

Datus Gateway streams every stage of the agent's processing back to the chat:

- **Thinking** — The agent's reasoning process
- **Tool calls** — Which tools are being invoked and their results
- **SQL** — Generated SQL queries displayed as code blocks
- **Markdown** — Final answers and explanations
- **Errors** — Any errors are reported immediately

### Chat Commands

Datus Gateway provides built-in slash commands that are intercepted before messages reach the agentic loop. You can type them with or without the `/` prefix.

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help` | — | Show all available commands |
| `/new` | `/reset`, `/clear` | Reset the current conversation session |
| `/verbose [level]` | — | Get or set the verbosity level for the current conversation |

#### `/help`

Displays a list of all registered commands with their descriptions.

#### `/new` / `/reset` / `/clear`

Resets the current conversation session and starts a fresh one. The bot will confirm the reset and display the new session ID.

#### `/verbose [level]`

Controls how much detail the bot shows during processing. When called without an argument, it displays the current verbosity level.

| Level | Aliases | Behavior |
|-------|---------|----------|
| `quiet` | `off` | Thinking + final output only (no tool calls) |
| `brief` | `on` | Thinking + tool summaries + final output |
| `detail` | `full` | Thinking + tool parameters/results + final output |

Examples:

```text
/verbose           # Show current verbosity level
/verbose quiet     # Only show thinking and final answer
/verbose brief     # Show tool call summaries
/verbose detail    # Show full tool call details
```

### Session Management

Each conversation (group chat, DM, or thread) automatically maps to a persistent session. To start a fresh session, send `/new`, `/reset`, or `/clear`.

The bot will confirm the reset with the new session ID.

### Datasource Override

Each channel can specify a `datasource` in its configuration to override the gateway's default datasource. This allows different channels to query different databases.

### Subagent Routing

Use the `subagent_id` field to route messages from a specific channel to a dedicated sub-agent, enabling specialized behavior per channel.

## Troubleshooting

### SDK Not Installed

```text
ImportError: slack_sdk is required for the Slack adapter. Install it with: pip install slack-sdk[socket_mode]
```

Install the required SDK package for your platform. See the [Installation](#installation) section.

### Invalid Credentials

If the bot connects but immediately disconnects or logs authentication errors, verify that:

- Slack: `app_token` starts with `xapp-` and `bot_token` starts with `xoxb-`
- Feishu: `app_id` and `app_secret` match the self-built app credentials

### Bot Not Receiving Messages

- **Slack**: Ensure the bot is invited to the channel and event subscriptions are enabled.
- **Feishu**: Ensure the WebSocket subscription mode is selected and `im.message.receive_v1` event is added.

### Debug Logging

Enable verbose logging to diagnose connection issues:

```bash
datus-gateway --config conf/agent.yml --debug
```

Or set the environment variable:

```bash
export DATUS_LOG_LEVEL=DEBUG
```

## Next Steps

- [Configuration](../configuration/introduction.md) — Full agent configuration reference
- [Chatbot](../web_chatbot/introduction.md) — Web-based chatbot interface
- [Datasources](../configuration/datasources.md) — Datasource configuration details
- [Subagent](../subagent/introduction.md) — Sub-agent configuration and customization
