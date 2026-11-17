# WeCom AI Bot

Connect a Subagent to a WeCom **AI Bot in API mode**, using an outbound WebSocket
long connection. A public callback URL is not required. An ordinary group message
push webhook is not a bidirectional bot and cannot be used here.

## Setup

1. Create an AI Bot in WeCom, select **API mode**, then **long connection**.
2. Copy its **Bot ID** and **Secret** and set its allowed users.
3. Configure the channel below, or run `datus-gateway configure`.
4. Start `datus-gateway --config /path/to/agent.yml` (add `--daemon` for background mode).
5. Send a text message in a direct conversation, or add the bot to an internal group and @mention it.

```yaml
channels:
  wecom_sales:
    adapter: wecom
    enabled: true
    subagent_id: sales_analyst  # omit for the default chat agent
    extra:
      bot_id: ${WECOM_BOT_ID}
      secret: ${WECOM_BOT_SECRET}
```

The SaaS Access panel saves these credentials encrypted and binds the connection
to the edited Subagent. Saving is required to apply the channel switch. The standalone
gateway reads YAML on startup; restart it after changing configuration.

## Conversation and delivery

- Only text is currently supported. Other message types receive an explanatory reply.
- Direct conversations retain history per user. Group history is isolated per group
  and sender; WeCom does not use Slack-style threads. `/reset` resets that scope.
- Replies begin with a processing placeholder, then update the same stream with
  accumulated Markdown. The final update sets `finish=true`.
- A turn still running after 300 seconds closes its stream and delivers the final
  result using `aibot_send_msg`. Oversized results are also delivered separately,
  split on UTF-8 boundaries. A missing/rejected stream-close ACK does not prevent
  proactive delivery. Platform permissions/rate limits still apply.
- One bot should have one active gateway owner. Backend deployments use their
  existing Redis leader election; standalone instances must not compete for the same bot.
- A server `disconnected_event` means another connection took ownership of the bot.
  The displaced adapter closes and stops automatic reconnection. Resolve duplicate
  bot configurations before explicitly restarting it (or re-enabling the channel).
- Network reconnection uses exponential backoff. Heartbeats are protocol `ping` frames,
  not chat-message polling. Pending callbacks are cancelled on connection loss;
  there is no durable message queue or exactly-once delivery across process restarts.

Advanced YAML-only `extra` options: `ws_url` (default
`wss://openws.work.weixin.qq.com`), `heartbeat_interval` (30 seconds),
`request_timeout` (5 seconds), and `stream_timeout` (300 seconds). Use `ws_url`
only with a trusted endpoint: bot credentials are sent to it.

[WeCom protocol documentation](https://developer.work.weixin.qq.com/document/path/101463)
