# API 介绍

Datus REST API 将 Agent 的 chat 循环、知识库 explorer、数据库 catalog 与语义模型管理以 HTTP 接口的形式
对外暴露。通过 `datus-api` 命令启动。

API 服务与 `datus` CLI、`datus-mcp` MCP 服务共享同一套配置、知识库与 Agent 能力 — 三种入口,同一个引擎。

| 入口 | 适用场景 |
|------|----------|
| **CLI**(`datus`) | 本地交互式开发 |
| **MCP**(`datus-mcp`) | 嵌入到其他 Agent 中 |
| **REST API**(`datus-api`) | Web 前端、服务化、自动化 |

## 鉴权模型

开源版本采用基于请求头的身份识别方案,无 token 概念。调用方通过 `X-Datus-User-Id` 请求头标识自己,值需匹配
`^[A-Za-z0-9_-]+$`。未提供时请求落入默认未隔离的会话;提供时,该 user id 用于隔离各用户的会话与工具状态。

```
X-Datus-User-Id: alice
```

Namespace 隔离由 `--namespace` CLI 参数(或 `DATUS_NAMESPACE` 环境变量)单独控制,决定加载 `agent.yml`
中的哪个 namespace 的数据库与知识库。

## 响应封装 {#response-envelope}

绝大多数 JSON 响应都包裹在统一的 `Result[T]` 结构中:

```json
{
  "success": true,
  "data":    { ... },
  "errorCode":    null,
  "errorMessage": null
}
```

| 字段           | 类型   | 说明 |
|----------------|--------|------|
| `success`      | bool   | 成功为 `true`,失败为 `false` |
| `data`         | object | 接口专属业务数据,失败时为 `null` |
| `errorCode`    | string | 稳定的机器可读错误码 |
| `errorMessage` | string | 人类可读错误描述 |

流式接口(`/chat/stream`、`/chat/resume`)返回 `text/event-stream`,不使用 `Result` 包装,详见
[Chat](chat.zh.md)。

## 全局 URL 前缀

所有 v1 接口位于 `/api/v1` 前缀下;健康检查 `/health` 与 OpenAPI/Swagger UI(`/docs`、`/openapi.json`)
位于应用根路径。

## 下一步

- [部署](deployment.zh.md) — 安装并启动 API 服务
- [Chat](chat.zh.md) — Chat 接口与 SSE 流式协议
- [知识库](knowledge_base.zh.md) — 知识库构建与平台文档接口（SSE 流式）
