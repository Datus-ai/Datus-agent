# API 部署

## 安装

API 服务随主 `datus` 包一起发布,只需安装一次依赖:

```bash
uv sync
```

`datus-api` 控制台脚本会自动注册。

## 启动

### 前台启动（默认）

```bash
datus-api --host 0.0.0.0 --port 8000

# 或通过 uv 运行
uv run datus-api --host 0.0.0.0 --port 8000
```

### 后台守护进程模式

以守护进程方式在后台运行 API 服务：

```bash
# 后台启动
datus-api --daemon --port 8000

# 查看状态
datus-api --action status

# 停止
datus-api --action stop

# 重启
datus-api --action restart
```

所有守护进程命令也支持 `uv run`，例如 `uv run datus-api --daemon --port 8000`。

默认情况下，PID 文件存储在 `~/.datus/run/datus-agent-api.pid`，守护进程日志写入 `logs/datus-agent-api.log`。可通过参数覆盖：

```bash
datus-api --daemon --pid-file /var/run/datus-api.pid --daemon-log-file /var/log/datus-api.log
```

## CLI 参数

| 参数              | 默认值                  | 说明 |
|-------------------|-------------------------|------|
| `--config`        | (自动解析)              | `agent.yml` 路径 |
| `--datasource`     | `default`               | `agent.services.datasources` 中的数据源键名 |
| `--output-dir`    | `./output`              | 生成产物目录 |
| `--log-level`     | `INFO`                  | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `--host`          | `127.0.0.1`             | 监听地址 |
| `--port`          | `8000`                  | 监听端口 |
| `--reload`        | 关闭                    | 文件变更自动重载(仅开发) |
| `--workers`       | `1`                     | uvicorn worker 进程数 |
| `-v`, `--version` | —                       | 打印版本并退出 |
| `--daemon`        | 关闭                    | 以守护进程方式后台运行 |
| `--action`        | `start`                 | 守护进程操作：`start`、`stop`、`restart`、`status` |
| `--pid-file`      | `~/.datus/run/datus-agent-api.pid` | PID 文件路径 |
| `--daemon-log-file` | `logs/datus-agent-api.log` | 守护进程日志文件路径 |

`--reload` 与 `--workers > 1` 互斥，服务会发出告警并退回单 worker 模式。
`--daemon` 与 `--reload` 互斥。

## 环境变量

| 变量                  | 等价参数        | 说明 |
|-----------------------|-----------------|------|
| `DATUS_CONFIG`        | `--config`      | 空字符串触发默认查找 |
| `DATUS_DATASOURCE`     | `--datasource`   | 默认 `default` |
| `DATUS_OUTPUT_DIR`    | `--output-dir`  | 默认 `./output` |
| `DATUS_LOG_LEVEL`     | `--log-level`   | 默认 `INFO` |
| `DATUS_CORS_ORIGINS`  | —               | 逗号分隔来源,默认 `*` |

`DATUS_CORS_ORIGINS` 非 `*` 时,CORS 中间件会启用 `allow_credentials=true`。

## 配置文件解析优先级

`datus-api` 按以下顺序解析 agent 配置文件:

1. 显式设置的 `--config`(或 `DATUS_CONFIG`)
2. 当前工作目录下的 `./conf/agent.yml`
3. `~/.datus/conf/agent.yml`

与 `datus` CLI 行为一致。使用标准的 `~/.datus` 安装路径时无需指定 `--config`。

## 内置端点

启动后,无论路由是否加载成功,以下端点恒可用:

| 路径                | 说明 |
|---------------------|------|
| `GET /`             | 服务 banner 与版本指针 |
| `GET /health`       | 健康检查(无需鉴权) |
| `GET /docs`         | Swagger UI |
| `GET /openapi.json` | OpenAPI 3 规范 |

## curl 快速上手

```bash
# 1. 启动服务
datus-api --port 8000 &

# 2. 健康检查
curl http://127.0.0.1:8000/health

# 3. 列出 catalog(以用户 alice 身份)
curl -H 'X-Datus-User-Id: alice' \
  'http://127.0.0.1:8000/api/v1/catalog/list'

# 4. 流式 chat
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Datus-User-Id: alice' \
  -d '{"message": "上周新增用户数是多少?"}'
```

## 生产部署提示

- 建议在 nginx/traefik 等反向代理后运行,由代理终止 TLS。
- 在反向代理上关闭对 SSE 接口的响应缓冲,确保事件不被延迟下发。
- 多 worker 运行时,请在代理层启用粘性会话(sticky session),使 SSE 续传请求落到仍持有任务的那个 worker。
