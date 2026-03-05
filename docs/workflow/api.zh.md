# 工作流 API

## 介绍

Datus Agent Workflow API Service 通过 REST 接口对外提供“自然语言 → SQL”的工作流能力，便于将智能 SQL 生成、执行与工作流管理集成到你的应用中。

## 快速开始

### 启动服务
```bash
# 启动 API 服务
python -m datus.api.server --host 0.0.0.0 --port 8000

# 多进程
python -m datus.api.server --workers 4 --port 8000

# 守护进程（后台）
python -m datus.api.server --daemon --port 8000
```

## 认证

### OAuth2 Client Credentials

API 使用客户端凭证模式：

#### 1）获取 Token
```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=your_client_id&client_secret=your_client_secret&grant_type=client_credentials"
```

#### 2）调用时携带 Token
```bash
curl -X POST "http://localhost:8000/workflows/run" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{"workflow": "fixed", "namespace": "your_db", "task": "Show me users"}'
```

### 配置
创建 `auth_clients.yml`：
```yaml
clients:
  your_client_id: your_client_secret
  another_client: another_secret

jwt:
  secret_key: your-jwt-secret-key-change-in-production
  algorithm: HS256
  expiration_hours: 2
```

## 接口

### 认证
#### POST /auth/token
获取 JWT 访问令牌。

**请求：**
```http
POST /auth/token
Content-Type: application/x-www-form-urlencoded

client_id=your_client_id&client_secret=your_client_secret&grant_type=client_credentials
```

**响应：**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 7200
}
```

### 工作流执行
#### POST /workflows/run
执行工作流，将自然语言转换为 SQL。

**通用请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `workflow` | string | ✅ | 工作流名称（nl2sql、reflection、fixed、metric_to_sql） |
| `namespace` | string | ✅ | 数据库命名空间 |
| `task` | string | ✅ | 自然语言任务描述 |
| `mode` | string | ✅ | 执行模式（sync 或 async） |
| `task_id` | string | ❌ | 自定义任务 ID（幂等） |
| `catalog_name` | string | ❌ | Catalog 名 |
| `database_name` | string | ❌ | 数据库名 |
| `schema_name` | string | ❌ | Schema 名 |
| `current_date` | string | ❌ | 时间表达基准日期 |
| `domain` | string | ❌ | 业务域 |
| `layer1` | string | ❌ | 业务层级 1 |
| `layer2` | string | ❌ | 业务层级 2 |
| `ext_knowledge` | string | ❌ | 额外业务上下文 |

#### 同步模式（mode: "sync"）
**请求头：**
```yaml
Authorization: Bearer your_jwt_token
Content-Type: application/json
```

**请求体：**
```json
{
  "workflow": "nl2sql",
  "namespace": "your_database_namespace",
  "task": "Show me monthly revenue by product category",
  "mode": "sync",
  "catalog_name": "your_catalog",
  "database_name": "your_database"
}
```

**响应：**
```json
{
  "task_id": "client_20240115143000",
  "status": "completed",
  "workflow": "nl2sql",
  "sql": "SELECT DATE_TRUNC('month', order_date) as month, product_category, SUM(amount) as revenue FROM orders WHERE order_date >= '2023-01-01' GROUP BY month, product_category ORDER BY month, revenue DESC",
  "result": [ ... ],
  "metadata": {
    "execution_time": 12.5,
    "nodes_executed": 5,
    "reflection_rounds": 0
  },
  "error": null,
  "execution_time": 12.5
}
```

#### 异步模式（mode: "async"）
**请求头：**
```yaml
Authorization: Bearer your_jwt_token
Content-Type: application/json
Accept: text/event-stream
Cache-Control: no-cache
```

**请求体：**
```json
{
  "workflow": "nl2sql",
  "namespace": "your_database_namespace",
  "task": "Show me monthly revenue by product category",
  "mode": "async",
  "catalog_name": "your_catalog",
  "database_name": "your_database"
}
```

**响应（SSE 流）：**
```
event: started
:data: {"task_id": "client_20240115143000", "workflow": "nl2sql"}
...
event: done
:data: {"task_id": "client_20240115143000", "status": "completed", "total_time": 15.2}
```

### 反馈提交
#### POST /workflows/feedback
提交对本次执行质量的反馈。

**请求：**
```json
{
  "task_id": "client_20240115143000",
  "status": "success"
}
```

**响应：**
```json
{
  "task_id": "client_20240115143000",
  "acknowledged": true,
  "recorded_at": "2024-01-15T14:30:15Z"
}
```

## 工作流类型

### reflection
- 具反思与纠错能力；可适配并重试；适合复杂或不确定问题

### fixed
- 确定性路径；无自适应；适合明确需求

### metric_to_sql
- 基于业务指标；包含时间解析；适合标准化 BI

## 配置

### 服务配置
```bash
python -m datus.api.server \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --reload \
  --debug
```

### 服务参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--host` | 监听地址 | `127.0.0.1` |
| `--port` | 端口 | `8000` |
| `--workers` | 进程数 | `1` |
| `--reload` | 代码变更自动重载 | `False` |
| `--debug` | 调试模式 | `False` |
| `--daemon` | 后台运行 | `False` |

## 最佳实践

### 安全
- 生产环境务必使用强随机的 JWT 秘钥
- 定期轮换客户端凭证
- 上线限流策略
- 使用 HTTPS

### 性能
- 长耗时任务建议用异步模式
- 根据负载合理配置 workers 数量
- 多进程时关注内存占用
- 同步请求在客户端设置超时

### 错误处理
- 检查响应状态码
- 对瞬时失败实现重试
- 异步流注意断连重连
- 记录详细错误日志

## 结语

Workflow API 为你的应用提供强大且灵活的“自然语言 → SQL”集成能力，支持多种执行模式、实时进度、完善认证，助力构建智能数据分析应用。
