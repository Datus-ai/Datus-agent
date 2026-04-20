# 知识库 API

知识库接口负责 KB 构建操作，通过 Server-Sent Events (SSE) 实时推送进度。所有接口位于 `/api/v1/kb` 前缀下。

## KB 组件构建

### `POST /api/v1/kb/bootstrap`

启动知识库组件构建，通过 SSE 流式推送进度。组件包括 metadata、semantic model、metrics、ext knowledge 和 reference SQL。

**请求体**：

| 字段                  | 类型      | 默认值          | 说明 |
|----------------------|----------|----------------|------|
| `components`         | string[] | _（必填）_       | 要构建的组件：`metadata`、`semantic_model`、`metrics`、`ext_knowledge`、`reference_sql` |
| `strategy`           | string   | `incremental`  | `check`（仅检查）、`overwrite`（重建）、`incremental`（增量更新） |
| `schema_linking_type`| string   | `full`         | metadata 专用：`table`、`view`、`mv`、`full` |
| `catalog`            | string   | `""`           | metadata catalog 过滤（Snowflake、StarRocks） |
| `database_name`      | string   | `""`           | metadata 数据库过滤 |
| `success_story`      | string?  | `null`         | 项目根目录的相对路径，指向 success-story CSV |
| `subject_tree`       | string[]?| `null`         | 预定义层级分类 |
| `sql_dir`            | string?  | `null`         | 项目根目录的相对路径，指向 `.sql` 文件目录 |
| `ext_knowledge`      | string?  | `null`         | 项目根目录的相对路径，指向外部知识 CSV |

**响应**：`text/event-stream`，参见下方 [SSE 事件格式](#sse-events)。

### `POST /api/v1/kb/bootstrap/{stream_id}/cancel`

取消正在运行的 KB 构建流。

**响应**：`Result[dict]`，`data = { stream_id, cancelled: true|false }`。

---

<a id="platform-documentation-bootstrap"></a>

## 平台文档构建

### `POST /api/v1/kb/bootstrap-docs`

启动平台文档构建，通过 SSE 流式推送进度。从 GitHub 仓库、网站或本地目录抓取文档并写入平台文档向量库。

**请求体**：

| 字段              | 类型      | 默认值       | 说明 |
|-------------------|----------|-------------|------|
| `platform`        | string   | _（必填）_   | 平台名称（如 `snowflake`、`duckdb`、`starrocks`） |
| `build_mode`      | string   | `overwrite` | `check`（仅查看状态）或 `overwrite`（重建） |
| `pool_size`       | int      | `4`         | 线程池大小（1–16） |
| `source_type`     | string?  | `null`      | `github`、`website` 或 `local` |
| `source`          | string?  | `null`      | GitHub 仓库 `owner/repo`、URL 或本地路径 |
| `version`         | string?  | `null`      | 文档版本（未提供时自动检测） |
| `github_ref`      | string?  | `null`      | GitHub 分支 / tag / commit |
| `github_token`    | string?  | `null`      | GitHub API token |
| `paths`           | string[]?| `null`      | 需要抓取的路径列表（GitHub 专用） |
| `chunk_size`      | int?     | `null`      | 分块目标大小（字符数） |
| `max_depth`       | int?     | `null`      | 网站最大爬取深度 |
| `include_patterns`| string[]?| `null`      | 包含规则（正则） |
| `exclude_patterns`| string[]?| `null`      | 排除规则（正则） |

仅 `platform` 为必填。其余字段未提供时从 `agent.yml` 中的 `agent.document.<platform>` 配置读取。
参见 [平台文档 — 在 agent.yml 中配置](../knowledge_base/platform_doc.zh.md#agent-yml-configuration)。

**响应**：`text/event-stream`，参见下方 [SSE 事件格式](#sse-events)。

**示例**：

```bash
# 检查现有存储状态
curl -N -X POST http://localhost:8000/api/v1/kb/bootstrap-docs \
  -H "Content-Type: application/json" \
  -d '{"platform": "starrocks", "build_mode": "check"}'

# 从 GitHub 完整重建
curl -N -X POST http://localhost:8000/api/v1/kb/bootstrap-docs \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "starrocks",
    "source": "StarRocks/starrocks",
    "source_type": "github",
    "github_ref": "main",
    "paths": ["docs/en"],
    "build_mode": "overwrite"
  }'
```

### `POST /api/v1/kb/bootstrap-docs/{stream_id}/cancel`

取消正在运行的平台文档构建流。

**响应**：`Result[dict]`，`data = { stream_id, cancelled: true|false }`。

---

<a id="sse-events"></a>

## SSE 事件格式

两个 bootstrap 接口均通过 Server-Sent Events 推送进度。每个事件格式：

```
id: <stream_id>
event: <stage>
data: <json>
```

JSON 载荷结构（`BootstrapKbEvent`）：

| 字段         | 类型    | 说明 |
|-------------|--------|------|
| `stream_id` | string | 唯一流标识 |
| `component` | string | 组件名称（`metadata`、`semantic_model`、`platform_doc`、`all` 等） |
| `stage`     | string | 生命周期阶段（见下表） |
| `message`   | string?| 可读状态消息 |
| `error`     | string?| 错误信息（失败时） |
| `progress`  | object?| `{ total, completed, failed }` 进度计数 |
| `payload`   | object?| 附加数据（组件相关的结果） |
| `timestamp` | string | ISO 格式时间戳 |

### 生命周期阶段

| 阶段               | 含义 |
|-------------------|------|
| `task_started`    | 组件构建已启动 |
| `task_validated`  | 项目已验证，总数确认 |
| `task_processing` | 组件正在处理 |
| `task_completed`  | 组件构建完成 |
| `task_failed`     | 组件构建失败 |
| `item_started`    | 单个项目开始处理 |
| `item_completed`  | 单个项目处理完成 |
| `item_failed`     | 单个项目处理失败 |

### SSE 流示例

```
id: abc-123
event: task_started
data: {"stream_id":"abc-123","component":"platform_doc","stage":"task_started","timestamp":"2025-01-01T00:00:00"}

id: abc-123
event: task_validated
data: {"stream_id":"abc-123","component":"platform_doc","stage":"task_validated","progress":{"total":1036,"completed":0,"failed":0},"timestamp":"2025-01-01T00:00:01"}

id: abc-123
event: item_completed
data: {"stream_id":"abc-123","component":"platform_doc","stage":"item_completed","message":"docs/en/intro.md","timestamp":"2025-01-01T00:00:02"}

id: abc-123
event: task_completed
data: {"stream_id":"abc-123","component":"platform_doc","stage":"task_completed","message":"Processed 1036 docs, 5200 chunks","payload":{"platform":"starrocks","version":"3.5.15","total_docs":1036,"total_chunks":5200},"timestamp":"2025-01-01T00:01:30"}
```

### 取消操作

向 cancel 端点 POST 即可取消正在运行的构建，`stream_id` 从 SSE 事件中获取：

```bash
curl -X POST http://localhost:8000/api/v1/kb/bootstrap-docs/abc-123/cancel
```
