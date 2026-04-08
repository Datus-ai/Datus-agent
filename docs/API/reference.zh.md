# API 参考

所有 v1 接口挂载在全局前缀 `/api/v1` 下。响应均使用
[`Result[T]` 封装](introduction.zh.md#响应封装),并支持 `X-Datus-User-Id` 请求头进行按用户的会话隔离。

本页覆盖 **Database**、**Table**、**Explorer** 三组接口。Chat 接口见 [Chat](chat.zh.md) 独立页面。

---

## Database

### `GET /api/v1/catalog/list`

列出当前 namespace 可见的 catalog/数据库。

**Query**:

| 参数                  | 类型   | 默认值 |
|-----------------------|--------|--------|
| `datasource_id`       | string | 当前 namespace |
| `catalog_name`        | string | `""`   |
| `database_name`       | string | `""`   |
| `schema_name`         | string | `""`   |
| `include_sys_schemas` | bool   | `false`|

**响应**:`Result[DatabasesData]` — `databases: DatabaseInfo[]`。

---

## Table

### `GET /api/v1/table/detail?table=db.schema.tbl`

返回指定全限定表名的字段、索引、行数与描述。

**响应**:`Result[GetTableDetailData]` — `{ table: { name, description, rows, columns, indexes } }`。

### `GET /api/v1/semantic_model?table=db.schema.tbl`

获取某张表配置的 SemanticModel YAML。

**响应**:`Result[GetSemanticModelData]` — `{ yaml }`。

### `POST /api/v1/semantic_model`

保存或更新某张表的 SemanticModel YAML。

**Body**:`{ table, yaml }`

**响应**:`Result[dict]`。

### `POST /api/v1/semantic_model/validate`

校验 SemanticModel YAML 但不持久化。

**Body**:`{ table, yaml }`

**响应**:`Result[ValidateSemanticModelData]` — `{ valid, invalid_message? }`。

---

## Explorer

Explorer 管理一个分层的主题树,叶子节点有四种:目录、metric、reference SQL、knowledge。
所有节点使用从根开始的字符串数组路径作为标识,例如 `["sales", "core", "active_users"]`。

### 主题树

| 方法与路径                     | Body                                             | 说明 |
|--------------------------------|--------------------------------------------------|------|
| `GET /subject/list`            | —                                                | 返回完整嵌套树 |
| `POST /subject/create`         | `{ subject_path }`                               | 在 `subject_path` 下创建目录 |
| `POST /subject/rename`         | `{ type, subject_path, new_subject_path }`       | 重命名或移动节点 |
| `DELETE /subject/delete`       | `{ type, subject_path }`                         | 删除节点 |

`type` 取值:`directory`、`metric`、`reference_sql`、`knowledge`。

### Metric

| 方法与路径                     | Body                       | 说明 |
|--------------------------------|----------------------------|------|
| `POST /subject/metric`         | `{ subject_path }`         | 获取 metric `{ name, yaml }` |
| `POST /subject/metric/create`  | `{ subject_path, yaml }`   | 通过 YAML 创建 metric |
| `POST /subject/metric/edit`    | `{ subject_path, yaml }`   | 更新 metric YAML |

### Reference SQL

| 方法与路径                             | Body                                                     | 说明 |
|----------------------------------------|----------------------------------------------------------|------|
| `POST /subject/reference_sql`          | `{ subject_path }`                                       | 获取 `{ name, sql, summary, search_text }` |
| `POST /subject/reference_sql/create`   | `{ subject_path, name, sql, summary, search_text }`      | 创建条目 |
| `POST /subject/reference_sql/edit`     | `{ subject_path, name, sql, summary, search_text }`      | 更新条目 |

### Knowledge

| 方法与路径                          | Body                                                              | 说明 |
|-------------------------------------|-------------------------------------------------------------------|------|
| `POST /subject/knowledge`           | `{ subject_path }`                                                | 获取 `{ name, search_text, explanation }` |
| `POST /subject/knowledge/create`    | `{ subject_path, name, search_text, explanation }`                | 创建条目 |
| `POST /subject/knowledge/edit`      | `{ subject_path, search_text, explanation }`                      | 更新条目 |
