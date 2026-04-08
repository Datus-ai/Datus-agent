# API Reference

All v1 endpoints are mounted under the global prefix `/api/v1`. Responses are wrapped in the
[`Result[T]` envelope](introduction.md#response-envelope) and accept the `X-Datus-User-Id` header for per-user
session isolation.

This page covers the **Database**, **Table**, and **Explorer** endpoints. Chat endpoints are documented
separately on the [Chat](chat.md) page.

---

## Database

### `GET /api/v1/catalog/list`

List the catalogs/databases visible in the current namespace.

**Query**:

| Param                 | Type   | Default |
|-----------------------|--------|---------|
| `datasource_id`       | string | current namespace |
| `catalog_name`        | string | `""`    |
| `database_name`       | string | `""`    |
| `schema_name`         | string | `""`    |
| `include_sys_schemas` | bool   | `false` |

**Response**: `Result[DatabasesData]` — `databases: DatabaseInfo[]`.

---

## Table

### `GET /api/v1/table/detail?table=db.schema.tbl`

Return columns, indexes, row count, and description for a fully-qualified table name.

**Response**: `Result[GetTableDetailData]` — `{ table: { name, description, rows, columns, indexes } }`.

### `GET /api/v1/semantic_model?table=db.schema.tbl`

Fetch the SemanticModel YAML configured for a table.

**Response**: `Result[GetSemanticModelData]` — `{ yaml }`.

### `POST /api/v1/semantic_model`

Save or update a SemanticModel YAML for a table.

**Body**: `{ table, yaml }`

**Response**: `Result[dict]`.

### `POST /api/v1/semantic_model/validate`

Validate a SemanticModel YAML without persisting it.

**Body**: `{ table, yaml }`

**Response**: `Result[ValidateSemanticModelData]` — `{ valid, invalid_message? }`.

---

## Explorer

The explorer manages a hierarchical subject tree containing four kinds of leaves: directories, metrics,
reference SQLs, and knowledge entries. All node identifiers are list-of-strings paths from the root, e.g.
`["sales", "core", "active_users"]`.

### Subject tree

| Method & path                  | Body                                             | Description |
|--------------------------------|--------------------------------------------------|-------------|
| `GET /subject/list`            | —                                                | Return the full nested tree |
| `POST /subject/create`         | `{ subject_path }`                               | Create a directory under `subject_path` |
| `POST /subject/rename`         | `{ type, subject_path, new_subject_path }`       | Rename or move a node |
| `DELETE /subject/delete`       | `{ type, subject_path }`                         | Delete a node |

`type` is one of `directory`, `metric`, `reference_sql`, `knowledge`.

### Metrics

| Method & path                  | Body                       | Description |
|--------------------------------|----------------------------|-------------|
| `POST /subject/metric`         | `{ subject_path }`         | Get metric `{ name, yaml }` |
| `POST /subject/metric/create`  | `{ subject_path, yaml }`   | Create metric from YAML |
| `POST /subject/metric/edit`    | `{ subject_path, yaml }`   | Update metric YAML |

### Reference SQL

| Method & path                          | Body                                                     | Description |
|----------------------------------------|----------------------------------------------------------|-------------|
| `POST /subject/reference_sql`          | `{ subject_path }`                                       | Get `{ name, sql, summary, search_text }` |
| `POST /subject/reference_sql/create`   | `{ subject_path, name, sql, summary, search_text }`      | Create entry |
| `POST /subject/reference_sql/edit`     | `{ subject_path, name, sql, summary, search_text }`      | Update entry |

### Knowledge

| Method & path                       | Body                                                              | Description |
|-------------------------------------|-------------------------------------------------------------------|-------------|
| `POST /subject/knowledge`           | `{ subject_path }`                                                | Get `{ name, search_text, explanation }` |
| `POST /subject/knowledge/create`    | `{ subject_path, name, search_text, explanation }`                | Create entry |
| `POST /subject/knowledge/edit`      | `{ subject_path, search_text, explanation }`                      | Update entry |
