# Storage Schema Definitions and Field Structures - Comprehensive Report

## Current Status (Branch: for_saas)

### Git Changes
- Modified file: `datus/storage/subject_tree/store.py` (formatting changes in TableDefinition)
- Staged changes exist for the same file

---

## 1. BaseEmbeddingStore (Core Infrastructure)

**Location:** `datus/storage/base.py` (lines 40-621)

### Support for Multi-tenant/SaaS Features
The base class already has built-in support for:
- `extra_fields`: Optional list of PyArrow fields to add to schema
- `default_values`: Dictionary of field names to default values
- `scope_indices`: List of fields to create scalar indices on
- `table_prefix`: Prefix for table names (e.g., "tb_" for SaaS)

### Key Constructor Parameters
```python
def __init__(self, 
    table_name: str,
    embedding_model: EmbeddingModel,
    schema: Optional[pa.Schema] = None,
    vector_source_name: str = "definition",
    vector_column_name: str = "vector",
    unique_columns: Optional[List[str]] = None,
    db: Optional[VectorDatabase] = None,
    table_prefix: str = "",
    extra_fields: Optional[List[pa.Field]] = None,           # SaaS: multi-tenant fields
    default_values: Optional[Dict[str, Any]] = None,         # SaaS: auto-fill on writes
    scope_indices: Optional[List[str]] = None,               # SaaS: indices for scope fields
)
```

### Multi-tenant Field Handling
1. **Schema Extension** (lines 119-120):
   - If `extra_fields` provided, they're appended to base schema
   - Example: `workspace_id`, `creator_id`, `updator_id`

2. **Default Values Application** (lines 168-175, 336, 355, 374):
   - `_apply_default_values()` fills missing fields before writes
   - Called in: `store_batch()`, `store()`, `upsert_batch()`
   - Ensures consistent tenant data on every write

3. **Scope Filtering** (lines 177-184):
   - `_apply_scope_filter()` combines user WHERE with scope filter
   - Applied to all queries to ensure data isolation

4. **Scalar Index Creation** (lines 156-158, 582-590):
   - Indices created on scope fields at table initialization
   - Improves query performance for tenant filtering

---

## 2. Storage Registry Configuration System

**Location:** `datus/storage/registry.py` (lines 1-234)

### Two-Level Configuration Architecture

#### 1. Deployment-Level (Application Startup)
Set once via `configure_storage_defaults()`:
```python
configure_storage_defaults(
    scope_fields=["workspace_id"],          # Read-time filters
    table_prefix="tb_",                     # Schema structure
    extra_fields=[                          # Schema structure
        pa.field("workspace_id", pa.string()),
        pa.field("creator_id", pa.string()),
        pa.field("updator_id", pa.string()),
    ],
)
```

**What it does:**
- Configures ALL future storage instances with these defaults
- `extra_fields` are appended to every store's schema
- `table_prefix` applied to all table names
- `scope_fields` also become `scope_indices` (line 78)

#### 2. Request-Level (Per Request)
Set via `agent_config.request_context`:
```python
agent_config.request_context = {
    "workspace_id": "ws_abc",
    "creator_id": "user_123",
    "updator_id": "user_123"
}
```

**Behavior:**
- ALL request_context fields are auto-filled on writes
- Only `scope_fields` are used as WHERE filters on reads
- Creates "scoped views" for data isolation

### Storage Instance Creation Paths

1. **Singleton via `get_storage()`** (lines 89-106):
   - Caches per (factory_name, namespace)
   - Applies deployment defaults automatically

2. **Scoped View via `get_rag_storage()`** (lines 109-136):
   - Used by RAG classes (ReferenceSqlRAG, MetricRAG, etc.)
   - Creates shallow copy with independent scope + write defaults
   - Shares underlying db/table/locks with singleton

3. **Scoped View Creation** (lines 139-167):
   - `create_scoped_view()` creates shallow copy
   - Merges write_defaults: `{**storage._default_values, **write_defaults}`
   - Ensures singleton and view are in sync

---

## 3. Schema Definitions by Store

### 3.1 ReferenceSQL Store
**File:** `datus/storage/reference_sql/store.py` (lines 17-43)

**Base Columns** (via `base_schema_columns()`):
- `name`: string
- `subject_id`: int64
- `created_at`: string

**Additional Columns:**
- `id`: string (unique)
- `sql`: string (SQL query)
- `comment`: string
- `summary`: string
- `search_text`: string (embedding source)
- `filepath`: string
- `tags`: string
- `vector`: list[float32] (embedding_dim)

**Key Features:**
- Subject-based: extends `BaseSubjectEmbeddingStore`
- Unique on `id`
- Creates indices: id, name, filepath, subject_index
- Full-text search on: sql, name, summary, tags, search_text

---

### 3.2 Metric Store
**File:** `datus/storage/metric/store.py` (lines 18-53)

**Base Columns:**
- `name`, `subject_id`, `created_at` (from `base_schema_columns()`)

**Additional Columns:**
- `id`: string (unique, e.g., "metric:dau")
- `semantic_model_name`: string
- `description`: string (embedding source)
- `vector`: list[float32]
- `metric_type`: string (simple|derived|ratio|cumulative)
- `measure_expr`: string
- `base_measures`: list[string]
- `dimensions`: list[string]
- `entities`: list[string]
- `catalog_name`, `database_name`, `schema_name`: string
- `sql`: string
- `yaml_path`: string
- `updated_at`: timestamp(ms)

**Key Features:**
- Subject-based
- Unique on `id`
- Creates indices: semantic_model_name, id, catalog_name, database_name, schema_name, subject_index
- Full-text search on: description, name

---

### 3.3 Semantic Model Store
**File:** `datus/storage/semantic_model/store.py` (lines 19-70)

**Columns (No base_schema_columns):**
- `id`: string (unique, e.g., "table:orders", "column:orders.amount")
- `kind`: string (table|column|entity, NOT metric)
- `name`: string (physical name)
- `fq_name`: string (fully qualified)
- `semantic_model_name`: string
- `catalog_name`, `database_name`, `schema_name`: string
- `table_name`: string (context for filtering)
- `description`: string (embedding source)
- `vector`: list[float32]
- `is_dimension`, `is_measure`, `is_entity_key`, `is_deprecated`: bool
- `expr`: string (SQL expression)
- `column_type`: string
- `agg`: string (SUM|COUNT|COUNT_DISTINCT|etc)
- `create_metric`: bool
- `agg_time_dimension`: string
- `is_partition`: bool
- `time_granularity`: string
- `entity`: string
- `yaml_path`: string
- `updated_at`: timestamp(ms)

**Key Features:**
- NOT subject-based (direct BaseEmbeddingStore)
- Unique on `id`
- Creates indices: kind, table_name, id
- Full-text search on: description, name, fq_name

---

### 3.4 Schema Metadata Store
**File:** `datus/storage/schema_metadata/store.py` (lines 26-71)

**Columns (Simple Base Schema):**
- `identifier`: string (unique: catalog.database.schema.table.type)
- `catalog_name`: string
- `database_name`: string
- `schema_name`: string
- `table_name`: string
- `table_type`: string (table|view|mv)
- `{vector_source_name}`: string (e.g., "metadata", "properties")
- `vector`: list[float32]

**Key Features:**
- Simple base, no subject-based inheritance
- NOT unique constraint explicitly defined
- Supports filtering by catalog/database/schema/table_type

---

### 3.5 Document Store
**File:** `datus/storage/document/store.py` (lines 42-72)

**Columns:**
- `chunk_id`: string (unique, deduplication)
- `chunk_text`: string (embedding source)
- `chunk_index`: int32
- `title`: string
- `titles`: list[string] (page-internal headings)
- `nav_path`: list[string] (site navigation)
- `group_name`: string
- `hierarchy`: string (full combined path)
- `version`: string
- `source_type`: string
- `source_url`: string
- `doc_path`: string
- `keywords`: list[string]
- `language`: string
- `created_at`: string
- `updated_at`: string
- `content_hash`: string
- `vector`: list[float32]

**Key Features:**
- Platform-isolated via separate DocumentStore per platform
- Unique on `chunk_id`
- Full-text search on: chunk_text, keywords
- Per-platform storage paths

---

### 3.6 External Knowledge Store
**File:** `datus/storage/ext_knowledge/store.py` (lines 17-41)

**Base Columns:**
- `name`, `subject_id`, `created_at` (from `base_schema_columns()`)

**Additional Columns:**
- `id`: string (unique)
- `search_text`: string (embedding source)
- `explanation`: string
- `vector`: list[float32]

**Key Features:**
- Subject-based
- Unique on `id`
- Creates indices: subject_index
- Full-text search on: search_text, explanation

---

### 3.7 Task Store (RDB-backed)
**File:** `datus/storage/task/store.py` (lines 20-37)

**Table Definition:**
- `id`: INTEGER PRIMARY KEY, autoincrement
- `task_id`: TEXT (not null, unique)
- `task_query`: TEXT (not null)
- `sql_query`: TEXT
- `sql_result`: TEXT
- `status`: TEXT (default: "running")
- `user_feedback`: TEXT
- `created_at`: TEXT (not null)
- `updated_at`: TEXT (not null)

**Key Features:**
- RDB-backed (NOT vector-based)
- Unique on `task_id`
- Index on `task_id`

---

### 3.8 Feedback Store (RDB-backed)
**File:** `datus/storage/feedback/store.py` (lines 20-32)

**Table Definition:**
- `id`: INTEGER PRIMARY KEY, autoincrement
- `task_id`: TEXT (not null, unique)
- `status`: TEXT (not null)
- `created_at`: TEXT (not null)

**Key Features:**
- RDB-backed
- Unique on `task_id`
- Index on `task_id`

---

## 4. Subject Tree Store (RDB-backed)
**File:** `datus/storage/subject_tree/store.py` (lines 28-43, 70-180)

### Base Table Definition
```python
_SUBJECT_NODES_TABLE = TableDefinition(
    table_name="subject_nodes",
    columns=[
        ColumnDef(name="node_id", col_type="INTEGER", pk=True, autoincrement=True),
        ColumnDef(name="parent_id", col_type="INTEGER"),
        ColumnDef(name="name", col_type="TEXT", nullable=False),
        ColumnDef(name="description", col_type="TEXT", default=""),
        ColumnDef(name="created_at", col_type="TEXT", nullable=False),
        ColumnDef(name="updated_at", col_type="TEXT", nullable=False),
    ],
    indices=[
        IndexDef(name="idx_subject_parent_id", columns=["parent_id"]),
        IndexDef(name="idx_subject_parent_name", columns=["parent_id", "name"], unique=True),
    ],
    constraints=["UNIQUE(parent_id, name)"],
)
```

### Dynamic Configuration (Lines 76-114)
Reads from storage registry defaults:
- `table_prefix`: Applied to table name (e.g., "tb_subject_nodes")
- `extra_fields`: Converted from PyArrow fields to ColumnDef, appended
  - `col_type` hardcoded to "TEXT", `default` set to ""
- `scope_indices`: Added as additional indices for tenant isolation
  - Pattern: `idx_subject_{column_name}`

### Multi-tenant Support (Lines 123-145)
- `set_request_context()`: Configures tenant isolation
  - `request_context`: All key-value pairs auto-filled on writes
  - `scope_fields`: Which fields become WHERE filters
- `_base_where`: Merged into every query
- `_write_defaults`: Auto-filled on every insert/update

---

## 5. Path and Namespace Building
**Location:** `datus/configuration/agent_config.py`

### Properties
```python
@property
def rag_storage_path(self) -> str:
    # Returns: {rag_base_path}/datus_db_{namespace}
    return rag_storage_path(self._current_namespace, self.rag_base_path)

def document_storage_path(self, platform: str) -> str:
    # Returns: {rag_base_path}/document/{platform}/
    return os.path.join(self.rag_base_path, "document", platform)

def document_storage_base_path(self) -> str:
    # Returns: {rag_base_path}/document/
    return os.path.join(self.rag_base_path, "document")

def sub_agent_storage_path(self, sub_agent_name: str):
    # Returns: {rag_base_path}/sub_agents/{sub_agent_name}
    return os.path.join(self.rag_base_path, "sub_agents", sub_agent_name)
```

### Base Path Initialization
- CLI mode: `{home}/.datus/data` (from `_init_dirs()`)
- SaaS mode: Set explicitly via `request_context` + `rag_base_path` kwarg
- SaaS mode skips `init_backends()` in central location to avoid global mutation

### Namespace Building Function
```python
def rag_storage_path(namespace: str, rag_base_path: str = "data") -> str:
    return os.path.join(rag_base_path, f"datus_db_{namespace}")
```

---

## 6. Backend Initialization
**Location:** `datus/storage/backend_holder.py` (lines 29-56)

### Initialization Flow
1. Called from `AgentConfig.__init__()` → `init_backends()`
2. Sets global singletons:
   - `_config`: StorageBackendConfig
   - `_data_dir`: Root data directory
   - `_namespace`: Current namespace
3. Resets backend instances for lazy re-initialization

### Per-Store Database Creation
```python
def create_rdb_for_store(store_db_name: str) -> RdbDatabase:
    backend = _get_rdb_backend()
    return backend.connect(_namespace, store_db_name)
    # store_db_name: "subject_tree", "task", "feedback"
```

### Vector Database Creation
```python
def create_vector_connection(namespace: str = "") -> VectorDatabase:
    backend = get_vector_backend()
    return backend.connect(namespace=namespace or _namespace)
```

---

## 7. Current Field Usage - NONE FOUND

### Important Finding:
**NO fields like `workspace_id`, `datasource_id`, `creator_id`, `updator_id`, `namespace` currently exist in any store's schema.**

These are:
1. **Documented in registry.py as SaaS example** (lines 66-71)
2. **Support infrastructure is built** (extra_fields, default_values, scope_indices)
3. **But NOT IMPLEMENTED in current schemas**

### Where They Would Be Used:
- Example in `registry.py` shows for SaaS deployments
- Would be added via `configure_storage_defaults(extra_fields=[...])`
- Would be auto-filled via `configure_storage_defaults(scope_fields=[...])`

---

## 8. Current Git Changes Summary

### Modified File: `datus/storage/subject_tree/store.py`
- Formatting change in `SubjectTreeStore.__init__()` (lines 94-97)
- Table name building split across multiple lines for readability
- No functional changes to schema or multi-tenant support

---

## 9. Pattern Summary: Subject-Based vs. Direct Stores

### Subject-Based Stores
Use `BaseSubjectEmbeddingStore` which adds:
- Automatic subject_path → subject_node_id conversion
- Base columns: name, subject_id, created_at
- Methods: batch_store(), batch_upsert(), search_with_subject_filter()
- delete_entry(), search_all() with subject filtering

**Stores:**
- ReferenceSqlStorage
- MetricStorage
- ExtKnowledgeStore

### Direct Embedding Stores
Use `BaseEmbeddingStore` directly:
- No subject hierarchy
- Custom schema definitions
- Direct batch operations

**Stores:**
- SemanticModelStorage
- BaseMetadataStorage (abstract)
- DocumentStore

---

## 10. Key Findings

### Already Implemented
1. Multi-tenant schema extension via `extra_fields`
2. Default value auto-fill via `default_values`
3. Scope filtering via `scope_filter` + `scope_indices`
4. Singleton + scoped view pattern for isolation
5. Per-request context injection via `request_context`
6. Two-level configuration (deployment + request)

### Not Yet Implemented
1. No actual multi-tenant fields in any schema
2. SubjectTreeStore has template for dynamic columns but needs activation
3. RDB stores (task, feedback) have no multi-tenant support yet
4. Document store has no multi-tenant support

### Gaps for SaaS Deployment
1. Configure storage defaults at startup
2. Pass request_context in each request
3. Add extra_fields to schemas
4. Update SubjectTreeStore initialization to create extra columns
5. Similar RDB schema extension for task/feedback stores

