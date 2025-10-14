# Sub-Agent Scoped KB Bootstrap Design

## 1. Background

- `.subagent` commands (`datus/cli/sub_agent_commands.py:33`) provide CRUD over `SubAgentConfig` via the interactive wizard (`datus/cli/sub_agent_wizard.py`).
- `ScopedContext` (tables, metrics, sqls) is persisted as comma-delimited strings; prompts (`datus/agent/node/gen_sql_agentic_node.py:670`) merely condition LLM output and do not constrain retrieval.
- Knowledge base bootstrap is centralized in `Agent.bootstrap_kb` (`datus/agent/agent.py:560`), writing all components into the namespace’s LanceDB directory `rag_storage_path()` (`datus/configuration/agent_config.py:408`).
- CLI completers (`datus/cli/autocomplete.py:535`, `724`, `755`) read from the global LanceDB stores, so scoped sub-agents still see every table/metric/sql.

## 2. Goals

1. Allow sub-agents to materialize a private LanceDB containing only metadata/metrics/sql_history defined in their `ScopedContext`.
2. Trigger scoped bootstrap from the existing `.subagent` workflow (after creation or on demand).
3. Persist the scoped storage location in `SubAgentConfig` so runtime components can switch to it.
4. Reuse current bootstrap logic where possible; avoid duplicating ingestion pipelines.
5. When the user executes `datus/main.py bootstrap` to overwrite/incrementally update the corresponding data, the LanceDB needs to be adjusted according to the subagent configuration.

## 3. Non-Goals

- Changing default namespace-level bootstrap behavior.
- Extending scoped bootstrap to documents or external knowledge.
- Drastically altering the wizard UI beyond minimal prompts.

## 4. Proposed Architecture

### 4.1 Config Extensions

```python
class ScopedContext(BaseModel):
    tables: Optional[str]
    metrics: Optional[str]
    sqls: Optional[str]

    def as_lists(self) -> ScopedContextLists:
        # split by comma/newline → trimmed lists
```

```python
class SubAgentConfig(BaseModel):
    ...
    scoped_kb_path: Optional[str] = None
```

- `ScopedContext.as_lists()` normalizes strings into deterministic lists (split on comma/newline, remove empties).
- Scoped KB components are fixed to three segments: `metadata` (tables & table_value), `metrics`, and `sql_history`.
- `SubAgentManager.save_agent` persists `scoped_kb_path` when provided; otherwise existing behavior remains.

### 4.2 Scoped Bootstrap Service

New module `datus/utils/sub_agent_bootstrap.py`:

```python
class SubAgentBootstrapper:
    def __init__(self, *, sub_agent: SubAgentConfig, agent_config: AgentConfig,
                 db_manager: DBManager, console: Console | None = None):
        ...

    def compute_storage_path(self) -> str:
        base = agent_config.rag_storage_path()
        return os.path.join(base, "sub_agents", slugify(sub_agent.system_prompt))

    def run(self, components: Sequence[str], strategy: Literal["overwrite", "plan"]) -> BootstrapResult:
        # orchestrate metadata/metrics/sql_history pipelines
```

Responsibilities:

- Ensure target directory exists; wipe component folders on `overwrite`.
- Convert `ScopedContext` to filter sets (tables, metrics, sqls).
- Delegate component-specific bootstraps; collect result statuses/messages.
- Update `SubAgentConfig` with `scoped_kb_path`.

### 4.3 Component Pipelines

#### Metadata (Tables & Table Value)

1. Source: `SchemaWithValueRAG` on global path (`schema_metadata.lance`, `schema_value.lance`).
2. Filter:
   - Parse table identifiers into `(catalog, database, schema, table)`.
   - When qualifiers are missing, use current namespace/default database.
   - Support wildcard suffix `*` for schema/table prefixes.
3. Target: new `SchemaWithValueRAG` pointing to scoped path.
   - Collect schema/value rows with `_search_all` and `build_where` conditions.
   - Use `store_batch` to insert, `after_init()` to create indices.

#### Metrics

1. Source: `SemanticMetricsRAG` at namespace path.
2. Filter: interpret entries like `domain.layer1.layer2.metric`; allow wildcard segments.
3. Target: `SemanticMetricsRAG` at scoped path.
   - Retrieve matching semantic models and metrics.
   - `store_batch` with filtered data, then `after_init()`.

#### SQL History

1. Source: `SqlHistoryRAG` (`sql_history.lance`).
2. Filter: hierarchical key `domain.layer1.layer2.name`; partial keys fallback to prefix match.
3. Target: new `SqlHistoryRAG` at scoped directory.
   - Use existing upsert to persist filtered items.
   - Rebuild taxonomy indices if available.
   - Record skipped entries (not found) as warnings.

### 4.4 CLI Integration

Two complementary approaches:

1. **Automatic Bootstrap Prompt**
   - After wizard completion (when scoped context present), ask whether to bootstrap immediately.
   - If user agrees, invoke `SubAgentBootstrapper.run` automatically and surface results before returning to REPL.

2. **Manual Command**
   - Extend `.subagent` help and parser to include:

     ```
     .subagent bootstrap <name> [--components metadata,metrics,sql_history] [--plan]
     ```

   - Workflow:
     1. Resolve `SubAgentConfig` via `SubAgentManager.get_agent`.
     2. Instantiate `SubAgentBootstrapper` with active `DatusCLI` context (`agent_config`, `db_manager`).
     3. Run bootstrap; print per-component summary and target path.
     4. On success, update configuration file (persist path).

- `.subagent list` enhancement: add “Scoped KB” column (path or “—”) so operators see which agents already have scoped storage.

### 4.5 Runtime Consumption

- Retrieval utilities (`ContextSearchTools`, metric/sql search handlers, completers) should prefer the scoped LanceDB when `SubAgentConfig.scoped_kb_path` exists:
  - Inject scoped path into tool initialization (pass via constructor or context override).
  - Fallback to global path if scoped bootstrap missing or stale.
- Prompts still receive `ScopedContext` entries for transparency.

## 5. Error Handling & Observability

- Each component returns `{component, status, message, details}`; aggregate into CLI report.
- On partial failure: mark component status `error`, keep others `success`.
- Log skipped identifiers (e.g., table not found) with actionable hints.
- Optional `--plan` flag: simulate filters, list candidate records, skip write.

## 6. Testing

- **Unit Tests**
  - `ScopedContext.as_lists()` conversions covering commas, newlines, whitespace, duplicates.
  - Metadata filter building for single DB vs cataloged DBs; wildcard coverage.
  - Metrics/sql filters verifying hierarchical parsing.

- **Integration Tests**
  - Construct multiple `ScopedContext` variants (narrow table list, wildcard metrics, selective SQL histories).
  - Run bootstrapper to materialize scoped KBs; for each, execute representative queries via search utilities and confirm recall contains only the scoped records (no leakage from global store).
  - CLI command test with mocked `DatusCLI` invoking `.subagent bootstrap`.

- **Regression**
  - Ensure global bootstrap unaffected.
  - Confirm fallback behavior when `scoped_kb_path` absent (Use default storage_path).

## 7. Open Questions

1. Should scoped bootstrap copy from global LanceDB (as proposed) or re-ingest from source DB/LLM? Copying is faster but inherits existing global state.
2. Versioning: do we need timestamps or hashes to detect stale scoped stores?
3. Concurrency: enforce locks if multiple sub-agent bootstraps run in parallel?

## 8. Implementation Plan

1. Schema updates (`ScopedContext`, `SubAgentConfig`) + normalization helpers.
2. Implement `SubAgentBootstrapper` with metadata pipeline; add metrics/sql next.
3. Add CLI command wiring and configuration persistence updates (automatic + manual triggers).
4. Teach retrieval utilities/completers to respect scoped KB path.
5. When the user executes `datus/main.py bootstrap` to overwrite/incrementally update the corresponding data, the LanceDB needs to be adjusted according to the subagent configuration.
6. Add tests and docs updates (CLI usage example, troubleshooting).

