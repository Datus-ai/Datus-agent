# Metadata Full-text Search

Metadata full-text search (FTS) is an opt-in alternative to vector retrieval for finding relevant tables. It searches the text already present in table metadata and does not generate embeddings for that metadata.

FTS is useful when questions contain recognizable business terms, table or column names, SQL fragments, or other exact text. Keep the default vector mode when users commonly describe a concept with words that do not appear in the metadata.

!!! note "Scope"
    The setting only changes **metadata retrieval** used for table discovery and schema linking. Metrics, reference SQL, reference templates, platform documents, and other knowledge-base components keep their own retrieval behavior. FTS also searches only sample rows captured during bootstrap; it does not query every source-table row.

## FTS and vector modes

| | `fts` | `vector` |
|---|---|---|
| Best for | Keywords, identifiers, SQL text, and exact business terms | Semantic similarity and paraphrased questions |
| Metadata embeddings | Not generated | Required |
| Default | No | Yes |
| Fallback | Does not fall back to vector search | Not applicable |

FTS indexes the following metadata text:

- Table name, qualified name, identifier, and table type
- Table or view definition (DDL)
- Sample rows collected by `bootstrap-kb`
- Attached table semantic profiles, when available, including descriptions, columns, relationships, and AI context

Search results remain scoped to the active datasource and any catalog, database, schema, table type, or subagent restrictions applied by Datus Agent.

## Prerequisites

- Configure the datasource and give its database user permission to read metadata and sample rows.
- Use an FTS-capable vector storage backend. The built-in LanceDB backend works without additional configuration.
- If `storage.vector` selects an external adapter, make sure that adapter version supports the Datus FTS contract. Unsupported backends fail explicitly instead of silently using vector search.

## Enable FTS

Add the following setting to the `agent.yml` used by both bootstrap and normal Datus Agent sessions:

```yaml
kb:
  search:
    mode: fts
```

The default is `vector` when this section is omitted.

`bootstrap-kb` also accepts `--kb_search_mode fts`, but that option overrides the mode only for the current command and is not saved. Persist the YAML setting before using the rebuilt metadata in chat, API, background sync, or agent runs.

## Build the initial index

The first FTS build, or a switch from vector mode, requires a full metadata rebuild:

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components metadata \
  --kb_update_strategy overwrite
```

`overwrite` replaces metadata only for the selected datasource and creates the FTS index with the current index format. It does not rebuild unrelated knowledge-base components.

To apply a one-time command-line override while testing, add `--kb_search_mode fts` to the command. Remember that subsequent sessions still read the mode from `agent.yml`.

## Verify the index

Run a read-only check with the same configuration:

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components metadata \
  --kb_update_strategy check
```

A successful result reports `search_mode=fts` together with `schema_size` and `value_size`. The check also verifies that the FTS index exists and matches the current index specification.

After that, use Datus normally. For example, start the CLI, select the datasource, and ask a table-discovery question:

```text
datus
/datasource <your_datasource>
Which table contains customer order status?
```

Metadata tools and schema linking automatically use the configured FTS path.

## Keep metadata current

After a successful full build, use an incremental update for routine schema changes:

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components metadata \
  --kb_update_strategy incremental
```

Incremental mode upserts new or changed metadata and updates the existing index. It requires an already-ready FTS index and does not create or repair a missing index.

Run `overwrite` again when:

- Enabling FTS for the first time or switching storage backends
- Datus reports an index status of `missing`, `legacy`, or `version_mismatch`
- An earlier build was interrupted or the index is incomplete
- An upgrade explicitly requires rebuilding the FTS index

## Switch back to vector search

Change the configuration and rebuild metadata so the vector store and embeddings are current:

```yaml
kb:
  search:
    mode: vector
```

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components metadata \
  --kb_update_strategy overwrite
```

## Troubleshooting

### The backend does not support FTS

An error mentioning an `FTS-capable vector backend` means the backend selected by `storage.vector` does not expose the required FTS operations. Use the built-in LanceDB backend, install an adapter version with FTS support, or keep `kb.search.mode: vector`.

### The index is missing or incompatible

Errors containing `missing`, `legacy`, or `version_mismatch` are intentionally not hidden by a vector fallback. Rebuild metadata with `--kb_update_strategy overwrite`.

### Incremental update fails on a new installation

Incremental mode only maintains an existing healthy index. Run one `overwrite` build first, then use `incremental` for later changes.

### Bootstrap used FTS but normal sessions still use vector search

If the build command used `--kb_search_mode fts` without updating `agent.yml`, only that command used FTS. Add `kb.search.mode: fts` to the persistent configuration and run the read-only check again.

### Results miss conceptually related tables

FTS ranks text matches; it does not infer semantic similarity. Add clearer business descriptions or semantic profiles to the metadata, search with terms that appear in the schema, or switch back to vector mode for paraphrase-heavy workloads.

!!! tip "Embeddings for other components"
    FTS avoids embeddings for metadata only. If you also build metrics, reference SQL, documents, or other vector-backed components, keep their embedding configuration and credentials available.
