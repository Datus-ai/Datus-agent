# Platform Documentation

## Introduction

`platform-doc` ingests official platform documentation (for example Snowflake, StarRocks, Polaris) into a dedicated vector store, so the agent can verify platform-specific SQL syntax and features before generating queries.

The end-to-end pipeline is:

```text
Fetch → Parse → Clean → Chunk → Embed → Store
```

## Storage Scope

- **Namespace-independent**: documents are stored per platform and shared across namespaces.
- **Default location**: `~/.datus/data/document/<platform>/`.
- **Selection key**: the `platform` argument decides which document store is queried.

## Build the Documentation Store

Platform docs are initialized with the dedicated `platform-doc` command (separate from `bootstrap-kb`).

### Command Syntax

```bash
datus-agent platform-doc \
  --platform <platform_name> \
  --source <source> \
  --source-type <github|website|local> \
  --update-strategy <check|overwrite> \
  [options]
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--platform` | `default` | Platform name used as the storage key (recommended). |
| `--source` | - | GitHub repo `owner/repo`, website URL, or local path. Required unless configured in `agent.yml`. |
| `--source-type` | `local` | Source type: `github`, `website`, or `local`. |
| `--version` | auto | Document version label. If omitted, Datus tries to detect the latest version. |
| `--github-ref` | default branch | Git ref (branch or tag) for GitHub sources. |
| `--paths` | `docs README.md` | Paths to fetch for GitHub sources only. |
| `--include-patterns` | - | Include patterns (glob for local, regex for website). |
| `--exclude-patterns` | - | Exclude patterns (glob for local, regex for website). |
| `--chunk-size` | `1024` | Target chunk size in characters (see details below). |
| `--max-depth` | `2` | Maximum crawl depth for website sources. |
| `--pool-size` | `4` | Worker threads for processing. |
| `--update-strategy` | `check` | `check` (status-only) or `overwrite` (rebuild). |

#### `--chunk-size` Details

`--chunk-size` is a **soft limit**: to preserve semantic integrity, individual paragraphs and code blocks may exceed the target size (up to a hard maximum of **2048** characters). Chunks smaller than **256** characters are automatically merged with their neighbors.

- **Default**: 1024 characters
- **Recommended range**: 512–2048
- **Tuning**: larger values produce fewer, coarser chunks; smaller values produce more, finer-grained chunks

#### Supported File Types

- **local / github**: `.md`, `.markdown`, `.html`, `.htm`, `.rst`, `.txt`

### Examples

**1) GitHub (default branch)**

!!! tip
    Set `GITHUB_TOKEN` environment variable or configure `github_token` in `agent.yml` to avoid GitHub API rate limiting.

```bash
datus-agent platform-doc \
  --platform starrocks \
  --source StarRocks/starrocks \
  --source-type github \
  --update-strategy overwrite \
  --paths docs/en
```

**2) GitHub (specific tag or branch)**

```bash
datus-agent platform-doc \
  --platform starrocks \
  --source StarRocks/starrocks \
  --source-type github \
  --version 4.0.5 \
  --github-ref 4.0.5 \
  --update-strategy overwrite \
  --paths docs/en
```

**3) GitHub (versioned docs branch)**

```bash
datus-agent platform-doc \
  --platform polaris \
  --source apache/polaris \
  --source-type github \
  --github-ref versioned-docs \
  --update-strategy overwrite \
  --paths 1.2.0 1.3.0
```

**4) Website crawl**

```bash
datus-agent platform-doc \
  --platform snowflake \
  --source https://docs.snowflake.com/en/sql-reference \
  --source-type website \
  --version latest \
  --update-strategy overwrite \
  --max-depth 2
```

**5) Local directory**

```bash
datus-agent platform-doc \
  --platform duckdb \
  --source /path/to/duckdb-docs \
  --source-type local \
  --version v1.0.0 \
  --update-strategy overwrite
```

## Configure in `agent.yml` (Optional)

You can store per-platform fetch configs in `agent.document` and then run the command with only `--platform`. CLI arguments override YAML values.

```yaml
agent:
  document:
    tavily_api_key: ${TAVILY_API_KEY}
    starrocks:
      type: github
      source: StarRocks/starrocks
      paths: ["docs/sql-reference"]
      chunk_size: 1024
      github_token: ${GITHUB_TOKEN}
    snowflake:
      type: website
      source: https://docs.snowflake.com/en/
      max_depth: 3
      include_patterns: ["/en/user-guide", "/en/sql-reference"]
    local_docs:
      type: local
      source: /path/to/docs
      include_patterns: ["*.md"]
      exclude_patterns: ["CHANGELOG.md"]
```

Run with:

```bash
datus-agent platform-doc --platform starrocks --github-ref 4.0.5
```

## Using the Tools in Datus

Once documents are ingested and platform doc tools are enabled, the agent gains four tools:

| Tool | Purpose | Key Args |
|---|---|---|
| `list_document_nav` | Browse the navigation tree and discover document titles. | `platform`, `version` |
| `get_document` | Retrieve full content for one document by hierarchy path. | `platform`, `titles`, `version` |
| `search_document` | Semantic search by keywords. | `platform`, `keywords`, `version`, `top_n` |
| `web_search_document` | Optional web fallback (Tavily). | `keywords`, `include_domains`, `max_results` |

To expose these tools in custom nodes or subagents, include `platform_doc_tools` (or specific `platform_doc_search_tools.*`) in the tool configuration.

**Recommended call order**: `list_document_nav` → `get_document` → `search_document` → `web_search_document` (fallback).

**`get_document` expects one document at a time**. Pass a single hierarchy path like:

```text
titles=["DDL", "CREATE TABLE"]
```

To retrieve multiple documents, call it multiple times.

### CLI shortcut (Datus-CLI)

Inside `datus-cli`, use:

```bash
!sd
!search_document
```

This runs `search_document` interactively and shows matched chunks.

## Notes and Troubleshooting

- **GitHub API limits**: set `GITHUB_TOKEN` (or `github_token` in config) to avoid rate limiting.
- **Website crawling**: `--paths` is ignored for website sources. Use `--include-patterns` / `--exclude-patterns` instead.
- **No results**: verify the `platform` name matches the store you created and pass the right `version` if multiple versions exist.
- **Web fallback**: `web_search_document` requires `TAVILY_API_KEY`.
