# Tool Commands `!`

## 1. Overview

Tool commands (prefixed with `!`) provide specialized AI-powered capabilities and utility operations within the Datus-CLI environment. These commands enable schema discovery, metrics search, SQL reference search, and other intelligent data operations without leaving the interactive session.

## 2. Command Categories

### 2.1 Schema Discovery Commands

#### `!sl` / `!schema_linking`
Perform intelligent schema linking to discover relevant tables and columns for your query.

```bash
!sl user purchase information
!schema_linking sales data by region
```

Features:

- Semantic search for relevant database tables
- Table definition (DDL) display
- Sample data preview
- Configurable matching methods: fast, medium, slow, from_llm
- Adjustable top_n results

Interactive prompts guide you through:

- Catalog/database/schema selection
- Number of tables to match
- Matching method preference

### 2.2 Search & Discovery Commands

#### `!sm` / `!search_metrics`
Use natural language to search for corresponding metrics in your data catalog.

```bash
!sm monthly active users
!search_metrics revenue growth rate
```

Allows filtering by:

- Domain
- Layer1 (business layer)
- Layer2 (sub-layer)
- Top N results

#### `!sq` / `!search_sql`
Search historical SQL queries using natural language descriptions.

```bash
!sq queries about user retention
!search_sql monthly sales reports
```

Returns:

- SQL query text with syntax highlighting
- Query summary and comments
- Tags and categorization
- Domain/layer metadata
- File path and relevance distance

### 2.3 Utility Commands

#### `!save`
Save the last query result to a file.

```bash
!save
```

Interactive options:

- File type: json, csv, sql, or all
- Output directory (defaults to ~/.datus/output)
- Custom filename

#### `!bash <command>`
Execute safe bash commands (security restricted).

```bash
!bash pwd
!bash ls -la
!bash cat config.yaml
```

**Security**: Only whitelisted commands are allowed:

- `pwd` - Print working directory
- `ls` - List files
- `cat` - Display file contents
- `head` - Show file beginning
- `tail` - Show file end
- `echo` - Display text

Commands not in the whitelist will be rejected with a security warning.

## 3. Best Practices

1. **Start with Schema Linking** - Use `!sl` to discover relevant tables before writing queries
2. **Leverage Search** - Use `!sm` and `!sq` to find existing metrics and queries before creating new ones
3. **Save Results** - Use `!save` to preserve important query results
4. **Security First** - Be aware of bash command restrictions when using `!bash`

## 4. Security Considerations

- Tool commands run with the same privileges as the Datus-CLI process
- Bash commands are restricted to a whitelist of safe operations
- `!bash` commands timeout after 10 seconds to prevent hanging
- All operations are logged for audit purposes
- API credentials and database connections are handled securely