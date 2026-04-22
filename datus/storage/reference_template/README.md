# Reference Template Storage Module

This module handles the storage, processing, and analysis of parameterized Jinja2 SQL template files. It provides functionality to extract templates from `.j2` files, analyze them with LLM, and store them in a searchable knowledge base with parameter metadata.

## Data Flow

```text
Template Files → File Processor → Parameter Analysis → LLM Analysis → Storage
      ↓                ↓                ↓                   ↓            ↓
  Split blocks     Validation      Type inference        Metadata     Vector DB
  Extract params   + Cleaning     + column resolution   Extraction    + Search
                                  + sample values
```

### Processing Pipeline

1. **File Processing**: Extract template blocks from `.j2`/`.jinja2` files (split by `;`)
2. **Parameter Extraction**: Use `jinja2.meta.find_undeclared_variables()` to discover parameters
3. **Validation**: Validate Jinja2 syntax for each template block
4. **Parameter Analysis**: Static SQL AST analysis to determine parameter types, resolve column references, and query sample values from the database
5. **LLM Analysis**: Extract business metadata (name, summary, search_text, tags, subject_tree) using SqlSummaryAgenticNode
6. **Merge**: Combine statically-analyzed parameter types with LLM-generated descriptions
7. **Storage**: Store enriched data in vector store for semantic search
8. **Indexing**: Create search indices for efficient retrieval

## Parameter Type System

During bootstrap, each `{{ variable }}` placeholder is analyzed using SQL AST parsing to determine its type and resolve column references. Table aliases (e.g., `T1 → frpm`) are automatically resolved to real table names.

### Parameter Types

| Type | Detection Rule | Enrichment |
|------|---------------|------------|
| `dimension` | `WHERE col = '{{param}}'` (quoted string in equality) | `column_ref`: resolved `table.column`; `sample_values`: top 10 most common values from DB |
| `column` | `GROUP BY {{param}}`, `SELECT {{param}}` (column name position) | `table_refs`: list of tables from FROM/JOIN; `sample_values`: column names from describe_table |
| `keyword` | `ORDER BY expr {{param}}` (sort direction position) | `allowed_values`: `["ASC", "DESC"]` |
| `number` | `LIMIT {{param}}`, `col > {{param}}` (numeric context) | — |

### Example: Parameter Analysis

Given this template:
```sql
SELECT {{group_column}}, COUNT(*) AS school_count,
  AVG(`Free Meal Count (Ages 5-17)` * 1.0 / NULLIF(`Enrollment (Ages 5-17)`, 0)) AS avg_free_rate
FROM frpm
WHERE `Educational Option Type` = '{{school_type}}'
GROUP BY {{group_column}}
ORDER BY school_count {{sort_order}}
LIMIT {{limit}}
```

The analyzer produces:
```json
[
  {
    "name": "group_column",
    "type": "column",
    "table_refs": ["frpm"],
    "sample_values": ["CDSCode", "County Name", "District Name", "School Name", ...],
    "description": "Column name to group results by"
  },
  {
    "name": "school_type",
    "type": "dimension",
    "column_ref": "frpm.`Educational Option Type`",
    "sample_values": ["Traditional", "Continuation School", "Alternative School of Choice", ...],
    "description": "Type of educational option to filter by"
  },
  {
    "name": "sort_order",
    "type": "keyword",
    "allowed_values": ["ASC", "DESC"],
    "description": "Sort direction for results"
  },
  {
    "name": "limit",
    "type": "number",
    "description": "Maximum number of rows to return"
  }
]
```

### Alias Resolution

Templates with table aliases are handled correctly:
```sql
SELECT T2.Phone
FROM satscores AS T1
INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode
WHERE T1.County = '{{county}}'
```

The alias `T1` is resolved to `satscores`, producing `column_ref: "satscores.County"` (not `T1.County`).

## Agent Tools

Three tools are available for LLM agents to interact with reference templates:

| Tool | Purpose |
|------|---------|
| `search_reference_template` | Semantic search by natural language intent. Returns name, parameters, summary, tags (no template body). |
| `get_reference_template` | Exact lookup by subject_path + name. Returns full template, parameters with sample_values, summary. |
| `execute_reference_template` | Render template with parameters AND execute the SQL, returning query results in one step. |

### Dedicated System Prompt: `ref_tpl`

A specialized system prompt (`ref_tpl_system_1.0.j2`) restricts the agent to template-only execution:
- MUST search templates first
- If matched, use `execute_reference_template` — never write SQL manually
- If no match, report "no matching template found" and stop

Configure in `agent.yml`:
```yaml
agentic_nodes:
  template_executor:
    model: deepseek-v3
    system_prompt: ref_tpl
    prompt_version: '1.0'
    max_turns: 10
    tools: context_search_tools.list_subject_tree, reference_template_tools.search_reference_template, reference_template_tools.get_reference_template, reference_template_tools.execute_reference_template
```

## Configuration

### Build Modes

- **`overwrite`**: Replace all existing data
- **`incremental`**: Only process new items (based on template content hash)
- **`validate-only`**: Only validate template files without LLM analysis or storage

### Performance Tuning

- **`pool_size`**: Number of concurrent async tasks for LLM analysis (default: 1)
- **Parallel processing**: Items are processed asynchronously with semaphore-controlled concurrency
- Requires `--subject_tree` for parallel mode; without it, falls back to serial (pool_size=1)

## Data Schema

### Input Format (from J2 files)
```python
{
    "template": "SELECT * FROM t WHERE dt > '{{start_date}}'",  # Raw Jinja2 template
    "parameters": '[{"name": "start_date"}]',                    # Auto-extracted JSON
    "comment": "",                                                # Optional comment
    "filepath": "/path/to/source.j2"                             # Source file path
}
```

**Notes:**
- Templates are validated for Jinja2 syntax correctness
- Parameters are automatically extracted from `{{ variable }}` expressions
- A single file can contain multiple templates separated by `;`

### Output Format (after bootstrap)
```python
{
    "id": "md5_hash_of_template_content",
    "name": "daily_free_rate_query",                                    # Generated by LLM
    "template": "SELECT * FROM t WHERE dt > '{{start_date}}'",         # Raw Jinja2 content
    "parameters": '[{"name": "start_date", "type": "dimension", ...}]', # Enriched by sqlglot + LLM
    "comment": "",
    "filepath": "/path/to/source.j2",
    "summary": "Query free meal rates filtered by date range",          # Generated by LLM
    "search_text": "daily_free_rate: free meal rate query by date...",   # Used for vector embedding
    "subject_path": ["analytics", "education", "free_rate"],            # Generated by LLM
    "tags": "free_rate,education,daily",                                # Generated by LLM
    "vector": [0.1, 0.2, ...]                                          # Embedding from search_text
}
```

**Required fields for storage:**
- `name`, `template`, `summary`, `search_text`, `subject_path` (all generated by SqlSummaryAgenticNode)
- Items missing any required fields will be skipped with a warning

## Usage

```bash
# Basic usage - initialize reference templates with overwrite mode
python -m datus.main bootstrap-kb \
  --datasource your_datasource \
  --components reference_template \
  --template_dir /path/to/template/directory \
  --kb_update_strategy overwrite

# Incremental update - only process new template files
python -m datus.main bootstrap-kb \
  --datasource your_datasource \
  --components reference_template \
  --template_dir /path/to/template/directory \
  --kb_update_strategy incremental

# With predefined subject tree categories and parallel processing
python -m datus.main bootstrap-kb \
  --datasource your_datasource \
  --components reference_template \
  --template_dir /path/to/template/directory \
  --kb_update_strategy overwrite \
  --subject_tree "Analytics/User/Activity,Reporting/Sales/Monthly" \
  --pool_size 4
```
