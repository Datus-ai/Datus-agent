# Reference Template Intelligence

## Overview

Bootstrap-KB Reference Template is a knowledge base component that processes, analyzes, and indexes parameterized Jinja2 SQL templates. It transforms raw `.j2` template files into a searchable repository with semantic search, parameter metadata extraction, and server-side rendering capabilities.

## Core Value

### What Problem Does It Solve?

- **SQL Stability**: LLM-generated SQL can vary between runs, causing production inconsistencies
- **Parameterized Queries**: Repetitive queries that differ only by parameters (dates, regions, thresholds)
- **Template Discovery**: No efficient way to find existing templates by business intent
- **Controlled Output**: Need to constrain SQL generation to pre-approved query patterns
- **Parameter Ambiguity**: LLM doesn't know what values are valid for each parameter

### What Value Does It Provide?

- **Stable SQL Output**: Render pre-defined templates with parameters instead of generating SQL from scratch
- **Parameter Intelligence**: Automatically infers parameter types, resolves column references, and provides sample values from the database
- **Semantic Search**: Find templates using natural language descriptions
- **Server-Side Rendering**: Jinja2 rendering happens server-side with strict undefined checking

## Usage

### Basic Command

```bash
# Initialize Reference Template component
datus-agent bootstrap-kb \
    --datasource <your_datasource> \
    --components reference_template \
    --template_dir /path/to/template/directory \
    --kb_update_strategy overwrite
```

### Key Parameters

| Parameter | Required for reference_template | Description | Example |
|-----------|---------------------------------|-------------|---------|
| `--datasource` | Yes | Database datasource | `analytics_db` |
| `--components` | Yes | Components to initialize. Set this to `reference_template`; otherwise the reference template component is not initialized. | `reference_template` |
| `--template_dir` | Yes | Directory or single file containing J2 template files. If omitted, the reference template store is initialized empty and no template entries are loaded. | `/templates/queries` |
| `--kb_update_strategy` | No | Update strategy. Defaults to `check`; use `overwrite` or `incremental` when loading templates intentionally. | `overwrite`/`incremental` |
| `--validate-only` | No | Only validate, don't store | |
| `--pool_size` | No | Concurrent processing threads. CLI default is `4`; learning mode without `--subject_tree` forces effective concurrency to `1`. | `8` |
| `--subject_tree` | No | Predefined subject tree categories | `Analytics/User/Activity,Reporting/Sales/Monthly` |

### Subject Tree Categorization

Subject tree provides a hierarchical taxonomy for organizing templates by domain. This is the same mechanism used by Reference SQL.

**Predefined Mode** (with `--subject_tree`):

```bash
datus-agent bootstrap-kb \
    --datasource analytics_db \
    --components reference_template \
    --template_dir /path/to/templates \
    --kb_update_strategy overwrite \
    --subject_tree "Analytics/User/Activity,Reporting/Sales/Monthly"
```

**Learning Mode** (without `--subject_tree`):

The system reuses existing categories and creates new ones as needed.

## Template File Format

### Supported Extensions

- `.j2` — Standard Jinja2 template extension
- `.jinja2` — Alternative Jinja2 extension

### Single Template File

Each `.j2` file can contain a single SQL template with Jinja2 parameters:

```sql
SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` AS free_rate
FROM frpm
WHERE `Educational Option Type` = '{{school_type}}'
  AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL
ORDER BY free_rate {{sort_order}}
LIMIT {{limit}}
```

### Multi-Template File

Multiple templates in one file, separated by semicolons (`;`):

```sql
SELECT T2.Zip
FROM frpm AS T1
INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode
WHERE T1.`District Name` = '{{district_name}}'
  AND T1.`Charter School (Y/N)` = 1;
SELECT T1.Phone
FROM schools AS T1
INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds
WHERE T1.County = '{{county}}'
  AND T2.NumTstTakr < {{max_test_takers}}
```

### Jinja2 Syntax Support

- **Variables**: `{{ variable_name }}` — extracted as template parameters
- **Conditionals**: `{% if condition %}...{% endif %}`
- **Loops**: `{% for item in items %}...{% endfor %}`
- **Comments**: `{# comment #}`

Semicolons inside Jinja2 block structures (`{% if %}`, `{% for %}`, etc.) are not treated as template delimiters.

### Format Requirements

1. **Semicolon Delimiter**: Templates in a multi-template file must be separated by `;`. Put each template delimiter at the end of a statement line; multiple templates on the same physical line are not reliably split.
2. **Valid Jinja2**: Templates must pass Jinja2 syntax validation
3. **SQL Content**: Templates should produce valid SQL when rendered

## Parameter Type System

During bootstrap, each `{{ variable }}` placeholder is automatically analyzed to determine its type and context within the SQL. Table aliases (e.g., `T1`, `T2`) are resolved to real table names.

### Parameter Types

| Type | Detection | Enrichment |
|------|-----------|------------|
| `dimension` | Appears in `WHERE col = '{{param}}'` | `column_ref` with real table.column; `sample_values` with top 10 most common values from the database |
| `column` | Appears in `GROUP BY {{param}}` or `SELECT {{param}}` | `table_refs` listing involved tables; `sample_values` with available column names |
| `keyword` | Appears in `ORDER BY expr {{param}}` | `allowed_values` with valid keywords (e.g., `["ASC", "DESC"]`) |
| `number` | Appears in `LIMIT {{param}}` or comparison operators | — |

### Example

Given this template:

```sql
SELECT {{group_column}}, COUNT(*) AS school_count
FROM frpm
WHERE `Educational Option Type` = '{{school_type}}'
GROUP BY {{group_column}}
ORDER BY school_count {{sort_order}}
LIMIT {{limit}}
```

The bootstrap process produces:

```json
[
  {
    "name": "group_column",
    "type": "column",
    "table_refs": ["frpm"],
    "sample_values": ["CDSCode", "County Name", "District Name", "School Name", "..."],
    "description": "Column name to group results by"
  },
  {
    "name": "school_type",
    "type": "dimension",
    "column_ref": "frpm.`Educational Option Type`",
    "sample_values": ["Traditional", "Continuation School", "Charter School", "..."],
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

When a parameter's SQL context can be resolved, this gives the LLM concrete candidate values or allowed keywords to use when calling `execute_reference_template`. Parameters whose context cannot be inferred are still stored, but may not include sample values.

## Tools

After bootstrapping, template tools are available only when the reference template store contains entries. The complete tool set is:

### `search_reference_template`

Search templates by natural language query. Returns matching templates with name, raw template body, parameter metadata, summary, and tags.

### `get_reference_template`

Retrieve a specific template by `subject_path` + `name`. Returns full template content, enriched parameters with `sample_values`, and summary.

### `render_reference_template`

Render a template with provided parameter values using Jinja2. Returns the final SQL string without executing it. Uses `StrictUndefined` mode — missing parameters produce actionable error messages listing expected vs. provided parameters.

### `execute_reference_template`

Render a template and immediately execute the resulting SQL (read-only). Combines `render_reference_template` + `read_query` in a single step. Returns both the rendered SQL and the query result rows.

Note: `execute_reference_template` is exposed only when the tool instance has a database function tool. Chat and GenSQL nodes create or reuse a DB tool when wiring reference template tools, so they normally expose `execute_reference_template` when templates exist. Direct `ReferenceTemplateTools` instances without `db_func_tool` expose only search/get/render.

## Template-Only Mode

For use cases where the agent should only execute pre-approved templates (no ad-hoc SQL), a dedicated system prompt `ref_tpl` is available:

```yaml
agentic_nodes:
  template_executor:
    model: deepseek-v3
    system_prompt: ref_tpl
    prompt_version: '1.0'
    max_turns: 10
    tools: context_search_tools.list_subject_tree, reference_template_tools.search_reference_template, reference_template_tools.get_reference_template, reference_template_tools.execute_reference_template
```

In this mode, the agent:

- **MUST** search templates first for every question
- **MUST** use `execute_reference_template` when a match is found — never writes SQL manually
- Reports "no matching template found" if no template matches, and stops

## Data Flow

```text
Template Files (.j2)  →  File Processor  →  LLM Analysis  →  Parameter Analysis  →  Storage  →  Tools
       |                       |                  |                    |                |           |
   Parse blocks          Validate J2         Generate summary,   Infer types,       Vector DB   search/
   Extract params        Extract params      search_text,        merge descriptions + Indices  get/render/
   Split by ;            Filter invalid      subject_tree        query sample values            execute
```

### Processing Pipeline

1. **File Discovery**: Find `.j2`/`.jinja2` files in the template directory
2. **Block Splitting**: Split multi-template files by semicolons (respecting Jinja2 blocks)
3. **Validation**: Validate Jinja2 syntax for each template block
4. **Parameter Extraction**: Extract undeclared variables via `jinja2.meta.find_undeclared_variables()`
5. **LLM Analysis**: Generate business summary, search text, subject tree, tags, and parameter descriptions using `SqlSummaryAgenticNode` in workflow mode with `storage_type="reference_template"`
6. **Parameter Analysis**: Infer parameter types, resolve table aliases to real table names, and query sample values from the database
7. **Merge**: Combine statically-analyzed parameter types with LLM-generated descriptions and keyword allowed values
8. **Storage**: Store enriched template data in vector store
9. **Indexing**: Create search indices for efficient retrieval

## Summary

Reference Template transforms parameterized SQL templates into an intelligent, searchable knowledge base. It bridges the gap between flexible LLM-driven SQL generation and the stability requirements of production environments.

**Key Features:**

- **Parameterized SQL**: Define query patterns with Jinja2 variables
- **Parameter Intelligence**: Automatically infers types (`dimension`, `column`, `keyword`, `number`), resolves column references, and provides sample values
- **Semantic Search**: Find templates by business intent
- **One-Step Execution**: Search, render, and execute templates in a single tool call
- **Template-Only Mode**: Dedicated system prompt to restrict agents to pre-approved templates only
- **Subject Tree Organization**: Hierarchical classification for template discoverability
