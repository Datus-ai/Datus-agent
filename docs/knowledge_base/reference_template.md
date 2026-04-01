# Reference Template Intelligence

## Overview

Bootstrap-KB Reference Template is a knowledge base component that processes, analyzes, and indexes parameterized Jinja2 SQL templates. It transforms raw `.j2` template files into a searchable repository with semantic search, parameter metadata extraction, and server-side rendering capabilities.

## Core Value

### What Problem Does It Solve?

- **SQL Stability**: LLM-generated SQL can vary between runs, causing production inconsistencies
- **Parameterized Queries**: Repetitive queries that differ only by parameters (dates, regions, thresholds)
- **Template Discovery**: No efficient way to find existing templates by business intent
- **Controlled Output**: Need to constrain SQL generation to pre-approved query patterns

### What Value Does It Provide?

- **Stable SQL Output**: Render pre-defined templates with parameters instead of generating SQL from scratch
- **Parameter Awareness**: Automatically extracts and exposes template parameters for LLM-driven filling
- **Semantic Search**: Find templates using natural language descriptions
- **Server-Side Rendering**: Jinja2 rendering happens server-side with strict undefined checking

## Usage

### Basic Command

```bash
# Initialize Reference Template component
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components reference_template \
    --template_dir /path/to/template/directory \
    --kb_update_strategy overwrite
```

### Key Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--namespace` | Yes | Database namespace | `analytics_db` |
| `--components` | Yes | Components to initialize | `reference_template` |
| `--template_dir` | Yes | Directory containing J2 template files | `/templates/queries` |
| `--kb_update_strategy` | Yes | Update strategy | `overwrite`/`incremental` |
| `--validate-only` | No | Only validate, don't store | |
| `--pool_size` | No | Concurrent processing threads (default: 4) | `8` |
| `--subject_tree` | No | Predefined subject tree categories | `Analytics/User/Activity,Reporting/Sales/Monthly` |

### Subject Tree Categorization

Subject tree provides a hierarchical taxonomy for organizing templates by domain. This is the same mechanism used by Reference SQL.

**Predefined Mode** (with `--subject_tree`):

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
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

1. **Semicolon Delimiter**: Each template in a multi-template file must end with `;`
2. **Valid Jinja2**: Templates must pass Jinja2 syntax validation
3. **SQL Content**: Templates should produce valid SQL when rendered

## Tools

After bootstrapping, three tools are available to agents:

### `search_reference_template`

Search templates by natural language query. Returns matching templates with parameter metadata.

### `get_reference_template`

Retrieve a specific template by `subject_path` + `name`. Returns full template content and parameter list.

### `render_reference_template`

Render a template with provided parameter values using Jinja2. Returns the final SQL string. Uses `StrictUndefined` mode — missing parameters produce actionable error messages listing expected vs. provided parameters.

## Data Flow

```
Template Files (.j2)  -->  File Processor  -->  LLM Analysis  -->  Storage  -->  Tools
       |                        |                    |               |            |
   Parse blocks           Validate J2          Generate          Vector DB    search/
   Extract params         Extract params       summary &         + Indices   get/render
   Split by ;             Filter invalid       search_text
```

### Processing Pipeline

1. **File Discovery**: Find `.j2`/`.jinja2` files in the template directory
2. **Block Splitting**: Split multi-template files by semicolons (respecting Jinja2 blocks)
3. **Validation**: Validate Jinja2 syntax for each template block
4. **Parameter Extraction**: Extract undeclared variables via `jinja2.meta.find_undeclared_variables()`
5. **LLM Analysis**: Generate business summary and search text using SqlSummaryAgenticNode
6. **Storage**: Store enriched template data in vector store
7. **Indexing**: Create search indices for efficient retrieval

## Summary

Reference Template transforms parameterized SQL templates into an intelligent, searchable knowledge base. It bridges the gap between flexible LLM-driven SQL generation and the stability requirements of production environments.

**Key Features:**
- **Parameterized SQL**: Define query patterns with Jinja2 variables
- **Automatic Parameter Discovery**: Extract parameters from templates without manual annotation
- **Semantic Search**: Find templates by business intent
- **Server-Side Rendering**: Strict rendering with clear error messages for missing parameters
- **Subject Tree Organization**: Hierarchical classification for template discoverability
