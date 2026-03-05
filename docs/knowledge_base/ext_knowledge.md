# External Knowledge Intelligence

## Overview

Bootstrap-KB External Knowledge is a component that processes, stores, and indexes domain-specific business knowledge to create an intelligent searchable knowledge base. It transforms business rules and concepts into a structured repository with semantic search capabilities.

## Core Value

### What Problem Does It Solve?

- **Business Knowledge Silos**: Domain knowledge scattered across teams without centralized access
- **Terminology Ambiguity**: Different interpretations of business terms across the organization
- **Context Gap**: SQL agents lacking understanding of business-specific concepts
- **Knowledge Onboarding**: New team members struggling to understand domain-specific terminology

### What Value Does It Provide?

- **Unified Knowledge Base**: Centralized repository for business terminology and rules
- **Semantic Search**: Find relevant knowledge using natural language queries
- **Agent Context Enhancement**: Enriches SQL generation with domain understanding
- **Knowledge Preservation**: Captures business expertise in a structured, searchable format

## Usage

### Basic Command

```bash
# From CSV (direct import)
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components ext_knowledge \
    --ext_knowledge /path/to/knowledge.csv \
    --kb_update_strategy overwrite

# From success story (AI generation)
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components ext_knowledge \
    --success_story /path/to/success_story.csv \
    --kb_update_strategy overwrite
```

### Key Parameters

| Parameter              | Required | Description                                                       | Example                           |
| ---------------------- | -------- | ----------------------------------------------------------------- | --------------------------------- |
| `--namespace`          | ✅       | Database namespace                                                | `analytics_db`                    |
| `--components`         | ✅       | Components to initialize                                          | `ext_knowledge`                   |
| `--ext_knowledge`      | ⚠️       | Path to knowledge CSV file (required if no `--success_story`)     | `/data/knowledge.csv`             |
| `--success_story`      | ⚠️       | Path to success story CSV file (required if no `--ext_knowledge`) | `/data/success_story.csv`         |
| `--kb_update_strategy` | ✅       | Update strategy                                                   | `overwrite`/`incremental`         |
| `--subject_tree`       | ❌       | Predefined subject tree categories                                | `Finance/Revenue,User/Engagement` |
| `--pool_size`          | ❌       | Concurrent processing threads, default is 1                       | `8`                               |

## Data Source Formats

### Direct Import

Import pre-defined knowledge entries directly from a CSV file.

#### CSV Format

| Column         | Required | Description                | Example                       |
| -------------- | -------- | -------------------------- | ----------------------------- |
| `subject_path` | Yes      | Hierarchical category path | `Finance/Revenue/Metrics`     |
| `name`         | Yes      | Knowledge entry name       | `GMV Definition`              |
| `search_text`  | Yes      | Searchable business term   | `GMV`                         |
| `explanation`  | Yes      | Detailed description       | `Gross Merchandise Volume...` |

#### Example CSV

```csv
subject_path,name,search_text,explanation
Finance/Revenue/Metrics,GMV Definition,GMV,"Gross Merchandise Volume (GMV) represents the total value of merchandise sold through the platform, including both paid and unpaid orders."
User/Engagement/DAU,DAU Definition,DAU,"Daily Active Users (DAU) counts unique users who performed at least one activity within a calendar day."
User/Engagement/Retention,Retention Rate,retention rate,"The percentage of users who return to the platform after their first visit, typically measured at Day 1, Day 7, and Day 30 intervals."
```

### AI Generation

Generate knowledge automatically from question-SQL pairs using AI agent.

#### CSV Format

| Column         | Required | Description                           | Example                                 |
| -------------- | -------- | ------------------------------------- | --------------------------------------- |
| `question`     | Yes      | Business question or query intent     | `What is the total GMV for last month?` |
| `sql`          | Yes      | SQL query that answers the question   | `SELECT SUM(amount) FROM orders...`     |
| `subject_path` | No       | Hierarchical category path (optional) | `Finance/Revenue/Metrics`               |

#### Example CSV

```csv
question,sql,subject_path
"What is the total GMV for last month?","SELECT SUM(amount) as gmv FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)",Finance/Revenue/Metrics
"How many daily active users do we have?","SELECT COUNT(DISTINCT user_id) as dau FROM user_activity WHERE activity_date = CURDATE()",User/Engagement/DAU
"What is our 7-day retention rate?","SELECT COUNT(DISTINCT d7.user_id) / COUNT(DISTINCT d0.user_id) as retention FROM users d0 LEFT JOIN users d7 ON d0.user_id = d7.user_id",User/Engagement/Retention
```

#### How AI Generation Works

The success story mode uses GenExtKnowledgeAgenticNode which supports two operating modes:

- **Workflow Mode**: When `question` and `gold_sql` are provided as structured fields (e.g., from CSV batch processing), they are used directly.
- **Agentic Mode**: When only a `user_message` is provided (e.g., interactive chat), the system uses a lightweight LLM to parse and extract the question and reference SQL from the message.

The generation pipeline:

1. **Analyze Question-SQL Pairs**: Understands the business intent behind each query
2. **Extract Business Concepts**: Identifies key terminology, rules, and patterns
3. **Generate Knowledge Entries**: Creates structured knowledge with search_text and explanation
4. **Classify Categories**: Assigns appropriate subject paths for organization
5. **Verify SQL**: Uses `verify_sql` tool to compare agent-generated SQL against the hidden reference SQL. If verification fails, the system provides comparison feedback (match rate, column differences, data preview, and improvement suggestions) generated via `CompareAgenticNode`
6. **Retry on Failure**: If verification fails, the system automatically retries up to `max_verification_retries` times (default: 3), injecting retry prompts that guide the agent to correct or create knowledge entries and re-verify

## Update Strategies

### 1. Overwrite Mode

Clears existing knowledge and loads fresh data:

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components ext_knowledge \
    --ext_knowledge /path/to/knowledge.csv \
    --kb_update_strategy overwrite
```

### 2. Incremental Mode

Adds new knowledge entries while preserving existing ones. Entries with the same `subject_path + name` are automatically updated (upsert):

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components ext_knowledge \
    --success_story /path/to/success_story.csv \
    --kb_update_strategy incremental
```

## Subject Tree Categorization

Subject tree provides a hierarchical taxonomy for organizing knowledge entries.

### 1. Predefined Mode

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components ext_knowledge \
    --success_story /path/to/success_story.csv \
    --kb_update_strategy overwrite \
    --subject_tree "Finance/Revenue/Metrics,User/Engagement/DAU"
```

### 2. Learning Mode

When no subject_tree is provided, the system:

- Reuses existing categories from the Knowledge Base
- Creates new categories as needed based on content
- Builds taxonomy organically over time

## Integration with SQL Agent

External knowledge integrates with the SQL generation agent through context search tools:

1. **Automatic Context Retrieval**: When generating SQL, the agent queries relevant business knowledge
2. **Term Resolution**: Ambiguous business terms are resolved using stored definitions
3. **Rule Application**: Business rules stored in knowledge base guide SQL logic

### Example Workflow

1. User asks: "Calculate the GMV for last month"
2. Agent searches knowledge for "GMV"
3. Finds definition: "GMV = total value of merchandise including paid and unpaid orders"
4. Generates SQL with correct business logic

## Summary

The Bootstrap-KB External Knowledge component transforms scattered business knowledge into an intelligent, searchable knowledge base.

**Key Features:**

- **Dual Import Modes**: Direct CSV import or AI-driven generation from success stories
- **Dual Operating Modes**: Workflow mode (structured input) and agentic mode (free-form user message with LLM parsing)
- **Unified Repository**: Centralized storage for business terminology and rules
- **Semantic Search**: Find knowledge using natural language queries
- **Hierarchical Organization**: Navigate knowledge through subject path taxonomy
- **Flexible Classification**: Support both predefined and learning modes
- **Agent Integration**: Enhances SQL generation with domain context
- **Upsert Deduplication**: Entries with the same `subject_path + name` are automatically updated rather than duplicated
- **SQL Verification Loop**: AI-generated knowledge is validated by comparing SQL results against a hidden reference, with automatic retry and feedback
- **Batch Knowledge Retrieval**: `get_knowledge` supports fetching multiple entries by paths in a single call

By implementing external knowledge, teams can ensure consistent understanding of business concepts and enable intelligent SQL generation with domain awareness.

## Best Practices: Building Knowledge Base End-to-End

This section walks through the complete process of building an external knowledge base using two approaches, demonstrated with the `california_schools` database.

### Scenario

The goal is to build knowledge that enables the SQL agent to correctly answer:

> "Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names."

The key business knowledge is:

- **Full name** means first name + last name
- There are at most **3 administrators** for each school (`AdmFName1/AdmLName1`, `AdmFName2/AdmLName2`, `AdmFName3/AdmLName3`)
- **SAT Scores >= 1500** refers to the `NumGE1500` column in the `satscores` table

The expected SQL:

```sql
SELECT T2.AdmFName1, T2.AdmLName1, T2.AdmFName2, T2.AdmLName2, T2.AdmFName3, T2.AdmLName3
FROM satscores AS T1
INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode
ORDER BY T1.NumGE1500 DESC
LIMIT 1
```

### Approach 1: Bootstrap from Success Stories (Batch Workflow)

Best for bulk import from existing question-SQL pairs, initial KB builds, and CI/CD pipelines.

#### Step 1: Prepare the Success Story CSV

Create a CSV file (e.g., `success_story.csv`) with `question` and `sql` columns. Optionally include `subject_path` for pre-classification:

```csv
question,sql,subject_path
"Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names.","SELECT T2.AdmFName1, T2.AdmLName1, T2.AdmFName2, T2.AdmLName2, T2.AdmFName3, T2.AdmLName3 FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1",Education/SAT/Administrators
```

#### Step 2: Run the Bootstrap Command

```bash
datus-agent bootstrap-kb \
    --namespace california_schools \
    --components ext_knowledge \
    --success_story /path/to/success_story.csv \
    --kb_update_strategy overwrite \
    --subject_tree "Education/SAT/Administrators,Education/SAT/Scores,Education/Schools"
```

#### Step 3: What Happens Internally

1. Each CSV row is read and converted into an `ExtKnowledgeNodeInput` with `question` and `gold_sql` passed as structured fields (workflow mode)
2. A `GenExtKnowledgeAgenticNode` is created in `workflow` mode for each row
3. The agent analyzes the question-SQL pair, extracts business concepts, and generates a knowledge YAML file
4. The agent calls `verify_sql` to compare its generated SQL against the hidden reference SQL. If `match_rate < 100%`, `CompareAgenticNode` generates improvement suggestions, and the agent retries (up to `max_verification_retries` times, default 3)
5. Once verification passes, the knowledge entry is **automatically saved** to the Knowledge Base (no user confirmation needed in workflow mode)

#### Bootstrap Log Output

During execution, the bootstrap process outputs logs that track progress and verification status:

```log
[info     ] Verification status updated: passed=True, match_rate=1.0 [datus.agent.node.gen_ext_knowledge_agentic_node]
[info     ] Agentic loop ended. Verification passed: True, attempt: 1/4 [datus.agent.node.gen_ext_knowledge_agentic_node]
[info     ] Successfully upserted 2 items in batch [datus.storage.subject_tree.store]
[info     ] Successfully upserted 2 external knowledge entries to Knowledge Base [datus.cli.generation_hooks]
[info     ] Successfully saved to database: Upserted 2 knowledge entries: SAT Score Record Types, Admin Full Names Columns [datus.agent.node.gen_ext_knowledge_agentic_node]
[info     ] Auto-saved to database: Education_SAT_Administrators_knowledge.yaml [datus.agent.node.gen_ext_knowledge_agentic_node]
[info     ] Generated knowledge for: Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names. [datus.storage.ext_knowledge.ext_knowledge_init]
[info     ] Final Result: {'status': 'success', 'message': 'ext_knowledge bootstrap completed, knowledge_size=2'} [__main__]
```

#### Step 4: Expected Output

The system generates a knowledge YAML file like:

```yaml
name: "SAT Score Record Types"
search_text: "SAT scores highest school district record type rtype filter"
explanation: "When querying SAT data for 'schools' with highest scores: (1) The satscores table has rtype column where 'S'=School level, 'D'=District level; (2) When question mentions 'school' without explicit specification, do NOT add rtype='S' filter - the question may refer to any educational entity including districts; (3) District-level records often have higher aggregate numbers than individual schools."
subject_path: "Education/SAT/Administrators"
created_at: "2025-01-15T10:00:00Z"
---
name: "Admin Full Names Columns"
search_text: "administrator full name school multiple administrators"
explanation: "When retrieving administrator 'full names' from schools table: (1) Include ALL administrator columns: AdmFName1, AdmLName1, AdmFName2, AdmLName2, AdmFName3, AdmLName3; (2) Schools may have multiple administrators (up to 3); (3) Even if most entries only have one administrator, query all 6 columns to ensure complete coverage of 'full names' as requested."
subject_path: "Education/SAT/Administrators"
created_at: "2025-01-15T10:00:00Z"
```

#### Step 5: Verify the Result

Start the cli, first use `@subject` to browse the generated knowledge entries, then test with the original question:

```bash
datus-agent --namespace california_schools
```

```
# Browse the knowledge tree and entries
Datus> @subject
# Should show Education/SAT/Administrators and Education/SAT/Scores with the generated knowledge entries
```

```
# Test with the original question
Datus> Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names.
```

The agent should:

1. Search the knowledge base and retrieve entries about `NumGE1500` and administrator full name columns
2. Generate SQL that joins `satscores` with `schools`, orders by `NumGE1500 DESC`, and selects all 6 administrator name columns (`AdmFName1`, `AdmLName1`, `AdmFName2`, `AdmLName2`, `AdmFName3`, `AdmLName3`)
3. Return results matching the expected SQL output

### Approach 2: Subagent Interactive Mode

Best for ad-hoc knowledge creation, exploring and debugging, or refining individual entries.

#### Step 1: Start the REPL

```bash
datus-agent --namespace california_schools
```

#### Step 2: Invoke the Subagent

Use the `/gen_ext_knowledge` slash command, pasting the question and reference SQL together in the message:

```
Datus> /gen_ext_knowledge Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names. Reference SQL: SELECT T2.AdmFName1, T2.AdmLName1, T2.AdmFName2, T2.AdmLName2, T2.AdmFName3, T2.AdmLName3 FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1
```

#### Step 3: What Happens Internally

1. The system creates a `GenExtKnowledgeAgenticNode` in `interactive` mode
2. Since only `user_message` is provided (no structured `question`/`gold_sql` fields), a lightweight LLM **parses the message** to extract the question and reference SQL separately
3. The agent enters its agentic loop: analyzes the question, queries the database schema, generates knowledge entries, and writes them to a YAML file
4. `verify_sql` validates the generated SQL against the hidden reference. If it fails, the agent automatically retries with targeted feedback

#### Step 4: Confirm Database Sync

In interactive mode, `GenerationHooks` intercepts the `write_file` call and prompts for confirmation:

```
[Knowledge Generated] School Administrator Full Names
Sync this knowledge entry to the Knowledge Base? [y/n]: y
```

Approve to save the entry to the Knowledge Base. You can also decline and edit the YAML file manually before re-syncing.

### When to Use Which

| Aspect | Bootstrap (batch) | Subagent (interactive) |
|--------|-------------------|------------------------|
| Use case | Bulk import from existing Q&A pairs | Ad-hoc knowledge creation/refinement |
| Input | CSV file with `question`, `sql` columns | Free-form message in REPL |
| Gold SQL handling | Passed directly as structured fields | Parsed from user message by LLM |
| DB save | Automatic (no confirmation) | User confirms via hook prompt |
| Verification | Automatic retry loop | Automatic retry loop |
| Best for | Initial KB build, CI/CD pipelines | Exploring, debugging, single-entry fixes |

### Tips

1. **Emphasize knowledge usage when verifying**: When testing in the CLI, add "Please search the knowledge base first" to your question to ensure the agent uses stored knowledge rather than relying solely on its own reasoning.
2. **Pre-build subject tree**: Create the subject tree structure in advance via `@subject`. Subsequent runs without `--subject_tree` will automatically reuse existing categories (learning mode), giving you control over taxonomy without requiring `subject_path` in every CSV row.
3. **Iterate for stability**: Run bootstrap multiple times with `incremental` mode. Each run may produce improved knowledge entries, and the upsert mechanism ensures existing entries are updated with better content.
