# Context Command `@`

## 1. Overview

The Context Command `@` allows you to inject rich contextual knowledge directly into your chat session. Instead of asking the model to guess your data structure, you can explicitly point it to the tables, metrics, or SQL snippets you want it to use — dramatically improving the accuracy of generated SQL.

Context items are organized into three separate trees:

- `@catalog` — your physical data structure (databases, schemas, tables)
- `@subject` — your semantic/business layer (domains, layers, semantic models, metrics)
- `@sql` — your historical knowledge (SQL history, reusable SQL snippets)

By combining these, you give Datus the same mental model you have — so it can reason with your data instead of guessing.

---

## 2. Basic Usage

You can summon the context browser by typing `@` and pressing Tab. Depending on which command you choose, you'll see a tree view you can drill down:

### @catalog
```
catalog
  └── database
      └── schema
          └── table
```

### @subject
```
domain
  └── layer1
      └── layer2
          └── semantic_model
              └── metrics
```

### @sql
```
domain
  └── layer1
      └── layer2
          └── sql_history / sql_snippet
```

When you select an item, it will be injected into the current chat turn as a reference — Datus will know its metadata, lineage, and purpose when generating SQL.

For the initialization of the context tree, you can refer to the Knowledge Base documentation.

## 3. Advanced Features

### Context Injection Modes

There are two ways to inject context:

#### Browse Mode
Type `@` (or `@catalog`, `@subject`, `@sql`) and press Tab to open the tree browser, navigating node by node until you reach the desired table, metric, or SQL.

#### Fuzzy Search Mode
Type `@` (or `@subject orders`) followed by a keyword, then press Tab. Datus will fuzzy search across all context trees and suggest best matches, so you don't need to remember exact paths.

### Examples

```bash
# Browse mode - navigate step by step
@ <Tab>
# Select catalog > my_database > public > customers

# Fuzzy search mode - quick keyword search
@subject revenue <Tab>
# Shows all revenue-related metrics across all domains
```

### Context Types

#### Physical Data (`@catalog`)
- Databases and schemas
- Table structures and column definitions
- Data types and constraints
- Foreign key relationships

#### Business Context (`@subject`)
- Business domains and data layers
- Semantic models and business logic
- Calculated metrics and KPIs
- Business glossary and definitions

#### Historical Knowledge (`@sql`)
- Previously executed SQL queries
- Reusable SQL snippets and templates
- Query patterns and best practices
- Optimization examples