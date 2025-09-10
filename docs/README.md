## 🎯 Overview

**Datus** is an AI-powered agent that transforms data engineering and metric management into a conversational experience.

![DatusArchitecure](assets/datus_architecture.svg)

With the **Datus Agent** you can:

- **Simplify Data Engineering Development:**
    - Enable data engineers to develop and debug using natural language, reducing entry barriers and increasing productivity.
- **Standardize and Manage Metrics:**
    - Extract and unify metrics consistently, ensuring your BI and AI tools always access accurate and reliable definitions.
- **Self-Improving:**
    - Convert iterative CoT reasoning workflows into structured datasets, enabling SFT and RL for ongoing, automatic improvements in model accuracy and performance.


## ✨ Why Choose Datus Agent?

## 🚀 Key Features

### 💬 **Conversational Data Engineering**

- **Natural Language Workflows** - Use `/` to execute complex task in plain language
- **Intelligent SQL Generation** - `!gen` creates optimized SQL with `!fix` for instant corrections
- **Live Workflow Monitoring** - `!darun_screen` shows real-time execution status
- **Schema Intelligence** - `!sl` provides smart table and column recommendations

### 📈 **Smart Metrics Management**

- **Automated Metric Generation** - `!gen_metrics` extracts business metrics from your queries
- **Semantic Model Creation** - `!gen_semantic_model` builds comprehensive data models
- **Streaming Analytics** - Real-time metric generation with `!gen_metrics_stream` variants
- **Context-Aware Operations** - `!set` manages different workflow contexts

### 🔄 **Self-Improving AI System**

- **Reasoning Mode** - `!reason` provides step-by-step analysis with detailed CoT for complex problems
- **Standard log Output -** Comprehensively record the user’s reasoning process to generate high-value data for subsequent model refinement and evolution


## 💡 Use Cases

Data Pipeline Development

```bash
# Natural language query execution
!reason "create a pipeline that aggregates daily sales by region"

# View recommended tables
!sl
# Schema linking found: sales_data, regions, daily_transactions

# Generate and refine SQL
!gen
# Generated: SELECT region_id, DATE(sale_date) as day, SUM(amount)...

!fix add product category grouping
# Updated SQL with category dimension added

```

Metric Standardization

```bash
# Check existing metrics
@subject

# Generate new metrics from analysis
!gen_metrics_stream
# Streaming metric generation...
# ✓ Monthly Active Users (MAU)
# ✓ Average Order Value (AOV)
# ✓ Customer Lifetime Value (CLV)

# Create semantic model
!gen_semantic_model
# Generated comprehensive data model with relationships

```

Intelligent Debugging

```
# Start debugging session
!dastart "debug ETL memory error"

# Explore context
@context_screen
# Visual display of current tables, schemas, and resources

# Run reasoning analysis
!reason_stream
# Analyzing: Large dataset (10TB) without partitioning detected
# Suggesting: Date-based partitioning, chunked processing

# Apply fix
!fix implement suggested partitioning stratege
```

## Get more

* 🚦 [Quick Start ](Quickstart.md)
* 🤝 [Contribution](Contribute.md)
* 📝 [Release Notes](Release_notes.md)
* 🌱 [Good First Issue](good_first_issue.md)
* 🏗️ [Architecture](Architecture.md)
