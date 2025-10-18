[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blueviolet?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)

## 🎯 Overview

**Datus** is a data engineering agent that transforms data engineering and metric management into a conversational experience.

![DatusArchitecure](assets/datus_architecture.svg)

<p align="center">
  <a href="https://datus.ai">Website</a> | <a href="https://docs.datus.ai/">Docs</a> | <a href="https://docs.datus.ai/getting_started/Quickstart/">QuickStart</a> | <a href="https://docs.datus.ai/release_notes/">ReleaseNotes</a> 
</p>


## 🚀 Key Features

### 🧩 Contextual Data Engineering  
Automatically builds a **living semantic map** of your company’s data — combining metadata, metrics, SQL history, and external knowledge — so engineers and analysts collaborate through context instead of raw SQL.

### 💬 Agentic Chat  
A **Claude-Code-like CLI** for data engineers.  
Chat with your data, recall tables or metrics instantly, and run agentic actions — all in one terminal.

### 🧠 Subagents for Every Domain  
Turn data domains into **domain-aware chatbots**.  
Each subagent encapsulates the right context, tools, and rules — making data access accurate, reusable, and safe.

### 🔁 Continuous Learning Loop  
Every query and feedback improves the model.  
Datus learns from success stories and user corrections to evolve reasoning accuracy over time.


---

## 🧰 Installation

```bash
pip install datus-agent==0.2.1
```

## 🧭 User Journey

### 1️⃣ Initial Exploration

A Data Engineer (DE) starts by chatting with the database using /chat.
They run simple questions, test joins, and refine prompts using @table or @file.
Each round of feedback (e.g., “Join table1 and table2 by PK”) helps the model improve accuracy.

```

```

2️⃣ Building Context

The DE imports SQL history and generates summaries or semantic models:

!gen_sql_summary  /gen_semantic_model


They edit or refine models in @subject, combining AI-generated drafts with human corrections.
Now, /chat can reason using both SQL history and semantic context.

3️⃣ Creating a Subagent

When the context matures, the DE defines a domain-specific chatbot (Subagent):

.subagent add chatbot


They describe its purpose, add rules, choose tools, and limit scope (e.g., 5 tables).
Each subagent becomes a reusable, scoped assistant for a specific business area.

4️⃣ Delivering to Analysts

The Subagent is deployed to a web interface:
http://localhost:8501/?subagent=wangzhe_new_commerial

Analysts chat directly, upvote correct answers, or report issues for feedback.
Results can be saved via !export.

5️⃣ Refinement & Iteration

Feedback from analysts loops back to improve the subagent:
engineers fix SQL, add rules, and update context.
Over time, the chatbot becomes more accurate, self-evolving, and domain-aware.