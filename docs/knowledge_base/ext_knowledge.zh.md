# 外部知识智能化

## 概述

Bootstrap-KB 外部知识是一个处理、存储和索引领域特定业务知识的组件，用于创建智能可搜索的知识库。它将业务规则和概念转化为具有语义搜索能力的结构化知识库。

## 核心价值

### 解决什么问题？

- **业务知识孤岛**：领域知识分散在各团队，缺乏集中访问
- **术语歧义**：组织内对业务术语有不同的理解
- **上下文缺失**：SQL Agent 缺乏对业务特定概念的理解
- **知识传承**：新团队成员难以理解领域特定术语

### 提供什么价值？

- **统一知识库**：业务术语和规则的集中存储库
- **语义搜索**：使用自然语言查询查找相关知识
- **Agent 上下文增强**：通过领域理解丰富 SQL 生成
- **知识沉淀**：以结构化、可搜索的格式捕获业务专业知识

## 使用方法

### 基本命令

```bash
# 从 CSV（直接导入）
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components ext_knowledge \
    --ext_knowledge /path/to/knowledge.csv \
    --kb_update_strategy overwrite

# 从 success story（AI 生成）
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components ext_knowledge \
    --success_story /path/to/success_story.csv \
    --kb_update_strategy overwrite
```

### 关键参数

| 参数                   | 必需 | 描述                                                            | 示例                              |
| ---------------------- | ---- | --------------------------------------------------------------- | --------------------------------- |
| `--namespace`          | ✅   | 数据库命名空间                                                  | `analytics_db`                    |
| `--components`         | ✅   | 要初始化的组件                                                  | `ext_knowledge`                   |
| `--ext_knowledge`      | ⚠️   | 知识 CSV 文件路径（如果没有 `--success_story` 则必需）          | `/data/knowledge.csv`             |
| `--success_story`      | ⚠️   | Success story CSV 文件路径（如果没有 `--ext_knowledge` 则必需） | `/data/success_story.csv`         |
| `--kb_update_strategy` | ✅   | 更新策略                                                        | `overwrite`/`incremental`         |
| `--subject_tree`       | ❌   | 预定义主题树分类                                                | `Finance/Revenue,User/Engagement` |
| `--pool_size`          | ❌   | 并发处理线程数，默认为 1                                        | `8`                               |

## 数据源格式

### 直接导入

从 CSV 文件直接导入预定义的知识条目。

#### CSV 格式

| 列             | 必需 | 描述             | 示例                          |
| -------------- | ---- | ---------------- | ----------------------------- |
| `subject_path` | 是   | 层级分类路径     | `Finance/Revenue/Metrics`     |
| `name`         | 是   | 知识条目名称     | `GMV Definition`              |
| `search_text`  | 是   | 可搜索的业务术语 | `GMV`                         |
| `explanation`  | 是   | 详细说明         | `Gross Merchandise Volume...` |

#### 示例 CSV

```csv
subject_path,name,search_text,explanation
Finance/Revenue/Metrics,GMV Definition,GMV,"Gross Merchandise Volume (GMV) represents the total value of merchandise sold through the platform, including both paid and unpaid orders."
User/Engagement/DAU,DAU Definition,DAU,"Daily Active Users (DAU) counts unique users who performed at least one activity within a calendar day."
User/Engagement/Retention,Retention Rate,retention rate,"The percentage of users who return to the platform after their first visit, typically measured at Day 1, Day 7, and Day 30 intervals."
```

### AI 生成

使用 AI Agent 从问题-SQL 对自动生成知识。

#### CSV 格式

| 列             | 必需 | 描述                 | 示例                                    |
| -------------- | ---- | -------------------- | --------------------------------------- |
| `question`     | 是   | 业务问题或查询意图   | `What is the total GMV for last month?` |
| `sql`          | 是   | 回答问题的 SQL 查询  | `SELECT SUM(amount) FROM orders...`     |
| `subject_path` | 否   | 层级分类路径（可选） | `Finance/Revenue/Metrics`               |

#### 示例 CSV

```csv
question,sql,subject_path
"What is the total GMV for last month?","SELECT SUM(amount) as gmv FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)",Finance/Revenue/Metrics
"How many daily active users do we have?","SELECT COUNT(DISTINCT user_id) as dau FROM user_activity WHERE activity_date = CURDATE()",User/Engagement/DAU
"What is our 7-day retention rate?","SELECT COUNT(DISTINCT d7.user_id) / COUNT(DISTINCT d0.user_id) as retention FROM users d0 LEFT JOIN users d7 ON d0.user_id = d7.user_id",User/Engagement/Retention
```

#### AI 生成工作原理

success story 模式使用 GenExtKnowledgeAgenticNode，支持两种运行模式：

- **Workflow 模式**：当 `question` 和 `gold_sql` 作为结构化字段提供时（如 CSV 批量处理），直接使用。
- **Agentic 模式**：当仅提供 `user_message` 时（如交互式对话），系统使用轻量级 LLM 解析并提取问题和参考 SQL。

生成流程：

1. **分析问题-SQL 对**：理解每个查询背后的业务意图
2. **提取业务概念**：识别关键术语、规则和模式
3. **生成知识条目**：创建包含 search_text 和 explanation 的结构化知识
4. **分类归类**：分配适当的 subject path 进行组织
5. **验证 SQL**：使用 `verify_sql` 工具将 Agent 生成的 SQL 与隐藏的参考 SQL 进行对比。如果验证失败，系统通过 `CompareAgenticNode` 提供对比反馈（匹配率、列差异、数据预览和改进建议）
6. **失败重试**：如果验证失败，系统自动重试最多 `max_verification_retries` 次（默认 3 次），注入重试提示引导 Agent 修正或创建知识条目并重新验证

## 更新策略

### 1. 覆盖模式

清除现有知识并加载新数据：

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components ext_knowledge \
    --ext_knowledge /path/to/knowledge.csv \
    --kb_update_strategy overwrite
```

### 2. 增量模式

添加新知识条目同时保留现有条目。相同 `subject_path + name` 的条目会自动更新（upsert）：

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components ext_knowledge \
    --success_story /path/to/success_story.csv \
    --kb_update_strategy incremental
```

## 主题树分类

主题树提供层级分类法来组织知识条目。

### 1. 预定义模式

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components ext_knowledge \
    --success_story /path/to/success_story.csv \
    --kb_update_strategy overwrite \
    --subject_tree "Finance/Revenue/Metrics,User/Engagement/DAU"
```

### 2. 学习模式

当未提供 subject_tree 时，系统会：

- 复用知识库中的现有分类
- 根据内容按需创建新分类
- 随时间自然构建分类体系

## 与 SQL Agent 集成

外部知识通过上下文搜索工具与 SQL 生成 Agent 集成：

1. **自动上下文检索**：生成 SQL 时，Agent 查询相关业务知识
2. **术语解析**：使用存储的定义解析模糊的业务术语
3. **规则应用**：知识库中存储的业务规则指导 SQL 逻辑

### 示例工作流

1. 用户提问："Calculate the GMV for last month"
2. Agent 搜索 "GMV" 相关知识
3. 找到定义："GMV = total value of merchandise including paid and unpaid orders"
4. 生成包含正确业务逻辑的 SQL

## 总结

Bootstrap-KB 外部知识组件将分散的业务知识转化为智能、可搜索的知识库。

**核心特性：**

- **双重导入模式**：直接 CSV 导入或从 success story AI 驱动生成
- **双重运行模式**：Workflow 模式（结构化输入）和 Agentic 模式（自由文本输入，LLM 解析）
- **统一知识库**：业务术语和规则的集中存储
- **语义搜索**：使用自然语言查询查找知识
- **层级组织**：通过 subject path 分类体系导航知识
- **灵活分类**：支持预定义和学习两种模式
- **Agent 集成**：通过领域上下文增强 SQL 生成
- **Upsert 去重**：相同 `subject_path + name` 的条目自动更新而非重复创建
- **SQL 验证循环**：AI 生成的知识通过与隐藏参考 SQL 对比进行验证，支持自动重试和反馈
- **批量知识检索**：`get_knowledge` 支持通过路径列表一次获取多个条目

通过实施外部知识，团队可以确保对业务概念的一致理解，并实现具有领域感知能力的智能 SQL 生成。

## 最佳实践：端到端构建知识库

本节以 `california_schools` 数据库为例，演示使用两种方式构建外部知识库的完整流程。

### 场景

目标是构建知识，使 SQL Agent 能够正确回答：

> "Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names."

关键业务知识：

- **Full name** 指 first name + last name
- 每所学校最多有 **3 位管理员**（`AdmFName1/AdmLName1`、`AdmFName2/AdmLName2`、`AdmFName3/AdmLName3`）
- **SAT 成绩 >= 1500** 对应 `satscores` 表中的 `NumGE1500` 列

期望的 SQL：

```sql
SELECT T2.AdmFName1, T2.AdmLName1, T2.AdmFName2, T2.AdmLName2, T2.AdmFName3, T2.AdmLName3
FROM satscores AS T1
INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode
ORDER BY T1.NumGE1500 DESC
LIMIT 1
```

### 方式一：Bootstrap 批量构建（Workflow 模式）

适用于从已有问答对批量导入、初始知识库构建和 CI/CD 流水线。

#### 步骤 1：准备 Success Story CSV

创建 CSV 文件（如 `success_story.csv`），包含 `question` 和 `sql` 列，可选 `subject_path` 进行预分类：

```csv
question,sql,subject_path
"Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names.","SELECT T2.AdmFName1, T2.AdmLName1, T2.AdmFName2, T2.AdmLName2, T2.AdmFName3, T2.AdmLName3 FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1",Education/SAT/Administrators
```

#### 步骤 2：运行 Bootstrap 命令

```bash
datus-agent bootstrap-kb \
    --namespace california_schools \
    --components ext_knowledge \
    --success_story /path/to/success_story.csv \
    --kb_update_strategy overwrite \
    --subject_tree "Education/SAT/Administrators,Education/SAT/Scores,Education/Schools"
```

#### 步骤 3：内部执行流程

1. 读取每行 CSV，转换为 `ExtKnowledgeNodeInput`，`question` 和 `gold_sql` 作为结构化字段传入（Workflow 模式）
2. 为每行创建 `GenExtKnowledgeAgenticNode`（`workflow` 模式）
3. Agent 分析问题-SQL 对，提取业务概念，生成知识 YAML 文件
4. Agent 调用 `verify_sql` 将生成的 SQL 与隐藏参考 SQL 对比。如果 `match_rate < 100%`，`CompareAgenticNode` 生成改进建议，Agent 重试（最多 `max_verification_retries` 次，默认 3 次）
5. 验证通过后，知识条目**自动保存**到知识库（Workflow 模式无需用户确认）

#### Bootstrap 日志输出

执行过程中，bootstrap 进程输出跟踪进度和验证状态的日志：

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

#### 步骤 4：预期输出

系统生成的知识 YAML 文件示例：

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

#### 步骤 5：验证结果

启动 CLI，先用 `@subject` 浏览生成的知识条目，然后用原始问题测试：

```bash
datus-agent --namespace california_schools
```

```
# 浏览知识树和条目
Datus> @subject
# 应显示 Education/SAT/Administrators 和 Education/SAT/Scores 及其生成的知识条目
```

```
# 用原始问题测试
Datus> Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names.
```

Agent 应该：

1. 搜索知识库，检索关于 `NumGE1500` 和管理员全名列的知识条目
2. 生成 SQL：关联 `satscores` 和 `schools`，按 `NumGE1500 DESC` 排序，选择全部 6 个管理员名称列（`AdmFName1`、`AdmLName1`、`AdmFName2`、`AdmLName2`、`AdmFName3`、`AdmLName3`）
3. 返回与预期 SQL 输出匹配的结果

### 方式二：Subagent 交互模式

适用于临时知识创建、探索调试或优化单个条目。

#### 步骤 1：启动 CLI

```bash
datus-agent --namespace california_schools
```

#### 步骤 2：调用 Subagent

使用 `/gen_ext_knowledge` 斜杠命令，将问题和参考 SQL 一起粘贴到消息中：

```
Datus> /gen_ext_knowledge Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names. Reference SQL: SELECT T2.AdmFName1, T2.AdmLName1, T2.AdmFName2, T2.AdmLName2, T2.AdmFName3, T2.AdmLName3 FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1
```

#### 步骤 3：内部执行流程

1. 系统创建 `GenExtKnowledgeAgenticNode`（`interactive` 模式）
2. 由于仅提供了 `user_message`（没有结构化的 `question`/`gold_sql` 字段），轻量级 LLM **解析消息**，分别提取问题和参考 SQL
3. Agent 进入 agentic 循环：分析问题、查询数据库 schema、生成知识条目并写入 YAML 文件
4. `verify_sql` 将生成的 SQL 与隐藏参考进行验证。如果失败，Agent 自动重试并提供针对性反馈

#### 步骤 4：确认数据库同步

在交互模式下，`GenerationHooks` 拦截 `write_file` 调用并提示确认：

```
[Knowledge Generated] School Administrator Full Names
Sync this knowledge entry to the Knowledge Base? [y/n]: y
```

确认后将条目保存到知识库。也可以拒绝并手动编辑 YAML 文件后重新同步。

### 两种方式对比

| 方面 | Bootstrap（批量） | Subagent（交互） |
|------|-------------------|------------------|
| 使用场景 | 从已有问答对批量导入 | 临时知识创建/优化 |
| 输入 | 包含 `question`、`sql` 列的 CSV 文件 | CLI 中的自由文本消息 |
| Gold SQL 处理 | 作为结构化字段直接传入 | 由 LLM 从用户消息中解析 |
| 数据库保存 | 自动（无需确认） | 用户通过 hook 提示确认 |
| 验证 | 自动重试循环 | 自动重试循环 |
| 最适合 | 初始知识库构建、CI/CD 流水线 | 探索、调试、单条修复 |

### 使用技巧

1. **验证时强调使用知识库**：在 CLI 中测试时，在问题中加上"请先搜索知识库"，确保 Agent 使用已存储的知识而非仅依赖自身推理。
2. **预构建主题树**：通过 `@subject` 提前创建主题树结构。后续不指定 `--subject_tree` 运行时，系统会自动复用已有分类（学习模式），无需在每行 CSV 中都指定 `subject_path`。
3. **多次迭代提升稳定性**：使用 `incremental` 模式多次运行 bootstrap。每次运行可能生成更优的知识条目，upsert 机制确保已有条目以更好的内容更新。
