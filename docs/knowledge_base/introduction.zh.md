# 知识库简介

Datus Agent 知识库是一个多模态智能系统，将分散的数据资产转换为统一的、可搜索的存储库。可以将其视为"数据的 Google"，深入理解 SQL、业务指标和数据关系。

## 核心目的

- **数据发现**：查找相关的表、列和模式
- **查询智能**：理解业务意图并生成 SQL
- **知识保存**：捕获和组织 SQL 专业知识
- **语义搜索**：按含义查找信息，而非关键字

## 核心组件

### 1. [Schema 元数据](metadata.zh.md)

**目的**：理解数据库结构并提供智能表推荐。

- **存储**：表定义、列信息、样本数据、统计信息
- **能力**：按业务含义查找表、获取表结构、语义搜索
- **用途**：自动表选择、数据发现、schema 理解

### 2. [语义模型](semantic_model.zh.md)

**目的**：通过语义信息增强数据库 schema 以改进 SQL 生成。

- **存储**：表结构、维度、度量、实体关系
- **能力**：Schema linking、列使用模式、外键发现
- **用途**：准确的临时 SQL 生成、智能过滤、正确的 JOIN 构造

### 3. [业务指标](metrics.zh.md)

**目的**：管理和查询标准化的业务 KPI。

- **存储**：指标定义、主题树分类
- **能力**：通过 MetricFlow 直接查询指标、指标优先策略
- **用途**：一致的报告、消除重复 SQL、标准化定义

### 4. [Reference SQL](reference_sql.zh.md)

**目的**：捕获、分析和使 SQL 专业知识可搜索。

- **存储**：历史查询、LLM 摘要、查询模式、最佳实践
- **能力**：按意图查找查询、获取相似查询、学习模式
- **用途**：知识共享、通过示例优化、团队入职

### 5. [Reference Template](reference_template.zh.md)

**目的**：管理参数化 SQL 模板，实现稳定、可重复的查询生成。

- **存储**：Jinja2 模板、参数定义、LLM 摘要、主题树分类
- **能力**：按意图搜索模板、获取含参数元数据的模板、服务端渲染
- **用途**：生产环境稳定 SQL 输出、参数化报表查询、基于模板的 SQL 生成

### 6. [外部知识](ext_knowledge.zh.md)

**目的**：处理和索引领域特定业务知识，实现智能搜索。

- **存储**：业务术语、规则、概念、层级分类
- **能力**：业务术语语义搜索、上下文增强、术语解析
- **用途**：Agent 上下文增强、术语标准化、知识传承

### 7. [平台文档](platform_doc.zh.md)

**目的**：提供权威的平台文档，用于 SQL 生成与语法校验。

- **存储**：按平台与版本组织的官方文档分块
- **能力**：导航浏览、文档检索、语义搜索
- **用途**：在生成 SQL 前校验平台特有语法与能力

## 存储后端 {#storage-backends}

所有知识库组件的数据存储基于双轨架构：

- **向量数据库（Vector DB）**：存储 embedding 向量，驱动语义搜索（schema linking、文档搜索等）
- **关系数据库（RDB）**：存储结构化元数据（task、feedback、success story 等）

Datus Agent 通过 Registry + entry-point 机制支持可插拔存储后端，无需修改业务代码即可切换。

### 默认：LanceDB + SQLite
- 零配置，开箱即用
- 数据存储在 `data/datus_db_<namespace>/`
- 适用于开发、单机部署

### PostgreSQL (pgvector)
- 生产级后端，安装 `datus-storage-postgresql` 包后自动注册
- 向量存储：pgvector 扩展提供向量搜索
- 关系存储：原生 PostgreSQL 关系存储
- 通过 PostgreSQL schema 实现 namespace 隔离

### Namespace 隔离
- 每个 namespace 独立存储，互不干扰
  - LanceDB：每个 namespace 一个独立目录
  - PostgreSQL：每个 namespace 一个独立 schema

> 详细配置请参考 [存储配置](../configuration/storage.zh.md#storage-backends)

## 关键特性

- **统一搜索**：跨所有知识域的单一界面
- **语义搜索**：使用向量嵌入按含义查找
- **智能分类**：自动分类和组织
- **可扩展**：延迟加载、批处理、增量更新
