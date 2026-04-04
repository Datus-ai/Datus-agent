# 工作流节点

节点是 Datus Agent 工作流系统的基本构件。每个节点在“理解用户请求 → 生成 SQL → 执行 → 返回结果”的流程中完成一项明确职责。本文介绍节点类型、用途，以及它们如何在工作流中协同。

## 节点分类

### 1. 控制类节点（Control Nodes）

#### Reflect（反思） {#reflect-node}
- **用途**：评估结果并决定下一步
- **要点**：自适应 SQL 生成的核心智能
- **常见策略**：
  - 简单再生成（重试生成 SQL）
  - 文档检索（查找相关文档）
  - 结构再分析（重新审视库表结构）
  - 深度推理分析

#### Parallel（并行）
- **用途**：并行执行多个子节点
- **场景**：对比多种 SQL 生成策略

#### Selection（选择）
- **用途**：从多个候选中选优
- **场景**：在多条候选 SQL 中选择最佳

#### Subworkflow（子工作流）
- **用途**：执行嵌套工作流
- **场景**：复用复杂流程、模块化组合

### 2. 动作类节点（Action Nodes）

#### Schema Linking（结构关联） {#schema-linking-node}
- **用途**：理解用户问题并定位相关数据表
- **关键活动**：
  - 解析自然语言意图
  - 在知识库中搜索相关表
  - 提取表结构与样例数据
  - 将结构信息写入上下文
- **输出**：相关表结构与样例数据列表

#### Generate SQL（生成 SQL） {#generate-sql-node}
- **用途**：基于需求生成查询
- **要点**：
  - 利用大模型理解业务需求
  - 复用历史 SQL 模式
  - 可融合业务指标
  - 处理复杂逻辑
- **输出**：SQL 及执行计划

#### Execute SQL（执行 SQL） {#execute-sql-node}
- **用途**：在数据库中执行查询
- **关键活动**：
  - 连接目标库
  - 安全执行并处理错误
  - 返回结果或错误信息
  - 更新执行上下文
- **输出**：结果、耗时、错误信息

#### Output（输出） {#output-node}
- **用途**：最终结果呈现
- **特性**：
  - 结果格式化与展示
  - 清晰的报错提示
  - 性能指标展示
- **输出**：可读性好的结果

#### Reasoning（推理）
- **用途**：提供深入分析与解释
- **场景**：复杂业务逻辑的说明与校验

#### Fix（修复）
- **用途**：修正存在问题的 SQL
- **要点**：
  - 错误模式识别
  - 自动修正
  - 修正后验证
- **场景**：自动处理失败的 SQL

#### Generate Metrics（生成指标）
- **用途**：从 SQL 中抽取业务指标
- **关键活动**：分析查询 → 识别指标 → 生成定义 → 入库
- **输出**：指标定义及计算方式

#### Generate Semantic Model（生成语义模型）
- **用途**：为数据表创建语义模型
- **要点**：识别维度与度量、定义表语义、生成可复用模型
- **输出**：面向 BI 的语义模型定义

#### Search Metrics（搜索指标）
- **用途**：查找相关业务指标
- **场景**：复用既有口径，确保一致性

#### Compare（对比）
- **用途**：将 SQL 结果与预期对比
- **场景**：测试、验证与质检

#### Date Parser（时间解析）
- **用途**：解析时间表达
- **示例**：
  - “last month” → 具体日期范围
  - “Q3 2023” → 季度边界
  - “past 7 days” → 滚动窗口

#### Document Search（文档检索）
- **用途**：查找相关文档与上下文
- **场景**：为复杂问题补充领域知识

### 3. Agentic 类节点

具备对话式与自适应能力的高阶 AI 节点。

#### Chat Agentic
- **用途**：具工具调用能力的对话式交互
- **要点**：
  - 多轮对话
  - 工具调用
  - 上下文保持
  - 自适应回复
- **场景**：交互式 SQL 生成与迭代

## 实现要点

### 输入/输出结构
```python
class BaseNode:
    def setup_input(self, context: Context) -> NodeInput
    def run(self, input: NodeInput) -> NodeOutput
    def update_context(self, context: Context, output: NodeOutput) -> Context
```

### 上下文管理
```python
class Context:
    sql_contexts: List[SQLContext]
    table_schemas: List[TableSchema]
    metrics: List[BusinessMetric]
    reflections: List[Reflection]
    documents: List[Document]
```

### 错误处理
- 输入校验：必填参数与上下文检查
- 执行安全：处理数据库错误与超时
- 输出校验：确保输出格式正确
- 恢复机制：自动重试与降级策略

## 节点配置

### 模型分配
```yaml
nodes:
  schema_linking:
    model: "claude-3-sonnet"
    temperature: 0.1
  generate_sql:
    model: "gpt-4"
    temperature: 0.2
  reasoning:
    model: "claude-3-opus"
    temperature: 0.3
```

### SQL模板
```yaml
nodes:
  generate_sql:
    prompt_template: "generate_sql_system.j2"
    user_template: "generate_sql_user.j2"
```

### 资源限制
```yaml
nodes:
  execute_sql:
    timeout: 30
    max_rows: 10000
    memory_limit: "1GB"
```

## 最佳实践

### 选择与组合
1. 先做 Schema Linking，补足上下文
2. 复杂场景结合 Reasoning 与 Generate SQL
3. 引入 Reflect 提升稳健性
4. 用 Parallel 比较多种策略

### 性能优化
- 缓存表结构，跨工作流复用
- 依据复杂度选模型规模
- 控制 SQL 返回量
- 监控长任务的资源占用

### 故障恢复
- 逐步降级，尽量给出部分有用结果
- 瞬时失败自动重试
- 面向用户给出可操作的错误信息
- 记录详尽日志以便排错

## 进阶用法

### 自定义节点
```python
class CustomValidationNode(BaseNode):
    def run(self, input: ValidationInput) -> ValidationOutput:
        return ValidationOutput(is_valid=True, message="Validation passed")
```

### 动态工作流
```python
# 在反思节点中
if complexity_score > threshold:
    workflow.add_node("reasoning", after="current")

if needs_validation:
    workflow.add_node("compare", before="output")
```

### 组合示例
```python
# 并行多种 SQL 生成策略
parallel_node = ParallelNode([
    GenerateSQLNode(strategy="conservative"),
    GenerateSQLNode(strategy="aggressive"),
    GenerateSQLNode(strategy="metric_based")
])

# 选择最优结果
selection_node = SelectionNode(criteria="accuracy")
```

## 结语

节点是让 Datus Agent 工作流既高效又智能的模块化基石。理解各节点职责及其协作方式，有助于构建能够适应复杂需求、并持续产出高质量结果的 SQL 生成工作流。
