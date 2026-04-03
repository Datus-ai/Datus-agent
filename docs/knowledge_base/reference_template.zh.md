# Reference Template 智能化

## 概览

Bootstrap-KB Reference Template 是一个知识库组件，用于处理、分析和索引参数化的 Jinja2 SQL 模板。它将原始 `.j2` 模板文件转换为具有语义搜索、参数元数据提取和服务端渲染能力的可搜索存储库。

## 核心价值

### 解决什么问题？

- **SQL 稳定性**：LLM 生成的 SQL 在不同运行间可能存在差异，导致生产环境不一致
- **参数化查询**：重复查询仅在参数（日期、地区、阈值）上有所不同
- **模板发现**：没有高效的方法按业务意图查找现有模板
- **可控输出**：需要将 SQL 生成约束在预审批的查询模式中
- **参数歧义**：LLM 不知道每个参数的合法取值范围

### 提供什么价值？

- **稳定 SQL 输出**：基于预定义模板渲染参数，而非从头生成 SQL
- **参数智能化**：自动推断参数类型、解析列引用、并从数据库中提供候选值
- **语义搜索**：使用自然语言描述查找模板
- **服务端渲染**：Jinja2 渲染在服务端执行，使用严格的未定义变量检查

## 使用方法

### 基本命令

```bash
# 初始化 Reference Template 组件
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components reference_template \
    --template_dir /path/to/template/directory \
    --kb_update_strategy overwrite
```

### 关键参数

| 参数 | 必需 | 描述 | 示例 |
|------|------|------|------|
| `--namespace` | 是 | 数据库命名空间 | `analytics_db` |
| `--components` | 是 | 要初始化的组件 | `reference_template` |
| `--template_dir` | 是 | 包含 J2 模板文件的目录 | `/templates/queries` |
| `--kb_update_strategy` | 是 | 更新策略 | `overwrite`/`incremental` |
| `--validate-only` | 否 | 仅验证，不存储 | |
| `--pool_size` | 否 | 并发处理线程数（默认：1） | `8` |
| `--subject_tree` | 否 | 预定义主题树分类 | `Analytics/User/Activity,Reporting/Sales/Monthly` |

### 主题树分类

主题树提供了一个层级分类法，用于按域组织模板。与 Reference SQL 使用相同的机制。

**预定义模式**（使用 `--subject_tree`）：

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components reference_template \
    --template_dir /path/to/templates \
    --kb_update_strategy overwrite \
    --subject_tree "Analytics/User/Activity,Reporting/Sales/Monthly"
```

**学习模式**（不使用 `--subject_tree`）：

系统复用现有分类，并根据需要创建新分类。

## 模板文件格式

### 支持的扩展名

- `.j2` — 标准 Jinja2 模板扩展名
- `.jinja2` — 替代 Jinja2 扩展名

### 单模板文件

每个 `.j2` 文件包含一个带有 Jinja2 参数的 SQL 模板：

```sql
SELECT `Free Meal Count (Ages 5-17)` / NULLIF(`Enrollment (Ages 5-17)`, 0) AS free_rate
FROM frpm
WHERE `Educational Option Type` = '{{school_type}}'
  AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL
ORDER BY free_rate {{sort_order}}
LIMIT {{limit}}
```

### 多模板文件

一个文件中包含多个模板，用分号（`;`）分隔：

```sql
SELECT T2.Zip
FROM frpm AS T1
INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode
WHERE T1.`District Name` = '{{district_name}}'
  AND T1.`Charter School (Y/N)` = 1
;
SELECT T1.Phone
FROM schools AS T1
INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds
WHERE T1.County = '{{county}}'
  AND T2.NumTstTakr < {{max_test_takers}}
```

### Jinja2 语法支持

- **变量**：`{{ variable_name }}` — 自动提取为模板参数
- **条件语句**：`{% if condition %}...{% endif %}`
- **循环语句**：`{% for item in items %}...{% endfor %}`
- **注释**：`{# comment #}`

Jinja2 块结构（`{% if %}`、`{% for %}` 等）内部的分号不会被视为模板分隔符。

### 格式要求

1. **分号分隔符**：多模板文件中的模板必须用 `;` 分隔
2. **合法 Jinja2**：模板必须通过 Jinja2 语法验证
3. **SQL 内容**：模板渲染后应产生合法的 SQL

## 参数类型系统

在 Bootstrap 过程中，系统会自动分析每个 `{{ variable }}` 占位符的 SQL 上下文，确定其类型。表别名（如 `T1`、`T2`）会被自动解析为真实表名。

### 参数类型

| 类型 | 检测规则 | 元数据增强 |
|------|---------|-----------|
| `dimension` | 出现在 `WHERE col = '{{param}}'` 中 | `column_ref`：真实的 table.column；`sample_values`：数据库中出现频率最高的 10 个值 |
| `column` | 出现在 `GROUP BY {{param}}` 或 `SELECT {{param}}` 中 | `table_refs`：涉及的表列表；`sample_values`：可用的列名列表 |
| `keyword` | 出现在 `ORDER BY expr {{param}}` 中 | `allowed_values`：合法关键字（如 `["ASC", "DESC"]`） |
| `number` | 出现在 `LIMIT {{param}}` 或比较运算符中 | — |

### 示例

给定以下模板：

```sql
SELECT {{group_column}}, COUNT(*) AS school_count
FROM frpm
WHERE `Educational Option Type` = '{{school_type}}'
GROUP BY {{group_column}}
ORDER BY school_count {{sort_order}}
LIMIT {{limit}}
```

Bootstrap 过程会生成：

```json
[
  {
    "name": "group_column",
    "type": "column",
    "table_refs": ["frpm"],
    "sample_values": ["CDSCode", "County Name", "District Name", "School Name", "..."],
    "description": "用于分组的列名"
  },
  {
    "name": "school_type",
    "type": "dimension",
    "column_ref": "frpm.`Educational Option Type`",
    "sample_values": ["Traditional", "Continuation School", "Charter School", "..."],
    "description": "要筛选的教育选项类型"
  },
  {
    "name": "sort_order",
    "type": "keyword",
    "allowed_values": ["ASC", "DESC"],
    "description": "结果排序方向"
  },
  {
    "name": "limit",
    "type": "number",
    "description": "返回的最大行数"
  }
]
```

这使得 LLM 在调用 `execute_reference_template` 时能够准确知道每个参数应该填什么值。

## 工具

Bootstrap 完成后，Agent 可使用四个工具：

### `search_reference_template`

通过自然语言查询搜索模板。返回匹配的模板元数据（名称、类型、摘要、标签），不返回模板正文以节省 token。

### `get_reference_template`

通过 `subject_path` + `name` 精确获取特定模板。返回完整的模板内容、带有 `sample_values` 的增强参数信息和摘要。

### `render_reference_template`

使用提供的参数值渲染模板，返回最终 SQL 字符串但不执行。使用 Jinja2 的 `StrictUndefined` 模式 — 缺少参数时会产生可操作的错误信息，列出期望参数与已提供参数的对比。

### `execute_reference_template`

渲染模板并立即执行生成的 SQL（只读）。将 `render_reference_template` + `read_query` 合并为一步操作。返回渲染后的 SQL 和查询结果行。

注意：`execute_reference_template` 会自动创建内部数据库连接 — 使用模板工具时不需要单独配置 `db_tools`。

## 纯模板模式

当需要 Agent 只能执行预审批的模板（禁止自由编写 SQL）时，可使用专用的 `ref_tpl` 系统提示词：

```yaml
agentic_nodes:
  template_executor:
    model: deepseek-v3
    system_prompt: ref_tpl
    prompt_version: '1.0'
    max_turns: 10
    tools: context_search_tools.list_subject_tree, reference_template_tools.search_reference_template, reference_template_tools.get_reference_template, reference_template_tools.execute_reference_template
```

在此模式下，Agent：

- **必须**先搜索模板
- 找到匹配后**必须**使用 `execute_reference_template` — 绝不手动编写 SQL
- 无匹配时直接回复"未找到匹配的 SQL 模板"并停止

## 数据流

```text
模板文件 (.j2)  →  文件处理器  →  参数分析  →  LLM 分析  →  存储  →  工具
     |                |             |             |          |         |
  解析模板块      验证 J2 语法   推断类型       生成摘要   向量数据库  search/
  提取参数        过滤无效模板   解析别名      和搜索文本   + 索引     get/execute
  分号分割                     查询候选值
```

### 处理流程

1. **文件发现**：查找模板目录中的 `.j2`/`.jinja2` 文件
2. **模板分割**：按分号分割多模板文件（尊重 Jinja2 块结构）
3. **语法验证**：验证每个模板块的 Jinja2 语法
4. **参数提取**：通过 `jinja2.meta.find_undeclared_variables()` 提取未声明变量
5. **参数分析**：推断参数类型，将表别名解析为真实表名，从数据库查询候选值
6. **LLM 分析**：使用 SqlSummaryAgenticNode 生成业务摘要、搜索文本和参数描述
7. **合并**：将静态分析的参数类型与 LLM 生成的描述合并
8. **存储入库**：将增强后的模板数据存入向量数据库
9. **索引构建**：创建搜索索引以支持高效检索

## 总结

Reference Template 将参数化 SQL 模板转换为智能、可搜索的知识库。它弥补了灵活的 LLM 驱动 SQL 生成与生产环境稳定性需求之间的差距。

**关键特性：**

- **参数化 SQL**：使用 Jinja2 变量定义查询模式
- **参数智能化**：自动推断类型（`dimension`、`column`、`keyword`、`number`）、解析列引用、提供候选值
- **语义搜索**：按业务意图查找模板
- **一步执行**：搜索、渲染、执行模板合并为一次工具调用
- **纯模板模式**：专用系统提示词限制 Agent 只能使用预审批模板
- **主题树组织**：层级分类提升模板可发现性
