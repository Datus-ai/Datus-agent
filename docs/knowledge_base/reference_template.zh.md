# Reference Template 智能化

## 概览

Bootstrap-KB Reference Template 是一个知识库组件，用于处理、分析和索引参数化的 Jinja2 SQL 模板。它将原始 `.j2` 模板文件转换为具有语义搜索、参数元数据提取和服务端渲染能力的可搜索存储库。

## 核心价值

### 解决什么问题？

- **SQL 稳定性**：LLM 生成的 SQL 在不同运行间可能存在差异，导致生产环境不一致
- **参数化查询**：重复查询仅在参数（日期、地区、阈值）上有所不同
- **模板发现**：没有高效的方法按业务意图查找现有模板
- **可控输出**：需要将 SQL 生成约束在预审批的查询模式中

### 提供什么价值？

- **稳定 SQL 输出**：基于预定义模板渲染参数，而非从头生成 SQL
- **参数感知**：自动提取并暴露模板参数，供 LLM 驱动的参数填充
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
| `--pool_size` | 否 | 并发处理线程数（默认：4） | `8` |
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

## 工具

Bootstrap 完成后，Agent 可使用三个工具：

### `search_reference_template`

通过自然语言查询搜索模板。返回匹配的模板及其参数元数据。

### `get_reference_template`

通过 `subject_path` + `name` 精确获取特定模板。返回完整的模板内容和参数列表。

### `render_reference_template`

使用提供的参数值渲染模板。使用 Jinja2 的 `StrictUndefined` 模式 — 缺少参数时会产生可操作的错误信息，列出期望参数与已提供参数的对比。

## 数据流

```text
模板文件 (.j2)  -->  文件处理器  -->  LLM 分析  -->  存储  -->  工具
     |                  |              |            |          |
  解析模板块        验证 J2 语法     生成摘要     向量数据库   search/
  提取参数          过滤无效模板    和搜索文本    + 索引构建   get/render
  分号分割
```

### 处理流程

1. **文件发现**：查找模板目录中的 `.j2`/`.jinja2` 文件
2. **模板分割**：按分号分割多模板文件（尊重 Jinja2 块结构）
3. **语法验证**：验证每个模板块的 Jinja2 语法
4. **参数提取**：通过 `jinja2.meta.find_undeclared_variables()` 提取未声明变量
5. **LLM 分析**：使用 SqlSummaryAgenticNode 生成业务摘要和搜索文本
6. **存储入库**：将增强后的模板数据存入向量数据库
7. **索引构建**：创建搜索索引以支持高效检索

## 总结

Reference Template 将参数化 SQL 模板转换为智能、可搜索的知识库。它弥补了灵活的 LLM 驱动 SQL 生成与生产环境稳定性需求之间的差距。

**关键特性：**
- **参数化 SQL**：使用 Jinja2 变量定义查询模式
- **自动参数发现**：从模板中提取参数，无需手动标注
- **语义搜索**：按业务意图查找模板
- **服务端渲染**：严格渲染，缺少参数时提供清晰的错误信息
- **主题树组织**：层级分类提升模板可发现性
