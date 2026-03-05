# Reference SQL 智能化

## 概览

Bootstrap-KB Reference SQL是一个强大的组件，用于处理、分析和索引 SQL 查询文件，创建智能可搜索的存储库。它将带注释的原始 SQL 文件转换为具有语义搜索能力的结构化知识库。

## 核心价值

### 解决什么问题？

- **SQL 知识孤岛**：SQL 查询分散在文件中，缺乏组织
- **SQL 复用性**：难以找到满足相似需求的现有查询
- **查询发现**：没有高效的方法按业务意图搜索 SQL
- **知识管理**：SQL 专业知识锁定在个别开发人员的头脑中

### 提供什么价值？

- **智能组织**：自动分类和归类 SQL 查询
- **语义搜索**：使用自然语言描述查找 SQL 查询
- **知识保存**：以可搜索的格式捕获 SQL 专业知识
- **查询复用**：轻松发现和复用现有 SQL 模式

## 使用方法

### 基本命令

```bash
# 初始化Reference SQL组件
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components reference_sql \
    --sql_dir /path/to/sql/directory \
    --kb_update_strategy overwrite
```

### 关键参数

| 参数 | 必需 | 描述 | 示例 |
|-----------|----------|-------------|------------|
| `--namespace` | ✅ | 数据库命名空间 | `analytics_db` |
| `--components` | ✅ | 要初始化的组件 | `reference_sql` |
| `--sql_dir` | ✅ | 包含 SQL 文件的目录 | `/sql/queries` |
| `--kb_update_strategy` | ✅ | 更新策略 | `overwrite`/`incremental` |
| `--validate-only` | ❌ | 仅验证，不存储 |  |
| `--pool_size` | ❌ | 并发处理线程数，默认值为 4 | `8` |
| `--subject_tree` | ❌ | 用于分类的预定义主题树分类 | `Analytics/User/Activity,Analytics/Revenue/Daily` |

### 主题树分类

主题树提供了一个层级分类法，用于按域和层级组织 SQL 查询。这有助于保持一致的分类并提高查询的可发现性。

#### 格式

主题树分类遵循格式：`domain/layer1/layer2`

示例：`Analytics/User/Activity`、`Analytics/Revenue/Daily`、`Reporting/Sales/Monthly`

#### 使用模式

**1. 预定义模式（使用 --subject_tree）**

当你提供预定义分类时，SQL 查询将仅使用这些分类进行归类：

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components reference_sql \
    --sql_dir /path/to/sql/queries \
    --kb_update_strategy overwrite \
    --subject_tree "Analytics/User/Activity,Analytics/Revenue/Daily,Reporting/Sales/Monthly"
```

**2. 学习模式（不使用 --subject_tree）**

当未提供 subject_tree 时，系统在学习模式下运行：
- 复用知识库中的现有分类
- 分析相似的 SQL 查询以建议适当的分类
- 根据查询内容按需创建新分类
- 随时间有机地构建分类法

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components reference_sql \
    --sql_dir /path/to/sql/queries \
    --kb_update_strategy overwrite
```

**优势：**
- **一致性**：确保 SQL 查询遵循组织分类法
- **可发现性**：通过语义搜索使查询更易查找
- **知识管理**：以结构化方式组织 SQL 专业知识
- **模式识别**：将相似查询分组以提高复用性

## SQL 文件格式

### 期望格式

SQL 文件应使用注释块描述每个查询。每个 SQL 语句必须以分号（`;`）结束。

```sql
-- Daily active users count
-- Count unique users who logged in each day
SELECT
    DATE(created_at) as activity_date,
    COUNT(DISTINCT user_id) as daily_active_users
FROM user_activity
WHERE created_at >= '2025-01-01'
GROUP BY DATE(created_at)
ORDER BY activity_date;

-- Monthly revenue summary
-- Total revenue grouped by month and category
SELECT
    DATE_TRUNC('month', order_date) as month,
    category,
    SUM(amount) as total_revenue,
    COUNT(*) as order_count
FROM orders
WHERE order_date >= '2025-01-01'
GROUP BY DATE_TRUNC('month', order_date), category
ORDER BY month, total_revenue DESC;
```

### 格式要求

1. **分号分隔符**：每个 SQL 语句必须以分号（`;`）结束
2. **注释格式**：使用 SQL 行注释（`--`）描述查询（可选）
      - 紧接在 SQL 语句之前的注释将与该查询关联
      - 多行注释连接成单个描述
      - 如果不需要，可以省略注释
3. **仅 SELECT 查询**：仅处理和存储 `SELECT` 语句
4. **SQL 方言**：支持 MySQL、Hive 和 Spark SQL 方言
5. **参数占位符**：支持以下参数样式：
      - `#parameter#` - 井号分隔的参数
      - `:parameter` - 冒号前缀参数
      - `@parameter` - at 符号参数
      - `${parameter}` - shell 风格参数

### 文件组织

- 将所有 SQL 文件（`.sql` 扩展名）放在单个目录中
- 文件递归处理
- 无效的 SQL 条目记录到 `sql_processing_errors.log` 以供审查

## 总结

Bootstrap-KB Reference SQL组件将分散的 SQL 文件转换为智能、可搜索的知识库。它结合了高级 NLP 能力和强大的 SQL 处理，创建了一个强大的 SQL 发现和复用工具。

**关键特性：**
- **智能组织**：使用主题树分类法自动归类 SQL 查询
- **灵活分类**：支持预定义和学习模式进行分类
- **语义搜索**：使用自然语言描述查找 SQL 查询
- **知识保存**：以结构化、可搜索的格式捕获 SQL 专业知识
- **自动上下文检索**：分析相似查询以建议适当的分类

通过实施Reference SQL，团队可以打破知识孤岛，构建随时间增长的集体 SQL 智能资产。
