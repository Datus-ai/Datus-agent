# 工具命令 `!`

## 1. 概览

以 `!` 为前缀的工具命令，为 Datus-CLI 会话提供一系列 AI 加持的能力与实用操作。你可以在不离开交互式终端的前提下，完成模式发现、指标搜索、SQL 参考搜索等智能数据任务。

## 2. 命令分类

### 2.1 模式发现命令

#### `!sl` / `!schema_linking`
执行智能模式匹配，为你的问题查找相关的数据表与字段。

```bash
!sl user purchase information
!schema_linking sales data by region
```

特性：

- 语义级别搜索相关数据库表
- 展示表定义（DDL）
- 预览样例数据
- 可选匹配模式：fast、medium、slow、from_llm
- 可设置 top_n 返回数量

交互式提示将引导你完成：

- 选择 catalog / database / schema
- 指定匹配数据表数量
- 选择偏好的匹配方式

### 2.2 搜索发现命令

#### `!sm` / `!search_metrics`
使用自然语言在数据目录中搜索对应的指标。

```bash
!sm monthly active users
!search_metrics revenue growth rate
```

支持以下筛选条件：

- Domain
- Layer1（业务层）
- Layer2（子层）
- Top N 结果

#### `!sq` / `!search_sql`
使用自然语言描述搜索历史 SQL。

```bash
!sq queries about user retention
!search_sql monthly sales reports
```

返回内容包括：

- SQL 文本（语法高亮）
- 查询摘要与备注
- 标签与分类
- 域/层级元数据
- 文件路径与相关度（距离分数）

### 2.3 实用命令

#### `!save`
将最近一次查询结果保存到文件。

```bash
!save
```

交互式选项：

- 文件类型：json、csv、sql 或 all
- 输出目录（默认 `~/.datus/output`）
- 自定义文件名

#### `!bash <command>`
执行安全的 Bash 命令（有限制）。

```bash
!bash pwd
!bash ls -la
!bash cat config.yaml
```

**安全策略**：仅允许白名单命令：

- `pwd` —— 查看当前目录
- `ls` —— 列出目录内容
- `cat` —— 展示文件内容
- `head` —— 查看文件开头
- `tail` —— 查看文件末尾
- `echo` —— 输出文本

不在白名单内的命令会被拒绝并提示安全警告。

## 3. 最佳实践

1. **先做模式匹配** —— 使用 `!sl` 寻找相关表，再编写查询
2. **善用搜索** —— 使用 `!sm`、`!sq` 优先复用已有指标与查询
3. **保存成果** —— 通过 `!save` 保存重要的查询结果
4. **关注安全** —— 使用 `!bash` 时注意白名单限制

## 4. 安全注意事项

- 工具命令与 Datus-CLI 进程共享权限
- Bash 命令被限制在安全白名单内
- `!bash` 默认 10 秒超时，防止卡死
- 所有操作都会记录日志以供审计
- API 凭证与数据库连接均经过安全处理

