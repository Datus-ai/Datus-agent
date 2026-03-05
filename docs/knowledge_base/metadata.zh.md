# Schema 元数据智能化

## 简介

元数据模块主要用于使 LLM 能够根据用户问题快速匹配可能相关的表定义信息和样本数据。

当你使用 `bootstrap-kb` 命令时，我们会将你指定的数据源中创建表/视图/物化视图的 SQL 语句和样本数据初始化到向量数据库中。

此模块包含两种类型的信息：**表定义**和**样本数据**。

## 表定义的数据结构

| 字段名称       | 说明 | 支持的数据库类型 |
|------------------|-------------|-----------------------------|
| `catalog_name` | 数据库系统中的顶层容器。它通常表示数据库的集合，并提供关于它们的元数据，例如可用的 schema、表和安全设置 | StarRocks/Snowflake |
| `database_name` | 存储相关数据的逻辑容器。它通常将多个 schema 组合在一起，并为数据组织、安全和管理提供边界 | DuckDB/MySQL/StarRocks/Snowflake |
| `schema_name` | 数据库内的命名空间。它将表、视图、函数和过程等对象组织到逻辑组中。Schema 有助于避免名称冲突并支持基于角色的访问 | DuckDB/Snowflake |
| `table_type` | 数据库中的表类型，包括 `table`、`view` 和 `mv`（物化视图的缩写）。每个数据库都支持表和视图。DuckDB 和 Snowflake 支持物化视图 | 所有支持的数据库 |
| `table_name` | 表/视图/物化视图的名称 | 所有支持的数据库 |
| `definition` | 创建表/视图/物化视图的 SQL 语句 | 所有支持的数据库 |
| `identifier` | 当前表的唯一标识符，由 `catalog_name`、`database_name`、`schema_name` 和 `table_name` 组成。你不需要担心它，因为在大多数情况下你不需要它 | 所有支持的数据库 |

## 样本数据的数据结构

| 字段名称 | 说明 |
|------------|-------------|
| `catalog_name` | 同上 |
| `database_name` | 同上 |
| `schema_name` | 同上 |
| `table_type` | 同上 |
| `table_name` | 同上 |
| `sample_rows` | 当前表/视图/物化视图的样本数据。通常是当前表中的前 5 条记录 |
| `identifier` | 同上 |

## 如何构建

你可以使用 `datus-agent bootstrap-kb` 命令构建：

```bash
datus-agent bootstrap-kb --namespace <your_namespace> --kb_update_strategy [check/overwrite/incremental]
```

### 命令行参数说明

- `--namespace`：你的数据库配置对应的键
- `--kb_update_strategy`：执行策略，有三个选项：
    - `check`：检查当前构建的数据条目数
    - `overwrite`：完全覆盖现有数据
    - `incremental`：增量更新：如果现有数据已更改，则更新它并追加不存在的数据

## 使用示例

### 检查当前状态
```bash
datus-agent bootstrap-kb --namespace <your_namespace> --kb_update_strategy check
```

### 完全重建
```bash
datus-agent bootstrap-kb --namespace <your_namespace> --kb_update_strategy overwrite
```

### 增量更新
```bash
datus-agent bootstrap-kb --namespace <your_namespace> --kb_update_strategy incremental
```

## 最佳实践

### 数据库配置
- 确保你的数据库命名空间在 `agent.yml` 中正确配置
- 运行 bootstrap 命令前验证数据库连接性
- 使用具有系统表读取权限的适当凭据

### 更新策略选择
- 使用 `check` 在不做更改的情况下验证当前状态
- 使用 `overwrite` 进行初始设置或当 schema 发生重大变化时
- 使用 `incremental` 进行常规更新以捕获新表和更改

### 性能考虑
- 大型数据库在初始 bootstrap 期间可能需要时间处理
- 考虑在生产数据库的非高峰时段运行
- 监控磁盘空间，因为元数据存储在 LanceDB 本地

## 故障排除

### 常见问题
- **权限错误**：确保数据库用户有权访问系统/信息 schema 表
- **连接超时**：检查网络连接性和数据库可用性
- **大结果集**：如果数据库非常大，考虑过滤到特定 schema

### 验证
Bootstrap 完成后，验证元数据是否正确捕获：

- 检查 LanceDB 存储目录中的填充文件
- 通过 CLI 测试搜索功能
- 验证样本数据代表实际表内容
