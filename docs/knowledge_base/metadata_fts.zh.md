# 元数据全文检索

元数据全文检索（FTS）是查找相关数据表时可选的一种检索方式。它直接检索表元数据中的文本，不需要为这些元数据生成 embedding。

当用户问题中包含明确的业务词、表名、列名、SQL 片段或其它原文关键词时，FTS 更合适。如果用户经常使用元数据中没有出现的说法来描述业务概念，建议继续使用默认的向量检索。

!!! note "生效范围"
    此配置只改变表发现和 schema linking 使用的**元数据检索**。指标、Reference SQL、Reference Template、平台文档等其它知识库组件仍使用各自的检索方式。FTS 只会检索 bootstrap 时采集的样本数据，不会扫描数据源中每一行数据。

## FTS 与向量检索的区别

| | `fts` | `vector` |
|---|---|---|
| 适合场景 | 关键词、标识符、SQL 文本和明确的业务词 | 语义相似和改写后的自然语言问题 |
| 元数据 embedding | 不生成 | 需要生成 |
| 是否为默认值 | 否 | 是 |
| 检索失败时回退 | 不会自动回退到向量检索 | 不适用 |

FTS 会索引以下元数据文本：

- 表名、完整限定名、唯一标识和表类型
- 表或视图定义（DDL）
- `bootstrap-kb` 采集的样本数据
- 已关联的表语义描述，包括 description、列、关系和 AI context 等信息

检索结果仍受当前 datasource，以及 Datus Agent 应用的 catalog、database、schema、表类型和 subagent 范围限制。

## 前置条件

- 已配置 datasource，且数据库账号有读取元数据和样本数据的权限。
- 使用支持 FTS 的向量存储后端。内置 LanceDB 无需额外配置即可使用。
- 如果 `storage.vector` 使用外部适配器，请确认所安装的适配器版本实现了 Datus FTS 接口。不支持 FTS 的后端会直接报错，不会静默切换为向量检索。

## 启用 FTS

在 bootstrap 和日常运行共同使用的 `agent.yml` 中添加：

```yaml
kb:
  search:
    mode: fts
```

没有此配置时，默认值是 `vector`。

`bootstrap-kb` 也支持 `--kb_search_mode fts`，但该参数只覆盖当前命令，不会保存配置。要让 chat、API、后台同步和后续 agent 运行都使用 FTS，必须把配置写入 `agent.yml`。

## 首次构建索引

首次启用 FTS，或者从向量检索切换到 FTS 时，需要完整重建元数据：

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components metadata \
  --kb_update_strategy overwrite
```

`overwrite` 只替换当前 datasource 的元数据，并使用当前索引格式创建 FTS 索引，不会重建其它知识库组件。

临时测试时，可以在命令末尾添加 `--kb_search_mode fts`。请注意，后续运行仍会从 `agent.yml` 读取检索模式。

## 验证索引

使用同一份配置执行只读检查：

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components metadata \
  --kb_update_strategy check
```

检查成功时，结果会包含 `search_mode=fts`、`schema_size` 和 `value_size`，同时确认 FTS 索引存在且符合当前索引规范。

之后可以正常使用 Datus。例如启动 CLI、选择 datasource，然后询问数据表：

```text
datus
/datasource <your_datasource>
哪个表包含客户订单状态？
```

元数据工具和 schema linking 会自动使用已配置的 FTS 检索路径。

## 日常更新

完成首次完整构建后，日常 schema 变化可以使用增量更新：

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components metadata \
  --kb_update_strategy incremental
```

增量模式会更新或新增发生变化的元数据，并维护现有索引。它要求 FTS 索引已经处于可用状态，不会创建或修复缺失的索引。

以下情况需要重新执行 `overwrite`：

- 首次启用 FTS，或者切换存储后端
- Datus 报告索引状态为 `missing`、`legacy` 或 `version_mismatch`
- 上一次构建被中断，索引不完整
- 升级说明明确要求重建 FTS 索引

## 切回向量检索

修改配置并重建元数据，确保向量存储和 embedding 都是最新的：

```yaml
kb:
  search:
    mode: vector
```

```bash
datus-agent bootstrap-kb \
  --datasource <your_datasource> \
  --components metadata \
  --kb_update_strategy overwrite
```

## 故障排查

### 存储后端不支持 FTS

如果错误信息中出现 `FTS-capable vector backend`，说明 `storage.vector` 选择的后端没有提供所需的 FTS 能力。可以使用内置 LanceDB、安装支持 FTS 的适配器版本，或者继续使用 `kb.search.mode: vector`。

### 索引缺失或版本不兼容

如果错误信息中出现 `missing`、`legacy` 或 `version_mismatch`，Datus 不会用向量检索隐藏问题。请使用 `--kb_update_strategy overwrite` 重建元数据。

### 新环境执行 incremental 失败

增量模式只能维护已经存在且状态正常的索引。请先执行一次 `overwrite`，后续更新再使用 `incremental`。

### Bootstrap 使用了 FTS，但日常运行仍在使用向量检索

如果只在构建命令中使用了 `--kb_search_mode fts`，那么只有该命令临时使用 FTS。请将 `kb.search.mode: fts` 写入 `agent.yml`，然后重新执行只读检查。

### 无法找到语义相关但没有相同关键词的表

FTS 按文本匹配排序，不会推断语义相似性。可以为元数据补充更明确的业务描述或表语义信息，使用 schema 中实际出现的词进行检索，或者对大量改写表达的场景切回向量检索。

!!! tip "其它组件仍可能需要 embedding"
    FTS 只避免为元数据生成 embedding。如果还会构建指标、Reference SQL、文档或其它依赖向量检索的组件，请继续保留这些组件所需的 embedding 配置和凭据。
