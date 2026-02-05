## 平台文档工具需求

### 目标

为 Datus 增加平台文档检索能力，帮助大模型生成 SQL 时参考官方文档。

### 功能

**1. 文档来源**

- GitHub 仓库
- 官方网站
- 本地文件

**2. 处理流程**

```
抓取 → 解析 → 清洗 → 分段 → 向量存储
```

**3. 数据字段**

- `platform` / `version` - 平台和版本
- `titles` / `hierarchy` - 标题层级
- `chunk_text` - 文档内容
- `keywords` - 关键词

**4. CLI 命令**

```bash
datus bootstrap-kb --components document \
    --doc-source <path> \
    --doc-source-type <github|website|local> \
    --doc-platform <name>
```

**5. 搜索 API**
| 方法 | 功能 |
|------|------|
| `list_document_nav(platform, version?)` | 列出文档目录 |
| `get_document(platform, titles, version?)` | 按标题获取文档 |
| `search_document(platform, keywords, version?)` | 关键词搜索 |

### 技术选型

- 存储: LanceDB
- 向量化: FastEmbed
- 解析: 自实现 Markdown/HTML Parser