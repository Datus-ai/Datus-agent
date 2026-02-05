# 平台文档工具设计方案

## 概述

为 Datus 项目新增平台文档工具，抓取、解析、分段并存储数据库/BI平台官方文档，供大模型在生成 SQL/报告前检索参考。

---

## 1. 模块结构

```
datus/storage/document/
├── __init__.py
├── store.py                  # DocumentStore 存储类
├── doc_init.py               # 初始化入口
├── schemas.py                # 数据模型
├── fetcher/
│   ├── __init__.py
│   ├── base.py               # BaseFetcher 抽象基类
│   ├── local.py              # LocalFetcher 本地文件
│   ├── github.py             # GitHubFetcher
│   └── web.py                # WebFetcher
├── parser/
│   ├── __init__.py
│   ├── markdown.py           # MarkdownParser
│   ├── html.py               # HTMLParser
│   └── metadata.py           # MetadataExtractor
├── chunker/
│   ├── __init__.py
│   └── semantic.py           # SemanticChunker
└── cleaner/
    ├── __init__.py
    └── cleaner.py            # DocumentCleaner

datus/tools/search_tools/
└── search_tool.py            # SearchTool (搜索API)
```

---

## 2. 技术选型

| 组件 | 方案 | 说明 |
|------|------|------|
| 向量存储 | LanceDB | 已有基础设施 |
| 向量化 | FastEmbed | 轻量级嵌入 |
| GitHub 抓取 | PyGithub | 自动处理认证/分页 |
| 网页抓取 | httpx + BeautifulSoup4 | 已有依赖 |
| Markdown 解析 | markdown-it-py | 完整 AST |
| 文档分段 | 自实现 SemanticChunker | 语义感知 |

---

## 3. 数据模型

```python
@dataclass
class PlatformDocChunk:
    chunk_id: str              # 唯一标识
    chunk_text: str            # 文档内容
    chunk_index: int           # 分段序号
    title: str                 # 当前标题
    titles: List[str]          # 标题层级
    hierarchy: str             # 层级字符串 "A > B > C"
    nav_path: str              # 导航路径
    platform: str              # 平台名称
    version: str               # 文档版本
    source_type: str           # github/website/local
    source_url: str            # 来源 URL
    doc_path: str              # 文档路径
    keywords: List[str]        # 关键词
    language: str              # 语言
```

---

## 4. 处理流程

```
抓取 (Fetcher) → 清洗 (Cleaner) → 解析 (Parser) → 分段 (Chunker) → 存储 (Store)
```

**分段策略：**
1. 按 Markdown 标题 (h1-h3) 分割
2. 保持代码块完整性
3. 超大块递归按段落分割
4. 合并过小分段
5. 添加上下文 (titles, hierarchy)

---

## 5. CLI 命令

文档初始化使用 `platform-doc` 命令：

```bash
datus platform-doc --namespace <ns> \
    --source <path>                    # GitHub repo / URL / 本地路径
    --source-type <github|website|local> \
    --platform <name>                  # snowflake, duckdb 等
    [--version <version>]              # 可选，自动检测
    [--chunk-size <size>]              # 默认 1024
    [--max-depth <depth>]              # 网站爬取深度，默认 1
```

**示例：**
```bash
# 本地文档
datus platform-doc --namespace demo \
    --source /path/to/docs \
    --source-type local \
    --platform snowflake

# GitHub 文档
datus platform-doc --namespace demo \
    --source "snowflake/snowflake-arctic" \
    --source-type github \
    --platform snowflake
```

---

## 6. 搜索 API

搜索功能由 `SearchTool` 提供：

```python
class SearchTool:
    def list_document_nav(self, platform: str, version: Optional[str] = None) -> DocNavResult:
        """列出文档导航结构"""

    def get_document(self, platform: str, titles: List[str], version: Optional[str] = None) -> GetDocResult:
        """按标题层级获取文档"""

    def search_document(self, platform: str, keywords: List[str], version: Optional[str] = None, top_n: int = 5) -> DocSearchResult:
        """关键词语义搜索"""
```

---

## 7. 职责分离

| 模块 | 职责 |
|------|------|
| `platform-doc` | 初始化/写入 |
| `SearchTool` | 查询/读取 |

---

## 8. LLM Function Call 集成

### 8.1 PlatformDocSearchTool (`datus/tools/func_tool/platform_doc_search.py`)

为 LLM 提供文档检索 Function Call 能力，遵循现有 `ContextSearchTools` 模式：

```python
class PlatformDocSearchTool:
    def __init__(self, agent_config: AgentConfig)
    def available_tools(self) -> List[Tool]      # 无文档时返回空列表
    def list_document_nav(platform, version?) -> FuncToolResult
    def get_document(platform, titles, version?) -> FuncToolResult
    def search_document(platform, keywords, version?, top_n?) -> FuncToolResult
```

**关键设计：**
- 懒加载 `DocumentStore`，首次访问时检测文档是否存在（`_has_document` 标志）
- 无文档时 `available_tools()` 返回空列表，不注册任何工具
- 每个方法内部委托给 `SearchTool`，封装为统一的 `FuncToolResult`

### 8.2 list_document_nav 返回树形结构

导航树按 `nav_path` 分组，返回轻量级树形结构供 LLM 浏览钻取：

```
[
    {
        "name": "SQL Reference",
        "children": [
            {"name": "DDL", "children": [], "docs": ["CREATE TABLE", "ALTER TABLE"]}
        ],
        "docs": []
    }
]
```

**叶子节点命名规则：**
- `nav_path == titles` → 叶子名称取 `nav_path` 最后一个元素
- `nav_path != titles` → 叶子名称取文档 `title`

**多版本支持：**
- 传入 `version` 或仅有单一版本 → 返回扁平树
- 未传 `version` 且存在多版本 → 按版本顶层分组：
  ```
  [{"version": "v3.4.0", "tree": [...]}, {"version": "v3.3.0", "tree": [...]}]
  ```

### 8.3 get_document 使用说明

`titles` 参数表示单个文档的层级路径（父组 + 文档标题），所有元素 AND 匹配：
- 正确: `titles=["DDL", "CREATE TABLE"]` → 获取 DDL 下的 CREATE TABLE
- 错误: `titles=["DDL", "CREATE TABLE", "ALTER TABLE"]` → 不会返回结果
- 获取多个文档需多次调用

### 8.4 Prompt 模板集成

在 `sql_system_1.1.j2` 和 `chat_system_1.1.j2` 中增加 platform doc tools 的使用说明。

**模板上下文变量：**
- `has_platform_doc_tools`: 布尔值，在 `prepare_template_context()` 中通过 `bool(self._platform_doc_tool)` 设置
- 每个工具按 `native_tools` 条件渲染，避免未配置的工具出现在 prompt 中引发 LLM 强行调用

**条件渲染逻辑（sql_system_1.1.j2）：**
```jinja2
{% if has_platform_doc_tools -%}
    - Platform Documentation Tools for ...:
    {% if "platform_doc_search_tools" in native_tools or "platform_doc_search_tools.list_document_nav" in native_tools %}
        - list_document_nav: ...
    {% endif %}
    {% if "platform_doc_search_tools" in native_tools or "platform_doc_search_tools.get_document" in native_tools %}
        - get_document: ...
    {% endif %}
    {% if "platform_doc_search_tools" in native_tools or "platform_doc_search_tools.search_document" in native_tools %}
        - search_document: ...
    {% endif -%}
{% endif -%}
```

此模式与 `context_search_tools` 一致：
- `"platform_doc_search_tools" in native_tools` → 全部工具启用
- `"platform_doc_search_tools.get_document" in native_tools` → 仅启用单个工具

### 8.5 Agent 节点集成

| 文件 | 修改内容 |
|------|----------|
| `gen_sql_agentic_node.py` | `_setup_platform_doc_tools()` 初始化工具；`prepare_template_context()` 增加 `has_platform_doc_tools` 参数 |
| `chat_agentic_node.py` | `setup_tools()` 中调用 `_setup_platform_doc_tools()` |
| `sql_system_1.1.j2` | 在 Capabilities 区域增加条件渲染的工具描述 |
| `chat_system_1.1.j2` | 在 "You have access to" 区域增加工具描述 |

---

## 9. 微批处理优化

### 问题
原流程一次性将所有文档加载到内存（抓取→处理→存储），大型文档仓库（数百文件）峰值内存约为总文档大小的 3 倍。

### 方案：两阶段微批处理

**阶段 1 — 轻量元数据收集：**
- 收集所有 `file_paths`（字符串）
- 解析 `nav_map`（仅需路径 + 少量 index 文件）

**阶段 2 — 分批内容处理：**
- 每批 N 个文件：抓取内容 → 处理为 chunks → 立即存储
- 批次处理完后数据即释放，峰值内存降至约 0.3x

**关键文件：**
- `github_fetcher.py`: 增加 `collect_metadata()` + `fetch_batch()` 方法
- `doc_init.py`: 增加 `batch_size` 参数，提取 `_process_batch()` 辅助函数

---

## 10. 实现状态

- [x] 目录结构
- [x] 数据模型 (schemas.py)
- [x] DocumentStore 存储类
- [x] LocalFetcher
- [x] GitHubFetcher
- [x] WebFetcher
- [x] MarkdownParser
- [x] HTMLParser
- [x] SemanticChunker
- [x] DocumentCleaner
- [x] CLI 集成 (platform-doc)
- [x] SearchTool API
- [x] list_document_nav 树形结构（含多版本分组）
- [x] PlatformDocSearchTool Function Call 工具
- [x] get_document docstring 防误用说明
- [x] Prompt 模板集成（sql_system + chat_system）
- [x] 按 native_tools 条件渲染各工具描述
- [x] 单元测试（23 passed）
- [ ] 微批处理优化（已设计，待实现）
