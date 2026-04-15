# BI Dashboard Subagent Skill & Tool 设计文档

## 1. 整体架构

BI Dashboard 功能采用**三层分离**架构：

```
┌─────────────────────────────────────────────────────┐
│                  Agent Layer (本仓库)                 │
│  ChatAgenticNode → task(type="gen_dashboard")        │
│    → GenDashboardAgenticNode → BIFuncTool            │
│       → LLM Function Calling                         │
│  Skill Definitions (SKILL.md) → 工作流编排            │
│  CLI Bootstrap → 子 Agent 自动生成                    │
├─────────────────────────────────────────────────────┤
│              Adapter Registry (datus-bi-core)         │
│  BIAdapterBase / Mixin 接口 / AuthParam / 数据模型    │
│  adapter_registry → Entry Point 自动发现              │
├─────────────────────────────────────────────────────┤
│          Platform Adapters (独立包)                    │
│  datus-bi-superset / datus-bi-grafana                │
│  各平台 API 的具体实现                                │
└─────────────────────────────────────────────────────┘
```

核心设计原则：
- **Agent 层不依赖具体平台**：通过 `adapter_registry` 动态发现已安装的适配器
- **Mixin 决定能力**：工具按适配器实现的 Mixin 动态暴露，不做硬编码
- **数据物化桥接**：`write_query` 把源数据库查询结果写入 BI 平台自己的数据库，解耦源库与 BI 展示层

---

## 2. BIFuncTool — LLM 工具层

**文件**: `datus/tools/func_tool/bi_tools.py`

BIFuncTool 是 LLM Agent 与 BI 平台之间的桥梁，将 `datus_bi_core` 的适配器能力包装为 OpenAI Agents SDK 的 function tool。

### 2.1 工具清单与能力检测

工具按适配器实现的 Mixin 类型**动态暴露**：

| 工具 | 所需 Mixin | 说明 |
|------|-----------|------|
| `list_dashboards` | 基础（所有适配器） | 列出/搜索仪表盘 |
| `get_dashboard` | 基础 | 获取仪表盘详情 |
| `list_charts` | 基础 | 列出仪表盘下的图表 |
| `list_datasets` | 基础 | 列出数据集/数据源 |
| `create_dashboard` | `DashboardWriteMixin` | 创建仪表盘 |
| `update_dashboard` | `DashboardWriteMixin` | 更新仪表盘标题/描述 |
| `delete_dashboard` | `DashboardWriteMixin` | 删除仪表盘 |
| `create_chart` | `ChartWriteMixin` | 创建图表/面板 |
| `update_chart` | `ChartWriteMixin` | 更新图表配置 |
| `add_chart_to_dashboard` | `ChartWriteMixin` | 将图表添加到仪表盘 |
| `delete_chart` | `ChartWriteMixin` | 删除图表 |
| `create_dataset` | `DatasetWriteMixin` | 注册数据集 |
| `list_bi_databases` | `DatasetWriteMixin` | 列出 BI 平台数据库连接 |
| `delete_dataset` | `DatasetWriteMixin` | 删除数据集 |
| `write_query` | 需配置 `dataset_db_uri` | 物化查询结果到 BI 数据库 |

**能力检测机制**：`available_tools()` 方法通过 `isinstance(adapter, XxxMixin)` 判断适配器支持哪些操作，只返回适配器实际支持的工具。

### 2.2 write_query — 数据物化桥梁

`write_query` 是 BI 工具链中最关键的操作：

```
源数据库 ──(SQL查询)──→ read_connector ──(结果集)──→ dataset_db (BI平台数据库)
                                                         ↓
                                              物化为物理表 (table_name)
                                                         ↓
                                              BI 图表从此表查询
```

- 在**源数据库**上执行分析 SQL（通过 `read_connector`）
- 将结果写入 **BI 平台自有数据库**（通过 `dataset_db_uri` 配置的 SQLAlchemy 连接）
- 返回 `database_id` 供后续 `create_dataset` 使用

### 2.3 Grafana 特殊处理

- **Datasource UID 自动解析**：Grafana 的 `create_chart` 需要 datasource UID，BIFuncTool 自动通过 `datasource_name` 配置或数据库名称查找匹配的 datasource
- **无独立 Dataset 概念**：Grafana 图表直接嵌入 SQL，不需要先创建 Dataset
- **`update_chart` 不支持**：Grafana 面板需要删除后重建

---

## 3. Skill 定义 — LLM 工作流编排

Skill 以 SKILL.md 文件定义，通过 YAML frontmatter 注册元信息，markdown 正文作为 LLM 的工作流指令。

### 3.1 superset-dashboard Skill

**文件**: `skills/superset-dashboard/SKILL.md`

五步流程：

```
write_query → create_dataset → create_chart → create_dashboard → add_chart_to_dashboard
```

关键规则：
- **必须先 `write_query`**：Superset 图表查 BI 平台数据库，不直接查源库
- **`create_dataset` 必须用 BI 平台的 `database_id`**，不是源数据库 ID
- **Metrics 格式**：纯列名默认 `SUM(col)`，支持显式聚合 `AVG(price)`、`COUNT(id)` 等
- **`big_number` 图表**：单一指标，无需 x_axis 和 dimensions

### 3.2 grafana-dashboard Skill

**文件**: `skills/grafana-dashboard/SKILL.md`

三步流程：

```
write_query → create_dashboard → create_chart(sql=...)
```

与 Superset 的关键差异：
- **无 Dataset 步骤**：Grafana 图表直接嵌入 SQL 查询
- **必须先创建 Dashboard**：`create_chart` 需要 `dashboard_id` 参数
- **时间序列要求**：时间列必须别名为 `time`（`SELECT date_col AS time, ...`）
- **Datasource 自动配置**：从 `datasource_name` 配置自动解析，无需手动指定

### 3.3 Skill 注册属性

```yaml
user_invocable: false        # 用户不能直接调用，由 LLM 自动选择
disable_model_invocation: false  # 允许 LLM 自主调用
tags: [superset/grafana, dashboard, BI, visualization]
```

---

## 4. GenDashboardAgenticNode 集成

**文件**: `datus/agent/node/gen_dashboard_agentic_node.py`

### 4.1 BI 工具初始化流程

```python
def _setup_bi_tools(self):
    bi_platform = self.node_config.get("bi_platform")  # 从节点配置读取
    dash_cfg = self.agent_config.dashboard_config[bi_platform]

    # 1. 从 adapter_registry 获取适配器类
    adapter_cls = adapter_registry.get(bi_platform)

    # 2. 从 dataset_db 配置推导 dialect
    dialect = dash_cfg.dataset_db.get("dialect", "")
    # 或从 URI 自动推导: make_url(uri).get_backend_name()

    # 3. 实例化适配器
    adapter = adapter_cls(api_base_url=api_url, auth_params=auth_params, dialect=dialect)

    # 4. 创建 BIFuncTool（read_connector 在 init 阶段确定，不随会话切库变化）
    self.bi_func_tool = BIFuncTool(
        adapter,
        dataset_db_uri=...,        # BI 平台数据库连接串
        dataset_db_schema=...,     # 物化表的 schema
        read_connector=...,        # 源数据库连接器（初始化时确定）
        datasource_name=...,       # Grafana datasource 名称
    )

    # 5. 注册到工具列表和权限系统
    self.tools.extend(self.bi_func_tool.available_tools())
    self.tool_registry.register_tools("bi_tools", self.bi_func_tool.available_tools())
```

### 4.2 优雅降级

- `datus_bi_core` 未安装时：`ImportError` 被捕获，跳过 BI 工具初始化
- 适配器包未安装时：warning 日志，不影响其他功能

---

## 5. 配置模型

**文件**: `datus/configuration/agent_config.py`

### 5.1 DashboardConfig

```python
@dataclass
class DashboardConfig:
    platform: str                              # 平台标识: "superset" / "grafana"
    api_url: str = ""                          # API 地址
    username: str = ""                         # 登录认证
    password: str = ""
    api_key: str = ""                          # API Key 认证
    extra: Optional[Dict[str, Any]] = None     # 平台特定扩展参数
    dataset_db: Optional[Dict[str, Any]] = None  # 物化目标数据库
    # dataset_db 结构: {uri: "postgresql+psycopg2://...", schema: "public", datasource_name: "..."}
```

### 5.2 agent.yml 配置示例

```yaml
agent:
  dashboard:
    superset:
      api_url: "http://localhost:8088"
      username: "${SUPERSET_USER}"
      password: "${SUPERSET_PASSWORD}"
      dataset_db:
        uri: "${SUPERSET_DB_URI}"
        schema: "public"
    grafana:
      api_url: "http://localhost:3000"
      api_key: "${GRAFANA_API_KEY}"
      dataset_db:
        uri: "${GRAFANA_DB_URI}"
        datasource_name: "PostgreSQL"
```

所有敏感值支持 `${ENV_VAR}` 环境变量替换。

---

## 6. CLI Bootstrap — 从已有仪表盘生成子 Agent

**文件**: `datus/cli/bi_dashboard.py`

`datus-agent bootstrap-bi` 命令从一个已有的 BI 仪表盘自动生成子 Agent 配置。

### 6.1 Bootstrap 流程

```
选择平台/URL/认证
       ↓
连接仪表盘 → 加载图表列表
       ↓
用户选择图表 → 分两组：reference SQL / metrics
       ↓
DashboardAssembler 提取 SQL、表名、数据集
       ↓
生成以下资源：
  ├── Metadata（表结构元数据）
  ├── Reference SQL（RAG 知识库索引）
  ├── Semantic Model（语义模型）
  ├── Metrics（指标定义）
  └── 两个子 Agent 配置
```

### 6.2 生成的子 Agent

| 子 Agent | 类型 | 工具 | 用途 |
|----------|------|------|------|
| `{platform}_{dashboard}` | GenSQL | `context_search_tools`, `db_tools` | 基于仪表盘上下文的 SQL 生成 |
| `{platform}_{dashboard}_attribution` | GenReport | `semantic_tools`, `context_search_tools` | 指标归因分析 |

两个子 Agent 共享同一个 `ScopedContext`（包含表名、参考 SQL、指标的范围限定），确保查询聚焦在仪表盘相关的数据上。

### 6.3 DashboardAssembler

**文件**: `datus/tools/bi_tools/dashboard_assembler.py`

负责从仪表盘 API 提取结构化信息：
- **SQL 候选提取**：从图表配置中解析出 SQL 查询
- **表名规范化**：用当前数据库上下文补全不完整的表名限定符（catalog.database.schema.table）
- **去重逻辑**：处理部分限定名之间的包含关系（如 `schema.table` 与 `catalog.database.schema.table`）

---

## 7. 平台差异对比

| 维度 | Superset | Grafana |
|------|----------|---------|
| Dataset 概念 | 有（物理表/虚拟视图） | 无（SQL 嵌入面板） |
| 创建图表前置条件 | 需要 `dataset_id` | 需要 `dashboard_id` |
| SQL 位置 | Dataset 层 | Panel（图表）层 |
| 数据库连接 | `database_id` + dataset | `datasource_name` 自动解析 |
| 更新图表 | 支持 `update_chart` | 不支持，需删除重建 |
| 认证方式 | Login (用户名/密码) | API Key |
| 完整工作流 | 5 步 | 3 步 |

---

## 8. 数据流全景

```
用户自然语言请求
       ↓
ChatAgenticNode → task(type="gen_dashboard") → GenDashboardAgenticNode
       ↓ 选择 Skill
       ↓
┌──────────────────────────────────────────────────┐
│ Superset 流程:                                    │
│                                                   │
│ write_query(SQL, table_name)                      │
│   → 源 DB 执行 SQL → 结果写入 Superset DB         │
│   → 返回 database_id                              │
│                                                   │
│ create_dataset(name, database_id, [sql])          │
│   → 注册为 Superset 数据集                         │
│   → 返回 dataset_id                               │
│                                                   │
│ create_chart(type, title, dataset_id, metrics...) │
│   → 创建可视化图表                                 │
│   → 返回 chart_id                                 │
│                                                   │
│ create_dashboard(title) → dashboard_id            │
│ add_chart_to_dashboard(chart_id, dashboard_id)    │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│ Grafana 流程:                                     │
│                                                   │
│ write_query(SQL, table_name)                      │
│   → 源 DB 执行 SQL → 结果写入 Grafana DB          │
│                                                   │
│ create_dashboard(title) → dashboard_id            │
│                                                   │
│ create_chart(type, title, sql=..., dashboard_id)  │
│   → SQL 直接查询物化表                             │
│   → datasource 自动解析                            │
└──────────────────────────────────────────────────┘
```

---

## 9. 可扩展性设计

添加新 BI 平台只需：

1. **创建适配器包**（如 `datus-bi-metabase`），实现 `BIAdapterBase` 及相关 Mixin
2. **注册 Entry Point**：
   ```toml
   [project.entry-points."datus_bi_core.adapters"]
   metabase = "datus_bi_metabase:MetabaseAdapter"
   ```
3. **添加 Skill 文件**：`skills/metabase-dashboard/SKILL.md`，定义该平台的工作流规则
4. **无需修改 Agent 层代码**：`adapter_registry` 自动发现，`BIFuncTool` 按 Mixin 动态暴露工具
