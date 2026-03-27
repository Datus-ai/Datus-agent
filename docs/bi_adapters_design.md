# BI Adapters 设计文档

## 背景与目标

**现状：** `datus/tools/bi_tools/` 已有针对 Superset 的只读适配层，仅用于 `.bootstrap-bi` 命令提取 dashboard 信息来引导 sub-agent。

**目标：**
1. 参考 `datus-db-adapters` 的插件化多包模式，将 BI 工具抽成独立包 `datus-bi-adapters`
2. 扩展 Superset 适配器，支持 dashboard/chart/dataset 的增删改操作
3. 新增 Grafana 适配器（读+写）
4. 在 agent-1 中新增 `BIFuncTool`，将 BI 操作暴露为 LLM function tools
5. 支持通过 datus CLI 自然语言自动创建和修改 dashboard

---

## 总体架构

```
自然语言 (datus CLI REPL)
    │
    ▼ /bi_agent <用户需求>
bi_agent (agentic node, tools: bi_tools + db_tools.read_query)     [agent-1]
    │
    ▼
BIFuncTool  (LLM function calling 层)                              [datus/tools/func_tool/]
    │
    ▼
datus_bi_core.BIAdaptorBase + Mixins                               [datus-bi-core 包]
    │
    ├── SupersetAdaptor (读+写)                                     [agent-1 内置，datus/tools/bi_tools/superset/]
    └── datus_bi_grafana.GrafanaAdaptor (读+写)                     [datus-bi-grafana 包]
```

---

## 包结构

### 独立包：`datus-bi-core` + `datus-bi-grafana`

参照 `datus-db-adapters/` 的模式，核心抽象和外部适配器独立发包。Superset 与 agent-1 的 CLI（`.bootstrap-bi`）深度耦合，作为**内置适配器**保留在 agent-1 内（类似 SQLite/DuckDB 内置在 agent-1 的 `db_tools/` 中）。

```
datus-bi-adapters/
├── pyproject.toml                        # uv workspace
├── README.md
├── datus-bi-core/                        # 核心抽象层（独立包）
│   ├── pyproject.toml
│   └── datus_bi_core/
│       ├── __init__.py                   # 公开所有核心 API
│       ├── base.py                       # BIAdaptorBase(ABC) + AuthType/AuthParam
│       ├── mixins.py                     # 能力 Mixin（写操作按需组合）
│       ├── models.py                     # 所有数据模型（读+写两类）+ BICapability 枚举
│       ├── registry.py                   # BIAdaptorRegistry（entry_points + 懒加载）
│       └── exceptions.py                 # 结构化异常层级
│
└── datus-bi-grafana/                     # Grafana 适配器（独立包）
    ├── pyproject.toml                    # entry-point: grafana = "datus_bi_grafana:register"
    └── datus_bi_grafana/
        ├── __init__.py                   # register()
        └── adaptor.py                    # GrafanaAdaptor
```

### agent-1：Superset 内置 + import 路径更新

Superset 适配器保留在 agent-1 内部（与 CLI `.bootstrap-bi`、`dashboard_assembler` 深度耦合），`base_adaptor.py` 和 `registry.py` 的实现迁移到 `datus_bi_core`，直接更新 import 路径：

```
# 旧 import（删除）
from datus.tools.bi_tools.base_adaptor import BIAdaptorBase
from datus.tools.bi_tools.registry import adaptor_registry

# 新 import（替换为）
from datus_bi_core import BIAdaptorBase
from datus_bi_core.registry import adaptor_registry
```

迁移后 agent-1 目录结构：

```
datus/tools/bi_tools/
├── __init__.py                  # import 指向 datus_bi_core
├── dashboard_assembler.py       # 保留（CLI 专用逻辑）
└── superset/                    # 保留在 agent-1（内置适配器）
    ├── __init__.py              # register()，启动时注册到 datus_bi_core.registry
    ├── adaptor.py               # SupersetAdaptor（读+写，import from datus_bi_core）
    ├── util.py                  # QueryContext builder（保留）
    └── layout.py                # position_data 布局管理（新增）
    # base_adaptor.py — 已删除，迁移到 datus_bi_core
    # registry.py — 已删除，迁移到 datus_bi_core

datus/tools/func_tool/
└── bi_func_tools.py             # 新增，BIFuncTool（LLM function calling 层）

pyproject.toml                   # 新增 datus-bi-core, datus-bi-grafana 依赖
conf/agent.yml                   # 新增 grafana 配置段 + bi_agent 示例节点
```

---

## datus-bi-core：核心数据模型 + 接口

### 能力枚举（`models.py`）

```python
class BICapability(str, Enum):
    LIST_DASHBOARDS = "list_dashboards"
    DASHBOARD_WRITE = "dashboard_write"
    CHART_WRITE     = "chart_write"
    DATASET_WRITE   = "dataset_write"
    PUBLISH         = "publish"
```

所有 `capabilities={...}` 传参均使用 `BICapability` 枚举值，避免魔法字符串散落各处。

### 数据模型（`models.py`）

**现有模型（从 `base_adaptor.py` 迁移）：**
- `AuthParam`, `AuthType`, `ColumnInfo`, `MetricDef`, `DimensionDef`
- `DatasetInfo`, `QuerySpec`, `ChartInfo`

**`DashboardInfo` 新增 `version` 字段：**

```python
class DashboardInfo(BaseModel):
    id: Union[int, str]
    title: str
    url: str = ""
    chart_ids: List[Union[int, str]] = []
    version: Optional[int] = None  # Grafana dashboard 乐观锁版本号
```

**新增写操作输入模型：**

```python
class ChartSpec(BaseModel):
    """LLM 构造 chart 创建/修改请求的统一 spec"""
    chart_type: Literal["bar", "line", "pie", "table", "big_number", "scatter"]
    title: str
    description: str = ""
    dataset_id: Optional[int] = None   # 使用已有 dataset
    sql: Optional[str] = None          # 或直接提供 SQL
    x_axis: Optional[str] = None       # 时间轴/类别轴字段名
    metrics: Optional[List[str]] = None
    dimensions: Optional[List[str]] = None
    filters: Optional[List[Dict[str, Any]]] = None
    extra: Dict[str, Any] = {}

    @model_validator(mode="after")
    def check_data_source(self) -> "ChartSpec":
        if self.dataset_id is None and not self.sql:
            raise ValueError("must provide either dataset_id or sql")
        return self

class DatasetSpec(BaseModel):
    name: str
    sql: str
    database_id: int           # BI 平台内的数据库连接 ID
    schema: str = ""
    description: str = ""

class DashboardSpec(BaseModel):
    title: str
    description: str = ""
    extra: Dict[str, Any] = {}
```

**认证类型扩展（现有 `LOGIN` + 新增 `API_KEY`）：**

```python
class AuthType(Enum):
    LOGIN   = "login"    # Superset（用户名密码）
    API_KEY = "api_key"  # Grafana（Service Account Token）
```

### 能力 Mixin（`mixins.py`）

参考 `datus-db-adapters` 的 `CatalogSupportMixin` 模式，按需组合：

```python
class ListDashboardsMixin(ABC):
    @abstractmethod
    def list_dashboards(self, search: str = "", page_size: int = 20) -> List[DashboardInfo]: ...

class DashboardWriteMixin(ABC):
    @abstractmethod
    def create_dashboard(self, spec: DashboardSpec) -> DashboardInfo: ...
    @abstractmethod
    def update_dashboard(self, dashboard_id: Union[int, str], spec: DashboardSpec) -> DashboardInfo: ...
    @abstractmethod
    def delete_dashboard(self, dashboard_id: Union[int, str]) -> bool: ...
    @abstractmethod
    def publish_dashboard(self, dashboard_id: Union[int, str]) -> DashboardInfo:
        """发布 dashboard（草稿 → 可见）。Superset: PUT /api/v1/dashboard/{id}/publish"""
        ...

class ChartWriteMixin(ABC):
    @abstractmethod
    def create_chart(self, spec: ChartSpec, dashboard_id=None) -> ChartInfo:
        # dashboard_id：Grafana 必填（panel 嵌入 dashboard），Superset 可选
        ...
    @abstractmethod
    def update_chart(self, chart_id: Union[int, str], spec: ChartSpec) -> ChartInfo: ...
    @abstractmethod
    def delete_chart(self, chart_id: Union[int, str]) -> bool: ...
    @abstractmethod
    def add_chart_to_dashboard(self, dashboard_id: Union[int, str], chart_id: Union[int, str]) -> bool: ...

class DatasetWriteMixin(ABC):
    @abstractmethod
    def create_dataset(self, spec: DatasetSpec) -> DatasetInfo: ...
    @abstractmethod
    def update_dataset(self, dataset_id: Union[int, str], spec: DatasetSpec) -> DatasetInfo: ...
    @abstractmethod
    def list_bi_databases(self) -> List[Dict[str, Any]]: ...
```

### 抽象基类（`base.py`）

```python
class BIAdaptorBase(ABC):
    # 读操作（所有适配器必须实现）
    @abstractmethod def platform_name(self) -> str: ...
    @abstractmethod def auth_type(self) -> AuthType: ...
    @abstractmethod def parse_dashboard_id(self, url: str) -> Union[int, str]: ...
    @abstractmethod def get_dashboard_info(self, id) -> Optional[DashboardInfo]: ...
    @abstractmethod def list_charts(self, dashboard_id) -> List[ChartInfo]: ...
    @abstractmethod def get_chart(self, chart_id, dashboard_id) -> Optional[ChartInfo]: ...
    @abstractmethod def list_datasets(self, dashboard_id: Optional[str] = None) -> List[DatasetInfo]: ...
    @abstractmethod def get_dataset(self, dataset_id, dashboard_id) -> Optional[DatasetInfo]: ...

    # 注册时声明 capabilities（参考 db-adapters）
    @classmethod
    def register(cls, platform: str, auth_type: AuthType,
                 display_name: str = "",
                 capabilities: Set[BICapability] = None): ...
```

注意：`list_datasets` 的 `dashboard_id` 参数类型为 `Optional[str] = None`，而非空字符串默认值，便于适配器区分"未传入"与"传入空值"两种语义。

### 注册机制（`registry.py`）

迁移自 agent-1 的 `registry.py`，放入 `datus_bi_core/registry.py`，新增：
- `_capabilities: Dict[str, Set[BICapability]]` — 记录每个 platform 的能力集合
- `get_capabilities(platform) -> Set[BICapability]` — 供 `BIFuncTool.available_tools()` 查询
- `discover_adaptors()` → `entry_points(group="datus.bi_adaptors")` — 插件扩展点（与 `datus-db-adapters` 的 `datus.adapters` 对称）
- `_try_load_adaptor(platform)` → `importlib.import_module(f"datus_bi_{platform}")` — 懒加载未注册的适配器

### 结构化异常层级（`exceptions.py`）

```python
class DatusBiException(Exception): ...
class BIAuthError(DatusBiException): ...        # 401/403
class BIPermissionError(DatusBiException): ...  # 权限不足
class BINotFoundError(DatusBiException): ...    # 404
class BIConflictError(DatusBiException): ...    # 版本冲突（Grafana 乐观锁）
class BIValidationError(DatusBiException): ...  # 参数非法
class BIPlatformError(DatusBiException): ...    # 平台不可用
```

区分可重试错误（网络类 `BIPlatformError`）与不可重试错误（参数类 `BIValidationError`、权限类 `BIAuthError`），便于 agent 制定差异化的错误处理策略。

---

## Superset 适配器

### 类声明

```python
class SupersetAdaptor(BIAdaptorBase, ListDashboardsMixin,
                      DashboardWriteMixin, ChartWriteMixin, DatasetWriteMixin):
```

### 文件列表

| 文件 | 说明 |
|------|------|
| `datus/tools/bi_tools/superset/adaptor.py` | `SupersetAdaptor`（保留在 agent-1，扩展写操作，import from `datus_bi_core`） |
| `datus/tools/bi_tools/superset/util.py` | `QueryContext` builder（保留） |
| `datus/tools/bi_tools/superset/layout.py` | Superset dashboard 布局（`position_data`）管理模块：`add_panel(position_data, chart_id) -> dict`、`remove_panel(position_data, chart_id) -> dict`。需要专项单元测试（空 dashboard、已有行、含 tabs）。 |

### 写操作（Superset REST API）

| 方法 | API 端点 | HTTP 方法 |
|------|----------|-----------|
| `list_dashboards()` | `/api/v1/dashboard?q=...` | GET |
| `create_dashboard()` | `/api/v1/dashboard` | POST |
| `update_dashboard()` | `/api/v1/dashboard/{id}` | PUT |
| `delete_dashboard()` | `/api/v1/dashboard/{id}` | DELETE |
| `publish_dashboard()` | `/api/v1/dashboard/{id}/publish` | PUT |
| `create_chart()` | `/api/v1/chart` | POST |
| `update_chart()` | `/api/v1/chart/{id}` | PUT |
| `delete_chart()` | `/api/v1/chart/{id}` | DELETE |
| `add_chart_to_dashboard()` | `/api/v1/dashboard/{id}`（更新 position_data） | PUT |
| `create_dataset()` | `/api/v1/dataset` | POST |
| `update_dataset()` | `/api/v1/dataset/{id}` | PUT |
| `list_bi_databases()` | `/api/v1/database/` | GET |

### create_chart form_data 构建

基于现有 `util.py` 的 `ChartBuildQueryRegistry`，从 `ChartSpec` 反向生成 Superset `form_data`。优先支持 6 种最常用图表类型：

| ChartSpec.chart_type | Superset viz_type |
|---------------------|-------------------|
| `bar` | `echarts_timeseries_bar` |
| `line` | `echarts_timeseries_line` |
| `pie` | `pie` |
| `table` | `table` |
| `big_number` | `big_number_total` |
| `scatter` | `echarts_timeseries_scatter` |

```python
def _build_form_data(self, spec: ChartSpec) -> Dict:
    return {
        "viz_type": _CHART_TYPE_MAP[spec.chart_type],
        "datasource": f"{spec.dataset_id}__table",
        "metrics": spec.metrics or [],
        "groupby": spec.dimensions or [],
        "time_column": spec.x_axis,
        **spec.extra,
    }
```

### 注册

```python
# datus/tools/bi_tools/superset/__init__.py（agent-1 内置）
def register():
    from datus_bi_core import adaptor_registry
    from datus_bi_core.models import BICapability
    adaptor_registry.register(
        "superset", SupersetAdaptor,
        auth_type=AuthType.LOGIN,
        display_name="Apache Superset",
        capabilities={
            BICapability.LIST_DASHBOARDS,
            BICapability.DASHBOARD_WRITE,
            BICapability.CHART_WRITE,
            BICapability.DATASET_WRITE,
            BICapability.PUBLISH,
        },
    )
```

---

## Grafana 适配器

### 模型差异说明

| 概念 | Superset | Grafana |
|------|----------|---------|
| Chart | 独立 Chart 对象（有独立 API） | Panel（嵌入 Dashboard JSON，无独立 API） |
| Dataset | Virtual Dataset 对象 | DataSource（全局连接级，非 SQL 虚拟表） |
| Dashboard | Dashboard | Dashboard（带 `version` 乐观锁字段） |

**设计决策：接受差异，显式声明**
- `create_chart()` 对 Grafana 要求传入 `dashboard_id`，在函数内部完成 panel 追加 + dashboard 更新
- `add_chart_to_dashboard()` 对 Grafana：验证 panel 是否已在目标 dashboard 中；若 `chart_id` 属于另一 dashboard，则从源 dashboard 提取 panel 并追加到目标 dashboard。**不是 no-op**，始终返回有意义的状态。
- `GrafanaAdaptor` 不实现 `DatasetWriteMixin`（DataSource 是全局连接配置，不应由 agent 管理；Grafana chart 用 datasource UID + 查询 DSL）

### 类声明

```python
class GrafanaAdaptor(BIAdaptorBase, ListDashboardsMixin,
                     DashboardWriteMixin, ChartWriteMixin):
    # 不实现 DatasetWriteMixin（DataSource 是全局配置）
```

### 认证

```
Authorization: Bearer {api_key}    # Service Account Token 或 Legacy API Key
```

### Grafana HTTP API 映射

| 方法 | Grafana API |
|------|------------|
| `get_dashboard_info()` | `GET /api/dashboards/uid/{uid}` |
| `list_charts()` | 解析 `GET /api/dashboards/uid/{uid}` 响应中的 `.dashboard.panels[]` |
| `list_datasets()` | `GET /api/datasources` |
| `get_dataset()` | `GET /api/datasources/{id}` |
| `list_dashboards()` | `GET /api/search?type=dash-db&query={search}` |
| `create_dashboard()` | `POST /api/dashboards/db` |
| `update_dashboard()` | `POST /api/dashboards/db`（含 `id` + `overwrite: true`，需携带 `version` 实现乐观锁） |
| `delete_dashboard()` | `DELETE /api/dashboards/uid/{uid}` |
| `publish_dashboard()` | 更新 dashboard JSON 中 `meta.provisioned = false` 后 `POST /api/dashboards/db` |
| `create_chart()` | GET dashboard → 追加 panel → `POST /api/dashboards/db` |
| `update_chart()` | GET dashboard → 定位 panel by id → 修改 → `POST /api/dashboards/db` |
| `delete_chart()` | GET dashboard → 过滤掉 panel → `POST /api/dashboards/db` |
| `add_chart_to_dashboard()` | 验证或跨 dashboard 复制 panel → `POST /api/dashboards/db` |

### create_chart 实现要点

```python
def create_chart(self, spec: ChartSpec, dashboard_id=None) -> ChartInfo:
    if not dashboard_id:
        raise BIValidationError("Grafana requires dashboard_id to create a panel")
    # 1. GET dashboard JSON，获取当前 version
    # 2. 构建 panel dict（Grafana panel 格式）
    # 3. 追加到 dashboard["panels"]，分配新 panel_id
    # 4. POST /api/dashboards/db (overwrite: true, 携带 version 乐观锁)
    # 5. 返回 ChartInfo(id=panel_id, name=spec.title, ...)
```

**Grafana panel 类型映射：**

| ChartSpec.chart_type | Grafana panel type |
|---------------------|-------------------|
| `bar` | `barchart` |
| `line` | `timeseries` |
| `pie` | `piechart` |
| `table` | `table` |
| `big_number` | `stat` |

### 注册

```python
# datus_bi_grafana/__init__.py
def register():
    from datus_bi_core import adaptor_registry
    from datus_bi_core.models import BICapability
    adaptor_registry.register(
        "grafana", GrafanaAdaptor,
        auth_type=AuthType.API_KEY,
        display_name="Grafana",
        capabilities={
            BICapability.LIST_DASHBOARDS,
            BICapability.DASHBOARD_WRITE,
            BICapability.CHART_WRITE,
            BICapability.PUBLISH,
        },
    )
```

---

## BIFuncTool（agent-1 新增）

**文件：** `datus/tools/func_tool/bi_func_tools.py`

参照 `datus/tools/func_tool/database.py`（DBFuncTool）的设计模式：

```python
class BIFuncTool:
    def __init__(self, adaptor: BIAdaptorBase): ...

    # ── 读操作（所有适配器）────────────────────────────────────
    def list_dashboards(self, search: str = "") -> FuncToolResult:
        """列出 BI 平台上的 dashboard 列表，支持关键词搜索"""

    def get_dashboard(self, dashboard_id: str) -> FuncToolResult:
        """获取指定 dashboard 的详细信息，包含 chart_ids 列表"""

    def list_charts(self, dashboard_id: str) -> FuncToolResult:
        """列出指定 dashboard 下的所有 chart/panel"""

    def list_datasets(self, dashboard_id: Optional[str] = None) -> FuncToolResult:
        """列出可用的 dataset/datasource"""

    # ── 写操作：dashboard（DashboardWriteMixin）───────────────
    def create_dashboard(self, title: str, description: str = "") -> FuncToolResult:
        """创建新的空 dashboard"""

    def update_dashboard(self, dashboard_id: str,
                         title: str = "", description: str = "") -> FuncToolResult:
        """更新 dashboard 标题或描述"""

    def publish_dashboard(self, dashboard_id: str) -> FuncToolResult:
        """发布 dashboard（草稿 → 可见）"""

    # ── 写操作：chart（ChartWriteMixin）──────────────────────
    def create_chart(
        self,
        chart_type: Literal["bar", "line", "pie", "table", "big_number", "scatter"],
        title: str,
        sql: str = "",
        dataset_id: str = "",
        x_axis: str = "",
        metrics: List[str] = [],   # 指标字段列表
        dashboard_id: str = "",    # Grafana 必填，Superset 可选
        description: str = ""
    ) -> FuncToolResult:
        """创建图表。chart_type 支持：bar/line/pie/table/big_number/scatter"""

    def update_chart(self, chart_id: str, title: str = "",
                     chart_type: Literal["bar", "line", "pie", "table", "big_number", "scatter"] = "",
                     sql: str = "", metrics: List[str] = [], x_axis: str = "") -> FuncToolResult:
        """修改已有图表的类型、标题或数据配置"""

    def add_chart_to_dashboard(self, chart_id: str, dashboard_id: str) -> FuncToolResult:
        """将已有 chart 添加到指定 dashboard（Grafana：支持跨 dashboard 复制 panel）"""

    def create_full_dashboard(
        self,
        title: str,
        chart_specs: List[Dict],   # list of ChartSpec-like dicts
        database_id: str = "",
        description: str = ""
    ) -> FuncToolResult:
        """
        原子创建完整 dashboard（含 dataset、chart、布局、发布）。
        任一步骤失败时 best-effort 清理已创建资源。
        适合一次性创建场景，避免 LLM 多步调用出错。
        """

    # ── 写操作：dataset（DatasetWriteMixin）──────────────────
    def create_dataset(self, name: str, sql: str, database_id: str,
                       description: str = "") -> FuncToolResult:
        """基于 SQL 创建虚拟 dataset（Superset Virtual Dataset）"""

    def list_bi_databases(self) -> FuncToolResult:
        """列出 BI 平台内已配置的数据库连接（用于创建 dataset 时选择 database_id）"""

    # ── 工具注册（按 adaptor 能力动态返回）──────────────────
    def available_tools(self) -> List[Tool]:
        tools = [self.list_dashboards, self.get_dashboard,
                 self.list_charts, self.list_datasets]
        if isinstance(self.adaptor, DashboardWriteMixin):
            tools += [self.create_dashboard, self.update_dashboard,
                      self.publish_dashboard]
        if isinstance(self.adaptor, ChartWriteMixin):
            tools += [self.create_chart, self.update_chart,
                      self.add_chart_to_dashboard, self.create_full_dashboard]
        if isinstance(self.adaptor, DatasetWriteMixin):
            tools += [self.create_dataset, self.list_bi_databases]
        return [trans_to_function_tool(m) for m in tools]
```

---

## agentic node 集成（agent-1）

### `agentic_node.py` 工具解析

```python
# 参照 db_tools, context_search_tools 的注册方式
if "bi_tools" in tool_names:
    platform = node_config.get("bi_platform", "superset")
    dashboard_cfg = agent_config.dashboard_config.get(platform)
    adaptor_cls = adaptor_registry.get(platform)
    adaptor = adaptor_cls(
        api_base_url=dashboard_cfg.extra.get("api_url", ""),
        auth_params=AuthParam(
            username=dashboard_cfg.username,
            password=dashboard_cfg.password,
            api_key=dashboard_cfg.api_key,
            extra=dashboard_cfg.extra,
        ),
        dialect=agent_config.dialect,
    )
    bi_tool = BIFuncTool(adaptor)
    tools.extend(bi_tool.available_tools())
```

### `conf/agent.yml` 配置示例

```yaml
dashboard:
  superset:
    username: admin
    password: admin
    extra:
      provider: db
      api_url: http://localhost:8088
  grafana:
    api_key: ${GRAFANA_API_KEY}
    extra:
      api_url: http://localhost:3000

agentic_nodes:
  bi_agent:
    model: claude
    system_prompt: bi_dashboard_agent
    tools: bi_tools, db_tools.read_query
    bi_platform: superset      # 指定 BI 平台（superset 或 grafana）
    max_turns: 20
    rules:
      - "Always call list_bi_databases() or list_datasets() first before creating charts."
      - "For creating a full dashboard: use create_full_dashboard() for one-shot creation, or create_dataset → create_chart → create_dashboard → add_chart_to_dashboard for fine-grained control."
      - "Confirm with user before deleting any dashboard or chart."
      - "For Grafana, always provide dashboard_id in create_chart."
      - "Call publish_dashboard() after creation if the dashboard should be immediately visible."
```

---

## 完整文件清单

### 新建（datus-bi-adapters 仓库）

| 文件 | 说明 |
|------|------|
| `pyproject.toml` | uv workspace，members: datus-bi-core, datus-bi-grafana |
| `datus-bi-core/datus_bi_core/base.py` | `BIAdaptorBase` + `AuthType`/`AuthParam`（迁移自 agent-1 + 扩展） |
| `datus-bi-core/datus_bi_core/mixins.py` | 四个能力 Mixin（含 `publish_dashboard`） |
| `datus-bi-core/datus_bi_core/models.py` | 所有数据模型 + `BICapability` 枚举 |
| `datus-bi-core/datus_bi_core/registry.py` | `BIAdaptorRegistry`（迁移 + 扩展 capabilities 查询） |
| `datus-bi-core/datus_bi_core/exceptions.py` | 结构化异常层级（7 个类） |
| `datus-bi-grafana/datus_bi_grafana/adaptor.py` | `GrafanaAdaptor`（全新） |

### 修改（agent-1）

| 文件 | 变更 |
|------|------|
| `datus/tools/bi_tools/base_adaptor.py` | **删除**（实现迁移到 `datus_bi_core`） |
| `datus/tools/bi_tools/registry.py` | **删除**（实现迁移到 `datus_bi_core.registry`） |
| `datus/tools/bi_tools/__init__.py` | 更新 import 指向 `datus_bi_core` |
| `datus/tools/bi_tools/superset/adaptor.py` | 更新 import 指向 `datus_bi_core` + 新增写操作 |
| `datus/tools/bi_tools/superset/layout.py` | **新增** position_data 布局管理模块 |
| `datus/tools/bi_tools/dashboard_assembler.py` | 更新 import 路径指向 `datus_bi_core` |
| `datus/cli/bi_dashboard.py` | 更新 import 路径指向 `datus_bi_core` |
| `datus/tools/func_tool/bi_func_tools.py` | **新增** `BIFuncTool`（LLM function calling 层） |
| `datus/agent/node/agentic_node.py` | 新增 `bi_tools` 工具解析逻辑 |
| `pyproject.toml` | 新增 `datus-bi-core`, `datus-bi-grafana` 依赖 |
| `conf/agent.yml` | 新增 `grafana` 配置段 + `bi_agent` 示例节点 |

### 新增测试

| 文件 | 说明 |
|------|------|
| `datus-bi-core/tests/unit/test_registry.py` | `BIAdaptorRegistry` 单元测试 |
| `datus-bi-core/tests/unit/test_models.py` | `ChartSpec` 校验（含 `check_data_source` validator）、`DatasetSpec`、`BICapability` |
| `datus-bi-grafana/tests/unit/test_adaptor.py` | `GrafanaAdaptor` 单元测试（mock httpx） |
| `tests/unit_tests/tools/bi_tools/superset/test_adaptor.py` | `SupersetAdaptor` 写操作单元测试（mock httpx，agent-1 内） |
| `tests/unit_tests/tools/bi_tools/superset/test_layout.py` | `add_panel`/`remove_panel`（空 dashboard、已有行、含 tabs） |
| `tests/unit_tests/tools/func_tool/test_bi_func_tools.py` | `BIFuncTool.available_tools()` + `create_full_dashboard` 回滚逻辑 |

---

## 验证方案

### CI 级别（无外部依赖）

```bash
# datus-bi-adapters 仓库
uv run pytest datus-bi-core/tests/unit/ -v
uv run pytest datus-bi-grafana/tests/unit/ -v

# agent-1（含内置 Superset 适配器测试）
uv run pytest tests/unit_tests/tools/bi_tools/superset/ -v
uv run pytest tests/unit_tests/tools/func_tool/test_bi_func_tools.py -v
uv run pytest tests/unit_tests/ --cov=datus --cov-fail-under=80
uv run ruff format . && uv run ruff check --fix .
```

### 手动 E2E 验证

**场景 1 — 全新创建 dashboard（一次性模式）：**

```
> /bi_agent 帮我创建一个按月份统计订单数量的柱状图 dashboard

# 期望 LLM 调用序列（使用组合工具）：
# 1. list_bi_databases()
#    → 确认数据库 ID（如 database_id="1"）
# 2. create_full_dashboard(
#      title="订单分析 Dashboard",
#      chart_specs=[{
#        "chart_type": "bar",
#        "title": "月度订单量",
#        "sql": "SELECT DATE_TRUNC('month', order_date) AS month, COUNT(*) AS order_count FROM orders GROUP BY 1",
#        "x_axis": "month",
#        "metrics": ["order_count"]
#      }],
#      database_id="1"
#    )
#    → 内部依次执行：create_dataset → create_chart → create_dashboard → add_chart_to_dashboard → publish_dashboard
#    → 任一步失败时 best-effort 清理已创建资源
```

**场景 2 — 全新创建 dashboard（细粒度模式）：**

```
> /bi_agent 帮我创建一个按月份统计订单数量的柱状图 dashboard

# 期望 LLM 调用序列（细粒度控制）：
# 1. list_bi_databases()
#    → 确认数据库 ID（如 database_id="1"）
# 2. create_dataset(sql="SELECT DATE_TRUNC('month', order_date) AS month, COUNT(*) AS order_count FROM orders GROUP BY 1", name="monthly_orders", database_id="1")
#    → 返回 dataset_id
# 3. create_chart(chart_type="bar", title="月度订单量", dataset_id="{dataset_id}", x_axis="month", metrics=["order_count"])
#    → 返回 chart_id
# 4. create_dashboard(title="订单分析 Dashboard")
#    → 返回 dashboard_id
# 5. add_chart_to_dashboard(chart_id="{chart_id}", dashboard_id="{dashboard_id}")
# 6. publish_dashboard(dashboard_id="{dashboard_id}")
```

**场景 3 — 修改现有 dashboard：**

```
> /bi_agent 把订单分析 dashboard 里的柱状图改成折线图

# 期望 LLM 调用序列：
# 1. list_dashboards(search="订单分析")
# 2. list_charts(dashboard_id="...")
# 3. update_chart(chart_id="...", chart_type="line")
```

---

## 实现顺序

1. 创建 `datus-bi-adapters/` workspace + `datus-bi-core` 包（迁移 base/models/mixins/registry/exceptions）
2. agent-1：更新 Superset import 路径指向 `datus_bi_core`，新增写操作 + `layout.py`
3. 创建 `datus-bi-grafana` 包（`GrafanaAdaptor`）
4. agent-1：添加 `datus-bi-core` + `datus-bi-grafana` 依赖
5. agent-1：新增 `bi_func_tools.py`（含 `create_full_dashboard` 组合工具）
6. `agentic_node.py` 集成 `bi_tools` 工具解析
7. 全部 CI 单元测试（含 `layout.py` 专项测试）
8. `conf/agent.yml` 配置 + 代码格式检查

---

## 设计决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 包结构 | `datus-bi-core` + `datus-bi-grafana` 独立包；Superset 内置于 agent-1 | core 独立保证依赖方向正确；Superset 与 CLI 深度耦合故内置（类似 SQLite/DuckDB 内置于 db_tools）；Grafana 无历史耦合故独立 |
| Grafana panel 差异 | 接受差异，`create_chart` 对 Grafana 需要 `dashboard_id` | 假统一比显式差异更危险；调用方通过工具文档/LLM rules 知悉区别 |
| 写操作原子性 | 提供 `create_full_dashboard()` 组合工具 + 细粒度 API 并存 | LLM 多步调用有失败风险；组合工具降低风险，细粒度 API 保留灵活性 |
| Grafana DataSource | 不实现 `DatasetWriteMixin` | DataSource 是全局连接配置，不应由 agent 管理；Grafana chart 用 datasource UID + 查询 DSL |
| `add_chart_to_dashboard()` Grafana 行为 | 返回有意义状态而非 no-op | 避免调用方误判"操作成功但实际无变化"；对跨 dashboard 复制 panel 场景也有用 |
| ChartSpec 校验 | `model_validator` 强制 `dataset_id`/`sql` 二选一 | 尽早拦截无效 spec，减少 BI 平台 API 错误 |
| 错误类型 | 结构化层级（Auth/Permission/NotFound/Conflict/...） | 区分可重试错误（网络）与不可重试错误（参数非法），便于 agent 的错误处理策略 |
| `list_datasets` 签名 | `dashboard_id: Optional[str] = None` | 区分"未传入"与"传入空值"两种语义，避免适配器误判 |
| Capabilities | `BICapability` 枚举而非字符串集合 | 消除魔法字符串，IDE 可补全，重构安全 |
