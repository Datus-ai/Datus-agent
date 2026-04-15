# BI Tool / Adapter 接口设计

本文总结 `bi_adapter` 分支里 BI 能力相关的接口设计，覆盖两个仓库：

- `agent-1`：Agent 侧编排、LLM tool 暴露、CLI bootstrap、配置接入
- `../datus-bi-adapters`：平台抽象、数据模型、注册表、Superset/Grafana 适配器实现

这条分支的核心目标不是“再加一个 BI 功能模块”，而是把职责真正拆清楚：

- `agent-1` 只负责 agent 编排、参数校验、tool 暴露、工作流组织
- `datus-bi-adapters` 负责 BI 平台 API 适配和平台差异封装
- 两边通过稳定的 `datus_bi_core` 接口连接

## 1. 这条分支做了什么

### agent-1 侧

1. 新增 `BIFuncTool`，作为 LLM function calling 层，把平台能力转换成统一的 tool 接口。
2. `GenDashboardAgenticNode` 支持从 `agent.yml` 读取 `bi_platform` 和 `dashboard_config`，按配置动态挂载 BI tools；ChatAgenticNode 通过 `task(type="gen_dashboard")` 委托给该节点。
3. 原先放在 `agent-1` 内部的 Superset adapter / registry / util 基本被移除，只保留 agent 自己的逻辑。
4. `datus/tools/bi_tools/__init__.py` 退化为对 `datus_bi_core` 的兼容 re-export，避免老引用立即失效。
5. `pyproject.toml` 改成：
   - `datus-bi-core` 为主依赖
   - `datus-bi-superset` / `datus-bi-grafana` 作为 `bi` optional extras
6. `bi_dashboard.py` 和 `dashboard_assembler.py` 保留在 `agent-1`，因为它们属于 agent bootstrap / 提取编排，不属于任何单一 BI 平台。

### ../datus-bi-adapters 侧

1. 新建独立 uv workspace，拆成三个包：
   - `datus-bi-core`
   - `datus-bi-superset`
   - `datus-bi-grafana`
2. 定义统一抽象：
   - `BIAdapterBase`
   - `DashboardWriteMixin`
   - `ChartWriteMixin`
   - `DatasetWriteMixin`
3. 定义统一模型：
   - 读模型：`DashboardInfo` / `ChartInfo` / `DatasetInfo` / `QuerySpec`
   - 写模型：`DashboardSpec` / `ChartSpec` / `DatasetSpec`
4. 通过 entry points + `adapter_registry` 做适配器自动发现。
5. Superset / Grafana 的平台差异被下沉到各自 adapter 内部，不再散落在 `agent-1` 里。

## 2. 分层和边界

整体分层如下：

```text
ChatAgenticNode
    |
    | task(type="gen_dashboard")
    v
GenDashboardAgenticNode
    |
    v
BIFuncTool
    |
    v
datus_bi_core
    |- BIAdapterBase
    |- Write Mixins
    |- Models
    |- Registry
    |
    +--> datus_bi_superset.SupersetAdapter
    +--> datus_bi_grafana.GrafanaAdapter
```

另有一条“读取 BI 资产并生成子代理上下文”的旁路：

```text
bi_dashboard.py / DashboardAssembler
    |
    v
BIAdapterBase read-side methods
```

这条旁路只依赖 read-side 接口，不依赖写接口。这样 dashboard bootstrap 和 LLM dashboard creation 两条链路共用同一套 adapter 抽象，但互不耦合。

## 3. 仓库职责划分

| 仓库 | 保留内容 | 不再承担的内容 |
|---|---|---|
| `agent-1` | tool 暴露、配置解析、LLM 参数校验、工作流编排、dashboard bootstrap | 平台 API 细节、认证细节、平台对象序列化 |
| `datus-bi-adapters` | BI 平台抽象、数据模型、注册表、Superset/Grafana API 封装 | agent node 初始化、LLM tool 命名、跨工具编排 |

这个边界很重要，因为它决定了之后新增平台时，绝大多数代码应该落在 `datus-bi-adapters`，而不是继续把平台特化逻辑写回 `agent-1`。

## 4. agent-1 中的接口设计

### 4.1 GenDashboardAgenticNode 如何挂 BI 能力

`datus/agent/node/gen_dashboard_agentic_node.py` 的 `_setup_bi_tools()` 负责：

1. 从节点配置读取 `bi_platform`
2. 从 `dashboard_config` 读取该平台的认证和 `dataset_db` 配置
3. 调用 `datus_bi_core.adapter_registry.get(bi_platform)` 获取 adapter class
4. 实例化 adapter
5. 包装成 `BIFuncTool`
6. 把 `BIFuncTool.available_tools()` 注册进当前 node

这意味着：

- 是否启用 BI 能力由配置驱动，不是硬编码
- agent 侧不关心平台实现类细节，只关心注册表和统一构造参数

### 4.2 BIFuncTool 的角色

`datus/tools/func_tool/bi_tools.py` 是整个分支里最关键的 agent-side 适配层。它不直接实现平台 API，而是做四件事：

1. 把 adapter 能力转成 LLM 可调用的函数名
2. 把字符串参数转成 `ChartSpec` / `DashboardSpec` / `DatasetSpec`
3. 做 agent 侧输入校验和 fail-fast
4. 处理跨平台但不属于 adapter 的公共逻辑

典型例子：

- `metrics="revenue,count"` 在这里解析成 `List[str]`
- `dataset_id="12"` 在这里转成数值 ID
- `write_query()` 在这里完成，因为它本质上是“源数据库读 + BI 数据库写”的跨边界流程，不属于任何单一 BI 平台

### 4.3 动态 tool 暴露

`BIFuncTool.available_tools()` 按 adapter 是否实现 mixin 来决定暴露哪些工具：

| 条件 | 暴露工具 |
|---|---|
| 所有 adapter | `list_dashboards` `get_dashboard` `list_charts` `list_datasets` |
| `DashboardWriteMixin` | `create_dashboard` `update_dashboard` `delete_dashboard` |
| `ChartWriteMixin` | `create_chart` `update_chart` `add_chart_to_dashboard` `delete_chart` |
| `DatasetWriteMixin` | `create_dataset` `list_bi_databases` `delete_dataset` |
| 配置了 `dataset_db_uri` | `write_query` |

这个设计的好处是：

- capability 由类型系统表达，而不是靠 if/else 名字判断
- Grafana 不支持 dataset write 时，tool 层天然不会暴露 `create_dataset`
- 新平台只要实现对应 mixin，就自动进入 tool 集合

### 4.4 GenDashboardAgenticNode 的读连接初始化

`GenDashboardAgenticNode` 在 init 阶段完成 `_read_connector` 的设置，而不再依赖 `ChatAgenticNode._update_database_connection()` 动态回填。

- `_read_connector` 在节点初始化时从 agent 配置中确定，绑定到节点生命周期
- `write_query()` 从该初始 connector 读取源数据，写入 BI 平台自己的 `dataset_db`
- BI 工具从 ChatAgenticNode 提取出来后，不再随聊天会话切库而动态变化

这简化了连接所有权模型：GenDashboardAgenticNode 自身持有并管理读连接，而不是跟随父节点的状态变化。

### 4.5 为什么 `dashboard_assembler.py` 留在 agent-1

`DashboardAssembler` 依赖的是：

- `parse_dashboard_id`
- `get_dashboard_info`
- `list_charts`
- `get_chart`
- `list_datasets`
- `get_dataset`

它的职责是把平台资产整理成 reference SQL、metric SQL、table 列表，供 sub-agent bootstrap 使用。这个过程是 agent 逻辑，不是平台 API 逻辑，所以应该留在 `agent-1`。

换句话说：

- adapter 提供“平台对象的标准化读取”
- assembler 负责“把这些对象变成 agent 可消费的知识材料”

## 5. datus-bi-core 的接口设计

### 5.1 `BIAdapterBase`

`../datus-bi-adapters/datus-bi-core/datus_bi_core/base.py`

`BIAdapterBase` 定义所有平台都必须提供的 read/discovery 接口：

- `platform_name()`
- `auth_type()`
- `parse_dashboard_id(dashboard_url)`
- `get_dashboard_info(dashboard_id)`
- `list_charts(dashboard_id)`
- `get_chart(chart_id, dashboard_id=None)`
- `list_datasets(dashboard_id)`
- `get_dataset(dataset_id, dashboard_id=None)`
- `list_dashboards(search="", page_size=20)`

这里有两个设计点：

1. `list_dashboards` 现在放回 `BIAdapterBase`，不再单独维护 `ListDashboardsMixin`
2. `get_chart` / `get_dataset` 虽然不一定暴露给 LLM，但仍然是 adapter 的核心接口，因为 bootstrap 和 update 前校验都依赖它们

### 5.2 Write Mixins

`../datus-bi-adapters/datus-bi-core/datus_bi_core/mixins.py`

写接口拆成三个 mixin，而不是一个大而全的“可写 adapter”接口：

- `DashboardWriteMixin`
- `ChartWriteMixin`
- `DatasetWriteMixin`

这样做的原因是平台能力不对称：

- Superset 有 dataset 概念
- Grafana 没有独立 dataset 资源，只是 datasource + panel

如果强行把所有写接口塞进一个基类，就会让不支持的平台暴露伪接口，最后 tool 层还得额外绕开。

### 5.3 模型设计

`../datus-bi-adapters/datus-bi-core/datus_bi_core/models.py`

#### 读模型

| 模型 | 作用 |
|---|---|
| `DashboardInfo` | dashboard 标准返回 |
| `ChartInfo` | chart/panel 标准返回 |
| `DatasetInfo` | dataset/datasource 标准返回 |
| `QuerySpec` | chart 背后的 SQL/语义查询描述 |

#### 写模型

| 模型 | 作用 |
|---|---|
| `DashboardSpec` | dashboard 创建/更新请求 |
| `ChartSpec` | chart 创建/更新请求 |
| `DatasetSpec` | dataset 创建请求 |

有几个关键约束：

- 公共字段尽量平台无关，例如 `title`、`description`、`chart_type`
- 平台专有扩展放进 `extra`
- `ChartInfo.query` 允许既表达 SQL，也表达 semantic 查询

这让 adapter 可以既服务 dashboard creation，也服务 dashboard bootstrap。

### 5.4 Registry 设计

`../datus-bi-adapters/datus-bi-core/datus_bi_core/registry.py`

注册表负责三件事：

1. 统一注册 adapter class
2. 通过 Python entry points 自动发现安装的 adapter 包
3. 提供 metadata：`platform`、`display_name`、`auth_type`、`capabilities`

这个设计让 `agent-1` 不需要写死：

- `if platform == "superset": ...`
- `if platform == "grafana": ...`

只要包安装并注册，agent 就能发现它。

## 6. 两个平台 adapter 的设计差异

### 6.1 SupersetAdapter

`../datus-bi-adapters/datus-bi-superset/datus_bi_superset/adapter.py`

Superset 适配器实现了：

- `BIAdapterBase`
- `DashboardWriteMixin`
- `ChartWriteMixin`
- `DatasetWriteMixin`

它的特点是：

- 平台本身有明确的 dataset 概念
- chart 与 dataset 强绑定
- 能从 chart / dataset 中提取 SQL、metric、dimension、table 信息
- 支持 read + write 的完整闭环

因此 Superset 的主流程是：

```text
write_query
  -> create_dataset
  -> create_chart
  -> create_dashboard
  -> add_chart_to_dashboard
```

### 6.2 GrafanaAdapter

`../datus-bi-adapters/datus-bi-grafana/datus_bi_grafana/adapter.py`

Grafana 适配器实现了：

- `BIAdapterBase`
- `DashboardWriteMixin`
- `ChartWriteMixin`

没有实现 `DatasetWriteMixin`，因为：

- Grafana 的“数据源”不是 Superset 式 dataset
- panel 直接挂 datasource，并把 SQL 放在 `targets[].rawSql`
- panel 生命周期本身依赖 dashboard

因此 Grafana 的主流程是：

```text
write_query
  -> create_dashboard
  -> create_chart(sql=..., dashboard_id=...)
```

当前接口语义下，Grafana 有几个天然限制：

- `create_chart` 必须带 `dashboard_id`
- `get_chart` 读取时需要 `dashboard_id`
- `update_chart` / `delete_chart` 受 panel-in-dashboard 模型限制，语义上不如 Superset 自然

这不是 tool 设计问题，而是平台对象模型本身不同。

## 7. `write_query` 和 datasource / database 的所有权

`write_query()` 是这条分支里很重要的一条共享能力。

它的语义不是“在 BI 平台里执行查询”，而是：

1. 用当前 source DB connector 执行 `SELECT` / `WITH`
2. 把结果集 materialize 到 BI 平台自己的 `dataset_db`
3. 返回写入表名、行数，以及可解析出的 `database_id`

这让 agent 能把结果数据交给 BI 平台托管，而不是让 BI 平台回连 agent 当前的任意源数据库。

这里有两个所有权约束：

### 7.1 `dataset_db`

- 由 `agent.yml` 配置
- 是 BI 平台最终读取的数据存储
- `BIFuncTool` 只负责写入和查找，不负责在 BI 平台里“注册连接”

### 7.2 Grafana datasource

- `BIFuncTool._resolve_grafana_datasource_uid()` 只查找 datasource，不创建 datasource
- 优先按 `datasource_name` 匹配
- fallback 按 `dataset_db_uri` 里的 database name 匹配

也就是说，datasource 的所有权在 Grafana 自己，不在 agent。

## 8. 安全和容错设计

这条分支没有把 BI tool 当成“纯透传”，而是在 tool 层做了明确约束：

- `write_query` 只接受 `SELECT` / `WITH`
- 禁止多语句 SQL
- 表名必须匹配正则
- `if_exists` 只允许 `replace` / `append` / `fail`
- `update_dashboard` / `update_chart` 先读现有对象，再做 partial update
- 大多数 adapter / tool 方法都做异常捕获并返回结构化错误

这些校验放在 `BIFuncTool` 而不是 adapter，原因是：

- 这些约束是 agent 对 LLM 输入的防线
- 它们属于“对 tool contract 的保护”，不是平台 API 的本地规则

## 9. 这条分支在两个仓库里的关键文件

### agent-1

- `datus/agent/node/gen_dashboard_agentic_node.py`
- `datus/tools/func_tool/bi_tools.py`
- `datus/tools/bi_tools/__init__.py`
- `datus/tools/bi_tools/dashboard_assembler.py`
- `datus/cli/bi_dashboard.py`
- `pyproject.toml`

### ../datus-bi-adapters

- `datus-bi-core/datus_bi_core/base.py`
- `datus-bi-core/datus_bi_core/mixins.py`
- `datus-bi-core/datus_bi_core/models.py`
- `datus-bi-core/datus_bi_core/registry.py`
- `datus-bi-superset/datus_bi_superset/__init__.py`
- `datus-bi-superset/datus_bi_superset/adapter.py`
- `datus-bi-grafana/datus_bi_grafana/__init__.py`
- `datus-bi-grafana/datus_bi_grafana/adapter.py`
- 各包 `pyproject.toml`

## 10. 当前设计的结论

这个分支最终形成的是一种比较干净的“双层契约”：

### 第一层：agent-side contract

由 `BIFuncTool` 提供，面向 LLM / GenDashboardAgenticNode：

- tool 名称稳定
- 参数格式稳定
- 安全校验在这一层完成
- 能力按 mixin 动态暴露

### 第二层：platform-side contract

由 `datus_bi_core` 提供，面向各平台 adapter：

- 读接口统一
- 写接口按能力拆分
- 返回模型统一
- 平台特化通过 `extra` 承载

这使得后续新增平台时，路径比较明确：

1. 在 `../datus-bi-adapters` 实现新 adapter
2. 注册 entry point
3. 在 `agent.yml` 提供平台配置
4. `agent-1` 基本不用再改 tool 层主结构

## 11. 当前已知 caveat

1. Registry metadata 里 Grafana 注册为 `AuthType.API_KEY`，但 `GrafanaAdapter` 实现本身也支持 username/password。CLI 如何暴露这两种认证方式，后续可以再统一。
2. Grafana 的 panel 不是独立 chart 资源，所以 `update_chart` / `delete_chart` 语义天然偏弱。
3. `dataset_db` / datasource 目前都假设“BI 平台里已经存在或可被查到”，tool 层只做 lookup，不做平台侧资源声明式管理。

如果继续演进，这几个点会是下一轮接口收敛的重点。
