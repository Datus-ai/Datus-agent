# HTML 仪表盘生成指南

## 概览

`gen_visual_dashboard` 把一个问题——*"做一份门店销售总览"*、*"看看新用户激活漏斗"*——直接变成一份自包含的 **HTML 仪表盘**：一个带筛选器的可交互网页，包含 KPI 卡片、图表、表格，筛选条件一变，SQL 立刻重跑、视图即时刷新。生成完成后会自动在浏览器中打开，由本地的 `datus --web` 后端提供查询服务。

![HTML 仪表盘示例——筛选器栏、KPI 卡片、按地区 / 时间维度联动的图表](../assets/gen_visual_dsahboard_preview.png)

HTML 仪表盘是一个**活的数据视图**——每次切换日期、地区、产品线等筛选条件，仪表盘都会把 SQL 模板按新参数渲染后真实跑库取数。这跟 `gen_visual_report` 那种把数据**一次性固化**到页面里的长篇报告不一样：报告适合发邮件、做汇报；仪表盘适合需要反复探索、自助筛选的场景。

如果你要的是接 Superset / Grafana 的 BI 平台仪表盘，请使用 [`gen_dashboard`](gen_dashboard.zh.md)。

## 快速开始

先确认本地已经通过 `--web` 启动了 datus（HTML 仪表盘的筛选器需要这个后端处理查询）：

```bash
uv run datus --datasource <你的数据源> --web
```

把当前 agent 切到 `gen_visual_dashboard`——之前的 `/gen_visual_dashboard ...` slash 调用方式已经移除，统一通过 `/agent` 选择：

```bash
/agent gen_visual_dashboard
```

（或者直接 `/agent`（不带参数）打开 agent TUI，选中 `gen_visual_dashboard` → Enter 编辑、`s` 设为默认。）切好之后直接提问就行——说清楚问题、想看的筛选维度，以及关心的核心指标：

```bash
做一份门店销售总览仪表盘,支持按地区、月份筛选,核心指标包含 GMV、订单量、客单价
```

```bash
新用户激活漏斗仪表盘,可以按来源渠道、注册时段筛选,展示注册→首单→次日留存各环节转化
```

要编辑已有的仪表盘,按显示名或 slug 引用即可:

```bash
给仪表盘 store_sales_overview 的筛选器加一个「客单价区间」
```

```bash
修改「门店销售总览」,把按月聚合改成按周
```

仪表盘生成完成后,聊天面板会输出 HTML 文件的绝对路径,以及供他人启动后端时使用的 `datus --web --datasource <ds>` 命令——浏览器里把 HTML 打开就能用,关掉再开也不用重新生成。

## 用 metric 或你自己的 SQL 生成仪表盘

`gen_visual_dashboard` 接受两种同样合法的起点:

- **从 metric 出发** —— 用 `@Metrics <subject>.<group>.<metric>` 三段式直接引用已经生成的指标(subject 树路径 + 指标名),agent 会自动从语义层加载定义、维度和时间窗,并据此生成对应的筛选器。项目里已经沉淀了 metric 注册表时最适用(如何生成指标见 [Generate Metrics](gen_metrics.zh.md))。
  ```bash
  围绕 @Metrics revenue.daily.dau 和 @Metrics conversion.weekly.signup_rate 做一份运营总览,支持按地区、日期范围筛选
  ```

- **从 SQL 出发** —— 把你想要的 SQL 贴进 prompt,agent 会从里面识别出可参数化的条件(日期范围、枚举值、ID 列表等),把它们提升成筛选器,再组织图表。适合临时探索,或者还没有现成 metric 的场景。
  ```bash
  用下面这段 SQL 做一份商户对账仪表盘,把日期范围和门店 ID 做成筛选器:
      SELECT trade_date, store_id, SUM(amount) AS gmv, COUNT(*) AS orders
      FROM mart.merchant_daily
      WHERE trade_date BETWEEN '2025-01-01' AND '2025-12-31'
      GROUP BY trade_date, store_id
  ```

两种方式也可以混用——主 KPI 用 metric,某一处下钻或筛选项用你自己的 SQL。

## 仪表盘是怎么生成的

```mermaid
graph LR
    A[你的问题] --> B[识别筛选器<br/>和核心指标]
    B --> C[把 SQL 写成<br/>参数化模板]
    C --> D[组织视图 + 图表]
    D --> E[编译 HTML<br/>+ 启动后端]
    E --> F[在浏览器中打开<br/>实时筛选查询]
```

agent 会先理解你的问题,识别出哪些是筛选维度、哪些是核心指标,把对应的 SQL 写成参数化的 Jinja2 模板(参数声明 + 模板正文),保存到项目下的 `dashboards/<slug>/` 目录。HTML 视图在浏览器打开后,每次切换筛选器都会向本地 `datus --web` 后端的 `/api/v1/dashboard/query` 发起请求,后端把模板渲染成 SQL 跑库,把结果回填到图表。这套链路全程都在你自己的环境里,不会把数据外发。

如果之后要修改,再次调用 `gen_visual_dashboard` 即可原地编辑同一份仪表盘。

## 按模块独立修改

每份仪表盘都是由相互独立的模块组成的——筛选器、单张图表、KPI 卡片、数据表等等。你可以只针对 **其中一个** 模块做改动,其它部分不会被牵连:

```bash
把 store_sales_overview 的客单价图表改成按地区分组的柱状图
给 store_sales_overview 加一个「商品大类」筛选器
删掉客户构成饼图,这次不需要
调整一下 KPI 卡片的顺序,把 GMV 放在最前面
```

每一次调用都是定点修改:agent 会定位到对应的模块,只编辑它、只重写相关的 SQL 模板,其它的版面、筛选器和图表都保持原样——保持你之前审过的状态。

## agent 能用到的资源

为了把仪表盘做好,`gen_visual_dashboard` 会复用项目里已经配置好的全部能力:

- **语义层** —— 优先使用已定义的 metric 和维度,能匹配上的话不会临时写 SQL;维度也直接做成筛选器。
- **数据库** —— 读取 schema、取样,并基于已配置的数据源执行试跑(验证参数化模板能跑通)。
- **知识库** —— 提前查阅项目中已沉淀的参考 SQL 和业务术语表。
- **既有仪表盘** —— 当你让 agent 参考某个已有仪表盘时,它会读取那份仪表盘的筛选器布局和图表配置作为灵感。

这些都不需要你手动调用——写好 prompt 就行。

## 配置

`gen_visual_dashboard` 开箱即用,无需任何配置。下面这些在 `agent.yml` 中是可选项:

```yaml
agent:
  agentic_nodes:
    gen_visual_dashboard:
      model: claude              # 可选,默认使用当前配置的模型
      max_turns: 30              # 可选,默认 30
      web_host: localhost        # 可选,生成的 HTML 里筛选请求指向的 host
      web_port: 8501             # 可选,生成的 HTML 里筛选请求指向的 port
```

| 参数 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `model` | 否 | 使用的 LLM 模型 | 当前配置的默认模型 |
| `max_turns` | 否 | agent 最多迭代多少轮后强制结束 | 30 |
| `web_host` | 否 | 编译进 HTML 的查询后端 host | `localhost` |
| `web_port` | 否 | 编译进 HTML 的查询后端 port | `8501` |
| `query_endpoint` | 否 | 直接指定完整 URL,覆盖 `web_host` / `web_port` 的组装结果 | 由 `web_host` + `web_port` 组装 |

> 把 HTML 发给别人时,接收方需要在本地用相同的 `datus --web --datasource <ds>` 启动后端;如果换了 host / port,记得在 `agent.yml` 里同步改 `web_host` / `web_port`,或直接配 `query_endpoint`,这样 HTML 烘焙进去的查询地址才能对上。

## 怎么让 prompt 更有效

下面这些信息越明确,仪表盘越对路:

- **场景** —— *"门店销售总览"*、*"新用户激活漏斗"*、*"商户对账"*
- **核心指标** —— *"GMV、订单量、客单价"*、*"激活率、留存率"*
- **筛选维度** —— *"按地区 / 月份"*、*"按渠道 / 注册时段"*、*"按门店 ID"*
- **时间范围** —— *"近 90 天"*、*"2025 全年"*(也会自动转成筛选器的默认值)

省略也行,agent 会基于项目里的 metric 和你给的 SQL 自行推断,仅在意图严重模糊时才主动追问。
