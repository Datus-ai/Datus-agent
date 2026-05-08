# VSCode Extension 用户指南

## 概览

**Datus Studio** 是 Datus-agent 的官方 VSCode 插件，把 Datus 的数据工程能力直接搬进你日常使用的 IDE。它复用 Datus-agent 的 Web Server（HTTP 服务），通过统一的 Endpoint 连接后端，所有元数据、子代理、上下文、模型与数据源配置都来自同一个 Datus 实例。

面向场景：

- **数据工程师 / 分析师**：在写代码、调 SQL 的同时，在同一窗口完成自然语言问答、SQL 生成、结果可视化，避免在 IDE / 网页 / 终端之间来回切换。
- **AI Native 数据开发**：把 Object Catalog、Context（Metrics / Reference / Knowledge）、SubAgent、Chat、SQL Result、AI Chart 整合进 VSCode 的标准面板，按 IDE 习惯组织工作流。
- **多数据源协作**：在一个工程下连接多个数据库，切换 SubAgent、切换 Datasource、切换 Plan mode 都不离开编辑器。

![Datus Studio 主界面](../assets/vscode-main.png)

## 安装

当前发布渠道为 VSIX 直装：

1. 下载最新版本：[https://cdn.studio.datus.ai/vsc/release/datus-studio-vsix.zip](https://cdn.studio.datus.ai/vsc/release/datus-studio-vsix.zip)，解压 zip 得到 .vsix 文件。
2. 在 VSCode 中打开命令面板（`Cmd/Ctrl + Shift + P`），执行 **「Extensions: Install from VSIX...」**，选择刚下载的 `.vsix` 文件。
3. 安装完成后，活动栏会出现 Datus Studio 图标，并新增三个面板：左侧 **Datus: Object Explorer**、主面板 **Datus Studio**（Chat，你可以手动拖拽到右侧）、底部 **Datus Studio**（SQL / Chart）。

!!! tip "升级"
    重新下载最新的 `.vsix` 文件后再次执行 *Install from VSIX* 即可覆盖安装。

## 启动步骤

Datus Studio 自身不持有任何模型与数据库凭证，所有能力都由 Datus-agent 的 Web Server 提供。启动顺序如下。

### 1. 启动带 Web Server 的 Datus-agent

任意带有 `--web` 的启动方式都可以，关键是要让 Datus-agent 暴露 HTTP 服务。常见命令：

```bash
# 直接指定数据源
datus-cli --web --datasource <your_datasource>

# 指定配置文件 + 数据源（推荐：项目化配置）
datus-cli --web --config /path/to/conf/agent.yml --datasource <your_datasource>

# 自定义端口与监听地址
datus-cli --web --port 8080 --host 0.0.0.0
```

启动完成后，终端会打印实际的服务地址，例如：

```text
http://localhost:8501
```

将这个地址记下，下一步要填到插件 Settings 里。

!!! note "为什么必须 `--web`"
    VSCode 插件通过 HTTP 与 Datus-agent 通信。CLI 模式（不带 `--web`）只在终端内运行，没有 HTTP 端口，插件无法连接。

### 2. 在 VSCode 中配置 Endpoint

1. 打开活动栏的 **Datus Studio** Chat 面板，点击右上角齿轮图标进入 **「Datus · Settings」**。
2. 切到 **General** 页，将上一步的服务地址填入 **Endpoint**，例如 `http://localhost:8501`。
3. 同页还可以设置：
    - **Language**：插件界面语言（中 / 英）。
    - **Default Model**：默认对话使用的 LLM，下拉来自 Datus-agent 已配置的 providers。
    - **Feedback**：直接向 Datus 团队提交反馈或建议。

![Datus Studio 设置页](../assets/vscode-settings.png)

保存后，插件会自动 Reload 与 Datus-agent 建立连接，左侧 Object Explorer 会拉取数据库目录与已注册的子代理。

!!! tip "Models / Datasources 子页"
    **Models** 与 **Datasources** 两个子页是 Datus-agent 同名配置的可视化镜像，可在插件里直接编辑，写回到对应的 `agent.yml`。

## 核心功能

整个插件按 VSCode 习惯划分为 **左侧 Catalog Tree/Context Tree/SubAgent Explorer**、**右侧 Datus Studio Chat**、**底部 Datus Studio (SQL / Chart)** 三块。

### 1. Catalog Tree 与 Context Tree

左侧 **Datus: Object Explorer** 提供两个 Tab：

- **Catalog**：当前数据源下的 `database → schema → table` 树，对应 Datus-agent 抓取的元数据。
    - 点击表名 → 在编辑区打开 *Table Info* 标签页，展示 **Columns**（字段类型 / 是否可空 / 默认值 / PK）、**Indexes**、**Sample data**。
    - 顶部支持搜索与刷新，元数据变更后点刷新立即同步。
- **Context**：上下文知识树，与 Datus-agent 的 `subject/` 目录对齐：
    - **Metrics**：MetricFlow 指标定义。
    - **Reference**：Reference SQL 与 Reference Template。
    - **Knowledge**：External Knowledge / Platform Documentation。

Catalog 与 Context 中的任意节点都可以作为 Chat 的引用对象（在 Chat 输入框 `@` 选择），也可以作为 SubAgent 的可见范围（见下文）。

### 2. SubAgent 创建与管理

左下 **Agents** 区列出当前工程的全部子代理（含 Built-in 与自定义）。点击 **「+ Add Agent」** 进入四步向导：

1. **Description**：基本信息（名称、描述、面向场景）。
2. **Tools**：勾选可用工具 / MCP 工具。
3. **Objects**：选择该 SubAgent 可见的 **Catalog 范围** 与 **Context 范围**——同样使用上一节的两棵树，勾选到 schema / table / 知识节点级别。底部 *Selected* 区会以 `jeff_shop.*`、`Root.*` 等通配符形式回显。
4. **Rules**：补充行为规则、Few-shot 示例、对话风格等。

![创建 SubAgent 向导（Objects 步骤）](../assets/vscode-create-subagent.png)

向导提交后，子代理会写入 Datus-agent 的配置（与 CLI 中的 `/agent` 等价），随即在 Chat 面板的 SubAgent 下拉中可选。Built-in 子代理（`gen_sql`、`gen_metrics`、`gen_dashboard`、`gen_semantic_model`、`gen_report`、`scheduler`、…）只读展示，可在向导中查看其 Tools / Objects / Rules。

### 3. Datus Studio Chat 面板

右侧 **Datus Studio** 是与 Datus-agent 对话的主入口，对齐 Web Chatbot 但更贴近 IDE 工作流：

- **普通对话**：自然语言提问，Datus 会生成 SQL、调工具、并把结果以 Markdown / 表格 / 链接渲染回来。
- **SubAgent 切换**：输入框下方 **Main...** 下拉，选择 `gen_sql`、`gen_dashboard`、`gen_metrics` 或自定义子代理。切换后 Chat 上下文随即接入该子代理的可见范围与规则。
- **Datasource 切换**：右下 **数据库** 下拉，无需重启即可切换当前会话的数据源。
- **Plan mode**：开启后，Datus 会先输出执行计划与 SQL 草稿，等你确认再执行；适合生产数据上的高风险查询。
- **会话历史**：聊天历史按 `~/.datus/sessions/{project}/` 分项目存储，插件侧边可加载历史会话继续聊或对照查看。
- **代码联动**：在编辑器内选中一段 SQL 或文本，右键 *Datus: Add Selection to Chat*（命令 `datus.addSelectionToChat`），即可把当前选区作为上下文丢进 Chat。

### 4. SQL Result 与 AI Chart 面板

执行 SQL（来自 Chat 生成、手写、或 SubAgent 回写）后，结果会显示在底部 **Datus Studio** 面板，三个子 Tab：

- **Generated SQL**：本轮生成 / 执行的 SQL 全文，支持复制与一键回灌到编辑器。
- **Execute Result**：分页表格，支持排序、列宽调整、CSV / Excel 下载。
- **Chart**：基于 [ECharts](https://echarts.apache.org/) 的可视化区。

Chart 子 Tab 提供：

- **图表类型切换**：柱状图 / 折线图 / 散点图 / 饼图。
- **维度 / 度量绑定**：通过 *Showing `<measure>` By `<dimension>`* 选择字段；齿轮图标可调整轴、配色、堆叠等细节。
- **AI 解读**：图表上方自动给出由 LLM 生成的趋势 / 异常 / 对比说明。
- **导出**：右上角下载图标可导出 PNG / 数据。

![SQL Result 与 AI Chart 面板](../assets/vscode-sql-chart.png)

### 5. FileSystem Tools（文件操作接管）

Datus-agent 在自然语言对话与子代理执行过程中会调用一组文件工具（`read_file` / `write_file` / `edit_file` / `glob` / `grep`），用于读写本地文件、查找代码与配置。在 **VSCode 场景下，这组文件工具会被插件接管**：所有文件操作都不会落到 Datus-agent 进程所在的目录，而是统一作用于 **VSCode 当前打开的工作区目录**（`workspaceFolders[0]`）。

- **统一的根目录**：无论 Datus-agent 在哪台机器、以哪个工作目录启动，文件读写永远发生在你 VSCode 里看到的那个项目，避免出现 “SQL 写出来了但找不到文件” 的割裂感。
- **强制的路径沙箱**：所有传入路径都会校验前缀是否在工作区内。任何尝试越界（如 `../../etc/passwd`）都会以 `Invalid path` 拒绝。
- **白名单 + 体积限制**：仅允许文本类扩展名（`.txt` `.md` `.py` `.js` `.ts` `.json` `.yaml` `.yml` `.csv` `.sql` `.html` `.css` `.xml` 等）；单文件读取上限 **10MB**，`grep` 跳过 >1MB 的文件，`glob` 与 `grep` 自动忽略 `.git/`、`node_modules/`。

#### 使用建议

- **保持 VSCode 工作区干净**：因为读写都落在工作区根目录，建议每个 Datus 项目对应一个 VSCode 工作区文件夹，与 Datus-agent 的 `project_name`、`subject/` 目录一一对应，子代理生成的 `metric.yml`、`semantic_model.yml`、`reference_sql/*.sql` 才会落到正确位置。
- **未打开文件夹的情况**：VSCode 没有打开任何工作区时，所有文件工具都会返回错误。先用 *「File → Open Folder...」* 打开你的项目再发起对话。
- **超出白名单的文件**：二进制、Notebook、Office 文档等会被直接拒绝。需要让 Datus 处理这些文件时，先转换成支持的文本格式或在白名单中扩展（修改插件源码）。

## 常见问题

### 插件一直显示 "Disconnected"
检查 Endpoint 是否填对、Datus-agent 是否带 `--web` 启动；远程主机要确认防火墙放行了对应端口。

### 端口被占用
`datus --web --port 8080 ...` 自定义端口后，把 Settings → Endpoint 同步改成 `http://localhost:8080`。

### Catalog 看不到表
确认 Datus-agent 已对该数据源跑过元数据抓取（CLI 中的 `/init` 或 `/refresh-meta`）。插件本身只渲染后端返回的目录。

### Built-in 子代理可以编辑吗
不可以。Built-in 仅展示其结构，要定制请使用 *「+ Add Agent」* 新建一个独立子代理。

## 总结

Datus Studio VSCode 插件把 Datus-agent 的核心能力——元数据浏览、上下文管理、子代理编排、Chat 对话、SQL 执行与 AI 可视化——整合进一个 IDE 内的工作区，与 [CLI](../cli/introduction.md) 与 [Web Chatbot](../web_chatbot/introduction.md) 共用同一份后端配置与知识库，是开发态最顺手的入口。
