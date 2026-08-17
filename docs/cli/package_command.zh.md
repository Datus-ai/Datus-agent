# 打包命令

`datus package` 命令将当前项目目录导出为一个**自包含的 zip 包**。接收方解压、导出几个环境变量、运行一条初始化脚本，就得到一个可用的 Datus 项目 —— 不依赖共享的 `~/.datus`，无需手工改配置，也不会有任何凭据随包外流。

## 设计保证

| 保证 | 实现方式 |
|---|---|
| **自包含** | 生成的 `conf/agent.yml` 固定 `home: .` 和 `project_name`，解压出的目录就是完整运行时。接收方的 `~/.datus` 不会被读写。 |
| **零密钥** | `conf/agent.yml` 和 `conf/.mcp.json` 是**重新生成**的，从不直接拷贝。每个凭据字段都被替换为 `${VAR}` 占位符。打包前会对所有待写入文件做一次内容扫描，一旦发现疑似密钥就直接构建失败 —— 没有跳过开关。 |
| **只带源文件，不带索引** | 指标 / 语义模型 / Reference SQL 的 **YAML 源文件**随包发出，并附带生成的 `scripts/rebuild_kb.sh`。二进制 LanceDB 索引不入包，由接收方本地重建。 |

## 使用方法

```bash
# 交互式向导（常规用法）
datus package

# 非交互：按默认值打包全部内容
datus package -y
```

所有参数都通过向导收集 —— `-y/--yes` 是唯一的命令行开关，用于脚本 / 非 TTY 场景。请在项目根目录下执行，被打包的就是当前工作目录。

任意步骤按 `Ctrl+C` 均可中断。中断时不会留下任何产物；若中断发生在写 zip 的过程中，已写了一半的归档会被删除。

### 选项

| 选项 | 描述 |
|------|------|
| `-y`, `--yes` | 跳过向导，按默认值打包全部内容。stdin/stdout 不是 TTY 时必须使用。 |

### 退出码

| 退出码 | 含义 |
|---|---|
| `0` | 打包成功 |
| `1` | 构建失败（包括最终扫描发现密钥） |
| `2` | 参数错误，或非交互终端且未传 `--yes` |
| `3` | 未找到 agent 配置（`./conf/agent.yml` 或 `~/.datus/conf/agent.yml`） |
| `130` | 被 `Ctrl+C` 取消 |

## 向导步骤

向导是线性流程，内容为空的分类会被静默跳过。

| 步骤 | 询问内容 |
|---|---|
| **输出路径** | zip 的写入位置，默认 `./<project_name>.zip`，必须以 `.zip` 结尾。目标文件已存在时会先确认。 |
| **文件范围** | 打包全部文件，或提供逗号分隔的 **include / exclude 正则**。正则会当场校验，非法时重新询问。 |
| **Subagents** | 选择要携带的 `agent.yml` agentic node，其提示词模板会自动一并打包。 |
| **Skills** | 项目技能（`./.datus/skills`）与全局技能（`~/.datus/skills`），列表中标注来源。 |
| **指标数据源** | 选择要携带的 `subject/semantic_models/<datasource>` 目录。 |
| **主题域（Subject areas）** | 两层主题树。**同时**约束指标文档与 Reference SQL 摘要 —— 见下文。 |
| **插件** | 已安装的插件，写入 `scripts/install_plugins.sh`。 |
| **报告** / **看板** | 选择 `reports/` 与 `dashboards/` 下要包含的产物。 |
| **报告 dist** | 仅在选了报告时询问：是否内置 `web-artifact-render` dist 以便 `index.html` 通过 `file://` 打开；默认走 CDN。 |
| **确认摘要** | 列出全部选择的表格，最后确认。选择「否」以 `130` 退出且不写任何文件。 |

每个多选界面默认全选：`Space` 切换、`a` 全选/全不选、`Enter` 确认。整类取消时会二次确认，避免一次误触的 `Ctrl+C` 悄无声息地丢掉一整类内容。

### 主题树选择

主题域读自项目在向量库中的 subject tree，**最多渲染两层** —— 根节点及其直接子节点，更深的路径折叠进它的第二层父节点：

```
[✓] 营销分析 (24 reference SQL)
[✓]   └ 活动统计 (15 reference SQL)
[-]   └ 预算分析 (2 reference SQL)
[ ] 运营 (22 metrics)
[ ]   └ 活动 (22 metrics)
```

- 计数**向上汇总**，所以根节点直接说明「全选它要付多少代价」。
- 复选框**双向联动**：勾选父节点会连带勾选全部子节点；子节点全部选中时父节点自动勾选。部分选中的父节点标记为 `[-]`。
- 匹配按路径前缀进行：选 `营销分析` 保留整棵子树，选 `营销分析/活动统计` 则收窄到该分支。

指标文档通过 `subject_tree:` 标签匹配，并且是**按文档过滤而非按文件** —— 一个 metrics YAML 可能横跨多个主题域，只有命中的文档会随包发出。Reference SQL 通过 `subject/sql_summaries/` 中每份摘要的 `subject_tree` 字段匹配。

没有主题标签的指标文档和摘要仍会被打包，同时给出警告 —— 它们不属于任何主题域，否则会从所有过滤后的包中消失。

## 包结构

```
<project_name>.zip
├── README.md                    # 生成：快速上手 + 所需环境变量
├── requirements.txt             # 生成：固定版本的 datus 依赖
├── package_manifest.json        # 生成：格式、选择项、环境变量及其引用位置、逐文件 sha256
├── conf/
│   ├── agent.yml                # 生成：home: .，凭据为 ${VAR} 占位符
│   └── .mcp.json                # 生成（仅当配置了 MCP server）
├── .datus/
│   ├── config.yml               # 生成：固定 project_name / 默认数据源
│   └── skills/                  # 选中的项目技能
├── scripts/
│   ├── init.sh                  # 生成：装依赖 → 装插件 → 重建知识库
│   ├── install_plugins.sh       # 生成：逐个 datus plugin install --force
│   └── rebuild_kb.sh            # 生成：逐份源 YAML 执行 bootstrap-kb
├── subject/
│   ├── semantic_models/<ds>/    # 选中的语义模型与指标文档
│   └── sql_summaries/           # 选中的 Reference SQL 摘要
├── template/                    # 选中 subagent 的提示词模板
├── reports/ · dashboards/       # 选中的产物
└── ...                          # 项目的其余文件
```

`package_manifest.json` 记录包格式版本、精确的选择项、所需环境变量，以及每个文件的 `sha256` 和 `generated` / `project` 来源标记。

`env_vars`（格式版本 2）不再是一个变量名列表，而是每个变量一条记录，接收方可以据此看出每个占位符对应哪个配置项：

```json
"env_vars": [
  {
    "var": "OPENAI_API_KEY",
    "config_paths": ["models.gpt4.api_key", "providers.openai.api_key"],
    "preexisting": false
  }
]
```

`config_paths` 列出 `conf/agent.yml`（或 `conf/.mcp.json`）中引用该变量的所有字段。`preexisting` 为 `true` 表示源配置在上述每一处本来就写的是 `${VAR}`；为 `false` 表示其中至少有一处原本是字面量、由打包器替换而来——也就是说，该变量顶替的是一个原本明文的凭据。

### 永不入包的内容

| 分类 | 条目 |
|---|---|
| 运行时状态（顶层） | `sessions/` `data/` `logs/` `run/` `cache/` `save/` `trajectory/` `output*/` `.venv/` `.git/`，以及 REPL 的 `history` 文件 |
| 密钥与系统/编辑器垃圾 | `.env` `.DS_Store` `._*`（macOS AppleDouble 附属文件）`__MACOSX/` `__pycache__/` `*.swp` `*.swo` `*~` `*.duckdb.wal`，以及 `.Spotlight-V100/` 等卷元数据 |
| 二进制索引 | `data/` 下的 LanceDB 数据 —— 由 `scripts/rebuild_kb.sh` 重建 |

`reports/`、`dashboards/`、`template/` 由选择器接管：即便选了「打包全部文件」，也只有向导中勾选的内容会入包。

## 密钥处理

写入包之前，`agent.yml` 中的每个凭据字段都会被改写为 `${VAR}` 占位符：

```yaml
agent:
  home: .
  project_name: baisheng
  providers:
    deepseek:
      api_key: ${DEEPSEEK_API_KEY}
  services:
    datasources:
      starrocks:
        host: ${STARROCKS_HOST:-127.0.0.1}
        port: ${STARROCKS_PORT:-9030}
        password: ${STARROCKS_PASSWORD}
```

- 源配置中本来就写成 `${VAR}` 或 `${VAR:-default}` 的字段，会保留原变量名和默认值。
- 识别是**基于 schema** 的 —— 由字段的角色决定，而不是看值，因为明文密钥和普通字符串在值这一层无法区分。
- 数据库 URI 只替换密码部分，主机、端口、库名原样保留：`postgresql://svc:${DATUS_DS_PG_URI_PASSWORD}@db.example.com/warehouse`。
- 插件配置按各插件的 config schema 脱敏。schema 加载失败的插件会退化为替换**所有**字符串字段并给出警告 —— 默认偏安全。

暂存完成后会对整个包做一次内容扫描。一旦发现疑似真实密钥，构建**失败**并打印出问题文件及定位；修正源配置（或排除该文件）后重试即可。

生成的 `README.md` 会列出全部所需变量及其用途：

| 变量 | 用于 |
|---|---|
| `DEEPSEEK_API_KEY` | providers.deepseek.api_key |
| `STARROCKS_PASSWORD` | services.datasources.starrocks.password |

## 接收方使用

```bash
unzip baisheng.zip -d baisheng && cd baisheng
export DEEPSEEK_API_KEY=... STARROCKS_PASSWORD=...   # 见 README.md
bash scripts/init.sh
datus-api        # 或 `datus` 进入交互式控制台
```

`init.sh` 依次执行：安装依赖 → 安装插件 → 重建知识库。可以反复运行：pip 是幂等的，插件安装带 `--force`，知识库步骤为覆盖写。每一步也可单独执行 —— 修改过 subject YAML 之后单跑 `scripts/rebuild_kb.sh` 即可。

!!! note
    通过 pip 安装的 datus-agent **不会**自动加载 `.env` 文件。请在 shell 中导出变量，或执行 `set -a; source .env; set +a`。

`init.sh` 使用 `$PYTHON`（默认 `python3`）并优先选择 `uv` 而非 `pip`，因此在没有 `pip` 模块的 uv 虚拟环境中同样可用。需要指定解释器时设置 `PYTHON=/path/to/python`。

## 示例会话

```
$ datus package
Packaging project 'baisheng' from /Users/me/baisheng-project
Output zip path [/Users/me/baisheng-project/baisheng.zip]:
Package all files? [Y/n]: y
Subagents: Space toggles, 'a' toggles all, Enter confirms
...
Subject areas: Space toggles, 'a' toggles all, Enter confirms
...
             Package summary
  Item                Value
  Project             baisheng
  Output              /Users/me/baisheng-project/baisheng.zip
  Subject areas       营销分析/活动统计
  Metric datasources  starrocks
  ...
Build the package now? [Y/n]: y
✓ Package built: /Users/me/baisheng-project/baisheng.zip
  163 files, 4.2 MB uncompressed
  Subject areas: 营销分析/活动统计 → 15 reference-SQL summaries
  Receiver must export: DEEPSEEK_API_KEY, STARROCKS_PASSWORD
```

结果始终会明确说明这次选择到底产出了什么。手工数 zip 条目并不可靠 —— `unzip -l` 会把过长或中文文件名折行显示，让过滤正确的包看起来像没过滤。

## 相关文档

- [知识库 —— Reference SQL](../knowledge_base/reference_sql.md)
- [知识库 —— 指标](../knowledge_base/metrics.md)
- [配置 —— Agent](../configuration/agent.md)
