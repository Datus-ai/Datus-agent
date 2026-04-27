# Init 命令 `/init`

## 概览

`/init` 在当前项目目录生成 `AGENTS.md`,描述项目的架构、目录结构、已配置的
服务,以及当前数据源的数据资产——这是下游 `datus` 子 agent 与外部编码 agent
（Claude Code、Cursor 等）使用的上下文文件。

`/init` 完全在 REPL 内运行:复用当前会话已加载的 LLM（`/model`）与 agent
配置,无需单独的初始化步骤。`/init` **不接受任何参数**——用于补充 prompt 的
数据源就是当前 REPL 选中的那一个（启动时通过 `--datasource` 指定,或运行时
用 `/datasource` 切换）。

---

## 基本用法

```text
Datus> /init
```

处理流程:

1. 读取 `agent.yml`(必须已存在;请先用 `/datasource` 配置数据源、用 `/model`
   配置 LLM)。
2. 扫描当前目录（深度最多 3 层),自动跳过 `.git`、`node_modules`、
   `__pycache__` 等噪音目录。
3. 通过指示文件检测项目类型(`pyproject.toml`、`package.json`、
   `Dockerfile`、`dbt_project.yml` 等)。
4. 如果存在 `README.md`(或 `README.rst` / `README` / `readme.md`),读取其
   内容作为上下文。
5. 当 REPL 已选中数据源时,**自动**取该数据源的表清单加入 LLM 上下文。
6. 把上述信息一并交给当前 LLM 生成 `AGENTS.md`;LLM 失败时回退到模板骨架。
7. 把结果写入项目根目录的 `AGENTS.md`。

若 `AGENTS.md` 已存在,会询问 **overwrite**（覆盖）还是 **cancel**（取消）。

需要换数据源时,先用 `/datasource <name>` 切换,再运行 `/init`。

---

## 生成内容

输出文件结构:

| Section | 来源 |
|---------|------|
| `# <project-name>` | 目录名 + LLM 生成的一句话描述 |
| `## Architecture` | LLM,基于目录树、项目类型与 README 摘要 |
| `## Directory Map` | LLM,目录与其用途/入口/调用方的对照表 |
| `## Services` | 来自 `agent.services.datasources`（`agent.yml`) |
| `## Data Tables` | 仅当 REPL 已选中数据源时出现 |
| `## Artifacts` | LLM,如数据目录、语义模型、SQL 文件、配置等 |

---

## 前置条件

- 已配置 LLM。如未配置,先运行 `/model`。
- `~/.datus/conf/agent.yml` 非空。CLI 首次启动会自动创建最小化的
  `.datus/config.yml`,但若希望 Services 章节包含数据源,运行 `/init` 前请
  先用 `/datasource` 添加。

如果找不到 `agent.yml`,`/init` 会打印提示并直接退出,不写任何文件。

---

## 示例

```bash
# 用当前模型与当前数据源生成 AGENTS.md
Datus> /init

# 想换一个数据源,先切再 /init
Datus> /datasource duckdb-demo
Datus> /init
```

参见:[`/model`](model_command.zh.md)、[斜杠命令参考中的 `/datasource`](reference.zh.md)。
