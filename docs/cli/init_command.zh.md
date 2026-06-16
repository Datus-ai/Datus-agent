# Init 命令 `/init`

## 概览

`/init` 是一个轻量级快捷入口——它让 chat agent 按内置 **`init` skill**
做一次**轻量化**项目初始化。它产出的是一份快速、低成本的首轮结果——
`AGENTS.md` 清单加上廉价的文件类存储——而不为昂贵的向量库生成买单。
skill 会带着 agent 走以下流程：

1. 从 `README.md`、目录树和 `agent.yml` 推断项目目标与数据源范围——
   前期不提任何问题
2. 扫描文件与数据库元数据（目录树、表结构描述、样本行），
   并把所有内容归类到业务域
3. 直接落盘**廉价的文件类存储**：原子的、不可推断的业务事实经
   lite `extract-knowledge` 写入 `./knowledge/*.md`,跨会话的持久偏好
   写入 `memory`
4. 写出 **`AGENTS.md` 清单骨架** 到 `./AGENTS.md`——Architecture、
   Directory Map、Services、Data Assets、Recommended Tools、SQL
   Conventions 以及 Knowledge 索引——而向量索引段
   （`## Semantic Models` / `## Metrics` / `## Reference SQL`）留作
   指向 `/build-kb` 的占位

全程**没有确认门**：knowledge、memory、`AGENTS.md` 都是廉价可逆的
markdown 写入,skill 直接执行（仅在整体覆盖已存在的 `AGENTS.md` 前会
征询确认）。

`/init` 刻意**不触及**昂贵的向量索引存储
（`semantic_models` / `metrics` / `reference_sql`）。要构建它们——
包括把 validated-query 语料索引为 few-shot 检索样本——请随后运行
[`/build-kb`](build_kb_command.zh.md)。

由于 `/init` 走的是标准 chat 流水线,所以你会自动获得熟悉的 UX——
`ActionHistory` 事件流式渲染、**Ctrl+O** 切换 trace 详情、**ESC** 中断、
以及生成后继续追问以微调具体章节的能力。

skill 源文件位于 `datus/resources/skills/init/SKILL.md`,运行时由 skill
registry 直接从已安装的包内加载——不再向 `~/.datus/skills` 拷贝任何文件。
如需自定义行为,只要在 `./.datus/skills/init/SKILL.md`（项目级,优先级最高)
或 `~/.datus/skills/init/SKILL.md`（用户级）放一份同名 skill 即可遮蔽内置版本,
全程零代码改动。

---

## 基本用法

```text
> /init
> /init 这是一个销售分析数仓,重点关注订单域
```

`/init` 接受一段可选的自由文本描述。命令后的内容会原样转发给 skill,
作为目标/范围提示,由 skill 合并进推断出的上下文。不带参数时,skill
自行推断全部上下文。数据源范围默认是当前 REPL 选中的那一个（启动时
通过 `--datasource` 指定,或运行时用 `/datasource` 切换）。即便配置了多个
数据源,`/init` 也只初始化当前激活的那一个——要换目标请先用
`/datasource <name>` 切换。

执行时你会看到标准的 chat trace:`load_skill`、文件系统扫描
（`glob` / `grep` / `read_file`）、数据库元数据调用（`list_tables`、
`describe_table`）、针对原子事实的 `extract-knowledge` 一轮,最终以写入
`AGENTS.md` 的 `filesystem_tools.write_file` 收尾。不做 `explore` 扇出,
也没有 Generation Manifest——那些属于 `/build-kb`。

---

## 前置条件

- 已配置 LLM。如未配置先运行 `/model`——`/init` 需要 agent 来驱动每一步。
- `~/.datus/conf/agent.yml` 非空。先用 `/datasource` 添加数据源,这样它们
  才会出现在 skill 输出的"Services"表里。

需要换数据源时,先用 `/datasource <name>` 切换,再运行 `/init`。

---

## 与 `/build-kb` 的关系

| | `/init`（轻量） | [`/build-kb`](build_kb_command.zh.md)（重量） |
|---|---|---|
| 成本 | 低,单回合无确认门 | 高,扇出子代理 + 确认门 |
| 写入 | `AGENTS.md` 清单、`./knowledge/*.md`、`memory` | `semantic_models`、`metrics`、`reference_sql`（向量/LanceDB）,并刷新 `AGENTS.md` 的 KB 索引 |
| 范围 | 整个项目 | 可选文件/库表/业务域范围 |
| 确认 | 无（除覆盖 AGENTS.md 外） | 打印 Generation Manifest 后停下等确认 |

典型流程是先 `/init` 建图,再用 `/build-kb`（可限定范围）构建检索型知识库。

---

## 自定义输出形态

skill 是 `AGENTS.md` 形态的唯一信息源。想改章节结构、表格列名、摘要
风格,只需修改:

- **项目级覆盖**（推荐,适合一次性调整,优先级最高）:
  `./.datus/skills/init/SKILL.md`
- **用户级覆盖**（影响所有项目）:
  `~/.datus/skills/init/SKILL.md`
- **内置兜底**（随包发布,始终可用）:
  `datus/resources/skills/init/SKILL.md`

项目级 → 用户级 → 内置,同名 skill 按此顺序逐级遮蔽。

---

## 多轮微调

`/init` 本质是一次 chat,因此完成后可以继续追问、就地修改:

```text
> /init
… <流式输出,AGENTS.md 已写入> …
> 重写 Architecture 章节,重点放在数仓分层。
```

agent 会用 `filesystem_tools.write_file` 直接修改 `AGENTS.md`。

参见:[`/build-kb`](build_kb_command.zh.md)、[`/model`](model_command.zh.md)、[斜杠命令参考中的 `/datasource`](reference.zh.md)、[Skills 集成](../integration/skills.zh.md)。
