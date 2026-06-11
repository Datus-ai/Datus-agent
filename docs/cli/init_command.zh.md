# Init 命令 `/init`

## 概览

`/init` 是一个轻量级快捷入口——它让 chat agent 按内置 **`init` skill**
端到端地初始化项目工作区：构建知识库,最后以 `AGENTS.md` 总览收尾。
skill 本身会带着 agent 走完整流程：

1. 从 `README.md`、目录树和 `agent.yml` 推断项目目标与数据源范围——
   前期不提任何问题
2. 扫描文件与数据库元数据（目录树、表结构描述、样本行、行数统计），
   并把所有内容归类到业务域
3. 用只读 `explore` 子代理并行探索每个业务域（并发上限 3）
4. 输出 **Generation Manifest**（推断出的目标、数据源范围、每个待生成
   产物一行），然后**停下等待确认**——回复确认,或修正目标/范围/任意一行
5. 确认后按 `storage-classify` skill 把每个产物路由到对应存储：
   semantic models、metrics、reference SQL、knowledge,各由匹配的
   生成子代理产出
6. 最后生成精简的 `AGENTS.md` 总览并写入 `./AGENTS.md`
   （已存在时会先确认是否覆盖）

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
作为目标/范围提示,由 skill 合并进推断出的上下文和 Generation Manifest。
不带参数时,skill 自行推断全部上下文。数据源范围默认是当前 REPL 选中的
那一个（启动时通过 `--datasource` 指定,或运行时用 `/datasource` 切换）。

执行时你会看到标准的 chat trace:`load_skill`、文件系统扫描
（`glob` / `grep` / `read_file`）、数据库元数据调用（`list_tables`、
`describe_table`、`read_query`）、并行的 `explore` 子代理 `task` 调用,
然后在 Generation Manifest 处暂停。确认后,生成类 `task` 分批执行,
最终以写入 `AGENTS.md` 的 `filesystem_tools.write_file` 收尾。

---

## 前置条件

- 已配置 LLM。如未配置先运行 `/model`——`/init` 需要 agent 来驱动每一步。
- `~/.datus/conf/agent.yml` 非空。先用 `/datasource` 添加数据源,这样它们
  才会出现在 skill 输出的"Services"表里。

需要换数据源时,先用 `/datasource <name>` 切换,再运行 `/init`。

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

参见:[`/model`](model_command.zh.md)、[斜杠命令参考中的 `/datasource`](reference.zh.md)、[Skills 集成](../integration/skills.zh.md)。
