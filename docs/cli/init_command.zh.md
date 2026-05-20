# Init 命令 `/init`

## 概览

`/init` 是一个轻量级快捷入口——它把"按 `init` skill 的步骤为当前项目生成
`AGENTS.md`"作为一句话发给 chat agent。skill 本身会带着 agent 走完整流程：

1. 用 `ask_user` 询问项目目标
2. 用 `filesystem_tools` 扫描目录结构
3. 读取 `README.md`（若存在)
4. 询问要纳入哪些已配置服务,并支持用户描述未配置的额外服务
5. 对每个选中数据源的表清单做"按类别归档"（不会枚举所有表)
6. 生成 markdown 并写入 `./AGENTS.md`（已存在时会先确认是否覆盖）

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
```

`/init` 不接受任何参数。skill 中传给 `db_tools.list_tables` 的数据源是
当前 REPL 选中的那一个（启动时通过 `--datasource` 指定,或运行时用
`/datasource` 切换）。

执行时你会看到标准的 chat trace:`load_skill`、若干 `ask_user` 交互、
若干 `filesystem_tools.*` 调用、`db_tools.list_tables` 调用,最后是
`filesystem_tools.write_file`。

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
