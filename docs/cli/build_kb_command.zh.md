# Build KB 命令 `/build-kb`

## 概览

`/build-kb` 是一个快捷入口——它让 chat agent 按内置 **`build-kb` skill**
构建项目的**向量索引知识库**——`semantic_models`、`metrics` 和
`reference_sql`（由 LanceDB 支撑的存储）。它是轻量化
[`/init`](init_command.zh.md) 的重量级搭档:`/init` 写出 `AGENTS.md` 清单
加上廉价的文件类存储（`knowledge` / `memory`),而 `/build-kb` 负责跑昂贵
的生成并随后刷新 `AGENTS.md` 的 KB 索引。

skill 会带着 agent 走以下流程：

1. **解析范围**——从你传入的任意自由文本提示中解析出具体的文件
   （glob/路径）、库表、数据源或业务域。不带提示时覆盖整个项目。
   若 `AGENTS.md` 已存在,则复用其清单而非重新全量扫描。
2. **扫描并分类**范围内的文件与数据库元数据到业务域。validated-query
   语料会逐对枚举（绝不抽样),确保每个 `(question, SQL)` 样本都被索引。
3. 用只读 `explore` 子代理**并行探索**每个业务域（并发上限 3）。
4. 输出 **Generation Manifest**——解析出的范围,以及每个待生成产物一行
   ——然后**停下等待确认**。回复确认,或修正范围/任意一行。
5. 确认后按 `storage-classify` skill 把每个产物**路由**到对应存储:
   semantic models、metrics、reference SQL,各由匹配的生成子代理产出
   （每条 SQL 一次 `gen_sql_summary` 调用）。每个 `(question, SQL)` 对
   会**双路由**——既进 `reference_sql`（样本）,又经
   `extract-knowledge` 挖出背后的规则。
6. **刷新 `AGENTS.md` 的 KB 索引**——用 count + 检索工具更新
   `## Semantic Models` / `## Metrics` / `## Reference SQL`,并向
   `## Knowledge` 追加新挖出的条目。它不会重写 `/init` 拥有的清单段。
   若 `AGENTS.md` 缺失,则写出完整文件。

由于 `/build-kb` 走的是标准 chat 流水线,你会获得熟悉的 UX——
`ActionHistory` 事件流式渲染、**Ctrl+O** 切换 trace 详情、**ESC** 中断、
以及多轮微调。

skill 源文件位于 `datus/resources/skills/build-kb/SKILL.md`,运行时直接从
已安装的包内加载。可在 `./.datus/skills/build-kb/SKILL.md`（项目级,优先级
最高）或 `~/.datus/skills/build-kb/SKILL.md`（用户级）放同名 skill 覆盖。

---

## 基本用法

```text
> /build-kb
> /build-kb orders 和 order_items 表,以及 queries/*.sql,只覆盖 sales 域
> /build-kb 只处理 analytics 数据源
```

`/build-kb` 接受一段可选的自由文本范围。命令后的内容会原样转发给 skill,
由它解析成范围内的文件 / 库表 / 数据源 / 业务域。不带参数时覆盖整个项目,
但限定在**当前激活的数据源**——配置了多个时只构建这一个,除非提示中显式
点名了其他数据源。要换目标请先用 `/datasource <name>` 切换。

执行时你会看到:`load_skill`、限定范围的文件系统与数据库扫描、并行的
`explore` 子代理 `task` 调用、在 Generation Manifest 处暂停,确认后是分批
的生成 `task` 调用（`gen_semantic_model`、`gen_metrics`、`gen_sql_summary`、
`extract-knowledge`),最终以对 `AGENTS.md` 的 `edit_file` 收尾。

---

## 前置条件

- 已配置 LLM。如未配置先运行 `/model`。
- `~/.datus/conf/agent.yml` 非空,且已通过 `/datasource` 配置数据源。
- 推荐:先运行 [`/init`](init_command.zh.md),让 `AGENTS.md` 与初始的
  knowledge/memory 就位;`/build-kb` 随后复用该清单,只补充向量存储。
  它也可独立运行（缺 `AGENTS.md` 时会自行创建）。

---

## 与 `/init` 和 `/bootstrap` 的关系

- **对比 [`/init`](init_command.zh.md)**——`/init` 是轻量、无确认门的一轮
  （清单 + 文件类 knowledge/memory）。`/build-kb` 是重量、带确认门的一轮,
  构建检索型存储。典型流程:先 `/init`,再 `/build-kb`（可限定范围）。
- **对比 `/bootstrap`**——`/bootstrap` 是另一条独立的确定性 Python TUI
  流程,用于 bootstrap KB（schema / sql / semantic / metrics / knowledge）。
  `/build-kb` 是其 agentic、skill 驱动的等价物,能理解自由文本范围提示,
  并经 `storage-classify` 路由。想用引导式 TUI 选 `/bootstrap`;想让 agent
  自行划定范围并推理生成什么,选 `/build-kb`。

---

## 自定义输出形态

skill 是唯一信息源。想改行为,只需修改:

- **项目级覆盖**:`./.datus/skills/build-kb/SKILL.md`
- **用户级覆盖**:`~/.datus/skills/build-kb/SKILL.md`
- **内置兜底**:`datus/resources/skills/build-kb/SKILL.md`

项目级 → 用户级 → 内置,同名 skill 按此顺序逐级遮蔽。

参见:[`/init`](init_command.zh.md)、[`/model`](model_command.zh.md)、[斜杠命令参考中的 `/datasource`](reference.zh.md)、[Skills 集成](../integration/skills.zh.md)。
