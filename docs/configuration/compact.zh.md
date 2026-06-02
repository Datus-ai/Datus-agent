# 上下文压缩（Context Compaction）

随着对话变长，历史最终会逼近模型的上下文窗口上限。Datus 通过 `agent.compact` 下的两种互补压缩自动管理上下文，让你在一个长会话里持续工作，而无需手动清空、也不会撞上 token 上限。

| | Minor compact | Major compact |
|---|---|---|
| 做什么 | 把较早的工具 I/O 归档到磁盘 | 把整个会话总结成摘要 |
| 驱动方式 | 规则（user 轮次计数） | LLM（一次摘要调用） |
| 执行方式 | 同步，但很快（本地、无 LLM 调用） | 同步、阻塞运行循环 |
| 是否触及最近轮次 | 否 | 是 —— 替换全部历史 |
| 可恢复 | 是（归档文件） | 是（全历史 JSONL） |

## Minor compact

**触发条件** —— 在每个 user 轮次开始时，如果会话的 user 轮次超过 `keep_recent_user_turns`（默认 4），就会运行 minor compact。它的判据——user 轮次计数——只在轮次之间变化，因此每个轮次只判断一次，而不是每次工具调用之后都判断（那种逐次工具调用的检查留给 major，因为只有它的 token 占用会在轮次中途增长）。它是同步的，但由于只是纯本地的、基于规则的归档（无 LLM 调用），很快就能完成，几乎不会拖慢 agent。

**做什么** —— 对于保留窗口之外的*较早*轮次，任何长度超过 `archive_threshold` 字符（默认 1000）的工具调用参数或输出都会被移出实时对话并写入磁盘归档；原处留下一段简短的内联预览（`archive_preview_chars`，默认 1000；错误输出为 2×）并带上 `[DATUS_ARCHIVED]` 标记。

**产生的行为**

- 最近 `keep_recent_user_turns` 个轮次的完整工具 I/O 保持不变 —— 对话的活跃部分永远不会被削弱。
- 较早的大体积输出会缩成「预览 + 指针」；当模型需要细节时，仍可通过 `read_file` 读取归档恢复完整内容。
- 由于它只归档（不做摘要），不会丢失任何信息，也不消耗 LLM 调用。开销低、速度快、触发频繁。

## Major compact

**触发条件** —— 当最近一次模型调用的输入 token 达到上下文窗口的 `token_threshold`（默认 `0.9`，即 90%）时，会强制执行 major compact。该检查在每个 user 轮次开始时以及每次工具调用之后进行，因此它可以在**轮次中途**触发 —— 正好在上下文即将溢出时，而不必等到下一轮。`/compact` 可随时手动触发。

**做什么** —— 由模型把**整个**会话总结成一段摘要；随后清空会话历史，并用该摘要作为新的起点继续对话。压缩前的完整历史会被转存为一个 JSONL 文件，并在摘要末尾附上指向它的指针，使 agent 能通过 `read_file` 找回任意具体细节。

**产生的行为**

- 它是**同步、阻塞**的：运行循环会等待摘要写入后才继续，因为下一次模型调用必须看到压缩后的历史，而不是已超限的历史。
- 可见对话被折叠 —— 较早的轮次被摘要替换。在 CLI 中会先显示 `Compacting context…` 提示，随后展示摘要面板；在 API/print 流中则作为一条 `compact_summary` markdown 消息到达。
- 以一定的细节换取空间：摘要是精炼的，因此细粒度内容此时只存在于 JSONL 转存中（仍可通过 `read_file` 获取）。
- 一次 major compact 会额外消耗一次 LLM 调用（用于摘要），因此较为少见，只在接近上下文上限时触发。

## 自动 vs. 手动

- **自动** —— 两种压缩在会话过程中自行运行，你无需做任何事。major 在接近上下文上限时触发，minor 随轮次增长触发。
- **手动 `/compact`** —— 无论当前用量如何，都会立即执行一次 **major**。适合在开始一项较大的新任务前使用，获得一个干净、已摘要的起点。

## 参数

| 配置项 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `compact.major.enabled` | bool | `true` | 启用 LLM 驱动的全历史摘要。 |
| `compact.major.token_threshold` | float | `0.9` | 上下文窗口占用比例超过该值时强制 major（覆盖自动调度中的 minor 选择）。 |
| `compact.minor.enabled` | bool | `true` | 启用规则驱动的归档压缩。 |
| `compact.minor.keep_recent_user_turns` | int | `4` | 保留最近 N 个 user 轮次的原始工具 I/O；更早的轮次可被归档。 |
| `compact.minor.archive_threshold` | int | `1000` | 工具调用参数/输出超过该字符数时转存到磁盘。 |
| `compact.minor.archive_preview_chars` | int | `1000` | 归档标记中保留的内联预览长度（错误输出为 2×）。 |

```yaml title="agent.yml"
agent:
  compact:
    major:
      enabled: true
      token_threshold: 0.9
    minor:
      enabled: true
      keep_recent_user_turns: 4
      archive_threshold: 1000
      archive_preview_chars: 1000
```

如需完全关闭自动压缩，将 `major.enabled` 与 `minor.enabled` 都设为 `false`。即使 `major.enabled` 为 `false`，仍可手动运行 `/compact`。

## 注意事项

- Major compact 的触发读取当前轮次的**实时** token 用量（与 CLI 状态栏显示的是同一数值）。在全新会话或刚 `resume` 后，该信号从零开始，因此在至少有一次模型调用上报用量之前，第一轮不会触发 major compact。
- Minor compact 依据会话的 user 轮次计数判断是否触发，因此在 resume 之后仍能正确工作。
- 归档的工具 I/O 与全历史 JSONL 存放在会话的数据目录下；路径见 [存储（Storage）](storage.md)。
