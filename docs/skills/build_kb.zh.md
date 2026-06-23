# Build KB

`build-kb` 构建项目的向量索引知识库——语义模型、指标和参考 SQL——让你可以对数据进行语义检索。它是 [`init`](init.md) 的重量级对应步骤。

在 REPL 中使用 `/build-kb` 命令运行。

## 功能作用

- 扫描已配置的数据源（表、schema、关系）。
- 提出一份**生成清单（manifest）**——说明它计划构建什么——并等待你确认或调整。
- 确认后，生成并索引每类产物：语义模型、指标和参考 SQL。
- 刷新 `AGENTS.md` 的知识库索引部分。

它属于**重量级**层级：比 `init` 慢（数分钟），并且默认会在生成前请你确认一份清单。

## 何时使用

- 你想对数据进行语义检索（语义模型、指标、参考 SQL）。
- [`/init`](init.md) 的轻量清单不够用。
- 你想构建或重建知识库。

## 如何使用

```text
/build-kb
```

你可以附加自由文本提示，把生成范围聚焦到特定的文件、表或业务域：

```text
/build-kb only the orders and customers tables
```

一次典型的运行如下：

1. `build-kb` 扫描你的数据源，并**提出一份生成清单**，列出将要生成的内容。
2. 你审阅清单并确认或调整。
3. 它生成并索引已确认的产物，然后刷新 `AGENTS.md` 索引。
4. 它汇报生成并索引了什么（按产物类型给出数量）。

不带提示时，它会提出一份覆盖主要数据源的清单。重新运行会更新已有产物，而不是重复生成。

## 示例

`/build-kb` 后面的文本会作为指令转发给 agent，因此你可以用自然语言来调整这次运行。

### 限制范围

把扫描和生成限定到特定的表、文件或业务域，而不是全部数据源：

```text
/build-kb only the orders and order_items tables
/build-kb the sales domain, plus queries/*.sql
```

### 选择生成内容

只构建部分产物类型。这里生成语义模型和指标，但不生成参考 SQL：

```text
/build-kb semantic models and metrics only, skip reference SQL
```

### 跳过确认 { #skip-the-confirmation }

跳过清单审阅、直接开始生成——当你已经清楚范围、不需要调整计划时很方便：

```text
/build-kb the orders table, skip the manifest confirmation and generate directly
```

这些可以组合使用——例如*“only the orders domain, semantic models only, skip confirmation”*。

## Build KB 与 Init 的区别

先运行 [`/init`](init.md) 获得即时的轻量清单，再运行 `/build-kb` 构建向量索引知识库。完整对比见 [Init](init.md#init-vs-build-kb)。

## 注意事项

- 生成只覆盖已配置的数据源。
- 默认会执行清单确认，让你在任何工作开始前决定生成什么；你也可以让它跳过该确认（见[示例](#skip-the-confirmation)）。
