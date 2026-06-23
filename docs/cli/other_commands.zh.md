# Other Commands

本页汇总其余的 CLI 命令——运行时配置以及数据源 / 服务设置——它们没有各自的独立页面。

## 配置命令

这些斜杠命令在 REPL 内调整运行时行为。不带参数运行时会打开交互式选择器，也可接受快捷参数。

### `/model`

无需编辑 YAML 即可切换活跃的 LLM 提供商和模型。

```text
/model                       # 打开交互式选择器
/model openai/gpt-4.1        # 直接切换到某个 provider/model
/model openai                # 打开选择器并定位到某个 provider
```

选择器把选项分为 **Providers**、**Plans** 和 **Custom**（来自 `agent.models` 的自托管模型）。提供商凭据存放在 `agent.yml` 中；`/model` 只切换活跃选择，该选择会持久化到 `./.datus/config.yml` 的 `target` 下：

```yaml
target:
  provider: openai
  model: gpt-4.1
```

切换在下一次查询时生效——无需重启。

### `/effort`

控制 LLM 的推理努力级别。

```text
/effort                      # 打开选择器
/effort high                 # 直接设置
```

| 级别 | 行为 |
|------|------|
| `minimal` | 推理最少，最快 |
| `low` | 较少推理 |
| `medium` | 平衡（默认） |
| `high` | 最深入，最慢 |

更高的努力级别消耗更多 token 且耗时更长。并非所有提供商 / 模型都支持努力级别；不支持的模型会忽略此设置。该选择在会话期间持久化。

### `/language`

设置助手回复所使用的语言。

```text
/language                    # 打开选择器
/language zh                 # 直接设置
```

它仅影响助手的自然语言回复，不影响 SQL 或代码。该设置在会话期间持久化。

## 设置与服务命令

Datus 在 REPL 内通过斜杠命令完成配置。

### `/datasource`

添加、编辑、删除或切换数据源（DuckDB、SQLite、Snowflake、MySQL、PostgreSQL、StarRocks 等）。改动会写入 `agent.yml` 的 `services.datasources` 下。

### `/init` 和 `/build-kb`

`/init` 和 `/build-kb` 用于初始化项目工作区——它们委托给内置 skill。`/init` 做一次快速的轻量扫描；`/build-kb` 构建向量索引知识库。详见 [Init](../skills/init.md) 和 [Build KB](../skills/build_kb.md)。

### `datus upgrade`

一次性把 `datus-agent` 及所有已安装的 `datus-*` 适配器包升级到最新版本。可编辑 / 源码安装会被跳过。加 `--check` 可只报告最新版本而不安装。

```bash
datus upgrade
datus upgrade --check
```

在交互式启动时，若有更新版本，Datus 还会打印一行提示。设置 `DATUS_DISABLE_VERSION_CHECK=1` 可静默它。
