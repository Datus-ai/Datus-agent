# 语义层配置（Semantic Layer）

语义适配器统一配置在 `agent.services.semantic_layer` 下。**至少要配置一条**——空 section 不再静默 fallback 到 metricflow,而是 raise `No semantic layer configured`。

> **迁移说明**:旧版本会在 section 缺失时隐式注入 metricflow 默认值,新版要求显式写入 yaml 条目。`conf/agent.yml.example` 已经默认带上 `metricflow: {}`,新装用户开箱仍能用。

## 配置结构

```yaml
agent:
  services:
    semantic_layer:
      metricflow:
        timeout: 300
        config_path: ./conf/agent.yml   # 可选的高级覆盖项
        default: true                   # 全局默认:无 project pin 时被选用

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow

    gen_metrics:
      semantic_adapter: metricflow
```

## 选择规则

`AgentConfig.resolve_semantic_adapter` 解析活动语义适配器的顺序与 BI Dashboard / Scheduler 完全一致:

1. 调用处显式传入的 `adapter_type`(或 agentic node 上的 `semantic_adapter`)。
2. `./.datus/config.yml` 中的项目级 pin —— `semantic:` 字段。
3. YAML 全局 `default: true` 标志:`services.semantic_layer` 中至多一条可标 default,多于一条会在加载阶段直接报错。
4. 单条快捷:仅有一条 semantic adapter 时,自动使用它。
5. 否则抛错:
   - section 为空 → `No semantic layer configured ...`
   - 多条无 default → `Multiple semantic layers are configured ...`

`services.semantic_layer` 下的 key **必须等于 adapter type**(例如 `metricflow`)。如果同时写了 `type:` 字段,其值必须与 key 一致,否则 Datus 会在启动时抛出配置错误。比较时会先 lowercase + trim,因此 `MetricFlow`、` metricflow ` 都会被视为与 `metricflow` 匹配。

## MetricFlow 说明

- `config_path` 是可选项。
- Datus 默认会基于当前 `services.datasources` 中选中的数据源和项目语义模型目录自动构建运行时配置。
- MetricFlow 验证会直接读取配置中的项目语义模型目录，包括位于 gitignore 项目路径下的生成 YAML。
- 仅当你需要 MetricFlow 直接读取某个指定的 `agent.yml` 时才需要设置 `config_path`。

## 通过 CLI 配置（`/services`）

在 Datus REPL 中运行 `/services semantic`（或者从其他 tab 按 `Tab` 切过来）会进入配置 TUI 的 **Semantic** tab。该 tab 支持：

- 在尾部的 `+ Add new semantic` 行按 `Enter` 新增一个语义层。目前仅支持 `metricflow`（`datus-semantic-metricflow`），并且**无需任何参数** —— 在 type picker 中选中它即可。如果适配器包尚未安装，Datus 会自动执行 `pip install datus-semantic-metricflow` 并热加载注册表，无需重启进程。
- 用 `x` 删除条目；用 `t` 触发一次注册探测。
- `d` 切换**全局** `default: true`:按 `d` 把光标项设为默认,并自动清掉其他条目的 default。
- `p` 设置**项目级** default:值写入 `./.datus/config.yml` 的 `semantic: <name>`,只对当前项目生效,优先级高于全局标记。在已 pin 的行上再按一次 `p` 清除。
- 此 tab 不显示 `e edit`:metricflow 当前没有可编辑字段。

新建条目会写入 `~/.datus/conf/agent.yml`，形态为 `services.semantic_layer.<type>: {type: <type>}`。

首次进入交互式 REPL 时,Datus 会跑一遍 bootstrap:若尚无项目级 pin,而 YAML 中能解析出明确的默认值(单条快捷或唯一标 `default: true`),Datus 会自动写入项目级 pin。若多条都未标 default,启动时会弹出一个轻量选择器。CI / Docker 等无人值守环境可设置 `DATUS_DISABLE_SERVICE_BOOTSTRAP=1` 关闭。
