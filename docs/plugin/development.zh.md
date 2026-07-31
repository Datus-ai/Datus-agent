# 开发插件

推荐使用
[`datus-plugin-development`](https://github.com/Datus-ai/Datus-Plugins/blob/main/skills/datus-plugin-development/SKILL.md)
skill 开发 Datus 插件。这个 skill 来自
[`Datus-Plugins`](https://github.com/Datus-ai/Datus-Plugins) 项目。

你只需要提供想要封装的 SDK、API 规范或产品文档。skill 会帮助你确定命令范围、
生成插件，并验证最终结果。插件契约已经包含在 skill 中，因此本文只介绍开发流程，
不再重复容易过时的实现细节。

## 开始前的准备

请准备以下信息：

- 想要接入 Datus 的 SDK、REST API、OpenAPI 规范、CLI 文档或 README。可以提供
  本地路径，也可以提供公开 URL。
- 一个简短的命令名。例如，Airflow 插件对应 `datus airflow`。
- 对功能范围的要求，例如优先支持哪些操作，或明确排除哪些管理类功能。

你只需要提供原始资料，并说明希望插件完成什么。具体实现由开发 skill 处理。

## 安装开发 skill

### Codex

先添加 Datus 插件市场：

```bash
codex plugin marketplace add Datus-ai/Datus-Plugins
```

然后打开 `/plugins`，安装 **datus-plugin-development**，并新建一个会话。

### Claude Code

添加插件市场并安装插件：

```text
/plugin marketplace add Datus-ai/Datus-Plugins
/plugin install datus-plugin-development@datus-plugin
```

## 创建插件

调用 skill 时，传入文档位置和期望的命令名。

在 Codex 中：

```text
$datus-plugin-development ./docs/openapi.yaml，命令名使用 acme
```

在 Claude Code 中：

```text
/datus-plugin-development ./docs/openapi.yaml，命令名使用 acme
```

文档参数也可以是 URL，或一个包含多份相关文档的目录。如果你对功能范围、兼容性
或权限有额外要求，可以在同一条请求中说明。

## 开发流程

这个 skill 会按以下流程工作：

1. **阅读原始文档。** 梳理可用操作、认证方式、配置项和依赖。
2. **给出设计草案。** 草案会列出建议的命令、配置字段、权限行为、随插件提供的
   skills，以及暂不纳入的功能。
3. **等待你确认范围。** 这一步不会写代码。你可以确认方案、提出修改，或将被排除的
   功能重新加入范围。
4. **实现插件。** 方案确认后，skill 才会生成可运行的插件、配套 skills 和测试。
5. **验证结果。** skill 会运行相关测试，并检查本地安装、CLI 分发、skill 加载和
   打包是否正常。

先确认设计、再开始实现，可以避免把一个庞大的厂商 API 原样复制成同样庞大的 CLI，
同时确保最终范围仍由你决定。

## 试用生成的插件

实现完成后，可以安装本地项目并查看 CLI：

```bash
datus plugin install src:./datus-acme-plugin
datus acme --help
```

需要分发时，可以生成安装包：

```bash
datus plugin pack ./datus-acme-plugin -o ./dist
```

如果目标环境无法访问 package index，可以添加 `--with-deps`。两种安装包的区别见
[离线安装](introduction.zh.md#offline-install)。

## 延伸阅读

- [Development skill 源文件](https://github.com/Datus-ai/Datus-Plugins/blob/main/skills/datus-plugin-development/SKILL.md)：
  最新的开发流程和插件契约。
- [Datus-Plugins README](https://github.com/Datus-ai/Datus-Plugins#develop-your-own-plugin)：
  插件市场安装方式和项目示例。
- [Datus Plugins 贡献指南](https://github.com/Datus-ai/Datus-Plugins/blob/main/CONTRIBUTING.md)：
  工作区配置、测试、Pull Request 和发布流程。
- [插件介绍](introduction.zh.md)：面向使用者的安装、配置和管理说明。
