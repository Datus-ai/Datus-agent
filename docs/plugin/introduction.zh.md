# 插件

插件可以在不修改 Datus 本身的情况下，接入外部服务或增加一组专用命令。根据插件
提供的内容，你可以获得：

| 能力 | 作用 |
|---|---|
| CLI 命令 | 通过 `datus <plugin> ...` 操作外部服务 |
| Skills | 像使用项目和用户 skills 一样使用插件自带的 skills |
| Agent 上下文 | 让 agent 知道插件及其已配置的环境 |
| 权限规则 | Agent 执行敏感命令前先征得你的确认 |
| 工具保护 | 在指定的 agent 工具执行前检查、修改或拒绝调用 |

每个插件都是独立安装的 Python 包。安装完成后，Datus 会自动发现它，不需要额外注册。

如果你想开发自己的插件，请阅读[开发插件](development.zh.md)。

## 安装插件

安装已发布的插件时，直接传入包名即可：

```bash
datus plugin install datus-airflow-plugin
datus airflow --help
```

也可以明确指定安装来源：

```bash
datus plugin install pip:datus-airflow-plugin
datus plugin install src:./datus-airflow-plugin
datus plugin install whl:./dist/datus_airflow_plugin-0.3.0-py3-none-any.whl
datus plugin install git:https://github.com/acme/datus-airflow-plugin
datus plugin install zip:./dist/datus-airflow-plugin-0.3.0.zip
```

支持的来源包括：

| 来源 | 适用场景 |
|---|---|
| `pip:` | 从 Python 包索引安装。省略前缀时默认使用这种方式。 |
| `src:` | 安装本地插件项目。 |
| `whl:` | 安装已经构建好的 wheel 文件。 |
| `git:` | 从 Git 仓库安装。 |
| `zip:` | 安装由 `datus plugin pack` 或 `datus plugin export` 生成的包。 |

Datus 会把插件及其依赖安装到 `~/.datus/plugins/<name>/`。如果同名插件已经存在，
可以添加 `--force` 进行替换。

如果执行 `datus <name>` 后进入了聊天 REPL，而不是插件 CLI，请检查插件是否已经安装，
以及它是否在[当前项目中启用](#activating-plugins)。

## 配置插件

插件配置位于 `agent.yml` 的 `agent.plugins.<name>` 下。每个子项都是一个命名的
**配置环境（profile）**，通常对应生产、测试等不同环境：

```yaml
agent:
  plugins:
    airflow:
      prod:
        default: true
        base_url: https://airflow.example.com
        token: ${AIRFLOW_TOKEN}
      staging:
        base_url: https://airflow-staging.example.com
        token: ${AIRFLOW_STAGING_TOKEN}
```

凭据应使用 `${ENV_VAR}` 引用，不要直接写入配置文件。运行插件前，Datus 会读取对应的
环境变量。

Datus 按以下顺序查找配置文件：

1. 通过 `--config` 显式指定的文件。
2. 当前项目下的 `./conf/agent.yml`。
3. 用户目录下的 `~/.datus/conf/agent.yml`。

配置修改会在下次运行插件时生效，不需要重启。

### 托管 API 部署

在多租户 `datus-api` 部署中，`AuthProvider` 可以为每个请求提供插件配置，而不必
写入 `agent.yml`。插件和 skill 只使用当前请求的配置，不会读取其他租户的设置，也
不会回退到服务器上的本地项目配置。

这种模式仍支持 `--profile`，但会拒绝 `--config`，因为 `AuthProvider` 提供的配置
才是权威来源。当 agent 通过 Bash 运行插件时，每条命令应只包含一次直接的
`datus <name>` 调用。管道和重定向可以正常使用，但命令替换以及 `timeout`、`env`、
`xargs`、`sh -c` 等包装方式会被拒绝。

许多插件还会提供 `<name>-setup` skill。你可以让 agent 帮忙配置插件，它会收集必要
信息并创建 profile。

### 选择 profile

当插件配置了多个 profile 时，Datus 按以下顺序选择：

1. 命令行中的 `--profile <name>`。
2. 当前项目在 `./.datus/config.yml` 中选定的 profile。
3. 标记了 `default: true` 的 profile。
4. 仅有的一个 profile。

不需要配置的插件可以直接以空 profile 运行。如果存在多个 profile，但无法确定使用
哪一个，Datus 会提示你传入 `--profile`。

例如：

```bash
datus airflow --profile staging dags list
```

如果希望某个项目默认使用指定 profile，可以在 `./.datus/config.yml` 中设置：

```yaml
plugins:
  airflow:
    enabled: true
    active_profile: [staging]
```

## 管理插件

在终端中使用 `datus plugin`：

| 命令 | 作用 |
|---|---|
| `datus plugin install <source>` | 安装插件。添加 `--force` 可替换已有安装。 |
| `datus plugin list` | 查看已安装插件、版本、来源、profile 和项目启用状态。 |
| `datus plugin info <name>` | 查看一个插件的详细信息。 |
| `datus plugin upgrade <name>` | 按记录的 `pip:`、`git:` 或 `src:` 来源重新安装插件。 |
| `datus plugin uninstall <name>` | 卸载插件。 |
| `datus plugin enable <name>` | 在当前项目中启用插件。 |
| `datus plugin disable <name>` | 在当前项目中停用插件。 |
| `datus plugin pack [directory]` | 从插件项目生成可分发的 `.zip`。 |
| `datus plugin export <name>` | 将已安装插件导出为 `.zip`。 |

在聊天 REPL 中，`/plugins` 会打开交互式管理界面。你可以浏览插件、编辑 profile、
以环境变量引用的形式填写凭据，以及选择当前项目启用的插件和 profile。

## 在项目中启用插件 {#activating-plugins}

默认情况下，所有项目都可以使用已安装的插件。如果某个项目只应启用一部分插件，
可以在 `./.datus/config.yml` 中添加 `plugins:`：

```yaml
plugins:
  airflow:
    enabled: true
    active_profile: [staging]
  internal-admin:
    enabled: false
```

一旦添加这个配置段，它就会成为当前项目的插件列表。没有列出的插件，以及标记为
`enabled: false` 的插件，都不会在该项目中加载。对应的 CLI 命令、skills、agent
上下文、权限规则和工具保护也会一并停用。

也可以通过命令修改同一设置：

```bash
datus plugin enable airflow
datus plugin enable airflow --profile staging
datus plugin disable internal-admin
```

这是项目级设置，与[全局插件开关](#disabling-the-plugin-system)相互独立。

## 与 agent 配合使用

你既可以在终端中直接运行插件，也可以让 agent 在任务中使用它：

- 插件自带的 skills 会出现在 `/skill list` 中。
- 配置完成后，插件可以把可用环境告诉 agent，让 agent 判断何时应该使用它。
- Setup skill 可以引导你创建 profile，同时避免在配置中写入明文凭据。
- 在聊天 REPL 中，`!<plugin> ...` 可以直接运行插件命令，并把结果带回当前对话。
  详见[工具和插件命令](../cli/execution_command.zh.md)。

插件上下文会在会话开始时生成。如果你在会话中修改了 profile，请新建会话，让 agent
读取更新后的环境信息。

在不允许 agent 修改 `agent.yml` 的托管环境中，对应的 setup skill 不会显示。此时需要由
管理员完成配置。

## 权限

Agent 运行的插件命令会经过 Datus 权限系统。插件可以将自己的命令标记为：

- `allow`：agent 可以直接运行。
- `ask`：运行前需要用户确认。
- `deny`：agent 不得运行。

这些规则只约束 agent。你在终端中直接输入的命令不受影响。

插件只能为自己的 `datus <name> ...` 命令声明规则，不能控制其他插件或无关的 shell
命令。你在 `agent.yml` 中设置的规则始终优先，插件不能覆盖用户设置的 `deny`。

如果在确认 `ask` 命令时选择 **allow (project)**，Datus 会把本次匹配到的命令规则
保存到当前项目的 `.datus/config.yml`。

## 离线安装 {#offline-install}

先在能够访问网络的机器上生成安装包：

```bash
# 在插件项目目录中执行
datus plugin pack -o ./dist
datus plugin pack --with-deps -o ./dist
```

默认生成的安装包只包含插件 wheel，目标机器仍需访问 Python 包索引才能安装依赖。

添加 `--with-deps` 后，安装包会包含插件和全部依赖 wheel，可用于完全离线安装。如果
依赖中包含原生代码，请使用与目标环境相同的操作系统和 Python 版本构建。

安装生成的文件：

```bash
datus plugin install zip:./dist/datus-airflow-plugin-0.3.0.zip
```

也可以导出已经安装的插件：

```bash
datus plugin export airflow -o ./dist
```

## 从其他目录加载插件 {#plugin-paths}

通过 `agent.plugin_paths`，可以加载不在 `~/.datus/plugins/` 中管理的插件：

```yaml
agent:
  plugin_paths:
    - /opt/shared/datus-plugins/airflow
    - $DATUS_PLUGIN_HOME/internal
```

每一项都必须指向一个已经安装好的插件目录，而不是包含多个插件的父目录。这种方式
适合多个项目或机器共享集中部署的插件。如果外部目录与 `~/.datus/plugins/` 中存在
同名插件，Datus 会优先使用后者。

## 关闭插件系统 {#disabling-the-plugin-system}

在 `agent.yml` 中设置全局开关：

```yaml
agent:
  plugins_enabled: false
```

关闭后，插件命令、自带 skills、agent 上下文、权限规则和工具保护都会停用。默认值为
`true`。

## 下一步

- 使用 `datus-plugin-development` skill [开发插件](development.zh.md)。
- 了解 [skills](../skills/introduction.zh.md) 的发现和使用方式。
