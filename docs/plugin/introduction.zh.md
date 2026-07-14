# Plugin 介绍

**plugin**(插件)是一个可安装的 Python 包,在不修改 Datus 本身的前提下对其进行扩展。
`datus plugin install` 会把每个插件装到独立目录 `~/.datus/plugins/{name}/`(依赖一并
vendored 进目录内),根据插件打包的内容,你可以获得:

| 功能面 | 提供什么 |
|---|---|
| CLI 子命令 | `datus <plugin> ...` 运行插件自己的命令行界面 |
| Skills | 插件自带的 skill 出现在 `/skill list` 中,与项目、用户 skill 并列 |
| Agent 感知 | 插件在 agent 的 system prompt 中描述自己和已配置的环境,模型会主动选用它 |
| Bash 权限 | 插件预先声明哪些子命令 agent 可以直接执行、哪些需要确认 |
| 工具 transformer | 插件可以在 agent 的工具调用执行前改写参数或拒绝调用(如强制 SQL 作用域策略) |

每个已启用插件的目录会在启动时追加到 `sys.path`,其 `datus.plugins` entry-point 因此
在每次调用时被发现——无需重启,也没有任何注册步骤。(用旧方式直接 `pip install` 装进
同一 Python 环境的插件仍会作为兜底被发现。)

想开发自己的插件?见[开发指南](development.zh.md)。

## 安装插件

`datus plugin install` 接收 `{type}:{src}` 形式的安装源,把插件装到
`~/.datus/plugins/{name}/`。类型前缀**必填**:

```bash
datus plugin install pip:datus-plugin-hello                       # PyPI 包名
datus plugin install src:./datus-plugin-hello                     # 本地项目目录
datus plugin install whl:./dist/hello-1.0-py3-none-any.whl        # 本地 wheel 文件
datus plugin install git:https://github.com/acme/datus-plugin-hello   # git 仓库
datus plugin install zip:./dist/datus-plugin-hello-1.0.0.zip      # 离线 bundle(见下)

datus hello Ada          # 子命令立即可用
```

`pip`、`src`、`whl`、`git` 会从 package index 解析依赖(需要网络);插件及其依赖通过
`pip install --target` 一并 vendored 进插件目录。`datus plugin install` 会把安装方式
记录到 `~/.datus/plugins/{name}/datus-plugin.json`,以便之后
`datus plugin upgrade <name>` 用同样的方式重新拉取。若插件已安装,传 `--force` 可替换。

如果 `datus <name>` 落进了 REPL 而不是运行插件,说明它未安装,或未在本项目激活
(见[激活插件](#activating-plugins))。

### 离线安装(内网/气隙环境) {#offline-install}

离线 **bundle** 就是一个普通 `.zip`,内含一个 wheelhouse(插件 wheel,以及可选的每个
依赖 wheel)。在**有网**的机器上用 `datus plugin pack` 打好包,把文件拷过去,再用
`zip:` 安装:

```bash
# 有网机器,在插件的项目目录下
datus plugin pack --with-deps -o ./dist     # 插件 wheel + 全部依赖 wheel
datus plugin pack -o ./dist                 # 仅插件 wheel(默认)

# 目标机器
datus plugin install zip:./dist/datus-plugin-hello-1.0.0.zip
```

`pack` 默认**仅打插件 wheel**——包体小,但目标机器在安装时要从 index 解析依赖(需要
网络)。加 `--with-deps` 会把每个依赖 wheel 都打进去,从而**完全离线**安装
(`pip install --no-index --find-links`)。安装前会对照 bundle manifest 逐一校验每个
wheel 的 checksum;若 bundle 是为不同的 Python 版本或平台构建的,安装会带明确提示拒绝,
除非传 `--force`(checksum 仍强制校验)。含依赖 bundle 是同平台快照,请在与目标机
OS/Python 匹配的机器上构建。

### 导出已安装插件

`datus plugin export <name>` 能把一个已安装插件重新导出成可分发的 `.zip`——`zip:`
来源的安装会字节级返回其保留的原始 bundle,而 `pip`/`src`/`whl`/`git` 来源则按记录的
安装源重新打包(需要网络)。

## 配置

插件在 `agent.yml` 的 `agent.plugins.<name>` 下配置,`<name>` 之下的每个键是一个
**profile**——一套命名环境(endpoint、凭据、选项)。一个插件可以有任意多个 profile:

```yaml
agent:
  plugins:
    hello:
      prod:
        default: true              # 省略 --profile 时选它(见下)
        greeting: Hi
        token: ${HELLO_TOKEN}      # 密钥优先用 ${ENV_VAR}
      staging:
        greeting: Yo
```

Datus 按以下顺序解析 config 文件:显式 `--config` → `./conf/agent.yml`(项目级)→
`~/.datus/conf/agent.yml`(用户默认)。把 profile 写进你的 datus 会话实际加载的那个文件。

`${VAR}` 引用会按 profile 从环境变量展开——密钥请一律使用它,不要写明文。配置修改对
下一次 `datus <plugin>` 调用即刻生效,无需重启。

有些插件自带 `<name>-setup` skill,可以替你写好这段配置——见
[与 agent 配合使用](#agent)。

### 哪个 profile 生效 {#which-profile-runs}

执行 `datus <name> ...` 时,激活 profile 按以下顺序解析:

1. 命令行显式 `--profile <p>`(`datus hello --profile staging ...`)。
2. 项目 pin —— `./.datus/config.yml`(见下)。
3. 标了 `default: true` 的 profile(超过一个 → 报错)。
4. 唯一 profile(只配了一个时直接用)。
5. 完全没有 `agent.plugins.<name>` 配置段 → 插件以空配置运行(config-free
   插件仍可工作)。
6. 多个 profile 且无法判定 → Datus 报错,提示传 `--profile`。

### 按项目固定 profile

想让某个项目始终使用特定 profile 而不必每次敲 `--profile`,在项目的
`./.datus/config.yml` 里用 `plugins.<name>.active_profile` pin 住它。当只有
一个 profile 激活时,它就成为 `datus <plugin>` 的默认:

```yaml
plugins:
  hello:
    enabled: true
    active_profile:
      - staging
```

## 激活插件 {#activating-plugins}

`./.datus/config.yml` 的 `plugins:` 段还决定**本项目激活哪些已安装插件**。
它可选,但一旦写了即为权威:

- **整段省略** —— 全部已安装插件及其所有 profile 都激活。这是默认。
- **写了它** —— 即成为白名单。每项为
  `{enabled: bool, active_profile: [<profile>, ...]}`。段里未列出(或列出但
  `enabled: false`)的插件在本项目**不加载**:其 `datus <plugin>` 子命令被拒绝,
  自带 skill、system-prompt 段、工具 transformer、bash 规则一律跳过。
  `active_profile` 用于收窄激活的 profile(省略即"全部 profile")。

```yaml
plugins:
  hello:
    enabled: true
    active_profile: [staging]   # 只激活 'staging'
  noisy-plugin:
    enabled: false              # 已安装但本项目关闭
```

不必手改文件也能切换激活:

```bash
datus plugin enable hello                    # 激活(全部 profile)
datus plugin enable hello --profile staging  # 激活并固定到 'staging'
datus plugin disable noisy-plugin            # 本项目停用
```

项目里第一次 `enable`/`disable` 会先用全部已安装插件(均 enabled)播种白名单,
再应用你的改动,这样关掉一个插件绝不会悄悄停用其余插件。

这是**按项目**激活,与全局总开关
[`agent.plugins_enabled`](#disabling-the-plugin-system)(在任何地方关掉整个系统)
是两回事。

## 管理插件

`datus plugin` 是管理入口(即使插件已停用它也始终可用):

| 命令 | 作用 |
|---|---|
| `datus plugin install '{type}:{src}'` | 从 `pip:` / `src:` / `whl:` / `git:` / `zip:` 安装到 `~/.datus/plugins/`(`--force` 替换已有安装)。 |
| `datus plugin pack [dir]` | 从插件项目目录(默认 `./`)构建可分发的 wheelhouse `.zip`;`--with-deps` 打入依赖,`-o` 指定输出目录。 |
| `datus plugin export <name>` | 把已安装插件导出成 `.zip`(`-o` 指定输出目录)。 |
| `datus plugin upgrade <name>` | 按记录的安装源重装(pip/git/src)。 |
| `datus plugin uninstall <name>` | 删除插件目录 `~/.datus/plugins/{name}/`。 |
| `datus plugin list` | 列出已安装插件:包、版本、来源、已配置 profile、本项目激活状态。 |
| `datus plugin info <name>` | 查看单个插件的 profile、配置 schema 与激活状态。 |
| `datus plugin enable/disable <name>` | 切换本项目激活状态。 |

在 REPL 内,`/plugins` 打开交互式管理器完成同样的事:浏览已安装插件,增删改
全局 profile(表单由插件的配置 schema 驱动,密钥以 `${ENV_VAR}` 引用形式输入),
并切换本项目激活哪些插件与 profile。

## 与 agent 配合使用 {#agent}

除了自己在终端执行 `datus <name> ...`,插件还与 agent 深度集成:

- **Skills** —— 插件自带的 skill 出现在 `/skill list`,可以像其他 skill 一样调用。
- **Prompt 感知** —— 已配置的插件会把自己的环境列表写进 agent 的 system prompt,
  模型因此知道插件的存在并主动选用。可以问 agent"配置了哪些 plugin?"来查看它
  掌握的信息。prompt 段落在会话启动时刷新;会话中途的配置修改要到下一个会话才可见。
- **引导式配置** —— 已安装但未配置的插件通常会在 prompt 中声明自己,并指向自带的
  `<name>-setup` skill。让 agent 帮你配置,它会收集必填项并替你写入 profile
  (密钥以 `${VAR}` 形式引用,绝不写明文)。

## Agent bash 权限

当 **agent**(而非你本人)通过它的 bash 工具执行插件 CLI 时,命令会经过 Datus 的
权限层。插件可以按权限 profile(`normal` / `auto`)预先声明:哪些子命令可以直接
放行(`allow`)、哪些必须确认(`ask`)、哪些直接拦截(`deny`)。若无任何声明,
agent 发起的每条插件命令都会弹出确认。

实际含义:

- **插件声明只作用于自己的命名空间。** 插件只能为 `datus <自己的名字> ...` 声明
  规则——碰不到 `rm`、其他插件或任何别的命令。
- **你的规则永远优先。** 你写在 `agent.yml` `permissions.bash_commands` 下的
  `deny` 规则压过插件的任何 `allow`(判定固定为 deny > ask > allow),且插件声明
  永远改不了 profile 的默认姿态。
- **`ask` 可按项目放宽。** agent 命中插件声明的 `ask` 子命令时,确认框会提供
  **allow (project)** 选项——选择后把命中的 pattern 原样持久化到项目
  `.datus/config.yml` 的 `bash_allow` 列表,此后该子命令直接放行。授权绝不会
  扩大到 pattern 之外,插件的 `deny` 规则也不受影响。
- **只有 agent 受门控。** 你自己在终端敲 `datus <name> ...` 不受任何影响。
- `dangerous` 权限 profile 设计上忽略一切命令级 bash 规则,包括插件声明。

## 关闭 plugin 系统 {#disabling-the-plugin-system}

`agent.yml` 中的 `agent.plugins_enabled: false` 是总开关,关闭**全部** plugin
功能——`datus <plugin>` 分发、插件自带 skill、prompt 注入(含 setup 引导)、
权限声明与工具 transformer 一律失效。建议在 API/web 部署中关闭,避免 agent
被引导去修改配置文件。默认值为 `true`。

## 下一步

- [Plugin 开发](development.zh.md) —— 从一个最小的 `hello` 命令到完整契约,开发你
  自己的插件。
- [Skills](../skills/introduction.zh.md) —— skill 的工作机制,包括插件自带的 skill。
