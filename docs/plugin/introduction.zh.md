# Plugin 介绍

**plugin**(插件)是一个可安装的 Python 包,在不修改 Datus 本身的前提下对其进行扩展。
把插件安装到与 `datus` 相同的 Python 环境中,根据插件打包的内容,你可以获得:

| 功能面 | 提供什么 |
|---|---|
| CLI 子命令 | `datus <plugin> ...` 运行插件自己的命令行界面 |
| Skills | 插件自带的 skill 出现在 `/skill list` 中,与项目、用户 skill 并列 |
| Agent 感知 | 插件在 agent 的 system prompt 中描述自己和已配置的环境,模型会主动选用它 |
| Bash 权限 | 插件预先声明哪些子命令 agent 可以直接执行、哪些需要确认 |
| 工具 transformer | 插件可以在 agent 的工具调用执行前改写参数或拒绝调用(如强制 SQL 作用域策略) |

Datus 通过 `datus.plugins` Python entry-point 组在每次调用时发现插件——安装或升级插件
无需重启,也没有任何注册步骤。

想开发自己的插件?见[开发指南](development.zh.md)。

## 安装插件

把插件包安装到与 `datus` 相同的环境:

```bash
pip install datus-plugin-hello
datus hello Ada          # 子命令立即可用
```

如果 `datus <name>` 落进了 REPL 而不是运行插件,说明该包没有安装在 `datus`
所运行的环境里。

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
`./.datus/config.yml` 里 pin 住它:

```yaml
plugins:
  hello: staging
```

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
