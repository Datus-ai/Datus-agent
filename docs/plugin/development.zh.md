# Plugin 开发

本指南讲解如何开发一个 Datus plugin:从一个最小可用的 `hello` 命令出发,逐步覆盖
完整契约。关于 plugin 是什么、用户如何安装配置、profile 如何解析,请先阅读
[介绍](introduction.zh.md)。

plugin 是一个可安装的 Python 包,通过 `datus.plugins` entry-point 组被发现。
最关键的约束:

- **plugin 绝不 `import datus.*`**,也不依赖任何共享 SDK。
- **整个契约就是一个声明式文件**——随包分发的 `datus-plugin.yml`。它声明你的 CLI
  入口函数、tool transformer、skills 目录、系统提示词模板、bash 权限规则和配置
  schema。你唯一要写的 Python 只是普通函数。

Datus 是 *配置 broker*——它负责读 `agent.yml`、展开 `${VAR}`、解析激活 profile,
然后用一个普通 `dict` 调用你声明的 `cli` 函数。读取 manifest 不会执行你的任何
代码:skills、权限、提示词、配置 schema 的收集全程不 import 你的包;只有 `cli`
函数(在 `datus <name> ...` 分发时)和声明的 tool transformer 才会被惰性导入。

## 前置条件

- 一个能安装到与 `datus` 同一环境的 Python 包。
- 已安装 `datus`(`pip install datus-agent` 或源码 checkout)。
- Python 3.12+——plugin 运行在 datus 自己的解释器里(`datus-agent` 声明
  `requires-python >= 3.12`),你的代码和依赖必须与之兼容。

## 快速开始:最小 plugin

**1. 包结构**

```text
datus-plugin-hello/
├── pyproject.toml
└── datus_plugin_hello/
    ├── __init__.py
    ├── datus-plugin.yml      # manifest——整个 plugin 契约
    └── cli.py
```

**2. manifest**(`datus_plugin_hello/datus-plugin.yml`)

```yaml
manifest_version: 1
description: "Say hello to someone."
cli: datus_plugin_hello.cli:main
```

**3. CLI 函数**(`datus_plugin_hello/cli.py`)

```python
from __future__ import annotations


def main(argv: list[str], profile: dict) -> int:
    # `profile` 是解析好的 agent.plugins.hello.<profile> 字典
    # (已由 datus 完成 ${VAR} 展开)。空字典也没问题。
    greeting = profile.get("greeting", "Hello")
    name = argv[0] if argv else "world"
    print(f"{greeting}, {name}!")
    return 0
```

**4. 注册 entry-point**(`pyproject.toml`)

```toml
[project]
name = "datus-plugin-hello"
version = "0.1.0"
dependencies = []                      # 注意:不要依赖 datus

[project.entry-points."datus.plugins"]
hello = "datus_plugin_hello"           # 包名——不是类

[tool.setuptools.package-data]
datus_plugin_hello = ["datus-plugin.yml"]
```

entry-point 的值是你的**包名**——一条纯粹的"名字 → 包"映射,不是代码引用。
entry-point 的名字(`hello`)唯一决定 CLI 命令(`datus hello`)和配置键
(`agent.plugins.hello`)——包名可以随意取。三个名字是**保留字**,永远不会分发给
plugin:`upgrade`、`skill`、`plugin`。注册在这些名字下的 plugin 不可达
(`datus plugin install` 也会拒绝),以 `-` 开头的名字则完全无法分发。

manifest 是包数据而非 Python 代码——务必确保它进入 wheel。Hatchling 默认打包包
目录下所有文件;setuptools 需要上面的 `[tool.setuptools.package-data]` 配置。
用 `unzip -l dist/*.whl | grep datus-plugin.yml` 验证(`datus plugin install`
和 `datus plugin pack` 都会拒绝缺失 manifest 的包)。

**5. 安装并运行**

```bash
datus plugin install src:./datus-plugin-hello   # 安装到 ~/.datus/plugins/hello/
datus hello Ada          # -> Hello, Ada!
```

开发期想要快速的改-跑循环,也可以直接 `pip install -e datus-plugin-hello` 到
datus 自己的环境——这类 plugin 作为回退同样会被发现,只是没有
`~/.datus/plugins/` 目录。

以上就是一个完整的 plugin。下面的内容全部是可选扩展面。

## manifest 参考

`datus-plugin.yml` 位于包根。只有 `manifest_version` 必填,其余键全部可选:

| 键 | 类型 | 用途 |
|---|---|---|
| `manifest_version` | int,**必填** | 必须是 `1`。比 datus 能理解的更新的版本会被拒绝并警告"requires newer datus"。 |
| `description` | 字符串 | 一行摘要,`datus plugin info` 展示。 |
| `cli` | 代码引用 | `module.path:function`,在 `datus <name> ...` 时以 `main(argv, profile)` 调用。见[实现 CLI 入口](#实现-cli-入口)。没有它时 `datus <name>` 以退出码 2 结束。 |
| `tool_transformers` | 映射 | 工具 pattern → 代码引用(或引用列表),改写或拒绝 agent 的工具调用。见[工具参数 transformer](#工具参数-transformer)。 |
| `permissions` | 映射 | 你自己 CLI 命名空间的 bash 权限规则,按权限 profile 分组——纯 YAML,零代码。见[CLI bash 权限](#cli-bash-权限)。 |
| `system_prompt` | 路径 | 包内相对路径,指向渲染进 agent 系统提示词的 Jinja2 模板。见[系统提示词模板](#系统提示词模板)。 |
| `skills` | 路径 | 包内相对路径,指向捆绑的 skill 目录。见[捆绑 skills](#捆绑-skills)。 |
| `config_schema` | JSON Schema | 内联 object schema,描述一个 profile——驱动 `/plugins` TUI 表单并在保存前校验。见[配置 schema 与校验](#配置-schema-与校验)。 |

**代码引用**是形如 `module.path:function` 的点分字符串。路径均相对包目录,且不
允许逃逸出去。manifest 解析是防御式的:某一段格式错误只会被警告并丢弃,其余部分
照常可用;只有 `manifest_version` 缺失/不支持(或 YAML 不可读)才会整体拒绝。

机器可读的契约在 `datus/plugins/base.py`;本表与该 docstring 保持同步。

## 配置:Datus 交给你什么

用户在 `agent.plugins.<name>` 下配置你的 plugin,`<name>` 下的每个键都是一个
**profile**(一个环境):

```yaml
agent:
  plugins:
    hello:
      prod:
        default: true
        greeting: Hi
        token: ${HELLO_TOKEN}      # secret 建议用 ${ENV_VAR}
      staging:
        greeting: Yo
```

Datus 把它解析成 `agent.plugins.<name>.<profile> -> dict`,**按 profile 展开
`${VAR}`**,并注入一个等于 profile 名的 `name` 键。哪个 profile 字典进入你的
`cli` 函数由 Datus 决定——显式 `--profile`、项目 pin、`default: true`、唯一
profile,或在完全未配置时给一个空字典。完整解析顺序见
[介绍](introduction.zh.md#which-profile-runs);这些逻辑你一行都不用写,你的函数
只管接收解析好的 `dict`。

本地测试时,把 profile 写进你的 datus 会话实际加载的那个配置文件(显式
`--config` → `./conf/agent.yml` → `~/.datus/conf/agent.yml`)。

## 配置 schema 与校验

声明一个 `config_schema`——描述**单个 profile** 的内联 JSON Schema——之后
`/plugins` TUI 会渲染出真正的表单(而非自由键值编辑),Datus 也会在保存前用它
校验候选 profile:

```yaml
config_schema:
  type: object
  required: [token, s3]
  properties:                    # 属性顺序 == TUI 字段顺序
    token:
      type: string
      description: "API token"
      x-secret: true             # TUI 中掩码显示,提示词渲染时被剥离
    greeting:
      type: string
      description: "Greeting word"
      default: "Hi"
    s3:                          # 嵌套 object 会展开为点分表单字段
      type: object
      required: [secret_access_key]
      properties:
        region: {type: string, default: us-east-1}
        secret_access_key: {type: string, x-secret: true}
```

语义:

- **`x-secret: true`** 标记 secret 字段:TUI 掩码显示并提示用户输入
  `${ENV_VAR}` 引用,系统提示词渲染器会剥离它(见
  [系统提示词模板](#系统提示词模板))。它是属性级扩展关键字——JSON Schema
  校验器会忽略它。
- **`required`** 成员标记表单必填字段;**`default`** 会直接预填为字段初始值。
  留空的字段(无 default 且尚未输入)会以浅色占位符显示其 `description`。
- **嵌套 object** —— 带自身 `properties` 的 `type: object` 属性在 TUI 中展开
  为每个叶子一个字段,字段名为点分路径(`s3.region`、`s3.secret_access_key`);
  提交时按嵌套结构重新组装后再保存。object 上的 `x-secret: true` 会把所有叶子
  标记为 secret;叶子只有在整条祖先路径都必填时才是表单必填。系统提示词的
  白名单剥离也按同样规则递归过滤已声明的嵌套 object。
- **自由 object 会整体透传。** **没有**自身 `properties` 的 `type: object`
  属性(自由字典)*不会*逐键过滤:TUI 把它当作单个字段,系统提示词渲染器会把
  它存储的整个值原样送进提示词。被跳过的只是逐键剥离——字段本身仍须先声明——
  因此这种字段里嵌套的任何 secret 都会到达 LLM。要么声明其子 `properties`
  (以获得递归过滤),要么把整个 object 标记为 `x-secret: true`。
- **校验**在原始候选字典上运行 `jsonschema`(用户刚输入、**尚未** `${VAR}` 展开
  的值)。含 `${ENV_VAR}` 占位符的值被视为不透明——针对它们的
  pattern/enum/format 违规会被抑制,但缺失 `required` 字段仍会报错。真正的运行
  时校验放在你的 `cli` 函数里。
- **TUI 输入的值都是字符串**,所以优先用 `type: string` 配合 `pattern` /
  `enum` 约束;其他类型留给手写 `agent.yml` 的场景。
- schema 本身非法(被 JSON Schema 元 schema 拒绝)时只会警告并视为不存在——TUI
  回退到自由编辑。

## 实现 CLI 入口

manifest 的 `cli` 指向一个以 `main(argv, profile)` 调用的函数:

```text
datus hello --profile staging greet Ada
                └── 被剥离 ──┘ └── argv = ["greet", "Ada"] ──┘
```

只有出现在**第一个非选项 token 之前**的 `--profile` / `--config` 会被当作
Datus 全局参数消费;从第一个命令 token 起的一切都属于 plugin。因此
`datus hello greet --profile staging` 会原样传入
`["greet", "--profile", "staging"]`——你的子命令可以自由定义自己的 `--profile`。

返回整数退出码(`None` 视为 `0`)。建议的约定:

| 退出码 | 含义 |
|---|---|
| `0` | 成功 |
| `1` | 运行时错误 |
| `2` | 用法错误 |
| `3` | 配置错误 |
| `8` | 缺少可选依赖 |

直接抛异常也可以——Datus 会捕获 `cli` 函数的异常并映射为退出码 `1`,不会让 CLI
崩溃——但显式返回退出码能给用户更清晰的信号。

## 配方:把函数和 API 包装成 CLI

`cli` 函数收到的是原始 `argv` 列表,路由方式完全自由。以下是四种常见模式,从最
快到最丰富。

### A. 字典分发——几个函数,零依赖

```python
def main(argv, profile):
    if not argv:
        print("usage: datus toolbox <add|upper> ...")
        return 2
    cmd, rest = argv[0], argv[1:]
    handlers = {"add": _add, "upper": _upper}
    handler = handlers.get(cmd)
    if handler is None:
        print(f"unknown command: {cmd}")
        return 2
    return handler(rest)


def _add(args):          # datus toolbox add 1 2 3
    print(sum(float(a) for a in args))
    return 0


def _upper(args):        # datus toolbox upper hello
    print(" ".join(args).upper())
    return 0
```

### B. argparse——带类型的参数、flag、自动 usage/`-h`

标准库,零额外依赖。`argparse` 在 `-h` 或错误用法时打印 usage 并抛
`SystemExit`;Datus 将其转成对应退出码(`-h` 为 0,用法错误为 2),这正是常规
CLI 行为。

```python
import argparse


def main(argv, profile):
    parser = argparse.ArgumentParser(prog="datus toolbox")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="sum numbers")
    p_add.add_argument("nums", nargs="+", type=float)

    p_grep = sub.add_parser("grep", help="filter lines in a file")
    p_grep.add_argument("pattern")
    p_grep.add_argument("path")
    p_grep.add_argument("-i", "--ignore-case", action="store_true")

    ns = parser.parse_args(argv)      # -h / 错误用法时抛 SystemExit
    if ns.cmd == "add":
        print(sum(ns.nums))
        return 0
    if ns.cmd == "grep":
        return _grep(ns.pattern, ns.path, ns.ignore_case)


def _grep(pattern, path, ignore_case):
    needle = pattern.lower() if ignore_case else pattern
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            hay = line.lower() if ignore_case else line
            if needle in hay:
                print(line.rstrip())
    return 0
```

### C. 包装 REST API

从 profile 读取端点和凭证(Datus 已展开 `${VAR}`),把子命令映射为请求。凭证
只放在 profile 里——绝不硬编码,也绝不回显。

```python
import argparse
import json


def main(argv, profile):
    import requests  # plugin 可以有自己的依赖

    base = profile.get("api_base_url")
    if not base:
        print("no api_base_url configured for the profile")
        return 3
    headers = {}
    if profile.get("token"):
        headers["Authorization"] = f"Bearer {profile['token']}"

    parser = argparse.ArgumentParser(prog="datus petstore")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-pets")
    p_get = sub.add_parser("get-pet")
    p_get.add_argument("id")
    ns = parser.parse_args(argv)

    base = base.rstrip("/")
    if ns.cmd == "list-pets":
        resp = requests.get(f"{base}/pets", headers=headers, timeout=30)
    else:
        resp = requests.get(f"{base}/pets/{ns.id}", headers=headers, timeout=30)

    if resp.status_code >= 400:
        print(f"error {resp.status_code}: {resp.text}")
        return 1
    print(json.dumps(resp.json(), indent=2))
    return 0
```

对应配置:

```yaml
agent:
  plugins:
    petstore:
      prod:
        default: true
        api_base_url: https://api.example.com/v1
        token: ${PETSTORE_TOKEN}
```

### D. Typer / Click——最丰富的体验,一个额外依赖

命令面很大时,[Typer](https://typer.tiangolo.com/) 这类框架能提供帮助文本、类型
转换和补全。由于 Datus 每次调用你的入口函数,而 Typer app 是模块级对象,可通过
模块全局变量把当前 profile 暴露给命令读取。

```python
import typer

app = typer.Typer(add_completion=False)
_ACTIVE_PROFILE: dict = {}


@app.command("greet")
def greet(name: str, loud: bool = False):
    greeting = _ACTIVE_PROFILE.get("greeting", "Hello")
    msg = f"{greeting}, {name}!"
    print(msg.upper() if loud else msg)


def main(argv, profile):
    global _ACTIVE_PROFILE
    _ACTIVE_PROFILE = profile
    try:
        # standalone_mode=False 阻止 Click 自己调用 sys.exit,
        # 这样我们能返回退出码并保证清理 profile。
        app(args=argv, standalone_mode=False)
        return 0
    except SystemExit as exc:      # -h / 用法错误
        return int(exc.code or 0)
    except typer.Exit as exc:
        return exc.exit_code
    finally:
        _ACTIVE_PROFILE = {}
```

把 `typer` 加进你自己包的 `dependencies`(plugin 的依赖归它自己——只是不能有
`datus`)。

## 捆绑 skills

在 manifest 里声明一个包内相对路径的 skills 目录,Datus 启动时即发现这些 skill
(出现在 `/skill list`,与项目和用户 skill 并列)——零代码:

```yaml
skills: skills
```

目录与打包:

```text
datus_plugin_hello/
├── datus-plugin.yml
└── skills/
    └── hello/
        └── SKILL.md
```

最小的 `SKILL.md` 是 YAML frontmatter 加 markdown 说明(frontmatter 遵循
Skills 系统使用的 [agentskills.io](https://agentskills.io) 规范):

```markdown
---
name: hello
description: Say hello to someone via the `datus hello` CLI
---

# Hello

Run `datus hello <name>` to greet someone. ...
```

完整 frontmatter 字段参考见 [Skills](../skills/introduction.zh.md) 文档。

确保 skill 文件被打进 wheel(它们是数据,不是 Python 代码)。Hatchling 默认打包
包目录下所有文件,除非文件被 VCS 忽略(那样需要在
`[tool.hatch.build.targets.wheel] artifacts` 列出);setuptools 必须显式声明:

```toml
[tool.setuptools.package-data]
datus_plugin_hello = ["datus-plugin.yml", "skills/**/*", "prompts/*"]
```

构建后用 `unzip -l dist/*.whl | grep SKILL.md` 验证。

## 系统提示词模板

plugin 可以预先告诉 agent 它是什么、配置了哪些环境——让模型主动选择它而不是靠
猜。在 manifest 里声明一个 Jinja2 模板:

```yaml
system_prompt: prompts/system.md.j2
```

```jinja
## Hello
Say hello via `datus hello <name>`.

{% if profiles %}
Environments ({{ profiles | length }}):
{% for name, cfg in profiles.items() %}
- {{ name }}: {{ cfg.get("greeting", "?") }}
{% endfor %}
{% else %}
Installed but not configured.
{% if config_mutable %}
Run the `hello-setup` skill to configure an environment.
{% else %}
Configuration is managed by the deployment administrator — tell the user to
contact them.
{% endif %}
{% endif %}
```

渲染上下文:

| 变量 | 值 |
|---|---|
| `plugin_name` | 你的 entry-point 名 |
| `profiles` | `dict[str, dict]`——plugin 的 profile 映射,**已收窄到项目激活的 profile**(`./.datus/config.yml` 的 `plugins.<name>.active_profile`)且**已剥离 secret**(见下) |
| `config_path` | 已加载的 agent 配置文件路径,或 `None`(配置只读时恒为 `None`) |
| `config_mutable` | agent 可编辑配置文件时为 `True`;托管部署(多租户 chat API / gateway)下为 `False`。需要引入该变量的 datus-agent 版本及以上——旧版本中引用它的模板会渲染失败(严格模式)并跳过该 section |

已安装但未配置的 plugin(或 pin 无匹配)渲染时 `profiles == {}`——用
`{% if profiles %}` 输出一段"已安装未配置"提示而不是从提示词中消失,并在该
分支内检查 `config_mutable`:仅当它为 `True` 时指向捆绑的 setup skill,否则
提示用户联系管理员。

只要有任一 plugin 贡献了 section,Datus 会前置自己的 `## Plugins` 导语,写明已
加载的配置文件和 `agent.plugins.<plugin>.<profile>` 结构——你的模板永远不需要
硬编码配置路径。只读模式下导语会换成禁止建议修改配置、请联系管理员的措辞。

!!! note "secret 是结构性剥离的"
    渲染出的文本会进入 LLM 上下文,而 profile 的值在提示词构建时已完成
    `${VAR}` 展开(是真实明文 secret)——所以 Datus 在模板看到 profiles
    **之前**就做了过滤:只有在你的 `config_schema` 中声明且**未**标记
    `x-secret: true` 的字段才会通过;未声明的字段同样被丢弃,已声明的嵌套
    object(带自身 `properties` 的 `type: object`)也按同样规则递归过滤。
    唯一例外是**自由** object 字段(声明为 `type: object` 但没有 `properties`):
    它的整个值会原样透传,若可能存放敏感内容,请把它标记为 `x-secret: true`。
    没有 `config_schema` 时,模板只能拿到 profile 名和空字典。模板若引用被剥离的
    字段会渲染失败(严格模式)并跳过该 section——永远不可能泄漏。

模板错误(文件缺失、语法错误、未定义变量)只会记录日志并跳过该 section——绝不
影响提示词构建。模板以严格模式(`StrictUndefined`)渲染,笔误会出现在日志里,
而不是变成悄悄错误的提示词文本。

## CLI bash 权限

当 **agent**(而非人)通过它的 bash 工具运行你的 CLI——例如模型决定执行
`datus hello greet Ada`——命令会经过 Datus 的权限层。没有声明时,agent 发出的
每条 plugin 命令都会请求用户确认。manifest 的 `permissions` 键按权限 profile 声
明你的哪些子命令可以自动运行(`allow`)、哪些必须确认(`ask`)、哪些被阻止
(`deny`)——纯 YAML,零代码:

```yaml
permissions:
  normal:
    allow: ["greet:*"]
    ask: ["config set:*"]
  auto:
    allow: ["greet:*", "config set:*"]
```

语义:

- **模式相对于你的命名空间。**Datus 给每个模式加 `datus <name> ` 前缀,
  `greet:*` 变成 `datus hello greet:*`。plugin 永远影响不到 `datus <name>` 之外
  的命令——不能碰 `rm`,也不能碰别的 plugin。
- **模式语法**与 `agent.yml` 的 `permissions.bash_commands` 一致:`cmd` 精确
  匹配,`cmd:*` 前缀匹配,`cmd:glob` 前缀匹配且第一个参数需满足 glob(如
  `greet:A*`)。裸 `:*` 覆盖整个命名空间。
- **profile 键**:只接受 `normal` 和 `auto`。`dangerous` profile 按设计忽略所有
  命令级 bash 规则;`dangerous` 键会被警告并丢弃。
- **用户永远优先。**用户在 `agent.yml` 写的 `deny` 覆盖 plugin 的 `allow`
  (deny > ask > allow,与声明顺序无关),plugin 声明也永远改不了 profile 的默认
  姿态。
- **`ask` 规则可以按项目放宽。**agent 撞上你声明的 `ask` 子命令时,确认提示会
  提供"allow (project)"选项——选择后把精确匹配到的模式(如
  `datus hello config set:*`)持久化到项目 `.datus/config.yml` 的 `bash_allow`
  列表,此后该子命令自动运行。授权只精确匹配:永远不会扩大到命名空间的其余部
  分,你的 `deny` 规则也不受影响。(用户自己在 `agent.yml` 写的 `ask` 规则没有
  这个选项——放宽它们应该改用户自己的配置。)
- **范围**:只有 agent 的 bash 工具被管控。人在终端里敲 `datus hello ...` 永远
  不受影响。`plugins_enabled: false` 会连同插件系统的其余部分一起禁用收集(见
  [介绍](introduction.zh.md#disabling-the-plugin-system))。
- **`--profile` 对匹配透明。**`datus hello --profile prod config set x` 匹配的
  规则(和项目授权)与不带该参数的形式相同——前导的 datus 全局 flag 在求值前被
  归一化掉。`--config <path>` 刻意**不**归一化:把 datus 指向另一个配置文件会重
  绑凭证,这类调用总是回退到确认。
- 畸形声明(类型错误、未知键、空模式)只会记录日志并跳过——绝不影响 Datus 启动。

只读子命令声明为 `normal` 下的 `allow`,变更状态的声明为 `ask`;只有重复执行无
害的常规变更才在 `auto` 下提升为 `allow`。

## 工具参数 transformer

manifest 的 `tool_transformers` 键让 plugin 拦截 **agent 的工具调用**——在工具
执行前检查并改写参数,或直接拒绝调用。典型用例是 SQL 策略强制:给每条
`execute_sql` 查询追加租户范围谓词,使用部署注入的请求 principal。

```yaml
tool_transformers:
  "db_tools.execute_sql": datus_plugin_scoped_sql.transformers:enforce_tenant_scope
```

```python
# datus_plugin_scoped_sql/transformers.py
def enforce_tenant_scope(tool_name, args, context):
    tenant_id = (context.get("principal") or {}).get("tenant", {}).get("id")
    if not tenant_id:
        raise PermissionError("missing principal.tenant.id; cannot scope query")
    args["sql"] = add_where_predicate(args["sql"], f"tenant_id = '{tenant_id}'")
    return args
```

语义:

- **声明形状**:工具 pattern 到代码引用(或引用列表)的映射。pattern 使用 proxy
  语法——裸工具名(`execute_sql`),或带 fnmatch glob 的 `category.method`
  (`db_tools.*`)。
- **transformer 签名**:`transformer(tool_name, args, context) -> dict`,同步或
  异步均可。返回(可能已修改的)参数字典则继续执行。**抛异常即拒绝**:工具不会
  运行,模型把你的异常消息当作普通工具失败收到。返回非 dict 同样拒绝,fail
  closed。
- **`context`** 是普通字典,含 `node_name`、`principal`(请求级调用方属性,部署
  未设置时为空)、`project_root` 和 `agent_config`(存活的 agent 配置对象——用
  `context["agent_config"].get_plugin_profile("<name>")` 读取你自己的 profile;
  鸭子类型访问,绝不为此 `import datus.*`)。它在每次调用时重建,请求级的值总是
  新鲜的。
- **加载是惰性的**:被引用的模块在为 agent 节点首次收集 transformer 时才导入,
  而不是在 manifest 加载时。导入失败(或不可调用)的引用会被警告并跳过——plugin
  其余的 transformer 仍然生效。
- **覆盖范围**:transformer 包装 agent 的 `FunctionTool` 层,两条执行路径
  (SDK Runner 与原生 loop)都经过它。它**不**覆盖对工具方法的直接 Python 调用
  (如 reference-template 执行)或代理给外部 client 的工具——必须在这些路径上
  存活的服务端强制应放进工具层本身(见 `agent.sql_policy`)。
- **信任模型**:transformer 在进程内运行,能完整访问每个被匹配工具调用的参数。
  它们是受信代码,与插件面的其余部分一样受 `plugins_enabled` 总开关管控。
- 改写 SQL 时用 SQL parser 或数据库安全的查询构造器——策略谓词绝不用字符串拼接。
- 收集成功但应用失败的声明会中止 agent 节点,而不是在没有强制的情况下静默运行。

## 捆绑 setup skill

`pip install` 之后手工编辑 YAML 是最大的摩擦。把 `<name>-setup` skill 与主
skill 一起分发,让 agent 自己收集值并写好 profile:

```text
datus_plugin_hello/
└── skills/
    ├── hello/
    │   └── SKILL.md
    └── hello-setup/
        └── SKILL.md
```

setup `SKILL.md` 应按顺序覆盖:

1. **何时使用**——plugin 未配置,或用户想加一个环境。
2. **配置结构**——`agent.plugins.<name>.<profile>` 的完整 YAML 模板,用注释标出
   必填/可选/secret 字段。
3. **询问用户**——列出必须来自用户的字段(端点、认证方式……)。secret 字段要求
   agent 让用户导出环境变量,YAML 里写 `${VAR}` 引用——绝不写字面 secret。
4. **写入配置**——写进 `## Plugins` 提示词导语指出的文件,第一个 profile 标
   `default: true`。
5. **验证**——一条便宜的只读命令(如 `datus hello version`)。
   `datus <plugin>` 每次调用都重新加载配置,profile 立即生效;提示词里的环境列
   表下个会话刷新。

在 setup skill 的 frontmatter 里声明 `requires_mutable_config: true`。配置只读
的部署(多租户 chat API / gateway)中,Datus 会把该 skill 从
`<available_skills>` 中隐藏并拒绝 `load_skill`——agent 会引导用户联系管理员,
而不是走一遍它做不到的配置编辑。同时在正文保留一条保护性说明,作为旧版
datus-agent(会忽略该 frontmatter 字段)的兜底:若当前环境无法编辑配置文件,
agent 应告知用户去服务器上改 `agent.yml`。

一个完整的最小 `hello-setup/SKILL.md`:

````markdown
---
name: hello-setup
description: Configure an environment profile for the `datus hello` plugin
requires_mutable_config: true
---

# Hello Setup

Use this skill when `datus hello` is installed but has no configured
environment, or when the user wants to add another one.

## Config structure

Profiles live under `agent.plugins.hello.<profile>` in the config file named
by the `## Plugins` section of the system prompt:

```yaml
agent:
  plugins:
    hello:
      prod:
        default: true            # mark the first profile as default
        greeting: Hi             # required
        token: ${HELLO_TOKEN}    # secret — reference an env var, never a literal
```

## Steps

1. Ask the user for `greeting` and which environment variable holds the
   token. Have the user export the variable; write `${VAR}` into the YAML —
   never a literal secret.
2. Write the profile into the config file above; mark the first profile
   `default: true`.
3. Verify with a cheap read-only call: `datus hello Ada`.

If this environment cannot edit the config file (API / web deployment), tell
the user to edit `agent.yml` on the server instead.
````

## 端到端验证你的 plugin

`pip install -e` 之后,每个扩展面都可以在不重启任何东西的情况下验证(plugin 每
次调用都会被发现):

- **CLI 分发**——在任意目录运行 `datus <name> ...`。如果落进了 REPL,说明
  entry point 缺失/拼错,或 manifest 被拒绝(看日志);`datus <name>` 打印
  "declares no CLI command" 说明 manifest 加载了但没有 `cli` 键。用
  `pip show -f your-package` 检查 `entry_points.txt` 和随包的
  `datus-plugin.yml`。
- **Skills**——启动 `datus` 执行 `/skill list`;plugin 捆绑的 skill 与项目、
  用户 skill 并列出现。
- **提示词注入**——在单测里直接渲染模板(见下节)。要确认真实会话里生效,启动
  `datus` 问 agent"配置了哪些 plugin?"——答案来自注入的 section。注意配置修改
  对下一次 `datus <plugin>` 调用立即生效,但提示词 section 只在下个会话刷新。

## 测试你的 plugin

因为 Datus 是 broker,单测直接用普通 dict 调用你的函数——不需要 `agent.yml`,
不需要 import Datus。manifest 和模板用普通的 YAML/Jinja2 工具验证:

```python
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from datus_plugin_hello.cli import main

PKG = Path(__file__).parent.parent / "datus_plugin_hello"


def test_cli_uses_profile_greeting(capsys):
    rc = main(["Ada"], {"name": "prod", "greeting": "Hi"})
    assert rc == 0
    assert "Hi, Ada!" in capsys.readouterr().out


def test_manifest_is_valid():
    manifest = yaml.safe_load((PKG / "datus-plugin.yml").read_text())
    assert manifest["manifest_version"] == 1
    assert manifest["cli"] == "datus_plugin_hello.cli:main"
    # 所有声明的路径都存在于包内
    assert (PKG / manifest["skills"]).is_dir()
    assert (PKG / manifest["system_prompt"]).is_file()


def test_prompt_template_renders_without_secrets():
    env = Environment(loader=FileSystemLoader(PKG), undefined=StrictUndefined)
    template = env.get_template("prompts/system.md.j2")
    # datus 渲染前会剥离未声明 / x-secret 字段;此处模拟同样的输入。
    ctx = {"plugin_name": "hello", "config_path": None, "config_mutable": True}
    text = template.render(profiles={"prod": {"greeting": "Hi"}}, **ctx)
    assert "## Hello" in text
    text = template.render(profiles={}, **ctx)
    assert "hello-setup" in text
    # 只读部署下绝不能把 agent 指向 setup skill
    text = template.render(profiles={}, plugin_name="hello", config_path=None, config_mutable=False)
    assert "hello-setup" not in text
```

## 分发离线安装包

在有网络的机器上、从 plugin 项目目录用 `datus plugin pack` 打出 wheelhouse
`.zip`:

```bash
datus plugin pack -o ./dist               # 只含 plugin wheel(默认)
# → ./dist/datus-plugin-hello-1.0.0.zip

datus plugin pack --with-deps -o ./dist   # plugin wheel + 全部依赖 wheel
```

bundle 是一个 zip,内含 `datus-bundle.json` 清单和 `wheels/` wheelhouse。
`pack` 会验证 wheel 声明了模块引用式的 `datus.plugins` entry point **并且**捆绑
了它的 `datus-plugin.yml`——缺少任一项的 wheel 在任何依赖下载之前就被拒绝。用户
通过 `datus plugin install zip:./….zip` 安装(见
[离线安装](introduction.zh.md#offline-install))。选择哪种口味:

- **默认(只含 plugin wheel)**——bundle 很小;目标机器在安装时从 package index
  解析依赖(需要网络)。
- **`--with-deps`**——捆绑所有传递依赖 wheel,安装完全离线
  (`pip install --no-index --find-links`)。`pack` 使用
  `pip download --only-binary=:all:`,因此每个依赖都必须发布 wheel(不接受只有
  sdist 的包)。纯 Python 依赖集可跨平台;含原生扩展的依赖会让 bundle 变成
  **同平台快照**——在与目标 OS/Python 匹配的机器上构建。
- **声明 `Requires-Python`。**在 `pyproject.toml` 里设置;`pack` 会把它拷进清
  单,安装时可以尽早拒绝不匹配的解释器(报错清晰,可用 `--force` 覆盖)。

## 约束检查清单

发布前逐项确认:

- [ ] 包内任何地方都**没有** `import datus`(`grep -rn "import datus" your_pkg/`)。
- [ ] `pyproject.toml` **不**依赖 `datus` 或共享 plugin SDK。
- [ ] `datus.plugins` entry-point 的值是**包名**(没有 `:Class` 部分)。
- [ ] entry-point 名不是保留名(`upgrade`、`skill`、`plugin`)且不以 `-` 开头。
- [ ] `datus-plugin.yml` 位于包根,声明 `manifest_version: 1`,并进入 wheel(`unzip -l dist/*.whl`)。
- [ ] `cli` 函数签名为 `main(argv: list[str], profile: dict) -> int | None`,成功路径不调用 `sys.exit()`。
- [ ] secret 配置字段在 `config_schema` 中标记了 `x-secret: true`。
- [ ] 系统提示词模板用 `{% if profiles %}` 处理 `profiles == {}` 的情况。
- [ ] `permissions` 模式是命名空间相对的(不带 `datus <name>` 前缀——Datus 会加),`normal` 下变更状态的子命令是 `ask`。
- [ ] skill 文件和提示词模板都打进了 wheel。
- [ ] entry-point 名与期望的 `datus <name>` 命令和 `agent.plugins.<name>` 配置键一致。

## 参考

- **entry-point 组**:`datus.plugins`——每个 plugin 一条,值为 plugin 的**包名**。
- **契约的唯一权威**:`datus/plugins/base.py`(`datus-plugin.yml` manifest 规格)。
- **相关**:[Plugin 介绍](introduction.zh.md)、[Skills](../skills/introduction.zh.md)。
