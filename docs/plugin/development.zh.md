# Plugin 开发

本指南讲解如何开发一个 Datus plugin:从一个最小可用的 `hello` 命令出发,逐步覆盖
完整契约。关于 plugin 是什么、用户如何安装配置、profile 如何解析,请先阅读
[介绍](introduction.zh.md)。

plugin 是一个可安装的 Python 包,通过 `datus.plugins` entry-point 组被发现。
最关键的约束:**plugin 绝不 `import datus.*`,也不依赖任何共享 SDK**。契约只是一小组
Datus 按结构调用的方法名(鸭子类型)。Datus 是 *配置 broker*——它负责读 `agent.yml`、
展开 `${VAR}`、解析激活 profile,用一个普通 `dict` 构造你的 plugin 并调用它。你只需实现方法。

## 前置条件

- 一个能安装到与 `datus` 同一环境的 Python 包。
- 已安装 `datus`(`pip install datus-agent` 或源码 checkout)。
- Python 3.12+——plugin 运行在 datus 自己的解释器里(`datus-agent` 声明
  `requires-python >= 3.12`),你的代码和依赖必须与之兼容。

## 快速开始:最小 plugin

一个 plugin 就是注册在 `datus.plugins` 下的一个类。下面是最小可用示例——一个 `hello` 命令。

**1. 包结构**

```
datus-plugin-hello/
├── pyproject.toml
└── datus_plugin_hello/
    ├── __init__.py
    └── plugin.py
```

**2. plugin 类**(`datus_plugin_hello/plugin.py`)

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional


class HelloPlugin:
    def __init__(self, profile: Optional[Dict[str, Any]] = None) -> None:
        # `profile` 是解析好的 agent.plugins.hello.<profile> 字典
        # (已由 datus 完成 ${VAR} 展开)。空字典也没问题。
        self.profile: Dict[str, Any] = profile or {}

    def run_cli(self, argv: List[str]) -> int:
        greeting = self.profile.get("greeting", "Hello")
        name = argv[0] if argv else "world"
        print(f"{greeting}, {name}!")
        return 0
```

**3. 注册 entry-point**(`pyproject.toml`)

```toml
[project]
name = "datus-plugin-hello"
version = "0.1.0"
dependencies = []                      # 注意:不要依赖 datus

[project.entry-points."datus.plugins"]
hello = "datus_plugin_hello.plugin:HelloPlugin"
```

CLI 命令名(`datus hello`)和配置键(`agent.plugins.hello`)**只由 entry-point
名(`hello`)决定**——类名和模块结构随意。有两个名字是**保留字**,永远不会分发给
plugin:`upgrade` 和 `skill`,注册成这两个名字的 plugin 会静默失效;以 `-` 开头的
名字也无法被分发。

**4. 安装并运行**

```bash
pip install -e datus-plugin-hello
datus hello Ada          # -> Hello, Ada!
```

这就是一个完整的 plugin。下面的内容都是可选的扩展面。

## 契约

Datus 在 entry-point 解析出的类上**按方法名**调用下列成员。你的类不导入、不继承 Datus 的任何东西。

| 成员 | 形态 | 用途 |
|---|---|---|
| `PluginClass(profile: dict)` | 构造函数 | Datus 以**关键字参数**方式传入解析好的 `agent.plugins.<name>.<profile>` 字典(已展开环境变量)——即 `PluginClass(profile=...)`,因此参数必须命名为 `profile`。config-free plugin 可忽略其值。 |
| `run_cli(self, argv: list[str]) -> int \| None` | 实例方法 | 执行子命令。`argv` 是 `datus <plugin>` 之后的全部参数(Datus 自己的 `--profile`/`--config` 已剥离)。返回退出码,`None` 视为 `0`。 |
| `skills_dir() -> str \| None` | **可选**,类级 | 返回打包的 skill 目录。见[打包 skill](#skill)。 |
| `system_prompt(profiles: dict[str, dict]) -> str \| None` | **可选**,类级 | 返回注入 system prompt 的 markdown 段。见[注入 system prompt](#system-prompt)。 |
| `cli_permissions() -> dict \| None` | **可选**,类级 | 按权限 profile 声明本 plugin CLI 命名空间内的 bash 权限规则。见 [CLI bash 权限](#cli-permissions)。 |
| `tool_transformers() -> dict \| None` | **可选**,类级 | 声明工具参数 transformer,在 agent 的工具调用执行前改写参数或拒绝调用。见 [工具参数 transformer](#tool-transformers)。 |

!!! warning "`skills_dir` 与 `system_prompt` 必须类级可取"
    Datus 在**启动期、无激活 profile** 时就解析这两者(skill 发现和 prompt 构建都发生在
    任何命令执行之前)。请声明为 `@classmethod` / `@staticmethod`(`skills_dir` 也可以是普通类属性)——
    它们不能依赖 `__init__`。

## 配置:Datus 交给你什么

用户在 `agent.plugins.<name>` 下配置你的 plugin,`<name>` 之下的每个键是一个
**profile**(一套环境):

```yaml
agent:
  plugins:
    hello:
      prod:
        default: true
        greeting: Hi
        token: ${HELLO_TOKEN}      # 密钥优先用 ${ENV_VAR}
      staging:
        greeting: Yo
```

Datus 把它解析成 `agent.plugins.<name>.<profile> -> dict`,**逐 profile 展开 `${VAR}`**,
并注入一个等于 profile 名的 `name` 键。哪个 profile 字典进入你的构造函数由 Datus
决定——显式 `--profile`、项目 pin、`default: true`、唯一 profile,或在完全未配置时
的空字典。完整解析顺序见[介绍](introduction.zh.md#which-profile-runs);这些逻辑你一行
都不用写,构造函数只管接收解析好的 `dict`。

本地测试时,把 profile 写进你的 datus 会话实际加载的那个 config 文件
(显式 `--config` → `./conf/agent.yml` → `~/.datus/conf/agent.yml`)。

## 实现 `run_cli`

`argv` 是剥离了 Datus 全局参数后的命令尾部:

```
datus hello --profile staging greet Ada
                └── 已剥离 ──┘ └── argv = ["greet", "Ada"] ──┘
```

只有出现在**第一个非选项 token 之前**的 `--profile` / `--config` 会被当作 Datus
全局参数消费;从第一个命令 token 起,后面的一切都属于 plugin。因此
`datus hello greet --profile staging` 会把 `["greet", "--profile", "staging"]`
原样传给你——你的子命令完全可以定义自己的 `--profile` 选项。

返回整数退出码。建议采用的约定:

| 退出码 | 含义 |
|---|---|
| `0` | 成功 |
| `1` | 运行时错误 |
| `2` | 用法错误 |
| `3` | 配置错误 |
| `8` | 缺少可选依赖 |

抛异常也可以——Datus 会捕获 `run_cli` 抛出的异常并映射为退出码 `1`,不会让 CLI 崩溃——
但返回明确的退出码能给用户更清晰的信号。

## 食谱:把函数/API 快速封装成 CLI

`run_cli` 收到的是原始 `argv` 列表,所以你可以任意路由。下面是四种常见模式,从最快到最完善。

### A. 字典分发 —— 几个函数,零依赖

暴露少量函数最快的方式:用第一个 token 映射到 handler,每个 handler 拿到 `argv` 剩余部分。

```python
class ToolboxPlugin:
    def __init__(self, profile=None):
        self.profile = profile or {}

    def run_cli(self, argv):
        if not argv:
            print("usage: datus toolbox <add|upper> ...")
            return 2
        cmd, rest = argv[0], argv[1:]
        handlers = {"add": self._add, "upper": self._upper}
        handler = handlers.get(cmd)
        if handler is None:
            print(f"unknown command: {cmd}")
            return 2
        return handler(rest)

    def _add(self, args):          # datus toolbox add 1 2 3
        print(sum(float(a) for a in args))
        return 0

    def _upper(self, args):        # datus toolbox upper hello
        print(" ".join(args).upper())
        return 0
```

### B. argparse —— 带类型的参数、开关、自动 usage/`-h`

标准库,无额外依赖。`argparse` 在 `-h` 或用法错误时会打印用法并抛 `SystemExit`;
Datus 会把它作为退出码透出(`-h` 为 0,用法错误为 2),这正是 CLI 的惯例行为。

```python
import argparse

class ToolboxPlugin:
    def __init__(self, profile=None):
        self.profile = profile or {}

    def run_cli(self, argv):
        parser = argparse.ArgumentParser(prog="datus toolbox")
        sub = parser.add_subparsers(dest="cmd", required=True)

        p_add = sub.add_parser("add", help="sum numbers")
        p_add.add_argument("nums", nargs="+", type=float)

        p_grep = sub.add_parser("grep", help="filter lines in a file")
        p_grep.add_argument("pattern")
        p_grep.add_argument("path")
        p_grep.add_argument("-i", "--ignore-case", action="store_true")

        ns = parser.parse_args(argv)      # -h / 用法错误时抛 SystemExit
        if ns.cmd == "add":
            print(sum(ns.nums))
            return 0
        if ns.cmd == "grep":
            return self._grep(ns.pattern, ns.path, ns.ignore_case)

    def _grep(self, pattern, path, ignore_case):
        needle = pattern.lower() if ignore_case else pattern
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                hay = line.lower() if ignore_case else line
                if needle in hay:
                    print(line.rstrip())
        return 0
```

### C. 封装 REST API

从 profile 里读端点和凭据(Datus 已展开 `${VAR}`),再把子命令映射到请求。
凭据保留在 profile 中——绝不硬编码,也绝不回显。

```python
import argparse
import json

class PetstorePlugin:
    def __init__(self, profile=None):
        self.profile = profile or {}

    def run_cli(self, argv):
        import requests  # plugin 可以依赖自己的库

        base = self.profile.get("api_base_url")
        if not base:
            print("no api_base_url configured for the profile")
            return 3
        headers = {}
        if self.profile.get("token"):
            headers["Authorization"] = f"Bearer {self.profile['token']}"

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

### D. Typer / Click —— 最完善的体验,一个额外依赖

命令面较大时,[Typer](https://typer.tiangolo.com/) 这类框架能自动给你帮助文本、
类型转换和补全。由于 Datus 每次调用都会重新构造你的 plugin,而 Typer app 是模块级对象,
需要通过一个模块全局变量把激活 profile 暴露给各命令读取。

```python
import typer

app = typer.Typer(add_completion=False)
_ACTIVE_PROFILE: dict = {}


@app.command("greet")
def greet(name: str, loud: bool = False):
    greeting = _ACTIVE_PROFILE.get("greeting", "Hello")
    msg = f"{greeting}, {name}!"
    print(msg.upper() if loud else msg)


class GreeterPlugin:
    def __init__(self, profile=None):
        self.profile = profile or {}

    def run_cli(self, argv):
        global _ACTIVE_PROFILE
        _ACTIVE_PROFILE = self.profile
        try:
            # standalone_mode=False 阻止 Click 自行 sys.exit,
            # 这样我们能返回退出码,并始终清理 profile。
            app(args=argv, standalone_mode=False)
            return 0
        except SystemExit as exc:      # -h / 用法
            return int(exc.code or 0)
        except typer.Exit as exc:
            return exc.exit_code
        finally:
            _ACTIVE_PROFILE = {}
```

把 `typer` 加进你的包 `dependencies`(plugin 的依赖是它自己的——只是别依赖 `datus`)。

## 打包 skill {#skill}

如果你的包附带 skill 目录,通过类级 `skills_dir()` 暴露它,Datus 会在启动期发现这些 skill
(它们会出现在 `/skill list`,与项目、用户 skill 并列)。

```python
class HelloPlugin:
    @classmethod
    def skills_dir(cls) -> str:
        from pathlib import Path
        return str(Path(__file__).parent / "skills")
```

目录结构与打包:

```
datus_plugin_hello/
└── skills/
    └── hello/
        └── SKILL.md
```

最小的 `SKILL.md` 由 YAML frontmatter 加 markdown 指令组成(frontmatter 遵循
Skills 系统采用的 [agentskills.io](https://agentskills.io) 规范):

```markdown
---
name: hello
description: Say hello to someone via the `datus hello` CLI
---

# Hello

Run `datus hello <name>` to greet someone. ...
```

完整的 frontmatter 字段参考见 [Skills](../skills/introduction.zh.md) 文档。

确保 skill 文件被打进 wheel(它们是数据文件,不是 Python 模块)。Hatchling 默认
会打包包目录下的所有文件,除非文件被 VCS ignore(此时需列进
`[tool.hatch.build.targets.wheel] artifacts`),否则无需额外配置。setuptools
则必须显式声明:

```toml
[tool.setuptools.package-data]
datus_plugin_hello = ["skills/**/*"]
```

构建后用 `unzip -l dist/*.whl | grep SKILL.md` 验证。

## 注入 system prompt {#system-prompt}

plugin 可以在对话一开始就告诉 agent:自己是什么、配了哪些环境——让模型主动选用,而不是盲猜。
通过类级 `system_prompt(profiles)` 暴露:

```python
class HelloPlugin:
    @classmethod
    def system_prompt(cls, profiles):
        if not profiles:
            # 已安装但未配置:指向 setup skill,而不是从 prompt 里消失。
            return (
                "## Hello (installed, not configured)\n"
                "The `datus hello` CLI is installed but has no environment "
                "configured.\nRun the `hello-setup` skill to configure one."
            )
        envs = "\n".join(
            f"- {name}: {cfg.get('greeting', '?')}"
            for name, cfg in profiles.items()
        )
        return (
            "## Hello\n"
            "Say hello via `datus hello <name>`.\n"
            f"Environments ({len(profiles)}):\n{envs}"
        )
```

Datus 传入该 plugin 的**全部** profile 映射(所有环境,不只激活的那个),并把返回的 markdown
追加到每个 agentic 节点的 system prompt。**已安装但未配置**的 plugin 会收到 `{}`——此时应返回
一段简短的"已安装,未配置"提示,指向自带的 setup skill(见下一节),让 agent 能引导用户完成配置。
只有确实无话可说时才返回 `None`。

只要有任一 plugin 贡献了内容,Datus 会在这些片段前面加上自己的 `## Plugins` 前导块,写明实际
加载的 config 文件位置与 `agent.plugins.<plugin>.<profile>` 结构——你的文本无需硬编码任何路径。

!!! danger "绝不暴露密钥"
    返回文本会进入 LLM 上下文。Datus 交给你的是完整 profile 字典——其中含 `password`、
    secret、access key——但你只能输出**非敏感字段**(endpoint、region、环境名)。
    Datus 从不自行拼接 profile 值;把凭据挡在 prompt 之外是 plugin 的责任。请用字段白名单。

## CLI bash 权限 {#cli-permissions}

当 **agent**(而非人类用户)通过它的 bash 工具执行你的 CLI——例如模型自主决定运行
`datus hello greet Ada`——命令会经过 Datus 的权限层。若无任何声明,这类命令每次都会
弹出确认。类级 `cli_permissions()` 让 plugin 按权限 profile 声明:哪些子命令可以直接
放行(`allow`)、哪些必须确认(`ask`)、哪些直接拦截(`deny`):

```python
class HelloPlugin:
    @classmethod
    def cli_permissions(cls):
        return {
            "normal": {"allow": ["greet:*"], "ask": ["config set:*"]},
            "auto":   {"allow": ["greet:*", "config set:*"]},
        }
```

语义:

- **pattern 相对于你的命名空间。** Datus 会给每条 pattern 加上 `datus <name> ` 前缀,
  `greet:*` 实际生效为 `datus hello greet:*`。plugin 永远无法影响 `datus <name>`
  之外的命令——碰不到 `rm`,也碰不到其他 plugin。
- **pattern 语法**与 `agent.yml` 的 `permissions.bash_commands` 一致:`cmd` 精确匹配,
  `cmd:*` 前缀匹配,`cmd:glob` 前缀匹配且第一个参数需满足 glob(如 `greet:A*`)。
  单写 `:*` 表示覆盖整个命名空间。
- **profile 键**:只接受 `normal` 与 `auto`。`dangerous` profile 设计上忽略一切
  命令级 bash 规则,声明 `dangerous` 键会被告警并丢弃。
- **用户永远优先。** `agent.yml` 里用户的 `deny` 规则压过 plugin 的 `allow`
  (无论声明顺序,判定固定为 deny > ask > allow),且 plugin 声明永远改不了
  profile 的默认姿态。
- **`ask` 规则可按项目放宽。** agent 命中你声明的 `ask` 子命令时,确认框会提供
  "allow (project)" 选项——选择后把命中的 pattern 原样(如
  `datus hello config set:*`)持久化到项目 `.datus/config.yml` 的 `bash_allow`
  列表,此后该子命令直接放行。授权是精确匹配:不会扩大到命名空间的其他部分,
  你的 `deny` 规则也不受影响。(用户自己写在 `agent.yml` 里的 `ask` 规则没有
  这个选项——放宽自己的姿态应该改自己的配置。)
- **作用范围**:只管 agent 的 bash 工具。人类在终端直接敲 `datus hello ...` 不受
  任何影响。`plugins_enabled: false` 会连同 plugin 系统一起停用规则收集
  (见[介绍](introduction.zh.md#disabling-the-plugin-system))。
- **`--profile` 对匹配透明。** `datus hello --profile prod config set x` 与
  不带该旗标的形式命中同样的规则(和同样的项目授权)——前导的 datus 全局旗标
  会在判定前被归一化掉。`--config <path>` 有意*不*归一化:指向另一个配置文件
  等于换绑凭据,这类调用一律回落到确认。
- 声明格式有误(类型不对、未知键、空 pattern)只会记日志并跳过,绝不会影响
  Datus 启动。

建议:只读子命令在 `normal` 下声明为 `allow`,会改状态的声明为 `ask`;仅当重复执行
无害时,才在 `auto` 下把常规状态变更提升为 `allow`。

## 工具参数 transformer {#tool-transformers}

类级 `tool_transformers()` 让 plugin 拦截 **agent 的工具调用**——在工具执行前检查并
改写参数,或者直接拒绝这次调用。典型用例是 SQL 策略:给每条 `execute_sql` 查询
追加租户作用域谓词,值来自部署侧注入的请求 principal。

```python
class ScopedSqlPlugin:
    @classmethod
    def tool_transformers(cls):
        return {"db_tools.execute_sql": enforce_tenant_scope}


def enforce_tenant_scope(tool_name, args, context):
    tenant_id = (context.get("principal") or {}).get("tenant", {}).get("id")
    if not tenant_id:
        raise PermissionError("missing principal.tenant.id; cannot scope query")
    args["sql"] = add_where_predicate(args["sql"], f"tenant_id = '{tenant_id}'")
    return args
```

语义:

- **声明形状**:dict,键为工具 pattern,值为单个 transformer 或 transformer 列表。
  pattern 使用 proxy 语法——裸工具名(`execute_sql`),或带 fnmatch glob 的
  `category.method`(`db_tools.*`)。
- **transformer 签名**:`transformer(tool_name, args, context) -> dict`,同步或
  异步均可。返回(可能已修改的)参数 dict 则继续执行。**抛异常即拒绝**:工具
  不会执行,模型收到你的异常消息(以普通工具失败的形式)。返回非 dict 同样
  按拒绝处理,fail-closed。
- **`context`** 是普通 dict,含 `node_name`、`principal`(请求级调用方属性,
  部署未注入时为空)、`project_root` 与 `agent_config`(运行中的 agent 配置
  对象——通过 `context["agent_config"].get_plugin_profile("<name>")` 读取
  自己的 profile;只做鸭子类型访问,不要为此 import `datus.*`)。每次调用
  都会重建,请求级数据永远是新鲜的。
- **覆盖范围**:transformer 包装的是 agent 的 `FunctionTool` 层,两条执行路径
  (SDK Runner 与 native loop)都经过它。但它**不覆盖**对工具方法的 Python
  直接调用(如 reference-template 执行),也不覆盖代理给外部客户端执行的工具——
  必须在这些路径上也生效的服务端强制,应放在工具层本身(见 `agent.sql_policy`)。
- **信任模型**:transformer 在进程内运行,能看到所有命中工具调用的完整参数,
  属于受信代码;与 plugin 系统其余表面一样受 `plugins_enabled` 总开关门控。
- 改写 SQL 时请使用 SQL parser 或数据库安全的查询构造器——策略谓词绝不要用
  字符串拼接。
- 声明格式有误(非 dict、非 callable 条目、空 pattern)只会记日志并跳过,绝不
  影响 Datus 启动。但收集成功后应用失败会直接中止 agent 节点,而不是在丢失
  强制的情况下静默运行。

## 自带 setup skill {#setup-skill}

`pip install` 之后手工改 YAML 是最大的使用摩擦。在主 skill 旁边再带一个 `<name>-setup`
skill,让 agent 替用户收集配置并写入 profile:

```
datus_plugin_hello/
└── skills/
    ├── hello/
    │   └── SKILL.md
    └── hello-setup/
        └── SKILL.md
```

setup 的 `SKILL.md` 按顺序覆盖:

1. **何时使用**——plugin 未配置,或用户要新增环境。
2. **配置结构**——`agent.plugins.<name>.<profile>` 的完整 YAML 模板,用注释标注
   必填 / 可选 / 敏感字段。
3. **询问用户**——列出必须由用户提供的字段(endpoint、认证方式等)。密钥类字段要指示
   agent 让用户自己导出环境变量,YAML 里写 `${VAR}` 占位——绝不把明文密钥写入文件。
4. **写入配置**——写到 prompt 的 `## Plugins` 前导块中给出的 config 文件,第一个
   profile 标 `default: true`。
5. **验证**——用一条低成本只读命令(如 `datus hello version`)。`datus <plugin>`
   每次调用都会重新加载配置,新 profile 立即可用;prompt 里的环境列表下次会话刷新。

加一条守护说明:若当前环境不可编辑 config 文件(API / VSCode / web 部署),
agent 应改为告知用户在服务端手工编辑 `agent.yml`。

一个完整的最小 `hello-setup/SKILL.md`:

````markdown
---
name: hello-setup
description: Configure an environment profile for the `datus hello` plugin
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

`pip install -e` 之后,每个功能面都可以直接检查,无需重启任何东西
(plugin 在每次调用时被发现):

- **CLI 分发**——在任意目录执行 `datus <name> ...`。如果它落进了 REPL 而不是你的
  plugin,说明 entry point 缺失或名字不对;用 `pip show -f your-package` 检查
  `entry_points.txt`。
- **Skills**——启动 `datus` 并执行 `/skill list`;plugin 自带的 skill 会与项目、
  用户 skill 并列出现。
- **Prompt 注入**——最简单的检查是单测直接调用 `system_prompt()`(见下一节)。
  要确认它进入了真实会话,启动 `datus` 后问 agent"配置了哪些 plugin?"——答案
  就来自注入的段落。注意:配置修改对下一次 `datus <plugin>` 调用立即生效,但
  prompt 段落要到下一个会话才刷新。

## 测试你的 plugin

因为 Datus 是 broker,单测只需用普通 dict 构造你的 plugin——不需要 `agent.yml`,不导入 Datus:

```python
from datus_plugin_hello.plugin import HelloPlugin

def test_run_cli_uses_profile_greeting(capsys):
    rc = HelloPlugin(profile={"name": "prod", "greeting": "Hi"}).run_cli(["Ada"])
    assert rc == 0
    assert "Hi, Ada!" in capsys.readouterr().out

def test_system_prompt_lists_envs_without_secrets():
    text = HelloPlugin.system_prompt({
        "prod": {"name": "prod", "greeting": "Hi", "token": "s3cr3t"},
    })
    assert "## Hello" in text
    assert "s3cr3t" not in text          # 密钥绝不能泄漏

def test_system_prompt_unconfigured_points_to_setup_skill():
    text = HelloPlugin.system_prompt({})
    assert "not configured" in text
    assert "hello-setup" in text
```

## 约束自检清单

发布前逐项确认:

- [ ] 包内任何地方都**不** `import datus`(`grep -rn "import datus" your_pkg/`)。
- [ ] `pyproject.toml` **不**依赖 `datus` 或任何共享 plugin SDK。
- [ ] `__init__` 以名为 `profile` 的关键字参数接收 profile(Datus 调用 `PluginClass(profile=...)`)。
- [ ] entry-point 名不是保留字(`upgrade`、`skill`),且不以 `-` 开头。
- [ ] `skills_dir`、`system_prompt` 与 `cli_permissions` 类级可取(`@classmethod` / `@staticmethod` / 类属性)。
- [ ] `system_prompt` 只输出非敏感字段。
- [ ] `cli_permissions` 的 pattern 相对命名空间书写(不要自带 `datus <name>` 前缀——Datus 会加),且改状态的子命令在 `normal` 下是 `ask`。
- [ ] `run_cli` 返回 int(或 `None`),成功路径上不调用 `sys.exit()`。
- [ ] skill 文件已打进 wheel。
- [ ] `datus.plugins` 的 entry-point 名与目标 `datus <name>` 命令、`agent.plugins.<name>` 配置键一致。

## 参考

- **Entry-point 组**:`datus.plugins`——一个 plugin 一条,解析到一个 plugin **类**。
- **契约权威来源**:`datus/plugins/base.py`(文档化的 `DatusPlugin` 协议)。
- **相关**:[Plugin 介绍](introduction.zh.md)、[Skills](../skills/introduction.zh.md)。
