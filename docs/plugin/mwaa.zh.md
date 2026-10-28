# MWAA 插件

MWAA 插件（`datus-mwaa-plugin`）把 Datus agent 连接到 Amazon Managed Workflows
for Apache Airflow。它工作在环境层面：查看 MWAA 环境、签发 Airflow UI 登录链接和
CLI 令牌、列出环境当前运行的 DAG 并读取源码，以及把 Airflow CLI 命令透传给 MWAA
的 REST 端点。创建、修改或删除环境不在其范围内。

## 安装

```bash
datus plugin install datus-mwaa-plugin
```

要求 datus-agent >= 0.3.8。AWS 凭据按标准 boto3 链解析，也可以通过下面
`mwaa-setup` 收集的 profile 字段提供。其他安装来源和 profile 管理方式见
[插件](introduction.zh.md)。

## Skills

| Skill | 作用 |
|---|---|
| `mwaa` | 查看环境、签发令牌、读取当前 DAG 及其源码 |
| `mwaa-setup` | 创建 profile（区域、凭据、默认环境） |
| `mwaa-dag-export` | 在明确确认范围后导出环境的 DAG 源码 |

### mwaa

借助这个 skill，agent 可以：

- 列出 MWAA 环境并查看详情——包括 Airflow 版本、状态、Web 服务器地址，以及环境
  读取 DAG 的 S3 桶和路径；
- 签发一次性的 Airflow UI 登录链接，或 CLI 令牌加 Web 服务器主机名；
- 列出环境当前运行的 DAG 并读取某个 DAG 的源码。两者都通过短时 MWAA Web 会话
  直接调用环境的 Airflow REST API——从不读取 S3 桶；
- 通过 MWAA 的 REST 端点执行 Airflow CLI 命令。这是不透明的透传——被包裹的命令
  可能有破坏性——因此 agent 总是先请求确认，而且 MWAA 并不支持所有 Airflow
  子命令。日常 DAG 操作建议改用指向该环境的 [Airflow 插件](airflow.zh.md)。

### mwaa-setup

让 agent 帮你配置插件，这个 skill 会收集 AWS 区域、凭据来源（默认 AWS 链、命名
profile、以 `${ENV_VAR}` 引用的密钥，或要 assume 的角色），以及可选的默认环境名，
然后通过列出环境来验证 profile。IAM 主体需要 `airflow:ListEnvironments`、
`airflow:GetEnvironment`、`airflow:CreateWebLoginToken` 和
`airflow:CreateCliToken` 四项权限。生成的 profile 形如：

```yaml
agent:
  plugins:
    mwaa:
      prod:
        default: true
        region: us-east-1
        environment: prod-airflow  # 可选的默认环境
        # 凭据：标准 AWS 链，或 profile / 密钥 / role_arn
```

### mwaa-dag-export

Airflow 插件导出工作流的 MWAA 对应版本，保证完全一致：以环境的 Airflow API 为
唯一权威来源（skill 绝不枚举 MWAA 的 S3 DAG 前缀），范围可用自然语言反复调整、
每次调整后重新计算，在你确认当前方案之前不写入、不上传，导出结果附带含校验和的
`dag-export-manifest.json`，绝不包含凭据或令牌。上传按目的地 URI 路由——
`s3://` 走 [S3 插件](s3.zh.md)。

## 在 agent 中使用

配置好 profile 后，可以这样提出请求：

- **「给我一个 prod MWAA UI 的登录链接。」** —— 返回一次性 web-login URL。
- **「prod 环境现在跑着哪些 DAG？把 sales_daily 的源码给我看看。」**
- **「看下 analytics-airflow 这个环境的详情——它从哪个桶读 DAG？」**
- **「帮我在 us-east-1 配置 mwaa 插件。」** —— 触发 `mwaa-setup`。

以上查看类操作都不需要确认；只有 Airflow CLI 透传总是先确认，因为它的内容对
[权限系统](introduction.zh.md#permissions)来说是不透明的。

## 编排工作流

### 用 Airflow 插件做细粒度 DAG 操作

MWAA 插件管环境，[Airflow 插件](airflow.zh.md)管 DAG。触发、暂停、清理、看日志
这些带权限分级的操作，应该给 Airflow 插件配置一个指向 MWAA 环境 Web 服务器的
profile——主机名和令牌由本插件提供。直接让 agent 把两者接起来：

```text
把 airflow 插件指向 prod MWAA 环境，然后触发 sales_daily 并等待运行结束。
```

### 用 S3 插件部署 DAG

MWAA 环境从环境详情里显示的 S3 桶和前缀读取 DAG。MWAA 插件不含 S3 传输能力，
所以部署是一个组合动作：agent 先通过 [S3 插件](s3.zh.md)上传文件，再通过本插件
验证 DAG 出现：

```text
把 dags/sales_daily.py 上传到 prod MWAA 环境的 DAG 目录，并确认 DAG 已经出现。
```

反过来，「把 prod MWAA 环境的所有 DAG 导出到 s3://backup/mwaa/2026-08-24/」会
运行 `mwaa-dag-export`——源码始终来自 Airflow API，上传由 S3 插件完成。

## 相关文档

- [插件](introduction.zh.md) —— 安装来源、profile、项目启用与权限
- [Airflow 插件](airflow.zh.md) —— 对同一环境做细粒度 DAG 操作
- [S3 插件](s3.zh.md) —— 向环境的桶上传 DAG
- [Skills](../skills/introduction.zh.md) —— skills 的发现和加载方式
