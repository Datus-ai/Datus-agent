# Airflow 插件

Airflow 插件（`datus-airflow-plugin`）通过 Airflow 的稳定 REST API，把 Datus agent
连接到远端的 Apache Airflow 2.x 或 3.x 部署，Datus 所在机器不需要安装任何 Airflow
组件。配置好 profile 之后，你就可以在对话里用自然语言让 agent 查看、触发和排查
DAG，管理 variables、connections 和 pools，以及导出或部署 DAG 源码。

## 安装

```bash
datus plugin install datus-airflow-plugin
```

要求 datus-agent >= 0.3.8。安装完成后，自带的 skills 会出现在 `/skill list` 中，
会话开始时 agent 会自动发现插件及其已配置的环境。其他安装来源和 profile 管理方式
见[插件](introduction.zh.md)。

## Skills

| Skill | 作用 |
|---|---|
| `airflow` | 对已配置的 Airflow 环境做日常操作 |
| `airflow-setup` | 通过对话引导创建环境 profile |
| `airflow-dag-export` | 在明确确认范围后导出或部署 DAG 源码 |

### airflow

核心操作 skill。借助它，agent 可以：

- 列出并查看 DAG、运行记录、任务状态和任务日志；
- 触发 DAG 运行（可以等待运行结束），清理失败的运行或任务以便重跑；
- 暂停和恢复 DAG，在部署后检查 import 错误；
- 管理实例级的 variables、connections 和 pools；
- 创建和跟踪 backfill，读取服务端版本与健康状态（assets 和 backfill API 需要
  Airflow 3）。

Profile 可以设置作用域护栏：`dag_id_prefix` 把所有针对 DAG 的操作限制在匹配前缀的
DAG 上，`allow_commands` 限制可用的命令组。这些限制会出现在 agent 的上下文里，
agent 会主动遵守，而不是靠失败来试探边界。它们用来防止误操作，不是安全边界——
真正的租户隔离仍然要靠 Airflow 服务端。

### airflow-setup

用对话完成配置。让 agent 帮你配置插件，这个 skill 会收集 Airflow Web 服务器地址、
认证方式（静态 API token，或用户名加密码），以及 `dags_folder` 部署 URI、上述作用域
护栏等可选项。密钥一律写成 `${ENV_VAR}` 引用，绝不写明文；写好后 skill 会用一次
只读调用验证 profile 可用。生成的 profile 形如：

```yaml
agent:
  plugins:
    airflow:
      prod:
        default: true
        api_base_url: https://airflow.example.com/api/v1  # /api/v1 对应 Airflow 2，/api/v2 对应 Airflow 3
        username: admin
        password: ${AIRFLOW_PASSWORD}
        dags_folder: s3://my-bucket/dags/  # 可选的部署 URI；
                                           # 存储凭据属于 s3 插件，不写在这里
```

### airflow-dag-export

一个必须经过明确确认的工作流，用于导出、备份、迁移或部署 DAG 源码：

1. **发现** —— 以线上 Airflow API 为唯一权威来源。skill 通过 API 列出活跃 DAG
   集合，逐个获取 Python 源码，绝不去扫描调度器背后的 `dags_folder` 存储。
2. **提议与调整** —— 给出一份完整方案：环境、选中的 DAG、文件、目的地和传输
   方式。你可以用自然语言调整范围——按 DAG id、通配或正则、owner、tag、暂停
   状态、源码关键字或引用的 connection id 过滤——每次调整后它都会重新计算并
   展示方案。
3. **确认后写入** —— 在你确认当前这份方案之前，不会向目的地写任何东西。导出
   结果附带 `dag-export-manifest.json`，记录每个 DAG、文件和校验和，绝不包含
   凭据。
4. **上传** —— 由目的地 URI 决定传输方式：本地路径直接复制，`s3://` 走
   [S3 插件](s3.zh.md)，`gs://` 和 `abfs://` 分别走 GCS 和 ADLS 插件。Airflow
   插件自身不含任何对象存储客户端。

## 在 agent 中使用

配置好 profile 后，可以这样提出请求：

- **「prod 里昨晚哪些 DAG 失败了？」** —— agent 会列出最近失败的运行，逐层查看
  任务状态，并调出失败任务的日志。
- **「触发 sales_daily，等它跑完告诉我结果。」** —— agent 先请求确认，然后启动
  运行并轮询到成功或失败。
- **「把昨天 sales_daily 那次运行里失败的任务清掉重跑。」**
- **「给 sales_daily 补跑一月第一周的数据，先 dry-run。」**
- **「帮我给 staging 集群配置 airflow 插件。」** —— 触发 `airflow-setup`。

Agent 执行的命令都经过 Datus [权限系统](introduction.zh.md#permissions)：只读操作
直接执行；常规可逆操作（暂停、清理运行、设置 variable）在 `normal` 模式下需要
确认一次；启动运行、删除对象、批量导入、涉及 connection 密钥的操作任何模式下都
必须先确认。

## 编排工作流

### 用 S3 插件部署 DAG

当调度器从对象存储同步 DAG（生产环境的常见做法）时，把本插件和
[S3 插件](s3.zh.md)配合使用：

1. 把 profile 的 `dags_folder` 指向调度器读取的桶，例如
   `s3://my-bucket/dags/`。存储凭据配置在 S3 插件自己的 profile 里，Airflow
   插件不会接触它们。
2. 让 agent 部署。它会通过 S3 插件上传 DAG 文件，然后通过 Airflow API 轮询
   DAG 列表和 import 错误，直到新 DAG 被解析成功。
3. 在同一段对话里继续：触发 DAG，跟踪运行结果。

也可以一句话交代整个流程：

```text
创建 dags/hello_world.py，里面一个任务打印运行日期；用 s3 插件把它上传到
Airflow 的 DAG 根目录，确认 DAG 出现后触发它，并等待运行完成。
```

同样的组合反过来就是备份和迁移——「把 prod 的所有活跃 DAG 导出到
s3://backup/airflow/2026-08-24/」会驱动 `airflow-dag-export`，由 S3 插件承担
传输。

### 托管在 Amazon MWAA 上的 Airflow

如果 Airflow 托管在 Amazon MWAA 上，[MWAA 插件](mwaa.zh.md)负责环境本身——登录
链接、令牌、环境详情——再把本插件指向该环境的 Web 服务器，就能获得上面这套
细粒度、带权限分级的 DAG 操作能力。

## 相关文档

- [插件](introduction.zh.md) —— 安装来源、profile、项目启用与权限
- [S3 插件](s3.zh.md) —— `s3://` DAG 部署背后的传输层
- [MWAA 插件](mwaa.zh.md) —— Amazon 托管的 Airflow 环境
- [数据工程快速开始](../getting_started/data_engineering_quickstart.zh.md) —— 包含发布每日 Airflow DAG 的端到端流水线
- [Skills](../skills/introduction.zh.md) —— skills 的发现和加载方式
