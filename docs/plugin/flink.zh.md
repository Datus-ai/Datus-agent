# Flink 插件

Flink 插件（`datus-flink-plugin`）让 Datus agent 帮你完成 Apache Flink 作业的本地
验证、SQL 打包，以及基于 Apache Flink Kubernetes Operator 的部署和运维。它是一个
skill-only 插件：你只需要描述目标，Agent 会调用相应 skill，并把所有 Kubernetes
资源操作交给 [K8s 插件](k8s.zh.md)；目标是 Amazon EKS 时，再由
[EKS 插件](eks.zh.md)提供集群发现和短期认证。

本文重点介绍在已经安装 Flink Kubernetes Operator 的 EKS 集群上部署 Flink 作业。
Operator、CRD、webhook 和集群级 RBAC 的安装属于 Kubernetes 管理员职责，不由这些
插件执行。

## 安装和启用

按依赖顺序安装三个插件：

```bash
datus plugin install datus-eks-plugin
datus plugin install datus-k8s-plugin
datus plugin install datus-flink-plugin
```

安装完成后，在当前项目启用这些插件。其他安装来源和项目启用方式见
[插件](introduction.zh.md)。

## Skills

| Skill | 作用 |
|---|---|
| `flink-local-dev` | 在本机 MiniCluster 中验证 Flink SQL，不写生产 sink |
| `flink-sql` | 为 Application Mode 选择并验证与 Flink 版本匹配的 SQL 入口 |
| `flink-k8s-operator` | 构建、部署、升级和排查 Operator 管理的 Flink 作业 |

### flink-local-dev

当你要求在本地验证 Flink SQL 时，Agent 会保持生产 SQL 不变，通过单独的本地
overlay 把 source 限制在开发环境，并把每个 sink 替换为 `print`、`blackhole` 或
本地文件表。缺少本地执行约束、存在未替换的生产 sink，或本地凭据可能被 Git 跟踪
时，它会拒绝运行并说明原因。

```text
帮我在本地验证 orders/job.sql，别写入生产环境。
```

### flink-sql

验证通过后，让 Agent 为生产环境确定 SQL 入口。Flink 1.x 通常需要版本匹配的
runner JAR；Flink 2.x 可以使用系统 classpath 中的 `SqlDriver`。Agent 会检查 SQL、
connector 和 filesystem 依赖，并把确定的镜像及作业字段交给
`flink-k8s-operator`，不会为了填充字段而虚构 JAR。

```text
把这个 SQL 准备成 Flink 1.20 的生产作业。
```

### flink-k8s-operator

这个 skill 支持 Application Cluster、Session Cluster、`FlinkSessionJob` 和
`FlinkStateSnapshot`。你提供业务作业和目标环境，Agent 负责构建、发布前检查、
自定义资源方案、部署、状态观察，以及安全的暂停、恢复、升级和删除流程。

## 在 EKS 上部署 Flink 作业

### 1. 检查基础设施前提

先让 Agent 做只读检查：

```text
帮我看看 flink-prod 能不能在 flink-jobs 部署 Flink 作业，先别做修改。
```

Agent 会确认：

- EKS 集群是否健康，当前 AWS 身份是否指向预期账号和集群；
- 目标 namespace 是否在 K8s profile 的允许范围内；
- 集群是否提供所需版本的 `FlinkDeployment`，Session 模式是否提供
  `FlinkSessionJob`，快照流程是否提供 `FlinkStateSnapshot`；
- Datus 使用的 Kubernetes 身份能否读取和创建目标自定义资源；
- Flink 作业 service account 是否拥有管理 TaskManager 所需的 Pod、Service、
  ConfigMap 权限，以及读取 Deployment owner reference 的权限；
- 作业需要访问 S3 等 AWS 服务时，service account 是否已配置 IRSA；
- 作业镜像 registry 是否能从 EKS 工作节点访问。

如果缺少 Operator、CRD、webhook、access entry 或集群级 RBAC，Agent 会列出证据和
管理员需要完成的事项，不会尝试绕过边界或自行安装。

### 2. 配置 EKS 与 K8s 环境

如果 profile 尚不存在，用自然语言给出集群和 namespace 信息：

```text
帮我配置 flink-prod：EKS 集群是 analytics-prod（us-east-1），使用
datus-flink-operator role，namespace 是 flink-jobs。
```

Agent 会分别运行 `eks-setup` 和 `k8s-setup`，建立同名且自动关联的 profile。它会
展示非敏感配置摘要，并验证目标，但不会输出短期 Kubernetes token 或把明文密钥
写入配置。缺少必须信息时，Agent 会在写入前询问你。

### 3. 验证并发布作业镜像

SQL 作业先经过 `flink-local-dev` 和 `flink-sql`；JVM 或 PyFlink 项目则使用项目已有
的测试与构建方式。告诉 Agent 目标版本和 registry，同时要求不可变镜像标识：

```text
为 orders-enrichment 构建 Flink 1.20 生产镜像并发布到现有 ECR，推送前问我。
```

EKS plugin 不负责 ECR 登录、仓库创建或镜像推送。Agent 只能使用已经准备好的
registry 与认证，并且在推送这个外部变更前请求确认。

### 4. 生成部署方案

让 Agent 先规划，不要直接写入集群：

```text
帮我规划 orders-enrichment 在 flink-prod 的部署，先给我方案，不要发布。
```

Agent 会把方案写到项目的 `deploy/flink/<name>/` 下，并明确展示：

- 集群实际提供的 Flink API version；
- 镜像、Flink version、入口类或脚本与参数；
- Application 或 Session 模式、并行度和计算资源；
- namespace、service account、checkpoint/savepoint 存储；
- `stateless`、`savepoint` 或 `last-state` 升级策略；
- 引用的 Secret、IRSA 和 image pull 配置，不包含任何明文凭据。

首次部署默认使用 `stateless`。只有作业已进入稳定状态、持久 checkpoint 与 HA 前提
已验证后，才适合改为 `savepoint` 或 `last-state`。

### 5. 校验并确认部署

```text
先验证这个部署方案，没问题再问我要不要发布。
```

Agent 会通过 K8s plugin 做 Server-Side Apply dry-run。校验成功只说明资源格式和当前
身份可接受该请求，不代表 Flink 作业已经启动。你确认后，Agent 才会正式提交资源。

### 6. 观察运行结果

```text
发布后帮我等到它正常运行，失败就告诉我原因。
```

Agent 使用有界的状态读取，不会无限等待。它会区分：

- 自定义资源已创建；
- JobManager 已就绪；
- Flink job 真正达到 `RUNNING`，或有界批作业达到 `FINISHED`；
- `status.error`、Pod 状态和 warning event 是否显示失败。

不能只因为资源存在或 JobManager 为 `READY` 就宣告成功。

### 7. 诊断失败

```text
orders-enrichment 没跑起来，帮我查下原因，先不要改东西。
```

Agent 会优先读取最小必要状态字段，并在需要时查看 init container 或上一实例日志。
修复应进入镜像、manifest、Secret 或基础设施配置，而不是临时修改运行中的 Pod。

## 生命周期管理

| 你的请求 | Agent 的行为 |
|---|---|
| “暂停 orders-enrichment，但保留可恢复状态” | 核对升级模式和状态存储，展示 patch，确认后暂停 |
| “从上次状态恢复作业” | 验证 checkpoint/savepoint 路径、权限和版本兼容性后再恢复 |
| “把镜像升级到新版本” | 修改受版本控制的 manifest，保留已确认的升级策略并展示差异 |
| “为当前作业创建快照” | 创建唯一的 `FlinkStateSnapshot`，等待完成并报告持久化路径 |
| “删除这个有状态作业” | 先询问是否需要最终 savepoint；Session Job 先于 Session Cluster 删除 |

所有会改变镜像仓库、项目文件或集群状态的步骤都会先展示目标和影响，并请求确认。

## 相关文档

- [K8s 插件](k8s.zh.md) —— namespace 资源操作、状态、日志与权限
- [EKS 插件](eks.zh.md) —— EKS 集群发现和短期 Kubernetes 认证
- [插件](introduction.zh.md) —— 安装来源、profile、项目启用与权限
- [Skills](../skills/introduction.zh.md) —— skills 的发现和加载方式
