# K8s 插件

K8s 插件（`datus-k8s-plugin`）让你直接用自然语言要求 Datus agent 查看和操作
namespace 范围内的 Kubernetes 数据工作负载。Agent 可以发现资源、读取状态与日志、
分析事件、检查权限，以及在确认后应用或修改 manifest，包括 Pod、Deployment、Job
和 `FlinkDeployment` 等自定义资源。

插件刻意不提供集群管理员能力：不能跨所有 namespace 查询，也不能操作集群级资源；
每次访问都受所选环境的 namespace allowlist 限制。连接 Amazon EKS 时，它从
[EKS 插件](eks.zh.md)取得集群信息和短期认证。

## 安装和启用

```bash
datus plugin install datus-k8s-plugin
```

连接 EKS 时还要安装 EKS plugin：

```bash
datus plugin install datus-eks-plugin
```

安装完成后，在当前项目启用所需插件。其他安装来源和项目启用方式见
[插件](introduction.zh.md)。

## Skills

| Skill | 作用 |
|---|---|
| `k8s` | 查看、诊断和变更 namespace 内的 Kubernetes 工作负载 |
| `k8s-setup` | 创建 managed-cloud 或 kubeconfig 类型的 K8s 环境 |

### k8s

核心 skill 会先确认环境和 namespace，再按问题选择最小必要的资源状态、事件、日志、
指标或权限检查。对于异步资源，它会设置最长耗时和检查次数，每次同时检查目标状态与
错误，不会在已经出现具体失败后继续盲目等待。

### k8s-setup

设置 skill 会询问 managed Kubernetes provider 或 kubeconfig、默认 namespace 和
允许访问的 namespace。云端身份和短期认证保留在 provider plugin 中，不会复制到
K8s 环境配置。

## 连接 Amazon EKS

告诉 Agent EKS 集群、认证方式和 namespace：

```text
帮我配置 analytics-prod，连接 us-east-1 的 analytics-prod-eks，默认使用 analytics
namespace。
```

Agent 会先使用 `eks-setup` 配置云端环境，再用 `k8s-setup` 创建同名 K8s 环境，使
两者自动关联。它会验证：

- 当前 AWS caller identity 和目标 EKS 集群；
- Kubernetes server 与所选环境是否一致；
- 默认 namespace 是否存在且位于 allowlist；
- 当前 Kubernetes 身份是否拥有请求的 namespace 权限。

短期 bearer token 不会展示给模型或持久化到项目。AWS 身份能够查看 EKS 并不代表
自动拥有 Kubernetes 权限；EKS access entry 与 namespace RBAC 必须分别配置。

如果 EKS 与 K8s 环境需要使用不同名称，可以直接说明关联关系：

```text
让 flink-prod 使用 analytics-prod EKS，只访问 flink-jobs。
```

## 连接 kubeconfig 环境

已有 kubeconfig 时，只把路径和 context 告诉 Agent：

```text
用 ./conf/kubeconfig.yaml 的 staging context 配一个环境，只访问
analytics-staging。
```

Agent 会从当前项目解析相对路径，并拒绝通过 `..` 或 symlink 跳出项目。省略 context
时会使用 kubeconfig 当前选择的 context，但 Agent 会在首次访问前把最终集群身份展示
给你确认。

## 查看和诊断工作负载

| 你的请求 | Agent 的行为 |
|---|---|
| “analytics-prod 里哪些 Pod 不健康？” | 汇总 READY、重启次数和实际等待原因，再查看相关 warning event |
| “为什么 pipeline-api 一直重启？” | 先确认 Pod 和容器状态，再读取上一实例及必要的 init container 日志 |
| “orders 这个 FlinkDeployment 现在是什么状态？” | 分别读取 job state、status.error、JobManager/TaskManager 状态，避免只看单一字段 |
| “哪个 Pod 的内存使用最高？” | 读取 namespace 内 Pod 指标并按内存排序，说明 metrics 不可用时的原因 |
| “当前身份能否创建 Job？” | 对目标 namespace 做授权检查，不尝试通过失败的写操作试探权限 |

资源表会区分 `Running` 与真正 Ready，也会显示 `Init:CrashLoopBackOff`、
`ImagePullBackOff` 等等待原因。Agent 优先读取回答问题所需的单个状态字段；只有需要
理解完整 spec 或 conditions 时才读取整个对象。

## 观察异步资源

请在请求中给出成功条件和最长观察时间：

```text
帮我看着 analytics-staging 的 daily-etl，跑完或失败时告诉我。
```

对于 Kubernetes Deployment、StatefulSet 和 DaemonSet，Agent 可以跟踪 rollout；
对于 Flink 等自定义资源，它会有界地重复读取资源自身的状态与错误。资源已创建、Pod
处于 Running 或 controller 已就绪，不一定等于业务工作负载成功。

## 应用和变更资源

从自然语言描述目标，要求 Agent 先展示 proposal：

```text
先检查 deploy/pipeline.yaml，没问题的话问我是否发布到 analytics-prod。
```

```text
把 analytics 里的 pipeline-api 扩容到 4 个副本。
```

```text
删除 analytics 里的 daily-backfill Job。
```

Agent 使用 Server-Side Apply 处理本地 YAML/JSON manifest。创建、应用、删除、patch、
扩缩容、标签/注解、重启和进入容器检查都需要确认。服务端 dry-run 不持久化资源，但
Agent 仍会清楚说明它正在验证哪个环境和 namespace。

## 权限与安全边界

- 资源发现、状态、详情、日志、事件、指标和授权检查属于只读操作；
- 环境只能访问 allowlist 中的 namespace，未配置时默认为默认 namespace；
- 禁止跨全部 namespace、集群级资源、kubeconfig 修改和身份模拟；
- 不支持交互式 shell、attach、文件复制、port-forward 或 proxy；
- 容器内检查只能是一次性、非交互命令，每次都需要确认；
- Agent 不会在运行容器里临时修补文件或配置，持久修复必须进入镜像或 manifest。

## 与 Flink 插件编排

[Flink 插件](flink.zh.md)的 `flink-k8s-operator` skill 负责理解作业并生成
`FlinkDeployment` 等资源，K8s plugin 负责在目标 EKS namespace 中执行发现、校验、
部署和诊断。

```text
把 orders 发布到 analytics-prod，并帮我跟踪运行结果。
```

Operator、CRD、webhook 和集群级 RBAC 不属于 K8s plugin 的能力范围，必须由管理员
预先安装。

## 相关文档

- [EKS 插件](eks.zh.md) —— EKS 集群发现和 Kubernetes provider 认证
- [Flink 插件](flink.zh.md) —— 通过 Operator 部署和管理 Flink 作业
- [插件](introduction.zh.md) —— 安装来源、环境配置、项目启用与权限
- [Skills](../skills/introduction.zh.md) —— skills 的发现和加载方式
