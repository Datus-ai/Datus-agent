# EKS 插件

EKS 插件（`datus-eks-plugin`）让你通过自然语言要求 Datus agent 查看 Amazon Elastic
Kubernetes Service 控制面，并为 [K8s 插件](k8s.zh.md)提供短期 Kubernetes 认证。
它直接使用 AWS SDK，不需要你生成或维护 kubeconfig，也不会把短期 bearer token
展示给模型。

首个版本的 EKS 运维能力全部是只读的：Agent 可以查看集群、node group、add-on、
access entry、Fargate profile、更新和升级 insight，但不能创建、修改或删除这些
AWS 资源。

## 安装和启用

```bash
datus plugin install datus-eks-plugin
```

如果还需要操作集群内的 namespace 工作负载，同时安装 K8s plugin：

```bash
datus plugin install datus-k8s-plugin
```

安装完成后，在当前项目启用所需插件。其他安装来源和项目启用方式见
[插件](introduction.zh.md)。

## Skills

| Skill | 作用 |
|---|---|
| `eks` | 查看 EKS 集群、计算资源、访问配置、更新和当前 AWS 身份 |
| `eks-setup` | 创建 EKS 环境并安全验证连接 |

### eks

核心 skill 从环境配置取得固定的 cluster name、region 和 AWS 认证。Agent 会先确认
当前 caller identity 与目标集群，再按问题读取 node group、add-on、access entry、
Fargate profile、update 或 upgrade insight。它会优先给出紧凑摘要，需要完整字段时
再展示脱敏后的 AWS 响应。

### eks-setup

设置 skill 会询问环境名、EKS cluster name、region 和认证来源。支持标准 AWS
credential chain、`~/.aws/config` 中的命名 profile、环境变量引用的临时密钥，以及
AssumeRole。Agent 只保存 `${ENV_VAR}` 引用，不会写入明文 credential 或 ExternalId。

## 配置 EKS 环境

推荐使用标准 AWS 链或 AssumeRole：

```text
帮我配置 analytics-prod EKS：集群是 analytics-prod-eks（us-east-1），使用
engineering AWS profile 和 datus-eks-operator role。
```

Agent 会先复述将要保存的非敏感字段。缺少 ExternalId 环境变量名、源 profile 或
region 等必要信息时，它会在写配置前询问。验证结果会明确显示：

- 当前 AWS account、principal ARN 和 AssumeRole 后的身份；
- 目标 EKS cluster name、region、状态和 Kubernetes 版本；
- endpoint 与 CA 是否可供 K8s provider 使用；
- 调用失败时，是凭据、IAM、region、cluster name 还是网络问题。

仅在标准链或 AssumeRole 无法使用时，才提供临时密钥的环境变量名：

```text
给 analytics-temp 使用现有的 AWS 临时凭据环境变量。
```

## 查看 EKS 环境

| 你的请求 | Agent 的行为 |
|---|---|
| “analytics-prod 是否健康？” | 确认 caller identity，读取集群状态、Kubernetes 版本、endpoint 和网络摘要 |
| “这个集群有哪些 node group？” | 列出 node group；指定一个后再展示容量、实例类型、版本和健康问题 |
| “关键 add-on 是否正常？” | 汇总 add-on 版本与状态，并指出降级或失败信息 |
| “这个 IAM role 能进入集群吗？” | 查看匹配的 access entry 及其访问策略，不尝试创建或修改 |
| “升级前有什么风险？” | 查看 upgrade insight 和近期 update，按状态与类别归纳待处理项 |
| “是否使用了 Fargate？” | 列出 Fargate profile 及 selector、subnet 和状态摘要 |

当你只要求概览时，Agent 不会倾倒完整 AWS 响应；需要某个资源的全部字段时，可以
明确提出：

```text
帮我看下 analytics-prod 的访问配置和网络。
```

发现缺少 access entry、add-on 或升级修复时，Agent 会报告对象、状态和管理员需要
采取的动作，不会直接创建或更新 AWS 资源。

## 作为 K8s provider

让 Agent 同时配置云端与 namespace 环境：

```text
让 analytics-prod K8s 使用同名 EKS，默认访问 analytics，也允许
analytics-staging。
```

Agent 会在内部读取 EKS endpoint 和 CA，并为每次 Kubernetes 连接获取短期凭据。
EKS 与 K8s 环境同名时自动关联；名称不同时，你可以在请求中明确 provider 环境名。

EKS access entry 只决定 AWS identity 能否进入 Kubernetes 认证链。是否能读取 Pod、
部署 `FlinkDeployment` 或访问某个 namespace，仍由 Kubernetes RBAC 决定。Agent 会
分别报告 IAM、access entry 和 RBAC 三层结果，避免把其中一层成功误判为完整授权。

## 与 Flink 部署配合

[Flink 插件](flink.zh.md)部署作业前，可以让 Agent 用 EKS plugin 确认控制面，再由
K8s plugin 检查 namespace 与 Operator 资源：

```text
帮我确认 analytics-prod 能不能在 flink-jobs 部署 orders-enrichment，先别改资源。
```

EKS plugin 不负责 ECR 登录、仓库或镜像，也不安装 Flink Operator、CRD、webhook
和集群级 RBAC。这些前提缺失时，Agent 会把问题交给相应管理员处理。

## 权限与安全边界

- 所有公开 EKS 运维操作都是只读的，不创建、修改或删除 AWS 资源；
- credential 和 ExternalId 只保存环境变量引用；
- 短期 Kubernetes token 仅供 K8s provider 内部使用，不写入项目或模型输出；
- Agent 不修改 kubeconfig，也不管理 ECR 镜像；
- AWS IAM、EKS access entry 与 Kubernetes RBAC 是三层独立授权；
- EKS plugin 管控制面查询和认证，namespace 工作负载由 K8s plugin 管理；
- Flink Operator、CRD、webhook 和集群级 RBAC 由管理员安装。

## 相关文档

- [K8s 插件](k8s.zh.md) —— 使用 EKS provider 操作 namespace 资源
- [Flink 插件](flink.zh.md) —— 在 EKS 上通过 Operator 部署 Flink 作业
- [插件](introduction.zh.md) —— 安装来源、环境配置、项目启用与权限
- [Skills](../skills/introduction.zh.md) —— skills 的发现和加载方式
