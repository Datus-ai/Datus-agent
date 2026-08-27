# EKS Plugin

The EKS plugin (`datus-eks-plugin`) lets you ask the Datus agent in natural
language to inspect the Amazon Elastic Kubernetes Service control plane and
provide short-lived Kubernetes authentication to the [K8s plugin](k8s.md). It
uses the AWS SDK directly, requires no kubeconfig maintenance, and does not
expose short-lived bearer tokens to the model.

All EKS operational capabilities in the first version are read-only. The agent
can inspect clusters, node groups, add-ons, access entries, Fargate profiles,
updates, and upgrade insights, but it cannot create, change, or delete those
AWS resources.

## Installation and Activation

```bash
datus plugin install datus-eks-plugin
```

Also install the K8s plugin when the project needs namespace workload
operations:

```bash
datus plugin install datus-k8s-plugin
```

Enable the required plugins for the current project after installation. See
[Plugins](introduction.md) for other installation sources and project
activation.

## Skills

| Skill | Purpose |
|---|---|
| `eks` | Inspect EKS clusters, compute, access configuration, updates, and the current AWS identity |
| `eks-setup` | Create an EKS environment and verify it safely |

### eks

The core skill takes the fixed cluster name, region, and AWS authentication
from the environment configuration. The agent confirms the caller identity and
target cluster first, then reads node groups, add-ons, access entries, Fargate
profiles, updates, or upgrade insights as needed. It prefers a compact summary
and presents a redacted AWS response only when complete fields are needed.

### eks-setup

The setup skill asks for an environment name, EKS cluster name, region, and
authentication source. It supports the standard AWS credential chain, a named
profile in `~/.aws/config`, temporary keys referenced through environment
variables, and AssumeRole. The agent stores `${ENV_VAR}` references only and
never writes literal credentials or an ExternalId.

## Configuring an EKS Environment

Prefer the standard AWS chain or AssumeRole:

```text
Configure analytics-prod for the analytics-prod-eks cluster in us-east-1,
using the engineering AWS profile and the datus-eks-operator role.
```

The agent restates the non-secret fields before saving them. If an ExternalId
variable name, source profile, region, or other required detail is missing, it
asks before writing configuration. Verification reports:

- the current AWS account, principal ARN, and post-AssumeRole identity;
- target EKS cluster name, region, status, and Kubernetes version;
- whether the endpoint and CA are available to the K8s provider; and
- whether a failure comes from credentials, IAM, region, cluster name, or
  connectivity.

Use temporary-key environment variables only when the standard chain or
AssumeRole is unavailable:

```text
Use the existing AWS temporary-credential environment variables for
analytics-temp.
```

## Inspecting an EKS Environment

| Your request | Agent behavior |
|---|---|
| "Is analytics-prod healthy?" | Confirm caller identity and read cluster status, Kubernetes version, endpoint, and a network summary |
| "Which node groups does this cluster have?" | List node groups, then show capacity, instance types, version, and health issues for the selected one |
| "Are the critical add-ons healthy?" | Summarize add-on versions and states and surface degraded or failed details |
| "Can this IAM role enter the cluster?" | Inspect the matching access entry and access policies without trying to create or change them |
| "What could block the next upgrade?" | Inspect upgrade insights and recent updates and group open work by status and category |
| "Does this cluster use Fargate?" | List Fargate profiles and summarize selectors, subnets, and state |

For an overview, the agent does not dump complete AWS responses. Ask
explicitly when complete fields are necessary:

```text
Show me the access configuration and networking for analytics-prod.
```

When an access entry, add-on, or upgrade remediation is missing, the agent
reports the object, state, and administrator action rather than creating or
updating the AWS resource.

## Using EKS as a K8s Provider

Ask the agent to configure the cloud and namespace environments together:

```text
Make analytics-prod K8s use the matching EKS environment, with analytics as
the default namespace and analytics-staging also allowed.
```

The agent reads the EKS endpoint and CA internally and obtains a short-lived
credential for each Kubernetes connection. Matching EKS and K8s environment
names link automatically; when names differ, state the provider environment in
your request.

An EKS access entry only places an AWS identity in the Kubernetes
authentication path. Kubernetes RBAC still determines whether it may read
Pods, deploy a `FlinkDeployment`, or access a namespace. The agent reports IAM,
access-entry, and RBAC results separately instead of treating one successful
layer as complete authorization.

## Working with Flink Deployment

Before the [Flink plugin](flink.md) deploys a job, ask the agent to use the EKS
plugin for control-plane checks and the K8s plugin for namespace and Operator
resources:

```text
Can analytics-prod deploy orders-enrichment in flink-jobs? Check without
changing anything.
```

The EKS plugin does not manage ECR login, repositories, or images, and does not
install the Flink Operator, CRDs, webhook, or cluster-level RBAC. When one of
those prerequisites is missing, the agent reports the appropriate
administrator action.

## Permissions and Security Boundaries

- Every public EKS operation is read-only and does not create, update, or
  delete AWS resources.
- Credentials and ExternalId values are stored only as environment-variable
  references.
- Short-lived Kubernetes tokens are used internally by the K8s provider and
  are not persisted or exposed to the model.
- The agent does not mutate kubeconfig or manage ECR images.
- AWS IAM, EKS access entries, and Kubernetes RBAC are three independent
  authorization layers.
- The EKS plugin owns control-plane inspection and authentication; the K8s
  plugin owns namespace workload operations.
- An administrator installs the Flink Operator, CRDs, webhook, and
  cluster-level RBAC.

## Related Docs

- [K8s plugin](k8s.md) — operate namespace resources through the EKS provider
- [Flink plugin](flink.md) — deploy Flink jobs to EKS through the Operator
- [Plugins](introduction.md) — install sources, environment configuration, activation, and permissions
- [Skills](../skills/introduction.md) — how skills are discovered and loaded
