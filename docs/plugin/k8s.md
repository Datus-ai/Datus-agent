# K8s Plugin

The K8s plugin (`datus-k8s-plugin`) lets you ask the Datus agent in natural
language to inspect and operate namespace-scoped Kubernetes data workloads.
The agent can discover resources, read status and logs, analyze events, check
authorization, and apply or change manifests after confirmation. This includes
Pods, Deployments, Jobs, and custom resources such as `FlinkDeployment`.

The plugin deliberately has no cluster-administrator capability. It cannot
query every namespace or operate cluster-scoped resources, and every request
is constrained by the selected environment's namespace allowlist. On Amazon
EKS it obtains cluster information and short-lived authentication from the
[EKS plugin](eks.md).

## Installation and Activation

```bash
datus plugin install datus-k8s-plugin
```

Also install the EKS plugin when connecting to EKS:

```bash
datus plugin install datus-eks-plugin
```

Enable the required plugins for the current project after installation. See
[Plugins](introduction.md) for other installation sources and project
activation.

## Skills

| Skill | Purpose |
|---|---|
| `k8s` | Inspect, diagnose, and change namespace-scoped Kubernetes workloads |
| `k8s-setup` | Create a managed-cloud or kubeconfig-backed K8s environment |

### k8s

The core skill confirms the environment and namespace first, then chooses the
smallest necessary resource status, events, logs, metrics, or authorization
checks. For asynchronous resources it sets a maximum duration and check count,
inspecting the target state and errors together instead of waiting blindly
after a concrete failure has appeared.

### k8s-setup

The setup skill asks for a managed-Kubernetes provider or kubeconfig, a default
namespace, and the allowed namespaces. Cloud identity and short-lived
authentication stay in the provider plugin and are not copied into the K8s
environment configuration.

## Connecting to Amazon EKS

Tell the agent the cluster, authentication method, and namespace:

```text
Configure analytics-prod for the analytics-prod-eks cluster in us-east-1, using
analytics as the default namespace.
```

The agent uses `eks-setup` for the cloud environment and `k8s-setup` for the
same-named K8s environment, linking them automatically. It verifies:

- the effective AWS caller identity and target EKS cluster;
- that the Kubernetes server matches the selected environment;
- that the default namespace exists and is inside the allowlist; and
- that the current Kubernetes identity has the requested namespace access.

The short-lived bearer token is not exposed to the model or persisted to the
project. Permission to inspect EKS does not automatically grant Kubernetes
access; configure the EKS access entry and namespace RBAC separately.

When the EKS and K8s environments need different names, state the relationship
explicitly:

```text
Make flink-prod use analytics-prod EKS and allow only flink-jobs.
```

## Connecting through kubeconfig

For an existing kubeconfig, give the agent its path and context:

```text
Use the staging context in ./conf/kubeconfig.yaml and allow only
analytics-staging.
```

The agent resolves relative paths from the current project and rejects a path
that escapes through `..` or a symlink. If you omit the context, kubeconfig's
current context is used, but the agent presents the effective cluster identity
before its first operation.

## Inspecting and Diagnosing Workloads

| Your request | Agent behavior |
|---|---|
| "Which Pods in analytics-prod are unhealthy?" | Summarize readiness, restarts, and actual waiting reasons, then inspect related warning events |
| "Why does pipeline-api keep restarting?" | Confirm Pod and container state, then read the previous instance and any relevant init-container logs |
| "What is the current state of the orders FlinkDeployment?" | Read job state, status.error, and JobManager/TaskManager state separately rather than trusting one field |
| "Which Pod uses the most memory?" | Read namespace Pod metrics, sort by memory, and explain when metrics are unavailable |
| "May the current identity create Jobs?" | Check authorization in the target namespace without probing through a failing write |

Resource summaries distinguish `Running` from actually Ready and expose
waiting reasons such as `Init:CrashLoopBackOff` and `ImagePullBackOff`. The
agent prefers the individual status field that answers the question and reads
the complete object only when it needs the full spec or conditions.

## Observing Asynchronous Resources

Include the success condition and observation limit in your request:

```text
Watch daily-etl in analytics-staging and tell me when it finishes or fails.
```

For Kubernetes Deployments, StatefulSets, and DaemonSets, the agent can follow
their rollout. For custom resources such as Flink, it performs bounded reads of
the resource's own status and error fields. A created resource, a Running Pod,
or a ready controller does not necessarily mean the business workload
succeeded.

## Applying and Changing Resources

Describe the outcome and ask the agent to present a proposal first:

```text
Check deploy/pipeline.yaml, then ask me whether to deploy it to analytics-prod.
```

```text
Scale pipeline-api in analytics to four replicas.
```

```text
Delete the daily-backfill Job from analytics.
```

The agent uses Server-Side Apply for local YAML or JSON manifests. Creating,
applying, deleting, patching, scaling, labeling, annotating, restarting, or
entering a container for a diagnostic probe requires confirmation. A
server-side dry-run persists nothing, but the agent still identifies the exact
environment and namespace being validated.

## Permissions and Safety Boundaries

- Resource discovery, status, details, logs, events, metrics, and authorization
  checks are read-only.
- An environment can reach only its allowed namespaces; the default namespace
  is the only allowed one when no wider allowlist is configured.
- Cross-namespace-all queries, cluster-scoped resources, kubeconfig mutation,
  and identity impersonation are blocked.
- Interactive shells, attach, file copy, port forwarding, and proxying are
  unsupported.
- An in-container diagnostic is one non-interactive command and requires
  confirmation every time.
- The agent never patches files or configuration inside a running container;
  durable fixes belong in the image or manifest.

## Orchestrating with the Flink Plugin

The [Flink plugin](flink.md) `flink-k8s-operator` skill understands the job and
generates resources such as `FlinkDeployment`. The K8s plugin performs
discovery, validation, deployment, and diagnosis in the target EKS namespace.

```text
Deploy orders to analytics-prod and track the result.
```

The Operator, CRDs, webhook, and cluster-level RBAC are outside the K8s
plugin's scope and must be installed by an administrator first.

## Related Docs

- [EKS plugin](eks.md) — EKS discovery and Kubernetes provider authentication
- [Flink plugin](flink.md) — deploy and manage Flink jobs through the Operator
- [Plugins](introduction.md) — install sources, environment configuration, activation, and permissions
- [Skills](../skills/introduction.md) — how skills are discovered and loaded
