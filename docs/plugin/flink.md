# Flink Plugin

The Flink plugin (`datus-flink-plugin`) lets the Datus agent validate Apache
Flink jobs locally, package SQL applications, and deploy and operate jobs
through the Apache Flink Kubernetes Operator. It is a skill-only plugin: you
describe the outcome, the agent loads the relevant skill, and every Kubernetes
resource operation is delegated to the [K8s plugin](k8s.md). On Amazon EKS,
the [EKS plugin](eks.md) supplies cluster discovery and short-lived
authentication.

This page focuses on deploying a Flink job to an EKS cluster where the Flink
Kubernetes Operator is already installed. Installing the Operator, CRDs,
webhook, and cluster-level RBAC remains a Kubernetes administrator task and is
not performed by these plugins.

## Installation and Activation

Install the three plugins in dependency order:

```bash
datus plugin install datus-eks-plugin
datus plugin install datus-k8s-plugin
datus plugin install datus-flink-plugin
```

Enable them for the current project after installation. See
[Plugins](introduction.md) for other installation sources and project
activation.

## Skills

| Skill | Purpose |
|---|---|
| `flink-local-dev` | Validate Flink SQL in a local MiniCluster without writing to production sinks |
| `flink-sql` | Select and verify the version-compatible SQL entry point for Application Mode |
| `flink-k8s-operator` | Build, deploy, upgrade, and diagnose Operator-managed Flink jobs |

### flink-local-dev

When you ask for local validation, the agent leaves production SQL unchanged
and uses a separate local overlay. Sources are restricted to development
systems and every sink is replaced with `print`, `blackhole`, or a local
filesystem table. It refuses to run when local execution is not pinned, a
production sink remains unshadowed, or Git could track local credentials.

```text
Validate orders/job.sql locally without writing to production.
```

### flink-sql

After validation, ask the agent to settle the production SQL entry point.
Flink 1.x normally needs a version-matched runner JAR; Flink 2.x can use the
`SqlDriver` on the system classpath. The agent verifies SQL, connectors, and
filesystem dependencies and hands the resulting image and job fields to
`flink-k8s-operator` without inventing a JAR merely to fill a field.

```text
Prepare this SQL as a production job for Flink 1.20.
```

### flink-k8s-operator

This skill supports Application Clusters, Session Clusters,
`FlinkSessionJob`, and `FlinkStateSnapshot`. You provide the job and target
environment; the agent handles the build, deployment preflight, custom-resource
proposal, deployment, observation, and safe suspend, resume, upgrade, and
deletion workflows.

## Deploying a Flink Job on EKS

### 1. Check infrastructure prerequisites

Start with a read-only request:

```text
Can flink-prod deploy a Flink job in flink-jobs? Check without changing anything.
```

The agent verifies:

- that the EKS cluster is healthy and the effective AWS identity points to the
  expected account and cluster;
- that the target namespace is inside the K8s profile's allowlist;
- that the cluster serves the required `FlinkDeployment` API, plus
  `FlinkSessionJob` for Session mode and `FlinkStateSnapshot` for snapshots;
- that the Kubernetes identity used by Datus may read and create the target
  custom resources;
- that the Flink job service account can manage the Pods, Services, and
  ConfigMaps needed for TaskManagers and read their Deployment owner
  references;
- that IRSA is configured when the job needs AWS services such as S3; and
- that EKS worker nodes can reach the job image registry.

If the Operator, a CRD, webhook, access entry, or cluster RBAC is missing, the
agent reports the evidence and administrator action instead of bypassing the
boundary or installing it itself.

### 2. Configure the EKS and K8s environment

If no profile exists, provide the cluster and namespace in natural language:

```text
Configure flink-prod for the analytics-prod EKS cluster in us-east-1, using
the datus-flink-operator role and the flink-jobs namespace.
```

The agent runs `eks-setup` and `k8s-setup`, creating linked profiles with the
same name. It shows a non-secret configuration summary and verifies the target
without exposing a short-lived Kubernetes token or writing literal secrets.
When required information is missing, it asks before writing configuration.

### 3. Validate and publish the job image

SQL jobs pass through `flink-local-dev` and `flink-sql`. JVM and PyFlink
projects use their existing test and build workflows. Give the agent the target
version and registry and require an immutable image identity:

```text
Build a Flink 1.20 production image for orders-enrichment and publish it to our
existing ECR repository. Ask before pushing.
```

The EKS plugin does not log in to ECR, create repositories, or push images.
The agent may use only a registry and authentication that are already
available, and it asks before making the external push.

### 4. Propose the deployment

Ask for a plan before any cluster write:

```text
Plan the orders-enrichment deployment on flink-prod. Show me the proposal
without deploying it.
```

The agent writes the proposal under `deploy/flink/<name>/` and makes these
decisions visible:

- the Flink API version actually served by the cluster;
- image, Flink version, entry class or script, and arguments;
- Application or Session mode, parallelism, and compute resources;
- namespace, service account, and checkpoint/savepoint storage;
- `stateless`, `savepoint`, or `last-state` upgrade policy; and
- referenced Secrets, IRSA, and image-pull configuration, without literal
  credentials.

Use `stateless` for a first deployment. Move to `savepoint` or `last-state`
only after the job is stable and durable checkpoints and HA prerequisites have
been verified.

### 5. Validate and confirm deployment

```text
Validate the proposal first, then ask me whether to deploy it.
```

The agent asks the K8s plugin for a Server-Side Apply dry-run. Passing it proves
only that the API schema and current identity accept the request; it does not
prove the Flink job has started. The agent submits the resource only after you
confirm.

### 6. Observe the result

```text
After deployment, watch it until it is healthy and tell me if it fails.
```

The agent uses bounded status reads rather than waiting indefinitely. It
distinguishes among:

- the custom resource existing;
- the JobManager being ready;
- the Flink job actually reaching `RUNNING`, or `FINISHED` for a bounded batch
  job; and
- `status.error`, Pod state, or warning events reporting a failure.

The resource existing or the JobManager reporting `READY` is not sufficient
to declare success.

### 7. Diagnose a failure

```text
orders-enrichment did not start. Find out why without changing anything.
```

The agent reads the smallest necessary status fields and, when relevant,
inspects init-container or previous-instance logs. Fixes belong in the image,
manifest, Secret, or infrastructure configuration rather than as temporary
changes to a running Pod.

## Lifecycle Management

| Your request | Agent behavior |
|---|---|
| "Suspend orders-enrichment but preserve recoverable state." | Verify upgrade mode and state storage, show the proposed change, then suspend after confirmation |
| "Restore the job from its latest state." | Validate checkpoint/savepoint path, authorization, and version compatibility before restoring |
| "Upgrade the job to the new image." | Update the version-controlled manifest, preserve the confirmed upgrade policy, and show the diff |
| "Create a snapshot for the running job." | Create a uniquely named `FlinkStateSnapshot`, observe it to completion, and report its durable path |
| "Delete this stateful job." | Ask whether a final savepoint is required and delete Session Jobs before their Session Cluster |

Every step that changes a registry, project file, or cluster state presents its
target and impact and asks for confirmation first.

## Related Docs

- [K8s plugin](k8s.md) — namespace resource operations, status, logs, and permissions
- [EKS plugin](eks.md) — EKS discovery and short-lived Kubernetes authentication
- [Plugins](introduction.md) — install sources, profiles, activation, and permissions
- [Skills](../skills/introduction.md) — how skills are discovered and loaded
