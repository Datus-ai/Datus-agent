# Dashboard Copilot

只需一条命令，即可将您的 BI 仪表盘转化为智能 AI 子代理。本指南将带您完成 Superset + PostgreSQL 的部署、Datus 配置，以及使用 `bootstrap-bi` 命令从仪表盘自动生成上下文和子代理。

## 为什么选择 Dashboard Copilot？

传统 BI 仪表盘是静态的——它们展示预定义的图表和指标，但用户无法提出后续问题或探索预构建内容之外的数据。**Datus Dashboard Copilot 将这些静态仪表盘转化为动态分析助手**，能够：

- 使用与仪表盘相同的数据和业务逻辑回答临时问题
- 在指标发生意外变化时进行根因分析
- 生成与仪表盘语义模型保持一致的新 SQL 查询
- 提供归因分析，解释指标变化的驱动因素

只需一条命令，Datus 就能从现有仪表盘中提取所有上下文——SQL 查询、表关系、指标定义和业务逻辑——并创建像您的仪表盘一样理解数据的 AI 子代理。

Bootstrap 过程会自动生成两个专门的子代理：一个是通过 SQL 工具实现自助取数的 **GenSQL 子代理**，另一个是通过指标工具提供分析、下钻和归因报告的 **GenReport 子代理**。

![Dashboard to Agent 架构](../assets/dashboard_to_agent.png)

## 前置条件

开始之前，请确保您已具备：

- Docker Desktop 已安装并运行
- Kubernetes CLI (`kubectl`)
- Helm 包管理器
- Python 3.12 并已安装 Datus

## 步骤 1：部署 Superset + PostgreSQL

首先，安装所需的基础设施工具。

### 安装依赖

=== "macOS"

    ```bash
    brew install --cask docker
    brew install helm kubectl minikube
    ```

=== "Linux"

    ```bash
    # 安装 Docker
    curl -fsSL https://get.docker.com | sh

    # 安装 kubectl
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl && sudo mv kubectl /usr/local/bin/

    # 安装 Helm
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

    # 安装 minikube
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    sudo install minikube-linux-amd64 /usr/local/bin/minikube
    ```

=== "Windows"

    ```powershell
    # 使用 Chocolatey 安装
    choco install docker-desktop
    choco install kubernetes-helm
    choco install kubernetes-cli
    choco install minikube
    ```

### 启动 Minikube

```bash
minikube start --driver=docker
```

### 部署 Superset

添加 Superset Helm 仓库并部署：

```bash
# 添加 Superset Helm 仓库
helm repo add superset https://apache.github.io/superset
helm repo update

# 使用示例配置部署 Superset
helm upgrade --install superset superset/superset -n default -f ./examples-values.yaml --wait --timeout 30m
```
[下载 examples-values.yaml](../assets/examples-values.yaml){ .md-button }

!!! tip "自定义配置"
    您可以通过修改 `examples-values.yaml` 来自定义部署。有关可用选项，请参阅 [Superset Helm Chart 文档](https://github.com/apache/superset/tree/master/helm/superset)。

### 等待 Pod 就绪

监控部署状态，直到所有 Pod 运行正常：

```bash
kubectl get pods -n default -w
```

等待所有 Pod 显示 `Running` 状态后再继续。

### 设置端口转发

将 Superset 和 PostgreSQL 服务暴露到本地：

```bash
# 转发 Superset UI（端口 8088）
kubectl port-forward -n default service/superset 8088:8088 > /dev/null 2>&1 &

# 转发 PostgreSQL（端口 15432）
kubectl port-forward -n default svc/superset-postgresql 15432:5432 > /dev/null 2>&1 &
```

现在您可以通过 [http://localhost:8088](http://localhost:8088) 访问 Superset，默认凭据为 `admin/admin`。

## 步骤 2：配置 Datus

配置 Datus 以连接 PostgreSQL 数据库和 Superset 仪表盘。

### 安装所需扩展

```bash
pip install datus-postgresql datus-semantic-metricflow
```

### 更新 agent.yml

将以下配置添加到您的 `~/.datus/agent.yml`：

```yaml
agent:
  namespace:
    superset:
      type: postgresql
      host: 127.0.0.1
      port: 15432
      username: superset
      password: superset
      database: examples
  dashboard:
    superset:
      username: admin
      password: admin
      extra:
        provider: db
```

!!! note "配置说明"
    - **namespace**：定义用于 SQL 执行的数据库连接
    - **dashboard**：定义用于仪表盘访问的 BI 平台凭据

## 步骤 3：从仪表盘 Bootstrap

现在使用 `bootstrap-bi` 命令从 Superset 仪表盘自动生成上下文和子代理。我们将以世界银行数据仪表盘为例。

### 运行 Bootstrap 命令

```bash
datus-agent bootstrap-bi --namespace superset
```

### 交互流程

该命令将引导您完成交互式流程：

```{ .yaml .no-copy }
Select BI platform (superset): superset
Dashboard URL: http://localhost:8088/superset/dashboard/world_health/?native_filters_key=4X5gjZkIbnU
API base URL (e.g. https://host) (http://localhost:8088): http://localhost:8088
```

系统将显示仪表盘信息和提取的图表：

![从仪表盘图表提取 SQL](../assets/worldbank_boostrapbi_extract_sql.png)

选择要包含的图表和表。Bootstrap 过程将自动执行：

**1. 构建元数据和参考 SQL**

系统分析每个图表的 SQL 查询并生成完整文档：

![参考 SQL 生成](../assets/worldbank_bootstrapbi_reference_sql.png)

**2. 生成语义模型**

Datus 创建包含度量、维度和关系的语义模型：

![语义模型生成](../assets/worldbank_bootstrapbi_semantic_model.png)

**3. 提取指标**

系统从仪表盘查询中识别并验证指标：

![指标提取](../assets/worldbank_bootstrapbi_metrics.png)

### 输出

Bootstrap 完成后，您将获得可直接使用的子代理：

```{ .yaml .no-copy }
Subagent `superset_world_bank_s` saved.
Subagent `superset_world_bank_s` bootstrapped.
Attribution Sub-Agent `superset_world_bank_s_attribution` saved.
```

## 步骤 4：使用归因分析

Bootstrap 生成的归因子代理提供强大的指标分析能力。

### 示例查询

```bash
Datus> /superset_world_bank_s_attribution compare 2014 and 2004, find the reason of population growth
```

代理使用仪表盘中的指标和维度执行多步分析：

![归因分析过程](../assets/worldbank_population_attribution1.png)

### 归因分析能力

归因子代理提供：

- **自动维度重要性排序** - 识别哪些维度对指标变化影响最大
- **增量贡献计算** - 量化每个因素对整体变化的贡献
- **根因识别** - 精确定位驱动指标变动的具体值

### 示例输出

分析生成包含关键发现的完整报告：

![归因分析结果](../assets/worldbank_population_attribution2.png)

报告包括：

- **整体增长指标** - 总人口、增长率和农村人口百分比的对比
- **主要区域贡献者** - 哪些区域推动了人口增长最多
- **主要国家贡献者** - 各国对变化的贡献
- **结论** - 解释指标变动的关键洞察摘要

## 生成的子代理说明

`bootstrap-bi` 命令创建两种类型的子代理：

### GenSQL 子代理

主子代理（如 `superset_world_bank_s`）提供：

- 在仪表盘语义范围内生成 SQL
- 使用提取的元数据进行上下文感知查询
- 来自仪表盘图表的参考 SQL 模式

**使用示例：**
```bash
/superset_world_bank_s show top 10 countries by life expectancy in 2020
```

### GenReport 子代理（归因）

归因子代理（如 `superset_world_bank_s_attribution`）提供：

- 指标对比和趋势分析
- 指标变化的根因分析
- 维度级别的归因报告

**使用示例：**
```bash
/superset_world_bank_s_attribution why did healthcare spending increase between 2010 and 2020?
```

## 下一步

现在您已经拥有了由仪表盘驱动的子代理，可以探索更多功能：

- **[子代理介绍](../subagent/introduction.md)** - 了解更多子代理功能
- **[知识库](../knowledge_base/introduction.md)** - 管理和扩展您的上下文
- **[指标](../knowledge_base/metrics.md)** - 定义和管理您的指标
- **[语义模型](../knowledge_base/semantic_model.md)** - 自定义您的语义层
