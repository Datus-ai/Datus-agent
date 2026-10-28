# 数据库 adapters

Datus 使用基于 registry 的数据库 adapter 系统。SQLite 与 DuckDB 随 Agent 内置；其他服务型数据库和云数仓都由独立版本的 `datus-<type>` 包提供，并通过 `datus.adapters` entry-point group 自动发现。

连接 profile 统一放在 [Datasources](../configuration/datasources.md) 下。每个子页面是字段、默认值、认证规则和命名空间行为的准确信息源；本页只说明 adapter 如何安装与加载。

## 安装与发现

推荐通过 `/datasource` 管理器安装：

```bash
datus
```

运行 `/datasource`，选择数据库类型并填写 profile。缺少 adapter 时，Datus 会把 `datus-<type>` 安装到当前解释器，加载 entry point，校验 profile，并在保存前测试连接。

受控部署可以显式安装，并与其他依赖一起锁定版本：

```bash
pip install datus-postgresql
```

安装 adapter 会注册四类行为：

- 对外公开的 `type` 名称，例如 `postgresql`；
- 负责拒绝未知字段并提供默认值的 Pydantic 连接模型；
- connector factory 与命名空间/URI handler；
- 可选的数据库专用 SQL 指引或 skill。

Adapter 必须安装在运行 `datus` 的同一个 Python 环境中。安装到另一个 virtual environment 不会被当前进程发现。

## 运行时流程

```text
agent.services.datasources.<name>
        ↓
Agent 保留通用字段与 adapter 专用字段
        ↓
adapter 连接模型校验类型、默认值与约束
        ↓
registry 按 <type> 创建 connector
        ↓
SQL 执行、元数据发现与命名空间切换
```

Adapter 包负责数据库专用配置；Datus 负责外围 profile 名、`type`、`default`、环境变量展开和当前 datasource 选择。因此连接字段按 datasource 独立成页，而不是维护一张只能近似描述所有数据库的共享表。

## 开发 adapter

数据库 adapters 位于 [`Datus-ai/datus-db-adapters`](https://github.com/Datus-ai/datus-db-adapters) workspace。新增 adapter 应：

1. 依赖 `datus-db-core`，适用时复用 `datus-sqlalchemy`；
2. 定义严格的 Pydantic 配置模型；
3. 实现 connector contract，并准确声明 namespace capability；
4. 暴露与 datasource `type` 同名的 `datus.adapters` entry point；
5. 通过共享 adapter contract 与 TPC-H 测试；
6. 在本文档中增加一组中英文 datasource 详细页。

应以仓库当前 reference adapter 和测试标准为准，不要原样复制旧 package。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| Adapter 不在列表中 | 把 `datus-<type>` 安装到运行 `datus` 的解释器，然后重新打开 `/datasource`。 |
| 安装成功但加载失败 | 直接 import `datus_<type>`，暴露缺失 native library 或依赖冲突。 |
| Profile 拒绝多余字段 | 只使用对应 datasource 页面列出的字段；adapter 模型有意使用严格校验。 |
| 元数据层级错误 | 检查 adapter 声明的 catalog/database/schema capability，以及 profile 是否使用文档规定的 namespace 字段。 |

## 下一步

- [配置 datasources](../configuration/datasources.md)
- [在 CLI 中使用 `/datasource`](../cli/other_commands.md#datasource)
- [配置 SQL policy 与只读模式](../configuration/sql_policy.md)
