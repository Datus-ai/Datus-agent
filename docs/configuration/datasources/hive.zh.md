# Hive datasource

Hive adapter 通过 PyHive/Thrift 连接 HiveServer2 的二进制端点，也可连接兼容此接口的星环 Inceptor。Hive database 是表的命名空间，可通过 `configuration` 传递会话属性。

!!! note "Adapter 版本要求"
    下文的 Kerberos 配置、Inceptor 元数据改进和结果协议需要包含对应改动的 `datus-hive` 构建。已发布的 `datus-hive` 0.1.3 不包含 `kerberos_service_name`、`kerberos` 可选依赖组、元数据修复和 `server_type`。使用这些功能前需安装包含对应改动的构建；仅安装 GSSAPI 依赖无法补上缺少的 Adapter 配置。

下方命令固定到 [Adapter PR #163](https://github.com/Datus-ai/datus-db-adapters/pull/163) 已验证的源码提交，其中也包含 [Adapter PR #158](https://github.com/Datus-ai/datus-db-adapters/pull/158) 的 Kerberos 和元数据改动。在包含结果类型修复的安装包发布前，使用此源码构建。命令需要 Git 和 Python 3.12 或更高版本，两种安装方式使用同一提交。源码构建目前仍使用 0.1.3 版本号，因此使用 `--force-reinstall` 替换已经安装的 PyPI 0.1.3 包。

## 安装

```bash
pip install --force-reinstall 'datus-hive @ git+https://github.com/Datus-ai/datus-db-adapters.git@84a407a6c5572947c01038d01e9f96580583be4a#subdirectory=datus-hive'
```

将 Adapter 安装到运行 `datus` 的同一 Python 环境。使用 Kerberos 时，先选择支持 Kerberos 的 Adapter 构建，再安装可选的 GSSAPI 依赖：

```bash
pip install --force-reinstall 'datus-hive[kerberos] @ git+https://github.com/Datus-ai/datus-db-adapters.git@84a407a6c5572947c01038d01e9f96580583be4a#subdirectory=datus-hive'
```

Kerberos 扩展构建需要系统 Kerberos 库、开发头文件和 C 编译器。例如 Debian/Ubuntu 使用 `libkrb5-dev`；还需安装 `kinit`、`klist` 等 Kerberos 客户端工具。

## 连接配置

```yaml
agent:
  services:
    datasources:
      hive_warehouse:
        type: hive
        host: ${HIVE_HOST:-127.0.0.1}
        port: 10000
        username: hive
        database: default
        auth: NONE
        timeout_seconds: 30
```

此配置用于服务端采用 `NONE` 认证的场景。连接启用认证的服务端时，按下文选择匹配的认证模式。

## 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `host` | string | 否 | `127.0.0.1` | HiveServer2 地址。 |
| `port` | integer | 否 | `10000` | HiveServer2 Thrift 端口。 |
| `server_type` | string | 否 | `hive` | 结果协议：Apache Hive 使用 `hive`，Inceptor 使用 `inceptor`。连接 Inceptor 时需显式设置，此字段不选择认证方式。 |
| `username` | string | 是 | — | Hive 用户。Kerberos 的认证身份来自票据缓存。 |
| `password` | string | 否 | 空字符串 | `LDAP` 或 `CUSTOM` 的密码；`NONE` 和 `KERBEROS` 不填写。 |
| `database` | string | 否 | 连接时使用 `default` | 初始 Hive database。 |
| `auth` | string | 否 | `None` → PyHive `NONE` | 显式选择 `NONE`、`LDAP`、`CUSTOM` 或 `KERBEROS`，不会自动探测认证方式。 |
| `kerberos_service_name` | string | `KERBEROS` 时必填 | `None` | 服务端 principal 的 service 部分，通常为 `hive`，不能为空；其他模式不填写。 |
| `configuration` | mapping | 否 | `{}` | Hive 会话属性，值会被规范化为字符串。 |
| `timeout_seconds` | integer | 否 | `30` | 连接超时秒数。 |

## 认证

### NONE、LDAP 和 CUSTOM

不填写 `auth` 时使用 PyHive 的 `NONE` 模式，该模式仍使用 SASL PLAIN，不会自动选择 LDAP 或 Kerberos。提供密码时必须显式填写 `LDAP` 或 `CUSTOM`。

```yaml
agent:
  services:
    datasources:
      hive_ldap:
        type: hive
        host: ${HIVE_HOST}
        port: 10000
        database: default
        username: ${HIVE_USER}
        password: ${HIVE_PASSWORD}
        auth: LDAP
      hive_custom:
        type: hive
        host: ${HIVE_HOST}
        port: 10000
        database: default
        username: ${HIVE_USER}
        password: ${HIVE_PASSWORD}
        auth: CUSTOM
```

两种模式都发送用户名和密码。`LDAP` 由服务端连接配置的目录进行验证，`CUSTOM` 由服务端配置的自定义认证 provider 验证。目录或 provider 需要由服务端管理员配置；数据源配置负责选择客户端认证模式，不会开启服务端认证。

### Kerberos

启动 Datus 前先获取票据。例如，使用 keytab 和供 Datus 进程单独使用的票据缓存：

```bash
export KRB5_CONFIG=/path/to/krb5.conf
export KRB5CCNAME=FILE:/path/to/datus.ccache
kinit -kt /path/to/alice.keytab alice@EXAMPLE.COM
klist
```

在同一环境中启动 Datus，并使用以下配置：

```yaml
agent:
  services:
    datasources:
      hive_kerberos:
        type: hive
        host: hs2.example.com
        port: 10000
        database: default
        username: alice
        auth: KERBEROS
        kerberos_service_name: hive
```

此示例对应服务端 principal `hive/hs2.example.com@EXAMPLE.COM`。`kerberos_service_name` 填写其中的 service 部分 `hive`。连接地址应与服务端 principal 的主机名匹配，并确保 DNS、realm 配置、KDC 连通性和时钟同步正常。

PyHive 使用 SASL GSSAPI 和当前票据缓存完成认证。配置仍需提供 `username`，但修改该字段不会改变票据对应的认证身份。不要填写 `password`，Adapter 会拒绝 Kerberos 密码配置。票据获取和续期由应用或部署环境负责；Adapter 不会执行 `kinit`、续期过期票据或修改进程级 Kerberos 配置。创建新连接时需要有效凭据。

## 会话配置

`configuration` 会作为 Hive session configuration 发送。YAML boolean 转为 `true`/`false` 字符串，其他 scalar 会字符串化，`null` 会变成空字符串。

只添加目标服务端支持的引擎参数。例如，使用 Tez 的 Hive 部署可能接受以下配置：

```yaml
configuration:
  hive.execution.engine: tez
  hive.vectorized.execution.enabled: true
```

## 星环 Inceptor

使用 `type: hive` 和 `server_type: inceptor` 连接 Inceptor 兼容 HiveServer2 的二进制 Thrift 端点：

```yaml
agent:
  services:
    datasources:
      inceptor_warehouse:
        type: hive
        server_type: inceptor
        host: ${INCEPTOR_HOST}
        port: 10000
        database: default
        username: ${INCEPTOR_USER}
        password: ${INCEPTOR_PASSWORD}
        auth: LDAP
```

将地址、端口和认证方式替换为 Inceptor 管理员提供的值。服务端启用对应模式时，可使用相同的 `NONE`、`LDAP`、`CUSTOM` 或 Kerberos 客户端配置。连接 Inceptor Kerberos 端点时，使用上方 Kerberos 示例，并增加 `server_type: inceptor`。此连接路径无需厂商 JDBC 驱动。

`server_type` 为当前连接选择结果解码器。Inceptor 与 Apache Hive 的类型编号存在差异，因此连接 Apache Hive 时保留默认的 `hive`。Adapter 不会自动探测服务端类型。

包含元数据修复的 Adapter 会将普通视图排除在表列表之外，通过视图接口单独获取；列注释按字段名读取，避免 Inceptor 的额外 `DESCRIBE` 字段导致取值错位。更新后的 Adapter 已在 Apache Hive 4.0.1 和 TDH 社区开发版 2024.5（Inceptor 8.37.3）上验证数据库、表、视图、列及注释、DDL 和采样数据获取。

表/视图发现要求服务端支持 `SHOW VIEWS`，并授予元数据读取权限；发现查询失败会向调用方报错。

Inceptor 验证覆盖 `NONE`、`LDAP`、`CUSTOM` 和客户端到 HiveServer2 的 Kerberos 认证，尚未验证完整 TDH/Guardian 集群的 Kerberos 安全配置或全部企业功能。Adapter 配置未暴露 Guardian Token/CAS 认证、HTTP/HTTPS 传输和 ZooKeeper 服务发现能力。会话引擎参数及 SQL 功能仍取决于目标服务端。

### 结果类型与中文

Inceptor 结果解码器处理 CHAR、VARCHAR、两类 interval、VARCHAR2、TIME、GEO、JSON 和 CURSOR 句柄，保留 NULL、Unicode 字符串和 CHAR 尾部空格。TIME 和 interval 保留服务端文本，JSON 返回 JSON 文本，GEO 保留服务端文本或二进制数据。遇到未知类型编号时，错误会包含类型编号和列名。

普通查询、聚合及上述结果类型已在 TDH 社区开发版 2024.5（Inceptor 8.37.3）上验证。原生 CURSOR 句柄取数及清理由存储过程测试验证；公共 `execute_sql` 工具尚未提供 OUT 参数绑定接口。其他服务端版本需要在对应端点上验证。

普通字符串值和列注释中的中文已通过真实引擎测试，包括 VARCHAR2 的 list、CSV、pandas 和 Arrow 返回格式。在本地测试环境中，包含中文的 `JSON_PARSE` 结果在 Adapter 转换前就已经变为问号；服务端将该原生 JSON 转成 STRING 后仍是相同的问号。具体原因尚未定位，应在目标服务端验证原生 JSON 的中文内容，不能据此认定 CE 版普遍不支持。

可运行以下 SQL 检查 Inceptor 结果协议：

```sql
SELECT CAST('中文' AS VARCHAR2(20)) AS probe_text;

SELECT column_name
FROM system.columns_v
WHERE database_name = 'system' AND table_name = 'columns_v'
LIMIT 5;
```

## 验证连接

```bash
datus --config conf/agent.yml --datasource hive_warehouse
```

运行 `/databases` 和 `/tables`，再查询一张小表，验证 HiveServer2 与底层 metastore 访问。

验证启用认证的数据源时，将 `hive_warehouse` 替换为对应配置名，并检查已知对象的普通视图、列注释、DDL 和采样数据。认证成功与 SQL 权限是两回事，登录用户仍需拥有目标数据库及对象的访问权限。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| Thrift 连接失败 | 检查 HiveServer2 host/port、服务健康和网络路由。 |
| SASL/认证错误 | 让 `auth`、用户名/密码与服务端 HiveServer2 认证模式一致。 |
| Inceptor 查询出现 `Error details: 24` 或 `KeyError(24)` | PyHive 按 Apache Hive 的类型定义解释了 Inceptor 的 VARCHAR2 结果元数据。安装包含 Adapter PR #163 的构建，并设置 `server_type: inceptor`。 |
| `server_type` 被拒绝 | 当前 Adapter 构建尚未包含结果协议选项，需安装上方固定的源码提交或包含该改动的发布版本。 |
| `500006` 中包含 `MAPRED-SPARK`、stage 失败或服务端异常 | `500006` 是通用执行错误包装。查看完整底层错误和 Inceptor executor 日志；结果类型修复无法解决服务端执行失败。 |
| `kerberos_service_name` 被拒绝，或安装时提示缺少 `kerberos` extra | 当前 Adapter 构建尚未包含 Kerberos 改动，需安装同时包含该字段和可选依赖组的构建。 |
| 没有 Kerberos 凭据或票据已过期 | 在 Datus 进程的环境中运行 `klist`，检查 `KRB5CCNAME`，创建新连接前重新获取或续期票据。 |
| 找不到 Kerberos 服务端 principal | 检查 `host`、realm/DNS 配置和 `kerberos_service_name` 中的 service 部分，不要将完整 principal 填入此字段。 |
| 登录成功，但查询或元数据访问被拒绝 | 请服务端管理员检查用户的数据库授权和当前角色。 |
| 会话选项不生效 | 在 `configuration` 中使用准确 Hive property key，并确认服务端允许覆盖。 |

## 参考

- [HiveServer2 clients](https://cwiki.apache.org/confluence/display/Hive/HiveServer2+Clients)
- [HiveServer2 认证配置](https://hive.apache.org/docs/latest/admin/setting-up-hiveserver2/)
- [星环 Inceptor 连接指南](https://www.transwarp.cn/doc/inceptor/9.5/quick-start--connect-database)
