# 技能命令

`datus skill` 命令提供了管理本地技能和与 Town 技能市场交互的 CLI 接口。

## 子命令

| 子命令 | 描述 |
|--------|------|
| `login` | 与 Town 市场进行认证 |
| `logout` | 清除已保存的认证令牌 |
| `list` | 列出所有本地已安装的技能 |
| `search <query>` | 在市场中搜索技能 |
| `install <name> [version]` | 从市场安装技能 |
| `publish <path>` | 将本地技能发布到市场 |
| `info <name>` | 显示技能详情 |
| `update` | 更新所有市场安装的技能 |
| `remove <name>` | 移除本地已安装的技能 |

## 全局选项

| 选项 | 描述 |
|------|------|
| `--marketplace <url>` | 覆盖 `agent.yml` 中的市场 URL |

## 使用方法

### 认证

```bash
# 交互式登录
datus skill login --marketplace http://datus-marketplace:9000

# 非交互式登录（使用环境变量避免密码泄露到 shell 历史记录）
DATUS_PASSWORD='***' datus skill login --marketplace http://datus-marketplace:9000 --email user@example.com --password "$DATUS_PASSWORD"

# 登出
datus skill logout --marketplace http://datus-marketplace:9000
```

### 列出本地技能

```bash
datus skill list
```

输出：
```
┌──────────────────┬─────────┬─────────────┬─────────────────────────┐
│ Name             │ Version │ Source      │ Tags                    │
├──────────────────┼─────────┼─────────────┼─────────────────────────┤
│ sql-optimization │ 1.0.0   │ marketplace │ sql, optimization       │
│ report-generator │ 1.0.0   │ local       │ report, analysis        │
└──────────────────┴─────────┴─────────────┴─────────────────────────┘
```

### 搜索市场

```bash
datus skill search sql
datus skill search --marketplace http://localhost:9000 report
```

### 安装技能

```bash
# 安装最新版本
datus skill install sql-optimization

# 安装指定版本
datus skill install sql-optimization 1.0.0
```

### 发布技能

```bash
# 从技能目录发布（必须包含 SKILL.md）
datus skill publish ./skills/sql-optimization

# 指定所有者发布
datus skill publish ./skills/sql-optimization --owner "murphy"
```

### 技能详情

```bash
datus skill info sql-optimization
```

### 更新技能

```bash
datus skill update
```

### 移除技能

```bash
datus skill remove sql-optimization
```

## REPL 等效命令

所有 `datus skill` 子命令在 REPL 中都可以作为 `.skill` 命令使用：

```
datus> .skill list
datus> .skill search sql
datus> .skill install sql-optimization
datus> .skill publish ./skills/my-skill
datus> .skill info sql-optimization
datus> .skill update
datus> .skill remove sql-optimization
```

## 配置

市场设置可以在 `agent.yml` 中配置：

```yaml
skills:
  directories:
    - ~/.datus/skills
    - ./skills
  marketplace_url: "http://localhost:9000"
  auto_sync: false
  install_dir: "~/.datus/skills"
```

有关技能创建、权限和市场工作流的更多详情，请参阅 [Skills 集成](../integration/skills.md)。
