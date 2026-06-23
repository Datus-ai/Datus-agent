# Skill 命令 `/skill`

## 概览

`/skill` 是管理本地技能（skill）以及与 **Town Skills Marketplace** 交互的统一入口。
一个 prompt_toolkit Application 在同一事件循环里承载 Tab 切换、详情下钻、
搜索、登录与删除二次确认；脚本场景则用对应的非交互子命令。

Marketplace 安装的 skill 默认存于 `~/.datus/skills/<skill-name>/`；
项目级 `./.datus/skills/` 优先于全局目录。

---

## 基本用法

### 交互式 TUI

直接输入 `/skill` 打开浏览器：

```text
/skill
```

TUI 有三个 Tab，使用 **Tab / Shift+Tab** 或 **←/→** 切换：

| Tab | 说明 |
|-----|------|
| **Installed** | 已落地到磁盘的 skill（本地与 marketplace 来源混合） |
| **Marketplace** | Town Marketplace 上发布的 skill |
| **Published** | 当前登录用户已发布的 skill |

Tab 内按键：

| 按键 | 行为 |
|------|------|
| **↑ / ↓** | 移动选中项 |
| **Enter** | 打开当前 skill 详情面板 |
| **/** | 在 Marketplace tab 中按关键字过滤 |
| **i** | 安装当前高亮的 Marketplace skill |
| **u** | 更新当前高亮的 marketplace 来源 skill |
| **x** | 删除当前高亮的本地 skill（按两次确认） |
| **l** | 打开登录表单（Marketplace tab） |
| **Esc / q** | 关闭面板 |

### 子命令快捷入口

每个子命令要么把 TUI 跳到预置状态，要么以非交互方式直接执行（便于脚本调用）：

| 命令 | 行为 |
|------|------|
| `/skill list` | 打开 TUI 并切到 Installed tab |
| `/skill search <query>` | 打开 TUI 并切到 Marketplace tab，预填过滤词 |
| `/skill login [url]` | 打开登录表单（提供 URL 时预填） |
| `/skill logout` | 清除已保存的 marketplace 凭据 |
| `/skill install <name> [version]` | 非交互安装，`version` 默认为 `latest` |
| `/skill publish <path> [--owner <name>]` | 从含 `SKILL.md` 的目录非交互发布 |
| `/skill info <name>` | 以表格形式打印本地 + marketplace 详情 |
| `/skill update` | 批量升级所有 marketplace 来源 skill |
| `/skill remove <name>` | 删除本地 skill（删除文件前会询问） |
| `/skill help` | 打印命令参考表格 |

---

## 登录认证

发布或 promote 操作需要登录 Town 账号：

1. 执行 `/skill login`（也可指定 URL：`/skill login http://my-marketplace:9000`）
2. 在表单中输入邮箱与密码，凭据会换取 JWT 并保存到本地；密码本身不会持久化
3. 使用 `/skill logout` 清除当前 marketplace 的 token

token 按 marketplace URL 维度隔离，可在同一台机器登录多个 marketplace。

---

## 配置

Skill 搜索路径与 marketplace 地址在 `agent.yml` 中配置：

```yaml
skills:
  directories:
    - ~/.datus/skills        # 全局，跨项目共享
    - ./.datus/skills        # 项目级，优先级高于全局
  marketplace_url: "http://localhost:9000"
  install_dir: "~/.datus/skills"
  auto_sync: false
```

同名 skill 时，`./.datus/skills/<name>/` 会覆盖全局目录。

---

## 示例

```bash
# 打开 Installed tab
/skill

# 在 marketplace 中搜索 "sql"
/skill search sql

# 安装指定版本
/skill install sql-optimization 1.0.0

# 发布本地 skill
/skill publish ./skills/sql-optimization --owner murphy

# 查看详情（本地 + marketplace）
/skill info sql-optimization

# 升级所有 marketplace 来源的 skill
/skill update
```

skill 编写、权限模型与 marketplace 流程详见 [Skills 集成](../skills/introduction.md)。
