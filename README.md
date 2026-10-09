# Obsidian-AIInfra

AI Infra 求职学习笔记 —— 基于 Obsidian 的知识库。

## 项目结构

```
.
├── 学习笔记/
│   ├── images/          # 学习笔记板块的所有图片
│   ├── 基础概念/         # 分布式并行、张量并行、MoE 等
│   ├── 训练/            # FSDP、Megatron
│   ├── 强化学习/         # Verl、算法基础
│   └── 推理/            # 推理优化、推理框架
├── AI生成/
│   └── images/          # AI 生成板块图片（预留）
├── .obsidian/           # Obsidian 配置（含 Git 自动同步插件）
├── .githooks/           # 格式转换脚本（Obsidian ↔ GitHub）
└── README.md
```

## 新机器克隆使用

```bash
git clone https://github.com/phdddd/Obsidian-AIInfra.git
```

用 Obsidian 打开文件夹作为 vault 即可。仓库包含插件代码和可选的 Markdown 转换脚本；API key、会话和插件运行状态需要在每台电脑上单独配置。

## Claude / Claudian 配置

Claudian 的 API key 是每台电脑的本机机密，不能放进 Git。仓库只同步笔记、插件代码和公开配置；以下文件必须保持本机私有：

```text
.claudian/claudian-settings.json
.claudian/sessions/
.obsidian/plugins/realclaudian/data.json
```

这些路径已经写入 `.gitignore`。新电脑 clone 后，先安装并启用 Claudian，再在 Obsidian 的 `设置 -> Claudian -> Claude` 中配置：

```text
ANTHROPIC_API_KEY=你的 API key
ANTHROPIC_BASE_URL=https://www.blackaicoding.com
```

不要把 `/v1` 加到 `ANTHROPIC_BASE_URL`，也不要把 key 写入 README、笔记、`.obsidian/plugins/realclaudian/data.json` 或任何 Git 文件。使用 API key 时不需要执行 Claude 官方 OAuth 的 `/login`。

Claudian 界面中模型选择 `Sonnet` 即可。Claude Code 的 `~/.claude/settings.json`（Windows 为 `%USERPROFILE%\\.claude\\settings.json`）使用别名把它映射到网关模型 `gpt-5.5`：

```json
{
  "env": {
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "gpt-5.5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5.5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5.5"
  }
}
```

配置后用 `claude --version` 确认 CLI 已安装，然后在 Claudian 中新建会话测试。验证成功应满足：回复正常，Claude Code 日志中的模型为 `gpt-5.5`，且不出现 `Not logged in` 或 `authentication_failed`。

### macOS

1. 安装 Claude Code CLI，并确认 `claude --version` 有输出。
2. 在 Claudian 设置中填写上面的两个环境变量。配置会保存到 vault 的 `.claudian/claudian-settings.json`，但该文件不会被 Git 跟踪。
3. 创建 Claude Code 用户配置：

   ```bash
   mkdir -p ~/.claude
   cat > ~/.claude/settings.json <<'JSON'
   {
     "env": {
       "ANTHROPIC_DEFAULT_OPUS_MODEL": "gpt-5.5",
       "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5.5",
       "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5.5"
     }
   }
   JSON
   ```

4. GitHub 推荐使用 SSH。生成 key 后，把公钥添加到 GitHub `Settings -> SSH and GPG keys`：

   ```bash
   ssh-keygen -t ed25519 -C "你的 GitHub 邮箱"
   pbcopy < ~/.ssh/id_ed25519.pub
   ssh -T git@github.com
   git remote set-url origin git@github.com:phdddd/Obsidian-AIInfra.git
   git ls-remote origin HEAD
   ```

   `ssh -T` 显示 `Hi <username>!` 后才说明 push 身份配置成功。

5. 如果使用 HTTPS，不要输入 GitHub 登录密码。创建 GitHub Personal Access Token，在 Git 提示时使用 GitHub 用户名和 token；macOS 会由 `osxkeychain` 保存：

   ```bash
   git config --global credential.helper osxkeychain
   git ls-remote origin HEAD
   ```

Obsidian Git 的 `commitMessageScript` 如果要使用仓库内的 PowerShell 图片转换脚本，macOS 需要先安装 PowerShell 7，并在插件设置中使用：

```text
pwsh -NoProfile -File ".githooks/convert-before-commit.ps1"; echo "vault backup: {{date}}"
```

安装方式：`brew install --cask powershell`。如果不需要图片格式转换，可以清空该脚本设置，Git 的 push/pull 鉴权不受影响。

### Windows

1. 安装 Claude Code CLI，并确认 PowerShell 中 `claude --version` 有输出。
2. 在 Claudian 设置中填写上面的两个环境变量。
3. 创建 `%USERPROFILE%\\.claude\\settings.json`，内容使用上面的模型映射 JSON。
4. GitHub 推荐使用 SSH：

   ```powershell
   ssh-keygen -t ed25519 -C "你的 GitHub 邮箱"
   Get-Content $env:USERPROFILE\\.ssh\\id_ed25519.pub | Set-Clipboard
   ssh -T git@github.com
   git remote set-url origin git@github.com:phdddd/Obsidian-AIInfra.git
   git ls-remote origin HEAD
   ```

   将剪贴板中的公钥添加到 GitHub `Settings -> SSH and GPG keys`。看到 `Hi <username>!` 后才说明 push 身份配置成功。

5. 如果使用 HTTPS，在 Git 提示中输入 GitHub 用户名和 Personal Access Token，不要输入普通登录密码。Git for Windows 通常自带 Git Credential Manager：

   ```powershell
   git config --global credential.helper manager-core
   git ls-remote origin HEAD
   ```

Windows 的 Obsidian Git `commitMessageScript` 可以使用：

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".githooks/convert-before-commit.ps1"; echo "vault backup: {{date}}"
```

### GitHub 身份验证排查

以下命令只读远程信息，不会上传笔记：

```bash
git remote -v
git ls-remote origin HEAD
```

如果出现 `could not read Username`，说明 HTTPS 没有凭据；如果出现 `Authentication failed`，说明 token 过期、权限不足或保存了旧 token；如果 SSH 出现 `Permission denied (publickey)`，说明当前公钥还没有添加到正确的 GitHub 账户。仓库为私有时，token 还必须拥有该仓库的读写权限。

Git 提交作者身份与 GitHub 登录身份是两回事。首次配置时建议设置：

```bash
git config --global user.name "你的 GitHub 用户名"
git config --global user.email "你的 GitHub 邮箱"
```

不要把 token、SSH 私钥或 API key 提交到仓库。API key 一旦误提交，即使后来删除当前文件，也仍可能存在于 Git 历史中，应立即撤销并重新生成。

## 自动同步

仓库包含 Obsidian Git 插件和格式转换脚本：

| 机制               | 作用                                            |
| ---------------- | --------------------------------------------- |
| Obsidian Git 插件  | 配置后可自动 commit + push                         |
| `.githooks/` 脚本  | 可选：提交前/后转换 Obsidian 图片链接格式             |
| 启动 pull          | 在 Obsidian Git 本机设置中开启后自动拉取             |

首次使用必须先配置 GitHub SSH 或 HTTPS token，并按操作系统设置 Obsidian Git 的提交脚本；完成后才适合开启自动同步。

## 注意事项

- 图片统一命名 `Pasted-image-YYYYMMDDHHmmss.png`（横线，无空格）
- 每个板块的图片存在各自 `images/` 子文件夹下
- `.obsidian/workspace.json` 已加入 `.gitignore`，不同步窗口布局
- Claudian 的 API key、会话和本机插件状态已加入 `.gitignore`，不同步到 GitHub
