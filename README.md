# Obsidian-AIInfra

AI Infra 求职学习笔记 —— 基于 Obsidian 的知识库。

## 项目结构

```
.
├── 学习笔记/           # 核心笔记：分布式并行、FSDP、Megatron、强化学习等
├── 学习计划安排/       # 月度/周度复习计划
├── images/            # 笔记中引用的图片
├── .obsidian/         # Obsidian 配置（含 Git 自动同步插件）
├── 欢迎.md            # Obsidian 欢迎页
├── 简历完善.md         # 简历相关
└── 常用命令.md         # 常用命令速查
```

## 快速开始（新机器）

### 1. 克隆仓库

```bash
git clone https://github.com/phdddd/Obsidian-AIInfra.git
```

> 国内用户如果无法访问 GitHub，需要配置代理：
> ```bash
> git config --global http.proxy http://127.0.0.1:7897
> git config --global https.proxy http://127.0.0.1:7897
> ```

### 2. 用 Obsidian 打开

1. 安装 [Obsidian](https://obsidian.md)
2. 打开 Obsidian → 点击 "Open folder as vault"
3. 选择刚才克隆的 `Obsidian-AIInfra` 文件夹
4. 如果提示 "Safe mode"，选择 "Turn off Safe Mode" 以启用 Git 插件

### 3. 自动同步已就绪

仓库已预装 **Obsidian Git** 插件，开箱即用：

| 功能        | 配置                        |
| --------- | ------------------------- |
| 自动 commit | 文件变更后自动提交                 |
| 自动 push   | 每 10 分钟自动推送到 GitHub       |
| 启动 pull   | 打开 Obsidian 时自动拉取最新内容     |
| 冲突处理      | push 前自动 pull，使用 merge 策略 |

右下角状态栏会显示 Git 同步状态，无需手动操作。

> 首次在新机器上编辑后 push 时，可能需要登录 GitHub 授权。Obsidian 内会弹出认证窗口。

### 4. 手动同步

如果需要立即同步，按 `Ctrl+P` 打开命令面板，搜索：
- `Git: Commit and push` — 手动提交并推送
- `Git: Pull` — 手动拉取

## 注意事项

- 图片使用标准 Markdown 语法（兼容 GitHub 渲染），不要使用 `![[...]]` wiki-link 格式
- 粘贴图片到 Obsidian 时会自动存为 `Pasted image xxx.png`，需要手动移动到 `images/` 文件夹并更新引用路径
- `.obsidian/workspace.json` 已加入 `.gitignore`，不会同步个人窗口布局

## 贡献

这是个人学习笔记，欢迎 Fork 和 Star。如有问题请提 Issue。
