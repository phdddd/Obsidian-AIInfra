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

用 Obsidian 打开文件夹作为 vault 即可。Git 插件和 hooks 预装好了，开箱即用。

## 自动同步

仓库预装 Obsidian Git 插件和格式转换 hooks：

| 机制               | 作用                                            |
| ---------------- | --------------------------------------------- |
| Obsidian Git 插件  | 每 10 分钟自动 commit + push                       |
| pre-commit hook  | 提交前自动将 `![[...]]` 转为 GitHub 兼容的 `![...](...)` |
| post-commit hook | 提交后自动还原为 Obsidian wiki-link 格式                |
| 启动 pull          | 打开 Obsidian 时自动拉取最新内容                         |

你只需要在 Obsidian 里正常编辑笔记，同步全自动。

## 注意事项

- 图片统一命名 `Pasted-image-YYYYMMDDHHmmss.png`（横线，无空格）
- 每个板块的图片存在各自 `images/` 子文件夹下
- `.obsidian/workspace.json` 已加入 `.gitignore`，不同步窗口布局