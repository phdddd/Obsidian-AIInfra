---
name: obsidian-vault-sync
description: Setup and maintain Obsidian vault-to-GitHub bidirectional sync with automatic format conversion. Use when setting up a cloned Obsidian vault on a new machine, configuring Obsidian Git plugin, fixing image display issues between Obsidian and GitHub, creating git hooks for format conversion, or troubleshooting the sync workflow in the Obsidian-AIInfra vault.
---

# Obsidian Vault GitHub Sync

Dual-format system: Obsidian wiki-links in the working tree, GitHub Markdown in commits. Scripts in `.githooks/` handle automatic conversion.

Vault repo: `https://github.com/phdddd/Obsidian-AIInfra`

## Installing the Skill in Codex

```powershell
$pluginDir = "$env:USERPROFILE\.codex\plugins\obsidian-vault-sync"
New-Item -ItemType Directory -Path "$pluginDir\skills\obsidian-vault-sync" -Force
Copy-Item -Recurse "skills/obsidian-vault-sync/*" "$pluginDir\skills\obsidian-vault-sync\" -Force
# Create .codex-plugin/plugin.json - see below
python "$env:CODEX_HOME\skills\.system\plugin-creator\scripts\validate_plugin.py" $pluginDir
python "$env:CODEX_HOME\skills\.system\plugin-creator\scripts\update_plugin_cachebuster.py" $pluginDir
# Restart Codex -> "/" -> "Obsidian Vault Sync"
```

## Architecture

```
Working tree (Obsidian)          Git commit (GitHub)
![[Pasted-image-xxx.png]]   <->   ![Pasted-image-xxx.png](../images/Pasted-image-xxx.png)
```

**Conversion layer**: `convert-to-github.ps1` (Obsidian->GitHub), `restore-obsidian.ps1` (GitHub->Obsidian). Both in `.githooks/`, triggered by git hooks or the Obsidian Git plugin's `commitMessageScript`.

## Directory Structure

Each top-level content folder has its own `images/` subfolder:

```
vault/
├── 学习笔记/
│   ├── images/            All images for this section
│   ├── 基础概念/
│   └── 训练/
├── AI生成/
│   └── images/
└── ...
```

Images: `Pasted-image-YYYYMMDDHHmmss.png` (hyphens, never spaces).

## New Machine Setup

### 1. Clone

```bash
git clone https://github.com/phdddd/Obsidian-AIInfra.git
```

Proxy config (China mainland):
```bash
git config --global http.proxy http://127.0.0.1:10090
git config --global https.proxy http://127.0.0.1:10090
git config --global http.sslVerify false
```

### 2. Open in Obsidian

Open the cloned folder as a vault. `.obsidian/` config (including Git plugin) is pre-packaged.

### 3. Verify Obsidian Git Plugin

Settings -> Community Plugins -> Git:
- Auto commit interval: 10 minutes
- Pull on startup: enabled

### 4. Activate Git Hooks

```bash
git config core.hooksPath .githooks
```

Hooks are pre-installed in `.githooks/` and `.git/hooks/`. If the plugin doesn't trigger them, `commitMessageScript` in the plugin's `data.json` handles conversion as fallback.

### 5. Set Attachment Folder

Obsidian Settings -> Files & Links -> Attachment folder path: `学习笔记/images`

## Conversion Scripts

### `convert-to-github.ps1` (pre-commit)

1. Renames `Pasted image *.png` -> `Pasted-image-*.png`
2. Converts `![[Pasted-image-xxx.png]]` -> `![Pasted-image-xxx.png](<relative>/Pasted-image-xxx.png)`
3. Relative path computed from file depth: `("../" * (depth-1)) + "images/"`

### `restore-obsidian.ps1` (post-commit / post-merge)

Converts `![Pasted-image-xxx.png](<any-path>/Pasted-image-xxx.png)` back to `![[Pasted-image-xxx.png]]` in all `.md` files.

## Troubleshooting

| Problem                        | Fix                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------- |
| GitHub images broken           | Run `convert-to-github.ps1` manually, then commit+push                          |
| Obsidian images broken         | Run `restore-obsidian.ps1` to convert back to wiki-links                        |
| Push fails: SSL error          | `git config --global http.sslVerify false`                                      |
| Push fails: connection refused | Check proxy at `127.0.0.1:10090` or try `127.0.0.1:7897`                        |
| Hooks not triggered by plugin  | Ensure `commitMessageScript` is set in plugin's `data.json`                     |
| Images paste with spaces       | Pre-commit hook auto-renames; or manually: `Pasted image *` -> `Pasted-image-*` |
