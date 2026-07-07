---
name: obsidian-vault-sync
description: Setup and maintain Obsidian vault-to-GitHub bidirectional sync with automatic format conversion. Use when setting up a cloned Obsidian vault on a new machine, configuring Obsidian Git plugin, fixing image display issues between Obsidian and GitHub, creating git hooks for format conversion, or troubleshooting the sync workflow in this vault (Obsidian-AIInfra).
---

# Obsidian Vault GitHub Sync

This vault uses a dual-format system: Obsidian wiki-links in the working tree, GitHub-compatible Markdown in commits. Three PowerShell scripts handle automatic conversion.

## Architecture

```
Working tree (Obsidian)          Git commit (GitHub)
![[Pasted-image-xxx.png]]   ←→   ![Pasted-image-xxx.png](../images/Pasted-image-xxx.png)
```

**Conversion layer**: `scripts/convert-to-github.ps1` (Obsidian→GitHub), `scripts/restore-obsidian.ps1` (GitHub→Obsidian). Both live in `.githooks/` and are triggered by git hooks OR the Obsidian Git plugin's `commitMessageScript`.

## Directory Structure Rules

Each top-level content folder has its own `images/` subfolder:

```
vault/
├── 学习笔记/
│   ├── images/           ← All images for this section
│   ├── 基础概念/
│   ├── 训练/
│   └── ...
├── AI生成/
│   └── images/           ← Images for this section (when needed)
└── ...
```

Images are named `Pasted-image-YYYYMMDDHHmmss.png` (hyphens, no spaces).

## New Machine Setup

### 1. Clone

```bash
git clone https://github.com/phdddd/Obsidian-AIInfra.git
```

If behind a proxy (China mainland), configure git first:
```bash
git config --global http.proxy http://127.0.0.1:10090
git config --global https.proxy http://127.0.0.1:10090
git config --global http.sslVerify false
```

### 2. Open in Obsidian

Open the cloned folder as a vault. The `.obsidian/` config (including the Git plugin) is already included.

### 3. Verify Obsidian Git Plugin

The plugin is pre-installed and pre-configured. Open Settings → Community Plugins → Git. Verify:
- Auto commit interval: 10 minutes (or adjust as needed)
- Pull on startup: enabled

### 4. Set Git Hooks

```bash
git config core.hooksPath .githooks
```

The hooks are already in `.githooks/` and `.git/hooks/`. If the Obsidian Git plugin doesn't trigger them, the `commitMessageScript` in the plugin's `data.json` handles conversion as a fallback.

### 5. Set Attachment Folder

In Obsidian Settings → Files & Links → Attachment folder path: set to the active section's images folder (e.g., `学习笔记/images`).

## Conversion Scripts

### `scripts/convert-to-github.ps1` (run before commit)

1. Renames any `Pasted image *.png` → `Pasted-image-*.png` (spaces→hyphens)
2. Converts `![[Pasted-image-xxx.png]]` → `![Pasted-image-xxx.png](<relative>/Pasted-image-xxx.png)`
3. Computes relative path based on file depth (e.g., `../images/` for files in subdirectories)

### `scripts/restore-obsidian.ps1` (run after commit/pull)

Converts `![Pasted-image-xxx.png](<any-path>/Pasted-image-xxx.png)` back to `![[Pasted-image-xxx.png]]` in all `.md` files.

### Manual Conversion

To manually convert the working tree before a push (if hooks/plugin script aren't working):

```powershell
# Convert Obsidian → GitHub
Get-ChildItem . -Recurse -Filter "*.md" | ForEach-Object {
    $c = [System.IO.File]::ReadAllText($_.FullName)
    $n = [regex]::Replace($c, '!\[\[Pasted-image-(\d+\.png)\]\]', {
        $fn = $args[0].Groups[1].Value
        # Compute depth from vault root
        $parts = ($_.FullName -replace [regex]::Escape($pwd.Path), '').TrimStart('\') -split '[/\\]'
        $depth = $parts.Count - 1
        $prefix = if ($depth -le 1) { 'images/' } else { ('../' * ($depth - 1)) + 'images/' }
        return "![Pasted-image-$fn]($($prefix)Pasted-image-$fn)"
    })
    if ($n -ne $c) { [System.IO.File]::WriteAllText($_.FullName, $n) }
}
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| GitHub images broken | Run `convert-to-github.ps1` manually, then commit+push |
| Obsidian images broken | Run `restore-obsidian.ps1` to convert back to wiki-links |
| Push fails with SSL error | Set `git config --global http.sslVerify false` |
| Push fails with connection error | Check proxy config or try different proxy port |
| Obsidian Git plugin doesn't trigger hooks | Use `commitMessageScript` in plugin's `data.json` |
| New images paste with spaces | Pre-commit hook auto-renames them, or manually rename `Pasted image *.png` → `Pasted-image-*.png` |
