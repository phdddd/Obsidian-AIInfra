# Image Format Specification

## Filename Convention

All images use hyphenated names: `Pasted-image-YYYYMMDDHHmmss.png`. Never use spaces.

## Format Rules

### Obsidian Wiki-Link (working tree only)
```
![[Pasted-image-20260524192022.png]]
```
No path component - Obsidian searches entire vault by filename. Never committed to git.

### GitHub Markdown (committed)
```
![Pasted-image-20260524192022.png](../images/Pasted-image-20260524192022.png)
```
Relative path from .md file to its section's `images/` folder.

### Path Calculation

| File location | Image path |
|---|---|
| `学习笔记/xxx.md` | `images/Pasted-image-xxx.png` |
| `学习笔记/基础概念/xxx.md` | `../images/Pasted-image-xxx.png` |
| `学习笔记/训练/xxx.md` | `../images/Pasted-image-xxx.png` |
| `AI生成/xxx.md` | `images/Pasted-image-xxx.png` |

Formula: `("../" * (depth - 1)) + "images/"` where depth = path component count.

## Obsidian Configuration

### Files & Links
- **Attachment folder path**: `学习笔记/images`
- **Always update internal links**: Enabled

### Obsidian Git Plugin (data.json)
```json
{
  "autoSaveInterval": 10,
  "autoPullOnBoot": true,
  "disablePush": false,
  "pullBeforePush": true,
  "syncMethod": "merge",
  "commitMessage": "vault backup: {{date}}",
  "commitMessageScript": "/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -ExecutionPolicy Bypass -File \".githooks/convert-before-commit.ps1\"; echo \"vault backup: {{date}}\""
}
```

## Git Configuration

### Proxy (phdddd's setup)
```bash
git config --global http.proxy http://127.0.0.1:10090
git config --global https.proxy http://127.0.0.1:10090
git config --global http.sslVerify false
```

### Hooks
```bash
git config core.hooksPath .githooks
```