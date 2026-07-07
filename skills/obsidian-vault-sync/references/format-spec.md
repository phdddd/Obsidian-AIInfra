# Image Format Specification

## Filename Convention

All images use hyphenated names:
- `Pasted-image-YYYYMMDDHHmmss.png`
- Example: `Pasted-image-20260524192022.png`

Never use spaces in image filenames. The pre-commit conversion renames any `Pasted image *.png` → `Pasted-image-*.png`.

## Format Rules

### Obsidian Wiki-Link (working tree only)

```
![[Pasted-image-20260524192022.png]]
```

- No path component — Obsidian searches entire vault by filename
- Used in working tree files for Obsidian rendering
- Never committed to git

### GitHub Markdown (committed)

```
![Pasted-image-20260524192022.png](../images/Pasted-image-20260524192022.png)
```

- Relative path from the .md file to its section's `images/` folder
- Path depth computed from file location

### Path Calculation

| File location | Image path |
|---|---|
| `学习笔记/xxx.md` | `images/Pasted-image-xxx.png` |
| `学习笔记/基础概念/xxx.md` | `../images/Pasted-image-xxx.png` |
| `学习笔记/训练/xxx.md` | `../images/Pasted-image-xxx.png` |
| `AI生成/xxx.md` | `images/Pasted-image-xxx.png` |
| `AI生成/sub/xxx.md` | `../images/Pasted-image-xxx.png` |

Rule: `("../" * (depth - 1)) + "images/"` where depth = number of path components in the file's relative path.

## Obsidian Configuration

### Files & Links Settings
- **Attachment folder path**: `学习笔记/images` (or the active section's images folder)
- **Always update internal links**: Enabled

### Obsidian Git Plugin Settings (data.json)
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

The `commitMessageScript` runs the conversion before every commit as a fallback when git hooks don't fire.

## Git Configuration

### .gitignore
```
.obsidian/workspace.json
.obsidian/workspace-mobile.json
.trash/
.DS_Store
Thumbs.db
```

### Proxy (China mainland users)
```bash
git config --global http.proxy http://127.0.0.1:PORT
git config --global https.proxy http://127.0.0.1:PORT
git config --global http.sslVerify false
```
