# ============================================================
# Pre-commit: Convert Obsidian wiki-links -> GitHub Markdown
# ============================================================
$vault = git rev-parse --show-toplevel

# --- Rename newly pasted images: "Pasted image xxx.png" -> "Pasted-image-xxx.png" ---
Get-ChildItem -Path $vault -Recurse -Filter "Pasted image *.png" -ErrorAction SilentlyContinue | ForEach-Object {
    $newName = $_.Name -replace ' ', '-'
    if ($_.Name -ne $newName) {
        $newPath = Join-Path $_.DirectoryName $newName
        Rename-Item -LiteralPath $_.FullName -NewName $newName -Force
        git add -- $_.FullName $newPath 2>$null
    }
}

# --- Convert wiki-links in staged .md files ---
$staged = git -c core.quotepath=false diff --cached --name-only --diff-filter=ACM -- '*.md'
foreach ($file in $staged) {
    $path = Join-Path $vault $file
    if (-not (Test-Path -LiteralPath $path)) { continue }

    $content = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
    $newContent = [regex]::Replace($content, '!\[\[Pasted image ([^\]|]+)(?:\|[^\]]+)?\]\]', {
        $fn = $args[0].Groups[1].Value
        $fnHyphen = $fn -replace ' ', '-'
        return "![Pasted image $fn](images/Pasted-image-$fnHyphen)"
    })

    if ($newContent -ne $content) {
        [System.IO.File]::WriteAllText($path, $newContent, [System.Text.Encoding]::UTF8)
        git add -- $file
    }
}
