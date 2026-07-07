# ============================================================
# Pre-commit: Obsidian -> GitHub Markdown
# Computes relative image path based on file location
# ============================================================
$vault = git rev-parse --show-toplevel

# 1. Rename newly pasted images: "Pasted image xxx.png" -> "Pasted-image-xxx.png"
Get-ChildItem -Path $vault -Recurse -Filter "Pasted image *.png" -ErrorAction SilentlyContinue | ForEach-Object {
    $newName = $_.Name -replace ' ', '-'
    if ($_.Name -ne $newName) {
        $newPath = Join-Path $_.DirectoryName $newName
        Rename-Item -LiteralPath $_.FullName -NewName $newName -Force
        git add -- $_.FullName $newPath 2>$null
    }
}

# 2. Convert wiki-links in staged .md files
$staged = git -c core.quotepath=false diff --cached --name-only --diff-filter=ACMR -- '*.md'
foreach ($file in $staged) {
    $path = Join-Path $vault $file
    if (-not (Test-Path -LiteralPath $path)) { continue }

    $content = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)

    # Compute relative path to the section's images/ folder
    $parts = $file -split '[/\\]'
    $depth = $parts.Count - 1  # 1 = in section root, 2+ = in subdir
    if ($depth -le 1) { $imgPrefix = "images/" }
    else { $imgPrefix = ("../" * ($depth - 1)) + "images/" }

    $newContent = [regex]::Replace($content, '!\[\[Pasted-image-(\d+\.png)\]\]', {
        $fn = $args[0].Groups[1].Value
        return "![Pasted-image-$fn]($($imgPrefix)Pasted-image-$fn)"
    })

    if ($newContent -ne $content) {
        [System.IO.File]::WriteAllText($path, $newContent, [System.Text.Encoding]::UTF8)
        git add -- $file
    }
}
