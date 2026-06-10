# ============================================================
# Post-merge: Convert GitHub Markdown -> Obsidian wiki-links
# ============================================================
$vault = git rev-parse --show-toplevel

$changed = git -c core.quotepath=false diff-tree --name-only -r HEAD@{1} HEAD -- '*.md' 2>$null
if (-not $changed) { exit 0 }

foreach ($file in $changed) {
    $path = Join-Path $vault $file
    if (-not (Test-Path -LiteralPath $path)) { continue }

    $content = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
    $newContent = $content

    # Pattern 1: ![Pasted image xxx.png](images/Pasted-image-xxx.png)
    $newContent = [regex]::Replace($newContent, '!\[(Pasted image \d+\.png)\]\(images/Pasted-image-\d+\.png\)', {
        return "![[$($args[0].Groups[1].Value)]]"
    })

    # Pattern 2: ![Pasted image xxx.png](<Pasted image xxx.png>)
    $newContent = [regex]::Replace($newContent, '!\[(Pasted image \d+\.png)\]\(<Pasted image \d+\.png>\)', {
        return "![[$($args[0].Groups[1].Value)]]"
    })

    if ($newContent -ne $content) {
        [System.IO.File]::WriteAllText($path, $newContent, [System.Text.Encoding]::UTF8)
    }
}
