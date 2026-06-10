# ============================================================
# Post-commit: Restore Obsidian wiki-link format in working tree
# ============================================================
$vault = git rev-parse --show-toplevel

$committed = git -c core.quotepath=false diff-tree --name-only -r HEAD~1 HEAD -- '*.md' 2>$null
if (-not $committed) { exit 0 }

foreach ($file in $committed) {
    $path = Join-Path $vault $file
    if (-not (Test-Path -LiteralPath $path)) { continue }

    $content = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
    $newContent = [regex]::Replace($content, '!\[(Pasted image \d+\.png)\]\(images/Pasted-image-\d+\.png\)', {
        $alt = $args[0].Groups[1].Value
        return "![[$alt]]"
    })

    if ($newContent -ne $content) {
        [System.IO.File]::WriteAllText($path, $newContent, [System.Text.Encoding]::UTF8)
    }
}
