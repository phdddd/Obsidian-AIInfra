# Post-merge: GitHub Markdown -> Obsidian wiki-links (merge-changed files only)
$vault = git rev-parse --show-toplevel
$changed = git -c core.quotepath=false diff-tree --name-only -r HEAD@{1} HEAD -- '*.md' 2>$null
if (-not $changed) { exit 0 }
foreach ($file in $changed) {
    $path = Join-Path $vault $file
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $c = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
    $n = [regex]::Replace($c, '!\[Pasted-image-(\d+\.png)\]\((?:\.\./)*images/Pasted-image-\d+\.png\)', {
        "![[Pasted-image-$($args[0].Groups[1].Value)]]"
    })
    if ($n -ne $c) { [System.IO.File]::WriteAllText($path, $n, [System.Text.Encoding]::UTF8) }
}
