# Post-commit: GitHub Markdown -> Obsidian wiki-links (all files)
$vault = git rev-parse --show-toplevel
Get-ChildItem $vault -Recurse -Filter "*.md" -ErrorAction SilentlyContinue | ForEach-Object {
    $c = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
    # Match any relative path: images/ or ../images/ or ../../images/
    $n = [regex]::Replace($c, '!\[Pasted-image-(\d+\.png)\]\((?:\.\./)*images/Pasted-image-\d+\.png\)', {
        "![[Pasted-image-$($args[0].Groups[1].Value)]]"
    })
    if ($n -ne $c) { [System.IO.File]::WriteAllText($_.FullName, $n, [System.Text.Encoding]::UTF8) }
}
