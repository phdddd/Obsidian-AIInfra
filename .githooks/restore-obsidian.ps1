# Post-commit: GitHub Markdown -> Obsidian wiki-links (all files)
$vault = git rev-parse --show-toplevel
Get-ChildItem $vault -Recurse -Filter "*.md" -ErrorAction SilentlyContinue | ForEach-Object {
    $c = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
    $n = [regex]::Replace($c, '!\[Pasted-image-(\d+\.png)\]\(images/Pasted-image-\d+\.png\)', { "![[Pasted-image-$($args[0].Groups[1].Value)]]" })
    $n = [regex]::Replace($n, '!\[(Pasted image \d+\.png)\]\(images/Pasted-image-\d+\.png\)', { "![[$($args[0].Groups[1].Value)]]" })
    $n = [regex]::Replace($n, '!\[(Pasted image \d+\.png)\]\(<Pasted image \d+\.png>\)', { "![[$($args[0].Groups[1].Value)]]" })
    if ($n -ne $c) { [System.IO.File]::WriteAllText($_.FullName, $n, [System.Text.Encoding]::UTF8) }
}
