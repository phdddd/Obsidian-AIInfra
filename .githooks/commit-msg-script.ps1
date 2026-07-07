# Convert Obsidian -> GitHub before commit, then output commit message
$vault = git rev-parse --show-toplevel
Get-ChildItem $vault -Recurse -Filter '*.md' -ErrorAction SilentlyContinue | ForEach-Object {
    $c = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
    $n = [regex]::Replace($c, '!\[\[Pasted-image-(\d+\.png)\]\]', {
        $fn = $args[0].Groups[1].Value
        $parts = $_.FullName.Replace($vault, '').TrimStart('\') -split '[/\\]'
        $depth = $parts.Count - 1
        if ($depth -le 1) { $prefix = 'images/' } else { $prefix = ('../' * ($depth - 1)) + 'images/' }
        return "![Pasted-image-$fn]($prefixPasted-image-$fn)"
    })
    if ($n -ne $c) { [System.IO.File]::WriteAllText($_.FullName, $n, [System.Text.Encoding]::UTF8) }
}
Write-Output 'vault backup: {{date}}'