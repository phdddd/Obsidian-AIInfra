$vault = git rev-parse --show-toplevel
Get-ChildItem $vault -Recurse -Filter '*.md' -ErrorAction SilentlyContinue | ForEach-Object {
    $c = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
    $n = [regex]::Replace($c, '!\[\[Pasted-image-(\d+\.png)\]\]', {
        $fn = $args[0].Groups[1].Value
        $relPath = $_.FullName.Replace($vault, '').TrimStart('\')
        $parts = $relPath -split '[/\\]'
        $depth = $parts.Count - 1
        if ($depth -le 1) { $prefix = 'images/' } else { $prefix = ('../' * ($depth - 1)) + 'images/' }
        return "![Pasted-image-$fn]($($prefix)Pasted-image-$fn)"
    })
    if ($n -ne $c) { [System.IO.File]::WriteAllText($_.FullName, $n, [System.Text.Encoding]::UTF8) }
}