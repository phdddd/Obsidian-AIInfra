# ============================================================
# Post-commit: Restore Obsidian wiki-link format in all files
# ============================================================
$vault = git rev-parse --show-toplevel

Get-ChildItem $vault -Recurse -Filter "*.md" -ErrorAction SilentlyContinue | ForEach-Object {
    $content = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
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
        [System.IO.File]::WriteAllText($_.FullName, $newContent, [System.Text.Encoding]::UTF8)
    }
}
