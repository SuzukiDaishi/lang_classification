# Install the ggml DLLs for the Windows x64 release build.
param([string]$Archive)

$ErrorActionPreference = 'Stop'
$target = Join-Path $PSScriptRoot 'target'
$output = Join-Path $target 'release'
$null = New-Item -ItemType Directory -Force -Path $output
if (-not $Archive) {
    $Archive = Join-Path $target 'llama-b12413-bin-win-vulkan-x64.zip'
    if (-not (Test-Path -LiteralPath $Archive)) {
        $url = 'https://github.com/ggml-org/llama.cpp/releases/download/b12413/llama-b12413-bin-win-vulkan-x64.zip'
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile "$Archive.part"
        Move-Item -LiteralPath "$Archive.part" -Destination $Archive
    }
}
$expected = '418dc12ea8189332e947ff2363cf8903a29b6ee68bf6d106dc86583f329ccf28'
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash -ne $expected) {
    throw "Unexpected ggml archive SHA-256: $Archive"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $Archive).Path)
try {
    foreach ($entry in $zip.Entries) {
        $name = $entry.FullName
        $subdir = switch -Regex ($name) {
            '^(ggml|ggml-base|libomp)\.dll$' { '.' }
            '^ggml-cpu-[a-z0-9]+\.dll$' { 'cpu' }
            '^ggml-vulkan\.dll$' { 'vulkan' }
        }
        if ($subdir) {
            $dir = Join-Path $output $subdir
            $null = New-Item -ItemType Directory -Force -Path $dir
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $dir $name), $true)
        }
    }
} finally {
    $zip.Dispose()
}
