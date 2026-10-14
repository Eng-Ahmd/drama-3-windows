[CmdletBinding()]
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [switch]$SkipFishReference
)

$ErrorActionPreference = 'Stop'
$s2Commit = '2c33261938da1a41d713768b1b391b4d368d7d2c'
$ggmlCommit = '57ea0bc119d722d74594196cc5b494a34dd87be4'
$fishCommit = '1ff3c3ed2f80d3106a83909dd25b83129c3d816f'

function Get-GitHead([string]$Path) {
    return (git -C $Path rev-parse HEAD).Trim()
}

function Ensure-PinnedClone([string]$Path, [string]$Url, [string]$Commit, [string[]]$SparsePaths = @()) {
    if (Test-Path -LiteralPath $Path) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path '.git'))) {
            throw "Refusing to replace existing non-Git directory: $Path"
        }
        $origin = (git -C $Path remote get-url origin).Trim()
        if ($origin -ne $Url -or (Get-GitHead $Path) -ne $Commit) {
            throw "Existing checkout is not the requested pristine pin: $Path (origin=$origin, HEAD=$(Get-GitHead $Path))"
        }
        $changes = git -C $Path status --porcelain
        if ($LASTEXITCODE -ne 0 -or $changes) { throw "Existing checkout has local changes; inspect it manually: $Path" }
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
        git clone --filter=blob:none --no-checkout $Url $Path
        if ($LASTEXITCODE -ne 0) { throw "Clone failed: $Url" }
        if ($SparsePaths.Count) {
            git -C $Path sparse-checkout init --no-cone
            if ($LASTEXITCODE -ne 0) { throw "Could not initialize sparse checkout: $Path" }
            $patterns = Join-Path $Path '.git/info/sparse-checkout'
            Set-Content -LiteralPath $patterns -Value $SparsePaths -Encoding ascii
        }
        git -C $Path fetch --depth 1 origin $Commit
        if ($LASTEXITCODE -ne 0) { throw "Fetch failed for $Commit" }
        git -C $Path checkout --detach $Commit
        if ($LASTEXITCODE -ne 0) { throw "Checkout failed for $Commit" }
    }
}

$s2Path = Join-Path $Root 'upstream/s2.cpp'
Ensure-PinnedClone $s2Path 'https://github.com/rodrigomatta/s2.cpp.git' $s2Commit
git -C $s2Path submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw 'Could not initialize pinned s2.cpp submodules.' }
$actualGgml = Get-GitHead (Join-Path $s2Path 'ggml')
if ($actualGgml -ne $ggmlCommit) { throw "GGML pin mismatch: expected $ggmlCommit, got $actualGgml" }

if (-not $SkipFishReference) {
    $fishPath = Join-Path $Root 'upstream/fish-speech'
    $fishSparse = @('/fish_speech/models/text2semantic/fish_omni/**', '/fish_speech/models/text2semantic/errors.py', '/fish_speech/models/text2semantic/sampling.py')
    Ensure-PinnedClone $fishPath 'https://github.com/fishaudio/fish-speech.git' $fishCommit $fishSparse
}

Write-Host "Verified s2.cpp $s2Commit and GGML $ggmlCommit."
if (-not $SkipFishReference) { Write-Host "Verified sparse fish-speech reference $fishCommit." }
