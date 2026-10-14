[CmdletBinding()]
param([string]$Root = (Split-Path -Parent $PSScriptRoot))

$ErrorActionPreference = 'Stop'
$sourceRoot = [IO.Path]::GetFullPath($Root)
$target = [IO.Path]::GetFullPath((Join-Path $sourceRoot 'release-staging/source'))
$ownershipMarker = Join-Path (Split-Path -Parent $target) '.source-export-managed'
if (-not $target.StartsWith($sourceRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Export destination escaped project root: $target"
}

# Source only: every path is individually named. Never recursively copy a project directory.
$allowlist = @(
    'README.md', 'NOTICE.md', 'LICENSE',
    'LICENSES/FISH_AUDIO_RESEARCH_LICENSE.md',
    'LICENSES/FISH_AUDIO_COMMUNITY_LICENSE.md', 'LICENSES/GGML_MIT_LICENSE.txt',
    'LICENSES/THIRD_PARTY_NOTICES.md',
    'pyproject.toml', 'requirements-runtime.txt',
    'data/source/drama3-dd64a85-sha256.json',
    '.github/workflows/ci.yml',
    'fish_drama3/__init__.py', 'fish_drama3/cli.py', 'fish_drama3/direct.py',
    'fish_drama3/generation.py', 'fish_drama3/text_chunks.py',
    'fish_drama3/convert.py', 'fish_drama3/gguf_stream.py', 'fish_drama3/native_quant.py',
    'fish_drama3/hybrid.py', 'fish_drama3/codec_adapter.py', 'fish_drama3/watchdog.py',
    'fish_drama3/qualification.py',
    'examples/generate_model_card_demos.py',
    'data/qualification/cases.jsonl',
    'data/qualification/lifecycle-cases.jsonl',
    'data/qualification/windows-010-lifecycle-report.json',
    'native/CMakeLists.txt', 'native/cuda/CMakeLists.txt',
    'native/codec_bridge.cpp', 'native/codec_ops.cpp', 'native/experts_bridge.cpp',
    'native/model_inspect.cpp', 'native/attention_fixture.cpp',
    'native/gdn_fixture.cpp', 'native/moe_fixture.cpp',
    'tests/test_cli.py', 'tests/test_codec_adapter.py', 'tests/test_direct.py',
    'tests/test_generation.py', 'tests/test_text_chunks.py',
    'tests/test_gguf_stream.py', 'tests/test_hybrid.py',
    'tests/test_native_quant.py', 'tests/test_qualification.py', 'tests/test_watchdog.py',
    'scripts/bootstrap.ps1', 'scripts/build.ps1', 'scripts/package-source.ps1',
    'docs/public/INSTALL.md', 'docs/public/USER_GUIDE.md',
    'docs/public/MODEL_CARD.md', 'docs/public/QUALITY.md'
)
$requiredPaths = @(
    'README.md', 'NOTICE.md', 'LICENSE', 'pyproject.toml', 'requirements-runtime.txt',
    'data/source/drama3-dd64a85-sha256.json',
    'LICENSES/FISH_AUDIO_RESEARCH_LICENSE.md',
    'LICENSES/FISH_AUDIO_COMMUNITY_LICENSE.md', 'LICENSES/GGML_MIT_LICENSE.txt',
    'LICENSES/THIRD_PARTY_NOTICES.md',
    'fish_drama3/__init__.py', 'fish_drama3/cli.py', 'fish_drama3/direct.py',
    'fish_drama3/generation.py', 'fish_drama3/text_chunks.py',
    'fish_drama3/convert.py', 'fish_drama3/gguf_stream.py', 'fish_drama3/native_quant.py',
    'fish_drama3/hybrid.py', 'fish_drama3/codec_adapter.py',
    'fish_drama3/watchdog.py',
    'fish_drama3/qualification.py',
    'examples/generate_model_card_demos.py',
    'native/CMakeLists.txt', 'native/cuda/CMakeLists.txt',
    'native/codec_bridge.cpp', 'native/codec_ops.cpp', 'native/experts_bridge.cpp',
    'native/model_inspect.cpp', 'native/attention_fixture.cpp', 'native/gdn_fixture.cpp',
    'native/moe_fixture.cpp', 'tests/test_direct.py', 'tests/test_gguf_stream.py',
    'tests/test_generation.py', 'tests/test_text_chunks.py',
    'tests/test_cli.py', 'tests/test_codec_adapter.py', 'tests/test_native_quant.py',
    'tests/test_hybrid.py', 'tests/test_qualification.py', 'scripts/bootstrap.ps1',
    'scripts/build.ps1', 'tests/test_watchdog.py', 'data/qualification/cases.jsonl',
    'data/qualification/lifecycle-cases.jsonl',
    'data/qualification/windows-010-lifecycle-report.json',
    'docs/public/INSTALL.md', 'docs/public/USER_GUIDE.md',
    'docs/public/MODEL_CARD.md', 'docs/public/QUALITY.md',
    'scripts/package-source.ps1'
)
foreach ($relativePath in $requiredPaths) {
    $source = Join-Path $sourceRoot ($relativePath -replace '/', [IO.Path]::DirectorySeparatorChar)
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required release source file is missing: $relativePath"
    }
}

if (Test-Path -LiteralPath $target) {
    $legacyMarker = Join-Path $target '.drama3-source-export'
    if (-not (Test-Path -LiteralPath $ownershipMarker) -and
        -not (Test-Path -LiteralPath $legacyMarker)) {
        throw "Refusing to replace unmarked export directory: $target"
    }
    $allowedPaths = @($allowlist | ForEach-Object { ($_ -replace '/', [IO.Path]::DirectorySeparatorChar).ToLowerInvariant() })
    foreach ($existing in Get-ChildItem -LiteralPath $target -File -Force -Recurse) {
        $relative = $existing.FullName.Substring($target.Length + 1).ToLowerInvariant()
        if ($relative -ne '.drama3-source-export' -and $relative -notin $allowedPaths) {
            throw "Refusing to replace export containing an unallowlisted path: $relative"
        }
    }
    Remove-Item -LiteralPath $target -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $target | Out-Null
Set-Content -LiteralPath $ownershipMarker -Value 'Generated by scripts/package-source.ps1' -Encoding utf8

foreach ($relativePath in $allowlist) {
    $source = Join-Path $sourceRoot ($relativePath -replace '/', [IO.Path]::DirectorySeparatorChar)
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
    $destination = Join-Path $target ($relativePath -replace '/', [IO.Path]::DirectorySeparatorChar)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}

$forbidden = @('.git', 'upstream', 'builds', 'checkpoints', 'models', '.hf-cache', 'artifacts', 'staging', 'docs/FULL_PLAN.md', 'docs/WINDOWS_PROTOTYPE.md', 'fish_drama3/voice_variants.py', 'fish_drama3/reference_codec.py', 'fish_drama3/diagnose_source.py', 'fish_drama3/source_experts.py', 'tests/test_source_experts.py')
foreach ($entry in $forbidden) {
    $escaped = [regex]::Escape(($entry -replace '/', [IO.Path]::DirectorySeparatorChar))
    if (Get-ChildItem -LiteralPath $target -Force -Recurse | Where-Object { $_.FullName -match "[\\/]$escaped([\\/]|$)" }) {
        throw "Forbidden path was included in source export: $entry"
    }
}
Write-Host "Created source-only export at $target ($($allowlist.Count) allowlisted paths checked)."
