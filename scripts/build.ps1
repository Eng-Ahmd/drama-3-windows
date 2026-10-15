[CmdletBinding()]
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [string]$Generator = 'Ninja',
    [string]$CudaToolkitRoot
)

$ErrorActionPreference = 'Stop'
$upstream = Join-Path $Root 'upstream/s2.cpp'
if (-not (Test-Path -LiteralPath (Join-Path $upstream 'CMakeLists.txt'))) {
    throw 'Pinned donor checkout is missing. Run scripts/bootstrap.ps1 first.'
}
$s2Pin = '2c33261938da1a41d713768b1b391b4d368d7d2c'
$ggmlPin = '57ea0bc119d722d74594196cc5b494a34dd87be4'
if ((git -C $upstream rev-parse HEAD).Trim() -ne $s2Pin) { throw 'Donor HEAD differs from the required pristine pin.' }
if ((git -C (Join-Path $upstream 'ggml') rev-parse HEAD).Trim() -ne $ggmlPin) { throw 'GGML HEAD differs from the required pin.' }
if ((git -C $upstream status --porcelain) -or (git -C (Join-Path $upstream 'ggml') status --porcelain)) {
    throw 'Donor checkout contains local changes; refusing to build against it.'
}

$cpuBuild = Join-Path $Root 'builds/s2-cpu'
$cudaBuild = Join-Path $Root 'builds/s2-cuda'
cmake -S $upstream -B $cpuBuild -G $Generator -DCMAKE_BUILD_TYPE=Release -DS2_AUTO_APPLY_LOCAL_PATCHES=OFF
if ($LASTEXITCODE -ne 0) { throw 'CPU donor configuration failed.' }
cmake --build $cpuBuild --target s2 --config Release --parallel 1
if ($LASTEXITCODE -ne 0) { throw 'CPU donor build failed.' }

$cudaRoot = $CudaToolkitRoot
if (-not $cudaRoot -and (Test-Path 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe')) {
    $cudaRoot = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3'
}
if (-not $cudaRoot -and $env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH }
if (-not $cudaRoot) { throw 'CUDA Toolkit 13.3 or newer is required; set -CudaToolkitRoot to its installation path.' }
$nvccPath = Join-Path $cudaRoot 'bin/nvcc.exe'
if (-not (Test-Path -LiteralPath $nvccPath)) { throw "nvcc.exe was not found under the selected CUDA toolkit: $cudaRoot" }
$nvccVersionText = (& $nvccPath --version 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "Could not query CUDA compiler version: $nvccPath" }
$versionMatch = [regex]::Match($nvccVersionText, 'release\s+(\d+)\.(\d+)')
if (-not $versionMatch.Success) { throw "Could not parse CUDA compiler version from nvcc output: $nvccPath" }
$toolkitVersion = [version]::new([int]$versionMatch.Groups[1].Value, [int]$versionMatch.Groups[2].Value)
if ($toolkitVersion -lt [version]'13.3') { throw "CUDA Toolkit 13.3 or newer is required; selected toolkit reports $toolkitVersion ($cudaRoot)." }
$cudaArgs = @('-S', $upstream, '-B', $cudaBuild, '-G', $Generator, '-DCMAKE_BUILD_TYPE=Release', '-DS2_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=120', '-DS2_AUTO_APPLY_LOCAL_PATCHES=OFF', "-DCUDAToolkit_ROOT=$cudaRoot")
cmake @cudaArgs
if ($LASTEXITCODE -ne 0) { throw 'CUDA donor configuration failed.' }
cmake --build $cudaBuild --target s2 --config Release --parallel 1
if ($LASTEXITCODE -ne 0) { throw 'CUDA donor build failed.' }

$nativeBuild = Join-Path $Root 'builds/drama3-inspect'
$nativeOutput = Join-Path $nativeBuild 'Release'
$nativeOutputArgs = @("-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$nativeOutput", "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$nativeOutput")
if ($Generator -like 'Visual Studio*') {
    $nativeOutputArgs = @("-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=$nativeOutput", "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=$nativeOutput")
}
cmake -S (Join-Path $Root 'native') -B $nativeBuild -G $Generator -DCMAKE_BUILD_TYPE=Release -DDONOR_BUILD=$cpuBuild @nativeOutputArgs
if ($LASTEXITCODE -ne 0) { throw 'Native bridge configuration failed.' }
cmake --build $nativeBuild --config Release --parallel 1
if ($LASTEXITCODE -ne 0) { throw 'Native bridge build failed.' }
$nativeCudaBuild = Join-Path $Root 'builds/drama3-cuda'
$nativeCudaOutput = Join-Path $nativeCudaBuild 'Release'
$nativeCudaOutputArgs = @("-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$nativeCudaOutput", "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$nativeCudaOutput")
if ($Generator -like 'Visual Studio*') {
    $nativeCudaOutputArgs = @("-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=$nativeCudaOutput", "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=$nativeCudaOutput")
}
cmake -S (Join-Path $Root 'native/cuda') -B $nativeCudaBuild -G $Generator -DCMAKE_BUILD_TYPE=Release -DDONOR_BUILD=$cudaBuild -DCPU_DONOR_BUILD=$cpuBuild @nativeCudaOutputArgs
if ($LASTEXITCODE -ne 0) { throw 'Native CUDA bridge configuration failed.' }
cmake --build $nativeCudaBuild --config Release --parallel 1
if ($LASTEXITCODE -ne 0) { throw 'Native CUDA bridge build failed.' }
Write-Host "CPU, CUDA 13.3+ SM 120, and CPU/CUDA native bridge targets built under $Root/builds."
