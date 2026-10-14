# Installation

This guide installs or builds the Windows release `0.1.0` on Windows 11. The current environment has been exercised with Python 3.13, PyTorch 2.13.0+cu132, Visual Studio 2022, and CUDA Toolkit 13.3. A clean-machine installation has not yet been qualified.

## Prerequisites

- Windows 11 x64 and Python 3.13.
- The prebuilt CPU libraries require AVX2, FMA and F16C instructions. The CUDA
  libraries target SM120; other GPU architectures require a compatible source build.
- Visual Studio 2022 C++ build tools, CMake, and Ninja are required only when compiling the runtime from source.
- For CUDA builds, an NVIDIA driver and CUDA Toolkit 13.3. The observed target was an RTX 5080 Laptop GPU with 16 GiB of VRAM.
- The converted [model and codec bundles](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF), accessed under the applicable licenses. The original pinned checkpoint is needed only for local conversion. This source repository does not bundle weights.
- Enough local disk for the source checkpoint, model and codec bundles, and build outputs. A complete converted model bundle is about 21.67 GiB and the codec about 1.46 GiB.

## Use the prebuilt Windows runtime

The `fish-drama3-windows-runtime-0.1.0-win-x64.zip` release archive contains the tested CPU and CUDA runtime DLLs. Extract it at the repository root so the files land under `builds/drama3-inspect/Release` and `builds/drama3-cuda/Release`; do not rename those directories. Check the included `SHA256SUMS.txt` against the extracted DLL and license files before use.

Install the Microsoft Visual C++ 2015–2022 x64 Redistributable separately; it is not included. CUDA inference also requires an NVIDIA driver and CUDA Toolkit 13.3 or newer. NVIDIA and Microsoft DLLs are not redistributed in the archive. The Python environment and pinned Fish Speech reference checkout are still required; run `scripts/bootstrap.ps1` to fetch and verify source pins. The prebuilt archive avoids compiling the runtime libraries. The included CPU GGML library also provides the pinned quantizer for local conversion; a source build is optional when using this archive.

The archive does not include model weights. Use it with a converted model and codec bundle that you have obtained or built under their applicable licenses. Use `--gpu-expert-layers 0` for the CPU expert bridge; CUDA expert placement requires a compatible GPU and the CUDA prerequisites above.
## Create the Python environment

Create and activate a virtual environment in PowerShell, then install the CUDA-enabled PyTorch build from the official PyTorch wheel index. Visual Studio C++ tools are needed only for source builds. PyTorch is intentionally separate from the PyPI requirements file so a generic CPU or unrelated CUDA wheel is not selected accidentally.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu132 torch==2.13.0+cu132
python -m pip install -r requirements-runtime.txt
python -m pip install -e .
```

The pinned Python dependencies are NumPy 2.2.6, safetensors 0.6.2, gguf 0.18.0, tokenizers 0.22.1, SoundFile 0.13.1, psutil 7.0.0, SciPy 1.18.1, loguru 0.7.3, and tqdm 4.67.1. Confirm the environment before continuing:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The expected PyTorch version is `2.13.0+cu132`. CUDA availability should be `True` for CUDA inference. CPU-only operation is useful for development but has not been qualified for full-model inference.

## Fetch pinned source and build native libraries

Bootstrap fetches the exact donor and reference commits, checks the GGML submodule commit, and refuses to replace an existing checkout whose origin, revision, or working tree differs. It fetches the Fish Speech reference only in the sparse paths needed by the runtime.

```powershell
pwsh -File scripts/bootstrap.ps1
pwsh -File scripts/build.ps1 -CudaToolkitRoot "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
```

The build compiles the pinned pristine donor for CPU and CUDA, then builds the project CPU and CUDA native bridges in sequence. The CUDA profile checks that the selected toolkit is 13.3 or newer and targets SM 120 for the tested RTX 5080 profile. Selection order is the explicit `-CudaToolkitRoot`, the standard installed CUDA 13.3 path, then `CUDA_PATH`; an older toolkit such as 12.8 is rejected. Each native build uses one build worker to avoid a known parallel MSBuild hang. Build outputs remain under the ignored `builds/` directory.

The recorded pins are:

- Drama 3 checkpoint: `dd64a85cd82e9521f7202610ede97b367fc83946`
- Fish Speech reference: `1ff3c3ed2f80d3106a83909dd25b83129c3d816f`
- s2.cpp: `2c33261938da1a41d713768b1b391b4d368d7d2c`
- GGML: `57ea0bc119d722d74594196cc5b494a34dd87be4`

## Download the converted model and codec

The [Hugging Face bundle](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF)
contains ready-to-use model and codec files with their verification manifests.
Install the download client and fetch each bundle into the directories used below:

```powershell
python -m pip install "huggingface_hub>=1.0,<2"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Eng-Ahmd/Drama-3-Windows-GGUF', local_dir='models/drama3-windows', allow_patterns=['model/*', 'codec/*', 'LICENSE', 'LICENSES/*', 'NOTICE.md'])"
fish-drama3 synth --bundle models/drama3-windows/model --codec models/drama3-windows/codec/codec.gguf --text "Hello. This is local speech synthesis." --output outputs/hello.wav --gpu-expert-layers 12
```

This download needs about 23.13 GiB for the model and codec. Bootstrap, Python
and native libraries are still required; converting the original checkpoint is
optional when using these bundles. The CLI verifies each complete file against
its manifest before loading it.

## Convert a local checkpoint

Place the complete pinned checkpoint files in a local directory such as `checkpoints/drama-3`. Conversion verifies the source files against [`data/source/drama3-dd64a85-sha256.json`](../../data/source/drama3-dd64a85-sha256.json), then writes a new immutable model bundle. Do not edit the pin file to accept different weights.

```powershell
fish-drama3-convert --checkpoint checkpoints/drama-3 --output models/drama3-q4k-q8
python -m fish_drama3.codec_adapter build `
  --source models/drama3-q4k-q8/codec.fp32.safetensors `
  --destination models/drama3-codec-adapter-v3
```

The converter produces `model.gguf` plus `codec.fp32.safetensors`; the second command builds the separate `codec.gguf` bundle used by the native codec adapter. Both commands need the pinned CPU GGML/native libraries built above. Each output directory must not already exist. Review both manifests before use; conversion manifests do not claim speech or runtime qualification.

## Run a short synthesis

```powershell
fish-drama3 synth `
  --bundle models/drama3-q4k-q8 `
  --codec models/drama3-codec-adapter-v3/codec.gguf `
  --text "Hello. This is a short local test." `
  --output outputs/hello.wav `
  --gpu-expert-layers 12
```

The CLI checks model and codec hashes against their manifests, refuses to overwrite an existing output, and writes PCM16 WAV plus a JSON run record. It requires EOS by default; if the frame limit is reached, it fails without publishing audio. `--allow-partial` is an explicit opt-in to incomplete audio, and the record marks it incomplete. The current default frame limit is 512.

For a first run, use a short prompt and preserve the JSON record alongside the WAV. The GPU expert-layer count controls expert placement only; it does not guarantee a particular speed or memory requirement.

## Verify the checkout

The CPU contract tests avoid model downloads. The native quantization tests require the pinned CPU GGML library, so run them after the build:

```powershell
python -m pytest tests/test_direct.py tests/test_gguf_stream.py tests/test_native_quant.py
```

The CI job also runs watchdog logic tests but excludes the live NVIDIA sensor check; CI does not claim GPU qualification. Additional native fixture tests require the pinned Fish Speech reference and corresponding built fixtures.

## Licensing

The runtime contains separately licensed model-related and codec-donor components. The s2.cpp-derived codec is subject to the Fish Audio Research License and requires a separate written agreement for commercial use. Review [NOTICE.md](../../NOTICE.md) and the component texts under [`LICENSES/`](../../LICENSES/THIRD_PARTY_NOTICES.md) before building or redistributing.

