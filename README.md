# Fish Audio Drama 3 for Windows

This repository contains the **Windows release 0.2.0** for running Fish Audio Drama 3 on Windows. It combines a PyTorch model path with native GGML expert execution and a native audio codec. It is a hybrid Windows runtime, not a standalone C++ model engine.

The project includes a direct converter for the pinned Drama 3 checkpoint, an offline synthesis CLI, native CPU and CUDA components, and guarded evaluation tools. Model weights and compiled binaries are separate from this source tree. Download the converted model and codec from [Hugging Face](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF), then follow the installation guide. Use runtime 0.1.1 or newer for the current bundles. Local conversion from the pinned checkpoint is also supported. Obtain model files under their applicable license terms and install the [prebuilt Windows libraries](https://github.com/Eng-Ahmd/drama-3-windows/releases) or build them locally. Source code is hosted at the [GitHub repository](https://github.com/Eng-Ahmd/drama-3-windows).

[**Listen to the original / Windows audio comparisons**](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF#listen-original-vs-windows) — expressive tags, contextual edits and chained edits, generated from the original voice reference.

## Generate in your browser

The local web app requires the 0.2.0 Python runtime and matching native bridge. It provides model load/unload, sample scripts, supported generation settings, voice references, playback and WAV/MP3 downloads with saved settings. Install the runtime using the [installation guide](docs/public/INSTALL.md), then run:

```powershell
python -m fish_drama3.webui
```

Automatic placement keeps 3% of VRAM free after dense weights, request caches and a 256 MiB expert-workspace allowance. Manual placement supports 0–40 GPU expert layers; its initial value of 16 is not a fit guarantee. The continuous worker guard also preserves 8 GiB free host RAM. This is a memory budget, not a compute-utilization target.

Open [127.0.0.1:7860](http://127.0.0.1:7860). See the [web app guide](docs/public/WEB_UI.md) for model paths, controls and FFmpeg setup for MP3 export.

## Podcast example

[**Listen to The Small Signal · 3:16**](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF#podcast--the-small-signal): a complete [592-word podcast script](examples/samples/podcast.txt), generated in **one request without text chunks, retries or speech edits**. Load the sample in the studio to try it yourself.

[WAV](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/demos/podcast-small-signal.wav) · [MP3](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/demos/podcast-small-signal.mp3) · [Importable settings and provenance](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/demos/podcast-small-signal-settings.json).

The recording ended normally at 4216 frames and measured 0.34% ASR WER (2 edits / 592 words), with no clipped samples. Prefill and generation took 520.00 seconds; CPU codec decoding added 223.34 seconds. These exclude model loading and MP3 export. This is one checked example; long-form output can still vary and automated transcription does not replace listening.

## Start here

- [Installation](docs/public/INSTALL.md) covers the Windows toolchain, Python environment, pinned sources, conversion, and a first synthesis command.
- [Browser app](docs/public/WEB_UI.md) covers local generation, sample loading, settings and audio export.
- [User guide](docs/public/USER_GUIDE.md) describes conversion, voice references, output behavior, and guarded runs.
- [Model card](docs/public/MODEL_CARD.md) documents the checkpoint, quantized bundle, measured results, and limitations.
- [Quality and qualification](docs/public/QUALITY.md) records the current evidence and unresolved release gates.
- [Attribution and licensing](NOTICE.md) explains the separate model, reference-code, and codec-donor terms.

## Runtime at a glance

The inference path keeps the backbone and audio tower in PyTorch, runs routed experts through native GGML CPU/CUDA libraries, and uses the native codec to encode reference audio and decode generated acoustic codes. The converter emits a custom `fish_drama3` GGUF architecture. It requires this matching runtime; generic llama.cpp does not provide Drama 3 support.

The speech model has 35.9B total parameters, matching the pinned upstream checkpoint. The separate codec is approximately 0.393B stored tensor elements. Quantization reduces storage precision, not parameter count.

The conversion recipe stores expert weights as Q4_K, the backbone as Q8_0, sensitive vectors as F32, and the audio tower in BF16 and the learned codec weights in F32, preserving the codec RoPE tables in BF16. A complete model bundle is about 21.67 GiB and its codec is about 1.46 GiB. Conversion is direct from the pinned FP8/BF16 source without a full FP32 model intermediate.

## Current evidence and limits

The native codec reproduced all 850 reference acoustic codes in the checked example. Decoding 87 frames produced the same 178,176 samples as the reference decode, with relative RMSE 1.74×10⁻⁶. One matched English voice-reference example had 0.0 Whisper WER for both the original and generated audio. A separate guarded CLI run completed at frame 169, producing 7.848 seconds of audio in 39 seconds; ASR recognized the requested words but inserted “Haha” (WER 0.0714). This is one sample, not a speech-quality or speed benchmark.

The CLI uses bounded text segments with the pinned model's audio-history conditioning by default. On the checked three-sentence passage, this completed all text in 14.12 seconds of audio and removed the unchunked run's long silent interval. Four short English checks had zero ASR WER; the checked Chinese plain example had zero CER. These are individual checks, with settings and limitations recorded in the [quality report](docs/public/QUALITY.md).

Generation is stochastic: the recorded numbers prompt stalled at seed 124, while the default seed 123 completed both sentences. Frame limits fail without publishing incomplete audio unless explicitly requested. A guarded two-hour repeated-request soak passed with 713 successful speech requests and 143 validation recovery cycles. Larger acoustic holdouts, cancellation recovery, clean-machine installation and varied-content coverage remain documented work.

## Licensing

Built with Fish Audio. Drama 3 materials retain the Fish Audio Community License v1.0. The s2.cpp-derived codec and its derivative work are separately governed by the Fish Audio Research License, which requires a separate written Fish Audio agreement for commercial use. GGML components retain the MIT license. See [NOTICE.md](NOTICE.md) and the complete texts in [LICENSES](LICENSES/THIRD_PARTY_NOTICES.md).

This repository does not grant rights to model weights, reference audio, or commercial use. Review all component terms before use or redistribution.
