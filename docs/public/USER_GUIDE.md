# User guide

Fish Drama 3 for Windows is an offline synthesis runtime. It combines PyTorch execution for the backbone and audio tower with native GGML expert kernels and a native codec. The current build is not a generic llama.cpp model and does not provide a standalone C++ execution path.

## Use the browser app

After completing installation, run `python -m fish_drama3.webui` (or `fish-drama3-web`) and open [http://127.0.0.1:7860](http://127.0.0.1:7860). The app loads the model once, exposes supported expert and sampling settings, and saves WAV or MP3 with a settings JSON. MP3 export needs FFmpeg on `PATH`. See the [browser app guide](WEB_UI.md) for model paths, sample loading, reference audio and exports.

The podcast example loads the full script. The studio defaults to single-pass generation with 6144 audio frames and supports up to 8192, within a 10,240-row prompt-plus-output limit. Chunked generation is optional. A verified one-pass podcast example is linked in the browser app guide. This is one checked script; a completed request alone does not prove that every word was spoken. See the browser app guide for memory monitoring and output limits. These studio limits are separate from the CLI limits below.

The studio defaults to **automatic expert placement** with **3% of total VRAM kept free**. It first loads the dense model and reserves request caches, then places as many complete expert banks as fit using measured free memory, with an additional 256 MiB workspace allowance. This targets memory occupancy, not GPU compute utilization; selecting more banks does not guarantee a utilization percentage or speedup. Whole-bank granularity can leave more than 3% free.

Choose **Manual** to request 0–40 GPU expert layers; the manual field starts at 16. This is a requested count, not a claim that 16 layers fit every machine or request. Both modes enforce the chosen reserve. Continuous monitoring stops the worker below the 3% VRAM reserve or 8 GiB free host RAM.

Automatic placement and the configurable native reserve require the **0.2.0 Python runtime and matching 0.2.0 native CUDA bridge**. Earlier DLLs do not export the required reserve-budget ABI. The model bundles themselves remain compatible with runtime 0.1.1 or newer.

## Generate speech

After [installation](INSTALL.md) and conversion, use a new output path for each run:

```powershell
fish-drama3 synth `
  --bundle models/drama3-q4k-q8 `
  --codec models/drama3-codec-adapter-v3/codec.gguf `
  --text "Good morning. The train leaves at half past nine." `
  --output outputs/morning.wav `
  --gpu-expert-layers 12
```

The command checks the SHA-256 values recorded in the model and codec manifests. It refuses to replace an existing WAV or JSON report. On success it writes a 44.1 kHz PCM16 WAV and a JSON sidecar with the model and codec identities, prompt, sampling settings, finish reason, timings, and audio summary.

Defaults are seed 123, temperature 0.7, top-p 0.7, top-k 30, 512 maximum frames, eight CPU threads, and twelve GPU expert layers. Change them explicitly when comparing runs and keep the JSON sidecars. These settings do not guarantee deterministic results across different hardware or software environments.

Long text is split at natural boundaries into segments of at most 200 characters.
The preceding segment's generated audio conditions the next segment through the
pinned model's history API. The final codec decode runs on the combined acoustic
codes. `--max-frames` is one shared budget for the entire request; all segments
must reach EOS for a complete output. The JSON sidecar records each segment.
Use `--chunk-length` to adjust this bound, or `0` to reproduce unsegmented runs.
At most 32 segments are accepted; speaker markers and expression tags stay whole.

## Voice references

For voice-reference conditioning, provide an audio file and its transcript together:

```powershell
fish-drama3 synth `
  --bundle models/drama3-q4k-q8 `
  --codec models/drama3-codec-adapter-v3/codec.gguf `
  --text "Your generated text goes here." `
  --reference-audio references/voice.wav `
  --reference-text "The words spoken in the reference recording." `
  --output outputs/with-reference.wav
```

The current CLI accepts audio up to 60 seconds, averages multichannel input to mono, and resamples it to 44.1 kHz with SciPy when needed. The run record includes the source file hash, source and target sample rates, and resampler name. Use only audio you have permission to use and the speaker's consent for cloning.

## Complete and partial outputs

The CLI treats EOS as the normal completion condition. If generation reaches `--max-frames` first, it fails and does not publish a WAV by default. Increase the frame limit (maximum 2048) and try again. `--allow-partial` explicitly permits a limit-terminated result; the JSON report marks it `complete: false`. A finite WAV is not evidence that the requested text was fully spoken.

The frozen preview includes a numbers prompt that currently reaches its 384-frame limit without EOS, and ASR recognizes only the first sentence. This is an unresolved failure. See the [quality report](QUALITY.md) before interpreting a result as successful speech coverage.

## Guarded runs

The watchdog can launch a child process while checking NVIDIA GPU free memory, Windows host memory, and system commit. Its current abort thresholds are 1 GiB free GPU memory, 8 GiB free RAM, and 90% system commit. These are safety cutoffs, not targets for stressing the machine. A sensor failure or threshold violation stops the child. An output directory containing a `STOP` file requests a manual stop.

```powershell
python -m fish_drama3.watchdog --output runs/watchdog -- fish-drama3 synth `
  --bundle models/drama3-q4k-q8 `
  --codec models/drama3-codec-adapter-v3/codec.gguf `
  --text "A short guarded test." `
  --output outputs/guarded.wav
```

The watchdog requires `nvidia-smi` and exactly one visible NVIDIA GPU. GPU memory used by native libraries is not fully represented by PyTorch's allocator counters; treat allocator statistics as partial measurements.

## Interpreting results

Component parity and a single ASR-matched example do not establish general voice quality, speaker similarity, expressive fidelity, artifact-free output, or real-time performance. Production qualification and clean-machine reproducibility have not been established. Read [Quality and qualification](QUALITY.md) and the [model card](MODEL_CARD.md) for measured evidence and remaining limitations.
