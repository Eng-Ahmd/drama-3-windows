---
license: other
license_name: fish-audio-community-license-v1.0
license_link: https://huggingface.co/fishaudio/drama-3/blob/dd64a85cd82e9521f7202610ede97b367fc83946/LICENSE
base_model: fishaudio/drama-3
base_model_relation: quantized
pipeline_tag: text-to-speech
language:
  - en
  - zh
tags:
  - fish-audio
  - drama-3
  - gguf
  - windows
  - voice-cloning
  - multi-speaker
  - emotional-speech-generation
  - moe
---

![Drama 3 for Windows](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/assets/drama3-windows.svg)

# Drama 3 · Windows GGUF

**Expressive speech, voice cloning and contextual editing on your own hardware.**

A Q4_K / Q8_0 release of [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3), paired with a Windows inference engine that places routed experts across system RAM and CUDA. Built with Fish Audio.

[**Download the model**](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/tree/main/model) · [**Windows runtime**](https://github.com/Eng-Ahmd/drama-3-windows/releases/latest) · [**Installation**](https://github.com/Eng-Ahmd/drama-3-windows/blob/main/docs/public/INSTALL.md)

- **35.9B MoE** — 256 experts per layer, eight selected per token.
- **Expressive voice cloning** — reference-conditioned speech with emotion and event tags.
- **Audio-history conditioning** — contextual edits and continuity between long-text segments.
- **Local CPU/CUDA execution** — configurable expert placement for limited GPU memory.
- **44.1 kHz audio** — 21.67 GiB model and 1.46 GiB codec.

## Listen: original vs Windows

The three demos from the [original model card](https://huggingface.co/fishaudio/drama-3#example-usage), reproduced with the same voice reference, requested text and prior-take context. Original recordings are provided by Fish Audio; the Windows recordings below were generated with this quantized release.

### Shared voice reference

The original **en0** reference clip used by all three examples:

<audio controls preload="metadata" style="width:100%;max-width:100%;"><source src="https://github.com/user-attachments/files/33393249/en0.wav" type="audio/wav"></audio>

### 1 · Expressive point tags

A single voice moves from a sigh to spoken dialogue and laughter, using `[sighs]` and `[laughs]` in the prompt.

<table style="display:table;width:100%;table-layout:fixed;">
<tr><th width="50%">Fish Audio original</th><th width="50%">Windows · Q4_K / Q8_0</th></tr>
<tr>
<td><audio controls preload="metadata" style="width:100%;max-width:100%;"><source src="https://github.com/user-attachments/files/33393205/A1_point_tags.wav" type="audio/wav"></audio></td>
<td><audio controls preload="metadata" style="width:100%;max-width:100%;"><source src="https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/demos/point-tags.wav" type="audio/wav"></audio></td>
</tr>
</table>

### 2 · Edit with context

Change **Wednesday → Thursday** in a spoken report reminder. The earlier take is supplied as acoustic history, preserving the voice and context while revising the line.

<table style="display:table;width:100%;table-layout:fixed;">
<tr><th width="50%">Fish Audio original</th><th width="50%">Windows · Q4_K / Q8_0</th></tr>
<tr>
<td><audio controls preload="metadata" style="width:100%;max-width:100%;"><source src="https://github.com/user-attachments/files/33393156/K1_edit_word_with_context.wav" type="audio/wav"></audio></td>
<td><audio controls preload="metadata" style="width:100%;max-width:100%;"><source src="https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/demos/contextual-edit.wav" type="audio/wav"></audio></td>
</tr>
</table>

### 3 · Chained edit

Change the deadline from **noon → five**, with both earlier takes retained as acoustic history.

<table style="display:table;width:100%;table-layout:fixed;">
<tr><th width="50%">Fish Audio original</th><th width="50%">Windows · Q4_K / Q8_0</th></tr>
<tr>
<td><audio controls preload="metadata" style="width:100%;max-width:100%;"><source src="https://github.com/user-attachments/files/33393164/K5_second_edit_chained.wav" type="audio/wav"></audio></td>
<td><audio controls preload="metadata" style="width:100%;max-width:100%;"><source src="https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/demos/chained-edit.wav" type="audio/wav"></audio></td>
</tr>
</table>

<details>
<summary>Demo settings and reproduction</summary>

Local recordings use seed 123, temperature 0.7, top-p 0.7, top-k 30 and twelve GPU expert layers. The source demos do not publish their random seeds or sampling settings. Generation is stochastic; timing, expression and nonverbal events can differ.

The contextual examples include generated acoustic tokens from the earlier takes, not just their transcripts. After installing the runtime, download the shared reference and generate the demos:

```powershell
Invoke-WebRequest -Uri "https://github.com/user-attachments/files/33393249/en0.wav" -OutFile en0.wav
python -m fish_drama3.watchdog --output runs/demo-watchdog -- python -m examples.generate_model_card_demos --bundle models/drama3-windows/model --codec models/drama3-windows/codec/codec.gguf --reference en0.wav --output runs/demos
```

 Run [the demo generator](https://github.com/Eng-Ahmd/drama-3-windows/blob/main/examples/generate_model_card_demos.py) through the runtime watchdog to reproduce the sequence. [Demo settings and measurements](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/blob/main/demos/generation.json).

</details>

## Podcast · The Small Signal

**A complete 592-word podcast in one generation request.** This 3 minute 16 second recording uses the full script at once: no text chunks, retries, selected takes or speech edits.

<audio controls preload="metadata" style="width:100%;max-width:100%;"><source src="https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/demos/podcast-small-signal.mp3" type="audio/mpeg"></audio>

[WAV](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/demos/podcast-small-signal.wav) · [MP3](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/demos/podcast-small-signal.mp3) · [Full script](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/blob/main/demos/podcast-small-signal.txt) · [Importable settings and provenance](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/resolve/main/demos/podcast-small-signal-settings.json) · [Quality checks](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/blob/main/demos/podcast-small-signal-quality.json)

Generated through the 0.2.0 browser studio with the **en0** reference above, seed 123, temperature 0.7, top-p 0.7, top-k 30 and a 6144-frame budget. Automatic placement selected **15 GPU expert layers** with a 3% VRAM reserve; the cache context was 10,240 rows. The model ended normally at **4216 frames / 195.79 seconds**. The codec decoded the continuous acoustic stream in bounded context windows (1792 frames plus 256 context), without editing speech.

On Windows 11 with an RTX 5080 Laptop (16 GiB VRAM), 128 GiB RAM and 8 CPU threads, acoustic generation including prefill took **520.00 seconds**; CPU codec decoding added **223.34 seconds**, for **743.34 seconds synthesis**. These exclude model loading (**27.26 seconds**) and MP3 export. This is one recorded run, not an aggregate speed benchmark.

The untouched WAV has **0.34% ASR WER** (2 edits / 592 reference words), no full-scale clipped samples, peak magnitude 0.54089 and a longest measured near-silent interval of 0.06 seconds. Automated transcription is approximate and does not replace listening or measure speaker similarity. This single example establishes that this full script completed in one request; other text, seeds and environments can still omit or repeat words, stop early or pause.

To try it, load **The Small Signal** in the studio, upload the shared **en0** reference and import the linked settings. Public settings use standard download paths; adjust them for your installation. Automatic expert placement and seeded output can differ with available memory, hardware and runtime.

## Model & runtime

| Component | Configuration |
|---|---|
| Speech model | 35.9B MoE, hybrid full / linear attention |
| Routed experts | Q4_K · 8 of 256 selected per token |
| Backbone | Q8_0; sensitive vectors in F32 |
| Audio tower | BF16 · 10 acoustic codebooks |
| Codec | Native reference encoder / decoder; F32 learned weights |
| Output | Mono 44.1 kHz PCM16 WAV |
| Download size | Model 21.67 GiB · codec 1.46 GiB |
| Engine | Windows runtime 0.1.1 or newer |

**Execution path:** PyTorch runs the backbone and audio tower; GGML CPU/CUDA runs routed experts; the native codec encodes the voice reference and decodes the generated audio. The `fish_drama3` GGUF files require this matching engine.

### Measured generation

Windows 11 · RTX 5080 Laptop, 16 GiB VRAM · 128 GiB RAM · 12 GPU expert layers · 8 CPU threads. These are warm-model, single-request measurements for the recordings above, with CPU codec processing.

| Example | Audio length | Generation | Gen + decode | RTF |
|---|---:|---:|---:|---:|
| Expressive tags | 7.85 s | 40.30 s | 47.87 s | 6.10 |
| Contextual edit | 4.27 s | 13.38 s | 17.28 s | 4.04 |
| Chained edit | 4.37 s | 14.35 s | 18.32 s | 4.20 |

RTF is synthesis time divided by audio length. Timings include model prefill, generation and native codec decode; they exclude one-time model loading, reference encoding and file writing. Model loading took 23.04 seconds for this run.


Generation is offline. The external CLI watchdog reserves at least 1 GiB GPU memory and 8 GiB host RAM; expert placement is configurable. Prebuilt CUDA libraries target SM120. The documented environment uses Python 3.13, PyTorch 2.13.0+cu132, CUDA Toolkit 13.3 and the Visual C++ x64 runtime.

## Browser generation app

The [local web app](https://github.com/Eng-Ahmd/drama-3-windows/blob/main/docs/public/WEB_UI.md) lets you load and unload the model, load the full podcast script, customize supported generation and expert-placement settings, play results and save WAV or MP3 with a settings JSON. Single pass is the default; chunked generation is optional. The studio starts with a 6144-frame budget and supports up to 8192 frames within a 10,240-row prompt-plus-output context limit. The studio requires the 0.2.0 Python runtime and matching native bridge. Automatic placement loads dense weights and request caches first, then fits complete expert banks against actual free memory with a 256 MiB workspace allowance and 3% total VRAM reserve. Manual mode supports 0–40 layers, initially 16; fit depends on hardware and request size. Continuous monitoring preserves 3% free VRAM and 8 GiB free host RAM. The percentage governs memory occupancy, not compute utilization. MP3 export requires FFmpeg. After installing the current runtime source, start it with `python -m fish_drama3.webui`, then open `http://127.0.0.1:7860`.

## Download & run

Install the [Windows runtime and Python environment](https://github.com/Eng-Ahmd/drama-3-windows/blob/main/docs/public/INSTALL.md), then download the model and codec with their manifests:

```powershell
python -m pip install "huggingface_hub>=1.0,<2"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Eng-Ahmd/Drama-3-Windows-GGUF', local_dir='models/drama3-windows', allow_patterns=['model/*', 'codec/*', 'LICENSE', 'LICENSES/*', 'NOTICE.md'])"
fish-drama3 synth --bundle models/drama3-windows/model --codec models/drama3-windows/codec/codec.gguf --text "Every voice has a story. Let's make yours heard." --output outputs/hello.wav --gpu-expert-layers 12
```

For voice cloning, add `--reference-audio` and its `--reference-text`. Long text uses bounded segments with audio-history conditioning. Complete output requires EOS; the CLI verifies file hashes and preserves existing results.

The root-level `Drama-3-35.9B-Q4_K-Q8_0.gguf` is an identical alias of `model/model.gguf`; download one copy of the model.

## Quality & validation

- **Reference codec:** all 850 checked acoustic codes matched; decode relative RMSE **1.74 × 10⁻⁶** against the pinned reference.
- **Intelligibility:** four short English checks had **0.0 ASR WER**; a Chinese plain-speech check had **0.0 CER**.
- **Runtime stability:** a guarded two-hour run completed **713 speech requests** and **143 validation recovery cycles**, with identical acoustic codes for the repeated seeded request.

These are targeted checks, not aggregate speech-quality or speaker-similarity scores. Seed-dependent incomplete generation and the evaluation method are documented in the [quality report](https://github.com/Eng-Ahmd/drama-3-windows/blob/main/docs/public/QUALITY.md).

## Credits & license

Model and reference implementation: [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) and [Fish Speech](https://github.com/fishaudio/fish-speech). Native codec integration derives from [s2.cpp](https://github.com/rodrigomatta/s2.cpp); backend and quantization components use [GGML](https://github.com/ggml-org/ggml). [audio.cpp](https://github.com/0xShug0/audio.cpp) and [Jmica's Drama 3 GGUF](https://huggingface.co/Jmica/Drama-3-GGUF) informed the porting survey. Pinned revisions and component attribution are in [NOTICE.md](https://github.com/Eng-Ahmd/drama-3-windows/blob/main/NOTICE.md).

Model-derived weights retain the **Fish Audio Community License v1.0**. The donor-derived codec is separately governed by the **Fish Audio Research License**, which requires a written Fish Audio agreement for commercial use. GGML components retain their MIT license. See [LICENSE](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/blob/main/LICENSE) and [component licenses](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF/tree/main/LICENSES).

Use voice references with the necessary rights and speaker consent. This is an independent Windows integration. **Built with Fish Audio.**
