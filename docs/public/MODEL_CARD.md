---
license: other
license_name: Fish Audio Community License v1.0
license_link: https://huggingface.co/fishaudio/drama-3/blob/dd64a85cd82e9521f7202610ede97b367fc83946/LICENSE
base_model: fishaudio/drama-3
base_model_relation: quantized
pipeline_tag: text-to-speech
tags:
  - fish-audio
  - drama-3
  - text-to-speech
  - windows
---

# Fish Audio Drama 3 Windows runtime — 0.1.0

This card documents the Windows runtime `0.1.0` and its local conversion recipe for the pinned Fish Audio Drama 3 checkpoint. A Hugging Face repository is planned for converted model and codec bundles; until it is published, create them with the source conversion workflow. The original checkpoint is available from [Fish Audio's pinned source](https://huggingface.co/fishaudio/drama-3/tree/dd64a85cd82e9521f7202610ede97b367fc83946). Source code: [GitHub repository](https://github.com/Eng-Ahmd/drama-3-windows).

## Model and runtime

The converted model bundle is approximately 21.67 GiB, with a separate V3 codec bundle of approximately 1.46 GiB. These bundle sizes are provided to plan local conversion and storage; the weights are not part of the source-code repository.

The source checkpoint combines FP8 and BF16 tensors. The converter streams it into a custom `fish_drama3` GGUF model bundle using this recipe:

- Expert matrices: Q4_K across the 256-expert banks; the model routes to eight experts per token.
- Backbone matrices: Q8_0.
- Sensitive vectors: F32.
- Audio tower matrices: BF16. Codec learned weights: F32, with original BF16 RoPE tables.
- GGUF custom architecture: `fish_drama3`, requiring this repository's matching runtime rather than generic llama.cpp support.

The converted model contains 925 tensors and 23,270,209,984 bytes (about 21.67 GiB). The separate V3 codec bundle contains 458 tensors and 1,568,779,744 bytes (about 1.46 GiB). Those sizes describe the current local conversion and are not download artifacts attached to this card.

### Runtime compatibility

Inference is hybrid: PyTorch executes the backbone and audio tower, native GGML CPU/CUDA libraries execute routed experts, and the native codec handles reference encoding and audio decoding. This is not a fully native C++ model graph. The custom `fish_drama3` GGUF architecture requires the matching project runtime; generic llama.cpp does not support it. The measured target was Windows 11 with a 16 GiB RTX 5080 Laptop GPU, Visual Studio 2022, CUDA Toolkit 13.3, and PyTorch 2.13.0+cu132. A clean-machine installation has not been established.

## Intended use

The Windows runtime supports local Drama 3 inference, development, and evaluation. Synthesis is offline and uses this matching runtime; generic GGUF engines do not implement its model graph. Use voice-reference audio only when you have the necessary rights and the speaker's consent.

## Evaluation

The native codec reproduced all 850 reference acoustic codes in the checked example. Decoding the corresponding 87 frames returned the same 178,176 samples as the reference decode, with relative RMSE 1.74×10⁻⁶. This is codec component evidence; it does not establish perceptual parity across general audio.

In one matched English voice-reference example, the reference and generated audio both scored 0.0 WER with the same Faster Whisper base.en evaluator. The original clip is 9.195 seconds long and the generated clip is 7.895 seconds long; these are audio durations, not synthesis latency. Listening found the generated voice close in that example, with a difference in the sigh. There is no objective speaker-similarity baseline.

A separate guarded CLI run reached EOS at frame 169, producing 7.848 seconds of audio in 39 seconds. ASR recognized the requested words but inserted “Haha” (WER 0.0714). This is a single-run observation, not a speed or aggregate quality result.

One 12-GPU-expert-layer run on the 16 GiB target ended with 2.67 GiB of GPU memory free. Native allocations are not fully reflected by PyTorch allocator counters, and this observation does not establish a maximum safe profile.

Runtime 0.1.0 generates long requests in bounded text segments with audio-history conditioning. The checked three-sentence passage completed all text in 14.12 seconds of audio, with a longest measured near-silent interval of 20 ms. The short reference-clone regression produced the same PCM WAV bytes as the previous CLI path.

Four short English checks had zero ASR WER. Chinese plain speech had zero CER with a pinned multilingual evaluator; a cross-language clone had 13.33% CER from two homophonous character substitutions. These are individual intelligibility checks rather than aggregate quality or speaker-similarity scores.

The recorded numbers prompt can stall at seed 124; the default seed 123 completed both sentences. Finite partial audio is marked incomplete and requires explicit opt-in. The [quality report](https://github.com/Eng-Ahmd/drama-3-windows/blob/main/docs/public/QUALITY.md) retains the failing runs, controlled source-history comparisons and remaining long-session, cancellation, acoustic-holdout and clean-machine checks.

## Data and provenance

The pinned upstream sources are:

- [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3/tree/dd64a85cd82e9521f7202610ede97b367fc83946), revision `dd64a85cd82e9521f7202610ede97b367fc83946`.
- [Fish Speech](https://github.com/fishaudio/fish-speech/tree/1ff3c3ed2f80d3106a83909dd25b83129c3d816f), revision `1ff3c3ed2f80d3106a83909dd25b83129c3d816f`, used for selected reference model code.
- [s2.cpp](https://github.com/rodrigomatta/s2.cpp/tree/2c33261938da1a41d713768b1b391b4d368d7d2c), revision `2c33261938da1a41d713768b1b391b4d368d7d2c`.
- [GGML](https://github.com/ggml-org/ggml/tree/57ea0bc119d722d74594196cc5b494a34dd87be4), revision `57ea0bc119d722d74594196cc5b494a34dd87be4`.

The donor-derived native build compiles the s2.cpp codec and logging sources with local precision overlays and uses its headers. GGML supplies backend and quantization components. Published [Jmica Drama 3 GGUF](https://huggingface.co/Jmica/Drama-3-GGUF) and [audio.cpp](https://github.com/0xShug0/audio.cpp/tree/a4a4c07ff9af10c437c777eeabf31677eea62668) material was consulted as references; Jmica model artifacts and audio.cpp source files are not redistributed by this repository. The s2.cpp checkout is fetched at the pinned revision during bootstrap rather than bundled in the source archive.

The source pin manifest [source pin manifest](https://github.com/Eng-Ahmd/drama-3-windows/blob/main/data/source/drama3-dd64a85-sha256.json) records hashes for the checkpoint files used by conversion.

## Licensing

The original model and model-derived weights are governed by the Fish Audio Community License v1.0. This runtime also includes s2.cpp-derived codec work governed separately by the Fish Audio Research License. That donor license requires a separate written Fish Audio agreement for commercial use; the model's Community License does not replace or satisfy it. GGML components retain the MIT license. Read [NOTICE.md](https://github.com/Eng-Ahmd/drama-3-windows/blob/main/NOTICE.md) and all component license texts before use or redistribution.

## Citation and credit

Please retain the required Fish Audio attribution and identify the pinned sources when redistributing applicable components. This project is an independent integration and makes no endorsement claim on behalf of Fish Audio, s2.cpp contributors, or GGML contributors.

```text
Fish Audio Drama 3, checkpoint revision dd64a85cd82e9521f7202610ede97b367fc83946.
Fish Speech reference, revision 1ff3c3ed2f80d3106a83909dd25b83129c3d816f.
s2.cpp codec donor, revision 2c33261938da1a41d713768b1b391b4d368d7d2c.
GGML, revision 57ea0bc119d722d74594196cc5b494a34dd87be4.
```
