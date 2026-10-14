# Fish Audio Drama 3 for Windows

This repository contains the **Windows release 0.1.0rc1** for running Fish Audio Drama 3 on Windows. It combines a PyTorch model path with native GGML expert execution and a native audio codec. It is a hybrid Windows runtime, not a standalone C++ model engine.

The project includes a direct converter for the pinned Drama 3 checkpoint, an offline synthesis CLI, native CPU and CUDA components, and guarded evaluation tools. Model weights and compiled binaries are separate from this source tree. A Hugging Face repository is planned for the converted bundle and codec; until it is published, use the local conversion steps in the installation guide. Obtain model files under their applicable license terms and build native components locally. The planned [GitHub source repository](https://github.com/Eng-Ahmd/drama-3-windows) will host the source code.

## Start here

- [Installation](docs/public/INSTALL.md) covers the Windows toolchain, Python environment, pinned sources, conversion, and a first synthesis command.
- [User guide](docs/public/USER_GUIDE.md) describes conversion, voice references, output behavior, and guarded runs.
- [Model card](docs/public/MODEL_CARD.md) documents the checkpoint, quantized bundle, measured results, and limitations.
- [Quality and qualification](docs/public/QUALITY.md) records the current evidence and unresolved release gates.
- [Attribution and licensing](NOTICE.md) explains the separate model, reference-code, and codec-donor terms.

## Runtime at a glance

The inference path keeps the backbone and audio tower in PyTorch, runs routed experts through native GGML CPU/CUDA libraries, and uses the native codec to encode reference audio and decode generated acoustic codes. The converter emits a custom `fish_drama3` GGUF architecture. It requires this matching runtime; generic llama.cpp does not provide Drama 3 support.

The conversion recipe stores expert weights as Q4_K, the backbone as Q8_0, sensitive vectors as F32, and the audio tower in BF16 and the learned codec weights in F32, preserving the codec RoPE tables in BF16. A complete model bundle is about 21.67 GiB and its codec is about 1.46 GiB. Conversion is direct from the pinned FP8/BF16 source without a full FP32 model intermediate.

## Current evidence and limits

The native codec reproduced all 850 reference acoustic codes in the checked example. Decoding 87 frames produced the same 178,176 samples as the reference decode, with relative RMSE 1.74×10⁻⁶. One matched English voice-reference example had 0.0 Whisper WER for both the original and generated audio. These are component and single-example results, not aggregate speech-quality or speaker-similarity claims.

The preview also has an unresolved failure: the frozen numbers prompt at seed 124 reached its 384-frame limit without EOS, and ASR recognized only the first sentence. A diagnostic reconstructed BF16 expert arithmetic from the FP8 source weights and reached EOS at frame 128, while the native Q4 run hit the 384-frame limit without EOS. A controlled Q4-in-PyTorch-BF16 comparison is still running, so the cause is not yet isolated. Production qualification, real-time performance, and clean-machine installation have not been established. See the [quality report](docs/public/QUALITY.md) for the full record.

## Licensing

Built with Fish Audio. Drama 3 materials retain the Fish Audio Community License v1.0. The s2.cpp-derived codec and its derivative work are separately governed by the Fish Audio Research License, which requires a separate written Fish Audio agreement for commercial use. GGML components retain the MIT license. See [NOTICE.md](NOTICE.md) and the complete texts in [LICENSES](LICENSES/THIRD_PARTY_NOTICES.md).

This repository does not grant rights to model weights, reference audio, or commercial use. Review all component terms before use or redistribution.
