# Attribution and licensing notice

Built with Fish Audio.

This repository integrates the pinned Fish Audio Drama 3 model and Fish Speech reference implementation with a Windows runtime derived in part from the s2.cpp codec and GGML. The components keep their separate licenses. This notice is a guide to those components; it does not replace the license texts in [`LICENSES/`](LICENSES/THIRD_PARTY_NOTICES.md) or grant rights to model weights.

## Fish Audio Drama 3 and Fish Speech

The Drama 3 checkpoint and the selected Fish Speech reference code are associated with the Fish Audio Community License v1.0. The model's pinned source revision is `dd64a85cd82e9521f7202610ede97b367fc83946`; the reference code revision is `1ff3c3ed2f80d3106a83909dd25b83129c3d816f`.

The required attribution notice for the Community-licensed model is:

> These Fish Audio Materials are licensed under the Fish Audio Community License, Version 1.0. Copyright © 39 AI, INC. All Rights Reserved.

The pinned checkpoint license is available at [the model repository](https://huggingface.co/fishaudio/drama-3/blob/dd64a85cd82e9521f7202610ede97b367fc83946/LICENSE); the full text is included as [`LICENSES/FISH_AUDIO_COMMUNITY_LICENSE.md`](LICENSES/FISH_AUDIO_COMMUNITY_LICENSE.md). Model weights are not included in this source export. Obtain them separately and follow their access and redistribution terms.

## s2.cpp codec donor

The native codec is derived from [rodrigomatta/s2.cpp](https://github.com/rodrigomatta/s2.cpp), pinned at `2c33261938da1a41d713768b1b391b4d368d7d2c`. The codec and logging sources are compiled from that pristine source revision, with local precision overlays; the donor checkout itself is not modified. Donor headers are used by the bridge.

The required Research License attribution is:

> This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.

The Fish Audio Research License permits research and non-commercial use under its terms. **Commercial use of the s2.cpp-derived codec or its derivative works requires a separate written license agreement from Fish Audio.** The Drama 3 Community License does not replace or satisfy that requirement. See [`LICENSES/FISH_AUDIO_RESEARCH_LICENSE.md`](LICENSES/FISH_AUDIO_RESEARCH_LICENSE.md).

## GGML

[GGML](https://github.com/ggml-org/ggml) is pinned at `57ea0bc119d722d74594196cc5b494a34dd87be4` and retains its MIT license. The build uses the pinned GGML library and quantization operations. Local BF16 cached-RoPE handling is applied by project build overlays; no donor checkout is edited. See [`LICENSES/GGML_MIT_LICENSE.txt`](LICENSES/GGML_MIT_LICENSE.txt).

## Other references and dependencies

The project consulted [Jmica's Drama 3 GGUF repository](https://huggingface.co/Jmica/Drama-3-GGUF) and [audio.cpp](https://github.com/0xShug0/audio.cpp/tree/a4a4c07ff9af10c437c777eeabf31677eea62668) at revision `a4a4c07ff9af10c437c777eeabf31677eea62668` as survey references. No Jmica model artifacts or audio.cpp source files are redistributed here. The pinned s2.cpp source is fetched separately by the bootstrap script rather than bundled in the source archive.

Python dependencies, CUDA/cuBLAS, and Microsoft runtime components retain their own licenses and distribution terms. The source export does not bundle those binaries. A consolidated inventory is in [`LICENSES/THIRD_PARTY_NOTICES.md`](LICENSES/THIRD_PARTY_NOTICES.md).

## Required-use reminders

Do not use identifiable voice-reference audio without the speaker's prior consent or another legal right. Do not use Fish Audio materials or their outputs to train, distill, or improve other foundational or speech-generation models, and do not remove any applied watermark or provenance signal. The complete component licenses govern; review them before use or redistribution. No Fish Audio, s2.cpp contributor, or GGML contributor endorsement is implied.
