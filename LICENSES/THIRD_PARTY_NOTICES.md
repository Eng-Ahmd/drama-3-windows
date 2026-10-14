# Third-party notices

Built with Fish Audio.

These Fish Audio Materials are licensed under the Fish Audio Community
License, Version 1.0. Copyright © 39 AI, INC. All Rights Reserved.

This model is licensed under the Fish Audio Research License,
Copyright © 39 AI, INC. All Rights Reserved.

The second notice applies to the Research-licensed s2.cpp codec donor; it does
not change the Community license accompanying the Drama 3 weights.

## Source components

- [Fish Speech](https://github.com/fishaudio/fish-speech), revision
  `1ff3c3ed2f80d3106a83909dd25b83129c3d816f`: Community License v1.0.
  The runtime imports the pinned `fish_omni` implementation.
- [s2.cpp](https://github.com/rodrigomatta/s2.cpp), revision
  `2c33261938da1a41d713768b1b391b4d368d7d2c`: Fish Audio Research License.
  Codec and logging sources are compiled with the local precision overlays.
- [GGML](https://github.com/ggml-org/ggml), revision
  `57ea0bc119d722d74594196cc5b494a34dd87be4`: MIT.
  The donor's exact submodule supplies quantization and backend operations.

## Runtime dependencies

PyTorch (BSD-style), NumPy (BSD-3-Clause), SciPy (BSD-3-Clause), safetensors
(Apache-2.0), tokenizers (Apache-2.0), gguf (MIT), SoundFile (BSD-3-Clause),
libsndfile (LGPL-2.1-or-later), and psutil (BSD-3-Clause) retain their package
licenses. Dependencies installed by the environment are not relicensed by
this repository. NVIDIA CUDA/cuBLAS and Microsoft runtime redistribution must
follow their vendor terms; the source export does not bundle those binaries.

Model weights are distributed separately. Their original license and required
notices accompany each model bundle. Runtime and model licensing must be
assessed together for the intended use.
