# Runtime performance

Runtime 0.2.1 accelerates the GPU audio tower with CUDA graphs while keeping CPU expert routing outside captured graphs. One CUDA stream is shared across cache buckets. Graphs are prepared before automatic expert placement, preserving the selected VRAM reserve and workspace allowance. Unsupported graph capture falls back to eager execution.

The expert bridge reuses single-token CPU graphs and workspaces. CUDA prefill can pass activation tensors directly to the native bridge for 128-token calls; single-token decode keeps the measured faster host path. Runtime settings and expert timing counters accompany saved recordings.

## Matched local measurements

Test machine: Windows 11, Intel Core Ultra 9 275HX, NVIDIA GeForce RTX 5080 Laptop GPU (16 GiB), PyTorch 2.13.0+cu132. Tests used the same weights, script, seed, sampling, eight CPU threads and 15 GPU expert layers. Timings below exclude cold model loading and prefill warmup.

With the codec loaded first, the production OpenMP backend and active waiting, warm acoustic decode measured 8.92–8.98 frames/s without tower graphs and 11.35–11.49 frames/s with graphs: approximately 28% higher throughput. Warm generation for the capped 64-frame fixture took 7.48–7.56 seconds before and 5.93–6.03 seconds after. This is an acoustic-generation benchmark, not a complete podcast or a promise of end-to-end speedup.

Graph buffers used about 58 MiB after stream reuse, down from 282 MiB in the initial prototype. All seven cache buckets and 14 boundary positions through row 10,239 passed exact sampled-code and RNG checks. Matched waveform output was identical across the tested CPU policies.

For the same 64-frame fixture, combined warm generation and codec time favored OpenMP with active waiting: 9.36 seconds, versus 10.57 seconds with passive waiting and 10.05 seconds without OpenMP. The studio therefore defaults to active waiting and eight threads, respecting an existing `OMP_WAIT_POLICY` override. Active waiting can consume CPU while workers wait; CPU utilization alone does not measure useful throughput.

An experimental fused CPU codec attention path passed numerical checks but decoded a real 2,304-frame stream in 106.57 seconds versus 94.13 seconds for the existing path. It remains disabled. Codec windows remain unchanged; smaller windows were slower and changed the waveform in a separate study.

## Reproduce

The scripts use the runtime job lease. Unload the studio model before benchmarking. Separate CPU and CUDA library directories let experiments run without replacing installed libraries. Use fresh processes when comparing backends because Windows reuses loaded DLL filenames.

- `scripts/benchmark-tower.py`: matched eager/graph generation, cache boundaries, optional codec-first loading and waveform timing.
- `scripts/benchmark-experts.py`: CPU/CUDA call parity and timing; `--trace` rotates 25 CPU layers with fresh routes.
- `scripts/benchmark-codec.py`: waveform and shared-input encoder-code comparisons, including unchanged-window long streams.

Run each script with `--help` for its parameters. GPU regression tests are opt-in with `FISH_DRAMA3_TEST_CUDA_TOWER=1` or `FISH_DRAMA3_TEST_CUDA_EXPERTS=1`.

To compare eager tower execution, set `FISH_DRAMA3_TOWER_GRAPH=0` before starting the studio. GPU utilization remains workload-dependent: CPU expert execution, synchronization and CPU waveform decoding still limit the hybrid pipeline.
