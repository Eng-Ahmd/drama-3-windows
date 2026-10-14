# Quality and qualification

This Windows runtime has component numerical evidence. Component tests and
finite speech output are useful evidence; they do not establish production
quality, perceptual fidelity, or real-time speech performance.

## Frozen preview

`data/qualification/cases.jsonl` contains twelve versioned preview cases:
English, Chinese, numbers, names, tags, canonical speakers, plain speech,
cloning from the official `en0` reference, a longer request, and three invalid
inputs. The runner replays a seeded short anchor after the long/negative
sequence and compares repeated acoustic-code hashes. This is a preview,
not the required production corpus of at least 100 fixed cases or the
separate 300-case acoustic holdout.

The reference clone codes were encoded by the original pinned Python codec.
The native encoder subsequently matched all 850 reference codes exactly.
Current generation uses the hybrid PyTorch/GGML runtime and the native codec;
it is not a standalone native model graph.

## Guarded execution

The watchdog holds a shared workspace process lock before launching a GPU job.
Another guarded launcher fails without starting its child until that lock is
released. The operating system releases the lock when its owner exits. Keep the
same default lock path for all jobs. The synthesis CLI shares this lock when run
directly and uses its parent watchdog's lease when wrapped. Other applications
and inference entry points are outside this guard.

Run the qualification process through the external watchdog. It records the
actual GPU-free-memory, host-free-memory and system-commit thresholds supplied
by the parent environment. Current ordinary profile: twelve GPU expert layers;
current headroom thresholds: 1 GiB GPU free, 8 GiB host RAM free, 90% system
commit. These are abort thresholds, not deliberate memory-stress targets.
The `STOP` file in the watchdog output directory is a manual stop mechanism.
Sensor failures abort the child process rather than continuing unchecked.
The guardian uses the more conservative free-memory reading from NVML and
the child's actual CUDA telemetry when both are available; WDDM can make
those readings differ. New runs retain a source snapshot matching their
manifest so interrupted-run evidence remains reproducible after edits.

```powershell
python -m fish_drama3.watchdog --output runs/preview-watchdog -- python -m fish_drama3.qualification --mode preview --output runs/preview
```

Each request records started/completed events with flushed JSONL checkpoints,
phase timings, finish reason, memory observations, and deterministic code
hashes. Restart the same command with `--resume` to rerun the interrupted
request. Source manifest, corpus, sampling, criteria and settings must match.
Actual model and codec payload SHA-256 values are verified once before each
model load. A completed failed run cannot be resumed into a passing result;
retain its evidence and use a new output directory after investigating it.
Resume begins a new continuous-soak segment; separate interrupted segments
are not combined to claim an uninterrupted two-hour soak. Only one waveform
and code file per passing case is retained; failed outputs are retained
separately.

## Criteria fixed before measurement

Every valid request must finish with EOS. A token-limit finish is a failure.
Nonfinite waveforms and invalid semantic/residual code indices are rejected.
Repeated identical seeded requests must produce identical code hashes.
After warmup, observed process private-byte and device-allocation drift must
remain at or below 256 MiB on the repeated fixed workload. The runner records
Torch allocations/reservations, actual CUDA free memory, process RSS/private
bytes and cache capacity; declared cache growth is recorded separately.
CUDA free memory includes unrelated processes, so a failing observation needs
investigation before assigning a cause. Missing required metrics are not passes.

```powershell
python -m fish_drama3.watchdog --output runs/soak-watchdog -- python -m fish_drama3.qualification --mode soak --soak-seconds 7200 --min-requests 200 --failure-cycles 50 --output runs/soak
```

The soak runner requires the requested continuous warm duration, completed
valid requests and negative-input-to-successful-short recovery cycles.
These boundary-error recovery cycles are **not cancellation tests**. The
current upstream generation API lacks cooperative cancellation; cancellation
recovery remains unavailable until a real supported path is implemented.

## Acoustic evaluation

An optional local Faster Whisper evaluator records model-file hashes,
evaluator version, exact transcripts and decoded hypotheses. English WER
uses frozen NFKC/lowercase Unicode word-token normalization and full edit
distance; numbers are not silently rewritten into equivalent words.
The same evaluator and transcript must be applied to reference and generated
audio. ASR is an acoustic indicator, not proof of numerical fidelity, speaker
similarity, emotion fidelity, or lack of artifacts. Chinese CER, controlled
listening and speaker-similarity evaluation remain separate requirements.

The first matched English example produced zero WER for both official and
generated audio using Faster Whisper 1.2.1, CTranslate2 4.8.3 and the frozen
`Systran/faster-whisper-base.en` revision
`3d3d5dee26484f91867d81cb899cfcf72b96be6c`. This is one example, not an
aggregate quality result. The original seed-124 numbers case failed: it reached its 384-frame limit without EOS, and ASR recognized only the first sentence. A finite waveform does not make the request complete. A separate guarded CLI run reached EOS at frame 169 and produced 7.848 seconds of audio in 39 seconds. ASR recognized the requested words but inserted “Haha” (WER 0.0714). This single-run observation is not an aggregate quality or speed result. Completed controlled source-history comparisons are described below.

## Evaluation scope and known behavior

A separate run of the twelve functional cases used runtime 0.1.0 defaults: seed 123 for valid inputs, 200-character chunking and twelve GPU expert layers. All nine valid inputs reached EOS for every segment, all three invalid inputs were rejected, and the short anchor replay produced identical acoustic codes. Measured GPU, Torch and process memory drift at the final warmed replay was zero bytes at the same cache capacity. This is a functional completion check; it does not measure aggregate speech quality or replace the original mixed-seed corpus, whose seed-124 failure remains recorded.

Runtime 0.1.0 adds bounded paragraph generation with audio-history conditioning.
The previously stalled three-sentence passage completed at seed 123 in two
segments, both reaching EOS, for 304 frames and 14.12 seconds of audio. Frozen
ASR recognized all sentences; WER was 2.44% from `twenty` versus `20`. The longest
20 ms RMS interval below 1e-4 full scale was 20 ms, compared with the unsegmented
seed-123 run's 26.86-second silent tail and missing final sentence. This is one
controlled regression, not a guarantee for every paragraph. The short reference
clone regression produced a byte-identical PCM WAV to its prior CLI result.

Chinese evaluation used the separate multilingual `Systran/faster-whisper-base`
checkpoint at revision `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`, CPU int8,
beam size five and temperature zero. The frozen CER normalizer removes only
punctuation and whitespace after NFKC. Chinese plain speech had zero CER; the
cross-language clone had 13.33% CER from two homophonous character substitutions
(`克隆` versus `课龙`). This may reflect ASR ambiguity and does not establish
speaker similarity or overall Chinese quality.

A supplemental guarded run completed names, expressive tags, canonical speaker
tags, English and Chinese voice references, a longer passage, three invalid
inputs, and an identical seeded replay after those requests. All valid requests
reached EOS and the repeated request produced identical acoustic codes. This
supplement excludes the previously recorded numbers failure and does not count
as a pass of the full corpus.

Frozen English ASR returned zero WER for the names, tags, speakers and English
clone examples. The longer passage contained all three sentences; its 2.44% WER
was one numeric transcription difference (`twenty` versus `20`). Waveform analysis
also found an 8.14-second near-silent interval in that 21.69-second clip, using
20 ms windows below RMS 1e-4 full scale. Unsegmented generation can still exhibit this issue; the bounded-generation
regression is listed above. The English-only evaluator was not used for Chinese.

A separate teacher-forced comparison used the same 128-frame successful source
history with reconstructed source BF16 experts, native CPU Q4 experts, and Q4
weights evaluated through the pinned BF16 Torch expert arithmetic. Both Q4 paths
matched the source's top-ranked semantic token at the four measured endpoints.
At the final endpoint EOS ranked first in all three paths, with probabilities
0.3258, 0.3555 and 0.3361 respectively. Final-layer residual relative RMS drift
was 1.80% for native Q4 and 1.88% for Torch Q4 against that source control.
This comparison uses chunked recurrent execution and a shared Q8 backbone; it
does not establish original FP8 equivalence or waveform quality. It does not
identify an intrinsic EOS-kernel defect; generated-history and sampling
divergence remain under investigation.

The previously failing numbers prompt was also run with the CLI's default seed
123. It reached EOS at frame 179, produced 8.31 seconds of audio, and the frozen
ASR evaluator recognized both complete sentences. Literal WER was 47.06% because
the evaluator wrote `9.30`, `25` and `1234` instead of their spelled-out forms;
the frozen normalizer has not been altered to hide those substitutions. The
seed-124 failure remains recorded. If a request reaches its frame limit, use
another seed and verify the complete result rather than accepting partial audio.

Full-model FP8 reference execution currently requires a suitable 48 GB runner.
That source-baseline gate is unavailable locally and is explicitly recorded.
Production qualification also requires the larger frozen corpus, acoustic
holdout, cancellation/failure recovery, long streaming tests, clean-machine
packaging and independent review. Successful preview or soak completion never
automatically marks the model production-qualified.
