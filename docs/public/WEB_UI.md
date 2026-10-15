# Fish Audio Drama 3 · Windows Studio

Install the Windows runtime 0.2.0 source and matching native bridge, its Python dependencies and the matching native libraries using [INSTALL.md](INSTALL.md). Download the model and codec manifests as documented there. The app uses the installed runtime and has no additional web framework dependency.

Start the app from your runtime environment:

```powershell
python -m fish_drama3.webui
# Installed console entry point:
fish-drama3-web
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Keep the terminal running. The server listens locally on your computer.

## Load the model

Select the model bundle directory containing `manifest.json` and the codec GGUF file. Typical download paths are `models/drama3-windows/model` and `models/drama3-windows/codec/codec.gguf`. Choose the expert placement and CPU thread settings, then load the model. Loading verifies the bundle before inference.

The studio defaults to **automatic expert placement** with **3% of total VRAM kept free**. It first loads the dense model and reserves request caches, then places as many complete expert banks as fit using measured free memory, with an additional 256 MiB workspace allowance. This targets memory occupancy, not GPU compute utilization; selecting more banks does not guarantee a utilization percentage or speedup. The **Free VRAM reserve** control accepts 1–25% and defaults to 3%. Whole-bank granularity can leave more free than the chosen reserve.

Choose **Manual** to request 0–40 GPU expert layers; the manual field starts at 16. This is a requested count, not a claim that 16 layers fit every machine or request. Both modes enforce the chosen reserve. Continuous monitoring stops the worker below the chosen VRAM reserve or 8 GiB free host RAM.

Automatic placement and the configurable native reserve require the **0.2.0 Python runtime and matching 0.2.0 native CUDA bridge**. Earlier DLLs do not export the required reserve-budget ABI. The model bundles themselves remain compatible with runtime 0.1.1 or newer.

The model stays loaded across requests. Unload it when finished to release model memory and VRAM. Expert controls select routed expert placement and do not change the model's Q4_K / Q8_0 weight format.

## Generate and listen

Enter your script or load **The Small Signal**, the complete original 592-word [podcast script](../../examples/samples/podcast.txt). The default **Single pass** mode submits the full script in one request without text splitting or speech edits. The [published one-pass example](https://huggingface.co/Eng-Ahmd/Drama-3-Windows-GGUF#podcast--the-small-signal) completed the full script in one request without retries or speech edits: 195.79 seconds, with 0.34% ASR WER. This is one checked sample; other runs can omit or repeat words, stop early or pause. Automated transcription does not replace listening.

The codec may decode the generated acoustic stream in bounded context windows to limit memory. Those decoder windows do not split the text into separate synthesis requests or edit speech.

For voice cloning, provide a local reference audio file and its exact transcript. The sample metadata identifies the same public **en0** reference used in the model-card demos:

```powershell
Invoke-WebRequest -Uri "https://github.com/user-attachments/files/33393249/en0.wav" -OutFile en0.wav
```

Reference transcript: `We asked over twenty different people, and they all said it was his.`

Use the generation controls to customize the supported runtime settings. Each setting has an accessible **?** help button with its meaning and valid range; activate it with the keyboard or pointer to reveal the explanation.

Supported controls include:

- **Seed** accepts integers 0–4,294,967,295. **Temperature** must be finite and greater than 0. **Top-p** must be greater than 0 and at most 1. **Top-k** accepts positive integers up to 2,147,483,647. Defaults are 123, 0.7, 0.7 and 30. These control sampling, not voice identity.
- **Maximum frames** sets the total acoustic frame budget across the complete script. The studio defaults to 6144 frames. Single pass supports up to 8192 frames within a 10,240-row prompt-plus-output context. Smart segments support up to **65,536 total frames across the recording**, not per segment; each passage must fit its own prompt/context capacity. Reference audio also consumes prompt rows. One frame is approximately 46 ms of audio. Direct API requests that omit this setting retain a 512-frame fallback.
- **Generation mode** defaults to **Single pass**, which disables text chunking. Select **Chunked** to synthesize bounded text passages using audio-history conditioning.
- **Text chunk length** applies only to **Chunked** mode. Use 32–2000 characters to choose the passage bound. The 2000-character ceiling is an application safety bound, not a claim about the model's language capacity. Smart splitting prefers sentence and whitespace boundaries, preserves protected speaker/event tags, and never cuts an ordinary word internally. Unspaced CJK can split at valid character boundaries without separating combining marks. An indivisible word or tag that cannot fit produces a preflight error asking you to raise the limit or shorten it. A request is limited to 32 passages, and splitting preserves the original text exactly.
- **Allow partial audio** permits saving an incomplete result when the budget is reached. Leave it disabled when you require the complete script.
- **Output format and MP3 bitrate** control the saved export. MP3 offers 128, 192, 256 or 320 kbps; 192 is the default. WAV remains available as the original output.
- **CPU threads** accepts 1–256 workers, initially 8; more threads do not always improve speed.

A fixed seed helps compare runs, but output also depends on the prompt, reference, settings and runtime. Long scripts can need shorter passages and selected retries. Listen to the result before sharing it.

## Live progress and system vitals

Before generation, a budget estimate uses approximate speech units and an assumed pace of 160 words per minute; conservative advice allows 110 words per minute plus 15%. Preflight shows the actual smart-passage count and flags scripts that may exhaust the total frame budget. Estimated progress percentages compare generated work with this estimated duration and can diverge from the final recording. They are predictions, not measured text coverage. The separate frame-budget usage is an exact ratio of frames generated to the selected maximum. EOS means the model stopped normally; it does not guarantee every requested word was spoken.

The app reports the current operation: **verifying**, **loading**, **reference encoding**, **prefill**, **generating**, **decoding**, **exporting** or **complete**. Elapsed time, acoustic frames, frames per second and equivalent generated audio seconds describe work already performed. A frame-budget indicator shows consumed capacity; it is not a completion percentage because the model decides when to end speech. Generation can finish before the budget or reach it without speaking the full script.

System vitals refresh approximately every two seconds; freshness labels indicate the age of the most recent sample. Rolling histories show actual sampled readings rather than generated activity. Decorative SVG elements are interface artwork, not audio waveform measurements.

Available readings include: CPU utilization and logical thread count; RAM utilization and used/total memory; GPU utilization, memory, temperature and power. Unavailable sensor values remain unavailable rather than becoming zero. GPU compute utilization and VRAM occupancy measure different things.

GPU display statistics come from NVIDIA/NVML sampling. The worker's memory guard uses CUDA free-memory readings to enforce the chosen VRAM reserve; those values can differ from the displayed NVML memory figures because the APIs measure and sample differently. The host guard preserves at least 8 GiB free RAM. These monitors protect memory reserves and do not promise a particular GPU utilization or generation speed.

## Understanding generation speed

The current hybrid runtime executes CPU expert layers serially between GPU backbone operations. It also transfers activations and routing data between the PyTorch and native expert paths. GPU compute can wait during CPU work and synchronization, while the final native codec decode runs on CPU. GPU utilization below 100% can therefore reflect these pipeline bottlenecks; the percentage alone does not establish a driver fault. VRAM placement, CPU thread count and passage length can affect timing differently, so compare a short fixed script with the same settings before drawing speed conclusions.

## Save audio and settings

Completed output is available for playback and download as mono 44.1 kHz WAV. Save the accompanying settings JSON to retain the request and runtime settings for a later run.

**Project settings** also works before generation: **Export settings JSON** downloads the current project without needing a completed recording. **Save in browser** stores the project locally in that browser; **Import settings** restores a compatible `fish-drama3-studio-v1` JSON file. Project settings include the script and controls, but never bundle reference audio. Upload the reference clip again after importing settings when voice conditioning is required. If a model is already loaded, importing settings restores generation settings only. Unload the current model before restoring a different model or its load-time configuration.

When a new interface revision becomes available, the app can reload its UI while preserving the current project. Transient reference identifiers remain usable only within the same running server session; restarting the server requires uploading the reference clip again. A saved or exported project does not recreate that audio.

MP3 export requires `ffmpeg` on `PATH`. Confirm availability before starting the app:

```powershell
ffmpeg -version
```

The WAV is the original PCM output. MP3 is a lossy export; its waveform will differ from WAV. If FFmpeg is unavailable, use WAV output or install FFmpeg and restart the app.

The app runs inference locally and serves the browser interface from the same process. Stop the terminal process after unloading when you are finished.
