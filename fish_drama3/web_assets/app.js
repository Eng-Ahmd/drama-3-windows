'use strict';
const $ = (id) => document.getElementById(id);
const modelKeys = ['bundle', 'codec', 'placement_mode', 'headroom_percent', 'gpu_expert_layers', 'threads'];
const generationKeys = ['generation_mode', 'seed', 'temperature', 'top_p', 'top_k', 'max_frames', 'chunk_length', 'allow_partial', 'output_format', 'mp3_bitrate'];
const settingsKeys = [...modelKeys, ...generationKeys, 'text', 'reference_text'];
const numericKeys = new Set(['headroom_percent', 'gpu_expert_layers', 'threads', 'seed', 'temperature', 'top_p', 'top_k', 'max_frames', 'chunk_length', 'mp3_bitrate']);
const defaults = Object.fromEntries(settingsKeys.map((key) => [key, $(key).type === 'checkbox' ? $(key).checked : $(key).value]));
let state = {status: 'unloaded', mp3_available: false};
let examples = [];
let referenceId = null;
let pending = false;
let uploading = false;
let lastResult = '';
let connected = false;
let lastServerError = '';
let uiRevision = null;
let telemetryUpdatedAt = null;
let telemetryReceivedAt = null;
let transientRestore = null;
let generationPlan = null;
let planTimer = null;
let planVersion = 0;
let planning = false;
const telemetryHistory = {cpu: [], ram: [], gpu: [], vram: []};
const transientKey = 'fish-drama3.transient.v1';

function numeric(value) {
  return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)) ? Number(value) : null;
}

function rememberTelemetry() {
  const vitals = state.vitals;
  if (!vitals) return;
  const stamp = numeric(vitals.updated_at);
  if (stamp !== null && stamp === telemetryUpdatedAt) return;
  telemetryUpdatedAt = stamp;
  telemetryReceivedAt = Date.now();
  const total = numeric(state.memory?.gpu_total_bytes ?? vitals.gpu_total_bytes);
  const free = numeric(state.memory?.gpu_free_bytes ?? vitals.gpu_free_bytes);
  const values = {cpu: numeric(vitals.cpu_percent), ram: numeric(vitals.ram_percent), gpu: numeric(vitals.gpu_utilization), vram: total > 0 && free !== null ? (total - free) / total * 100 : total > 0 && numeric(vitals.gpu_used_bytes) !== null ? Number(vitals.gpu_used_bytes) / total * 100 : null};
  for (const [key, value] of Object.entries(values)) {
    telemetryHistory[key].push(value);
    if (telemetryHistory[key].length > 60) telemetryHistory[key].shift();
  }
}

function renderFreshness() {
  const stamp = telemetryUpdatedAt ? telemetryUpdatedAt * 1000 : telemetryReceivedAt;
  const age = stamp ? Math.max(0, Math.floor((Date.now() - stamp) / 1000)) : null;
  $('telemetry-freshness').textContent = !connected ? 'OFFLINE · reconnecting' : age === null ? 'Connecting sensors…' : `${age > 10 ? 'STALE' : 'LIVE'} · updated ${age}s ago`;
  $('telemetry-freshness').classList.toggle('fresh', connected && age !== null && age <= 10);
}

function renderSparklines() {
  for (const [key, samples] of Object.entries(telemetryHistory)) {
    let path = '';
    let move = true;
    samples.forEach((sample, index) => {
      if (sample === null) { move = true; return; }
      const x = (60 - samples.length + index) / 59 * 120;
      const y = 23 - Math.max(0, Math.min(100, sample)) / 100 * 21;
      path += `${move ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)} `;
      move = false;
    });
    $(`spark-${key}`).setAttribute('d', path);
  }
  renderFreshness();
}

const formatSeconds = (seconds) => {
  if (!Number.isFinite(seconds)) return '—';
  return seconds >= 60 ? `${Math.floor(seconds / 60)}m ${String(Math.floor(seconds % 60)).padStart(2, '0')}s` : `${seconds.toFixed(1)}s`;
};
const gib = (bytes) => (bytes / 2 ** 30).toFixed(1);

function renderVitals() {
  const vitals = connected ? state.vitals || {} : {};
  for (const [key, id] of [['cpu_percent', 'cpu'], ['ram_percent', 'ram'], ['gpu_utilization', 'gpu']]) {
    const value = numeric(vitals[key]);
    $(`vital-${id}`).textContent = value !== null ? `${value.toFixed(1)}%` : '—';
    $(`meter-${id}`).style.width = Number.isFinite(value) ? `${Math.max(0, Math.min(100, value))}%` : '0%';
  }
  $('cpu-threads').textContent = Number.isFinite(vitals.cpu_threads) ? `${vitals.cpu_threads}T` : '';
  const total = numeric(state.memory?.gpu_total_bytes ?? vitals.gpu_total_bytes);
  const free = numeric(state.memory?.gpu_free_bytes ?? vitals.gpu_free_bytes);
  const used = total !== null && free !== null ? total - free : numeric(vitals.gpu_used_bytes);
  const available = connected && Number.isFinite(total) && total > 0 && Number.isFinite(used);
  $('vital-vram').textContent = available ? `${gib(used)} / ${gib(total)}` : '—';
  $('vram-percent').textContent = available ? `${(used / total * 100).toFixed(1)}% used · GiB` : 'Memory telemetry unavailable';
  $('meter-vram').style.width = available ? `${Math.max(0, Math.min(100, used / total * 100))}%` : '0%';
  $('vital-vram').title = available ? `${(used / total * 100).toFixed(1)}% used · GiB · ${state.memory ? 'CUDA' : 'system'} memory telemetry` : 'VRAM telemetry unavailable';
  $('vital-ram').title = Number.isFinite(vitals.ram_used_bytes) && Number.isFinite(vitals.ram_total_bytes) ? `${gib(vitals.ram_used_bytes)} / ${gib(vitals.ram_total_bytes)} GiB` : 'RAM telemetry unavailable';
  const thermal = [];
  if (Number.isFinite(vitals.gpu_temperature)) thermal.push(`${Math.round(vitals.gpu_temperature)}°C`);
  if (Number.isFinite(vitals.gpu_power_watts)) thermal.push(`${Math.round(vitals.gpu_power_watts)} W`);
  $('vital-thermal').textContent = thermal.join(' / ') || '—';
  $('vitals-note').textContent = !connected ? 'Runtime disconnected' : vitals.error ? 'Some sensors unavailable' : Object.keys(vitals).length ? 'Live system telemetry · VRAM in GiB' : 'Awaiting runtime telemetry';
  renderSparklines();
}

function renderTransport() {
  const running = state.status === 'generating';
  const active = ['loading', 'generating', 'unloading'].includes(state.status);
  const stages = ['preparing', 'encoding_reference', 'prefill', 'generating', 'decoding', 'exporting'];
  const phase = ['verifying', 'loading'].includes(state.phase) ? 'preparing' : state.phase;
  const labels = {preparing: 'Preparing your studio', encoding_reference: 'Encoding your voice reference', prefill: 'Reading your script', generating: 'Creating your voice', decoding: 'Decoding audio on CPU', exporting: 'Saving your recording', complete: 'Your recording is ready', ready: 'Ready to bring your script to life', idle: 'Your next voice starts here'};
  const descriptions = {preparing: 'Verifying model assets and preparing memory for synthesis.', encoding_reference: 'The CPU codec is converting your reference into acoustic conditioning.', prefill: 'The CUDA backbone is processing your script and voice context before speech begins.', generating: 'Acoustic frames are being created. The counters below report actual generated audio.', decoding: 'The CPU codec is turning acoustic frames into a waveform. GPU compute may be near zero during this stage.', exporting: 'Writing the WAV, optional MP3, and reproducible request settings.', complete: 'Listen to the full recording, then download the audio and its settings.', ready: 'Your model is loaded. Refine the script and voice settings, then generate.', idle: 'Load your local model, add a script, and choose the voice you want to hear.'};
  const current = stages.indexOf(phase);
  const completed = Boolean(state.result) && !['loading', 'generating', 'unloading'].includes(state.status);
  $('phase-label').textContent = !connected ? 'Reconnecting to your studio' : uploading ? 'Adding your voice reference' : state.status === 'error' || state.error ? 'Generation needs attention' : labels[phase] || (running ? 'Creating your recording' : state.status === 'ready' ? labels.ready : labels.idle);
  $('phase-detail').textContent = !connected ? 'Your local runtime is not responding. The studio retries automatically.' : uploading ? 'Uploading and validating your reference clip. Your script stays right here.' : state.error || (state.status === 'error' ? state.message : descriptions[phase] || (running ? 'Your request is running. Live measurements appear as the runtime reports them.' : descriptions[state.status === 'ready' ? 'ready' : 'idle']));
  $('transport').dataset.activity = connected && (running || state.status === 'loading' || uploading) ? 'active' : 'idle';
  $('transport-light').classList.toggle('active', connected && (running || state.status === 'loading'));
  for (const element of document.querySelectorAll('[data-phase]')) {
    const index = stages.indexOf(element.dataset.phase);
    element.classList.toggle('active', connected && !completed && index === current && ['loading', 'generating'].includes(state.status));
    element.classList.toggle('done', connected && (completed || current > index));
    if (index === current && ['loading', 'generating'].includes(state.status)) element.setAttribute('aria-current', 'step');
    else element.removeAttribute('aria-current');
  }
  const metrics = active ? {} : state.result?.metrics || {};
  const frames = running ? state.progress?.frames : metrics.frames;
  const budget = running ? state.progress?.budget || Number($('max_frames').value) : metrics.max_frames || Number($('max_frames').value);
  $('generated-frames').textContent = Number.isFinite(frames) ? frames.toLocaleString() : '—';
  $('budget-stat').textContent = Number.isFinite(budget) ? budget.toLocaleString() : '—';
  $('generation-speed').textContent = running && Number.isFinite(state.progress?.frames_per_second) ? `${state.progress.frames_per_second.toFixed(1)} fps` : !running && Number.isFinite(metrics.frames) && metrics.generation_seconds > 0 ? `${(metrics.frames / metrics.generation_seconds).toFixed(1)} fps` : '—';
  $('produced-audio').textContent = Number.isFinite(frames) ? formatSeconds(frames * 2048 / 44100) : '—';
  $('elapsed').textContent = active && state.started_at ? `${formatSeconds(Math.max(0, Date.now() / 1000 - state.started_at))} elapsed` : metrics.synthesis_seconds ? `${formatSeconds(metrics.synthesis_seconds)} total` : '—';
  renderSpeechProgress();
  renderExpertPerformance();
}

function renderExpertPerformance() {
  const element = $('expert-performance');
  const active = ['loading', 'generating', 'unloading'].includes(state.status);
  const profile = state.performance || (!active ? state.result?.metrics?.expert_profile : null);
  const cpu = profile?.backends?.cpu;
  const gpu = profile?.backends?.gpu;
  const available = profile?.enabled && (cpu || gpu);
  element.hidden = !available;
  if (!available) return;
  const cpuSeconds = cpu?.wall_seconds || 0;
  const gpuSeconds = gpu?.wall_seconds || 0;
  const staging = (cpu?.host_staging_seconds || 0) + (gpu?.host_staging_seconds || 0);
  element.textContent = `Measured expert bridge · CPU ${cpuSeconds.toFixed(1)}s · GPU ${gpuSeconds.toFixed(1)}s · host staging / waits ${staging.toFixed(1)}s`;
  element.title = 'Cumulative blocking expert-call time for this request, including prefill. Host staging and waits are included in the CPU/GPU totals, not extra time. Backbone, audio tower and codec time are separate.';
}

function renderSpeechProgress() {
  const progress = state.progress || {};
  const preparing = ['loading', 'unloading'].includes(state.status);
  const metrics = preparing ? {} : state.result?.metrics || {};
  const running = state.status === 'generating';
  let percent = null;
  let label = 'Estimated speech progress';
  let detail = 'Speech length is estimated. Frame budget usage is separate.';
  if (running && state.phase === 'decoding' && progress.stage_total > 0 && Number.isFinite(progress.stage_completed)) {
    percent = Math.min(100, progress.stage_completed / progress.stage_total * 100);
    label = 'Audio decode progress';
    detail = `${progress.stage_completed} / ${progress.stage_total} codec windows · CPU waveform decoding`;
  } else if (running && (state.phase === 'generating' || (state.phase === 'prefill' && progress.completed_characters > 0))) {
    const estimatedFrames = progress.estimated_frames || state.plan?.estimated_frames || generationPlan?.estimated_frames;
    if (estimatedFrames > 0 && Number.isFinite(progress.frames)) percent = progress.frames / estimatedFrames * 100;
    const segmentCharacters = progress.segment_characters ?? generationPlan?.chunks?.[progress.segment_index - 1]?.characters;
    if (progress.total_characters > 0 && progress.segment_estimated_frames > 0 && Number.isFinite(segmentCharacters)) {
      const segmentRatio = progress.segment_complete ? 0 : Math.max(0, Math.min(1, (progress.segment_frames || 0) / progress.segment_estimated_frames));
      percent = ((progress.completed_characters || 0) + segmentRatio * segmentCharacters) / progress.total_characters * 100;
    }
    if (percent !== null) percent = Math.max(0, Math.min(99, percent));
    detail = progress.segment_count > 1 ? `Passage ${progress.segment_index || 1} / ${progress.segment_count} · completed passages plus estimated current speech` : 'Estimated from acoustic frames and expected speech length. It can reach 99% before speech ends.';
  } else if (!['loading', 'generating', 'unloading'].includes(state.status) && state.result && metrics.complete === true) {
    percent = 100;
    label = 'Generation complete';
    detail = 'The model emitted its end token and the recording was saved. Listen to verify the words.';
  } else if (running) {
    label = state.phase === 'decoding' ? 'Decoding audio' : state.phase === 'exporting' ? 'Exporting recording' : 'Preparing speech';
    detail = 'This stage does not report a measurable completion percentage.';
  } else if (metrics.complete === false && state.result) {
    label = 'Incomplete recording';
    detail = 'The frame limit was reached before the model finished.';
  }
  $('speech-progress-label').textContent = label;
  $('speech-progress-detail').textContent = detail;
  $('speech-percent').textContent = percent === null ? '—' : `${Math.floor(percent)}%`;
  $('speech-progress-fill').style.width = percent === null ? '0%' : `${percent}%`;
  $('speech-meter').classList.toggle('indeterminate', running && percent === null);
  $('speech-meter').setAttribute('aria-label', label);
  if (percent === null) $('speech-meter').removeAttribute('aria-valuenow');
  else $('speech-meter').setAttribute('aria-valuenow', String(Math.floor(percent)));
  const frames = running ? progress.frames : metrics.frames;
  const budget = running ? progress.budget : metrics.max_frames;
  $('frame-budget-used').textContent = Number.isFinite(frames) && budget > 0 ? `Frame budget used ${(frames / budget * 100).toFixed(1)}% (${frames.toLocaleString()} / ${budget.toLocaleString()})` : 'Frame budget used —';
}

function values(keys) {
  return Object.fromEntries(keys.map((key) => [key, $(key).type === 'checkbox' ? $(key).checked : numericKeys.has(key) ? Number($(key).value) : $(key).value]));
}

function applySettings(settings) {
  if (!Object.hasOwn(settings, 'placement_mode') && Object.hasOwn(settings, 'gpu_expert_layers')) {
    settings = {...settings, placement_mode: 'manual'};
  }
  if (!Object.hasOwn(settings, 'generation_mode') && Object.hasOwn(settings, 'chunk_length')) {
    settings = {...settings, generation_mode: Number(settings.chunk_length) === 0 ? 'single' : 'chunked'};
  }
  for (const key of settingsKeys) {
    if (!Object.hasOwn(settings, key)) continue;
    if ($(key).type === 'checkbox') $(key).checked = settings[key] === true;
    else $(key).value = settings[key];
  }
  updateCount();
  schedulePlan();
}

function updateCount() {
  $('character-count').textContent = `${$('text').value.length.toLocaleString()} characters`;
}

function showError(message) {
  $('error').textContent = message || '';
  $('error').hidden = !message;
}

async function request(url, body, options = {}) {
  const response = await fetch(url, body === undefined ? {cache: 'no-store', ...(options.method ? {} : {signal: AbortSignal.timeout(10000)}), ...options} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body), ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || data.message || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function safeUrl(value) {
  if (typeof value !== 'string' || !value) return '';
  try {
    const url = new URL(value, location.origin);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : '';
  } catch { return ''; }
}

function renderResult(result) {
  if (!result) {
    $('result').hidden = true;
    $('result-audio').pause();
    return;
  }
  const audioUrl = safeUrl(result.audio_url);
  if (!audioUrl) return;
  $('result').hidden = false;
  if (audioUrl !== lastResult) {
    $('result-audio').src = audioUrl;
    lastResult = audioUrl;
  }
  $('audio-download').href = audioUrl;
  $('audio-download').textContent = audioUrl.split('?')[0].endsWith('.mp3') ? 'Download MP3' : 'Download WAV';
  for (const [id, url] of [['wav-download', result.wav_url], ['settings-download', result.settings_url]]) {
    $(id).hidden = !safeUrl(url);
    if (safeUrl(url)) $(id).href = safeUrl(url);
  }
  const metrics = result.metrics || {};
  const summary = [];
  if (Number.isFinite(metrics.audio_seconds)) summary.push(`${formatSeconds(metrics.audio_seconds)} audio`);
  if (Number.isFinite(metrics.synthesis_seconds)) summary.push(`${formatSeconds(metrics.synthesis_seconds)} to synthesize`);
  if (metrics.frames && metrics.generation_seconds > 0) summary.push(`${(metrics.frames / metrics.generation_seconds).toFixed(1)} frames/s`);
  if (Number.isFinite(metrics.seed)) summary.push(`Seed ${metrics.seed}`);
  if (Number.isFinite(metrics.gpu_expert_layers)) summary.push(`${metrics.gpu_expert_layers} GPU expert layers`);
  $('metrics').textContent = [summary.join(' · '), result.warning].filter(Boolean).join('. ');
  $('result-status').textContent = metrics.complete === false ? 'Incomplete · frame limit reached' : 'Generation complete';
  $('result-status').classList.toggle('incomplete', metrics.complete === false);
}

function render() {
  const busy = !connected || pending || planning || uploading || ['loading', 'generating', 'unloading'].includes(state.status);
  if (state.model) {
    for (const key of modelKeys) {
      if (Object.hasOwn(state.model, key)) $(key).value = state.model[key];
    }
  }
  $('status').dataset.status = state.status;
  $('status-label').textContent = {unloaded: 'Model unloaded', loading: 'Loading model', ready: 'Model ready', generating: 'Generating audio', unloading: 'Unloading model', error: 'Needs attention'}[state.status] || 'Connecting';
  if (state.status === 'error') $('status-label').textContent = state.model ? 'Model error · unload to reset' : 'Model stopped · reload to continue';
  $('progress').textContent = uploading ? 'Uploading reference audio…' : state.message || (state.status === 'ready' ? 'Ready when you are.' : 'Load a model to begin.');
  if (state.status === 'generating' && state.started_at) {
    const elapsed = Math.max(0, Math.floor(Date.now() / 1000 - state.started_at));
    const frames = state.progress?.frames;
    $('progress').textContent = frames ? `Generated ${frames.toLocaleString()} frames · ${(frames * 2048 / 44100).toFixed(1)}s of audio · ${elapsed}s elapsed` : `Preparing speech generation · ${elapsed}s elapsed`;
  }
  $('load').disabled = busy || state.status === 'ready';
  $('unload').disabled = busy || !['ready', 'error'].includes(state.status);
  $('generate').disabled = busy || state.status !== 'ready';
  $('cancel').hidden = state.status !== 'generating';
  $('cancel').disabled = pending || !connected;
  for (const key of settingsKeys) $(key).disabled = busy || (modelKeys.includes(key) && state.status === 'ready');
  for (const id of ['examples', 'reference', 'load-sample-reference', 'clear-reference', 'save-settings', 'import-settings', 'reset-settings']) $(id).disabled = busy;
  for (const id of ['use-smart-segments', 'retry-smart-segments', 'risk-acknowledged']) $(id).disabled = busy;
  $('failure-guidance').hidden = !(state.error || (state.status === 'error' && state.message));
  $('clear-reference').hidden = !referenceId;
  $('output_format').querySelector('[value="mp3"]').disabled = !state.mp3_available;
  $('mp3_bitrate').disabled = busy || !state.mp3_available || $('output_format').value !== 'mp3';
  $('chunk_length').disabled = busy || $('generation_mode').value === 'single';
  $('max_frames').max = $('generation_mode').value === 'single' ? '8192' : '65536';
  const automatic = $('placement_mode').value === 'auto';
  $('gpu_expert_layers').disabled = busy || state.status === 'ready' || automatic;
  $('headroom_percent').disabled = busy || state.status === 'ready';
  $('placement-status').textContent = state.model?.placement ? `${state.model.placement_mode === 'auto' ? 'Automatic placement selected' : 'Loaded placement:'} ${state.model.gpu_expert_layers} GPU expert layers.` : automatic ? 'The runtime will select GPU expert layers when you load the model.' : 'The runtime will use your manual layer count.';
  const memory = state.memory;
  if (memory && Number.isFinite(memory.gpu_free_bytes) && Number.isFinite(memory.gpu_total_bytes) && memory.gpu_total_bytes > 0) {
    const free = memory.gpu_free_bytes / 2 ** 30;
    const total = memory.gpu_total_bytes / 2 ** 30;
    $('memory-status').textContent = `GPU memory: ${free.toFixed(2)} GiB free / ${total.toFixed(2)} GiB total (${(100 * memory.gpu_free_bytes / memory.gpu_total_bytes).toFixed(1)}% free).`;
  } else $('memory-status').textContent = '';
  const duration = Math.round(Number($('max_frames').value) * 2048 / 44100);
  $('frame-duration').textContent = `${$('max_frames').value} frames provide up to approximately ${Math.floor(duration / 60)}m ${String(duration % 60).padStart(2, '0')}s of audio.`;
  if (!state.mp3_available && $('output_format').value === 'mp3') $('output_format').value = 'wav';
  $('mp3-note').textContent = state.mp3_available ? 'FFmpeg ready · WAV original is always retained.' : 'MP3 unavailable. Install FFmpeg with libmp3lame and restart the studio, or export WAV.';
  if (state.status === 'error' && state.message) showError(state.message);
  const serverError = state.error || (state.status === 'error' ? state.message : '');
  if (serverError) { showError(serverError); lastServerError = serverError; }
  else if (lastServerError) { showError(''); lastServerError = ''; }
  if (!connected) {
    $('status-label').textContent = 'Runtime disconnected';
    $('progress').textContent = 'Cannot reach the runtime. Check that the server is running.';
  }
  renderResult(state.result);
  renderVitals();
  renderTransport();
}

function localPlan(settings, reason) {
  const words = settings.text.trim().split(/\s+/).filter(Boolean).length;
  const frames = Math.ceil(words / 2.7 * 44100 / 2048);
  const single = settings.generation_mode === 'single';
  const high = (single && words > 150) || frames > settings.max_frames * 0.9;
  return {mode: settings.generation_mode, words, characters: settings.text.length, segments: single ? 1 : Math.max(1, Math.ceil(settings.text.length / Math.max(32, settings.chunk_length || 200))), estimated_frames: frames, estimated_seconds: frames * 2048 / 44100, max_frames: settings.max_frames, budget_seconds: settings.max_frames * 2048 / 44100, recommended_chunked_frames: Math.min(65536, Math.max(512, Math.ceil(frames * 1.3))), risk: high ? 'high' : words > 80 ? 'warning' : 'low', warnings: [reason, ...(single && words > 150 ? ['A long single pass can omit or repeat words, stop early, or produce long pauses. Smart segments are recommended.'] : []), ...(frames > settings.max_frames * 0.9 ? ['The estimated speech is close to or beyond the total frame budget. Consider a larger budget or a shorter script.'] : [])], local: true};
}

function renderPlan(plan) {
  $('plan-card').hidden = !plan;
  if (!plan) return;
  $('plan-card').dataset.risk = plan.risk;
  $('plan-risk').textContent = plan.risk === 'high' ? 'HIGH RISK' : plan.risk === 'warning' ? 'REVIEW REQUEST' : 'READY TO REVIEW';
  const speechRange = plan.conservative_seconds ? `${formatSeconds(plan.estimated_seconds)}–${formatSeconds(plan.conservative_seconds)}` : formatSeconds(plan.estimated_seconds);
  $('plan-summary').textContent = `${plan.words.toLocaleString()} words · ${plan.segments} ${plan.segments === 1 ? 'passage' : 'passages'} · estimated ${speechRange} of speech · ${formatSeconds(plan.budget_seconds)} total frame allowance`;
  $('plan-warnings').replaceChildren();
  for (const warning of plan.warnings || []) {
    const item = document.createElement('li');
    item.textContent = warning;
    $('plan-warnings').append(item);
  }
  $('plan-source').textContent = plan.local ? 'Local estimate · runtime planner unavailable' : 'Runtime preflight · speech length is estimated';
  $('risk-ack-label').hidden = plan.risk !== 'high';
  $('use-smart-segments').hidden = plan.mode === 'chunked' && plan.risk === 'low';
}

async function refreshPlan() {
  const version = ++planVersion;
  const settings = {...values(generationKeys), text: $('text').value, reference_text: $('reference_text').value, reference_id: referenceId};
  if (settings.generation_mode === 'single') settings.chunk_length = 0;
  if (!settings.text.trim()) { generationPlan = null; renderPlan(null); return null; }
  let plan;
  try { plan = await request('/api/plan', settings, {signal: AbortSignal.timeout(10000)}); }
  catch (error) {
    const message = error.status === 400 ? error.message : error.status === 404 ? 'Restart the runtime to enable its request planner. This is a rough local estimate.' : 'Runtime planning is unavailable. This is a rough local estimate; reconnect before generating.';
    plan = localPlan(settings, message);
    plan.invalid = true;
    plan.risk = 'high';
  }
  if (version !== planVersion) return null;
  generationPlan = plan;
  renderPlan(plan);
  return plan;
}

function schedulePlan() {
  clearTimeout(planTimer);
  $('risk-acknowledged').checked = false;
  planTimer = setTimeout(refreshPlan, 400);
}

function useSmartSegments() {
  $('generation_mode').value = 'chunked';
  const passageLength = generationPlan?.recommended_chunk_length || Math.min(2000, Math.max(200, Math.ceil($('text').value.length / 24)));
  $('chunk_length').value = passageLength;
  const recommended = numeric(generationPlan?.recommended_chunked_frames);
  if (recommended !== null) $('max_frames').value = Math.min(65536, Math.max(Number($('max_frames').value), recommended));
  $('risk-acknowledged').checked = false;
  $('settings-message').textContent = `Smart segments selected: up to ${passageLength} characters per passage at smart boundaries, with ${Number($('max_frames').value).toLocaleString()} total frames. Your script, reference and sampling settings are retained.`;
  render();
  refreshPlan();
}

async function action(url, body) {
  pending = true;
  showError('');
  render();
  try {
    const next = await request(url, body);
    if (next.status) state = next;
  } catch (error) { showError(error.message); }
  finally { pending = false; render(); }
}

async function poll() {
  try {
    const next = await request('/api/state');
    if (next.ui_revision && uiRevision && next.ui_revision !== uiRevision && !pending && !uploading) {
      uiRevision = next.ui_revision;
      if (saveTransient()) { location.reload(); return; }
    }
    if (next.ui_revision && !uiRevision) uiRevision = next.ui_revision;
    if (referenceId && state.server_session_id && next.server_session_id && state.server_session_id !== next.server_session_id) {
      referenceId = null;
      $('reference').value = '';
      $('reference-status').textContent = 'The runtime restarted. Upload your matching voice reference again.';
    }
    state = next;
    connected = true;
    if (transientRestore) restoreReference();
    rememberTelemetry();
    render();
  } catch (error) {
    connected = false;
    render();
  } finally { setTimeout(poll, 1000); }
}

function saveTransient() {
  try {
    sessionStorage.setItem(transientKey, JSON.stringify({saved_at: Date.now(), settings: values(settingsKeys), reference_id: referenceId, reference_status: $('reference-status').textContent, server_session_id: state.server_session_id || null, example: $('examples').value}));
    return true;
  } catch {
    $('settings-message').textContent = 'A studio update is ready. Browser session storage is unavailable; export your settings before refreshing.';
    return false;
  }
}

function restoreReference() {
  if (transientRestore.reference_id && transientRestore.server_session_id && transientRestore.server_session_id === state.server_session_id) {
    referenceId = transientRestore.reference_id;
    $('reference-status').textContent = transientRestore.reference_status || 'Voice reference restored for this runtime session.';
  } else if (transientRestore.reference_id) {
    $('reference-status').textContent = 'The runtime restarted. Upload your matching voice reference again.';
  }
  transientRestore = null;
}

$('model-form').addEventListener('submit', (event) => {
  event.preventDefault();
  action('/api/load', values(modelKeys));
});
$('unload').addEventListener('click', () => action('/api/unload', {}));
$('cancel').addEventListener('click', () => action('/api/cancel', {}));
$('text').addEventListener('input', updateCount);
$('output_format').addEventListener('change', render);
$('generation_mode').addEventListener('change', render);
$('placement_mode').addEventListener('change', render);
$('max_frames').addEventListener('input', render);
$('generate-form').addEventListener('invalid', (event) => {
  const details = event.target.closest('details');
  if (details) details.open = true;
}, true);
$('generate-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const settings = values(generationKeys);
  if (settings.generation_mode === 'single') settings.chunk_length = 0;
  if (settings.temperature <= 0 || settings.top_p <= 0) return showError('Temperature and top P must be greater than zero.');
  if (settings.chunk_length !== 0 && settings.chunk_length < 32) return showError('Chunk length must be 0 (disabled) or between 32 and 2000.');
  if (!$('text').value.trim()) return showError('Add a script before generating.');
  if (referenceId && !$('reference_text').value.trim()) return showError('Add the exact transcript of your reference audio.');
  if (!referenceId && $('reference_text').value.trim()) return showError('Upload reference audio to accompany its transcript, or clear the transcript.');
  clearTimeout(planTimer);
  planning = true;
  render();
  let plan;
  try { plan = await refreshPlan(); }
  finally { planning = false; render(); }
  if (!plan) return showError('The request changed during planning. Review it and generate again.');
  if (plan.invalid) return showError(plan.warnings[0]);
  if (plan.risk === 'high' && !$('risk-acknowledged').checked) {
    showError('Review the high-risk request below. Use smart segments, or acknowledge the risk before generating anyway.');
    $('risk-acknowledged').focus();
    return;
  }
  action('/api/generate', {...settings, text: $('text').value, reference_id: referenceId, reference_text: $('reference_text').value, plan_acknowledged: $('risk-acknowledged').checked});
});

for (const key of [...generationKeys, 'text', 'reference_text']) {
  $(key).addEventListener('input', schedulePlan);
  $(key).addEventListener('change', schedulePlan);
}
$('use-smart-segments').addEventListener('click', useSmartSegments);
$('retry-smart-segments').addEventListener('click', useSmartSegments);

$('reference').addEventListener('change', async () => {
  const file = $('reference').files[0];
  referenceId = null;
  if (!file) { $('reference-status').textContent = 'Upload a short, clear voice clip to condition generation.'; return; }
  uploading = true;
  showError('');
  render();
  try {
    const data = await request('/api/reference', undefined, {method: 'POST', headers: {'Content-Type': file.type || 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name)}, body: file});
    referenceId = data.reference_id;
    $('reference-status').textContent = `Ready: ${data.name || file.name}`;
  } catch (error) { showError(error.message); $('reference-status').textContent = 'Upload failed. Choose your file again to retry.'; $('reference').value = ''; }
  finally { uploading = false; render(); }
});

$('clear-reference').addEventListener('click', () => {
  referenceId = null;
  $('reference').value = '';
  $('reference_text').value = '';
  $('reference-status').textContent = 'WAV, FLAC, MP3 or OGG · up to 60 seconds';
  schedulePlan();
  render();
});

$('examples').addEventListener('change', () => {
  const example = examples.find((item) => item.id === $('examples').value);
  $('load-sample-reference').hidden = !example?.reference_url;
  if (!example) return;
  const exampleSettings = {...example.settings, text: example.text};
  delete exampleSettings.reference_text;
  if (state.status !== 'unloaded') for (const key of modelKeys) delete exampleSettings[key];
  applySettings(exampleSettings);
  $('example-description').textContent = example.description || '';
  const audioUrl = safeUrl(example.audio_url);
  const transcriptUrl = safeUrl(example.transcript_url);
  $('sample-transcript').hidden = !transcriptUrl;
  if (transcriptUrl) $('sample-transcript').href = transcriptUrl;
  $('sample-preview').hidden = !audioUrl;
  $('sample-audio').pause();
  if (audioUrl) $('sample-audio').src = audioUrl;
  else $('sample-audio').removeAttribute('src');
  render();
});

$('load-sample-reference').addEventListener('click', async () => {
  const example = examples.find((item) => item.id === $('examples').value);
  if (!example?.reference_url) return;
  uploading = true;
  showError('');
  render();
  try {
    const data = await request('/api/sample-reference', {id: example.id});
    referenceId = data.reference_id;
    $('reference').value = '';
    $('reference_text').value = data.reference_text || '';
    $('reference-status').textContent = `Ready: ${data.name || 'Sample voice reference'}`;
    schedulePlan();
  } catch (error) { showError(error.message); }
  finally { uploading = false; render(); }
});

$('save-settings').addEventListener('click', () => {
  try {
    localStorage.setItem('drama3.settings.v1', JSON.stringify(values(settingsKeys)));
    $('settings-message').textContent = 'Settings and script saved in this browser. Reference audio is not saved.';
  } catch { showError('This browser could not save settings. Download settings after generation instead.'); }
});
$('export-settings').addEventListener('click', () => {
  const settings = {schema: 'fish-drama3-studio-v1', model: values(modelKeys), ...values([...generationKeys, 'text', 'reference_text']), reference_note: 'Reference audio is separate. Upload the same reference clip when restoring these settings.'};
  if (settings.generation_mode === 'single') settings.chunk_length = 0;
  const url = URL.createObjectURL(new Blob([JSON.stringify(settings, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'fish-drama3-settings.json';
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  $('settings-message').textContent = 'Settings JSON exported. Reference audio must be saved separately.';
});
$('reset-settings').addEventListener('click', () => {
  applySettings(defaults);
  try { localStorage.removeItem('drama3.settings.v1'); } catch { /* Storage may be disabled. */ }
  referenceId = null;
  $('reference').value = '';
  $('reference-status').textContent = 'Upload a short, clear voice clip to condition generation.';
  $('examples').value = '';
  $('load-sample-reference').hidden = true;
  $('example-description').textContent = '';
  $('sample-audio').pause();
  $('sample-preview').hidden = true;
  $('settings-message').textContent = 'Default settings restored. Saved browser settings cleared.';
  render();
});
$('import-settings').addEventListener('click', () => $('settings-file').click());
$('settings-file').addEventListener('change', async () => {
  try {
    const file = $('settings-file').files[0];
    if (!file) return;
    const data = JSON.parse(await file.text());
    if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Settings must be a JSON object.');
    referenceId = null;
    $('reference').value = '';
    $('reference-status').textContent = 'Settings imported. Upload the matching reference audio before generating.';
    applySettings({...data.model, ...data.generation, ...data.settings, ...data});
    $('settings-message').textContent = state.model ? 'Generation settings imported. The active model configuration is retained; unload first to import different model settings. Re-upload reference audio if needed.' : 'Settings imported. Upload reference audio separately if needed.';
    showError('');
    render();
  } catch (error) { showError(`Could not import settings: ${error.message}`); }
  finally { $('settings-file').value = ''; }
});

for (const button of document.querySelectorAll('.help')) {
  button.addEventListener('click', () => {
    const expanded = button.getAttribute('aria-expanded') === 'true';
    button.setAttribute('aria-expanded', String(!expanded));
    const panel = $(button.getAttribute('aria-controls'));
    panel.hidden = expanded;
  });
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      button.setAttribute('aria-expanded', 'false');
      $(button.getAttribute('aria-controls')).hidden = true;
    }
  });
}

async function initialize() {
  try {
    const saved = localStorage.getItem('drama3.settings.v1');
    if (saved) { applySettings(JSON.parse(saved)); $('settings-message').textContent = 'Your saved browser settings were restored. Re-upload reference audio if needed.'; }
  } catch { /* Invalid or unavailable storage does not prevent startup. */ }
  try {
    const saved = sessionStorage.getItem(transientKey);
    sessionStorage.removeItem(transientKey);
    if (saved) {
      const transient = JSON.parse(saved);
      if (transient.settings && Date.now() - transient.saved_at < 86400000) {
        applySettings(transient.settings);
        transientRestore = transient;
        $('settings-message').textContent = 'Studio updated. Your current script and settings were preserved for this session.';
      }
    }
  } catch { /* Session recovery is optional when browser storage is restricted. */ }
  try {
    const data = await request('/api/examples');
    examples = data.examples || [];
    for (const example of examples) {
      const option = document.createElement('option');
      option.value = example.id;
      option.textContent = example.title;
      $('examples').append(option);
    }
    if (transientRestore?.example) $('examples').value = transientRestore.example;
  } catch { $('example-description').textContent = 'Examples could not be loaded. You can still write your own script.'; }
  render();
  poll();
}
initialize();
setInterval(() => { renderFreshness(); if (['loading', 'generating', 'unloading'].includes(state.status)) renderTransport(); }, 1000);
