/* VoxAI frontend - vanilla JS, no dependencies.
 *
 * Flow: MediaRecorder captures the microphone -> POST /api/voice (SSE progress)
 * -> backend runs STT (faster-whisper) -> LLM (Ollama/Qwen3) -> TTS (Piper)
 * -> the returned WAV is played through an HTML <audio> element.
 *
 * The browser never keeps conversation history: it only holds the session id.
 */
(() => {
  'use strict';

  // ---------------------------------------------------------------- helpers
  const $ = (id) => document.getElementById(id);
  const el = {
    healthChip: $('health-chip'), statusPill: $('status-pill'),
    settingsToggle: $('settings-toggle'), settingsPanel: $('settings-panel'),
    chatLog: $('chat-log'), emptyState: $('empty-state'),
    errorBanner: $('error-banner'), errorText: $('error-text'), errorDismiss: $('error-dismiss'),
    latencyPanel: $('latency-panel'), latencyGrid: $('latency-grid'), latencyNote: $('latency-note'),
    micBtn: $('mic-btn'), startBtn: $('start-btn'), stopBtn: $('stop-btn'),
    timer: $('record-timer'), newBtn: $('new-btn'), clearBtn: $('clear-btn'),
    textForm: $('text-form'), textInput: $('text-input'), sendBtn: $('send-btn'),
    temperature: $('temperature'), temperatureOut: $('temperature-out'),
    maxTokens: $('max-tokens'), maxTokensOut: $('max-tokens-out'), autoplay: $('autoplay'),
    infoLlm: $('info-llm'), infoWhisper: $('info-whisper'), infoPiper: $('info-piper'),
    infoSession: $('info-session'),
  };

  const STATUS_TEXT = {
    ready: 'Ready',
    listening: 'Listening',
    transcribing: 'Transcribing',
    thinking: 'Thinking',
    speaking: 'Speaking',
    error: 'Error',
  };

  const state = {
    sessionId: sessionStorage.getItem('voxai.session') || null,
    busy: false,            // one voice/chat request at a time
    recording: false,
    permPending: false,     // getUserMedia prompt is open
    stream: null,           // getUserMedia stream
    recorder: null,         // MediaRecorder
    chunks: [],
    startedAt: 0,
    timerId: null,
    lastAudio: null,        // currently playing <audio>
    healthOk: null,         // last known health status
  };

  function setStatus(key) {
    el.statusPill.className = `pill pill-${key}`;
    el.statusPill.textContent = STATUS_TEXT[key] || key;
  }

  function fmtMs(ms) {
    if (ms === null || ms === undefined || Number.isNaN(ms)) return null;
    return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
  }

  // ---------------------------------------------------------------- errors
  function showError(message, hint) {
    el.errorText.innerHTML = '';
    const span = document.createElement('span');
    span.textContent = message;
    el.errorText.appendChild(span);
    if (hint) {
      const h = document.createElement('span');
      h.className = 'hint';
      h.textContent = hint;
      el.errorText.appendChild(h);
    }
    el.errorBanner.hidden = false;
    setStatus('error');
    // Return to Ready shortly after unless something else took over.
    clearTimeout(showError._t);
    showError._t = setTimeout(() => { if (!state.busy && !state.recording) setStatus('ready'); }, 4000);
  }
  function hideError() { el.errorBanner.hidden = true; }
  el.errorDismiss.addEventListener('click', hideError);

  async function readError(resp) {
    let body = null;
    try { body = await resp.json(); } catch (_) { /* not JSON */ }
    const message = (body && (body.error || body.message)) || `Request failed (HTTP ${resp.status}).`;
    return { message, hint: body && body.hint, code: body && body.code };
  }

  // ---------------------------------------------------------------- bubbles
  function ensureChatStarted() {
    if (el.emptyState && el.emptyState.parentNode) el.emptyState.remove();
  }

  function addBubble(role, text, metaHtml) {
    ensureChatStarted();
    const bubble = document.createElement('div');
    bubble.className = `bubble bubble-${role}`;
    const who = document.createElement('div');
    who.className = 'bubble-role';
    who.textContent = role === 'user' ? 'You' : 'VoxAI';
    const body = document.createElement('div');
    body.className = 'bubble-text';
    body.textContent = text;
    bubble.appendChild(who);
    bubble.appendChild(body);
    if (metaHtml) {
      const meta = document.createElement('div');
      meta.className = 'bubble-meta';
      meta.innerHTML = metaHtml;
      bubble.appendChild(meta);
    }
    el.chatLog.appendChild(bubble);
    el.chatLog.scrollTop = el.chatLog.scrollHeight;
    return bubble;
  }

  function addNotice(text) {
    ensureChatStarted();
    const n = document.createElement('div');
    n.className = 'notice-banner';
    n.textContent = text;
    el.chatLog.appendChild(n);
    el.chatLog.scrollTop = el.chatLog.scrollHeight;
    return n;
  }

  // ---------------------------------------------------------------- latency
  const LATENCY_ROWS = [
    ['recording_ms', 'Recording (browser)'],
    ['stt_ms', 'Speech → text'],
    ['llm_ms', 'LLM answer'],
    ['tts_ms', 'Text → speech'],
    ['model_load_ms', 'First-call model load'],
    ['backend_total_ms', 'Backend total'],
    ['total_ms', 'End-to-end total'],
  ];
  const SOURCE_LABEL = {
    measured_server: 'measured',
    measured_client: 'measured (browser)',
    measured_client_plus_server: 'measured (browser + server)',
    estimated: 'estimated',
  };

  function showLatency(latency, extraRows) {
    if (!latency) return;
    el.latencyGrid.innerHTML = '';
    const rows = (extraRows || []).concat(LATENCY_ROWS);
    for (const [key, label] of rows) {
      const value = fmtMs(latency[key]);
      const source = (latency.sources || {})[key];
      const row = document.createElement('div');
      row.className = 'latency-row';
      const lab = document.createElement('span');
      lab.className = 'label';
      lab.textContent = label;
      const val = document.createElement('span');
      val.className = value ? 'value' : 'value na';
      val.textContent = value ? `${value}${source ? ` · ${SOURCE_LABEL[source] || source}` : ''}` : '—';
      row.appendChild(lab);
      row.appendChild(val);
      el.latencyGrid.appendChild(row);
    }
    const streaming = latency.streaming === true;
    el.latencyNote.textContent =
      `${latency.mode === 'buffered' ? 'Buffered pipeline' : 'Pipeline'} · ` +
      `streaming: ${streaming ? 'yes' : 'no'}. ` + (latency.note || '');
    el.latencyPanel.hidden = false;
  }

  // ---------------------------------------------------------------- health
  async function pollHealth() {
    try {
      const resp = await fetch('/api/health', { cache: 'no-store' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const h = await resp.json();
      state.healthOk = h.status;
      el.healthChip.textContent = h.status === 'ok' ? 'backend online' : `backend ${h.status}`;
      el.healthChip.className = `chip ${h.status === 'ok' ? 'ok' : 'warn'}`;

      const o = h.ollama || {};
      el.infoLlm.textContent = o.reachable
        ? `${o.model} @ ${o.host} ${o.model_present ? '· ready' : '· model missing'}`
        : `offline — ${o.hint || 'start ollama serve'}`;
      el.infoLlm.style.color = (o.reachable && o.model_present) ? '' : 'var(--error)';

      const w = h.whisper || {};
      el.infoWhisper.textContent =
        `${w.model} · ${w.device}/${w.compute_type} · ${w.loaded ? `loaded (${w.load_seconds}s)` : w.state || 'idle'}`;

      const p = h.piper || {};
      el.infoPiper.textContent = p.model_present
        ? `${p.model_path.split(/[\\/]/).pop()} ${p.loaded ? '· loaded' : '· idle'}${p.hindi_present ? ' + hi' : ''}`
        : `${p.model_path} — missing`;

      const turns = window.__voxaiTurns || 0;
      const sid = state.sessionId ? `${state.sessionId.slice(0, 8)}…` : 'new';
      const sess = h.sessions || {};
      el.infoSession.textContent = `${sid} · ${turns} turn(s) · ${sess.active_sessions ?? 0} active on server`;
    } catch (err) {
      el.healthChip.textContent = 'backend unreachable';
      el.healthChip.className = 'chip err';
    }
  }

  // ---------------------------------------------------------------- session
  async function ensureSession() {
    if (state.sessionId) return state.sessionId;
    const resp = await fetch('/api/session/new', { method: 'POST' });
    const body = await resp.json();
    if (!resp.ok || !body.session_id) throw new Error(body.error || 'Could not create a session.');
    state.sessionId = body.session_id;
    sessionStorage.setItem('voxai.session', state.sessionId);
    return state.sessionId;
  }

  function adoptSession(id) {
    if (id && id !== state.sessionId) {
      state.sessionId = id;
      sessionStorage.setItem('voxai.session', id);
    }
  }

  // ---------------------------------------------------------------- recording
  function pickMimeType() {
    if (typeof MediaRecorder === 'undefined') return null;
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];
    for (const m of candidates) {
      if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) return m;
    }
    return '';
  }

  function extFor(mime) {
    if (!mime) return 'webm';
    if (mime.includes('mp4')) return 'm4a';
    if (mime.includes('ogg')) return 'ogg';
    return 'webm';
  }

  function setRecordingUi(on) {
    state.recording = on;
    el.micBtn.classList.toggle('recording', on);
    el.micBtn.setAttribute('aria-label', on ? 'Stop recording' : 'Start recording');
    el.startBtn.disabled = on || state.busy;
    el.stopBtn.disabled = !on;
    el.sendBtn.disabled = state.busy;
    el.textInput.disabled = state.busy;
  }

  async function startRecording() {
    if (state.recording || state.busy || state.permPending) return;
    hideError();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError('This browser does not expose the microphone API (getUserMedia).',
                'Use a recent Chrome, Edge or Firefox over http://localhost or HTTPS.');
      return;
    }
    if (typeof MediaRecorder === 'undefined') {
      showError('MediaRecorder is not available in this browser.');
      return;
    }
    // Tell the user what is happening while the browser's permission prompt
    // is open (it can stay open indefinitely if the user ignores it).
    state.permPending = true;
    const permNotice = addNotice('Waiting for microphone permission — allow the microphone when your browser asks.');
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      permNotice.remove();
      state.permPending = false;
      const map = {
        NotAllowedError: ['Microphone permission denied.', 'Click the padlock in the address bar and allow the microphone, then try again.'],
        NotFoundError: ['No microphone found.', 'Connect a microphone and retry.'],
        NotReadableError: ['The microphone is already in use by another application.', 'Close other apps using the mic and retry.'],
      };
      const [msg, hint] = map[err.name] || [`Could not open the microphone (${err.name}).`, ''];
      showError(msg, hint);
      return;
    }
    permNotice.remove();
    state.permPending = false;

    state.stream = stream;
    state.chunks = [];
    const mime = pickMimeType();
    let recorder;
    try {
      recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    } catch (err) {
      showError(`Could not start recording: ${err.message}`);
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    state.recorder = recorder;
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) state.chunks.push(e.data); };
    recorder.onstop = () => {
      const blob = new Blob(state.chunks, { type: recorder.mimeType || mime || 'audio/webm' });
      const ms = Date.now() - state.startedAt;
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(state.timerId);
      sendRecording(blob, ms, extFor(recorder.mimeType || mime));
    };
    recorder.onerror = (e) => {
      showError(`Recording failed: ${(e.error && e.error.message) || 'unknown error'}`);
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(state.timerId);
      setRecordingUi(false);
      setStatus('ready');
    };

    recorder.start(); // collect in one chunk; nothing is streamed to the backend
    state.startedAt = Date.now();
    el.timer.textContent = '0.0s';
    state.timerId = setInterval(() => {
      el.timer.textContent = `${((Date.now() - state.startedAt) / 1000).toFixed(1)}s`;
    }, 100);
    setRecordingUi(true);
    setStatus('listening');
  }

  function stopRecording() {
    if (!state.recording || !state.recorder) return;
    setRecordingUi(false);
    setStatus('transcribing');
    if (state.recorder.state !== 'inactive') state.recorder.stop();
  }

  // ---------------------------------------------------------------- SSE parse
  async function readSse(resp, handlers) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        let event = 'message';
        let data = '';
        for (const line of raw.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) data += line.slice(5).trim();
        }
        if (!data) continue;
        let payload;
        try { payload = JSON.parse(data); } catch (_) { continue; }
        if (handlers[event]) handlers[event](payload);
      }
    }
  }

  // ---------------------------------------------------------------- voice
  async function sendRecording(blob, recordingMs, ext) {
    if (blob.size === 0) {
      showError('The recording was empty.', 'Check the microphone and record again.');
      setRecordingUi(false);
      setStatus('ready');
      return;
    }
    state.busy = true;
    setRecordingUi(false);
    setStatus('transcribing');
    const started = performance.now();

    let transcriptShown = false;
    const onProgress = (p) => {
      if (p.stage === 'stt') setStatus('transcribing');
      else if (p.stage === 'llm') {
        setStatus('thinking');
        if (p.transcript) {
          addBubble('user', p.transcript,
            p.language ? `<span class="tag">${p.language}</span>` : '');
          transcriptShown = true;
        }
      } else if (p.stage === 'tts') setStatus('speaking');
    };
    const onDone = (result) => finishVoice(result, performance.now() - started, transcriptShown);
    const onError = (body) => {
      state.busy = false;
      setStatus('error');
      showError(body.error || 'The voice pipeline failed.', body.hint);
      if (!body.error) console.error('voice pipeline error', body);
      setTimeout(() => { if (!state.busy && !state.recording) setStatus('ready'); }, 3000);
    };

    try {
      await ensureSession();
      const form = new FormData();
      form.append('audio', blob, `recording.${ext}`);
      form.append('session_id', state.sessionId || '');
      form.append('recording_ms', String(recordingMs));
      form.append('temperature', el.temperature.value);
      form.append('max_tokens', el.maxTokens.value);
      form.append('stream', 'progress');

      const resp = await fetch('/api/voice', { method: 'POST', body: form });

      if (resp.headers.get('content-type') && resp.headers.get('content-type').includes('text/event-stream')) {
        await readSse(resp, {
          progress: onProgress,
          done: (payload) => onDone(payload),
          error: (payload) => onError({ error: payload.error, hint: payload.hint }),
        });
        if (state.busy) {
          // Stream closed without a final done/error event (connection dropped).
          onError({ error: 'The connection to the backend closed before the answer finished.',
                    hint: 'Check the Flask server log and try again.' });
        }
      } else if (resp.ok) {
        onDone(await resp.json());
      } else {
        onError(await readError(resp));
      }
    } catch (err) {
      state.busy = false;
      setStatus('error');
      showError(`Could not reach the backend: ${err.message}`, 'Is the Flask server still running?');
      setTimeout(() => { if (!state.busy && !state.recording) setStatus('ready'); }, 4000);
    }
  }

  function finishVoice(payload, wallMs, transcriptShown) {
    const result = payload; // SSE done event carries {ok, ...result}
    state.busy = false;
    adoptSession(result.session_id);
    window.__voxaiTurns = result.history_turns || 0;

    if (!transcriptShown && result.transcript) {
      addBubble('user', result.transcript,
        result.language ? `<span class="tag">${result.language}</span>` : '');
    } else if (transcriptShown && result.transcript) {
      // Update the last user bubble's meta with language probability.
      const last = el.chatLog.querySelector('.bubble-user:last-of-type .bubble-meta');
      if (last && result.language) {
        const prob = result.language_probability
          ? ` · conf ${(result.language_probability * 100).toFixed(0)}%` : '';
        last.innerHTML = `<span class="tag">${result.language}</span>${prob}`;
      }
    }

    const usage = result.usage || {};
    const lat = result.latency || {};
    const metaBits = [];
    if (usage.model) metaBits.push(`<span class="tag">${usage.model}</span>`);
    metaBits.push(`turn ${result.history_turns ?? '—'}`);
    const metaHtml = metaBits.join(' · ');

    const bubble = addBubble('assistant', result.response || '(no answer)', metaHtml);

    if (result.tts_error) {
      const warn = document.createElement('div');
      warn.className = 'tts-warning';
      warn.textContent = `Voice unavailable: ${result.tts_error.message}`;
      bubble.appendChild(warn);
    }

    if (result.audio_url) {
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.preload = 'auto';
      audio.src = result.audio_url;
      const warn = document.createElement('div');
      warn.className = 'tts-warning';
      warn.hidden = true;
      warn.textContent = 'Audio failed to load — the text answer above is still valid.';
      audio.addEventListener('error', () => { warn.hidden = false; });
      audio.addEventListener('play', () => { if (!state.busy) setStatus('speaking'); });
      audio.addEventListener('ended', () => { if (!state.busy && !state.recording) setStatus('ready'); });
      bubble.appendChild(audio);
      bubble.appendChild(warn);
      state.lastAudio = audio;
      if (el.autoplay.checked) {
        audio.play().then(() => setStatus('speaking')).catch(() => {
          // Autoplay blocked by the browser - controls are visible, user can press play.
          if (!state.busy) setStatus('ready');
        });
      } else if (!state.busy) {
        setStatus('ready');
      }
    } else if (!state.busy) {
      setStatus('ready');
    }

    // Client-side wall-clock (upload + full buffered processing).
    const clientLat = Object.assign({}, lat, {
      total_ms: lat.total_ms != null ? Math.max(lat.total_ms, Math.round(wallMs)) : Math.round(wallMs),
      sources: Object.assign({}, lat.sources, {
        total_ms: 'measured_client_plus_server',
      }),
    });
    showLatency(clientLat, [['wall_ms', 'Browser wall-clock']]);
    pollHealth();
  }

  // ---------------------------------------------------------------- text chat
  async function sendText(message) {
    state.busy = true;
    setRecordingUi(false);
    setStatus('thinking');
    hideError();
    const started = performance.now();
    try {
      await ensureSession();
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          session_id: state.sessionId,
          temperature: parseFloat(el.temperature.value),
          max_tokens: parseInt(el.maxTokens.value, 10),
        }),
      });
      const body = await resp.json();
      if (!resp.ok) {
        showError(body.error || `Chat failed (HTTP ${resp.status}).`, body.hint);
        setStatus('error');
        setTimeout(() => { if (!state.busy) setStatus('ready'); }, 3000);
        return;
      }
      addBubble('user', message);
      const wallMs = performance.now() - started;
      adoptSession(body.session_id);
      window.__voxaiTurns = body.history_turns || 0;
      const usage = body.usage || {};
      addBubble('assistant', body.response,
        `${usage.model ? `<span class="tag">${usage.model}</span> · ` : ''}turn ${body.history_turns ?? '—'}`);
      const lat = Object.assign({}, body.latency, { wall_ms: Math.round(wallMs) });
      showLatency(lat, [['wall_ms', 'Browser wall-clock']]);
      setStatus('ready');
      pollHealth();
    } catch (err) {
      showError(`Could not reach the backend: ${err.message}`, 'Is the Flask server still running?');
      setStatus('error');
      setTimeout(() => { if (!state.busy) setStatus('ready'); }, 4000);
    } finally {
      state.busy = false;
      setRecordingUi(false);
      // Success paths already chose the final status; keep Error visible if one is showing.
      if (el.statusPill.className.indexOf('pill-error') === -1) {
        setStatus((state.lastAudio && !state.lastAudio.ended && !state.lastAudio.paused && el.autoplay.checked) ? 'speaking' : 'ready');
      }
    }
  }

  // ---------------------------------------------------------------- session actions
  function resetChatView() {
    el.chatLog.innerHTML = '';
    el.chatLog.appendChild(el.emptyState || buildEmptyState());
    el.latencyPanel.hidden = true;
    el.latencyGrid.innerHTML = '';
    hideError();
    window.__voxaiTurns = 0;
    if (state.lastAudio) { try { state.lastAudio.pause(); } catch (_) {} state.lastAudio = null; }
  }

  function buildEmptyState() {
    const d = document.createElement('div');
    d.id = 'empty-state';
    d.className = 'empty-state';
    d.innerHTML = '<h2>Talk to VoxAI</h2><p>Press the microphone (or <strong>Start</strong>) and ask a question.</p>';
    return d;
  }

  async function newConversation() {
    if (state.busy || state.recording) return;
    try {
      const resp = await fetch('/api/session/new', { method: 'POST' });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body.error || 'session create failed');
      state.sessionId = body.session_id;
      sessionStorage.setItem('voxai.session', state.sessionId);
      resetChatView();
      addNotice('Started a new conversation — previous history stays on the server until it expires.');
      pollHealth();
    } catch (err) {
      showError(`Could not start a new conversation: ${err.message}`);
    }
  }

  async function clearHistory() {
    if (state.busy || state.recording) return;
    const old = state.sessionId;
    if (!old) { resetChatView(); return; }
    try {
      const resp = await fetch(`/api/session/${encodeURIComponent(old)}`, { method: 'DELETE' });
      if (resp.status === 404) {
        // Already gone (expired) - just move on.
      } else if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.error || `clear failed (HTTP ${resp.status})`);
      }
      const fresh = await fetch('/api/session/new', { method: 'POST' });
      const body = await fresh.json();
      state.sessionId = body.session_id || null;
      if (state.sessionId) sessionStorage.setItem('voxai.session', state.sessionId);
      else sessionStorage.removeItem('voxai.session');
      resetChatView();
      addNotice('History cleared on the server. This is a fresh session.');
      pollHealth();
    } catch (err) {
      showError(`Could not clear history: ${err.message}`);
    }
  }

  // ---------------------------------------------------------------- wiring
  el.micBtn.addEventListener('click', () => (state.recording ? stopRecording() : startRecording()));
  el.startBtn.addEventListener('click', startRecording);
  el.stopBtn.addEventListener('click', stopRecording);
  el.newBtn.addEventListener('click', newConversation);
  el.clearBtn.addEventListener('click', clearHistory);

  el.textForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = el.textInput.value.trim();
    if (!text || state.busy || state.recording) return;
    el.textInput.value = '';
    sendText(text);
  });

  el.settingsToggle.addEventListener('click', () => {
    const open = el.settingsPanel.hidden;
    el.settingsPanel.hidden = !open;
    el.settingsToggle.setAttribute('aria-expanded', String(open));
  });

  el.temperature.addEventListener('input', () => { el.temperatureOut.textContent = el.temperature.value; });
  el.maxTokens.addEventListener('input', () => { el.maxTokensOut.textContent = el.maxTokens.value; });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.recording) stopRecording();
  });

  // ---------------------------------------------------------------- boot
  (async function boot() {
    setStatus('ready');
    await pollHealth();
    setInterval(pollHealth, 8000);
    if (state.sessionId) {
      // Validate the stored session id; silently replace it if the server forgot it.
      try {
        const r = await fetch(`/api/session/${state.sessionId}/history`);
        if (r.status === 404) {
          state.sessionId = null;
          sessionStorage.removeItem('voxai.session');
        } else if (r.ok) {
          const b = await r.json();
          window.__voxaiTurns = b.turns || 0;
          if (b.history && b.history.length) {
            ensureChatStarted();
            for (const m of b.history) addBubble(m.role === 'user' ? 'user' : 'assistant', m.content);
          }
        }
      } catch (_) { /* backend will be retried by pollHealth */ }
    }
    pollHealth();
  })();
})();
