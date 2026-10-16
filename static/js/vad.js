/* VoxAI - end-of-speech detection (pure logic, no browser APIs).
 *
 * The browser feeds it small frames of microphone level (RMS from an
 * AnalyserNode); it decides when the user has stopped talking for the
 * configured quiet window (~1.5 s by default) and the recording should be
 * submitted automatically.  Kept free of DOM/WebAudio so the exact decision
 * logic is unit-testable in Node (tests/js/test_vad.mjs).
 *
 * Loaded as a plain <script> (window.VoxVAD) before app.js.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.VoxVAD = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var DEFAULTS = {
    silenceMs: 1500,        // ~1-2 s of continuous silence ends the utterance
    minSpeechMs: 250,       // require some real speech first (a noisy room never triggers)
    minRecordingMs: 800,    // never auto-stop instantly (catches the first word)
    speechThreshold: 0.02,  // RMS above which a frame counts as voiced
    maxRecordingMs: 60000,  // hard cap; manual Stop is always available
    maxFrameGapMs: 1000,    // clamp long gaps (stalled tab, dropped frames)
  };

  // Verdicts returned by push(): 'speaking' | 'silence' |
  // 'end_of_speech' (auto-submit now) | 'max_duration' (hard cap reached).
  function createSilenceDetector(overrides) {
    var cfg = {};
    var key;
    for (key in DEFAULTS) cfg[key] = DEFAULTS[key];
    if (overrides) for (key in overrides) cfg[key] = overrides[key];

    var startedAtMs = null;
    var lastFrameAtMs = null;
    var speechMs = 0;
    var silenceMs = 0;
    var speechSeen = false;

    function push(rms, nowMs) {
      if (startedAtMs === null) {
        startedAtMs = nowMs;
        lastFrameAtMs = nowMs;
      }
      var dt = Math.max(0, Math.min(cfg.maxFrameGapMs, nowMs - lastFrameAtMs));
      lastFrameAtMs = nowMs;

      var voiced = typeof rms === 'number' && isFinite(rms) && rms >= cfg.speechThreshold;
      if (voiced) {
        speechMs += dt;
        silenceMs = 0;
        speechSeen = true;
      } else {
        silenceMs += dt;
      }

      var elapsed = nowMs - startedAtMs;
      if (elapsed >= cfg.maxRecordingMs) return 'max_duration';
      if (voiced) return 'speaking';
      if (
        speechMs >= cfg.minSpeechMs &&
        silenceMs >= cfg.silenceMs &&
        elapsed >= cfg.minRecordingMs
      ) {
        return 'end_of_speech';
      }
      return 'silence';
    }

    return {
      push: push,
      reset: function () {
        startedAtMs = null;
        lastFrameAtMs = null;
        speechMs = 0;
        silenceMs = 0;
        speechSeen = false;
      },
      state: function () {
        return {
          speechMs: speechMs,
          silenceMs: silenceMs,
          speechSeen: speechSeen,
          startedAtMs: startedAtMs,
          config: cfg,
        };
      },
      config: cfg,
    };
  }

  // RMS of one AnalyserNode time-domain frame (bytes: 128 = digital silence).
  function frameRms(samples) {
    if (!samples || !samples.length) return 0;
    var sum = 0;
    for (var i = 0; i < samples.length; i++) {
      var v = (samples[i] - 128) / 128;
      sum += v * v;
    }
    return Math.sqrt(sum / samples.length);
  }

  return { DEFAULTS: DEFAULTS, createSilenceDetector: createSilenceDetector, frameRms: frameRms };
});
